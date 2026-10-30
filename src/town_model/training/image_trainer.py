"""Deterministic image-model optimization, evaluation, and resumable state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import pickle
import zipfile
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from town_model.contracts import FrozenImageFeatureCache, ImageBatch, VisionFeatureBatch
from town_model.dataset.image_data import DatasetSubset, PreparedDataset
from town_model.experiments.image_config import IMAGE_FEATURE_EXTRACTION_CONTRACT, ImageExperimentConfig
from town_model.models.image_embeddings import ImageOntologyModel
from town_model.vision.backbone import MobileNetV3LargeFPN
from town_model.vision.batching import collate_image_records
from town_model.vision.features import SharedVisualFeatureExtractor
from town_model.vision.preprocessing import ImageNetLetterbox
from town_model.vision.roi import GroundTruthROIProvider

from .image_objectives import (
    ImageObjectiveResult,
    ImageObjectiveTerms,
    UndefinedImageObjective,
    aggregate_effective_batch,
    image_objective_terms,
)


class ImageTrainingError(RuntimeError):
    """A numerical, data, or compatibility error that stops an image run."""


@dataclass(frozen=True, slots=True)
class ImageTrainingResult:
    status: str
    latest_checkpoint: Path
    best_checkpoint: Path | None
    global_step: int
    completed_epochs: int
    history: tuple[dict[str, Any], ...]
    best_primary_monitor: float | None
    stopped_reason: str | None
    initial_metrics: dict[str, Any] | None = None


def _hash_state(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _environment() -> dict[str, Any]:
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception as error:  # pragma: no cover - import failure has a more useful earlier error
        torchvision_version = f"unavailable: {error}"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device_for(config: ImageExperimentConfig) -> torch.device:
    requested = config.execution.device
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but this PyTorch runtime has no available CUDA device")
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)
    if device.type == "cpu" and config.purpose == "smoke" and not config.execution.allow_cpu_smoke:
        raise ValueError("CPU smoke runs are disabled by execution.allow_cpu_smoke")
    return device


def build_image_modules(
    dataset: PreparedDataset,
    config: ImageExperimentConfig,
    *,
    class_ids: tuple[int, ...] | None = None,
) -> tuple[SharedVisualFeatureExtractor, ImageOntologyModel, tuple[int, ...]]:
    """Construct matched shared modules and one independent classifier copy."""
    config.validate()
    _set_seed(config.training.seed)
    visual_backbone = MobileNetV3LargeFPN(
        pretrained=config.vision.pretrained_imagenet,
        trainable_backbone_blocks=config.vision.trainable_backbone_blocks,
    )
    policy = config.vision.visual_policy
    visual_backbone.set_trainable(
        trainable_fpn=policy != "frozen",
        trainable_backbone_blocks=(config.vision.trainable_backbone_blocks if policy == "trainable_fpn_last_blocks" else ()),
    )
    visual = SharedVisualFeatureExtractor(visual_backbone)
    selected_class_ids = tuple(class_ids if class_ids is not None else config.dataset.trainable_class_ids or dataset.canonical_class_ids)
    canonical_columns = {class_id: index for index, class_id in enumerate(dataset.canonical_class_ids)}
    if not selected_class_ids or len(set(selected_class_ids)) != len(selected_class_ids) or any(type(class_id) is not int or class_id not in canonical_columns for class_id in selected_class_ids):
        raise ValueError("selected class IDs must be a unique nonempty subset of authoritative trainable ontology IDs")
    if tuple(sorted(selected_class_ids)) != selected_class_ids:
        raise ValueError("explicit class IDs must preserve canonical ontology ordering")
    names_by_id = dict(zip(dataset.canonical_class_ids, dataset.class_names, strict=True))
    model = ImageOntologyModel(
        config.model.latent_dimensions,
        tuple(names_by_id[class_id] for class_id in selected_class_ids),
        family=config.model.family,
        pizza_mode=config.model.pizza_mode,
        seed=config.training.seed,
        classifier_seed=config.model.classifier_seed,
        houses=config.model.houses,
        windows=config.model.windows,
        min_width=config.model.min_width,
        window_mode=config.model.window_mode,
        window_target_volume=config.model.window_target_volume,
    )
    return visual, model, tuple(canonical_columns[class_id] for class_id in selected_class_ids)


def _optimizer(visual: SharedVisualFeatureExtractor, model: ImageOntologyModel, config: ImageExperimentConfig) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = []

    def add(parameters, name: str, lr: float, weight_decay: float) -> None:
        values = [parameter for parameter in parameters if parameter.requires_grad]
        if values:
            groups.append({"params": values, "name": name, "lr": lr, "weight_decay": weight_decay})

    add(visual.backbone.mobile_features.parameters(), "pretrained_backbone", config.training.learning_rate_pretrained, config.training.weight_decay_pretrained)
    add(visual.backbone.fpn.parameters(), "fpn", config.training.learning_rate_fpn, config.training.weight_decay_fpn)
    projection_modules = (model.component_projection, model.global_projection, model.composition, model.fusion)
    add((p for module in projection_modules for p in module.parameters()), "point_and_composition", config.training.learning_rate_projection, config.training.weight_decay_projection)
    add(model.classifier.parameters(), "classifier_geometry", config.training.learning_rate_classifier, config.training.weight_decay_classifier)
    if not groups:
        raise ValueError("image run has no trainable parameters")
    return torch.optim.AdamW(groups)


def _schedule(optimizer: torch.optim.Optimizer, config: ImageExperimentConfig, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = min(config.training.warmup_steps, max(0, total_steps - 1))

    def factor(step: int) -> float:
        if config.training.schedule == "constant":
            return 1.0
        if warmup and step < warmup:
            return max(1e-8, float(step + 1) / warmup)
        decay_steps = max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, (step - warmup) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if payload.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def _pack_targets(batch: ImageBatch, max_components: int, class_indices: tuple[int, ...]) -> tuple[Tensor, Tensor]:
    class_index = torch.tensor(class_indices, dtype=torch.long, device=batch.pizza_targets.device)
    targets = batch.pizza_targets.new_zeros((len(batch.sample_ids), max_components, len(class_indices)))
    known = torch.zeros_like(targets, dtype=torch.bool)
    for image_index, (sample_targets, sample_known) in enumerate(zip(batch.component_targets, batch.component_known_masks, strict=True)):
        count = sample_targets.shape[0]
        if count:
            targets[image_index, :count] = sample_targets.to(batch.pizza_targets.device).index_select(1, class_index)
            known[image_index, :count] = sample_known.to(batch.pizza_known_masks.device).index_select(1, class_index)
    return targets, known


def _metrics_for_level(
    predictions: Tensor,
    targets: Tensor,
    known: Tensor,
    class_names: tuple[str, ...],
    *,
    split: str,
) -> dict[str, Any]:
    predictions = predictions.detach().cpu().bool()
    targets = targets.detach().cpu().bool()
    known = known.detach().cpu().bool()
    if predictions.shape != targets.shape or known.shape != targets.shape or predictions.ndim != 2:
        raise ValueError("metric input arrays must have matching [N,C] shapes")
    per_class: list[dict[str, Any]] = []
    f1_values: list[float] = []
    tp_total = fp_total = fn_total = 0
    for column, name in enumerate(class_names):
        mask = known[:, column]
        target = targets[mask, column]
        predicted = predictions[mask, column]
        positive = int(target.sum().item())
        known_count = int(mask.sum().item())
        tp = int((target & predicted).sum().item())
        fp = int((~target & predicted).sum().item())
        fn = int((target & ~predicted).sum().item())
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 0.0
        exclusion = None
        if known_count == 0:
            exclusion = "all_targets_unknown_in_evaluated_split"
        elif positive == 0:
            exclusion = "no_known_positive_observations_in_evaluated_split"
        else:
            f1_values.append(f1)
        per_class.append({
            "class_name": name,
            "known_positive": positive,
            "known_negative": known_count - positive,
            "unknown": int((~mask).sum().item()),
            "predictions_on_known": int(predicted.sum().item()),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "f1": f1 if exclusion is None else None,
            "macro_exclusion_reason": exclusion,
        })
        tp_total += tp
        fp_total += fp
        fn_total += fn
    macro = sum(f1_values) / len(f1_values) if f1_values else None
    micro_denom = 2 * tp_total + fp_total + fn_total
    return {
        "split": split,
        "macro_f1": macro,
        "macro_classes_included": len(f1_values),
        "micro_f1": 2 * tp_total / micro_denom if micro_denom else 0.0,
        "known_positive": sum(row["known_positive"] for row in per_class),
        "known_negative": sum(row["known_negative"] for row in per_class),
        "unknown": sum(row["unknown"] for row in per_class),
        "per_class": per_class,
    }


def evaluate_image_model(
    visual: SharedVisualFeatureExtractor,
    model: ImageOntologyModel,
    subset: DatasetSubset,
    config: ImageExperimentConfig,
    *,
    class_indices: tuple[int, ...],
    device: torch.device,
    feature_cache: FrozenImageFeatureCache | None = None,
    component_positive_weights: Tensor | None = None,
    pizza_positive_weights: Tensor | None = None,
) -> dict[str, Any]:
    """Known-mask GT-ROI metrics and objective losses for one original split."""
    old_visual_mode = visual.training
    old_model_mode = model.training
    visual.eval()
    model.eval()
    pizzas_predictions: list[Tensor] = []
    pizzas_targets: list[Tensor] = []
    pizzas_known: list[Tensor] = []
    components_predictions: list[Tensor] = []
    components_targets: list[Tensor] = []
    components_known: list[Tensor] = []
    objective_windows: list[ImageObjectiveTerms] = []
    transform = ImageNetLetterbox(config.vision.resolution)
    with torch.no_grad():
        for start in range(0, len(subset.records), config.training.physical_batch_size):
            records = subset.records[start : start + config.training.physical_batch_size]
            batch = collate_image_records(subset.dataset, records, transform=transform)
            batch = _batch_to(batch, device)
            if feature_cache is not None:
                vision_features = _feature_batch_to(feature_cache.get(batch, diagnostics=config.diagnostics.include_visual_features), device)
            elif config.vision.visual_policy == "frozen":
                with torch.no_grad():
                    vision_features = visual(batch, GroundTruthROIProvider(), diagnostics=config.diagnostics.include_visual_features)
            else:
                vision_features = visual(batch, GroundTruthROIProvider(), diagnostics=config.diagnostics.include_visual_features)
            result = model(vision_features, batch.sample_ids)
            objective_windows.append(image_objective_terms(
                result,
                batch,
                family=model.family,
                geometry_loss=config.objective.geometry_loss,
                temperature=config.objective.temperature,
                hinge_margin=config.objective.hinge_margin,
                class_column_indices=class_indices,
                component_positive_weights=component_positive_weights,
                pizza_positive_weights=pizza_positive_weights,
            ))
            pizza_prediction = result.pizza_classifier.logits >= 0 if result.pizza_classifier.logits is not None else result.pizza_classifier.contains
            pizzas_predictions.append(pizza_prediction)
            class_idx = torch.tensor(class_indices, dtype=torch.long, device=device)
            pizzas_targets.append(batch.pizza_targets.index_select(1, class_idx))
            pizzas_known.append(batch.pizza_known_masks.index_select(1, class_idx))
            max_components = result.component_valid_mask.shape[1]
            component_targets, component_known = _pack_targets(batch, max_components, class_indices)
            component_prediction = result.component_classifier.logits >= 0 if result.component_classifier.logits is not None else result.component_classifier.contains
            components_predictions.append(component_prediction[result.component_valid_mask])
            components_targets.append(component_targets[result.component_valid_mask])
            components_known.append(component_known[result.component_valid_mask])
    visual.train(old_visual_mode)
    model.train(old_model_mode)
    pizza_metric = _metrics_for_level(
        torch.cat(pizzas_predictions), torch.cat(pizzas_targets), torch.cat(pizzas_known),
        tuple(model.class_names), split=subset.split,
    )
    if components_predictions and sum(tensor.shape[0] for tensor in components_predictions):
        component_metric = _metrics_for_level(
            torch.cat(components_predictions), torch.cat(components_targets), torch.cat(components_known),
            tuple(model.class_names), split=subset.split,
        )
    else:
        component_metric = _metrics_for_level(
            torch.empty((0, len(model.class_names)), dtype=torch.bool),
            torch.empty((0, len(model.class_names)), dtype=torch.bool),
            torch.empty((0, len(model.class_names)), dtype=torch.bool),
            tuple(model.class_names), split=subset.split,
        )
    primary = None
    if pizza_metric["macro_f1"] is not None and component_metric["macro_f1"] is not None:
        primary = 0.5 * pizza_metric["macro_f1"] + 0.5 * component_metric["macro_f1"]
    try:
        objective_result = aggregate_effective_batch(
            objective_windows,
            component_coefficient=config.objective.component_coefficient,
            pizza_coefficient=config.objective.pizza_coefficient,
        )
        objective_losses = {
            "total": float(objective_result.total.detach().cpu()),
            "component": float(objective_result.component_loss.detach().cpu()) if objective_result.component_loss is not None else None,
            "pizza": float(objective_result.pizza_loss.detach().cpu()) if objective_result.pizza_loss is not None else None,
            "undefined_tasks": list(objective_result.undefined_tasks),
        }
    except UndefinedImageObjective:
        objective_losses = {"total": None, "component": None, "pizza": None, "undefined_tasks": ["component", "pizza"]}
    return {
        "split": subset.split,
        "objective_losses": objective_losses,
        "objective_identity": config.objective_identity(),
        "pizza": pizza_metric,
        "component_gt_roi": component_metric,
        "primary_monitor": primary,
        "monitor_definition": "0.5 * pizza_macro_F1 + 0.5 * component_GT_ROI_macro_F1",
        "dataset_subset_id": subset.subset_id,
    }


def _feature_batch_to(features: VisionFeatureBatch, device: torch.device) -> VisionFeatureBatch:
    diagnostic = features.diagnostics
    if diagnostic is not None:
        diagnostic = type(diagnostic)(
            roi_features=diagnostic.roi_features.to(device) if diagnostic.roi_features is not None else None,
            global_features=diagnostic.global_features.to(device) if diagnostic.global_features is not None else None,
            roi_source=diagnostic.roi_source,
            normalized_roi_coordinates=diagnostic.normalized_roi_coordinates.to(device) if diagnostic.normalized_roi_coordinates is not None else None,
        )
    return VisionFeatureBatch(
        component_vectors=features.component_vectors.to(device),
        component_valid_mask=features.component_valid_mask.to(device),
        global_spatial_features=features.global_spatial_features.to(device),
        component_ids=features.component_ids,
        diagnostics=diagnostic,
    )


def _batch_to(batch: ImageBatch, device: torch.device) -> ImageBatch:
    return ImageBatch(
        images=batch.images.to(device),
        original_image_sizes=batch.original_image_sizes,
        processed_image_sizes=batch.processed_image_sizes,
        image_geometries=batch.image_geometries,
        sample_ids=batch.sample_ids,
        gt_component_boxes_xyxy=tuple(value.to(device) for value in batch.gt_component_boxes_xyxy),
        component_ids=batch.component_ids,
        component_detector_category_ids=batch.component_detector_category_ids,
        pizza_targets=batch.pizza_targets.to(device),
        pizza_known_masks=batch.pizza_known_masks.to(device),
        component_targets=tuple(value.to(device) for value in batch.component_targets),
        component_known_masks=tuple(value.to(device) for value in batch.component_known_masks),
        split_identity=batch.split_identity,
    )


def _gradient_activity(visual: SharedVisualFeatureExtractor, model: ImageOntologyModel) -> dict[str, Any]:
    groups = {
        "component_projection": tuple(model.component_projection.parameters()),
        "global_projection": tuple(model.global_projection.parameters()),
        "deepsets": tuple(model.composition.parameters()),
        "hybrid_fusion": tuple(model.fusion.parameters()),
        "classifier_geometry": tuple(model.classifier.parameters()),
        "fpn": tuple(visual.backbone.fpn.parameters()),
        "backbone_unfrozen": tuple(
            parameter for parameter in visual.backbone.mobile_features.parameters() if parameter.requires_grad
        ),
    }
    report = {}
    for name, parameters in groups.items():
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        gradients = [parameter.grad.detach().float() for parameter in trainable if parameter.grad is not None]
        norm = math.sqrt(sum(float(gradient.square().sum().item()) for gradient in gradients)) if gradients else 0.0
        nonzero = sum(int(torch.count_nonzero(gradient).item()) for gradient in gradients)
        element_count = sum(gradient.numel() for gradient in gradients)
        report[name] = {
            "status": "frozen" if not trainable else "active" if norm > 0 else "no_gradient",
            "gradient_l2": norm,
            "nonzero_gradient_elements": nonzero,
            "gradient_elements": element_count,
            "zero_gradient_fraction": (1.0 - nonzero / element_count) if element_count else None,
        }
    expected = {
        "component_projection": "component auxiliary supervision and/or composition",
        "global_projection": "global or hybrid pizza mode",
        "deepsets": "compositional or hybrid pizza mode",
        "hybrid_fusion": "hybrid pizza mode only",
        "classifier_geometry": "all supervised tasks",
        "fpn": "trainable visual feature condition",
        "backbone_unfrozen": "explicit trainable backbone block selection",
    }
    for name, description in expected.items():
        report[name]["expected_activity"] = description
    return report


def _geometry_is_valid(model: ImageOntologyModel) -> None:
    geometry = getattr(model.classifier, "geometry", None)
    if geometry is None:
        return
    if hasattr(geometry, "physical_boxes"):
        low, high = geometry.physical_boxes()
        arrays = (low, high)
    else:
        arrays = geometry.physical_geometry()
    for value in arrays:
        if not torch.isfinite(value).all() or torch.any(value < -1) or torch.any(value > 1):
            raise ImageTrainingError("nonfinite or out-of-domain trainable physical geometry")


def _assert_finite_gradients(modules: tuple[torch.nn.Module, ...], step: int) -> None:
    for module in modules:
        for name, parameter in module.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ImageTrainingError(f"nonfinite gradient at optimizer step {step}, parameter {name}")


def _compatibility_identity(
    dataset: PreparedDataset,
    train_subset: DatasetSubset,
    validation_subset: DatasetSubset,
    config: ImageExperimentConfig,
    class_ids: tuple[int, ...],
    visual_state_hash: str,
) -> dict[str, Any]:
    return {
        "config_hash": config.identity_hash(),
        "objective_identity": config.objective_identity(),
        "dataset_id": dataset.dataset_id,
        "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
        "class_manifest_sha256": dataset.class_manifest_sha256,
        "train_subset_id": train_subset.subset_id,
        "validation_subset_id": validation_subset.subset_id,
        "class_ids": list(class_ids),
        "roi_source": config.vision.roi_source,
        "visual_initial_state_hash": visual_state_hash,
        "architecture": "mobilenet_v3_large_fpn_image_ontology_v2",
        "feature_extraction_contract": IMAGE_FEATURE_EXTRACTION_CONTRACT,
    }


def _checkpoint_payload(
    visual: SharedVisualFeatureExtractor,
    model: ImageOntologyModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    identity: dict[str, Any],
    resolved_config: dict[str, Any],
    *,
    epoch: int,
    batch_cursor: int,
    epoch_order: list[str],
    global_step: int,
    best_primary: float | None,
    best_epoch: int | None,
    patience_count: int,
    history: list[dict[str, Any]],
    class_ids: tuple[int, ...],
    parents: list[dict[str, Any]],
    initial_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "image-training-checkpoint-v1",
        "identity": identity,
        "resolved_config": resolved_config,
        "visual_state": visual.state_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": _rng_state(),
        "epoch": epoch,
        "batch_cursor": batch_cursor,
        "epoch_order_sample_ids": epoch_order,
        "optimizer_step": global_step,
        "accumulation_window_open": False,
        "best_primary_monitor": best_primary,
        "best_epoch": best_epoch,
        "early_stopping_patience_count": patience_count,
        "history": history,
        "initial_metrics": initial_metrics,
        "class_ids": list(class_ids),
        "environment": _environment(),
        "parents": parents,
    }


def _load_checkpoint(path: Path, expected_identity: dict[str, Any], device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except (OSError, EOFError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
        raise ImageTrainingError(f"could not load checkpoint {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != "image-training-checkpoint-v1":
        raise ImageTrainingError(f"checkpoint {path} has an unsupported structure or schema")
    actual = payload.get("identity")
    if actual != expected_identity:
        differing = sorted(key for key in set(expected_identity) | set(actual or {}) if expected_identity.get(key) != (actual or {}).get(key))
        raise ImageTrainingError(f"checkpoint {path} is incompatible with this run; identity fields differ: {differing}")
    if payload.get("accumulation_window_open") is not False:
        raise ImageTrainingError("checkpoint is not at a safe optimizer/accumulation boundary")
    return payload


def _positive_weights_from_counts(
    positive_counts: tuple[int, ...] | list[int],
    *,
    weighting: str,
    cap: float,
    beta: float,
    require_coverage: bool,
    required_mask: tuple[bool, ...] | None = None,
) -> tuple[float, ...]:
    """Resolve positive weights from known-positive training support only.

    For proportional policies, raw values are scaled so the most-supported
    class has weight 1, then upper-capped. This transparent common scale leaves
    all uncapped class-to-class ratios unchanged and keeps the positive weight
    of the best-supported class aligned with the fixed known-negative weight.
    A zero-support class gets neutral weight 1 only in exploratory handling;
    it has no positive observations to reweight, and comparison runs fail.
    """
    if weighting == "none":
        return tuple(1.0 for _ in positive_counts)
    if weighting not in ("inverse_sqrt", "effective_number", "experimental_inverse_frequency"):
        raise ValueError(f"unsupported positive weighting policy {weighting!r}")
    if not math.isfinite(cap) or cap < 1:
        raise ValueError("positive weight cap must be finite and at least one")
    if not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("effective-number beta must be finite in [0,1)")
    if any(type(count) is not int or count < 0 for count in positive_counts):
        raise ValueError("known-positive support counts must be nonnegative integers")

    required = required_mask if required_mask is not None else tuple(True for _ in positive_counts)
    if len(required) != len(positive_counts):
        raise ValueError("required_mask must align with positive support counts")
    supported = [index for index, count in enumerate(positive_counts) if count > 0]
    if require_coverage:
        missing = [index for index, (count, needed) in enumerate(zip(positive_counts, required, strict=True)) if needed and count == 0]
        if missing:
            raise ValueError(f"comparison run has zero known-positive support at class positions {missing}")
    if not supported:
        return tuple(1.0 for _ in positive_counts)

    raw: dict[int, float] = {}
    for index in supported:
        count = positive_counts[index]
        if weighting == "inverse_sqrt":
            raw[index] = 1.0 / math.sqrt(count)
        elif weighting == "experimental_inverse_frequency":
            raw[index] = 1.0 / count
        elif beta == 0.0:
            raw[index] = 1.0
        else:
            # -expm1(n*log(beta)) is stable when beta is near one.
            raw[index] = (1.0 - beta) / (-math.expm1(count * math.log(beta)))

    common_scale = 1.0 / min(raw.values())
    maximum_support = max(positive_counts[index] for index in supported)
    weights = [1.0] * len(positive_counts)
    for index, raw_weight in raw.items():
        normalized = 1.0 if positive_counts[index] == maximum_support else raw_weight * common_scale
        weights[index] = min(cap, normalized)
    return tuple(weights)


def _positive_weights(
    subset: DatasetSubset,
    dataset: PreparedDataset,
    class_ids: tuple[int, ...],
    *,
    level: str,
    weighting: str,
    cap: float,
    beta: float,
    require_coverage: bool,
) -> tuple[float, ...]:
    if subset.split != "train" or subset.dataset.dataset_id != dataset.dataset_id:
        raise ValueError("positive class weights must be resolved from this dataset's original training subset only")
    if level not in ("pizza", "component"):
        raise ValueError("positive class weights require pizza or component supervision level")
    support_ids = tuple(subset.class_support["class_ids"])
    support_index = {class_id: index for index, class_id in enumerate(support_ids)}
    positive_counts = subset.class_support[level]["known_positive"]
    supervised_ids = set(
        dataset.pizza_supervision_class_ids if level == "pizza" else dataset.component_supervision_class_ids
    )
    missing = [
        class_id for class_id in class_ids
        if class_id in supervised_ids and positive_counts[support_index[class_id]] == 0
    ]
    if require_coverage and missing:
        names = {class_id: dataset.class_names[dataset.canonical_class_ids.index(class_id)] for class_id in missing}
        raise ValueError(f"comparison run lacks known-positive {level} training support for classes: {names}")
    selected_ids = set(class_ids)
    # Resolve the common normalization scale over the complete canonical
    # in-scope training vocabulary, then project weights to the selected model
    # classes. This keeps run weights stable when class-selection subsets vary.
    canonical_ids = dataset.canonical_class_ids
    scoped_counts = tuple(
        positive_counts[support_index[class_id]] if class_id in supervised_ids else 0
        for class_id in canonical_ids
    )
    required = tuple(class_id in selected_ids and class_id in supervised_ids for class_id in canonical_ids)
    all_weights = _positive_weights_from_counts(
        scoped_counts,
        weighting=weighting,
        cap=cap,
        beta=beta,
        require_coverage=require_coverage,
        required_mask=required,
    )
    weight_by_id = dict(zip(canonical_ids, all_weights, strict=True))
    return tuple(weight_by_id[class_id] for class_id in class_ids)


def validate_image_training_data(
    dataset: PreparedDataset,
    train_subset: DatasetSubset,
    validation_subset: DatasetSubset,
    class_ids: tuple[int, ...],
    *,
    require_training_coverage: bool,
    require_primary_monitor: bool,
) -> None:
    class_ids_for_level = {
        "pizza": set(dataset.pizza_supervision_class_ids),
        "component": set(dataset.component_supervision_class_ids),
    }
    support_index = {class_id: index for index, class_id in enumerate(train_subset.class_support["class_ids"])}
    validation_index = {class_id: index for index, class_id in enumerate(validation_subset.class_support["class_ids"])}
    for level, scoped_ids in class_ids_for_level.items():
        required = set(class_ids) & scoped_ids
        train_positive = train_subset.class_support[level]["known_positive"]
        validation_positive = validation_subset.class_support[level]["known_positive"]
        if require_training_coverage:
            missing = [class_id for class_id in sorted(required) if train_positive[support_index[class_id]] == 0]
            if missing:
                names = [dataset.class_names[dataset.canonical_class_ids.index(class_id)] for class_id in missing]
                raise ValueError(f"comparison run lacks known-positive {level} support in the training subset: {names}")
        if require_primary_monitor and not any(
            validation_positive[validation_index[class_id]] > 0 for class_id in required
        ):
            raise ValueError(f"primary validation monitor is undefined: no {level} class has known-positive validation support")


def _trainable_class_indices(dataset: PreparedDataset, config: ImageExperimentConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ids = tuple(config.dataset.trainable_class_ids or dataset.canonical_class_ids)
    if tuple(sorted(ids)) != ids or any(class_id not in dataset.canonical_class_ids for class_id in ids):
        raise ValueError("selected class IDs must preserve the canonical trainable class order")
    canonical_index = {class_id: index for index, class_id in enumerate(dataset.canonical_class_ids)}
    return ids, tuple(canonical_index[class_id] for class_id in ids)


def train_image_model(
    dataset: PreparedDataset,
    train_subset: DatasetSubset,
    validation_subset: DatasetSubset,
    config: ImageExperimentConfig,
    run_directory: Path | str,
    *,
    resume_from: Path | str | None = None,
    stop_requested: Callable[[], bool] | None = None,
    feature_cache: FrozenImageFeatureCache | None = None,
    snapshot_callback: Callable[[int, int, SharedVisualFeatureExtractor, ImageOntologyModel, bool], None] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ImageTrainingResult:
    """Train one classifier/composition pair on real prepared RGB images."""
    config.validate()
    if train_subset.dataset.dataset_id != dataset.dataset_id or validation_subset.dataset.dataset_id != dataset.dataset_id:
        raise ValueError("train/validation subsets must reference the loaded canonical dataset")
    if train_subset.split != "train" or validation_subset.split != "validation":
        raise ValueError("image training must use original train and validation partitions")
    train_ids = {record.sample_id for record in train_subset.records}
    validation_ids = {record.sample_id for record in validation_subset.records}
    if train_ids & validation_ids:
        raise ValueError("train and validation subset IDs overlap")
    class_ids, class_indices = _trainable_class_indices(dataset, config)
    for subset in (train_subset, validation_subset):
        for record in subset.records:
            if record.split not in ("train", "validation"):
                raise ValueError("test records are sealed from image-model training and validation")
    require_coverage = config.purpose == "comparison"
    validate_image_training_data(
        dataset,
        train_subset,
        validation_subset,
        class_ids,
        require_training_coverage=require_coverage,
        require_primary_monitor=config.purpose != "smoke",
    )
    component_weights = _positive_weights(
        train_subset, dataset, class_ids, level="component",
        weighting=config.objective.positive_weighting, cap=config.objective.positive_weight_cap,
        beta=config.objective.effective_number_beta, require_coverage=require_coverage,
    )
    pizza_weights = _positive_weights(
        train_subset, dataset, class_ids, level="pizza",
        weighting=config.objective.positive_weighting, cap=config.objective.positive_weight_cap,
        beta=config.objective.effective_number_beta, require_coverage=require_coverage,
    )
    device = _device_for(config)
    root = Path(run_directory).expanduser().resolve()
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    if device.type == "cpu":
        torch.set_num_threads(config.execution.cpu_threads)
    _set_seed(config.training.seed)
    visual, model, class_indices = build_image_modules(dataset, config, class_ids=class_ids)
    visual.to(device)
    model.to(device)
    visual_initial_hash = _hash_state(visual.state_dict())
    common_initial_hash = _hash_state(model.common_state_dict())
    classifier_initial_hash = _hash_state(model.classifier.state_dict())
    identity = _compatibility_identity(dataset, train_subset, validation_subset, config, class_ids, visual_initial_hash)
    if feature_cache is not None:
        if config.vision.visual_policy != "frozen":
            raise ValueError("frozen feature cache cannot be used with trainable visual features")
        records_by_id = {record.sample_id: record for record in (*train_subset.records, *validation_subset.records)}
        expected_cache_identity = {
            "dataset_id": dataset.dataset_id,
            "dataset_manifest_id": dataset.manifest_id,
            "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
            "class_manifest_sha256": dataset.class_manifest_sha256,
            "train_subset_id": train_subset.subset_id,
            "validation_subset_id": validation_subset.subset_id,
            "sample_ids": list(records_by_id),
            "sample_splits": {sample_id: record.split for sample_id, record in records_by_id.items()},
            "roi_source": config.vision.roi_source,
            "visual_state_sha256": visual_initial_hash,
            "backbone_initialization": visual.backbone.initialization_source,
            "backbone_architecture": "torchvision_mobilenet_v3_large_features_0_16",
            "feature_extraction_contract": IMAGE_FEATURE_EXTRACTION_CONTRACT,
            "fpn_input_indices": list(visual.backbone.feature_indices),
            "fpn_input_channels": list(visual.backbone.input_channels),
            "fpn_output_channels": visual.backbone.output_channels,
            "preprocessing": {
                "name": "rgb_letterbox_imagenet_normalize_v1",
                "resolution": config.vision.resolution,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "roi_size": config.vision.roi_size,
            },
            "torch": torch.__version__,
            "torchvision": visual.backbone.torchvision_version,
            "producer_device": str(device),
        }
        cache_identity = feature_cache.identity
        mismatches = [key for key, expected in expected_cache_identity.items() if cache_identity.get(key) != expected]
        if _hash_json(cache_identity)[:24] != feature_cache.cache_id:
            mismatches.append("cache_id")
        if mismatches:
            raise ValueError(f"frozen feature cache is incompatible with this run; identity fields differ: {sorted(mismatches)}")
    elif config.execution.feature_cache_directory:
        raise ValueError("feature_cache_directory is configured but the experiment service did not provide a validated cache")
    resolved_config = asdict(config)
    resolved_config["resolved_objective"] = config.objective_identity()
    resolved_config["objective_semantics"] = config.identity_dict()["objective_semantics"]
    resolved_config["feature_cache_identity"] = feature_cache.identity if feature_cache is not None else None
    resolved_config["resolved_data"] = {
        "dataset_id": dataset.dataset_id,
        "manifest_id": dataset.manifest_id,
        "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
        "class_manifest_sha256": dataset.class_manifest_sha256,
        "class_ids": list(class_ids),
        "train_subset": train_subset.to_manifest(),
        "validation_subset": validation_subset.to_manifest(),
        "class_column_indices": list(class_indices),
        "class_weight_vectors": {
            "component_positive": list(component_weights),
            "pizza_positive": list(pizza_weights),
            "policy": config.objective.positive_weighting,
            "semantics_version": config.objective_identity()["positive_weighting_semantics"],
            "formula": config.objective_identity()["positive_weighting_formula"],
            "normalization": "raw positive-count-proportional values are scaled over all positive-supported canonical classes in each level so the most-supported in-scope class has weight 1, then capped and projected to selected classes; known negatives remain exactly 1; unknowns are excluded",
            "cap": config.objective.positive_weight_cap,
        },
    }
    resolved_config["initialization"] = {
        "visual_state_sha256": visual_initial_hash,
        "common_neural_state_sha256": common_initial_hash,
        "classifier_specific_state_sha256": classifier_initial_hash,
        "backbone_initialization": visual.backbone.initialization_source,
        "fpn_initialization": "torchvision FeaturePyramidNetwork default initialization; matched by seed across compared families",
    }
    identity["common_neural_state_hash"] = common_initial_hash
    identity["classifier_family"] = model.family
    identity["pizza_mode"] = model.pizza_mode
    optimizer = _optimizer(visual, model, config)
    optimizer_steps_per_epoch = math.ceil(math.ceil(len(train_subset.records) / config.training.physical_batch_size) / config.training.accumulation_steps)
    total_steps = max(1, optimizer_steps_per_epoch * config.training.epochs)
    scheduler = _schedule(optimizer, config, total_steps)
    history: list[dict[str, Any]] = []
    global_step = 0
    epoch_start = 0
    batch_cursor_resume = 0
    epoch_order_resume: list[str] | None = None
    best_primary: float | None = None
    best_epoch: int | None = None
    patience_count = 0
    checkpoint_parents: list[dict[str, Any]] = []
    initial_metrics: dict[str, Any] | None = None
    latest_path = checkpoints / "latest.pt"
    best_path = checkpoints / "best-validation.pt"
    if resume_from is not None:
        resume_path = Path(resume_from).expanduser().resolve()
        payload = _load_checkpoint(resume_path, identity, device)
        visual.load_state_dict(payload["visual_state"], strict=True)
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        history = list(payload.get("history", []))
        initial_metrics = payload.get("initial_metrics")
        global_step = int(payload["optimizer_step"])
        epoch_start = int(payload["epoch"])
        batch_cursor_resume = int(payload["batch_cursor"])
        epoch_order_resume = list(payload["epoch_order_sample_ids"])
        best_primary = payload.get("best_primary_monitor")
        best_epoch = payload.get("best_epoch")
        patience_count = int(payload.get("early_stopping_patience_count", 0))
        _restore_rng(payload["rng_state"])
        checkpoint_parents = list(payload.get("parents", [])) + [{
            "checkpoint_sha256": hashlib.sha256(resume_path.read_bytes()).hexdigest(),
            "optimizer_step": int(payload["optimizer_step"]),
            "resume_source": str(resume_path),
        }]
    else:
        if config.purpose != "smoke":
            component_weight_tensor = torch.tensor(component_weights, dtype=torch.float32, device=device)
            pizza_weight_tensor = torch.tensor(pizza_weights, dtype=torch.float32, device=device)
            initial_metrics = {
                "training": evaluate_image_model(
                    visual, model, train_subset, config, class_indices=class_indices, device=device,
                    feature_cache=feature_cache,
                    component_positive_weights=component_weight_tensor,
                    pizza_positive_weights=pizza_weight_tensor,
                ),
                "validation": evaluate_image_model(
                    visual, model, validation_subset, config, class_indices=class_indices, device=device,
                    feature_cache=feature_cache,
                    component_positive_weights=component_weight_tensor,
                    pizza_positive_weights=pizza_weight_tensor,
                ),
            }
        initial_order = [record.sample_id for record in train_subset.records]
        random.Random(config.training.seed).shuffle(initial_order)
        epoch_order_resume = initial_order
        initial_checkpoint = _checkpoint_payload(
            visual, model, optimizer, scheduler, identity, resolved_config,
            epoch=0, batch_cursor=0, epoch_order=initial_order, global_step=0,
            best_primary=None, best_epoch=None, patience_count=0, history=[], class_ids=class_ids,
            parents=checkpoint_parents, initial_metrics=initial_metrics,
        )
        _atomic_torch_save(initial_checkpoint, latest_path)

    transform = ImageNetLetterbox(config.vision.resolution)
    last_epoch_completed = epoch_start
    stopped_reason: str | None = None
    for epoch in range(epoch_start, config.training.epochs):
        visual.train()
        model.train()
        order = list(epoch_order_resume) if epoch == epoch_start and epoch_order_resume is not None else [record.sample_id for record in train_subset.records]
        if not (epoch == epoch_start and epoch_order_resume is not None):
            random.Random(config.training.seed + epoch).shuffle(order)
        if set(order) != train_ids:
            raise ImageTrainingError("saved sampler order does not match this immutable train subset")
        records_by_id = {record.sample_id: record for record in train_subset.records}
        microbatches = [
            [records_by_id[sample_id] for sample_id in order[start : start + config.training.physical_batch_size]]
            for start in range(0, len(order), config.training.physical_batch_size)
        ]
        cursor = batch_cursor_resume if epoch == epoch_start else 0
        if cursor < 0 or cursor > len(microbatches):
            raise ImageTrainingError("saved sampler cursor is outside the selected train subset")
        pending: list[ImageObjectiveTerms] = []
        pending_microbatch_count = 0
        optimizer.zero_grad(set_to_none=True)
        for micro_index in range(cursor, len(microbatches)):
            if stop_requested is not None and stop_requested():
                stopped_reason = "graceful stop requested at a safe optimizer boundary"
                break
            batch = collate_image_records(dataset, microbatches[micro_index], transform=transform)
            batch = _batch_to(batch, device)
            if feature_cache is not None:
                vision_features = _feature_batch_to(feature_cache.get(batch, diagnostics=config.diagnostics.include_visual_features), device)
            elif config.vision.visual_policy == "frozen":
                with torch.no_grad():
                    vision_features = visual(batch, GroundTruthROIProvider(), diagnostics=config.diagnostics.include_visual_features)
            else:
                vision_features = visual(batch, GroundTruthROIProvider(), diagnostics=config.diagnostics.include_visual_features)
            result = model(vision_features, batch.sample_ids)
            for point_tensor in (result.component_points, result.global_pizza_points, result.composed_pizza_points, result.final_pizza_points):
                if not torch.isfinite(point_tensor).all():
                    raise ImageTrainingError(f"nonfinite image embedding at optimizer step {global_step + 1}")
            targets, known = _pack_targets(batch, result.component_valid_mask.shape[1], class_indices)
            del targets, known  # Image objective applies the canonical masks and class-ID selection.
            terms = image_objective_terms(
                result,
                batch,
                family=model.family,
                geometry_loss=config.objective.geometry_loss,
                temperature=config.objective.temperature,
                hinge_margin=config.objective.hinge_margin,
                class_column_indices=class_indices,
                component_positive_weights=torch.tensor(component_weights, dtype=torch.float32, device=device),
                pizza_positive_weights=torch.tensor(pizza_weights, dtype=torch.float32, device=device),
            )
            pending.append(terms)
            pending_microbatch_count += 1
            is_window_end = pending_microbatch_count >= config.training.accumulation_steps or micro_index + 1 == len(microbatches)
            if not is_window_end:
                continue
            try:
                objective: ImageObjectiveResult = aggregate_effective_batch(
                    pending,
                    component_coefficient=config.objective.component_coefficient,
                    pizza_coefficient=config.objective.pizza_coefficient,
                )
            except UndefinedImageObjective as error:
                raise ImageTrainingError(f"undefined image objective at epoch {epoch + 1}, step {global_step + 1}: {error}") from error
            if not torch.isfinite(objective.total):
                raise ImageTrainingError(f"nonfinite objective at epoch {epoch + 1}, optimizer step {global_step + 1}")
            objective.total.backward()
            _assert_finite_gradients((visual, model), global_step + 1)
            gradient_report = _gradient_activity(visual, model)
            parameters = [parameter for module in (visual, model) for parameter in module.parameters() if parameter.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, config.training.gradient_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            _geometry_is_valid(model)
            if any(not torch.isfinite(parameter).all() for module in (visual, model) for parameter in module.parameters()):
                raise ImageTrainingError(f"nonfinite trainable parameter after optimizer step {global_step}")
            history_row = {
                "epoch": epoch + 1,
                "optimizer_step": global_step,
                "training_objective": float(objective.total.detach().cpu()),
                "objective_identity": config.objective_identity(),
                "component_loss": float(objective.component_loss.detach().cpu()) if objective.component_loss is not None else None,
                "pizza_loss": float(objective.pizza_loss.detach().cpu()) if objective.pizza_loss is not None else None,
                "component_eligible_images": objective.component_eligible_images,
                "pizza_eligible_images": objective.pizza_eligible_images,
                "undefined_tasks": list(objective.undefined_tasks),
                "microbatches_in_effective_window": pending_microbatch_count,
                "gradient_activity": gradient_report,
                "learning_rates": {group.get("name", str(index)): group["lr"] for index, group in enumerate(optimizer.param_groups)},
            }
            history.append(history_row)
            next_cursor = micro_index + 1
            checkpoint = _checkpoint_payload(
                visual, model, optimizer, scheduler, identity, resolved_config,
                epoch=epoch, batch_cursor=next_cursor, epoch_order=order, global_step=global_step,
                best_primary=best_primary, best_epoch=best_epoch, patience_count=patience_count,
                history=history, class_ids=class_ids, parents=checkpoint_parents,
                initial_metrics=initial_metrics,
            )
            _atomic_torch_save(checkpoint, latest_path)
            if progress_callback is not None:
                progress_callback({"event": "optimizer_step", **history_row})
            pending.clear()
            pending_microbatch_count = 0
            if stop_requested is not None and stop_requested():
                stopped_reason = "graceful stop requested after optimizer update"
                break
        if stopped_reason is not None:
            break
        batch_cursor_resume = 0
        epoch_order_resume = None
        last_epoch_completed = epoch + 1
        component_weight_tensor = torch.tensor(component_weights, dtype=torch.float32, device=device)
        pizza_weight_tensor = torch.tensor(pizza_weights, dtype=torch.float32, device=device)
        training_metrics = evaluate_image_model(
            visual, model, train_subset, config, class_indices=class_indices, device=device,
            feature_cache=feature_cache,
            component_positive_weights=component_weight_tensor,
            pizza_positive_weights=pizza_weight_tensor,
        )
        validation = evaluate_image_model(
            visual, model, validation_subset, config, class_indices=class_indices, device=device,
            feature_cache=feature_cache,
            component_positive_weights=component_weight_tensor,
            pizza_positive_weights=pizza_weight_tensor,
        )
        primary = validation["primary_monitor"]
        history.append({
            "epoch": epoch + 1,
            "optimizer_step": global_step,
            "training_metrics": training_metrics,
            "validation": validation,
        })
        if primary is None and config.purpose != "smoke":
            raise ImageTrainingError(
                "primary validation monitor is undefined (pizza or GT-ROI component macro-F1 has no supported positives); "
                "comparison/exploratory runs do not fall back to another metric"
            )
        improved = primary is not None and (best_primary is None or primary > best_primary)
        if improved:
            best_primary = float(primary)
            best_epoch = epoch + 1
            patience_count = 0
        elif primary is not None:
            patience_count += 1
        next_order = [record.sample_id for record in train_subset.records]
        if epoch + 1 < config.training.epochs:
            random.Random(config.training.seed + epoch + 1).shuffle(next_order)
        else:
            next_order = []
        checkpoint = _checkpoint_payload(
            visual, model, optimizer, scheduler, identity, resolved_config,
            epoch=epoch + 1, batch_cursor=0, epoch_order=next_order, global_step=global_step,
            best_primary=best_primary, best_epoch=best_epoch, patience_count=patience_count,
            history=history, class_ids=class_ids, parents=checkpoint_parents,
            initial_metrics=initial_metrics,
        )
        _atomic_torch_save(checkpoint, latest_path)
        if improved and not config.training.exploratory_latest_only:
            _atomic_torch_save(checkpoint, best_path)
        will_stop_early = config.purpose != "smoke" and primary is not None and patience_count >= config.training.early_stopping_patience
        if snapshot_callback is not None:
            snapshot_callback(epoch + 1, global_step, visual, model, epoch + 1 == config.training.epochs or will_stop_early)
        if will_stop_early:
            stopped_reason = f"early stopping: primary validation monitor did not improve for {patience_count} checks"
            break
    status = "stopped" if stopped_reason and stopped_reason.startswith("graceful") else "completed"
    return ImageTrainingResult(
        status=status,
        latest_checkpoint=latest_path,
        best_checkpoint=best_path if best_path.exists() else None,
        global_step=global_step,
        completed_epochs=last_epoch_completed,
        history=tuple(history),
        best_primary_monitor=best_primary,
        stopped_reason=stopped_reason,
        initial_metrics=initial_metrics,
    )


def _evaluate_image_checkpoint(
    dataset: PreparedDataset,
    train_subset: DatasetSubset,
    validation_subset: DatasetSubset,
    evaluation_subset: DatasetSubset,
    config: ImageExperimentConfig,
    checkpoint_path: Path | str,
    *,
    class_ids: tuple[int, ...],
) -> dict[str, Any]:
    """Load a compatible saved checkpoint and evaluate without training/updating it."""
    config.validate()
    if train_subset.dataset.dataset_id != dataset.dataset_id or validation_subset.dataset.dataset_id != dataset.dataset_id:
        raise ValueError("training/validation subsets belong to a different canonical dataset")
    if train_subset.split != "train" or validation_subset.split != "validation":
        raise ValueError("checkpoint identity requires the original train and validation partitions")
    train_ids = {record.sample_id for record in train_subset.records}
    validation_ids = {record.sample_id for record in validation_subset.records}
    if train_ids & validation_ids:
        raise ValueError("training and validation subset IDs overlap")
    if any(record.split != "train" for record in train_subset.records) or any(record.split != "validation" for record in validation_subset.records):
        raise ValueError("training/validation subset records do not match their original partitions")
    if evaluation_subset.dataset.dataset_id != dataset.dataset_id:
        raise ValueError("evaluation subset belongs to a different canonical dataset")
    if evaluation_subset.split not in ("train", "validation", "test"):
        raise ValueError("evaluation subset must be one original dataset partition")
    expected_ids = train_ids if evaluation_subset.split == "train" else validation_ids if evaluation_subset.split == "validation" else None
    evaluation_ids = {record.sample_id for record in evaluation_subset.records}
    if expected_ids is not None and not evaluation_ids <= expected_ids:
        raise ValueError("evaluation subset is not contained in the matching resolved train/validation panel")
    if evaluation_subset.split == "test" and (evaluation_ids & (train_ids | validation_ids) or any(record.split != "test" for record in evaluation_subset.records)):
        raise ValueError("test evaluation IDs overlap train/validation or do not belong to the original test partition")
    device = _device_for(config)
    if device.type == "cpu":
        torch.set_num_threads(config.execution.cpu_threads)
    visual, model, class_indices = build_image_modules(dataset, config, class_ids=class_ids)
    visual.to(device)
    model.to(device)
    visual_hash = _hash_state(visual.state_dict())
    identity = _compatibility_identity(dataset, train_subset, validation_subset, config, class_ids, visual_hash)
    identity["common_neural_state_hash"] = _hash_state(model.common_state_dict())
    identity["classifier_family"] = model.family
    identity["pizza_mode"] = model.pizza_mode
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    payload = _load_checkpoint(checkpoint, identity, device)
    visual.load_state_dict(payload["visual_state"], strict=True)
    model.load_state_dict(payload["model_state"], strict=True)
    component_weights = _positive_weights(
        train_subset, dataset, class_ids, level="component",
        weighting=config.objective.positive_weighting, cap=config.objective.positive_weight_cap,
        beta=config.objective.effective_number_beta, require_coverage=config.purpose == "comparison",
    )
    pizza_weights = _positive_weights(
        train_subset, dataset, class_ids, level="pizza",
        weighting=config.objective.positive_weighting, cap=config.objective.positive_weight_cap,
        beta=config.objective.effective_number_beta, require_coverage=config.purpose == "comparison",
    )
    metrics = evaluate_image_model(
        visual,
        model,
        evaluation_subset,
        config,
        class_indices=class_indices,
        device=device,
        component_positive_weights=torch.tensor(component_weights, dtype=torch.float32, device=device),
        pizza_positive_weights=torch.tensor(pizza_weights, dtype=torch.float32, device=device),
    )
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_optimizer_step": int(payload["optimizer_step"]),
        "checkpoint_epoch": int(payload["epoch"]),
        "selected_class_ids": list(class_ids),
        "metrics": metrics,
        "training_protocol": "pre-detector exploratory GT-ROI; not detector-pretrained Phase 3 acceptance evidence",
    }


def evaluate_image_checkpoint(
    dataset: PreparedDataset,
    train_subset: DatasetSubset,
    validation_subset: DatasetSubset,
    evaluation_subset: DatasetSubset,
    config: ImageExperimentConfig,
    checkpoint_path: Path | str,
    *,
    class_ids: tuple[int, ...],
) -> dict[str, Any]:
    """Evaluate a saved checkpoint without changing global RNG/runtime settings."""
    rng = _rng_state()
    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    cpu_threads = torch.get_num_threads()
    try:
        return _evaluate_image_checkpoint(
            dataset,
            train_subset,
            validation_subset,
            evaluation_subset,
            config,
            checkpoint_path,
            class_ids=class_ids,
        )
    finally:
        _restore_rng(rng)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        if torch.get_num_threads() != cpu_threads:
            torch.set_num_threads(cpu_threads)
