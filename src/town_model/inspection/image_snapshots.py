"""Read-only numeric image/model traces; dashboard readers never load optimizers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from town_model.classifier_backend import raise_classifier_backend_unavailable
from town_model.models.geometry_diagnostics import GEOMETRY_DIAGNOSTIC_AXES

if TYPE_CHECKING:
    from town_model.dataset.image_data import DatasetSubset, PreparedDataset
    from town_model.experiments.image_config import ImageExperimentConfig
    from town_model.models.image_embeddings import ImageOntologyModel
    from town_model.vision.features import SharedVisualFeatureExtractor

SNAPSHOT_SCHEMA = "image-research-snapshot-v2"


@dataclass(frozen=True, slots=True)
class ImageResearchSnapshot:
    path: Path
    header: dict[str, Any]
    arrays: dict[str, np.ndarray]


def reference_panel(train: DatasetSubset, validation: DatasetSubset, maximum: int) -> tuple[tuple[str, str], ...]:
    """Stable balanced train/validation reference panel from existing subsets."""
    maximum = min(maximum, len(train.records) + len(validation.records))
    train_count = min(len(train.records), (maximum + 1) // 2)
    validation_count = min(len(validation.records), maximum - train_count)
    remaining = maximum - train_count - validation_count
    if remaining:
        extra_train = min(remaining, len(train.records) - train_count)
        train_count += extra_train
        remaining -= extra_train
    if remaining:
        validation_count += min(remaining, len(validation.records) - validation_count)
    selected = [("train", record.sample_id) for record in train.records[:train_count]]
    selected += [("validation", record.sample_id) for record in validation.records[:validation_count]]
    return tuple(selected)


def _numpy(value: torch.Tensor | None, *, dtype=None) -> np.ndarray | None:
    if value is None:
        return None
    result = value.detach().cpu().numpy()
    return result.astype(dtype, copy=False) if dtype is not None else result


def _classifier_geometry(model: ImageOntologyModel) -> dict[str, Any]:
    raise_classifier_backend_unavailable("historical image classifier geometry snapshot generation")
    from town_model.models.classifiers import BoxClassifier, NeuralBaseline, TownClassifier

    classifier = model.classifier
    if isinstance(classifier, NeuralBaseline):
        return {
            "family": "baseline",
            "weight": _numpy(classifier.linear.weight, dtype=np.float32).tolist(),
            "bias": _numpy(classifier.linear.bias, dtype=np.float32).tolist(),
            "boundary": "actual classifier logit == 0",
        }
    if isinstance(classifier, BoxClassifier):
        low, high = classifier.geometry.physical_boxes()
        return {
            "family": "box",
            "box_low": _numpy(low, dtype=np.float32).tolist(),
            "box_high": _numpy(high, dtype=np.float32).tolist(),
            "boundary": "closed physical box; membership includes boundary",
        }
    assert isinstance(classifier, TownClassifier)
    frame_low, frame_high, window_low, window_high = classifier.geometry.physical_geometry()
    return {
        "family": "town",
        "frame_low": _numpy(frame_low, dtype=np.float32).tolist(),
        "frame_high": _numpy(frame_high, dtype=np.float32).tolist(),
        "window_low": _numpy(window_low, dtype=np.float32).tolist(),
        "window_high": _numpy(window_high, dtype=np.float32).tolist(),
        "window_mode": classifier.geometry.window_mode,
        "boundary": "union of closed frame-minus-window houses, exact Phase 2 membership",
    }


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_image_snapshot(
    destination: Path | str,
    *,
    dataset: PreparedDataset,
    train_subset: DatasetSubset,
    validation_subset: DatasetSubset,
    visual: SharedVisualFeatureExtractor,
    model: ImageOntologyModel,
    config: ImageExperimentConfig,
    class_ids: tuple[int, ...],
    class_indices: tuple[int, ...],
    epoch: int,
    optimizer_step: int,
    latest_checkpoint: Path,
    identity_hash: str,
    feature_cache=None,
) -> ImageResearchSnapshot:
    """Persist a historical classifier inference trace (unavailable during reconstruction)."""
    raise_classifier_backend_unavailable("historical image classifier snapshot generation")
    from town_model.vision.batching import collate_image_records
    from town_model.vision.preprocessing import ImageNetLetterbox
    from town_model.vision.roi import GroundTruthROIProvider

    root = Path(destination).expanduser().resolve()
    if root.exists():
        existing = load_image_snapshot(root)
        checkpoint = Path(latest_checkpoint).expanduser().resolve()
        current_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if existing.header.get("identity_hash") != identity_hash or existing.header.get("optimizer_step") != optimizer_step or existing.header.get("source_checkpoint_sha256") != current_hash:
            raise FileExistsError(f"immutable image snapshot already exists with different source state: {root}")
        return existing
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    old_visual_mode, old_model_mode = visual.training, model.training
    python_rng = __import__("random").getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        visual.eval()
        model.eval()
        train_by_id = {record.sample_id: record for record in train_subset.records}
        validation_by_id = {record.sample_id: record for record in validation_subset.records}
        panel = reference_panel(train_subset, validation_subset, config.diagnostics.max_reference_samples)
        records = [train_by_id[sample_id] if split == "train" else validation_by_id[sample_id] for split, sample_id in panel]
        transform = ImageNetLetterbox(config.vision.resolution)
        selected_names = tuple(model.class_names)
        arrays_out: dict[str, list[np.ndarray]] = {
            key: [] for key in (
                "sample_ids", "source_splits", "sample_image_paths", "component_categories", "original_sizes_hw", "processed_sizes_hw", "component_offsets",
                "component_ids", "component_detector_category_ids", "component_boxes_xyxy_original",
                "component_points", "component_contains", "component_targets", "component_known_masks",
                "component_signed_margin", "component_logits", "pizza_points_global", "pizza_points_composed",
                "pizza_points_final", "pizza_contains", "pizza_targets", "pizza_known_masks",
                "pizza_signed_margin", "pizza_logits", "roi_feature_means", "normalized_roi_coordinates",
                "global_spatial_features",
            )
        }
        arrays_out["component_offsets"].append(np.asarray([0], dtype=np.int64))
        component_count_total = 0
        classifier_geometry_values: dict[str, np.ndarray] = {}
        diag_component_values: dict[str, list[np.ndarray]] = {}
        diag_pizza_values: dict[str, list[np.ndarray]] = {}
        torch_device = next(model.parameters()).device
        for start in range(0, len(records), config.training.physical_batch_size):
            batch_records = records[start : start + config.training.physical_batch_size]
            batch = collate_image_records(dataset, batch_records, transform=transform)
            batch = type(batch)(
                images=batch.images.to(torch_device),
                original_image_sizes=batch.original_image_sizes,
                processed_image_sizes=batch.processed_image_sizes,
                image_geometries=batch.image_geometries,
                sample_ids=batch.sample_ids,
                gt_component_boxes_xyxy=tuple(value.to(torch_device) for value in batch.gt_component_boxes_xyxy),
                component_ids=batch.component_ids,
                component_detector_category_ids=batch.component_detector_category_ids,
                pizza_targets=batch.pizza_targets.to(torch_device),
                pizza_known_masks=batch.pizza_known_masks.to(torch_device),
                component_targets=tuple(value.to(torch_device) for value in batch.component_targets),
                component_known_masks=tuple(value.to(torch_device) for value in batch.component_known_masks),
                split_identity=batch.split_identity,
            )
            with torch.no_grad():
                if feature_cache is not None:
                    features = feature_cache.get(batch, diagnostics=config.diagnostics.include_roi_feature_means)
                    features = type(features)(
                        component_vectors=features.component_vectors.to(torch_device),
                        component_valid_mask=features.component_valid_mask.to(torch_device),
                        global_spatial_features=features.global_spatial_features.to(torch_device),
                        component_ids=features.component_ids,
                        diagnostics=(
                            type(features.diagnostics)(
                                roi_features=features.diagnostics.roi_features.to(torch_device) if features.diagnostics.roi_features is not None else None,
                                global_features=features.diagnostics.global_features.to(torch_device) if features.diagnostics.global_features is not None else None,
                                roi_source=features.diagnostics.roi_source,
                                normalized_roi_coordinates=features.diagnostics.normalized_roi_coordinates.to(torch_device) if features.diagnostics.normalized_roi_coordinates is not None else None,
                            ) if features.diagnostics is not None else None
                        ),
                    )
                else:
                    features = visual(
                        batch, GroundTruthROIProvider(), diagnostics=config.diagnostics.include_roi_feature_means
                    )
                output = model(features, batch.sample_ids, diagnostics=True)
            for scoped_output in (output.component_classifier, output.pizza_classifier):
                for name, value in (scoped_output.diagnostics or {}).items():
                    axes = GEOMETRY_DIAGNOSTIC_AXES.get(name)
                    if axes is None:
                        raise ValueError(f"classifier diagnostic {name!r} has no declared semantic axes")
                    if axes[0] == "class":
                        parameter = _numpy(value)
                        previous = classifier_geometry_values.get(name)
                        if previous is not None and not np.array_equal(previous, parameter):
                            raise ValueError(f"class-level geometry diagnostic {name!r} changed across snapshot batches")
                        classifier_geometry_values[name] = parameter
            for row, split in enumerate(batch.split_identity):
                count_total = int(output.component_valid_mask[row].sum().item())
                cap = config.diagnostics.max_component_traces_per_sample
                count = min(count_total, cap)
                ids = batch.component_ids[row][:count]
                model_columns = torch.arange(len(model.class_names), dtype=torch.long, device=torch_device)
                target_columns = torch.tensor(class_indices, dtype=torch.long, device=torch_device)
                arrays_out["sample_ids"].append(np.asarray(batch.sample_ids[row]))
                arrays_out["source_splits"].append(np.asarray(split))
                arrays_out["sample_image_paths"].append(np.asarray(batch_records[row].image_path))
                arrays_out["original_sizes_hw"].append(np.asarray(batch.original_image_sizes[row], dtype=np.int32))
                arrays_out["processed_sizes_hw"].append(np.asarray(batch.processed_image_sizes[row], dtype=np.int32))
                arrays_out["component_ids"].extend(np.asarray(value) for value in ids)
                arrays_out["component_categories"].extend(np.asarray(batch_records[row].components[index].detector_category) for index in range(count))
                arrays_out["component_detector_category_ids"].extend(np.asarray(value, dtype=np.int32) for value in batch.component_detector_category_ids[row][:count])
                arrays_out["component_boxes_xyxy_original"].extend(_numpy(batch.gt_component_boxes_xyxy[row][:count], dtype=np.float32))
                arrays_out["component_points"].extend(_numpy(output.component_points[row, :count], dtype=np.float32))
                arrays_out["component_contains"].extend(_numpy(output.component_classifier.contains[row, :count].index_select(1, model_columns), dtype=np.bool_))
                arrays_out["component_targets"].extend(_numpy(batch.component_targets[row][:count].index_select(1, target_columns), dtype=np.bool_))
                arrays_out["component_known_masks"].extend(_numpy(batch.component_known_masks[row][:count].index_select(1, target_columns), dtype=np.bool_))
                if output.component_classifier.signed_margin is not None:
                    arrays_out["component_signed_margin"].extend(_numpy(output.component_classifier.signed_margin[row, :count].index_select(1, model_columns), dtype=np.float32))
                if output.component_classifier.logits is not None:
                    arrays_out["component_logits"].extend(_numpy(output.component_classifier.logits[row, :count].index_select(1, model_columns), dtype=np.float32))
                arrays_out["pizza_points_global"].append(_numpy(output.global_pizza_points[row], dtype=np.float32))
                arrays_out["pizza_points_composed"].append(_numpy(output.composed_pizza_points[row], dtype=np.float32))
                arrays_out["pizza_points_final"].append(_numpy(output.final_pizza_points[row], dtype=np.float32))
                arrays_out["pizza_contains"].append(_numpy(output.pizza_classifier.contains[row].index_select(0, model_columns), dtype=np.bool_))
                arrays_out["pizza_targets"].append(_numpy(batch.pizza_targets[row].index_select(0, target_columns), dtype=np.bool_))
                arrays_out["pizza_known_masks"].append(_numpy(batch.pizza_known_masks[row].index_select(0, target_columns), dtype=np.bool_))
                if output.pizza_classifier.signed_margin is not None:
                    arrays_out["pizza_signed_margin"].append(_numpy(output.pizza_classifier.signed_margin[row].index_select(0, model_columns), dtype=np.float32))
                if output.pizza_classifier.logits is not None:
                    arrays_out["pizza_logits"].append(_numpy(output.pizza_classifier.logits[row].index_select(0, model_columns), dtype=np.float32))
                if config.diagnostics.include_roi_feature_means:
                    arrays_out["roi_feature_means"].extend(_numpy(features.component_vectors[row, :count, :256], dtype=np.float32))
                    arrays_out["normalized_roi_coordinates"].extend(_numpy(features.component_vectors[row, :count, 256:], dtype=np.float32))
                    arrays_out["global_spatial_features"].append(_numpy(features.global_spatial_features[row], dtype=np.float32))
                component_count_total += count
                arrays_out["component_offsets"].append(np.asarray([component_count_total], dtype=np.int64))
                for name, value in (output.component_classifier.diagnostics or {}).items():
                    axes = GEOMETRY_DIAGNOSTIC_AXES[name]
                    if axes[0] == "point":
                        diag_component_values.setdefault(name, []).append(_numpy(value[row, :count]))
                for name, value in (output.pizza_classifier.diagnostics or {}).items():
                    axes = GEOMETRY_DIAGNOSTIC_AXES[name]
                    if axes[0] == "point":
                        diag_pizza_values.setdefault(name, []).append(_numpy(value[row]))
        arrays: dict[str, np.ndarray] = {}
        for key, values in arrays_out.items():
            if not values:
                continue
            if key == "component_offsets":
                arrays[key] = np.concatenate(values).astype(np.int64)
            elif key in {"sample_ids", "source_splits", "sample_image_paths", "component_categories", "component_ids"}:
                arrays[key] = np.asarray(values, dtype=np.str_)
            else:
                arrays[key] = np.asarray(values)
        arrays["component_count_total"] = np.asarray(component_count_total, dtype=np.int64)
        diagnostic_axes: dict[str, list[str]] = {}
        for name, value in classifier_geometry_values.items():
            key = f"classifier_geometry_{name}"
            arrays[key] = value
            diagnostic_axes[key] = list(GEOMETRY_DIAGNOSTIC_AXES[name])
        for name, values in diag_component_values.items():
            key = f"component_activity_{name}"
            arrays[key] = np.concatenate(values, axis=0)
            diagnostic_axes[key] = ["component_occurrence", *GEOMETRY_DIAGNOSTIC_AXES[name][1:]]
        for name, values in diag_pizza_values.items():
            key = f"pizza_activity_{name}"
            arrays[key] = np.stack(values, axis=0)
            diagnostic_axes[key] = ["image", *GEOMETRY_DIAGNOSTIC_AXES[name][1:]]
        _atomic_save_npz(staging / "arrays.npz", arrays)
        checkpoint = Path(latest_checkpoint).expanduser().resolve()
        header = {
            "schema_version": SNAPSHOT_SCHEMA,
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "identity_hash": identity_hash,
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "dataset_id": dataset.dataset_id,
            "manifest_id": dataset.manifest_id,
            "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
            "class_manifest_sha256": dataset.class_manifest_sha256,
            "dataset_root": str(dataset.root),
            "sample_ids": [sample_id for _, sample_id in panel],
            "source_splits": [split for split, _ in panel],
            "train_subset_id": train_subset.subset_id,
            "validation_subset_id": validation_subset.subset_id,
            "class_ids": list(class_ids),
            "class_names": list(selected_names),
            "family": model.family,
            "pizza_mode": model.pizza_mode,
            "latent_dimensions": model.dimensions,
            "roi_source": config.vision.roi_source,
            "preprocessing": {"resolution": config.vision.resolution, "name": "rgb_letterbox_imagenet_normalize_v1"},
            "component_cap_per_sample": config.diagnostics.max_component_traces_per_sample,
            "component_count_original": sum(len(record.components) for record in records),
            "component_count_traced": component_count_total,
            "component_feature_means_included": config.diagnostics.include_roi_feature_means,
            "training_protocol": "pre-detector exploratory GT-ROI; not detector-pretrained Pizzaïolo model",
            "geometry": _classifier_geometry(model),
            "diagnostic_axes_schema": diagnostic_axes,
            "arrays_sha256": hashlib.sha256((staging / "arrays.npz").read_bytes()).hexdigest(),
            "feature_cache_id": getattr(feature_cache, "cache_id", None),
        }
        (staging / "header.json").write_text(json.dumps(header, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        visual.train(old_visual_mode)
        model.train(old_model_mode)
        __import__("random").setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)
    return load_image_snapshot(root)


def load_image_snapshot(path: Path | str) -> ImageResearchSnapshot:
    """Validate and load only numeric research arrays and their JSON header."""
    root = Path(path).expanduser().resolve()
    try:
        header = json.loads((root / "header.json").read_text(encoding="utf-8"))
        arrays_path = root / "arrays.npz"
        if header.get("schema_version") != SNAPSHOT_SCHEMA:
            raise ValueError(
                f"unsupported snapshot schema {header.get('schema_version')!r}; "
                f"expected {SNAPSHOT_SCHEMA} with explicit diagnostic axes"
            )
        digest = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        if digest != header.get("arrays_sha256"):
            raise ValueError("snapshot numeric array checksum mismatch")
        with np.load(arrays_path, allow_pickle=False) as stored:
            arrays = {name: stored[name].copy() for name in stored.files}
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
        raise ValueError(f"cannot read image research snapshot at {root}: {error}") from error
    sample_count = len(arrays.get("sample_ids", ()))
    offsets = arrays.get("component_offsets")
    if offsets is None or offsets.shape != (sample_count + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("image snapshot has invalid component offsets")
    if int(offsets[-1]) != int(arrays.get("component_count_total", -1)):
        raise ValueError("image snapshot component offsets do not match traced occurrence count")
    if tuple(arrays.get("sample_ids", ())) != tuple(header.get("sample_ids", ())):
        raise ValueError("snapshot sample IDs do not match the validated header")
    axis_schema = header.get("diagnostic_axes_schema", {})
    if not isinstance(axis_schema, dict):
        raise ValueError("snapshot diagnostic axis schema is malformed")
    for key, axes in axis_schema.items():
        if key not in arrays or not isinstance(axes, list) or arrays[key].ndim != len(axes):
            raise ValueError(f"snapshot diagnostic {key!r} does not match its declared axes")
    for diagnostic_name, header_name in (
        ("box_low", "box_low"), ("box_high", "box_high"),
        ("frame_low", "frame_low"), ("frame_high", "frame_high"),
        ("window_low", "window_low"), ("window_high", "window_high"),
    ):
        array_key = f"classifier_geometry_{diagnostic_name}"
        if array_key in arrays and not np.array_equal(arrays[array_key], np.asarray(header["geometry"][header_name])):
            raise ValueError(f"snapshot class geometry diagnostic {diagnostic_name!r} differs from saved geometry")
    return ImageResearchSnapshot(root, header, arrays)


def discover_image_snapshots(run_directory: Path | str) -> list[Path]:
    root = Path(run_directory).expanduser().resolve() / "snapshots"
    if not root.is_dir():
        return []
    snapshots = []
    for candidate in sorted(root.glob("epoch-*")):
        if candidate.is_dir() and (candidate / "header.json").is_file() and (candidate / "arrays.npz").is_file():
            try:
                load_image_snapshot(candidate)
            except ValueError:
                continue
            snapshots.append(candidate)
    return snapshots
