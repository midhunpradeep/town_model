"""Deterministic synthetic fixtures, paired training, run persistence and reload."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import pickle
import sys
import time
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from town_model.models import (
    PointClassifier,
    SyntheticPointEncoder,
    make_classifier,
    reduce_supervision,
    unreduced_penalties,
    validate_min_width,
)

MODEL_FAMILIES = ("baseline", "box", "town")


class SavedArtifactError(ValueError):
    """A persisted PyTorch artifact is unreadable or structurally incompatible."""


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    seed: int = 42
    points: int = 320
    dimensions: int = 2
    steps: int = 60
    learning_rate: float = 0.03
    margin: float = 0.1
    train_fraction: float = 0.7
    unknown_fraction: float = 0.04
    houses: int = 2
    windows: int = 1
    min_width: float = 1e-5
    window_mode: str = "contained"
    window_target_volume: float = 0.25
    snapshot_interval: int = 10

    def validate(self) -> None:
        if self.seed < 0 or self.points < 20 or self.dimensions < 2 or self.steps < 1:
            raise ValueError("seed>=0, points>=20, dimensions>=2, and steps>=1 are required")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be finite and nonnegative")
        if not 0.5 <= self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in [0.5,1)")
        if not 0 <= self.unknown_fraction < 1:
            raise ValueError("unknown_fraction must be in [0,1)")
        if self.houses < 1 or self.windows < 1 or self.snapshot_interval < 1:
            raise ValueError("houses, windows, and snapshot_interval must be positive")
        if self.window_mode != "contained":
            raise ValueError("Phase 2 supports only window_mode='contained'")
        if not np.isfinite(self.window_target_volume) or not 0.0 < self.window_target_volume < 1.0:
            raise ValueError("window_target_volume must be finite and in (0,1)")
        validate_min_width(
            self.min_width,
            model_families=MODEL_FAMILIES,
            window_mode=self.window_mode,
        )


@dataclass(frozen=True, slots=True)
class SyntheticFixture:
    fixture_id: str
    kind: Literal["fixed", "mapping"]
    scenario: str
    latent_dimensions: int
    features: Tensor
    targets: Tensor
    known_mask: Tensor
    train_indices: Tensor
    validation_indices: Tensor
    class_names: tuple[str, ...]
    seed: int
    teacher_description: str

    @property
    def input_dimensions(self) -> int:
        return int(self.features.shape[1])


def _target_rule(points: Tensor, scenario: str) -> tuple[Tensor, tuple[str, ...]]:
    """Construct supervised labels only; returned labels never enter model input."""
    if scenario == "simple":
        labels = torch.stack((points[:, 0] >= 0.0, points[:, 0] + points[:, 1] >= 0.0), dim=1)
        return labels.float(), ("right_half", "positive_diagonal")
    if scenario == "window":
        in_frame = (points[:, 0].abs() <= 0.82) & (points[:, 1].abs() <= 0.82)
        in_closed_window = (points[:, 0].abs() <= 0.30) & (points[:, 1].abs() <= 0.30)
        in_lobe = (points[:, 0].abs() >= 0.55) & (points[:, 1].abs() <= 0.65)
        targets = torch.stack((in_frame & ~in_closed_window, in_lobe), dim=1)
        return targets.float(), ("frame_minus_window", "two_outer_lobes")
    raise ValueError(f"unknown synthetic scenario: {scenario}")


def make_fixture(
    kind: Literal["fixed", "mapping"],
    scenario: str,
    *,
    config: SyntheticConfig,
    input_dimensions: int = 4,
) -> tuple[SyntheticFixture, dict[str, Tensor] | None]:
    """Make a seeded shared fixture and optional teacher-only mapping state."""
    config.validate()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    if kind == "fixed":
        features = torch.rand((config.points, config.dimensions), generator=generator) * 2.0 - 1.0
        teacher_points = features
        teacher_state = None
        teacher_description = "fixed bounded points; classifier-only training"
    elif kind == "mapping":
        if input_dimensions < 2:
            raise ValueError("mapping input_dimensions must be >=2")
        features = torch.rand((config.points, input_dimensions), generator=generator) * 2.0 - 1.0
        weight = torch.randn((input_dimensions, config.dimensions), generator=generator) / (input_dimensions**0.5)
        bias = torch.randn((config.dimensions,), generator=generator) * 0.15
        teacher_points = torch.tanh(features @ weight + bias)
        teacher_state = {"weight": weight, "bias": bias}
        teacher_description = "supervision derived from a private deterministic tanh teacher mapping; raw features only are model inputs"
    else:
        raise ValueError("kind must be 'fixed' or 'mapping'")

    targets, class_names = _target_rule(teacher_points, scenario)
    known_mask = torch.rand(targets.shape, generator=generator) >= config.unknown_fraction
    permutation = torch.randperm(config.points, generator=generator)
    train_count = int(round(config.points * config.train_fraction))
    train_indices = permutation[:train_count].sort().values
    validation_indices = permutation[train_count:].sort().values
    fixture_arrays = {
        "features": features.to(torch.float32).numpy(),
        "targets": targets.to(torch.float32).numpy(),
        "known_mask": known_mask.to(torch.bool).numpy(),
        "train_indices": train_indices.numpy(),
        "validation_indices": validation_indices.numpy(),
    }
    content_hash = _fixture_arrays_hash(fixture_arrays)
    identity = {
        "kind": kind,
        "scenario": scenario,
        "seed": config.seed,
        "points": config.points,
        "dimensions": config.dimensions,
        "input_dimensions": features.shape[1],
        "class_names": class_names,
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "content_sha256": content_hash,
    }
    fixture_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    fixture = SyntheticFixture(
        fixture_id=fixture_id,
        kind=kind,
        scenario=scenario,
        latent_dimensions=config.dimensions,
        features=features.to(torch.float32),
        targets=targets.to(torch.float32),
        known_mask=known_mask.to(torch.bool),
        train_indices=train_indices,
        validation_indices=validation_indices,
        class_names=class_names,
        seed=config.seed,
        teacher_description=teacher_description,
    )
    return fixture, teacher_state


def _fixture_arrays_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ("features", "targets", "known_mask", "train_indices", "validation_indices"):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _state_hash(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("town-embeddings-research", "matplotlib", "streamlit", "pandas", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _clone_state(module: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _write_fixture(group_dir: Path, fixture: SyntheticFixture) -> None:
    group_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        group_dir / "fixture.npz",
        features=fixture.features.numpy(),
        targets=fixture.targets.numpy(),
        known_mask=fixture.known_mask.numpy(),
        train_indices=fixture.train_indices.numpy(),
        validation_indices=fixture.validation_indices.numpy(),
    )
    _json_write(
        group_dir / "fixture.json",
        {
            "fixture_id": fixture.fixture_id,
            "kind": fixture.kind,
            "scenario": fixture.scenario,
            "latent_dimensions": fixture.latent_dimensions,
            "seed": fixture.seed,
            "class_names": fixture.class_names,
            "teacher_description": fixture.teacher_description,
            "split_identity": {
                "train_count": int(fixture.train_indices.numel()),
                "validation_count": int(fixture.validation_indices.numel()),
                "split_hash": hashlib.sha256(
                    fixture.train_indices.numpy().tobytes() + fixture.validation_indices.numpy().tobytes()
                ).hexdigest(),
            },
            "dataset_identifier": fixture.fixture_id,
            "content_sha256": _fixture_arrays_hash({
                "features": fixture.features.numpy(),
                "targets": fixture.targets.numpy(),
                "known_mask": fixture.known_mask.numpy(),
                "train_indices": fixture.train_indices.numpy(),
                "validation_indices": fixture.validation_indices.numpy(),
            }),
        },
    )


def _make_encoder(fixture: SyntheticFixture, seed: int) -> SyntheticPointEncoder | None:
    if fixture.kind == "fixed":
        return None
    return SyntheticPointEncoder(fixture.input_dimensions, fixture.latent_dimensions, seed=seed)


def _model_settings(model: PointClassifier, config: SyntheticConfig, seed: int) -> dict[str, Any]:
    return {
        "family": model.family,
        "seed": seed,
        "class_names": list(model.class_names),
        "fixed_positive_classes": list(model.fixed_classes),
        "latent_dimensions": model.dimensions,
        "loss_mode": "bce" if model.family == "baseline" else "signed_distance",
        "margin": config.margin if model.family != "baseline" else None,
        "houses": config.houses if model.family == "town" else None,
        "windows_per_house": config.windows if model.family == "town" else None,
        "window_mode": config.window_mode if model.family == "town" else None,
        "window_target_volume": config.window_target_volume if model.family == "town" else None,
        "minimum_axis_width": config.min_width if model.family != "baseline" else None,
    }


def _subset_tensors(fixture: SyntheticFixture, indices: Tensor, *, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        fixture.features[indices].to(device),
        fixture.targets[indices].to(device),
        fixture.known_mask[indices].to(device),
    )


def _points_for(encoder: SyntheticPointEncoder | None, features: Tensor) -> Tensor:
    return features if encoder is None else encoder(features)


def _metric_counts(targets: Tensor, known: Tensor) -> dict[str, int]:
    return {
        "known": int(known.sum().item()),
        "positive": int(((targets == 1) & known).sum().item()),
        "negative": int(((targets == 0) & known).sum().item()),
        "unknown": int((~known).sum().item()),
    }


def _classification_metrics(output, targets: Tensor, known: Tensor, class_names: tuple[str, ...]) -> dict[str, Any]:
    prediction = output.contains
    per_class: dict[str, Any] = {}
    f1_supported: list[float] = []
    correct = (prediction == targets.bool()) & known
    for class_index, class_name in enumerate(class_names):
        class_known = known[:, class_index]
        truth = targets[:, class_index].bool()
        predicted = prediction[:, class_index]
        tp = int((class_known & truth & predicted).sum().item())
        fp = int((class_known & ~truth & predicted).sum().item())
        fn = int((class_known & truth & ~predicted).sum().item())
        support = int((class_known & truth).sum().item())
        f1 = 2.0 * tp / max(1, 2 * tp + fp + fn)
        per_class[class_name] = {
            "known": int(class_known.sum().item()),
            "positive_support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "accuracy": float(correct[:, class_index].sum().item() / max(1, int(class_known.sum().item()))),
            "f1": f1,
            "macro_included": support > 0,
        }
        if support > 0:
            f1_supported.append(f1)
    return {
        "known_label_accuracy": float(correct.sum().item() / max(1, int(known.sum().item()))),
        "macro_f1": float(np.mean(f1_supported)) if f1_supported else None,
        "per_class": per_class,
        "counts": _metric_counts(targets, known),
    }


def _loss_for(model: PointClassifier, output, targets: Tensor, known: Tensor, margin: float):
    mode = "bce" if model.family == "baseline" else "signed_distance"
    penalties = unreduced_penalties(output, targets, mode=mode, margin=margin)
    return reduce_supervision(penalties, targets, known), mode


def _configured_margin(config: dict[str, Any]) -> float:
    """Resolve an omitted margin to the documented default without erasing zero."""
    value = config.get("margin")
    return 0.1 if value is None else float(value)


def _evaluate_split(
    model: PointClassifier,
    encoder: SyntheticPointEncoder | None,
    fixture: SyntheticFixture,
    indices: Tensor,
    *,
    margin: float,
    device: torch.device,
) -> tuple[dict[str, Any], Tensor]:
    features, targets, known = _subset_tensors(fixture, indices, device=device)
    model.eval()
    if encoder is not None:
        encoder.eval()
    with torch.no_grad():
        points = _points_for(encoder, features)
        output = model(points)
        reduction, mode = _loss_for(model, output, targets, known, margin)
    metric = _classification_metrics(output, targets, known, model.class_names)
    metric.update({"loss": float(reduction.loss.item()), "objective": mode, "supervised_instances": int(reduction.supervised_instances.sum().item())})
    return metric, points.detach().cpu()


def _gradient_report(model: PointClassifier, encoder: SyntheticPointEncoder | None) -> dict[str, Any]:
    def group(parameters) -> dict[str, Any]:
        values = [parameter.grad.detach() for parameter in parameters if parameter.requires_grad and parameter.grad is not None]
        if not values:
            return {"norm_l2": 0.0, "parameters_with_grad": 0, "zero_fraction": 1.0}
        squared = sum(float(torch.sum(value.float() ** 2).item()) for value in values)
        count = sum(value.numel() for value in values)
        zero = sum(int((value == 0).sum().item()) for value in values)
        return {"norm_l2": float(squared**0.5), "parameters_with_grad": len(values), "zero_fraction": float(zero / max(count, 1))}

    if model.family == "town":
        geometry = model.geometry
        result = {
            "frames": group((geometry.raw_frame_low, geometry.raw_frame_width)),
            "windows": group((geometry.raw_window_position, geometry.raw_window_width)),
        }
    elif model.family == "box":
        result = {"geometry": group(model.geometry.parameters())}
    else:
        result = {"classifier": group(model.parameters())}
    result["point_encoder"] = group(encoder.parameters()) if encoder is not None else None
    return result


def _geometry_activity(model: PointClassifier, points: Tensor) -> dict[str, Any]:
    with torch.no_grad():
        output = model(points, diagnostics=True)
    if model.family == "baseline":
        return {"region_occupancy_fraction": output.contains.float().mean(dim=0).cpu().tolist(), "geometry": None}
    if model.family == "box":
        low, high = model.geometry.physical_boxes()
        return {
            "region_occupancy_fraction": output.contains.float().mean(dim=0).cpu().tolist(),
            "geometry": {"box_low": low.detach().cpu().tolist(), "box_high": high.detach().cpu().tolist()},
        }
    details = output.diagnostics or {}
    wl, wr = details["window_low"], details["window_high"]
    fl, fr = details["frame_low"], details["frame_high"]
    house_obs = details["house_contains"].sum(dim=0).cpu()
    window_obs = details["window_contains"].sum(dim=0).cpu()
    return {
        "region_occupancy_fraction": output.contains.float().mean(dim=0).cpu().tolist(),
        "geometry": {
            "frame_low": fl.detach().cpu().tolist(),
            "frame_high": fr.detach().cpu().tolist(),
            "window_low": wl.detach().cpu().tolist(),
            "window_high": wr.detach().cpu().tolist(),
        },
        "activity": {
            "house_membership_observations": house_obs.tolist(),
            "window_membership_observations": window_obs.tolist(),
            "inactive_houses": int((house_obs == 0).sum().item()),
            "inactive_windows": int((window_obs == 0).sum().item()),
        },
    }


def _snapshot(
    run_dir: Path,
    name: str,
    model: PointClassifier,
    encoder: SyntheticPointEncoder | None,
    fixture: SyntheticFixture,
    *,
    step: int,
) -> None:
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    if encoder is not None:
        encoder.eval()
    with torch.no_grad():
        train_points = _points_for(encoder, fixture.features[fixture.train_indices])
        validation_points = _points_for(encoder, fixture.features[fixture.validation_indices])
        train_output = model(train_points, diagnostics=True)
        validation_output = model(validation_points)
    payload = {
        "step": step,
        "activity": _geometry_activity(model, train_points),
        "activity_split": "train",
        "model_state": _clone_state(model),
        "encoder_state": _clone_state(encoder) if encoder is not None else None,
        "train_points": train_points.detach().cpu(),
        "validation_points": validation_points.detach().cpu(),
        "train_contains": train_output.contains.detach().cpu(),
        "validation_contains": validation_output.contains.detach().cpu(),
        "train_signed_margin": train_output.signed_margin.detach().cpu() if train_output.signed_margin is not None else None,
        "validation_signed_margin": validation_output.signed_margin.detach().cpu() if validation_output.signed_margin is not None else None,
        "train_logits": train_output.logits.detach().cpu() if train_output.logits is not None else None,
        "validation_logits": validation_output.logits.detach().cpu() if validation_output.logits is not None else None,
    }
    destination = snapshot_dir / f"{name}.pt"
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_torch_artifact(path: Path, artifact_name: str) -> Any:
    """Deserialize one trusted local PyTorch artifact and translate file failures."""
    try:
        # Checkpoints use Python pickle: only load artifacts from trusted run directories.
        return torch.load(path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        raise
    except (pickle.UnpicklingError, EOFError, RuntimeError, OSError) as error:
        raise SavedArtifactError(
            f"could not read or deserialize {artifact_name} at {path}: "
            f"{type(error).__name__}: {error}"
        ) from error


def _require_artifact_mapping(payload: Any, path: Path, artifact_name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SavedArtifactError(f"incompatible {artifact_name} at {path}: expected a mapping payload")
    return payload


def _run_one(
    group_dir: Path,
    fixture: SyntheticFixture,
    family: str,
    config: SyntheticConfig,
    *,
    device: torch.device,
    base_encoder_state: dict[str, Tensor] | None,
    encoder_seed: int,
) -> dict[str, Any]:
    run_dir = group_dir / family
    run_dir.mkdir(parents=True, exist_ok=False)
    model = make_classifier(
        family, config.dimensions, fixture.class_names,
        seed=config.seed, houses=config.houses, windows=config.windows,
        min_width=config.min_width, window_mode=config.window_mode,
        window_target_volume=config.window_target_volume,
    ).to(device)
    encoder = _make_encoder(fixture, encoder_seed)
    if encoder is not None:
        assert base_encoder_state is not None
        encoder.load_state_dict(base_encoder_state)
        encoder.to(device)
    model_initial = _clone_state(model)
    encoder_initial = _clone_state(encoder) if encoder is not None else None
    objective = "bce" if family == "baseline" else "signed_distance"
    resolved = {
        "schema_version": 2,
        "experiment": {"kind": fixture.kind, "scenario": fixture.scenario, "seed": config.seed, "dataset_id": fixture.fixture_id},
        "model": _model_settings(model, config, config.seed),
        "training": {
            "steps": config.steps,
            "learning_rate": config.learning_rate,
            "optimizer": "Adam",
            "batching": "full train split",
            "objective": objective,
            "margin": config.margin if objective == "signed_distance" else None,
            "class_weighting": "none",
            "positive_weights": None,
            "supervision_reduction": "per-instance mean over known trainable classes; unknowns excluded",
            "device": str(device),
            "snapshot_interval": config.snapshot_interval,
        },
        "data": {
            "fixture_id": fixture.fixture_id,
            "split_identity": json.loads((group_dir / "fixture.json").read_text(encoding="utf-8"))["split_identity"],
            "class_names": fixture.class_names,
            "unknown_fraction": config.unknown_fraction,
        },
    }
    _json_write(run_dir / "resolved_config.json", resolved)
    _json_write(run_dir / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "dependencies": _dependency_versions(),
        "device": str(device),
    })
    _atomic_torch_save({"model_state": model_initial, "encoder_state": encoder_initial}, run_dir / "initial_state.pt")
    _snapshot(run_dir, "initial", model, encoder, fixture, step=0)

    parameters = list(model.parameters()) + (list(encoder.parameters()) if encoder is not None else [])
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    train_features, train_targets, train_known = _subset_tensors(fixture, fixture.train_indices, device=device)
    history: list[dict[str, Any]] = []
    gradient_last: dict[str, Any] = {}
    started = time.time()

    for step in range(1, config.steps + 1):
        model.train()
        if encoder is not None:
            encoder.train()
        optimizer.zero_grad(set_to_none=True)
        train_points = _points_for(encoder, train_features)
        output = model(train_points)
        reduction, _ = _loss_for(model, output, train_targets, train_known, config.margin)
        loss = reduction.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training objective at {family} step {step}")
        loss.backward()
        gradient_last = _gradient_report(model, encoder)
        optimizer.step()

        if step == 1 or step % max(1, config.snapshot_interval) == 0 or step == config.steps:
            train_metric, _ = _evaluate_split(
                model, encoder, fixture, fixture.train_indices, margin=config.margin, device=device
            )
            val_metric, _ = _evaluate_split(
                model, encoder, fixture, fixture.validation_indices, margin=config.margin, device=device
            )
            history.append({"step": step, "training": train_metric, "validation": val_metric, "gradient_norms": gradient_last})
            _snapshot(run_dir, f"step_{step:05d}", model, encoder, fixture, step=step)

    train_metric, train_points = _evaluate_split(model, encoder, fixture, fixture.train_indices, margin=config.margin, device=device)
    val_metric, val_points = _evaluate_split(model, encoder, fixture, fixture.validation_indices, margin=config.margin, device=device)
    activity = _geometry_activity(model, train_points.to(device))
    final_snapshot = run_dir / "snapshots" / f"step_{config.steps:05d}.pt"
    if not final_snapshot.exists():
        _snapshot(run_dir, f"step_{config.steps:05d}", model, encoder, fixture, step=config.steps)
    encoder_hash = _state_hash(encoder_initial) if encoder_initial is not None else None
    metadata = {
        "run_id": f"{group_dir.name}/{family}",
        "family": family,
        "objective": objective,
        "fixture_id": fixture.fixture_id,
        "seed": config.seed,
        "class_names": list(fixture.class_names),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()) + (sum(parameter.numel() for parameter in encoder.parameters()) if encoder is not None else 0),
        "classifier_parameter_count": model.trainable_parameter_count,
        "encoder_parameter_count": sum(parameter.numel() for parameter in encoder.parameters()) if encoder is not None else 0,
        "initialization_info": (
            model.geometry.initialization_info
            if hasattr(model, "geometry")
            else {"seed": config.seed, "weight_initialization": "torch Linear default under isolated seed"}
        ),
        "shared_initial_encoder_sha256": encoder_hash,
        "settings": resolved,
        "training_metrics": train_metric,
        "validation_metrics": val_metric,
        "gradient_norms_last_step": gradient_last,
        "geometry_activity": activity,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "snapshot_names": sorted(path.stem for path in (run_dir / "snapshots").glob("*.pt")),
    }
    _json_write(run_dir / "metrics.json", metadata)
    with (run_dir / "history.jsonl").open("w", encoding="utf-8") as stream:
        for row in history:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    checkpoint = {
        "checkpoint_schema": 2,
        "family": family,
        "dimensions": config.dimensions,
        "class_names": fixture.class_names,
        "seed": config.seed,
        "model_config": {
            "houses": config.houses,
            "windows": config.windows,
            "min_width": config.min_width,
            "window_mode": config.window_mode,
            "window_target_volume": config.window_target_volume,
        },
        "encoder_config": None if encoder is None else {
            "input_dimensions": fixture.input_dimensions,
            "latent_dimensions": config.dimensions,
            "seed": encoder_seed,
            "hidden_dimensions": 16,
            "initial_sha256": encoder_hash,
        },
        "model_state": _clone_state(model),
        "encoder_state": _clone_state(encoder) if encoder is not None else None,
        "initial_model_state": model_initial,
        "initial_encoder_state": encoder_initial,
        "optimizer_state": optimizer.state_dict(),
        "step": config.steps,
        "config_identity": hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest(),
        "fixture_id": fixture.fixture_id,
    }
    _atomic_torch_save(checkpoint, run_dir / "checkpoint.pt")
    return metadata


def run_experiment(
    kind: Literal["fixed", "mapping"],
    scenario: str,
    output_root: Path | str,
    *,
    config: SyntheticConfig | None = None,
    device: str = "cpu",
    input_dimensions: int = 4,
) -> Path:
    """Train all three families on one shared fixture and persist a run group."""
    resolved_config = config or SyntheticConfig()
    resolved_config.validate()
    if device != "cpu":
        raise ValueError("Phase 2 experiment harness currently supports CPU only")
    output = Path(output_root).expanduser().resolve()
    fixture, _ = make_fixture(kind, scenario, config=resolved_config, input_dimensions=input_dimensions)
    config_identity = hashlib.sha256(json.dumps(asdict(resolved_config), sort_keys=True).encode()).hexdigest()[:8]
    group_name = f"{kind}-{scenario}-seed{resolved_config.seed}-d{resolved_config.dimensions}-{config_identity}"
    group_dir = output / group_name
    if group_dir.exists():
        raise FileExistsError(f"run group already exists and is immutable: {group_dir}")
    group_dir.mkdir(parents=True)
    _json_write(group_dir / "status.json", {"status": "running", "completed_families": []})
    _write_fixture(group_dir, fixture)
    _json_write(group_dir / "resolved_config.json", {"schema_version": 2, "experiment": kind, "scenario": scenario, **asdict(resolved_config)})
    encoder_seed = resolved_config.seed + 100_003
    base_encoder = _make_encoder(fixture, encoder_seed)
    base_encoder_state = _clone_state(base_encoder) if base_encoder is not None else None
    shared_hash = _state_hash(base_encoder_state) if base_encoder_state is not None else None
    run_metrics: dict[str, Any] = {}
    old_num_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        completed_families: list[str] = []
        for family in MODEL_FAMILIES:
            run_metrics[family] = _run_one(
                group_dir,
                fixture,
                family,
                resolved_config,
                device=torch.device(device),
                base_encoder_state=base_encoder_state,
                encoder_seed=encoder_seed,
            )
            completed_families.append(family)
            _json_write(group_dir / "status.json", {"status": "running", "completed_families": completed_families})
    except Exception:
        _json_write(group_dir / "status.json", {"status": "failed"})
        raise
    finally:
        torch.set_num_threads(old_num_threads)
    comparison = {
        "status": "completed",
        "experiment": kind,
        "scenario": scenario,
        "fixture_id": fixture.fixture_id,
        "seed": resolved_config.seed,
        "shared_initial_encoder_sha256": shared_hash,
        "identical_fixture": True,
        "identical_train_validation_split": True,
        "model_runs": {family: {"run_id": run_metrics[family]["run_id"], "objective": run_metrics[family]["objective"]} for family in MODEL_FAMILIES},
        "warning": "Synthetic diagnostic only; not evidence of pizza-image generalization or thesis research-question resolution.",
    }
    _json_write(group_dir / "comparison.json", comparison)
    _json_write(group_dir / "status.json", {"status": "completed"})
    return group_dir


def run_fixed_experiments(output_root: Path | str, *, config: SyntheticConfig | None = None) -> list[Path]:
    resolved = config or SyntheticConfig()
    return [run_experiment("fixed", scenario, output_root, config=resolved) for scenario in ("simple", "window")]


def _load_fixture(group_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads((group_dir / "fixture.json").read_text(encoding="utf-8"))
    with np.load(group_dir / "fixture.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    actual_hash = _fixture_arrays_hash(arrays)
    if actual_hash != metadata.get("content_sha256"):
        raise ValueError("synthetic fixture arrays do not match their persisted content hash")
    return metadata, arrays


def load_saved_run(run_path: Path | str, *, checkpoint: Path | str | None = None) -> tuple[PointClassifier, SyntheticPointEncoder | None, dict[str, Any], dict[str, np.ndarray]]:
    """Restore classifier/encoder and fixture arrays without invoking training."""
    run_dir = Path(run_path).expanduser().resolve()
    group_dir = run_dir.parent
    metadata = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    fixture_meta, arrays = _load_fixture(group_dir)
    checkpoint_path = Path(checkpoint).expanduser() if checkpoint else run_dir / "checkpoint.pt"
    saved = _require_artifact_mapping(
        _load_torch_artifact(checkpoint_path, "synthetic checkpoint"), checkpoint_path, "synthetic checkpoint"
    )
    required = {
        "checkpoint_schema", "family", "dimensions", "class_names", "seed", "model_config",
        "encoder_config", "model_state", "encoder_state", "optimizer_state", "step",
        "config_identity", "fixture_id",
    }
    missing = sorted(required - saved.keys())
    if missing:
        raise SavedArtifactError(f"incompatible checkpoint at {checkpoint_path}: missing fields {missing}")
    schema = saved["checkpoint_schema"]
    if type(schema) is not int or schema not in {1, 2}:
        raise SavedArtifactError(f"incompatible checkpoint at {checkpoint_path}: unsupported schema {schema!r}")
    if (
        not isinstance(saved["class_names"], (list, tuple))
        or not saved["class_names"]
        or not all(isinstance(name, str) for name in saved["class_names"])
    ):
        raise SavedArtifactError(f"incompatible checkpoint at {checkpoint_path}: invalid class_names")
    if isinstance(saved["dimensions"], bool) or not isinstance(saved["dimensions"], int) or saved["dimensions"] < 1:
        raise SavedArtifactError(f"incompatible checkpoint at {checkpoint_path}: invalid dimensions")
    if not isinstance(saved["family"], str) or saved["family"] not in MODEL_FAMILIES:
        raise SavedArtifactError(f"incompatible checkpoint at {checkpoint_path}: invalid model family")
    if isinstance(saved["step"], bool) or not isinstance(saved["step"], int) or saved["step"] < 0:
        raise SavedArtifactError(f"incompatible checkpoint at {checkpoint_path}: invalid training step")
    if not isinstance(saved["optimizer_state"], dict):
        raise SavedArtifactError(f"incompatible checkpoint at {checkpoint_path}: invalid optimizer_state")
    model_config = _require_artifact_mapping(saved["model_config"], checkpoint_path, "checkpoint model_config")
    expected_model_fields = {"houses", "windows", "min_width", "window_mode"}
    if saved["checkpoint_schema"] >= 2:
        expected_model_fields.add("window_target_volume")
    missing_model_fields = sorted(expected_model_fields - model_config.keys())
    if missing_model_fields:
        raise SavedArtifactError(
            f"incompatible checkpoint at {checkpoint_path}: model_config is missing {missing_model_fields}"
        )
    if saved["fixture_id"] != fixture_meta.get("fixture_id"):
        raise SavedArtifactError(f"checkpoint at {checkpoint_path} fixture identity does not match its run directory")
    if tuple(saved["class_names"]) != tuple(fixture_meta.get("class_names", ())):
        raise SavedArtifactError(f"checkpoint at {checkpoint_path} class ordering does not match its fixture manifest")
    expected_config_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    if saved["config_identity"] != expected_config_hash:
        raise SavedArtifactError(f"checkpoint at {checkpoint_path} resolved configuration identity does not match resolved_config.json")
    family = saved["family"]
    if not isinstance(saved["model_state"], dict):
        raise SavedArtifactError(f"incompatible checkpoint at {checkpoint_path}: invalid model_state")
    dimensions = saved["dimensions"]
    window_target_volume = model_config.get("window_target_volume", 0.72**dimensions)
    try:
        model = make_classifier(
            family,
            dimensions,
            tuple(saved["class_names"]),
            seed=int(saved["seed"]),
            houses=int(model_config["houses"]),
            windows=int(model_config["windows"]),
            min_width=float(model_config["min_width"]),
            window_mode=model_config["window_mode"],
            window_target_volume=float(window_target_volume),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise SavedArtifactError(f"incompatible model configuration in checkpoint at {checkpoint_path}: {error}") from error
    try:
        model.load_state_dict(saved["model_state"])
    except (RuntimeError, TypeError, ValueError) as error:
        raise SavedArtifactError(f"incompatible model state in checkpoint at {checkpoint_path}: {error}") from error
    encoder = None
    if saved["encoder_config"] is not None:
        enc_config = _require_artifact_mapping(saved["encoder_config"], checkpoint_path, "checkpoint encoder_config")
        if not isinstance(saved["encoder_state"], dict):
            raise SavedArtifactError(f"incompatible checkpoint at {checkpoint_path}: encoder_state is missing")
        try:
            encoder = SyntheticPointEncoder(
                int(enc_config["input_dimensions"]),
                int(enc_config["latent_dimensions"]),
                seed=int(enc_config["seed"]),
                hidden_dimensions=int(enc_config["hidden_dimensions"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise SavedArtifactError(f"incompatible encoder configuration in checkpoint at {checkpoint_path}: {error}") from error
        try:
            encoder.load_state_dict(saved["encoder_state"])
        except (RuntimeError, TypeError, ValueError) as error:
            raise SavedArtifactError(f"incompatible encoder state in checkpoint at {checkpoint_path}: {error}") from error
    elif saved["encoder_state"] is not None:
        raise SavedArtifactError(f"incompatible checkpoint at {checkpoint_path}: unexpected encoder_state")
    bundle = {"run_dir": run_dir, "fixture": fixture_meta, "checkpoint": saved, "resolved_config": metadata}
    return model, encoder, bundle, arrays


def evaluate_saved_run(run_path: Path | str, split: str = "validation", *, checkpoint: Path | str | None = None) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise ValueError("synthetic run evaluation split must be train or validation")
    model, encoder, bundle, arrays = load_saved_run(run_path, checkpoint=checkpoint)
    indices = arrays[f"{split}_indices"].astype(np.int64)
    features = torch.from_numpy(arrays["features"][indices]).to(torch.float32)
    targets = torch.from_numpy(arrays["targets"][indices]).to(torch.float32)
    known = torch.from_numpy(arrays["known_mask"][indices]).to(torch.bool)
    model.eval()
    if encoder is not None:
        encoder.eval()
    with torch.no_grad():
        points = features if encoder is None else encoder(features)
        output = model(points)
        reduction, objective = _loss_for(model, output, targets, known, _configured_margin(bundle["resolved_config"]["training"]))
    metrics = _classification_metrics(output, targets, known, model.class_names)
    return {
        "run_id": str(bundle["run_dir"].relative_to(bundle["run_dir"].parent.parent)),
        "split": split,
        "objective": objective,
        "loss": float(reduction.loss.item()),
        **metrics,
    }


def resume_saved_run(run_path: Path | str, additional_steps: int) -> dict[str, Any]:
    """Continue a completed Adam run from its checkpoint after config validation.

    Resumed steps append history and snapshots; the immutable resolved config is
    not edited. The checkpoint's saved optimizer state is restored verbatim.
    """
    if isinstance(additional_steps, bool) or not isinstance(additional_steps, int) or additional_steps < 1:
        raise ValueError("additional_steps must be a positive integer")
    model, encoder, bundle, arrays = load_saved_run(run_path)
    run_dir = bundle["run_dir"]
    checkpoint = bundle["checkpoint"]
    settings = bundle["resolved_config"]
    metric_path = run_dir / "metrics.json"
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    config = settings["training"]
    margin = _configured_margin(config)
    fixture_meta = bundle["fixture"]
    fixture = SyntheticFixture(
        fixture_id=fixture_meta["fixture_id"],
        kind=fixture_meta["kind"],
        scenario=fixture_meta["scenario"],
        latent_dimensions=int(checkpoint["dimensions"]),
        features=torch.from_numpy(arrays["features"]).float(),
        targets=torch.from_numpy(arrays["targets"]).float(),
        known_mask=torch.from_numpy(arrays["known_mask"]).bool(),
        train_indices=torch.from_numpy(arrays["train_indices"]).long(),
        validation_indices=torch.from_numpy(arrays["validation_indices"]).long(),
        class_names=tuple(fixture_meta["class_names"]),
        seed=int(fixture_meta["seed"]),
        teacher_description=str(fixture_meta["teacher_description"]),
    )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + (list(encoder.parameters()) if encoder is not None else []),
        lr=float(config["learning_rate"]),
    )
    if checkpoint.get("optimizer_state") is None:
        raise SavedArtifactError(f"checkpoint at {run_dir / 'checkpoint.pt'} has no optimizer state and cannot be resumed")
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise SavedArtifactError(f"incompatible optimizer state in checkpoint at {run_dir / 'checkpoint.pt'}: {error}") from error
    features, targets, known = _subset_tensors(fixture, fixture.train_indices, device=torch.device("cpu"))
    previous_step = int(checkpoint["step"])
    target_step = previous_step + additional_steps
    interval = int(config["snapshot_interval"])
    history = list(metrics.get("history", []))
    last_gradient: dict[str, Any] = {}
    old_num_threads = torch.get_num_threads()
    started = time.time()
    try:
        torch.set_num_threads(1)
        for step in range(previous_step + 1, target_step + 1):
            model.train()
            if encoder is not None:
                encoder.train()
            optimizer.zero_grad(set_to_none=True)
            points = _points_for(encoder, features)
            output = model(points)
            reduction, _ = _loss_for(model, output, targets, known, margin)
            if not torch.isfinite(reduction.loss):
                raise FloatingPointError(f"non-finite resumed objective at step {step}")
            reduction.loss.backward()
            last_gradient = _gradient_report(model, encoder)
            optimizer.step()
            if step % interval == 0 or step == target_step:
                train_metric, _ = _evaluate_split(model, encoder, fixture, fixture.train_indices, margin=margin, device=torch.device("cpu"))
                validation_metric, _ = _evaluate_split(model, encoder, fixture, fixture.validation_indices, margin=margin, device=torch.device("cpu"))
                history.append({"step": step, "training": train_metric, "validation": validation_metric, "gradient_norms": last_gradient, "resumed": True})
                _snapshot(run_dir, f"step_{step:05d}", model, encoder, fixture, step=step)
    finally:
        torch.set_num_threads(old_num_threads)

    train_metric, train_points = _evaluate_split(model, encoder, fixture, fixture.train_indices, margin=margin, device=torch.device("cpu"))
    validation_metric, _ = _evaluate_split(model, encoder, fixture, fixture.validation_indices, margin=margin, device=torch.device("cpu"))
    metrics["training_metrics"] = train_metric
    metrics["validation_metrics"] = validation_metric
    metrics["history"] = history
    metrics["gradient_norms_last_step"] = last_gradient
    metrics["geometry_activity"] = _geometry_activity(model, train_points)
    metrics["resume_segments"] = [*metrics.get("resume_segments", []), {"from_step": previous_step, "to_step": target_step, "additional_steps": additional_steps, "elapsed_seconds": time.time() - started}]
    metrics["snapshot_names"] = sorted(path.stem for path in (run_dir / "snapshots").glob("*.pt"))
    _json_write(metric_path, metrics)
    with (run_dir / "history.jsonl").open("w", encoding="utf-8") as stream:
        for row in history:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    checkpoint.update({
        "model_state": _clone_state(model),
        "encoder_state": _clone_state(encoder) if encoder is not None else None,
        "optimizer_state": optimizer.state_dict(),
        "step": target_step,
        "resume_count": int(checkpoint.get("resume_count", 0)) + 1,
    })
    _atomic_torch_save(checkpoint, run_dir / "checkpoint.pt")
    return metrics


def inspect_saved_run(run_path: Path | str) -> dict[str, Any]:
    run_dir = Path(run_path).expanduser().resolve()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    return {"path": str(run_dir), "config": config, "metrics": metrics, "environment": env}


def saved_geometry(run_path: Path | str, snapshot: str | None = None) -> dict[str, Any]:
    """Restore actual model state and snapshot-specific activity when available."""
    run_dir = Path(run_path).expanduser().resolve()
    chosen = snapshot or "final"
    model, encoder, bundle, arrays = load_saved_run(run_dir)
    snapshot_data: dict[str, Any] | None = None
    if chosen == "final":
        checkpoint = bundle["checkpoint"]
        model_state = checkpoint["model_state"]
        encoder_state = checkpoint["encoder_state"]
        step = int(checkpoint["step"])
        name = "final checkpoint"
        final_snapshot = run_dir / "snapshots" / f"step_{step:05d}.pt"
        if final_snapshot.is_file():
            snapshot_data = _require_artifact_mapping(
                _load_torch_artifact(final_snapshot, "geometry snapshot"), final_snapshot, "geometry snapshot"
            )
            if snapshot_data.get("step") != step:
                raise SavedArtifactError(f"incompatible geometry snapshot at {final_snapshot}: step does not match checkpoint")
    elif chosen == "initial":
        initial_snapshot = run_dir / "snapshots" / "initial.pt"
        if initial_snapshot.is_file():
            snapshot_data = _require_artifact_mapping(
                _load_torch_artifact(initial_snapshot, "initial geometry snapshot"), initial_snapshot, "geometry snapshot"
            )
            if snapshot_data.get("step") != 0:
                raise SavedArtifactError(f"incompatible initial geometry snapshot at {initial_snapshot}: expected step 0")
            model_state = snapshot_data.get("model_state")
            encoder_state = snapshot_data.get("encoder_state")
        else:
            initial_path = run_dir / "initial_state.pt"
            initial = _require_artifact_mapping(
                _load_torch_artifact(initial_path, "initial model state"), initial_path, "initial model state"
            )
            model_state = initial.get("model_state")
            encoder_state = initial.get("encoder_state")
        step = 0
        name = chosen
    else:
        snapshot_path = run_dir / "snapshots" / f"{chosen}.pt"
        if not snapshot_path.is_file():
            raise FileNotFoundError(f"snapshot not found: {snapshot_path}")
        snapshot_data = _require_artifact_mapping(
            _load_torch_artifact(snapshot_path, "geometry snapshot"), snapshot_path, "geometry snapshot"
        )
        model_state = snapshot_data.get("model_state")
        encoder_state = snapshot_data.get("encoder_state")
        if "step" not in snapshot_data:
            raise SavedArtifactError(f"incompatible geometry snapshot at {snapshot_path}: missing step")
        step = int(snapshot_data["step"])
        if step < 0:
            raise SavedArtifactError(f"incompatible geometry snapshot at {snapshot_path}: step cannot be negative")
        name = chosen
    if not isinstance(model_state, dict):
        raise SavedArtifactError(f"incompatible model state for snapshot {name!r} in {run_dir}")
    if encoder is not None and not isinstance(encoder_state, dict):
        raise SavedArtifactError(f"incompatible encoder state for snapshot {name!r} in {run_dir}")
    if encoder is None and encoder_state is not None:
        raise SavedArtifactError(f"unexpected encoder state for snapshot {name!r} in {run_dir}")
    try:
        model.load_state_dict(model_state)
        if encoder is not None:
            encoder.load_state_dict(encoder_state)
    except (RuntimeError, TypeError, ValueError) as error:
        raise SavedArtifactError(f"incompatible model state for snapshot {name!r} in {run_dir}: {error}") from error
    model.eval()
    if encoder is not None:
        encoder.eval()
    features = torch.from_numpy(arrays["features"]).float()
    with torch.no_grad():
        points = features if encoder is None else encoder(features)
        output = model(points, diagnostics=True)

    activity = snapshot_data.get("activity") if snapshot_data is not None else None
    activity_split = snapshot_data.get("activity_split") if snapshot_data is not None else None
    activity_source = "snapshot" if activity is not None else None
    if activity is None and chosen == "final":
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        activity = metrics.get("geometry_activity")
        if activity is not None:
            activity_source = "final_run_metrics"
            activity_split = "train"
    return {
        "name": name,
        "step": step,
        "points": points,
        "output": output,
        "snapshot_activity": activity,
        "snapshot_activity_split": activity_split,
        "snapshot_activity_step": step if activity is not None else None,
        "snapshot_activity_source": activity_source,
        "snapshot_activity_compatibility_note": (
            None if activity is not None else
            "This run artifact predates snapshot-specific activity diagnostics; historical activity is unavailable."
        ),
        "fixture": bundle["fixture"],
        "arrays": arrays,
        "model": model,
        "encoder": encoder,
        "run_dir": run_dir,
    }
