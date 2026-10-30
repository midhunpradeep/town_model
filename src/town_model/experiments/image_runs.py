"""Validated image-run identity, lifecycle, persistence, and orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
from typing import Any, Callable, Iterable

from town_model.dataset.image_data import DatasetSubset, PreparedDataset, deterministic_subset, load_prepared_dataset
from town_model.experiments.feature_cache import FrozenFeatureCache, prepare_frozen_feature_cache
from town_model.experiments.image_config import ImageExperimentConfig
from town_model.inspection.image_snapshots import save_image_snapshot
from town_model.training.image_trainer import (
    ImageTrainingResult,
    _device_for,
    build_image_modules,
    train_image_model,
    validate_image_training_data,
)

RUN_STATES = ("draft", "running", "completed", "stopped", "interrupted", "failed")
_ALLOWED_TRANSITIONS = {
    "draft": {"running", "failed"},
    "running": {"completed", "stopped", "interrupted", "failed"},
    "stopped": {"running"},
    "interrupted": {"running"},
    "failed": {"running"},
    "completed": set(),
}


class RunLifecycleError(RuntimeError):
    """Invalid local image-run state transition or writer ownership."""


@dataclass(frozen=True, slots=True)
class ResolvedImageExperiment:
    config: ImageExperimentConfig
    dataset: PreparedDataset
    train_subset: DatasetSubset
    validation_subset: DatasetSubset
    class_ids: tuple[int, ...]
    identity: dict[str, Any]
    identity_hash: str
    run_id: str

    def resolved_config(self) -> dict[str, Any]:
        return {
            "schema_version": self.config.schema_version,
            "training_config": asdict(self.config),
            "identity_hash": self.identity_hash,
            "identity": self.identity,
            "dataset": {
                "dataset_id": self.dataset.dataset_id,
                "manifest_id": self.dataset.manifest_id,
                "dataset_manifest_sha256": self.dataset.dataset_manifest_sha256,
                "class_manifest_sha256": self.dataset.class_manifest_sha256,
                "class_ids": list(self.class_ids),
                "class_names": [
                    self.dataset.class_names[self.dataset.canonical_class_ids.index(class_id)]
                    for class_id in self.class_ids
                ],
            },
            "train_subset": self.train_subset.to_manifest(),
            "validation_subset": self.validation_subset.to_manifest(),
            "training_protocol": "pre-detector exploratory GT-ROI; no Pizzaïolo detector-pretraining checkpoint",
        }


def _sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=str)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def resolve_image_experiment(
    config: ImageExperimentConfig,
    *,
    repository_root: Path | str | None = None,
) -> ResolvedImageExperiment:
    """Validate all inputs and identities before an output run directory exists."""
    config.validate()
    _device_for(config)
    base = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[3]
    dataset_path = config.resolve_dataset_path(base)
    dataset = load_prepared_dataset(dataset_path, validate=True)
    train_subset = deterministic_subset(
        dataset,
        "train",
        size=config.dataset.train_size,
        seed=config.dataset.selection_seed,
    )
    validation_subset = deterministic_subset(
        dataset,
        "validation",
        size=config.dataset.validation_size,
        seed=config.dataset.selection_seed + 1,
    )
    train_ids = {record.sample_id for record in train_subset.records}
    validation_ids = {record.sample_id for record in validation_subset.records}
    if train_ids & validation_ids:
        raise ValueError("original train and validation partitions unexpectedly share sample IDs")
    class_ids = tuple(config.dataset.trainable_class_ids or dataset.canonical_class_ids)
    if tuple(sorted(class_ids)) != class_ids or any(class_id not in dataset.canonical_class_ids for class_id in class_ids):
        raise ValueError("explicit class IDs must be a canonical-order subset of trainable ontology class IDs")
    validate_image_training_data(
        dataset,
        train_subset,
        validation_subset,
        class_ids,
        require_training_coverage=config.purpose == "comparison",
        require_primary_monitor=config.purpose != "smoke",
    )
    identity = {
        "training_config": config.identity_dict(),
        "dataset_id": dataset.dataset_id,
        "manifest_id": dataset.manifest_id,
        "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
        "class_manifest_sha256": dataset.class_manifest_sha256,
        "train_subset_id": train_subset.subset_id,
        "validation_subset_id": validation_subset.subset_id,
        "train_sample_ids": [record.sample_id for record in train_subset.records],
        "validation_sample_ids": [record.sample_id for record in validation_subset.records],
        "canonical_trainable_class_ids": list(class_ids),
        "roi_source": config.vision.roi_source,
    }
    identity_hash = _sha256_json(identity)
    return ResolvedImageExperiment(
        config=config,
        dataset=dataset,
        train_subset=train_subset,
        validation_subset=validation_subset,
        class_ids=class_ids,
        identity=identity,
        identity_hash=identity_hash,
        run_id=f"image-{identity_hash[:20]}",
    )


class RunWriterLock:
    """Exclusive local run writer; stale same-host locks are safely recoverable."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def _owner_is_alive(self) -> bool:
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
            if owner.get("hostname") != socket.gethostname():
                return True
            os.kill(int(owner["pid"]), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            # An incomplete/malformed owner record cannot be proven stale safely.
            return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as error:
                if attempt == 0 and not self._owner_is_alive():
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                raise RunLifecycleError(f"another process owns this image run lock: {self.path}") from error
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "hostname": socket.gethostname(), "created_at": _now()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            self.acquired = True
            return
        raise RunLifecycleError(f"could not acquire image run writer lock {self.path}")

    def release(self) -> None:
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.acquired = False

    def __enter__(self) -> "RunWriterLock":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_manifest(run_directory: Path) -> dict[str, Any]:
    try:
        payload = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunLifecycleError(f"could not read image run manifest in {run_directory}: {error}") from error
    if not isinstance(payload, dict) or payload.get("state") not in RUN_STATES:
        raise RunLifecycleError(f"invalid image run manifest in {run_directory}")
    return payload


def transition_run(run_directory: Path | str, new_state: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    if new_state not in RUN_STATES:
        raise RunLifecycleError(f"unknown image run state {new_state!r}")
    root = Path(run_directory).resolve()
    manifest = _read_manifest(root)
    old_state = manifest["state"]
    if new_state not in _ALLOWED_TRANSITIONS[old_state]:
        raise RunLifecycleError(f"invalid image run state transition {old_state!r} -> {new_state!r}")
    manifest["state"] = new_state
    manifest["updated_at"] = _now()
    if details:
        manifest.update(details)
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def _process_alive(manifest: dict[str, Any]) -> bool:
    if manifest.get("hostname") != socket.gethostname():
        return True
    try:
        os.kill(int(manifest["pid"]), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def inspect_run(run_directory: Path | str, *, refresh_interrupted: bool = True) -> dict[str, Any]:
    root = Path(run_directory).expanduser().resolve()
    manifest = _read_manifest(root)
    if refresh_interrupted and manifest["state"] == "running" and not _process_alive(manifest):
        manifest = transition_run(root, "interrupted", details={
            "interruption_reason": "writer process is no longer alive; latest atomic checkpoint remains authoritative"
        })
    return manifest


def request_stop(run_directory: Path | str) -> None:
    root = Path(run_directory).expanduser().resolve()
    manifest = inspect_run(root, refresh_interrupted=True)
    if manifest["state"] != "running":
        raise RunLifecycleError(f"cannot request graceful stop for run in state {manifest['state']!r}")
    (root / "stop.request").write_text(f"requested_at={_now()}\n", encoding="utf-8")


def run_image_experiment(
    config: ImageExperimentConfig,
    output_root: Path | str,
    *,
    repository_root: Path | str | None = None,
    resume: bool = False,
    stop_requested: Callable[[], bool] | None = None,
) -> tuple[Path, ImageTrainingResult]:
    """Run/resume one validated config; run artifacts are immutable by identity."""
    resolved = resolve_image_experiment(config, repository_root=repository_root)
    feature_cache: FrozenFeatureCache | None = None
    if config.execution.feature_cache_directory:
        base = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[3]
        cache_root = Path(config.execution.feature_cache_directory).expanduser()
        if not cache_root.is_absolute():
            cache_root = base / cache_root
        device = _device_for(config)
        visual, _, _ = build_image_modules(resolved.dataset, config, class_ids=resolved.class_ids)
        visual.to(device)
        feature_cache = prepare_frozen_feature_cache(
            cache_root,
            resolved.dataset,
            resolved.train_subset,
            resolved.validation_subset,
            visual,
            config,
            device=device,
        )
    root = Path(output_root).expanduser().resolve() / resolved.run_id
    manifest_path = root / "manifest.json"
    if root.exists() and not resume:
        raise FileExistsError(f"image run already exists and will not be overwritten: {root}")
    if not root.exists() and resume:
        raise FileNotFoundError(f"cannot resume an image run that does not exist: {root}")
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        _atomic_json(root / "resolved-config.json", resolved.resolved_config())
        _atomic_json(
            manifest_path,
            {
                "schema_version": "image-run-manifest-v1",
                "run_id": resolved.run_id,
                "identity_hash": resolved.identity_hash,
                "state": "draft",
                "created_at": _now(),
                "updated_at": _now(),
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "run_directory": str(root),
                "checkpoint_path": None,
                "error": None,
            },
        )
    with RunWriterLock(root / ".writer.lock"):
        manifest = _read_manifest(root)
        resolved_on_disk = json.loads((root / "resolved-config.json").read_text(encoding="utf-8"))
        if manifest.get("identity_hash") != resolved.identity_hash or resolved_on_disk.get("identity_hash") != resolved.identity_hash:
            raise RunLifecycleError("existing run identity/configuration does not match the validated request")
        state = manifest["state"]
        if resume:
            if state == "running":
                if _process_alive(manifest):
                    raise RunLifecycleError("cannot resume an image run that still has a live writer")
                transition_run(root, "interrupted", details={"interruption_reason": "stale running state discovered during resume"})
                state = "interrupted"
            if state not in ("stopped", "interrupted", "failed"):
                raise RunLifecycleError(f"cannot faithfully resume an image run in state {state!r}")
            checkpoint = root / "checkpoints" / "latest.pt"
            if not checkpoint.is_file():
                raise RunLifecycleError("run has no valid latest checkpoint to resume")
            transition_run(root, "running", details={
                "pid": os.getpid(), "hostname": socket.gethostname(), "resumed_at": _now(), "error": None
            })
            resume_from = checkpoint
        else:
            if state != "draft":
                raise RunLifecycleError(f"new run must be draft, found {state!r}")
            transition_run(root, "running", details={"pid": os.getpid(), "hostname": socket.gethostname(), "started_at": _now()})
            resume_from = None
        should_stop = stop_requested or (lambda: (root / "stop.request").exists())
        snapshot_paths: list[str] = []

        def snapshot_callback(epoch, step, visual, model, is_final):
            if epoch % config.diagnostics.snapshot_every_epochs != 0 and not is_final:
                return
            snapshot = save_image_snapshot(
                root / "snapshots" / f"epoch-{epoch:05d}",
                dataset=resolved.dataset,
                train_subset=resolved.train_subset,
                validation_subset=resolved.validation_subset,
                visual=visual,
                model=model,
                config=config,
                class_ids=resolved.class_ids,
                class_indices=tuple(resolved.dataset.canonical_class_ids.index(class_id) for class_id in resolved.class_ids),
                epoch=epoch,
                optimizer_step=step,
                latest_checkpoint=root / "checkpoints" / "latest.pt",
                identity_hash=resolved.identity_hash,
                feature_cache=feature_cache,
            )
            snapshot_paths.append(str(snapshot.path))

        try:
            result = train_image_model(
                resolved.dataset,
                resolved.train_subset,
                resolved.validation_subset,
                config,
                root,
                resume_from=resume_from,
                stop_requested=should_stop,
                feature_cache=feature_cache,
                snapshot_callback=snapshot_callback,
            )
            metrics = {
                "run_id": resolved.run_id,
                "identity_hash": resolved.identity_hash,
                "status": result.status,
                "optimizer_step": result.global_step,
                "completed_epochs": result.completed_epochs,
                "best_primary_monitor": result.best_primary_monitor,
                "stopped_reason": result.stopped_reason,
                "history": list(result.history),
                "snapshots": snapshot_paths,
            }
            _atomic_json(root / "metrics.json", metrics)
            state_after = "stopped" if result.status == "stopped" else "completed"
            transition_run(
                root,
                state_after,
                details={
                    "checkpoint_path": str(result.latest_checkpoint),
                    "optimizer_step": result.global_step,
                    "completed_epochs": result.completed_epochs,
                    "best_primary_monitor": result.best_primary_monitor,
                    "stopped_reason": result.stopped_reason,
                    "snapshot_paths": snapshot_paths,
                    "finished_at": _now(),
                },
            )
            try:
                (root / "stop.request").unlink()
            except FileNotFoundError:
                pass
            return root, result
        except KeyboardInterrupt:
            transition_run(root, "interrupted", details={
                "error": "KeyboardInterrupt",
                "checkpoint_path": str(root / "checkpoints" / "latest.pt"),
                "finished_at": _now(),
            })
            raise
        except Exception as error:
            transition_run(root, "failed", details={
                "error": f"{type(error).__name__}: {error}",
                "checkpoint_path": str(root / "checkpoints" / "latest.pt") if (root / "checkpoints" / "latest.pt").exists() else None,
                "failed_at": _now(),
            })
            raise


def run_image_matrix(
    config: ImageExperimentConfig,
    output_root: Path | str,
    *,
    repository_root: Path | str | None = None,
    families: Iterable[str] = ("baseline", "box", "town"),
    pizza_modes: Iterable[str] = ("global", "compositional", "hybrid"),
) -> Path:
    """Run a paired family × composition matrix on identical real-image subsets."""
    if config.purpose != "comparison":
        raise ValueError("image family×composition matrices require purpose='comparison' and its strict support/checkpoint rules")
    selected_families = tuple(families)
    selected_modes = tuple(pizza_modes)
    if not selected_families or any(value not in ("baseline", "box", "town") for value in selected_families):
        raise ValueError("families must be a nonempty subset of baseline, box, town")
    if not selected_modes or any(value not in ("global", "compositional", "hybrid") for value in selected_modes):
        raise ValueError("pizza_modes must be a nonempty subset of global, compositional, hybrid")
    planned = [
        replace(config, model=replace(config.model, family=family, pizza_mode=mode))
        for family in selected_families
        for mode in selected_modes
    ]
    resolved = [resolve_image_experiment(item, repository_root=repository_root) for item in planned]
    matrix_identity = {
        "schema_version": "image-matrix-v1",
        "run_ids": [item.run_id for item in resolved],
        "dataset_id": resolved[0].dataset.dataset_id,
        "train_subset_id": resolved[0].train_subset.subset_id,
        "validation_subset_id": resolved[0].validation_subset.subset_id,
        "matrix": [{"family": item.config.model.family, "pizza_mode": item.config.model.pizza_mode} for item in resolved],
    }
    matrix_id = f"matrix-{_sha256_json(matrix_identity)[:20]}"
    output = Path(output_root).expanduser().resolve()
    matrix_path = output / "matrices" / matrix_id
    if matrix_path.exists():
        raise FileExistsError(f"image matrix manifest already exists and will not be overwritten: {matrix_path}")
    results = []
    for item in planned:
        run_path, result = run_image_experiment(item, output, repository_root=repository_root)
        results.append({
            "run_id": run_path.name,
            "run_directory": str(run_path),
            "family": item.model.family,
            "pizza_mode": item.model.pizza_mode,
            "status": result.status,
        })
    matrix_path.mkdir(parents=True, exist_ok=False)
    _atomic_json(matrix_path / "comparison.json", {
        **matrix_identity,
        "matrix_id": matrix_id,
        "status": "completed",
        "runs": results,
        "training_protocol": "pre-detector exploratory GT-ROI; not detector-pretrained Phase 3 acceptance evidence",
        "created_at": _now(),
    })
    return matrix_path
