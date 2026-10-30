"""Strict, versioned frozen-visual feature cache used by experiment orchestration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
from typing import Any

import numpy as np
import torch

from town_model.contracts import FeatureDiagnostics, FrozenImageFeatureCache, ImageBatch, VisionFeatureBatch
from town_model.dataset.image_data import DatasetSubset, PreparedDataset
from town_model.experiments.image_config import ImageExperimentConfig
from town_model.training.image_trainer import _hash_state
from town_model.vision.backbone import MobileNetV3LargeFPN
from town_model.vision.batching import collate_image_records
from town_model.vision.features import SharedVisualFeatureExtractor
from town_model.vision.preprocessing import ImageNetLetterbox
from town_model.vision.roi import GroundTruthROIProvider

CACHE_SCHEMA = "frozen-vision-feature-cache-v1"


def _hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _acquire_cache_lock(path: Path) -> None:
    for attempt in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            try:
                owner = json.loads(path.read_text(encoding="utf-8"))
                if owner.get("hostname") != socket.gethostname():
                    raise RuntimeError(f"feature-cache writer lock belongs to another host: {path}") from error
                os.kill(int(owner["pid"]), 0)
                raise RuntimeError(f"a writer is already creating frozen feature cache {path.stem}") from error
            except ProcessLookupError:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            except PermissionError as permission_error:
                raise RuntimeError(f"feature-cache writer lock owner is alive but not signalable: {path}") from permission_error
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as invalid_owner:
                raise RuntimeError(f"cannot safely recover feature-cache writer lock {path}: {invalid_owner}") from error
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "hostname": socket.gethostname()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            return
    raise RuntimeError(f"could not acquire feature-cache writer lock {path}")


def _sample_filename(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode()).hexdigest() + ".npz"


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class FrozenFeatureCache(FrozenImageFeatureCache):
    """Read-only cache reader that validates the full producer identity once."""

    def __init__(self, directory: Path, identity: dict[str, Any], manifest: dict[str, Any]) -> None:
        self.directory = directory
        self.identity = identity
        self.cache_id = manifest["cache_id"]
        self._sample_files = dict(manifest["samples"])
        self._loaded: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = {}
        expected = set(identity["sample_ids"])
        if set(self._sample_files) != expected:
            raise ValueError("feature-cache sample index does not exactly match the resolved train/validation panel")
        for sample_id, entry in self._sample_files.items():
            path = directory / entry["path"]
            if not path.is_file() or _file_sha256(path) != entry["sha256"]:
                raise ValueError(f"feature-cache artifact is missing or corrupt for sample {sample_id}")

    @classmethod
    def open(cls, directory: Path, identity: dict[str, Any]) -> "FrozenFeatureCache":
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read frozen visual feature cache at {directory}: {error}") from error
        if manifest.get("schema_version") != CACHE_SCHEMA or manifest.get("identity") != identity:
            raise ValueError("frozen visual feature cache has incompatible producer identity or schema")
        if manifest.get("cache_id") != _hash_json(identity)[:24]:
            raise ValueError("frozen visual feature cache identity hash does not match its manifest")
        return cls(directory, identity, manifest)

    def _load_sample(self, sample_id: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        if sample_id not in self._sample_files:
            raise ValueError(f"sample {sample_id!r} is not present in the frozen feature cache")
        cached = self._loaded.get(sample_id)
        if cached is not None:
            return cached
        entry = self._sample_files[sample_id]
        path = self.directory / entry["path"]
        if _file_sha256(path) != entry["sha256"]:
            raise ValueError(f"frozen feature cache changed after opening, sample {sample_id}")
        try:
            with np.load(path, allow_pickle=False) as stored:
                stored_id = str(stored["sample_id"].item())
                vectors = stored["component_vectors"].astype(np.float32, copy=True)
                global_spatial = stored["global_spatial_features"].astype(np.float32, copy=True)
                component_ids = tuple(str(value) for value in stored["component_ids"].tolist())
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f"invalid feature-cache record for {sample_id}: {error}") from error
        if stored_id != sample_id or vectors.ndim != 2 or vectors.shape[1] != 260 or global_spatial.shape != (256, 7, 7):
            raise ValueError(f"feature-cache tensor contract mismatch for {sample_id}")
        if vectors.shape[0] != len(component_ids) or not np.isfinite(vectors).all() or not np.isfinite(global_spatial).all():
            raise ValueError(f"feature-cache values or component identities are invalid for {sample_id}")
        cached = vectors, global_spatial, component_ids
        self._loaded[sample_id] = cached
        return cached

    def get(self, batch: ImageBatch, *, diagnostics: bool = False) -> VisionFeatureBatch:
        entries = [self._load_sample(sample_id) for sample_id in batch.sample_ids]
        for sample_id, (_, _, component_ids), expected_ids in zip(
            batch.sample_ids, entries, batch.component_ids, strict=True
        ):
            if component_ids != expected_ids:
                raise ValueError(f"cached GT ROI identities are stale for sample {sample_id}")
        counts = [entry[0].shape[0] for entry in entries]
        max_count = max(counts, default=0)
        component_vectors = torch.zeros((len(entries), max_count, 260), dtype=torch.float32)
        valid = torch.zeros((len(entries), max_count), dtype=torch.bool)
        global_features = torch.stack([torch.from_numpy(entry[1]) for entry in entries])
        component_ids_out: list[tuple[str | None, ...]] = []
        for index, (vectors, _, ids) in enumerate(entries):
            count = vectors.shape[0]
            if count:
                component_vectors[index, :count] = torch.from_numpy(vectors)
                valid[index, :count] = True
            component_ids_out.append(ids + (None,) * (max_count - count))
        diagnostic = None
        if diagnostics:
            diagnostic = FeatureDiagnostics(
                roi_features=component_vectors[..., :256],
                global_features=global_features,
                roi_source=str(self.identity["roi_source"]),
                normalized_roi_coordinates=component_vectors[..., 256:],
            )
        return VisionFeatureBatch(
            component_vectors=component_vectors,
            component_valid_mask=valid,
            global_spatial_features=global_features,
            component_ids=tuple(component_ids_out),
            diagnostics=diagnostic,
        )


def prepare_frozen_feature_cache(
    directory: Path | str,
    dataset: PreparedDataset,
    train_subset: DatasetSubset,
    validation_subset: DatasetSubset,
    visual: SharedVisualFeatureExtractor,
    config: ImageExperimentConfig,
    *,
    device: torch.device,
) -> FrozenFeatureCache:
    """Create or strictly reuse deterministic eval-mode visual features.

    Cache entries contain only visual ROI/global features and occurrence IDs;
    no ontology targets, detector labels, scores, optimizer states, or test rows.
    """
    if config.vision.visual_policy != "frozen":
        raise ValueError("frozen visual feature cache cannot be used for trainable visual features")
    records_by_id = {record.sample_id: record for record in (*train_subset.records, *validation_subset.records)}
    if set(record.split for record in records_by_id.values()) & {"test"}:
        raise ValueError("feature cache may not contain test-partition records")
    sample_ids = list(records_by_id)
    visual_state_hash = _hash_state(visual.state_dict())
    identity: dict[str, Any] = {
        "dataset_id": dataset.dataset_id,
        "dataset_manifest_id": dataset.manifest_id,
        "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
        "class_manifest_sha256": dataset.class_manifest_sha256,
        "train_subset_id": train_subset.subset_id,
        "validation_subset_id": validation_subset.subset_id,
        "sample_ids": sample_ids,
        "sample_splits": {sample_id: records_by_id[sample_id].split for sample_id in sample_ids},
        "roi_source": config.vision.roi_source,
        "visual_state_sha256": visual_state_hash,
        "backbone_initialization": visual.backbone.initialization_source,
        "backbone_architecture": "torchvision_mobilenet_v3_large_features_0_16",
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
    cache_id = _hash_json(identity)[:24]
    base = Path(directory).expanduser().resolve()
    destination = base / cache_id
    if destination.exists():
        return FrozenFeatureCache.open(destination, identity)
    base.mkdir(parents=True, exist_ok=True)
    lock_path = base / f".{cache_id}.lock"
    _acquire_cache_lock(lock_path)
    staging: Path | None = None
    old_training = visual.training
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        for orphan in base.glob(f".{cache_id}.*"):
            if orphan.is_symlink():
                raise RuntimeError(f"refusing to recover symlinked frozen feature-cache staging path {orphan}")
            if orphan.is_dir():
                shutil.rmtree(orphan, ignore_errors=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{cache_id}.", dir=base))
        visual.eval()
        sample_manifest: dict[str, Any] = {}
        transform = ImageNetLetterbox(config.vision.resolution)
        with torch.no_grad():
            for sample_id in sample_ids:
                record = records_by_id[sample_id]
                batch = collate_image_records(dataset, (record,), transform=transform)
                batch = type(batch)(
                    images=batch.images.to(device),
                    original_image_sizes=batch.original_image_sizes,
                    processed_image_sizes=batch.processed_image_sizes,
                    image_geometries=batch.image_geometries,
                    sample_ids=batch.sample_ids,
                    gt_component_boxes_xyxy=tuple(box.to(device) for box in batch.gt_component_boxes_xyxy),
                    component_ids=batch.component_ids,
                    component_detector_category_ids=batch.component_detector_category_ids,
                    pizza_targets=batch.pizza_targets.to(device),
                    pizza_known_masks=batch.pizza_known_masks.to(device),
                    component_targets=tuple(target.to(device) for target in batch.component_targets),
                    component_known_masks=tuple(mask.to(device) for mask in batch.component_known_masks),
                    split_identity=batch.split_identity,
                )
                features = visual(batch, GroundTruthROIProvider())
                count = int(features.component_valid_mask[0].sum().item())
                filename = _sample_filename(sample_id)
                output_path = staging / filename
                arrays = {
                    "sample_id": np.asarray(sample_id),
                    "component_vectors": features.component_vectors[0, :count].detach().cpu().numpy().astype(np.float32),
                    "global_spatial_features": features.global_spatial_features[0].detach().cpu().numpy().astype(np.float32),
                    "component_ids": np.asarray([identity.occurrence_id for identity in GroundTruthROIProvider().provide(batch).identities[0]], dtype=np.str_),
                }
                _write_npz(output_path, **arrays)
                sample_manifest[sample_id] = {"path": filename, "sha256": _file_sha256(output_path)}
        manifest = {
            "schema_version": CACHE_SCHEMA,
            "cache_id": cache_id,
            "identity": identity,
            "samples": sample_manifest,
        }
        _atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
    except Exception:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_cuda)
        visual.train(old_training)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    return FrozenFeatureCache.open(destination, identity)
