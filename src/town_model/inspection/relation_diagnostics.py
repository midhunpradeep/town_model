"""Exact box and explicitly sampled town relation diagnostics.

This is an inspection layer: it has no training-loss or optimizer integration.
Town samples are always verified by the existing exact closed-frame,
closed-window-exclusion membership contract; a missing sampled witness never
proves a universal relation.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

import numpy as np
import torch

from town_model.dataset.ontology import OntologySnapshot, build_class_vocabulary
from town_model.models.classifiers import BoxClassifier, PointClassifier, TownClassifier
from town_model.models.geometry import evaluate_town_corners

Relation = Literal["containment", "intersection", "disjointness"]
RELATION_SCHEMA = "ontology-region-relation-diagnostic-v1"
_MAX_REPORTED_POINTS = 8


@dataclass(frozen=True, slots=True)
class GeometryRegion:
    class_id: int
    class_name: str
    family: Literal["box", "town"]
    dimensions: int
    box_low: np.ndarray | None = None
    box_high: np.ndarray | None = None
    frame_low: np.ndarray | None = None
    frame_high: np.ndarray | None = None
    window_low: np.ndarray | None = None
    window_high: np.ndarray | None = None
    geometry_reference: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if type(self.class_id) is not int or self.class_id < 0:
            raise ValueError("region class_id must be a nonnegative integer")
        if not self.class_name or type(self.dimensions) is not int or self.dimensions < 1:
            raise ValueError("region name and latent dimension are required")
        if self.family == "box":
            if self.box_low is None or self.box_high is None:
                raise ValueError("box region requires low/high corners")
            _validate_corners(self.box_low, self.box_high, (self.dimensions,))
        elif self.family == "town":
            if any(value is None for value in (self.frame_low, self.frame_high, self.window_low, self.window_high)):
                raise ValueError("town region requires frame and window corners")
            assert self.frame_low is not None and self.frame_high is not None
            assert self.window_low is not None and self.window_high is not None
            if self.frame_low.ndim != 2 or self.frame_low.shape[0] < 1 or self.frame_low.shape[-1] != self.dimensions:
                raise ValueError("town frames must have shape [H,d] with at least one house")
            if self.window_low.ndim != 3 or self.window_low.shape[0] != self.frame_low.shape[0] or self.window_low.shape[-1] != self.dimensions:
                raise ValueError("town windows must have shape [H,W,d]")
            _validate_corners(self.frame_low, self.frame_high, self.frame_low.shape)
            _validate_corners(self.window_low, self.window_high, self.window_low.shape)
        else:
            raise ValueError(f"unsupported geometric region family {self.family!r}")

    @property
    def geometry_sha256(self) -> str:
        payload: dict[str, Any] = {"family": self.family, "dimensions": self.dimensions}
        for name in ("box_low", "box_high", "frame_low", "frame_high", "window_low", "window_high"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = np.asarray(value, dtype=np.float32).tolist()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _validate_corners(low: np.ndarray, high: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    low = np.asarray(low)
    high = np.asarray(high)
    if low.shape != expected_shape or high.shape != expected_shape or low.dtype.kind not in "fiu" or high.dtype.kind not in "fiu":
        raise ValueError("region corners have invalid shape or numeric type")
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low < -1) or np.any(high > 1) or np.any(low > high):
        raise ValueError("region corners must be finite, ordered, and contained in [-1,1]")


def box_region(
    class_id: int,
    class_name: str,
    low: np.ndarray | list[float],
    high: np.ndarray | list[float],
    *,
    geometry_reference: dict[str, Any] | None = None,
) -> GeometryRegion:
    low_array = np.asarray(low, dtype=np.float32)
    high_array = np.asarray(high, dtype=np.float32)
    return GeometryRegion(class_id, class_name, "box", int(low_array.size), box_low=low_array, box_high=high_array, geometry_reference=geometry_reference)


def town_region(
    class_id: int,
    class_name: str,
    frame_low: np.ndarray | list,
    frame_high: np.ndarray | list,
    window_low: np.ndarray | list,
    window_high: np.ndarray | list,
    *,
    geometry_reference: dict[str, Any] | None = None,
) -> GeometryRegion:
    low_array = np.asarray(frame_low, dtype=np.float32)
    high_array = np.asarray(frame_high, dtype=np.float32)
    window_low_array = np.asarray(window_low, dtype=np.float32)
    window_high_array = np.asarray(window_high, dtype=np.float32)
    return GeometryRegion(
        class_id, class_name, "town", int(low_array.shape[-1]),
        frame_low=low_array, frame_high=high_array,
        window_low=window_low_array, window_high=window_high_array,
        geometry_reference=geometry_reference,
    )


def region_from_classifier(
    classifier: PointClassifier,
    *,
    class_id: int,
    class_index: int,
    geometry_reference: dict[str, Any] | None = None,
) -> GeometryRegion:
    """Adapt existing classifier geometry; baseline hyperplanes are not regions here."""
    if not 0 <= class_index < len(classifier.class_names):
        raise IndexError("class index is outside the classifier vocabulary")
    name = classifier.class_names[class_index]
    if isinstance(classifier, BoxClassifier):
        low, high = classifier.geometry.physical_boxes()
        return box_region(class_id, name, low[class_index].detach().cpu().numpy(), high[class_index].detach().cpu().numpy(), geometry_reference=geometry_reference)
    if isinstance(classifier, TownClassifier):
        frame_low, frame_high, window_low, window_high = classifier.geometry.physical_geometry()
        return town_region(
            class_id, name,
            frame_low[class_index].detach().cpu().numpy(), frame_high[class_index].detach().cpu().numpy(),
            window_low[class_index].detach().cpu().numpy(), window_high[class_index].detach().cpu().numpy(),
            geometry_reference=geometry_reference,
        )
    raise ValueError("neural baseline linear heads do not define bounded box/town regions")


def validate_snapshot_ontology_mapping(snapshot, ontology: OntologySnapshot) -> None:
    """Ensure snapshot class IDs/names agree with the canonical OWL vocabulary."""
    trainable = [item for item in build_class_vocabulary(ontology)["classes"] if item["trainable"]]
    expected = {item["class_id"]: item["name"] for item in trainable}
    for class_id, name in zip(snapshot.header.get("class_ids", ()), snapshot.header.get("class_names", ()), strict=True):
        if expected.get(class_id) != name:
            raise ValueError(
                f"snapshot class identity mismatch for ID {class_id}: "
                f"saved name {name!r}, canonical ontology name {expected.get(class_id)!r}"
            )


def region_from_snapshot(snapshot, class_index: int, *, ontology: OntologySnapshot | None = None) -> GeometryRegion:
    """Build a query region from immutable snapshot geometry without loading a checkpoint."""
    if ontology is not None:
        validate_snapshot_ontology_mapping(snapshot, ontology)
    header = snapshot.header
    class_ids = header.get("class_ids", [])
    class_names = header.get("class_names", [])
    if not 0 <= class_index < len(class_ids) or len(class_ids) != len(class_names):
        raise IndexError("snapshot class index is outside its ordered vocabulary")
    geometry = header.get("geometry", {})
    family = geometry.get("family")
    reference = {
        "snapshot_path": str(snapshot.path),
        "snapshot_schema": header.get("schema_version"),
        "arrays_sha256": header.get("arrays_sha256"),
        "source_checkpoint_sha256": header.get("source_checkpoint_sha256"),
    }
    class_id, name = class_ids[class_index], class_names[class_index]
    if family == "box":
        return box_region(class_id, name, geometry["box_low"][class_index], geometry["box_high"][class_index], geometry_reference=reference)
    if family == "town":
        return town_region(
            class_id, name,
            geometry["frame_low"][class_index], geometry["frame_high"][class_index],
            geometry["window_low"][class_index], geometry["window_high"][class_index],
            geometry_reference=reference,
        )
    raise ValueError("saved baseline hyperplanes are not box/town regions for these relation queries")


def _box_contains(region: GeometryRegion, points: np.ndarray) -> np.ndarray:
    assert region.box_low is not None and region.box_high is not None
    return np.all((points >= region.box_low) & (points <= region.box_high), axis=-1)


def _town_contains(region: GeometryRegion, points: np.ndarray) -> np.ndarray:
    assert region.frame_low is not None and region.frame_high is not None
    assert region.window_low is not None and region.window_high is not None
    values = evaluate_town_corners(
        torch.as_tensor(points, dtype=torch.float32),
        torch.as_tensor(region.frame_low[None, ...], dtype=torch.float32),
        torch.as_tensor(region.frame_high[None, ...], dtype=torch.float32),
        torch.as_tensor(region.window_low[None, ...], dtype=torch.float32),
        torch.as_tensor(region.window_high[None, ...], dtype=torch.float32),
    )
    return values.contains[:, 0].cpu().numpy()


def evaluate_region_membership(region: GeometryRegion, points: np.ndarray | list) -> np.ndarray:
    """Evaluate exact hard membership using the existing box/town semantics."""
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != region.dimensions or not np.isfinite(values).all():
        raise ValueError("membership points must be finite [N,d] within the region dimension")
    if np.any(values < -1) or np.any(values > 1):
        raise ValueError("membership points must lie in the bounded latent domain [-1,1]^d")
    return _box_contains(region, values) if region.family == "box" else _town_contains(region, values)


def _contains(region: GeometryRegion, points: np.ndarray) -> np.ndarray:
    return evaluate_region_membership(region, points)


def _box_pair(source: GeometryRegion, target: GeometryRegion, relation: Relation) -> dict[str, Any]:
    assert source.box_low is not None and source.box_high is not None
    assert target.box_low is not None and target.box_high is not None
    source_low, source_high = source.box_low, source.box_high
    target_low, target_high = target.box_low, target.box_high
    result: dict[str, Any] = {
        "verification_method": "exact_closed_axis_aligned_box_interval_algebra",
        "decision_scope": "exact",
        "sampling": None,
        "verified_witness_points": [],
        "verified_counterexample_points": [],
        "separating_axis_certificate": None,
    }
    intersection_low = np.maximum(source_low, target_low)
    intersection_high = np.minimum(source_high, target_high)
    intersects = bool(np.all(intersection_low <= intersection_high))
    if relation == "containment":
        contained = bool(np.all(source_low >= target_low) and np.all(source_high <= target_high))
        result["outcome"] = contained
        result["interpretation"] = "exact_proof" if contained else "exact_counterexample"
        if not contained:
            point = (source_low + source_high) / 2.0
            violations = np.flatnonzero((source_low < target_low) | (source_high > target_high))
            dimension = int(violations[0])
            point[dimension] = source_low[dimension] if source_low[dimension] < target_low[dimension] else source_high[dimension]
            if not _box_contains(source, point[None, :])[0] or _box_contains(target, point[None, :])[0]:
                raise RuntimeError("exact box containment counterexample failed membership verification")
            result["verified_counterexample_points"] = [point.astype(float).tolist()]
    elif intersects:
        witness = intersection_low
        if not (_box_contains(source, witness[None, :])[0] and _box_contains(target, witness[None, :])[0]):
            raise RuntimeError("exact box intersection witness failed membership verification")
        if relation == "intersection":
            result["outcome"] = True
            result["interpretation"] = "exact_verified_witness"
            result["verified_witness_points"] = [witness.astype(float).tolist()]
        else:
            result["outcome"] = False
            result["interpretation"] = "exact_verified_counterexample_to_disjointness"
            result["verified_counterexample_points"] = [witness.astype(float).tolist()]
    else:
        dimension = int(np.flatnonzero(intersection_low > intersection_high)[0])
        result["separating_axis_certificate"] = {
            "dimension": dimension,
            "source_interval": [float(source_low[dimension]), float(source_high[dimension])],
            "target_interval": [float(target_low[dimension]), float(target_high[dimension])],
            "gap": float(max(source_low[dimension] - target_high[dimension], target_low[dimension] - source_high[dimension])),
        }
        if relation == "intersection":
            result["outcome"] = False
            result["interpretation"] = "exact_proof_no_intersection"
        else:
            result["outcome"] = True
            result["interpretation"] = "exact_proof_disjoint"
    return result


def _region_frame_bounds(region: GeometryRegion) -> tuple[np.ndarray, np.ndarray]:
    if region.family == "box":
        assert region.box_low is not None and region.box_high is not None
        return region.box_low[None, :], region.box_high[None, :]
    assert region.frame_low is not None and region.frame_high is not None
    return region.frame_low, region.frame_high


def _town_samples(region: GeometryRegion, count: int, rng: np.random.Generator) -> np.ndarray:
    low, high = _region_frame_bounds(region)
    house_indices = rng.integers(0, len(low), size=count)
    draws = rng.random((count, region.dimensions), dtype=np.float32)
    return low[house_indices] + draws * (high[house_indices] - low[house_indices])


def _sampled_pair(source: GeometryRegion, target: GeometryRegion, relation: Relation, *, sample_count: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    uniform_domain = rng.uniform(-1.0, 1.0, size=(sample_count, source.dimensions)).astype(np.float32)
    source_frame_draws = _town_samples(source, sample_count, rng)
    target_frame_draws = _town_samples(target, sample_count, rng)
    candidates = np.concatenate((uniform_domain, source_frame_draws, target_frame_draws), axis=0)
    in_source = _contains(source, candidates)
    in_target = _contains(target, candidates)
    if relation == "containment":
        observed = in_source & ~in_target
        witness_mask = observed
        outcome = False if bool(observed.any()) else None
        interpretation = "sampled_verified_counterexample" if outcome is False else "inconclusive_no_sampled_counterexample"
        witness_key = "verified_counterexample_points"
    elif relation == "intersection":
        observed = in_source & in_target
        witness_mask = observed
        outcome = True if bool(observed.any()) else None
        interpretation = "sampled_verified_intersection_witness" if outcome is True else "inconclusive_no_sampled_intersection"
        witness_key = "verified_witness_points"
    else:
        observed = in_source & in_target
        witness_mask = observed
        outcome = False if bool(observed.any()) else None
        interpretation = "sampled_verified_counterexample_to_disjointness" if outcome is False else "inconclusive_no_sampled_intersection"
        witness_key = "verified_counterexample_points"
    points = candidates[np.flatnonzero(witness_mask)[:_MAX_REPORTED_POINTS]]
    # Membership was computed by exact classifier semantics for every sampled
    # candidate. Recheck the subset retained as human-readable witnesses.
    if relation == "containment" and len(points):
        assert np.all(_contains(source, points) & ~_contains(target, points))
    elif relation in ("intersection", "disjointness") and len(points):
        assert np.all(_contains(source, points) & _contains(target, points))
    return {
        "verification_method": "seeded_stratified_sampling_with_exact_town_membership",
        "decision_scope": "sampled_empirical",
        "outcome": outcome,
        "interpretation": interpretation,
        "reason": (
            "No decisive witness was observed; finite deterministic sampling cannot prove a universal relation or absence of overlap."
            if outcome is None else None
        ),
        "sampling": {
            "strategy": "N uniform draws from [-1,1]^d plus N uniform proposals from source house frames plus N proposals from target house frames; all candidate memberships evaluated exactly",
            "sample_count": sample_count,
            "sample_count_per_strategy": sample_count,
            "candidate_point_count": int(len(candidates)),
            "sample_count_total": int(len(candidates)),
            "sampling_domain": {"lower": -1.0, "upper": 1.0, "dimensions": source.dimensions},
            "seed": seed,
            "observed_relation_violations_or_witnesses": int(observed.sum()),
            "observed_containment_violations": int((in_source & ~in_target).sum()),
            "observed_intersection_witnesses": int((in_source & in_target).sum()),
        },
        "verified_witness_points": points.astype(float).tolist() if witness_key == "verified_witness_points" else [],
        "verified_counterexample_points": points.astype(float).tolist() if witness_key == "verified_counterexample_points" else [],
        "separating_axis_certificate": None,
    }


def diagnose_region_relation(
    source: GeometryRegion,
    target: GeometryRegion,
    relation: Relation,
    *,
    sample_count: int = 10_000,
    seed: int = 0,
    ontology: OntologySnapshot | None = None,
) -> dict[str, Any]:
    """Diagnose A⊆B, A∩B≠∅, or A∩B=∅ with explicit proof scope."""
    if relation not in ("containment", "intersection", "disjointness"):
        raise ValueError(f"unsupported region relation {relation!r}")
    if source.dimensions != target.dimensions:
        raise ValueError("relation regions must share the same latent dimension")
    if type(sample_count) is not int or sample_count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("sample_count must be positive and seed a nonnegative integer")
    exact_boxes = source.family == target.family == "box"
    core = _box_pair(source, target, relation) if exact_boxes else _sampled_pair(source, target, relation, sample_count=sample_count, seed=seed)
    ontology_query = "disjointness" if relation == "disjointness" else "containment" if relation == "containment" else "intersection"
    return {
        "schema_version": RELATION_SCHEMA,
        "source_region": {"class_id": source.class_id, "class_name": source.class_name, "family": source.family, "geometry_sha256": source.geometry_sha256},
        "target_region": {"class_id": target.class_id, "class_name": target.class_name, "family": target.family, "geometry_sha256": target.geometry_sha256},
        "queried_relation": relation,
        "outcome": core["outcome"],
        "interpretation": core["interpretation"],
        "reason": core.get("reason"),
        "verification_method": core["verification_method"],
        "decision_scope": core["decision_scope"],
        "latent_domain": {"lower": -1.0, "upper": 1.0, "dimensions": source.dimensions},
        "sampling": core["sampling"],
        "verified_witness_points": core["verified_witness_points"],
        "verified_counterexample_points": core["verified_counterexample_points"],
        "separating_axis_certificate": core["separating_axis_certificate"],
        "boundary_and_tolerance": {
            "coordinate_tolerance": 0.0,
            "box_boundaries": "closed; equality and boundary-touching count as intersection",
            "town_semantics": "union of closed house frames minus each house's closed windows; overlapping houses re-include points; exact existing membership verifies samples",
        },
        "ontology_relation": ontology.geometric_relation_metadata(source.class_name, target.class_name, ontology_query) if ontology is not None else {
            "status": "unknown", "basis": "ontology metadata was not supplied"
        },
        "geometry_references": {"source": source.geometry_reference, "target": target.geometry_reference},
    }


def _fsync_directory(directory: Path) -> None:
    """Persist directory-entry changes where directory fsync is supported."""
    if os.name == "nt":
        # Python does not expose a portable directory-handle fsync on Windows.
        return
    unsupported = {
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as error:
        if error.errno in unsupported:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


def save_relation_diagnostic(result: dict[str, Any], destination: Path | str) -> Path:
    """Atomically publish immutable JSON using same-directory no-clobber linking.

    The temporary file and destination share a directory, so ``os.link`` is a
    same-filesystem atomic create-if-absent operation on supported local
    filesystems. There is deliberately no replacement fallback. Hard-link or
    filesystem errors fail closed. A post-publication cleanup error can leave a
    hidden temporary hard link; it is harmless and recoverable by removing it
    only after confirming it is the same inode as the published destination.
    """
    # Normalize to an absolute lexical path without resolving the final entry:
    # an existing (including dangling) symlink is itself an occupied name.
    path = Path(os.path.abspath(os.fspath(Path(destination).expanduser())))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor_open = False
        with stream:
            json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"relation diagnostic is immutable and already exists: {path}"
            ) from error
        except OSError as error:
            raise type(error)(
                f"atomic no-clobber hard-link publication failed for {path}: {error}"
            ) from error
        _fsync_directory(path.parent)
        os.unlink(temporary)
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor_open:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return path
