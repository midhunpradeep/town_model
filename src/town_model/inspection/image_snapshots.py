"""Read-only checksummed access to persisted historical image snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from town_model.classifier_backend import raise_classifier_backend_unavailable

SNAPSHOT_SCHEMA = "image-research-snapshot-v2"


@dataclass(frozen=True, slots=True)
class ImageResearchSnapshot:
    """Stored historical header and arrays; loading does not reconstruct a model."""

    path: Path
    header: dict[str, Any]
    arrays: dict[str, np.ndarray]


def save_image_snapshot(*args, **kwargs) -> NoReturn:
    """Reject model-dependent snapshot generation until an approved backend exists."""
    raise_classifier_backend_unavailable("historical image classifier snapshot generation")


def _snapshot_integrity_error(root: Path, field: str, detail: str) -> ValueError:
    return ValueError(f"invalid image research snapshot at {root}: {field}: {detail}")


def load_image_snapshot(path: Path | str) -> ImageResearchSnapshot:
    """Validate and return stored numeric arrays without recomputing their meaning."""
    root = Path(path).expanduser().resolve()
    try:
        header = json.loads((root / "header.json").read_text(encoding="utf-8"))
        if not isinstance(header, dict):
            raise ValueError("snapshot header must be a JSON object")
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
    except (OSError, EOFError, json.JSONDecodeError, KeyError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"cannot read image research snapshot at {root}: {error}") from error

    sample_ids = arrays.get("sample_ids")
    if sample_ids is None or sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise _snapshot_integrity_error(root, "arrays.sample_ids", "must be a one-dimensional string array")
    header_sample_ids = header.get("sample_ids")
    if not isinstance(header_sample_ids, list) or any(not isinstance(value, str) for value in header_sample_ids):
        raise _snapshot_integrity_error(root, "header.sample_ids", "must be a list of sample ID strings")
    sample_id_values = sample_ids.tolist()
    if sample_id_values != header_sample_ids:
        raise _snapshot_integrity_error(root, "header.sample_ids", "does not match arrays.sample_ids")

    sample_count = len(sample_id_values)
    offsets = arrays.get("component_offsets")
    if (
        offsets is None
        or offsets.ndim != 1
        or offsets.shape != (sample_count + 1,)
        or not np.issubdtype(offsets.dtype, np.integer)
    ):
        raise _snapshot_integrity_error(
            root, "arrays.component_offsets", "must be a one-dimensional integer array matching sample_ids"
        )
    if offsets[0] != 0 or np.any(offsets < 0) or np.any(offsets[1:] < offsets[:-1]):
        raise _snapshot_integrity_error(root, "arrays.component_offsets", "must start at zero and be non-decreasing")
    component_count = arrays.get("component_count_total")
    if (
        component_count is None
        or component_count.ndim != 0
        or not np.issubdtype(component_count.dtype, np.integer)
    ):
        raise _snapshot_integrity_error(root, "arrays.component_count_total", "must be an integer scalar")
    if int(offsets[-1]) != int(component_count):
        raise _snapshot_integrity_error(
            root, "arrays.component_count_total", "does not match the final component offset"
        )

    if "diagnostic_axes_schema" not in header:
        raise _snapshot_integrity_error(
            root, "header.diagnostic_axes_schema", "is required by the snapshot schema"
        )
    axis_schema = header["diagnostic_axes_schema"]
    if not isinstance(axis_schema, dict):
        raise _snapshot_integrity_error(root, "header.diagnostic_axes_schema", "must be a JSON object")
    for key, axes in axis_schema.items():
        if not isinstance(key, str) or not key:
            raise _snapshot_integrity_error(root, "header.diagnostic_axes_schema", "contains an invalid array name")
        if not isinstance(axes, list) or any(not isinstance(axis, str) or not axis for axis in axes):
            raise _snapshot_integrity_error(
                root, f"header.diagnostic_axes_schema.{key}", "must be a list of non-empty axis names"
            )
        if key not in arrays:
            raise _snapshot_integrity_error(root, f"header.diagnostic_axes_schema.{key}", "declares a missing array")
        if arrays[key].ndim != len(axes):
            raise _snapshot_integrity_error(
                root, f"header.diagnostic_axes_schema.{key}",
                f"declares {len(axes)} axes for a {arrays[key].ndim}-dimensional array",
            )

    # This checks persisted metadata/array agreement only. It never derives a
    # region, membership, score, margin, or prediction from the stored values.
    geometry_arrays = [key for key in arrays if key.startswith("classifier_geometry_")]
    if geometry_arrays:
        geometry = header.get("geometry")
        if not isinstance(geometry, dict):
            raise _snapshot_integrity_error(
                root, "header.geometry", "must be a JSON object when classifier geometry arrays are stored"
            )
        for array_key in geometry_arrays:
            name = array_key.removeprefix("classifier_geometry_")
            geometry_field = f"header.geometry.{name}"
            if not name or name not in geometry:
                raise _snapshot_integrity_error(
                    root, geometry_field, f"is required by stored array {array_key!r}"
                )
            try:
                saved_geometry = np.asarray(geometry[name])
            except (TypeError, ValueError) as error:
                raise _snapshot_integrity_error(
                    root, geometry_field, f"cannot be represented as an array: {error}"
                ) from error
            if saved_geometry.shape != arrays[array_key].shape:
                raise _snapshot_integrity_error(
                    root, geometry_field,
                    f"shape {saved_geometry.shape} does not match stored array {array_key!r} shape {arrays[array_key].shape}",
                )
            if not np.array_equal(arrays[array_key], saved_geometry):
                raise _snapshot_integrity_error(
                    root, geometry_field, f"values do not match stored array {array_key!r}"
                )
    return ImageResearchSnapshot(root, header, arrays)


def discover_image_snapshots(run_directory: Path | str) -> list[Path]:
    """Return only persisted snapshots that pass the read-only integrity checks."""
    root = Path(run_directory).expanduser().resolve() / "snapshots"
    if not root.is_dir():
        return []
    found: list[Path] = []
    for candidate in sorted(root.glob("epoch-*")):
        if not candidate.is_dir() or not (candidate / "header.json").is_file() or not (candidate / "arrays.npz").is_file():
            continue
        try:
            load_image_snapshot(candidate)
        except ValueError:
            continue
        found.append(candidate)
    return found
