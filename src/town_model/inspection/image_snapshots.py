"""Read-only checksummed access to persisted historical image snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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


def load_image_snapshot(path: Path | str) -> ImageResearchSnapshot:
    """Validate and return stored numeric arrays without recomputing their meaning."""
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

    # This checks persisted metadata/array agreement only. It never derives a
    # region, membership, score, margin, or prediction from the stored values.
    geometry = header.get("geometry", {})
    for name in ("box_low", "box_high", "frame_low", "frame_high", "window_low", "window_high"):
        array_key = f"classifier_geometry_{name}"
        if array_key in arrays and not np.array_equal(arrays[array_key], np.asarray(geometry[name])):
            raise ValueError(f"snapshot class geometry diagnostic {name!r} differs from saved metadata")
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
