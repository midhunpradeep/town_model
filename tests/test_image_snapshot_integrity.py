from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from town_model.inspection.image_snapshots import discover_image_snapshots, load_image_snapshot


_DEFAULT = object()


def _snapshot_values() -> dict[str, np.ndarray]:
    return {
        "sample_ids": np.asarray(["pizza-a", "pizza-b"], dtype=np.str_),
        "component_offsets": np.asarray([0, 2, 3], dtype=np.int64),
        "component_count_total": np.asarray(3, dtype=np.int64),
        "component_ids": np.asarray(["occ-a", "occ-b", "occ-c"], dtype=np.str_),
        "classifier_geometry_box_low": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "classifier_geometry_box_high": np.asarray([[0.6, 0.7], [0.8, 0.9]], dtype=np.float32),
        "pizza_contains": np.asarray([[True, False], [False, True]], dtype=np.bool_),
        "component_activity_frame_margin": np.arange(12, dtype=np.float32).reshape(3, 2, 2),
    }


def _write_snapshot(
    path: Path,
    *,
    geometry: Any = _DEFAULT,
    diagnostic_axes_schema: Any = _DEFAULT,
    sample_ids_header: Any = _DEFAULT,
    arrays: dict[str, np.ndarray] | None = None,
) -> tuple[Path, dict[str, np.ndarray]]:
    path.mkdir(parents=True, exist_ok=True)
    stored_arrays = _snapshot_values() if arrays is None else arrays
    arrays_path = path / "arrays.npz"
    np.savez_compressed(arrays_path, **stored_arrays)
    default_geometry = {
        name: stored_arrays[array_name].tolist()
        for name, array_name in (
            ("box_low", "classifier_geometry_box_low"),
            ("box_high", "classifier_geometry_box_high"),
        )
        if array_name in stored_arrays
    }
    default_axes = {
        "classifier_geometry_box_low": ["class", "coordinate"],
        "classifier_geometry_box_high": ["class", "coordinate"],
        "component_activity_frame_margin": ["component_occurrence", "class", "house"],
    }
    header = {
        "schema_version": "image-research-snapshot-v2",
        "sample_ids": stored_arrays["sample_ids"].tolist() if sample_ids_header is _DEFAULT else sample_ids_header,
        "diagnostic_axes_schema": default_axes if diagnostic_axes_schema is _DEFAULT else diagnostic_axes_schema,
        "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        "geometry": default_geometry if geometry is _DEFAULT else geometry,
    }
    (path / "header.json").write_text(json.dumps(header), encoding="utf-8")
    return path, stored_arrays


def test_reader_roundtrips_nonempty_diagnostic_axes_and_stored_values_without_reconstruction(tmp_path: Path) -> None:
    snapshot_path, expected_arrays = _write_snapshot(tmp_path / "snapshot")

    snapshot = load_image_snapshot(snapshot_path)

    assert snapshot.header["sample_ids"] == ["pizza-a", "pizza-b"]
    assert snapshot.arrays["sample_ids"].tolist() == ["pizza-a", "pizza-b"]
    assert snapshot.arrays["component_offsets"].tolist() == [0, 2, 3]
    assert snapshot.arrays["component_count_total"].item() == 3
    assert snapshot.header["diagnostic_axes_schema"] == {
        "classifier_geometry_box_low": ["class", "coordinate"],
        "classifier_geometry_box_high": ["class", "coordinate"],
        "component_activity_frame_margin": ["component_occurrence", "class", "house"],
    }
    for name, expected in expected_arrays.items():
        assert snapshot.arrays[name].dtype == expected.dtype
        assert np.array_equal(snapshot.arrays[name], expected)
    assert np.array_equal(
        snapshot.arrays["classifier_geometry_box_low"],
        np.asarray(snapshot.header["geometry"]["box_low"], dtype=np.float32),
    )
    assert np.array_equal(
        snapshot.arrays["classifier_geometry_box_high"],
        np.asarray(snapshot.header["geometry"]["box_high"], dtype=np.float32),
    )


def test_geometry_metadata_is_optional_when_no_geometry_arrays_are_stored(tmp_path: Path) -> None:
    arrays = {
        "sample_ids": np.asarray(["pizza-a"], dtype=np.str_),
        "component_offsets": np.asarray([0, 0], dtype=np.int64),
        "component_count_total": np.asarray(0, dtype=np.int64),
        "pizza_contains": np.asarray([[True]], dtype=np.bool_),
    }
    snapshot_path, _ = _write_snapshot(
        tmp_path / "snapshot", arrays=arrays, geometry=None, diagnostic_axes_schema={}
    )

    snapshot = load_image_snapshot(snapshot_path)

    assert snapshot.header["geometry"] is None
    assert snapshot.arrays["pizza_contains"].tolist() == [[True]]


@pytest.mark.parametrize(
    ("geometry", "expected_field"),
    [
        ({"box_high": [[0.6, 0.7], [0.8, 0.9]]}, "header.geometry.box_low"),
        ([], "header.geometry"),
        (
            {"box_low": [[0.1, 0.2]], "box_high": [[0.6, 0.7], [0.8, 0.9]]},
            "header.geometry.box_low",
        ),
        (
            {"box_low": [[9.1, 0.2], [0.3, 0.4]], "box_high": [[0.6, 0.7], [0.8, 0.9]]},
            "header.geometry.box_low",
        ),
    ],
    ids=["missing-field", "non-mapping", "shape-mismatch", "value-mismatch"],
)
def test_reader_reports_controlled_geometry_integrity_errors(
    tmp_path: Path, geometry: Any, expected_field: str
) -> None:
    snapshot_path, _ = _write_snapshot(tmp_path / "invalid-snapshot", geometry=geometry)

    with pytest.raises(ValueError, match=expected_field.replace(".", r"\.")) as captured:
        load_image_snapshot(snapshot_path)

    assert str(snapshot_path.resolve()) in str(captured.value)


def test_reader_reports_malformed_geometry_values_as_integrity_errors(tmp_path: Path) -> None:
    snapshot_path, _ = _write_snapshot(
        tmp_path / "invalid-snapshot",
        geometry={"box_low": [[0.1], [0.2, 0.3]], "box_high": [[0.6, 0.7], [0.8, 0.9]]},
    )

    with pytest.raises(ValueError, match=r"header\.geometry\.box_low"):
        load_image_snapshot(snapshot_path)


@pytest.mark.parametrize(
    ("axes", "expected_field"),
    [
        ({"classifier_geometry_box_low": ["class"]}, "header.diagnostic_axes_schema.classifier_geometry_box_low"),
        ({"missing_diagnostic": ["image"]}, "header.diagnostic_axes_schema.missing_diagnostic"),
        ({"classifier_geometry_box_low": ["class", 4]}, "header.diagnostic_axes_schema.classifier_geometry_box_low"),
    ],
    ids=["wrong-axis-count", "missing-array", "malformed-axis-name"],
)
def test_reader_rejects_invalid_diagnostic_axis_declarations(
    tmp_path: Path, axes: dict[str, Any], expected_field: str
) -> None:
    snapshot_path, _ = _write_snapshot(tmp_path / "invalid-snapshot", diagnostic_axes_schema=axes)

    with pytest.raises(ValueError, match=expected_field.replace(".", r"\.")):
        load_image_snapshot(snapshot_path)


def test_reader_rejects_checksum_corruption_before_returning_arrays(tmp_path: Path) -> None:
    snapshot_path, _ = _write_snapshot(tmp_path / "corrupt-snapshot")
    arrays_path = snapshot_path / "arrays.npz"
    arrays_path.write_bytes(arrays_path.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_image_snapshot(snapshot_path)

    arrays_path.write_bytes(b"not a valid npz file")
    header_path = snapshot_path / "header.json"
    header = json.loads(header_path.read_text(encoding="utf-8"))
    header["arrays_sha256"] = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    header_path.write_text(json.dumps(header), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read image research snapshot"):
        load_image_snapshot(snapshot_path)


def test_reader_normalizes_malformed_identity_and_offset_shapes(tmp_path: Path) -> None:
    scalar_sample_ids = {
        "sample_ids": np.asarray("pizza-a", dtype=np.str_),
        "component_offsets": np.asarray([0], dtype=np.int64),
        "component_count_total": np.asarray(0, dtype=np.int64),
    }
    sample_snapshot, _ = _write_snapshot(
        tmp_path / "scalar-sample-ids", arrays=scalar_sample_ids, diagnostic_axes_schema={}
    )
    with pytest.raises(ValueError, match=r"arrays\.sample_ids"):
        load_image_snapshot(sample_snapshot)

    scalar_offsets = {
        "sample_ids": np.asarray(["pizza-a"], dtype=np.str_),
        "component_offsets": np.asarray(0, dtype=np.int64),
        "component_count_total": np.asarray(0, dtype=np.int64),
    }
    offset_snapshot, _ = _write_snapshot(
        tmp_path / "scalar-offsets", arrays=scalar_offsets, diagnostic_axes_schema={}
    )
    with pytest.raises(ValueError, match=r"arrays\.component_offsets"):
        load_image_snapshot(offset_snapshot)


def test_discovery_returns_all_valid_snapshots(tmp_path: Path) -> None:
    run = tmp_path / "run"
    first, _ = _write_snapshot(run / "snapshots" / "epoch-00001")
    second, _ = _write_snapshot(run / "snapshots" / "epoch-00002")

    assert discover_image_snapshots(run) == [first, second]


def test_discovery_skips_corrupt_geometry_and_keeps_valid_snapshots(tmp_path: Path) -> None:
    run = tmp_path / "run"
    valid, _ = _write_snapshot(run / "snapshots" / "epoch-00001")
    invalid, _ = _write_snapshot(run / "snapshots" / "epoch-00002", geometry={})

    assert discover_image_snapshots(run) == [valid]
    with pytest.raises(ValueError, match=r"header\.geometry\.box_low"):
        load_image_snapshot(invalid)


def test_discovery_returns_no_snapshots_when_every_candidate_is_corrupt(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_snapshot(run / "snapshots" / "epoch-00001", geometry={})
    _write_snapshot(run / "snapshots" / "epoch-00002", geometry=[])

    assert discover_image_snapshots(run) == []
