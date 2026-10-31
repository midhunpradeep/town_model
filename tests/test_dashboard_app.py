from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest


def test_dashboard_explorer_loads_shared_dataset_and_renders_annotations(prepared_dataset: Path) -> None:
    app_path = Path(__file__).parents[1] / "src" / "town_model" / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception

    app.sidebar.radio[0].set_value("Image explorer")
    app.sidebar.text_input[0].set_value(str(prepared_dataset))
    app.run()
    assert not app.exception
    assert any(item.value == "Image and annotation explorer" for item in app.subheader)
    assert any(item.label == "Pizza image" for item in app.selectbox)
    assert any("Source-inferred labels:" in item.value for item in app.markdown)
    assert len(app.dataframe) == 2


def test_dashboard_marks_corrupt_data_unverified_and_blocks_explorer(prepared_dataset: Path) -> None:
    samples_path = prepared_dataset / "samples.jsonl"
    records = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
    records[0]["pizza"]["ontology_targets"][0] = 1 - records[0]["pizza"]["ontology_targets"][0]
    samples_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    app_path = Path(__file__).parents[1] / "src" / "town_model" / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.sidebar.text_input[0].set_value(str(prepared_dataset))
    app.run()
    assert not app.exception
    assert any("Current integrity validation: FAIL" in item.value for item in app.error)
    assert not app.dataframe

    app.sidebar.radio[0].set_value("Image explorer")
    app.run()
    assert not app.exception
    assert any("Explorer disabled" in item.value for item in app.error)
    assert not any(item.label == "Pizza image" for item in app.selectbox)
    assert not app.get("imgs")


def test_dashboard_generate_page_is_available_without_triggering_generation() -> None:
    app_path = Path(__file__).parents[1] / "src" / "town_model" / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Generate dataset")
    app.run()
    assert not app.exception
    assert any(item.value == "Explicit dataset generation" for item in app.subheader)
    assert any(item.label == "Validate settings" for item in app.button)


def test_image_embeddings_displays_only_persisted_metadata_and_checksums(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run = runs_root / "historical-run"
    run.mkdir(parents=True)
    identity_hash = "stored-run-identity"
    (run / "manifest.json").write_text(json.dumps({
        "schema_version": "image-run-manifest-v1",
        "run_id": run.name,
        "identity_hash": identity_hash,
        "state": "completed",
    }))
    (run / "resolved-config.json").write_text(json.dumps({"identity_hash": identity_hash, "stored": True}))
    (run / "metrics.json").write_text(json.dumps({"historical_metric": 0.25}))

    snapshot_dir = run / "snapshots" / "epoch-00001"
    snapshot_dir.mkdir(parents=True)
    arrays_path = snapshot_dir / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        sample_ids=np.asarray([], dtype=np.str_),
        component_offsets=np.asarray([0], dtype=np.int64),
        component_count_total=np.asarray(0, dtype=np.int64),
    )
    arrays_sha256 = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    (snapshot_dir / "header.json").write_text(json.dumps({
        "schema_version": "image-research-snapshot-v2",
        "epoch": 1,
        "optimizer_step": 3,
        "identity_hash": identity_hash,
        "dataset_id": "historical-dataset",
        "manifest_id": "historical-manifest",
        "source_checkpoint_sha256": "stored-checkpoint-hash",
        "arrays_sha256": arrays_sha256,
        "sample_ids": [],
        "source_splits": [],
        "class_ids": [],
        "class_names": [],
        "component_count_original": 0,
        "component_count_traced": 0,
        "diagnostic_axes_schema": {},
        "geometry": {},
    }))

    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    app_path = Path(__file__).parents[1] / "src" / "town_model" / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Image Embeddings")
    app.sidebar.text_input[2].set_value(str(runs_root))
    app.run()
    assert not app.exception
    assert any("historical values only" in item.value.lower() or "persisted values only" in item.value.lower() for item in app.caption)
    assert any(item.label == "Checksummed historical snapshot metadata" for item in app.selectbox)
    assert not app.button
    assert {path: path.read_bytes() for path in before} == before
