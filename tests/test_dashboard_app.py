from __future__ import annotations

import json
from pathlib import Path

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
    # The dataset explorer now exposes the canonical pizza-target state/mask table
    # as well as the component-occurrence table.
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


def test_dashboard_image_embeddings_page_loads_without_training_or_checkpoints(tmp_path) -> None:
    app_path = Path(__file__).parents[1] / "src" / "town_model" / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Image Embeddings")
    app.sidebar.text_input[2].set_value(str(tmp_path / "empty-image-runs"))
    app.run()
    assert not app.exception
    assert any("Image Embeddings" in item.value for item in app.subheader)
    assert any("pre-detector exploratory embeddings" in item.value for item in app.warning)
    assert any("No Phase 3 image runs" in item.value for item in app.info)


def test_dashboard_image_embeddings_inspects_a_saved_run_without_loading_checkpoints(tmp_path, capsys, monkeypatch) -> None:
    import torch
    from town_model.cli import main

    repository = Path(__file__).parents[1]
    example = repository / "examples" / "pizza-48"
    config = repository / "configs" / "phase3_image_smoke.json"
    runs_root = tmp_path / "image-runs"
    assert main(["experiment", "image", "run", "--config", str(config), "--output", str(runs_root)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"

    def fail_checkpoint_load(*args, **kwargs):
        raise AssertionError("read-only dashboard must not deserialize a model/optimizer checkpoint")

    monkeypatch.setattr(torch, "load", fail_checkpoint_load)
    app_path = repository / "src" / "town_model" / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Image Embeddings")
    app.sidebar.text_input[0].set_value(str(example))
    app.sidebar.text_input[2].set_value(str(runs_root))
    app.run()
    assert not app.exception
    assert any(item.label == "Historical epoch snapshot" for item in app.selectbox)
    assert any(item.label == "GT component occurrence" for item in app.selectbox)
    assert any(item.label == "Trainable ontology class" for item in app.selectbox)
    assert len(app.image) >= 1
    assert not any("checkpoint" in item.value.lower() and "error" in item.value.lower() for item in app.error)


def test_dashboard_generate_page_is_available_without_triggering_generation() -> None:
    app_path = Path(__file__).parents[1] / "src" / "town_model" / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Generate dataset")
    app.run()
    assert not app.exception
    assert any(item.value == "Explicit dataset generation" for item in app.subheader)
    assert any(item.label == "Validate settings" for item in app.button)
    # The UI offers generation as a separate explicit control; no output is
    # created merely by loading or navigating to the page.
