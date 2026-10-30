from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from streamlit.testing.v1 import AppTest

from town_model.cli import main
from town_model.dashboard.geometry import load_geometry_view
from town_model.experiments.synthetic import (
    SavedArtifactError,
    SyntheticConfig,
    evaluate_saved_run,
    load_saved_run,
    resume_saved_run,
    run_experiment,
)


def _make_run(root: Path) -> Path:
    group = run_experiment(
        "fixed", "simple", root,
        config=SyntheticConfig(seed=12, points=30, steps=1, snapshot_interval=1),
    )
    return group / "town"


def test_invalid_and_truncated_checkpoints_raise_actionable_artifact_errors(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    checkpoint = run / "checkpoint.pt"
    valid_payload = checkpoint.read_bytes()
    assert evaluate_saved_run(run)["split"] == "validation"

    checkpoint.write_bytes(b"not a PyTorch checkpoint")
    with pytest.raises(SavedArtifactError, match="could not read or deserialize") as invalid:
        load_saved_run(run)
    assert str(checkpoint) in str(invalid.value)

    checkpoint.write_bytes(valid_payload[: len(valid_payload) // 2])
    with pytest.raises(SavedArtifactError, match="could not read or deserialize") as truncated:
        load_saved_run(run)
    assert str(checkpoint) in str(truncated.value)


def test_structurally_incompatible_checkpoint_is_rejected_without_losing_integrity_checks(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    checkpoint = run / "checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["fixture_id"] = "wrong-fixture"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="fixture identity"):
        load_saved_run(run)

    fixture_id = json.loads((run.parent / "fixture.json").read_text())["fixture_id"]
    payload["fixture_id"] = fixture_id
    payload["model_state"] = {}
    torch.save(payload, checkpoint)
    with pytest.raises(SavedArtifactError, match="incompatible model state"):
        load_saved_run(run)

    torch.save({"checkpoint_schema": 2}, checkpoint)
    with pytest.raises(SavedArtifactError, match="missing fields") as incompatible:
        load_saved_run(run)
    assert str(checkpoint) in str(incompatible.value)


def test_cli_and_resume_report_corruption_as_actionable_errors(tmp_path: Path, capsys) -> None:
    run = _make_run(tmp_path)
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_bytes(b"truncated checkpoint")

    with pytest.raises(SystemExit) as exit_info:
        main(["evaluate", str(run)])
    assert exit_info.value.code == 2
    stderr = capsys.readouterr().err
    assert str(checkpoint) in stderr
    assert "could not read or deserialize" in stderr
    assert "Traceback" not in stderr

    with pytest.raises(SavedArtifactError, match="could not read or deserialize"):
        resume_saved_run(run, additional_steps=1)


def test_dashboard_reader_reports_corrupt_checkpoint_as_failed_run(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_bytes(b"invalid bytes")
    result = load_geometry_view(run)
    assert result.view is None
    assert str(checkpoint) in result.error
    assert "could not read or deserialize" in result.error


def test_geometry_laboratory_app_test_survives_corrupt_checkpoint(tmp_path: Path) -> None:
    group = run_experiment(
        "fixed", "simple", tmp_path,
        config=SyntheticConfig(seed=13, points=30, steps=1, snapshot_interval=1),
    )
    checkpoint = group / "town" / "checkpoint.pt"
    checkpoint.write_bytes(b"invalid checkpoint bytes")
    app_path = Path(__file__).parents[1] / "src" / "town_model" / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Geometry Laboratory")
    run_root = next(widget for widget in app.sidebar.text_input if widget.label == "Synthetic run directory")
    run_root.set_value(str(tmp_path))
    app.run()
    next(widget for widget in app.selectbox if widget.label == "Classifier model").set_value("town")
    app.run()
    assert not app.exception
    assert any(str(checkpoint) in str(element.value) for element in app.error)
