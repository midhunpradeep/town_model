from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from town_model.experiments.synthetic import SyntheticConfig, load_saved_run, resume_saved_run, run_experiment


def test_resume_restores_adam_and_appends_steps_without_mutating_resolved_config(tmp_path: Path) -> None:
    group = run_experiment("mapping", "simple", tmp_path, config=SyntheticConfig(seed=4, points=40, steps=2, snapshot_interval=2))
    run = group / "box"
    config_before = (run / "resolved_config.json").read_bytes()
    resumed = resume_saved_run(run, additional_steps=2)
    assert (run / "resolved_config.json").read_bytes() == config_before
    assert resumed["resume_segments"][-1]["from_step"] == 2
    assert resumed["resume_segments"][-1]["to_step"] == 4
    assert resumed["history"][-1]["step"] == 4
    assert resumed["history"][-1]["resumed"] is True
    model, encoder, bundle, arrays = load_saved_run(run)
    assert bundle["checkpoint"]["step"] == 4
    assert encoder is not None
    assert (run / "snapshots" / "step_00004.pt").is_file()


def test_modified_fixture_arrays_are_rejected_on_reload(tmp_path: Path) -> None:
    group = run_experiment("fixed", "simple", tmp_path, config=SyntheticConfig(points=30, steps=1))
    run = group / "baseline"
    fixture = group / "fixture.npz"
    with np.load(fixture, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["features"][0, 0] += 0.1
    np.savez_compressed(fixture, **arrays)
    with pytest.raises(ValueError, match="content hash"):
        load_saved_run(run)


def test_incompatible_resolved_config_is_rejected_on_checkpoint_reload(tmp_path: Path) -> None:
    group = run_experiment("fixed", "simple", tmp_path, config=SyntheticConfig(points=30, steps=1))
    run = group / "baseline"
    config_path = run / "resolved_config.json"
    config = json.loads(config_path.read_text())
    config["training"]["learning_rate"] *= 2
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="resolved configuration"):
        load_saved_run(run)
