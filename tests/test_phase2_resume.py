from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from town_model.experiments.synthetic import SyntheticConfig, evaluate_saved_run, load_saved_run, resume_saved_run, run_experiment


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


def test_schema_one_checkpoint_without_window_policy_remains_loadable(tmp_path: Path) -> None:
    group = run_experiment("fixed", "window", tmp_path, config=SyntheticConfig(seed=21, points=30, steps=1))
    run = group / "town"
    resolved_path = run / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["schema_version"] = 1
    resolved["model"].pop("window_target_volume")
    resolved_path.write_text(json.dumps(resolved, indent=2, sort_keys=True))
    checkpoint_path = run / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["checkpoint_schema"] = 1
    checkpoint["model_config"].pop("window_target_volume")
    checkpoint["config_identity"] = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
    torch.save(checkpoint, checkpoint_path)

    model, _, _, _ = load_saved_run(run)
    assert model.geometry.window_axis_fraction == pytest.approx(0.72)
    assert evaluate_saved_run(run)["loss"] == pytest.approx(
        json.loads((run / "metrics.json").read_text())["validation_metrics"]["loss"], abs=1e-12
    )


def test_zero_margin_training_reload_and_default_margin(tmp_path: Path) -> None:
    zero_group = run_experiment(
        "fixed", "window", tmp_path / "zero",
        config=SyntheticConfig(seed=31, points=48, steps=2, margin=0.0, snapshot_interval=1),
    )
    for family in ("box", "town"):
        run = zero_group / family
        config = json.loads((run / "resolved_config.json").read_text())
        checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert config["schema_version"] == 2
        if family == "town":
            assert config["model"]["window_target_volume"] == 0.25
        assert config["training"]["margin"] == 0.0
        assert checkpoint["checkpoint_schema"] == 2
        assert checkpoint["config_identity"] == hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        saved_loss = json.loads((run / "metrics.json").read_text())["validation_metrics"]["loss"]
        assert evaluate_saved_run(run, "validation")["loss"] == pytest.approx(saved_loss, abs=1e-12)

    default_group = run_experiment(
        "fixed", "simple", tmp_path / "default",
        config=SyntheticConfig(seed=32, points=40, steps=1),
    )
    for family in ("box", "town"):
        run = default_group / family
        config = json.loads((run / "resolved_config.json").read_text())
        assert config["training"]["margin"] == 0.1
        metrics = json.loads((run / "metrics.json").read_text())
        assert evaluate_saved_run(run)["loss"] == pytest.approx(metrics["validation_metrics"]["loss"], abs=1e-12)


def test_zero_margin_split_resume_matches_uninterrupted_run_and_keeps_identity(tmp_path: Path) -> None:
    full_group = run_experiment(
        "fixed", "window", tmp_path / "full",
        config=SyntheticConfig(seed=43, points=48, steps=4, margin=0.0, unknown_fraction=0.0, snapshot_interval=2),
    )
    split_group = run_experiment(
        "fixed", "window", tmp_path / "split",
        config=SyntheticConfig(seed=43, points=48, steps=2, margin=0.0, unknown_fraction=0.0, snapshot_interval=2),
    )
    for family in ("baseline", "box", "town"):
        full_run = full_group / family
        split_run = split_group / family
        resolved_before = (split_run / "resolved_config.json").read_bytes()
        checkpoint_before = torch.load(split_run / "checkpoint.pt", map_location="cpu", weights_only=False)
        identity_before = checkpoint_before["config_identity"]
        resume_saved_run(split_run, 2)
        assert (split_run / "resolved_config.json").read_bytes() == resolved_before
        resumed_checkpoint = torch.load(split_run / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert resumed_checkpoint["config_identity"] == identity_before
        assert resumed_checkpoint["config_identity"] == checkpoint_before["config_identity"]
        resolved_margin = json.loads((split_run / "resolved_config.json").read_text())["training"]["margin"]
        assert resolved_margin == (None if family == "baseline" else 0.0)

        full_model, full_encoder, _, _ = load_saved_run(full_run)
        resumed_model, resumed_encoder, _, _ = load_saved_run(split_run)
        assert full_encoder is None and resumed_encoder is None
        for name, value in full_model.state_dict().items():
            assert torch.equal(value, resumed_model.state_dict()[name]), name
        full_optimizer = torch.load(full_run / "checkpoint.pt", map_location="cpu", weights_only=False)["optimizer_state"]
        resumed_optimizer = resumed_checkpoint["optimizer_state"]
        assert full_optimizer.keys() == resumed_optimizer.keys()
        assert full_optimizer["param_groups"] == resumed_optimizer["param_groups"]
        for key, value in full_optimizer["state"].items():
            for state_name, state_value in value.items():
                other = resumed_optimizer["state"][key][state_name]
                if isinstance(state_value, torch.Tensor):
                    assert torch.equal(state_value, other)
                else:
                    assert state_value == other
        reloaded = evaluate_saved_run(split_run, "validation")
        final_metrics = json.loads((split_run / "metrics.json").read_text())
        assert reloaded["loss"] == pytest.approx(final_metrics["validation_metrics"]["loss"], abs=1e-12)
