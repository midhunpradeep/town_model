from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from town_model.experiments.synthetic import (
    SyntheticConfig,
    evaluate_saved_run,
    load_saved_run,
    make_fixture,
    run_experiment,
    saved_geometry,
)


def test_fixture_splits_and_masks_are_deterministic_and_shared() -> None:
    config = SyntheticConfig(seed=17, points=50, steps=1)
    first, _ = make_fixture("fixed", "simple", config=config)
    second, _ = make_fixture("fixed", "simple", config=config)
    assert first.fixture_id == second.fixture_id
    assert torch.equal(first.features, second.features)
    assert torch.equal(first.targets, second.targets)
    assert torch.equal(first.known_mask, second.known_mask)
    assert not set(first.train_indices.tolist()) & set(first.validation_indices.tolist())
    assert len(first.train_indices) + len(first.validation_indices) == config.points
    assert first.class_names == ("right_half", "positive_diagonal")


def test_all_three_fixed_classifiers_train_reload_and_share_identical_fixture(tmp_path: Path) -> None:
    group = run_experiment("fixed", "simple", tmp_path, config=SyntheticConfig(seed=5, points=60, steps=3, snapshot_interval=1))
    comparison = json.loads((group / "comparison.json").read_text())
    assert comparison["identical_fixture"] and comparison["identical_train_validation_split"]
    fixture = np.load(group / "fixture.npz")
    assert fixture["features"].shape == (60, 2)
    assert {path.name for path in group.iterdir() if path.is_dir()} >= {"baseline", "box", "town"}
    for family in ("baseline", "box", "town"):
        run = group / family
        metrics = json.loads((run / "metrics.json").read_text())
        assert metrics["training_metrics"]["objective"] == ("bce" if family == "baseline" else "signed_distance")
        assert metrics["validation_metrics"]["counts"]["known"] > 0
        assert metrics["parameter_count"] > 0
        assert metrics["history"]
        assert metrics["gradient_norms_last_step"]
        reloaded = evaluate_saved_run(run, "validation")
        assert reloaded["loss"] == pytest.approx(metrics["validation_metrics"]["loss"])
        model, encoder, bundle, arrays = load_saved_run(run)
        assert model.family == family and encoder is None
        assert bundle["fixture"]["fixture_id"] == comparison["fixture_id"]
        assert arrays["features"].shape[0] == 60
        initial = saved_geometry(run, "initial")
        final = saved_geometry(run, "final")
        assert initial["points"].shape == final["points"].shape == (60, 2)
        assert final["output"].contains.shape == (60, 2)
        assert initial["snapshot_activity_step"] == 0
        assert initial["snapshot_activity_source"] == "snapshot"
        assert final["snapshot_activity_step"] == final["step"]
        assert final["snapshot_activity_source"] == "snapshot"


def test_fixed_nonconvex_fixture_has_closed_window_exclusion_and_multiple_regions() -> None:
    fixture, _ = make_fixture("fixed", "window", config=SyntheticConfig(points=100, seed=3))
    points = fixture.features
    labels = fixture.targets
    assert fixture.class_names == ("frame_minus_window", "two_outer_lobes")
    assert not torch.any(labels[:, 0].bool() & ((points[:, 0].abs() <= 0.3) & (points[:, 1].abs() <= 0.3)))
    assert torch.any(labels[:, 1] == 1) and torch.any(labels[:, 1] == 0)


def test_trainable_mapping_has_matched_initial_encoders_and_gradient_paths(tmp_path: Path) -> None:
    group = run_experiment("mapping", "window", tmp_path, config=SyntheticConfig(seed=19, points=60, steps=3, snapshot_interval=1))
    comparison = json.loads((group / "comparison.json").read_text())
    hashes = []
    for family in ("baseline", "box", "town"):
        run = group / family
        checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
        hashes.append(checkpoint["encoder_config"]["initial_sha256"])
        assert checkpoint["encoder_config"]["initial_sha256"] == comparison["shared_initial_encoder_sha256"]
        metrics = json.loads((run / "metrics.json").read_text())
        assert metrics["encoder_parameter_count"] > 0
        assert metrics["gradient_norms_last_step"]["point_encoder"]["norm_l2"] > 0
        if family == "box":
            assert metrics["gradient_norms_last_step"]["geometry"]["norm_l2"] > 0
        if family == "town":
            assert metrics["gradient_norms_last_step"]["frames"]["norm_l2"] > 0
            assert metrics["gradient_norms_last_step"]["windows"]["norm_l2"] > 0
        model, encoder, _, arrays = load_saved_run(run)
        assert encoder is not None
        assert encoder(torch.from_numpy(arrays["features"]).float()).abs().max() <= 1
    assert len(set(hashes)) == 1


def test_missing_snapshot_or_unknown_split_fails_clearly(tmp_path: Path) -> None:
    group = run_experiment("fixed", "simple", tmp_path, config=SyntheticConfig(points=30, steps=1, snapshot_interval=1))
    with pytest.raises(ValueError, match="split"):
        evaluate_saved_run(group / "baseline", "test")
    with pytest.raises(FileNotFoundError, match="snapshot"):
        saved_geometry(group / "baseline", "step_99999")
