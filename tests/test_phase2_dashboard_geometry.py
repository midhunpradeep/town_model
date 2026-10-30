from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from streamlit.testing.v1 import AppTest

from town_model.dashboard.geometry import (
    build_comparison_figure,
    build_history_figure,
    build_motion_figure,
    build_region_figure,
    comparison_rows,
    discover_run_groups,
    load_geometry_view,
)
from town_model.experiments.synthetic import SyntheticConfig, run_experiment
from town_model.models.classifiers import ClassifierOutput
from town_model.models.geometry import evaluate_town_corners


class _PlotTown:
    family = "town"

    def __init__(self, window_lows: list[list[float]], window_highs: list[list[float]], dimensions: int = 2):
        self.dimensions = dimensions
        self.frame_low = torch.full((1, 2, dimensions), -0.8)
        self.frame_high = torch.full((1, 2, dimensions), 0.8)
        self.window_low = torch.tensor(window_lows, dtype=torch.float32).reshape(1, 2, 1, dimensions)
        self.window_high = torch.tensor(window_highs, dtype=torch.float32).reshape(1, 2, 1, dimensions)
        self.geometry = SimpleNamespace(
            physical_geometry=lambda: (self.frame_low, self.frame_high, self.window_low, self.window_high),
            window_mode="contained",
        )

    def eval(self):
        return self

    def __call__(self, points: torch.Tensor):
        result = evaluate_town_corners(points, self.frame_low, self.frame_high, self.window_low, self.window_high)
        return ClassifierOutput(contains=result.contains, signed_margin=result.signed_margin)


def _plot_view(model: _PlotTown) -> dict:
    return {
        "model": model,
        "class_names": ("class",),
        "points": torch.zeros((1, model.dimensions), dtype=torch.float32),
        "sample_index": 0,
        "sample_id": "point-00000",
        "split_indices": np.array([0]),
        "split": "train",
        "targets": torch.ones((1, 1), dtype=torch.float32),
        "known_mask": torch.ones((1, 1), dtype=torch.bool),
    }


@pytest.mark.parametrize("dimensions,axes", [(2, (0, 1)), (3, (0, 2))])
def test_town_plot_keeps_overlap_reinclusion_and_window_boundaries_faithful(dimensions, axes) -> None:
    model = _PlotTown(
        [[-0.2] * dimensions, [0.6] * dimensions],
        [[0.2] * dimensions, [0.7] * dimensions],
        dimensions=dimensions,
    )
    fig, facts = build_region_figure(_plot_view(model), class_name="class", axes=axes, grid_size=91)
    ax = fig.axes[0]
    mesh = ax.collections[0]
    membership = mesh.get_array().reshape(91, 91)
    assert facts["selected_membership"] is True
    assert bool(membership[45, 45]) is True  # (0,0) is re-included by house 1.
    windows = [patch for patch in ax.patches if patch.get_label().startswith("house ") and "boundary (house-specific exclusion)" in patch.get_label()]
    assert len(windows) == (2 if dimensions == 2 else 1)
    assert all(not patch.get_fill() for patch in windows)
    assert all(patch.get_linestyle() == ":" for patch in windows)
    assert any(text.get_text() == "final town membership (model-evaluated)" for text in ax.get_legend().get_texts())
    plt.close(fig)

    # When both houses exclude the center, the final membership mesh has a real
    # hole, while the individual windows remain visible as outline-only patches.
    excluded = _PlotTown(
        [[-0.2, -0.2], [-0.1, -0.1]],
        [[0.2, 0.2], [0.1, 0.1]],
    )
    excluded_fig, excluded_facts = build_region_figure(_plot_view(excluded), class_name="class", grid_size=91)
    excluded_mesh = excluded_fig.axes[0].collections[0].get_array().reshape(91, 91)
    assert excluded_facts["selected_membership"] is False
    assert bool(excluded_mesh[45, 45]) is False
    outlined_windows = [patch for patch in excluded_fig.axes[0].patches if "boundary (house-specific exclusion)" in patch.get_label()]
    assert len(outlined_windows) == 2 and all(not patch.get_fill() for patch in outlined_windows)
    plt.close(excluded_fig)


def test_saved_run_discovery_selection_region_metrics_and_comparison(tmp_path: Path) -> None:
    group = run_experiment("fixed", "window", tmp_path, config=SyntheticConfig(seed=4, points=60, steps=2, snapshot_interval=1))
    assert discover_run_groups(tmp_path) == [group]
    result = load_geometry_view(group / "box", split="validation", sample_position=0, snapshot="initial")
    assert result.error is None and result.view is not None
    view = result.view
    fig, info = build_region_figure(view, class_name=view["class_names"][0], anchor_mode="zero")
    assert info["is_exact_region_slice"] is True
    assert isinstance(info["slice_is_empty"], bool)
    assert info["selected_score_name"] == "signed margin"
    plt.close(fig)
    baseline_view = load_geometry_view(group / "baseline", split="validation", sample_position=0).view
    baseline_figure, baseline_info = build_region_figure(baseline_view, class_name=baseline_view["class_names"][0])
    assert baseline_info["decision_boundary_source"] == "actual_logits_zero"
    assert baseline_info["selected_score_name"] == "logit"
    plt.close(baseline_figure)
    rows = comparison_rows(group)
    assert [row["family"] for row in rows] == ["baseline", "box", "town"]
    assert {row["objective"] for row in rows} == {"bce", "signed_distance"}
    comparison, facts = build_comparison_figure(group, class_name=view["class_names"][0], split="validation", sample_position=0)
    assert set(facts) == {"baseline", "box", "town"}
    assert len(comparison.axes) == 3
    plt.close(comparison)
    history = build_history_figure(group)
    assert len(history.axes) == 3
    plt.close(history)
    motion = build_motion_figure(load_geometry_view(group / "town").view, class_name="frame_minus_window")
    assert len(motion.axes) == 2
    plt.close(motion)


def test_snapshot_activity_tracks_initial_intermediate_and_final_geometry(tmp_path: Path) -> None:
    group = run_experiment(
        "fixed", "window", tmp_path,
        config=SyntheticConfig(seed=0, points=80, steps=20, learning_rate=0.08, unknown_fraction=0.0, snapshot_interval=5),
    )
    activities = {}
    for name, expected_step in (("initial", 0), ("step_00005", 5), ("final", 20)):
        result = load_geometry_view(group / "town", snapshot=name)
        assert result.error is None and result.view is not None
        view = result.view
        assert view["snapshot_activity_source"] == "snapshot"
        assert view["snapshot_activity_step"] == expected_step == view["step"]
        assert view["snapshot_activity_split"] == "train"
        activities[name] = view["snapshot_activity"]["activity"]["window_membership_observations"]
    total = lambda activity: sum(sum(sum(window_counts) for window_counts in house_counts) for house_counts in activity)
    initial_total = total(activities["initial"])
    final_total = total(activities["final"])
    assert initial_total == 52 and final_total == 78
    assert final_total >= initial_total * 1.4


def test_historical_snapshot_without_optional_activity_remains_readable(tmp_path: Path) -> None:
    group = run_experiment(
        "fixed", "simple", tmp_path,
        config=SyntheticConfig(seed=9, points=30, steps=2, snapshot_interval=1),
    )
    snapshot_path = group / "town" / "snapshots" / "step_00001.pt"
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    snapshot.pop("activity")
    snapshot.pop("activity_split")
    torch.save(snapshot, snapshot_path)
    result = load_geometry_view(group / "town", snapshot="step_00001")
    assert result.error is None and result.view is not None
    assert result.view["step"] == 1
    assert result.view["snapshot_activity"] is None
    assert "predates snapshot-specific activity" in result.view["snapshot_activity_compatibility_note"]


def test_high_dimensional_view_uses_actual_sample_anchor_and_explicit_slice(tmp_path: Path) -> None:
    group = run_experiment("fixed", "simple", tmp_path, config=SyntheticConfig(seed=8, points=40, dimensions=4, steps=1, snapshot_interval=1))
    view = load_geometry_view(group / "town", split="train", sample_position=2).view
    fig, info = build_region_figure(view, class_name="right_half", anchor_mode="sample", axes=(1, 3), grid_size=18)
    assert info["is_exact_region_slice"] is True
    assert info["point_markers_are_projections"] is True
    selected = view["points"][view["sample_index"]]
    assert info["anchor"] == selected.tolist()
    assert info["axes"] == (1, 3)
    plt.close(fig)


def test_invalid_or_missing_runs_are_reported_without_crashing(tmp_path: Path) -> None:
    assert discover_run_groups(tmp_path / "missing") == []
    missing = load_geometry_view(tmp_path / "missing" / "baseline")
    assert missing.view is None and "Could not load" in missing.error
    invalid = load_geometry_view(tmp_path / "missing" / "baseline", split="test")
    assert invalid.view is None and "train and validation" in invalid.error
    group = run_experiment("fixed", "simple", tmp_path, config=SyntheticConfig(points=30, steps=1))
    outside = load_geometry_view(group / "box", split="train", sample_position=1000)
    assert outside.view is None and "outside" in outside.error
    with pytest.raises(ValueError, match="missing or invalid"):
        comparison_rows(group / "not-a-run")


def test_streamlit_geometry_laboratory_loads_saved_runs_without_retraining(tmp_path: Path) -> None:
    group = run_experiment("fixed", "simple", tmp_path, config=SyntheticConfig(seed=12, points=40, steps=2, snapshot_interval=1))
    app_path = Path(__file__).parents[1] / "src" / "town_model" / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Geometry Laboratory")
    run_root = next(widget for widget in app.sidebar.text_input if widget.label == "Synthetic run directory")
    run_root.set_value(str(tmp_path))
    app.run()
    assert not app.exception
    assert any(element.value == "Geometry Laboratory" for element in app.subheader)
    assert any(widget.label == "Classifier model" for widget in app.selectbox)
    assert any(widget.label == "Concept / class" for widget in app.selectbox)
    assert any(widget.label == "Dataset split" for widget in app.selectbox)
    assert any(widget.label == "Checkpoint / snapshot" for widget in app.selectbox)
    assert any("Compare three models" in str(tab.label) for tab in app.get("tabs"))
    assert any(metric.label == "Exact membership" for metric in app.metric)
    assert any("Activity for selected snapshot" in str(item.value) for item in app.markdown)
    assert any("Final-step gradient diagnostics" in str(item.value) for item in app.markdown)
