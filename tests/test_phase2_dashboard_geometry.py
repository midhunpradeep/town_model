from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
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
