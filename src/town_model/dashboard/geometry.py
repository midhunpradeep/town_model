"""Streamlit-independent readers and faithful slice plots for synthetic runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import torch

from town_model.classifier_backend import raise_classifier_backend_unavailable
from town_model.experiments.synthetic import MODEL_FAMILIES, saved_geometry


@dataclass(frozen=True, slots=True)
class GeometryViewResult:
    view: dict[str, Any] | None
    error: str | None


def discover_run_groups(root: Path | str) -> list[Path]:
    """Return completed, dashboard-readable comparison groups below root."""
    base = Path(root).expanduser()
    if not base.is_dir():
        return []
    groups = []
    for candidate in sorted(base.iterdir()):
        if not candidate.is_dir() or not (candidate / "comparison.json").is_file():
            continue
        try:
            comparison = json.loads((candidate / "comparison.json").read_text(encoding="utf-8"))
            if comparison.get("status") == "completed" and all((candidate / family / "metrics.json").is_file() for family in MODEL_FAMILIES):
                groups.append(candidate)
        except (OSError, ValueError, TypeError):
            continue
    return groups


def load_geometry_view(
    run_path: Path | str,
    *,
    split: str = "train",
    sample_position: int = 0,
    snapshot: str = "final",
) -> GeometryViewResult:
    """Load historical model state/predictions (unavailable during reconstruction)."""
    raise_classifier_backend_unavailable("historical synthetic classifier geometry inspection")
    if split not in {"train", "validation"}:
        return GeometryViewResult(None, "Geometry Laboratory exposes train and validation splits only.")
    run = Path(run_path).expanduser().resolve()
    try:
        geometry = saved_geometry(run, snapshot)
        metadata = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
        arrays = geometry["arrays"]
        indices = arrays[f"{split}_indices"].astype(np.int64)
        if not 0 <= sample_position < len(indices):
            return GeometryViewResult(None, f"Point selection {sample_position} is outside the {split} split (size {len(indices)}).")
        sample_index = int(indices[sample_position])
        fixture = geometry["fixture"]
        view = {
            **geometry,
            "metrics": metadata,
            "config": config,
            "split": split,
            "split_indices": indices,
            "sample_position": sample_position,
            "sample_index": sample_index,
            "sample_id": f"point-{sample_index:05d}",
            "class_names": tuple(fixture["class_names"]),
            "targets": torch.from_numpy(arrays["targets"]).to(torch.float32),
            "known_mask": torch.from_numpy(arrays["known_mask"]).to(torch.bool),
        }
        return GeometryViewResult(view, None)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return GeometryViewResult(None, f"Could not load saved synthetic run at {run}: {error}")


def comparison_rows(group_path: Path | str) -> list[dict[str, Any]]:
    group = Path(group_path).expanduser()
    rows = []
    for family in MODEL_FAMILIES:
        run = group / family
        try:
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"missing or invalid {family} run in {group}: {error}") from error
        val = metrics["validation_metrics"]
        train = metrics["training_metrics"]
        gradients = metrics.get("gradient_norms_last_step", {})
        rows.append({
            "family": family,
            "objective": metrics["objective"],
            "train loss": train["loss"],
            "validation loss": val["loss"],
            "known-label accuracy": val["known_label_accuracy"],
            "macro F1": val["macro_f1"],
            "known / positive / negative / unknown": " / ".join(str(val["counts"][key]) for key in ("known", "positive", "negative", "unknown")),
            "parameters": metrics["parameter_count"],
            "gradient summary": gradients,
        })
    return rows


def _region_geometry(model, class_index: int) -> tuple[dict[str, torch.Tensor], str]:
    if model.family == "baseline":
        return {}, "baseline decision boundary from logits"
    if model.family == "box":
        low, high = model.geometry.physical_boxes()
        return {"box_low": low[class_index], "box_high": high[class_index]}, "closed physical box"
    fl, fr, wl, wr = model.geometry.physical_geometry()
    return {
        "frame_low": fl[class_index],
        "frame_high": fr[class_index],
        "window_low": wl[class_index],
        "window_high": wr[class_index],
    }, "town frames minus closed windows, unioned by house"


def _slice_intersects(low: torch.Tensor, high: torch.Tensor, fixed: torch.Tensor, axes: tuple[int, int]) -> bool:
    other = [index for index in range(fixed.shape[0]) if index not in axes]
    return all(float(low[index]) <= float(fixed[index]) <= float(high[index]) for index in other)


def _draw_physical_boundaries(ax, model, class_index: int, fixed: torch.Tensor, axes: tuple[int, int]) -> None:
    geometry, _ = _region_geometry(model, class_index)
    axis_x, axis_y = axes
    if model.family == "box":
        low, high = geometry["box_low"], geometry["box_high"]
        if _slice_intersects(low, high, fixed, axes):
            ax.add_patch(Rectangle((float(low[axis_x]), float(low[axis_y])), float(high[axis_x] - low[axis_x]), float(high[axis_y] - low[axis_y]), fill=False, edgecolor="#126782", linewidth=2, label="actual box boundary"))
    elif model.family == "town":
        frame_low, frame_high = geometry["frame_low"], geometry["frame_high"]
        window_low, window_high = geometry["window_low"], geometry["window_high"]
        for house_index in range(frame_low.shape[0]):
            low, high = frame_low[house_index], frame_high[house_index]
            if _slice_intersects(low, high, fixed, axes):
                ax.add_patch(Rectangle((float(low[axis_x]), float(low[axis_y])), float(high[axis_x] - low[axis_x]), float(high[axis_y] - low[axis_y]), fill=False, edgecolor="#126782", linestyle="--", linewidth=1.6, label=f"house {house_index} frame"))
            for window_index in range(window_low.shape[1]):
                wlow, whigh = window_low[house_index, window_index], window_high[house_index, window_index]
                if _slice_intersects(wlow, whigh, fixed, axes):
                    ax.add_patch(Rectangle(
                        (float(wlow[axis_x]), float(wlow[axis_y])),
                        float(whigh[axis_x] - wlow[axis_x]),
                        float(whigh[axis_y] - wlow[axis_y]),
                        fill=False,
                        edgecolor="#8c2b67",
                        linestyle=":",
                        linewidth=2.0,
                        label=f"house {house_index} window {window_index} boundary (house-specific exclusion)",
                    ))


def build_region_figure(
    view: dict[str, Any],
    *,
    class_name: str,
    anchor_mode: str = "zero",
    axes: tuple[int, int] = (0, 1),
    grid_size: int = 90,
):
    """Return historical classifier region facts (unavailable during reconstruction).

    For d>2, region values are evaluated on full-dimensional points with the
    remaining coordinates fixed to zero or the selected sample. Point markers
    are projections onto the displayed coordinates and are labeled as such.
    """
    raise_classifier_backend_unavailable("historical synthetic classifier region visualization")
    model = view["model"]
    class_index = view["class_names"].index(class_name)
    dims = model.dimensions
    if len(set(axes)) != 2 or min(axes) < 0 or max(axes) >= dims:
        raise ValueError("slice axes must be two distinct valid latent dimensions")
    if grid_size < 8:
        raise ValueError("grid_size must be >=8")
    if anchor_mode not in {"zero", "sample"}:
        raise ValueError("anchor_mode must be 'zero' or 'sample'")
    all_points: torch.Tensor = view["points"]
    sample_index = view["sample_index"]
    selected_point = all_points[sample_index]
    fixed = torch.zeros((dims,), dtype=torch.float32)
    if anchor_mode == "sample":
        fixed = selected_point.detach().cpu().clone()
    axis_x, axis_y = axes
    coordinates = torch.linspace(-1.0, 1.0, grid_size)
    xx, yy = torch.meshgrid(coordinates, coordinates, indexing="xy")
    full_points = fixed.repeat(grid_size * grid_size, 1)
    full_points[:, axis_x] = xx.reshape(-1)
    full_points[:, axis_y] = yy.reshape(-1)
    model.eval()
    with torch.no_grad():
        slice_output = model(full_points)
        selected_output = model(selected_point.reshape(1, -1))
    region = slice_output.contains[:, class_index].reshape(grid_size, grid_size).cpu().numpy()
    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    ax.pcolormesh(xx.numpy(), yy.numpy(), region.astype(float), shading="nearest", cmap="Greens", vmin=0, vmax=1, alpha=0.45)
    if slice_output.logits is not None:
        logits = slice_output.logits[:, class_index].reshape(grid_size, grid_size).detach().cpu().numpy()
        if float(logits.min()) <= 0 <= float(logits.max()):
            ax.contour(xx.numpy(), yy.numpy(), logits, levels=[0.0], colors=["#162b75"], linewidths=2, linestyles="-")
        decision_label = "neural baseline actual logit=0 boundary"
    else:
        decision_label = "exact membership + signed-margin geometry"
        _draw_physical_boundaries(ax, model, class_index, fixed, axes)

    indices = view["split_indices"]
    points_np = all_points[indices].detach().cpu().numpy()
    targets = view["targets"][indices, class_index].numpy().astype(bool)
    known = view["known_mask"][indices, class_index].numpy().astype(bool)
    positive = known & targets
    negative = known & ~targets
    unknown = ~known
    ax.scatter(points_np[positive, axis_x], points_np[positive, axis_y], s=24, c="#167d42", marker="o", edgecolors="white", linewidths=0.35, label="known positive (points projected)" if dims > 2 else "known positive")
    ax.scatter(points_np[negative, axis_x], points_np[negative, axis_y], s=27, c="#bf3f36", marker="x", label="known negative (points projected)" if dims > 2 else "known negative")
    ax.scatter(points_np[unknown, axis_x], points_np[unknown, axis_y], s=27, c="#777777", marker="^", label="unknown target (points projected)" if dims > 2 else "unknown target")
    ax.scatter([float(selected_point[axis_x])], [float(selected_point[axis_y])], marker="*", s=180, c="#ffbf00", edgecolors="black", linewidths=0.7, label=f"selected {view['sample_id']}", zorder=6)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel=f"latent coordinate {axis_x}", ylabel=f"latent coordinate {axis_y}")
    ax.set_aspect("equal", adjustable="box")
    anchor_label = "zero anchor" if anchor_mode == "zero" else f"anchor through {view['sample_id']}"
    ax.set_title(f"{model.family} · {class_name} · {view['split']} · {anchor_label}\n{decision_label}")
    handles, labels = ax.get_legend_handles_labels()
    if model.family == "town":
        handles.insert(0, Patch(facecolor="#70ad86", edgecolor="none", alpha=0.45))
        labels.insert(0, "final town membership (model-evaluated)")
    ax.legend(handles, labels, loc="best", fontsize="small")
    selected_contains = bool(selected_output.contains[0, class_index].item())
    if selected_output.signed_margin is not None:
        selected_score = float(selected_output.signed_margin[0, class_index].item())
        score_name = "signed margin"
    else:
        selected_score = float(selected_output.logits[0, class_index].item())
        score_name = "logit"
    info = {
        "class_name": class_name,
        "slice_is_empty": not bool(region.any()),
        "selected_sample_id": view["sample_id"],
        "selected_membership": selected_contains,
        "selected_score_name": score_name,
        "selected_score": selected_score,
        "selected_target": int(view["targets"][sample_index, class_index].item()),
        "selected_known": bool(view["known_mask"][sample_index, class_index].item()),
        "anchor": fixed.tolist(),
        "axes": axes,
        "is_exact_region_slice": True,
        "decision_boundary_source": "actual_logits_zero" if slice_output.logits is not None else None,
        "point_markers_are_projections": dims > 2,
    }
    return fig, info


def build_history_figure(group_path: Path | str):
    group = Path(group_path)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, family in zip(axes, MODEL_FAMILIES):
        metrics = json.loads((group / family / "metrics.json").read_text(encoding="utf-8"))
        rows = metrics["history"]
        steps = [row["step"] for row in rows]
        train = [row["training"]["loss"] for row in rows]
        valid = [row["validation"]["loss"] for row in rows]
        ax.plot(steps, train, marker="o", label=f"train {metrics['objective']}")
        ax.plot(steps, valid, marker="s", label=f"validation {metrics['objective']}")
        ax.set_title(f"{family} · {metrics['objective']}")
        ax.set_xlabel("optimization step")
        ax.set_ylabel("objective value (not cross-family comparable)")
        ax.legend(fontsize="small")
    return fig


def build_motion_figure(view: dict[str, Any], *, class_name: str):
    """Plot historical classifier parameter motion (unavailable during reconstruction)."""
    raise_classifier_backend_unavailable("historical synthetic classifier geometry motion visualization")
    run = view["run_dir"]
    run_meta = view["metrics"]
    names = ["initial", *[name for name in run_meta.get("snapshot_names", []) if name.startswith("step_")], "final"]
    snapshots = [saved_geometry(run, name) for name in names]
    model_family = view["model"].family
    class_index = view["class_names"].index(class_name)
    point_index = view["sample_index"]
    point_xy = np.array([[snap["points"][point_index, 0].item(), snap["points"][point_index, 1].item()] for snap in snapshots])
    steps = [int(snap["step"]) for snap in snapshots]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].plot(point_xy[:, 0], point_xy[:, 1], marker="o", color="#8c2b67")
    for step, (x, y) in zip(steps, point_xy):
        axes[0].annotate(str(step), (x, y), fontsize=8)
    axes[0].set(xlim=(-1, 1), ylim=(-1, 1), xlabel="latent coordinate 0", ylabel="latent coordinate 1", title=f"point trajectory · {view['sample_id']}")
    axes[0].set_aspect("equal", adjustable="box")
    if model_family == "baseline":
        norms = [float(torch.linalg.vector_norm(snap["model"].linear.weight[class_index]).item()) for snap in snapshots]
        axes[1].plot(steps, norms, marker="o", label="linear class-weight norm")
        bias = [float(snap["model"].linear.bias[class_index].item()) for snap in snapshots]
        axes[1].plot(steps, bias, marker="s", label="linear class bias")
        axes[1].set_title("actual baseline parameter motion")
    elif model_family == "box":
        centers = []
        for snap in snapshots:
            low, high = snap["model"].geometry.physical_boxes()
            centers.append(((low[class_index] + high[class_index]) / 2).detach().cpu().numpy())
        centers_np = np.asarray(centers)
        axes[1].plot(steps, centers_np[:, 0], marker="o", label="box center coordinate 0")
        axes[1].plot(steps, centers_np[:, 1], marker="s", label="box center coordinate 1")
        axes[1].set_title("actual box parameter motion")
    else:
        centers = []
        for snap in snapshots:
            fl, fr, _, _ = snap["model"].geometry.physical_geometry()
            centers.append(((fl[class_index] + fr[class_index]) / 2).detach().cpu().numpy().mean(axis=0))
        centers_np = np.asarray(centers)
        axes[1].plot(steps, centers_np[:, 0], marker="o", label="mean frame center coordinate 0")
        axes[1].plot(steps, centers_np[:, 1], marker="s", label="mean frame center coordinate 1")
        axes[1].set_title("actual town frame parameter motion")
    axes[1].set_xlabel("optimization step")
    axes[1].set_ylabel("coordinate / parameter value")
    axes[1].legend(fontsize="small")
    return fig


def build_comparison_figure(
    group_path: Path | str,
    *,
    class_name: str,
    split: str,
    sample_position: int,
    snapshot: str = "final",
    anchor_mode: str = "zero",
):
    raise_classifier_backend_unavailable("historical synthetic classifier geometry comparison")
    group = Path(group_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    facts = {}
    for ax, family in zip(axes, MODEL_FAMILIES):
        result = load_geometry_view(group / family, split=split, sample_position=sample_position, snapshot=snapshot)
        if result.view is None:
            raise ValueError(result.error)
        single, info = build_region_figure(result.view, class_name=class_name, anchor_mode=anchor_mode)
        source_ax = single.axes[0]
        source_ax.figure.canvas.draw()
        # Copy a faithful rendered panel as a raster; each panel remains in its own coordinates.
        rgba = np.asarray(source_ax.figure.canvas.buffer_rgba())
        ax.imshow(rgba)
        ax.set_axis_off()
        ax.set_title(f"{family} · {info['selected_score_name']}={info['selected_score']:.3g} · contains={info['selected_membership']}")
        facts[family] = info
        plt.close(single)
    fig.suptitle(f"Independent latent coordinate systems · {class_name} · {split} · selected point #{sample_position}")
    return fig, facts
