"""Read-only dashboard views for persisted real-image embedding experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
import torch

from town_model.dashboard.data import find_sample, load_dashboard_data
from town_model.inspection.image_snapshots import ImageResearchSnapshot, load_image_snapshot
from town_model.models.geometry import BoxGeometry, evaluate_town_corners


def discover_image_runs(root: Path | str) -> list[Path]:
    """List valid Phase 3 run directories without importing checkpoints/optimizers."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        return []
    found: list[Path] = []
    for candidate in sorted(base.iterdir()):
        if not candidate.is_dir():
            continue
        try:
            manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
            config = json.loads((candidate / "resolved-config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("schema_version") == "image-run-manifest-v1" and manifest.get("state") in {
            "draft", "running", "completed", "stopped", "interrupted", "failed"
        } and config.get("identity_hash") == manifest.get("identity_hash"):
            found.append(candidate)
    return found


def discover_snapshots(run_directory: Path | str) -> list[Path]:
    root = Path(run_directory).expanduser().resolve() / "snapshots"
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob("epoch-*") if path.is_dir() and (path / "header.json").is_file() and (path / "arrays.npz").is_file())


def component_slice(snapshot: ImageResearchSnapshot, sample_index: int) -> slice:
    offsets = snapshot.arrays["component_offsets"]
    if sample_index < 0 or sample_index + 1 >= len(offsets):
        raise IndexError("snapshot sample index is outside the persisted reference panel")
    return slice(int(offsets[sample_index]), int(offsets[sample_index + 1]))


def point_rows(snapshot: ImageResearchSnapshot, sample_index: int, *, level: Literal["pizza", "component"], class_index: int) -> list[dict[str, Any]]:
    """Build inspection rows from persisted model outputs and canonical masks."""
    classes = len(snapshot.header["class_names"])
    if not 0 <= class_index < classes:
        raise IndexError("selected trainable class is outside the snapshot class vocabulary")
    arrays = snapshot.arrays
    if level == "pizza":
        row = {
            "sample_id": str(arrays["sample_ids"][sample_index]),
            "source_split": str(arrays["source_splits"][sample_index]),
            "global_point": arrays["pizza_points_global"][sample_index].tolist(),
            "composed_point": arrays["pizza_points_composed"][sample_index].tolist(),
            "final_point": arrays["pizza_points_final"][sample_index].tolist(),
            "contains": bool(arrays["pizza_contains"][sample_index, class_index]),
            "target": int(arrays["pizza_targets"][sample_index, class_index]),
            "known": bool(arrays["pizza_known_masks"][sample_index, class_index]),
        }
        if "pizza_signed_margin" in arrays:
            row["signed_margin"] = float(arrays["pizza_signed_margin"][sample_index, class_index])
        if "pizza_logits" in arrays:
            row["logit"] = float(arrays["pizza_logits"][sample_index, class_index])
        return [row]
    rows = []
    section = component_slice(snapshot, sample_index)
    for index in range(section.start, section.stop):
        row = {
            "sample_id": str(arrays["sample_ids"][sample_index]),
            "component_id": str(arrays["component_ids"][index]),
            "detector_category": str(arrays["component_categories"][index]),
            "bbox_xyxy_original": arrays["component_boxes_xyxy_original"][index].tolist(),
            "point": arrays["component_points"][index].tolist(),
            "contains": bool(arrays["component_contains"][index, class_index]),
            "target": int(arrays["component_targets"][index, class_index]),
            "known": bool(arrays["component_known_masks"][index, class_index]),
        }
        if "component_signed_margin" in arrays:
            row["signed_margin"] = float(arrays["component_signed_margin"][index, class_index])
        if "component_logits" in arrays:
            row["logit"] = float(arrays["component_logits"][index, class_index])
        rows.append(row)
    return rows


def _exact_region(snapshot: ImageResearchSnapshot, class_index: int, points: np.ndarray, *, axes: tuple[int, int], anchor: np.ndarray):
    geometry = snapshot.header["geometry"]
    family = geometry["family"]
    dimensions = int(snapshot.header["latent_dimensions"])
    if points.ndim != 2 or points.shape[1] != dimensions:
        raise ValueError("geometry slice points must have the snapshot latent dimension")
    coordinates = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    full = np.repeat(anchor.astype(np.float32)[None, :], xx.size, axis=0)
    full[:, axes[0]] = xx.reshape(-1)
    full[:, axes[1]] = yy.reshape(-1)
    if family == "baseline":
        weight = np.asarray(geometry["weight"][class_index], dtype=np.float32)
        bias = float(geometry["bias"][class_index])
        scores = full @ weight + bias
        contains = scores >= 0
        score_label = "actual classifier logit (zero boundary)"
    elif family == "box":
        lows = torch.tensor(geometry["box_low"][class_index : class_index + 1], dtype=torch.float32).unsqueeze(1)
        highs = torch.tensor(geometry["box_high"][class_index : class_index + 1], dtype=torch.float32).unsqueeze(1)
        evaluation = BoxGeometry.from_physical(lows, highs).evaluate(torch.from_numpy(full))
        contains = evaluation.contains[:, 0].numpy()
        scores = evaluation.signed_margin[:, 0].numpy()
        score_label = "exact box signed margin; contains uses closed bounds"
    else:
        frame_low = torch.tensor(geometry["frame_low"][class_index : class_index + 1], dtype=torch.float32)
        frame_high = torch.tensor(geometry["frame_high"][class_index : class_index + 1], dtype=torch.float32)
        window_low = torch.tensor(geometry["window_low"][class_index : class_index + 1], dtype=torch.float32)
        window_high = torch.tensor(geometry["window_high"][class_index : class_index + 1], dtype=torch.float32)
        evaluation = evaluate_town_corners(
            torch.from_numpy(full), frame_low, frame_high, window_low, window_high
        )
        contains = evaluation.contains[:, 0].numpy()
        scores = evaluation.signed_margin[:, 0].numpy()
        score_label = "exact Phase 2 town signed margin; frame minus windows"
    return xx, yy, contains.reshape(xx.shape), np.asarray(scores).reshape(xx.shape), score_label


def build_point_projection_figure(snapshot: ImageResearchSnapshot, *, sample_index: int, level: Literal["pizza", "component"], class_index: int):
    """Optional PCA scatter for display only; stored raw latent coordinates are unchanged."""
    arrays = snapshot.arrays
    if level == "pizza":
        points = np.asarray(arrays["pizza_points_final"], dtype=np.float32)
        known = arrays["pizza_known_masks"][:, class_index].astype(bool)
        targets = arrays["pizza_targets"][:, class_index].astype(bool)
        selected_index = sample_index
    else:
        points = np.asarray(arrays["component_points"], dtype=np.float32)
        known = arrays["component_known_masks"][:, class_index].astype(bool)
        targets = arrays["component_targets"][:, class_index].astype(bool)
        selected_index = component_slice(snapshot, sample_index).start
    if len(points) == 0:
        raise ValueError("no retained points are available for display projection")
    centered = points - points.mean(axis=0, keepdims=True)
    if centered.shape[1] == 1:
        projected = np.column_stack((centered[:, 0], np.zeros((len(centered),), dtype=np.float32)))
    else:
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        axes = right[:2].T
        if axes.shape[1] < 2:
            axes = np.column_stack((axes, np.zeros((axes.shape[0],), dtype=np.float32)))
        projected = centered @ axes[:, :2]
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    ax.scatter(projected[known & targets, 0], projected[known & targets, 1], c="#167d42", marker="o", label="known positive")
    ax.scatter(projected[known & ~targets, 0], projected[known & ~targets, 1], c="#bf3f36", marker="x", label="known negative")
    ax.scatter(projected[~known, 0], projected[~known, 1], c="#777777", marker="^", label="unknown")
    if 0 <= selected_index < len(projected):
        ax.scatter([projected[selected_index, 0]], [projected[selected_index, 1]], c="#ffbf00", marker="*", s=170, edgecolors="black", label="selected reference")
    ax.set_title(f"PCA visualization only · {level} points · {snapshot.header['class_names'][class_index]}")
    ax.set_xlabel("display-only principal component 1")
    ax.set_ylabel("display-only principal component 2")
    ax.legend(fontsize="small")
    return fig


def _slice_intersects(low, high, anchor: np.ndarray, axes: tuple[int, int]) -> bool:
    other = [index for index in range(len(anchor)) if index not in axes]
    return all(float(low[index]) <= float(anchor[index]) <= float(high[index]) for index in other)


def build_image_geometry_figure(
    snapshot: ImageResearchSnapshot,
    *,
    sample_index: int,
    level: Literal["pizza", "component"],
    class_index: int,
    axes: tuple[int, int] = (0, 1),
    anchor_mode: Literal["zero", "sample"] = "sample",
):
    """Plot an exact raw-coordinate slice, retaining model coordinates and masks."""
    arrays = snapshot.arrays
    dimensions = int(snapshot.header["latent_dimensions"])
    if dimensions < 2 or len(set(axes)) != 2 or min(axes) < 0 or max(axes) >= dimensions:
        raise ValueError("geometry axes must be distinct valid latent coordinates")
    if level == "pizza":
        selected_point = arrays["pizza_points_final"][sample_index]
        all_points = arrays["pizza_points_final"]
        known = arrays["pizza_known_masks"][:, class_index].astype(bool)
        targets = arrays["pizza_targets"][:, class_index].astype(bool)
    else:
        selected_slice = component_slice(snapshot, sample_index)
        selected_points = arrays["component_points"][selected_slice]
        if len(selected_points) == 0:
            raise ValueError("selected sample has no retained component traces")
        selected_point = selected_points[0]
        all_points = arrays["component_points"]
        known = arrays["component_known_masks"][:, class_index].astype(bool)
        targets = arrays["component_targets"][:, class_index].astype(bool)
    anchor = np.zeros((dimensions,), dtype=np.float32) if anchor_mode == "zero" else np.asarray(selected_point, dtype=np.float32).copy()
    xx, yy, contains, scores, score_label = _exact_region(snapshot, class_index, all_points, axes=axes, anchor=anchor)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    ax.pcolormesh(xx, yy, contains.astype(np.float32), shading="nearest", cmap="Greens", vmin=0, vmax=1, alpha=0.42)
    if snapshot.header["family"] == "baseline":
        if float(scores.min()) <= 0 <= float(scores.max()):
            ax.contour(xx, yy, scores, levels=[0.0], colors=["#162b75"], linewidths=2)
    else:
        geometry = snapshot.header["geometry"]
        if snapshot.header["family"] == "box":
            low = geometry["box_low"][class_index]
            high = geometry["box_high"][class_index]
            if _slice_intersects(low, high, anchor, axes):
                ax.add_patch(Rectangle((low[axes[0]], low[axes[1]]), high[axes[0]] - low[axes[0]], high[axes[1]] - low[axes[1]], fill=False, edgecolor="#126782", linewidth=2, label="actual closed box"))
        else:
            for house, (low, high) in enumerate(zip(geometry["frame_low"][class_index], geometry["frame_high"][class_index], strict=True)):
                if _slice_intersects(low, high, anchor, axes):
                    ax.add_patch(Rectangle((low[axes[0]], low[axes[1]]), high[axes[0]] - low[axes[0]], high[axes[1]] - low[axes[1]], fill=False, edgecolor="#126782", linestyle="--", label=f"house {house} frame"))
                for window, (wlow, whigh) in enumerate(zip(geometry["window_low"][class_index][house], geometry["window_high"][class_index][house], strict=True)):
                    if _slice_intersects(wlow, whigh, anchor, axes):
                        ax.add_patch(Rectangle((wlow[axes[0]], wlow[axes[1]]), whigh[axes[0]] - wlow[axes[0]], whigh[axes[1]] - wlow[axes[1]], fill=False, edgecolor="#8c2b67", linestyle=":", label=f"house {house} window {window} exclusion"))
    positive = known & targets
    negative = known & ~targets
    unknown = ~known
    ax.scatter(all_points[positive, axes[0]], all_points[positive, axes[1]], s=22, c="#167d42", marker="o", edgecolors="white", linewidths=0.3, label="known positive (projected)")
    ax.scatter(all_points[negative, axes[0]], all_points[negative, axes[1]], s=24, c="#bf3f36", marker="x", label="known negative (projected)")
    ax.scatter(all_points[unknown, axes[0]], all_points[unknown, axes[1]], s=24, c="#777777", marker="^", label="unknown (projected)")
    ax.scatter([selected_point[axes[0]]], [selected_point[axes[1]]], marker="*", s=170, c="#ffbf00", edgecolors="black", linewidths=0.7, label="selected sample", zorder=7)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel=f"latent coordinate {axes[0]}", ylabel=f"latent coordinate {axes[1]}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{snapshot.header['family']} · {snapshot.header['class_names'][class_index]} · exact {level} region slice\n{score_label}")
    ax.legend(fontsize="small", loc="best")
    return fig


def _draw_snapshot_image(image: Image.Image, record, selected_component: str | None, *, alpha: float) -> Image.Image:
    rendered = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", rendered.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    opacity = round(255 * alpha)
    for component in record.components:
        x1, y1, x2, y2 = component.bbox_xyxy_original
        rgb = (245, 40, 45) if component.component_id == selected_component else (20, 190, 80)
        color = (*rgb, opacity)
        draw.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)), outline=color, width=3 if component.component_id == selected_component else 2)
        draw.text((x1 + 2, max(0, y1 - 12)), component.component_id.rsplit("/", 1)[-1], fill=color, stroke_width=1, stroke_fill=(0, 0, 0, opacity))
    return Image.alpha_composite(rendered, overlay).convert("RGB")


def render_image_embeddings(st, runs_root: Path, dataset_path: Path) -> None:
    st.subheader("Image Embeddings · pre-detector GT-ROI exploratory work")
    st.warning("Every run on this page uses ground-truth component ROIs. These are pre-detector exploratory embeddings, not detector-pretrained Phase 3 acceptance evidence and not Phase 4.")
    runs = discover_image_runs(runs_root)
    if not runs:
        st.info(f"No Phase 3 image runs with validated run manifests were found under `{runs_root}`.")
        return
    selected_run = st.selectbox("Image run", runs, format_func=lambda path: path.name, key="image_embedding_run")
    try:
        manifest = json.loads((selected_run / "manifest.json").read_text(encoding="utf-8"))
        config = json.loads((selected_run / "resolved-config.json").read_text(encoding="utf-8"))
        metrics = json.loads((selected_run / "metrics.json").read_text(encoding="utf-8")) if (selected_run / "metrics.json").is_file() else {}
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read this run's JSON provenance: {error}")
        return
    st.caption(f"State: `{manifest['state']}` · run `{manifest['run_id']}` · identity `{manifest['identity_hash']}`")
    st.json({"resolved_configuration": config, "run_manifest": manifest, "metrics": metrics})
    snapshot_paths = discover_snapshots(selected_run)
    if not snapshot_paths:
        st.info("This run has no completed, persisted numerical snapshots yet. Running checkpoints are not loaded by the dashboard.")
        return
    selected_snapshot_path = st.selectbox("Historical epoch snapshot", snapshot_paths, format_func=lambda path: path.name, index=len(snapshot_paths) - 1, key="image_snapshot")
    try:
        snapshot = load_image_snapshot(selected_snapshot_path)
    except (OSError, ValueError) as error:
        st.error(f"Snapshot failed its read-only integrity checks: {error}")
        return
    header, arrays = snapshot.header, snapshot.arrays
    st.info(f"Epoch {header['epoch']} · optimizer step {header['optimizer_step']} · source checkpoint SHA-256 `{header['source_checkpoint_sha256']}` · components traced {header['component_count_traced']}/{header['component_count_original']} under cap {header['component_cap_per_sample']}")
    st.caption("The dashboard reads only checksummed JSON/NPZ research snapshots. It does not load a model checkpoint, optimizer state, or run training.")
    st.caption(f"Dataset `{header['dataset_id']}` · ROI source `{header['roi_source']}` · family `{header['family']}` · pizza mode `{header['pizza_mode']}` · latent d={header['latent_dimensions']}")

    loaded = load_dashboard_data(dataset_path)
    if loaded.explorer is None or loaded.explorer.summary.integrity_status != "current_pass":
        st.error(f"Dataset image overlay is disabled: {loaded.error or 'current dataset integrity has not passed'}")
        return
    explorer = loaded.explorer
    manifest_path = dataset_path / "manifest.json"
    try:
        current_manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except OSError as error:
        st.error(f"Could not verify the selected dataset manifest: {error}")
        return
    if explorer.summary.dataset_id != header["dataset_id"] or current_manifest_hash != header["dataset_manifest_sha256"]:
        st.error("Selected dataset path does not match the exact validated dataset manifest used by this research snapshot.")
        return
    position = st.selectbox(
        "Reference sample",
        list(range(len(arrays["sample_ids"]))),
        format_func=lambda index: f"{arrays['sample_ids'][index]} · {arrays['source_splits'][index]}",
        key="image_snapshot_sample",
    )
    sample_id = str(arrays["sample_ids"][position])
    record = find_sample(explorer, sample_id)
    if record is None:
        st.error(f"Reference image `{sample_id}` is absent from the validated dataset.")
        return
    start, stop = component_slice(snapshot, position).start, component_slice(snapshot, position).stop
    component_ids = arrays["component_ids"][start:stop].tolist()
    if component_ids:
        component_selection = st.selectbox("GT component occurrence", ["(whole image)", *component_ids], key="image_snapshot_component")
        selected_component = None if component_selection == "(whole image)" else component_selection
    else:
        selected_component = None
        st.caption("No component occurrence was retained for this sample under the configured trace cap.")
    from town_model.application import DatasetSnapshotChangedError, load_verified_sample_image

    try:
        image = load_verified_sample_image(explorer, record)
        overlay_alpha = float(config.get("training_config", {}).get("visualization", {}).get("image_overlay_alpha", 0.6))
        st.image(_draw_snapshot_image(image, record, selected_component, alpha=overlay_alpha), caption=f"{sample_id} · {record.split} · GT boxes only", width="stretch")
    except (OSError, DatasetSnapshotChangedError) as error:
        st.error(f"Could not load a consistent verified source image: {error}")
        return

    classes = list(zip(header["class_ids"], header["class_names"], strict=True))
    selected_class_index = st.selectbox(
        "Trainable ontology class",
        list(range(len(classes))),
        format_func=lambda index: f"{classes[index][1]} · class_id={classes[index][0]}",
        key="image_embedding_class",
    )
    level = st.radio("Embedding/classifier level", ("pizza", "component"), horizontal=True, key="image_embedding_level")
    rows = point_rows(snapshot, position, level=level, class_index=selected_class_index)
    projection_default = config.get("training_config", {}).get("visualization", {}).get("point_projection", "none")
    projection = st.selectbox("Point display", ("raw latent / exact region", "PCA (display only)"), index=1 if projection_default == "pca" else 0, key="image_point_projection")
    if projection == "PCA (display only)":
        projection_figure = build_point_projection_figure(snapshot, sample_index=position, level=level, class_index=selected_class_index)
        st.pyplot(projection_figure, clear_figure=True)
        plt.close(projection_figure)
        st.caption("PCA is fitted only for this display. Classifier outputs, known masks, saved points, and exact geometric slices remain in the original model coordinate system.")
    st.markdown("#### Stored image-derived points and actual classifier outputs")
    if level == "pizza":
        st.dataframe([{
            **row,
            "point (raw latent)": row["final_point"],
            "known-mask state": "unknown" if not row["known"] else "positive" if row["target"] else "known negative",
            "score is": "logit" if "logit" in row else "signed geometric margin",
        } for row in rows], hide_index=True, width="stretch")
        st.caption("Global and DeepSets-composed points are retained as intermediates. For hybrid mode only the fused final point receives the configured pizza-level classifier/objective; no auxiliary intermediate pizza losses are implied.")
    elif not rows:
        st.info("No component traces are available for this sample.")
    else:
        display_rows = []
        for row in rows:
            display_rows.append({
                **row,
                "known-mask state": "unknown" if not row["known"] else "positive" if row["target"] else "known negative",
                "score is": "logit" if "logit" in row else "signed geometric margin",
            })
        st.dataframe(display_rows, hide_index=True, width="stretch")
    if rows:
        if level == "component" and selected_component is not None and "roi_feature_means" in arrays:
            selected_global_index = component_ids.index(selected_component) + start
            with st.expander("Selected GT-ROI pooled feature and coordinate trace"):
                st.json({
                    "component_id": selected_component,
                    "pooled_roi_feature_mean_256d": arrays["roi_feature_means"][selected_global_index].tolist(),
                    "normalized_original_xyxy_coordinates": arrays["normalized_roi_coordinates"][selected_global_index].tolist(),
                    "feature_cache_id": header.get("feature_cache_id"),
                })
                if "global_spatial_features" in arrays:
                    global_feature = arrays["global_spatial_features"][position]
                    st.caption(f"Global ROI spatial feature tensor: shape {global_feature.shape} · mean={float(global_feature.mean()):.5g} · std={float(global_feature.std()):.5g}")
        try:
            geometry = header["geometry"]
            if geometry["family"] == "town":
                geometry_info = {
                    "family": "town",
                    "selected_class_frame_low": geometry["frame_low"][selected_class_index],
                    "selected_class_frame_high": geometry["frame_high"][selected_class_index],
                    "selected_class_window_low": geometry["window_low"][selected_class_index],
                    "selected_class_window_high": geometry["window_high"][selected_class_index],
                    "membership": geometry["boundary"],
                }
            elif geometry["family"] == "box":
                geometry_info = {"family": "box", "low": geometry["box_low"][selected_class_index], "high": geometry["box_high"][selected_class_index], "membership": geometry["boundary"]}
            else:
                geometry_info = {"family": "baseline", "weight": geometry["weight"][selected_class_index], "bias": geometry["bias"][selected_class_index], "boundary": geometry["boundary"]}
            st.markdown("#### Actual saved geometry / boundary provenance")
            st.json(geometry_info)
            if int(header["latent_dimensions"]) >= 2:
                dimension = int(header["latent_dimensions"])
                x_column, y_column = st.columns(2)
                axis_x = x_column.selectbox("Raw latent x axis", list(range(dimension)), index=0, key="image_geometry_axis_x")
                axis_y_candidates = [item for item in range(dimension) if item != axis_x]
                axis_y = y_column.selectbox("Raw latent y axis", axis_y_candidates, index=0, key="image_geometry_axis_y")
                anchor_mode = st.radio("Other-coordinate anchor", ("sample", "zero"), horizontal=True, key="image_geometry_anchor")
                figure = build_image_geometry_figure(snapshot, sample_index=position, level=level, class_index=selected_class_index, axes=(axis_x, axis_y), anchor_mode=anchor_mode)
                st.pyplot(figure, clear_figure=True)
                plt.close(figure)
                st.caption("The region is evaluated exactly in the raw latent coordinate system. In d>2 it is a slice with other coordinates fixed, not a projection of the region; point markers are coordinate projections.")
        except (ValueError, IndexError, RuntimeError, KeyError) as error:
            st.warning(f"Exact geometry view is unavailable for this selection: {error}")

    st.markdown("#### Training and held-out metric trace")
    history = metrics.get("history", [])
    validation_rows = [row["validation"] for row in history if isinstance(row, dict) and isinstance(row.get("validation"), dict)]
    if validation_rows:
        st.dataframe([{
            "split": row["split"],
            "primary monitor": row["primary_monitor"],
            "pizza macro F1": row["pizza"]["macro_f1"],
            "pizza known positive / negative / unknown": f"{row['pizza']['known_positive']} / {row['pizza']['known_negative']} / {row['pizza']['unknown']}",
            "component GT-ROI macro F1": row["component_gt_roi"]["macro_f1"],
            "component known positive / negative / unknown": f"{row['component_gt_roi']['known_positive']} / {row['component_gt_roi']['known_negative']} / {row['component_gt_roi']['unknown']}",
        } for row in validation_rows], hide_index=True, width="stretch")
    if history:
        train_rows = [row for row in history if isinstance(row, dict) and "training_objective" in row]
        if train_rows:
            figure, axis = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
            axis.plot([row["optimizer_step"] for row in train_rows], [row["training_objective"] for row in train_rows], marker="o", label="effective-batch training objective")
            axis.set_xlabel("optimizer step")
            axis.set_ylabel("configured objective (not cross-family comparable)")
            axis.legend()
            st.pyplot(figure, clear_figure=True)
            plt.close(figure)
    st.caption("Validation uses only the original validation partition. Test rows are never used by training, early stopping, or this saved run metric trace.")
