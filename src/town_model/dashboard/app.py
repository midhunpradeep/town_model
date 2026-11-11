"""Permanent local Phase 1 dashboard built on shared application operations."""

from __future__ import annotations

import json
import queue
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageDraw

from town_model.classifier_backend import ClassifierBackendUnavailableError
from town_model.application import (
    DatasetSnapshotChangedError,
    load_verified_sample_image,
    validate_dataset_operation,
    validate_generation_request,
)
from town_model.config import DEFAULT_PIZZA_TYPES, DatasetConfig, load_config
from town_model.dataset.constants import SCHEMA_VERSION
from town_model.dashboard.data import class_names_for_targets, find_sample, load_dashboard_data
from town_model.dashboard.runtime import submit_generation
from town_model.models.bounded_box import bounded_box_contains, box_signed_clearance
from town_model.models.bounded_box_fixture import BOUNDED_BOX_INSPECTION_FIXTURE
from town_model.models.bounded_multi_box import (
    bounded_multi_box_contains,
    multi_box_signed_clearance,
)
from town_model.models.bounded_multi_box_fixture import BOUNDED_MULTI_BOX_INSPECTION_FIXTURE
from town_model.models.bound_box_inference import BoundBoxClassifier, BoundBoxGeometry
from town_model.models.bound_box_inference_fixture import BOUND_BOX_INFERENCE_INSPECTION_FIXTURE

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_CONFIG = REPOSITORY_ROOT / "dataset.toml"
DEFAULT_EXAMPLE_DATASET = REPOSITORY_ROOT / "examples" / "pizza-48"


def _class_labels(vector: tuple[int, ...], classes: tuple[dict[str, Any], ...]) -> list[str]:
    return class_names_for_targets(vector, classes)


def target_state_rows(targets: tuple[int, ...], known_mask: tuple[int, ...], classes: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Render authoritative target/mask states without deriving ontology labels."""
    if len(targets) != len(classes) or len(known_mask) != len(classes):
        raise ValueError("target vectors must match the canonical ontology manifest width")
    rows = []
    for class_info in classes:
        class_id = int(class_info["class_id"])
        known = bool(known_mask[class_id])
        target = bool(targets[class_id])
        rows.append({
            "class_id": class_id,
            "class": class_info["name"],
            "status": "fixed positive" if class_info.get("fixed_positive") else "unsupported" if class_info.get("unsupported") else "trainable",
            "target_state": "positive" if known and target else "known negative" if known else "unknown",
        })
    return rows


def _load_config_defaults() -> DatasetConfig:
    return load_config(DEFAULT_DATASET_CONFIG)


def _draw_annotations(image: Image.Image, components: tuple, selected_id: str | None) -> Image.Image:
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    for component in components:
        x1, y1, x2, y2 = component.bbox_xyxy_original
        selected = component.component_id == selected_id
        color = (230, 35, 45) if selected else (20, 200, 80)
        draw.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)), outline=color, width=3 if selected else 2)
        draw.text((x1 + 2, max(0, y1 - 12)), f"{component.detector_category} · {component.component_id.rsplit('/', 1)[-1]}", fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
    return rendered


def _render_overview(st, dataset_path: Path) -> None:
    if not dataset_path.is_dir():
        st.info("Select a prepared dataset path in the sidebar, or generate one from the **Generate dataset** page. Opening this dashboard never generates data.")
        return
    loaded = load_dashboard_data(dataset_path)
    if loaded.explorer is None:
        st.error(loaded.error)
        return
    explorer = loaded.explorer
    summary = explorer.summary
    st.subheader("Dataset overview")
    verified = summary.integrity_status == "current_pass"
    columns = st.columns(4)
    columns[0].metric("Samples", summary.sample_count if verified else "unverified")
    columns[1].metric("Train", summary.split_sizes.get("train", 0) if verified else "unverified")
    columns[2].metric("Validation", summary.split_sizes.get("validation", 0) if verified else "unverified")
    columns[3].metric("Test", summary.split_sizes.get("test", 0) if verified else "unverified")
    st.caption(f"Dataset `{summary.dataset_id}` · schema `{summary.schema_version}` · ontology SHA-256 `{summary.ontology.get('sha256', 'unknown')}`")

    if summary.integrity_status == "current_pass":
        validation = summary.validation or {}
        st.success(
            f"Current integrity validation: PASS · "
            f"{validation.get('error_count', 0)} errors · {validation.get('warning_count', 0)} warnings"
        )
    elif summary.integrity_status == "current_fail":
        st.error("Current integrity validation: FAIL. Historical manifest statistics are withheld.")
        for issue in (summary.validation or {}).get("errors", [])[:30]:
            st.error(f"{issue['code']}: {issue['message']}" + (f" · {issue['path']}" if issue.get("path") else ""))
    else:
        st.error(f"Current integrity validation is UNAVAILABLE: {summary.validation_error or 'no report was produced'}")
    if summary.saved_validation is not None:
        saved_status = "PASS" if summary.saved_validation.get("valid") else "FAIL"
        st.caption(
            f"Saved validation report (historical only; not evidence about current files): {saved_status}"
        )
    if st.button("Run dataset integrity validation", key="overview_validate"):
        with st.spinner("Checking ontology, assets, annotations, targets, and splits…"):
            report = validate_dataset_operation(dataset_path)
        st.session_state["validation_dataset_path"] = str(dataset_path)
        if report.valid:
            st.success(f"Validation passed with {len(report.warnings)} class-support warning(s).")
        else:
            st.error(f"Validation found {len(report.errors)} error(s).")
        for issue in report.errors:
            st.error(f"{issue.code}: {issue.message}" + (f" · {issue.path}" if issue.path else ""))
        for issue in report.warnings[:40]:
            st.warning(issue.message)
        st.rerun()

    if verified:
        st.markdown("#### Generator-type distribution")
        st.bar_chart(pd.Series(summary.generator_type_counts, name="samples").sort_index())
        st.markdown("#### Per-class supervision support")
        split = st.selectbox("Split", ("train", "validation", "test"), key="support_split")
        level = st.radio("Supervision level", ("pizza", "component"), horizontal=True, key="support_level")
        rows = summary.class_support_by_split.get(split, {}).get("classes", [])
        support_rows = [
            {
                "class_id": item["class_id"],
                "class": item["name"],
                "status": "fixed positive" if item["fixed_positive"] else "trainable" if item["trainable"] else "unsupported",
                "positive": item[level]["positive"],
                "known negative": item[level]["known_negative"],
                "unknown": item[level]["unknown"],
            }
            for item in rows
        ]
        if support_rows:
            frame = pd.DataFrame(support_rows)
            st.dataframe(frame, hide_index=True, width="stretch")
            no_positive = frame[(frame["status"] == "trainable") & (frame["positive"] == 0)]
            if not no_positive.empty:
                st.warning("Trainable classes with no positive support at this level/split: " + ", ".join(no_positive["class"].tolist()))
    else:
        st.warning("Support counts and sample totals are withheld until current artifacts pass validation.")

    with st.expander("Provenance and resolved configuration"):
        st.json({"source": summary.provenance, "ontology": summary.ontology, "vocabularies": summary.vocabularies})
        manifest = dict(explorer.manifest)
        if not verified:
            manifest.pop("statistics", None)
            st.caption("Manifest statistics are historical and withheld because current integrity did not pass.")
        st.json(manifest)
    if summary.validation and summary.integrity_status == "current_pass":
        for issue in summary.validation.get("warnings", [])[:40]:
            st.warning(issue["message"])


def _render_explorer(st, dataset_path: Path) -> None:
    if not dataset_path.is_dir():
        st.info("Select a prepared dataset path in the sidebar or explicitly generate one first.")
        return
    loaded = load_dashboard_data(dataset_path)
    if loaded.explorer is None:
        st.error(loaded.error)
        return
    explorer = loaded.explorer
    st.subheader("Image and annotation explorer")
    if explorer.summary.integrity_status != "current_pass":
        st.error("Explorer disabled: current dataset integrity has not passed validation.")
        if explorer.summary.integrity_status == "current_fail":
            for issue in (explorer.summary.validation or {}).get("errors", [])[:30]:
                st.error(f"{issue['code']}: {issue['message']}" + (f" · {issue['path']}" if issue.get("path") else ""))
        elif explorer.summary.validation_error:
            st.error(explorer.summary.validation_error)
        return
    include_test = st.checkbox("Include test split for explicit final inspection", value=False)
    splits = ("train", "validation", "test") if include_test else ("train", "validation")
    split = st.selectbox("Split", splits)
    selected_records = [record for record in explorer.records if record.split == split]
    if not selected_records:
        st.warning(f"No samples in {split} split.")
        return
    selected_id = st.selectbox(
        "Pizza image",
        [record.sample_id for record in selected_records],
        format_func=lambda sample_id: next(
            f"{sample_id} · {next(item.generator_type for item in selected_records if item.sample_id == sample_id)}"
            for item in selected_records if item.sample_id == sample_id
        ),
    )
    record = find_sample(explorer, selected_id)
    if record is None:
        st.error(f"Sample {selected_id} was not found in the loaded dataset")
        return

    component_labels = [
        f"{component.detector_category} · {component.component_id.rsplit('/', 1)[-1]} ({component.component_id})"
        for component in record.components
    ]
    selected_component_label = st.selectbox("Select a component occurrence", ("None", *component_labels))
    selected_component = None
    if selected_component_label != "None":
        selected_index = component_labels.index(selected_component_label)
        selected_component = record.components[selected_index]
    show_boxes = st.checkbox("Show ground-truth bounding boxes", value=True)

    try:
        image = load_verified_sample_image(explorer, record)
        if show_boxes:
            image = _draw_annotations(image, record.components, selected_component.component_id if selected_component else None)
        st.image(image, caption=f"{record.sample_id} · split={record.split} · type={record.generator_type}", width="stretch")
    except (OSError, DatasetSnapshotChangedError) as error:
        st.error(f"Could not open a consistent validated image for sample {record.sample_id}: {error}")
        return

    st.markdown("#### Whole-pizza ontology labels")
    st.caption("The source-inferred labels below are retained as audit provenance; ontology targets include corrected class ancestry.")
    st.write("Source-inferred labels:", ", ".join(record.pizza_source_labels))
    st.write("Corrected ontology targets:", ", ".join(_class_labels(record.pizza_targets, explorer.classes)))
    with st.expander("Canonical pizza target states and known masks"):
        st.dataframe(pd.DataFrame(target_state_rows(record.pizza_targets, record.pizza_known_mask, explorer.classes)), hide_index=True, width="stretch")

    st.markdown("#### Ground-truth component occurrences")
    component_table = [
        {
            "component_id": component.component_id,
            "detector_category": component.detector_category,
            "bbox_xyxy_original": component.bbox_xyxy_original,
            "corrected_ontology_targets": ", ".join(_class_labels(component.ontology_targets, explorer.classes)),
        }
        for component in record.components
    ]
    st.dataframe(pd.DataFrame(component_table), hide_index=True, width="stretch")
    if selected_component:
        st.markdown(f"#### Selected occurrence: `{selected_component.component_id}`")
        st.write("Detector identity (separate from ontology class IDs):", selected_component.detector_category)
        st.write("Original-image xyxy bounding box:", selected_component.bbox_xyxy_original)
        st.write("Corrected ontology targets:", ", ".join(_class_labels(selected_component.ontology_targets, explorer.classes)))
        with st.expander("Canonical component target states and known masks"):
            st.dataframe(pd.DataFrame(target_state_rows(selected_component.ontology_targets, selected_component.ontology_known_mask, explorer.classes)), hide_index=True, width="stretch")

    with st.expander("Raw generator metadata — audit only, never predictive input"):
        st.json({"audit": record.audit, "source_annotations": record.source_annotations})


def _generation_page(st) -> None:
    st.subheader("Explicit dataset generation")
    st.caption("Generation runs in a background worker. It never starts when the dashboard opens, and output paths are immutable/non-overwriting.")
    try:
        defaults = _load_config_defaults()
    except (OSError, ValueError) as error:
        st.error(f"Cannot load dataset.toml defaults: {error}")
        defaults = DatasetConfig(total_instances=48)

    with st.form("generation-settings"):
        total = st.number_input("Total images", min_value=3, max_value=100000, value=int(defaults.total_instances or 48), step=1)
        types = st.multiselect("Generator types", DEFAULT_PIZZA_TYPES, default=list(defaults.pizza_types))
        generation_seed = st.number_input("Generation seed", min_value=0, max_value=2**31 - 1, value=defaults.generation_seed, step=1)
        split_seed = st.number_input("Split seed", min_value=0, max_value=2**32 - 1, value=defaults.split_seed, step=1)
        ratio_columns = st.columns(3)
        train_ratio = ratio_columns[0].number_input("Train ratio", min_value=0.01, max_value=0.98, value=float(defaults.train_size), step=0.01, format="%.2f")
        valid_ratio = ratio_columns[1].number_input("Validation ratio", min_value=0.01, max_value=0.98, value=float(defaults.valid_size), step=0.01, format="%.2f")
        test_ratio = ratio_columns[2].number_input("Test ratio", min_value=0.01, max_value=0.98, value=float(defaults.test_size), step=0.01, format="%.2f")
        counts_text = st.text_input("Optional exact type-count JSON (blank means balanced)", value="")
        output_text = st.text_input("New output directory", value=str(REPOSITORY_ROOT / "artifacts" / "datasets" / "pizza-local"))
        validate_settings = st.form_submit_button("Validate settings")

    config: DatasetConfig | None = None
    error: str | None = None
    if validate_settings or st.session_state.get("generation_config_valid"):
        try:
            type_counts = None
            if counts_text.strip():
                parsed = json.loads(counts_text)
                if not isinstance(parsed, dict) or any(type(value) is not int for value in parsed.values()):
                    raise ValueError("Type-count JSON must map generator names to integer counts")
                type_counts = parsed
            config = DatasetConfig(
                total_instances=int(total),
                pizza_types=tuple(types),
                type_counts=type_counts,
                generation_seed=int(generation_seed),
                split_seed=int(split_seed),
                train_size=float(train_ratio),
                valid_size=float(valid_ratio),
                test_size=float(test_ratio),
            )
            request = validate_generation_request(config, Path(output_text).expanduser())
            output_path = Path(request["output_path"])
            counts = request["resolved_type_counts"]
            if validate_settings:
                st.session_state["generation_config_valid"] = True
                st.session_state["generation_resolved_config"] = request["resolved_config"]
                st.session_state["generation_validated_output"] = str(output_path)
                st.success("Settings are valid.")
                st.json(request)
        except (ValueError, OSError, json.JSONDecodeError) as exception:
            error = str(exception)
            st.session_state["generation_config_valid"] = False
            if validate_settings:
                st.error(error)

    future = st.session_state.get("generation_future")
    if config is not None and st.session_state.get("generation_config_valid"):
        same_config = config.resolved_dict() == st.session_state.get("generation_resolved_config")
        same_output = str(Path(output_text).expanduser().resolve()) == st.session_state.get("generation_validated_output")
        if not same_config or not same_output:
            st.info("Settings changed since validation. Validate the current settings before generation.")
        else:
            if st.button("Generate this dataset", type="primary", disabled=future is not None and not future.done()):
                progress_queue: queue.Queue[tuple[float, str]] = queue.Queue()
                try:
                    st.session_state["generation_future"] = submit_generation(config, Path(output_text).expanduser().resolve(), progress_queue)
                    st.session_state["generation_queue"] = progress_queue
                    st.session_state["generation_progress"] = (0.0, "Queued")
                    st.session_state["generation_error"] = None
                    st.session_state["generation_output"] = None
                    st.rerun()
                except (OSError, RuntimeError, ValueError) as exception:
                    st.error(f"Could not start generation: {exception}")

    _render_generation_monitor(st)
    if error:
        st.caption("Correct the settings and validate again before generating.")


def _render_generation_monitor(st) -> None:
    future = st.session_state.get("generation_future")
    if future is None:
        return

    def render_status() -> None:
        progress_queue = st.session_state.get("generation_queue")
        if progress_queue:
            while True:
                try:
                    st.session_state["generation_progress"] = progress_queue.get_nowait()
                except queue.Empty:
                    break
        fraction, message = st.session_state.get("generation_progress", (0.0, "Starting"))
        st.progress(float(fraction), text=message)
        if future.done():
            try:
                output = future.result()
                st.session_state["generation_output"] = str(output)
                st.session_state["generation_future"] = None
                st.success(f"Dataset generation and validation completed: {output}")
            except Exception as error:
                st.session_state["generation_error"] = str(error)
                st.session_state["generation_future"] = None
                st.error(f"Dataset generation failed: {error}")
            return
        st.info("Generation is running in the background. This page refreshes progress automatically.")

    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        fragment(run_every=1.0)(render_status)()
    else:
        render_status()

    completed = st.session_state.get("generation_output")
    if completed:
        st.code(completed)
        st.caption("Use this path in Dataset overview and Image explorer.")
    failed = st.session_state.get("generation_error")
    if failed:
        st.error(failed)


def _bounded_box_inspection_table() -> tuple[
    pd.DataFrame, int, tuple[tuple[str, float, float], ...]
]:
    fixture = BOUNDED_BOX_INSPECTION_FIXTURE
    points = torch.tensor(fixture.points, dtype=torch.float32, device="cpu")
    lower = torch.tensor(fixture.lower, dtype=torch.float32, device="cpu")
    upper = torch.tensor(fixture.upper, dtype=torch.float32, device="cpu")
    actual_membership = bounded_box_contains(points, lower, upper).tolist()
    actual_clearances = box_signed_clearance(points, lower, upper).tolist()
    rows = [
        {
            "Point": label,
            "Coordinates": coordinates,
            "Expected membership": expected_membership,
            "Actual membership": bool(actual_membership_value),
            "Agreement": expected_membership == bool(actual_membership_value),
            "Signed clearance": float(actual_clearance_value),
        }
        for (
            label,
            coordinates,
            expected_membership,
            actual_membership_value,
            actual_clearance_value,
        ) in zip(
            fixture.labels,
            fixture.points,
            fixture.expected_contains,
            actual_membership,
            actual_clearances,
            strict=True,
        )
    ]
    frame = pd.DataFrame(rows)
    membership_mismatch_count = sum(not row["Agreement"] for row in rows)
    clearance_mismatches = tuple(
        (label, float(expected), float(actual))
        for label, expected, actual in zip(
            fixture.labels, fixture.expected_clearances, actual_clearances, strict=True
        )
        if float(actual) != expected
    )
    return frame, membership_mismatch_count, clearance_mismatches


def _render_mathematical_inspector(st) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fixture = BOUNDED_BOX_INSPECTION_FIXTURE
    frame, membership_mismatch_count, clearance_mismatches = _bounded_box_inspection_table()

    figure, axis = plt.subplots(figsize=(6, 6))
    domain_patch = Rectangle(
        (-1.0, -1.0), 2.0, 2.0, fill=False, linestyle="--", edgecolor="black", linewidth=1.5
    )
    box_patch = Rectangle(
        fixture.lower,
        fixture.upper[0] - fixture.lower[0],
        fixture.upper[1] - fixture.lower[1],
        fill=False,
        edgecolor="tab:orange",
        linewidth=2,
    )
    axis.add_patch(domain_patch)
    axis.add_patch(box_patch)
    axis.scatter(
        [point[0] for point in fixture.points],
        [point[1] for point in fixture.points],
        color="tab:blue",
        zorder=3,
    )
    label_offsets = {
        "A": (5, 5), "B": (5, 5), "C": (5, -9),
        "D": (5, 5), "E": (5, 5), "F": (5, 5),
    }
    for label, (x, y) in zip(fixture.labels, fixture.points, strict=True):
        axis.annotate(label, (x, y), xytext=label_offsets[label], textcoords="offset points")
    axis.set_xlim(-1.4, 1.15)
    axis.set_ylim(-1.15, 1.15)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x₁")
    axis.set_ylabel("x₂")
    axis.legend(handles=[domain_patch, box_patch], labels=["Bounded domain Δ", "Physical box"])
    st.subheader("Mathematical Inspector")
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
    st.dataframe(frame, hide_index=True, width="stretch")
    if membership_mismatch_count:
        st.error(
            f"Membership mismatches: {membership_mismatch_count} of {len(fixture.labels)}"
        )
    else:
        st.success(
            f"Membership fixture agreement: {len(fixture.labels)}/{len(fixture.labels)}"
            " · membership mismatches: 0"
        )
    if clearance_mismatches:
        details = "; ".join(
            f"{label}: expected {expected:+.9g}, actual {actual:+.9g}"
            for label, expected, actual in clearance_mismatches
        )
        st.error(
            f"Signed-clearance mismatches: {len(clearance_mismatches)} of "
            f"{len(fixture.labels)} ({details})"
        )
    else:
        st.success(
            f"Signed-clearance fixture agreement: {len(fixture.labels)}/{len(fixture.labels)}"
            " · clearance mismatches: 0"
        )
    _render_multi_box_inspection(st)
    _render_bound_box_inference_inspection(st)


def _bounded_multi_box_inspection_matrices() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[tuple[str, int, bool, bool], ...],
    tuple[tuple[str, int, float, float], ...],
]:
    fixture = BOUNDED_MULTI_BOX_INSPECTION_FIXTURE
    points = torch.tensor(fixture.points, dtype=torch.float32, device="cpu")
    lower = torch.tensor(fixture.lower, dtype=torch.float32, device="cpu")
    upper = torch.tensor(fixture.upper, dtype=torch.float32, device="cpu")
    actual_membership = bounded_multi_box_contains(points, lower, upper).tolist()
    actual_clearances = multi_box_signed_clearance(points, lower, upper).tolist()

    membership_matrix = pd.DataFrame(
        actual_membership, index=fixture.labels, columns=fixture.box_labels
    )
    clearance_matrix = pd.DataFrame(
        actual_clearances, index=fixture.labels, columns=fixture.box_labels
    )
    membership_mismatches = tuple(
        (fixture.labels[row], box_index, expected, bool(actual_membership[row][box_index]))
        for row, expected_row in enumerate(fixture.expected_membership)
        for box_index, expected in enumerate(expected_row)
        if bool(actual_membership[row][box_index]) != expected
    )
    clearance_mismatches = tuple(
        (
            fixture.labels[row],
            box_index,
            expected,
            float(actual_clearances[row][box_index]),
        )
        for row, expected_row in enumerate(fixture.expected_clearances)
        for box_index, expected in enumerate(expected_row)
        if float(actual_clearances[row][box_index]) != expected
    )
    return membership_matrix, clearance_matrix, membership_mismatches, clearance_mismatches


def _render_multi_box_inspection(st) -> None:
    fixture = BOUNDED_MULTI_BOX_INSPECTION_FIXTURE
    (
        membership_matrix,
        clearance_matrix,
        membership_mismatches,
        clearance_mismatches,
    ) = _bounded_multi_box_inspection_matrices()
    total_cells = len(fixture.labels) * len(fixture.box_labels)

    st.markdown("#### Multi-box inference")
    st.markdown("Membership matrix")
    st.dataframe(membership_matrix, width="stretch")
    st.markdown("Signed-clearance matrix")
    st.dataframe(clearance_matrix, width="stretch")

    if membership_mismatches:
        details = "; ".join(
            f"point {point}, box index {box_index}: expected {expected}, actual {actual}"
            for point, box_index, expected, actual in membership_mismatches
        )
        st.error(
            f"Multi-box membership mismatches: {len(membership_mismatches)} of "
            f"{total_cells} ({details})"
        )
    else:
        st.success(f"Multi-box membership mismatches: 0 of {total_cells}")

    if clearance_mismatches:
        details = "; ".join(
            f"point {point}, box index {box_index}: expected {expected:+.9g}, actual {actual:+.9g}"
            for point, box_index, expected, actual in clearance_mismatches
        )
        st.error(
            f"Multi-box clearance mismatches: {len(clearance_mismatches)} of "
            f"{total_cells} ({details})"
        )
    else:
        st.success(f"Multi-box clearance mismatches: 0 of {total_cells}")


def _bound_box_inference_inspection() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[bool, ...],
    tuple[tuple[str, int, bool, bool], ...],
    tuple[tuple[str, int, float, float], ...],
    tuple[tuple[str, bool, bool], ...],
]:
    identity = BOUND_BOX_INFERENCE_INSPECTION_FIXTURE
    fixture = BOUNDED_MULTI_BOX_INSPECTION_FIXTURE
    points = torch.tensor(fixture.points, dtype=torch.float32, device="cpu")
    lower = torch.tensor(fixture.lower, dtype=torch.float32, device="cpu")
    upper = torch.tensor(fixture.upper, dtype=torch.float32, device="cpu")
    geometry = BoundBoxGeometry.from_ordered_rows(
        lower,
        upper,
        row_class_ids=identity.row_class_ids,
        class_manifest_sha256=identity.binding.class_manifest_sha256,
        expected_binding=identity.binding,
    )
    result = BoundBoxClassifier(identity.binding).infer(points, geometry)
    actual_membership = result.contains.tolist()
    actual_clearances = result.signed_coordinate_clearance.tolist()
    actual_domain = tuple(bool(value) for value in result.domain_contains.tolist())
    labels = fixture.labels
    membership_matrix = pd.DataFrame(
        actual_membership, index=labels, columns=identity.row_labels
    )
    clearance_matrix = pd.DataFrame(
        actual_clearances, index=labels, columns=identity.row_labels
    )
    membership_mismatches = tuple(
        (labels[row], column, expected, bool(actual_membership[row][column]))
        for row, expected_row in enumerate(fixture.expected_membership)
        for column, expected in enumerate(expected_row)
        if bool(actual_membership[row][column]) != expected
    )
    clearance_mismatches = tuple(
        (
            labels[row],
            column,
            expected,
            float(actual_clearances[row][column]),
        )
        for row, expected_row in enumerate(fixture.expected_clearances)
        for column, expected in enumerate(expected_row)
        if float(actual_clearances[row][column]) != expected
    )
    domain_mismatches = tuple(
        (label, expected, actual)
        for label, expected, actual in zip(
            labels, identity.expected_domain_contains, actual_domain, strict=True
        )
        if expected != actual
    )
    return (
        membership_matrix,
        clearance_matrix,
        actual_domain,
        membership_mismatches,
        clearance_mismatches,
        domain_mismatches,
    )


def _render_bound_box_inference_inspection(st) -> None:
    identity = BOUND_BOX_INFERENCE_INSPECTION_FIXTURE
    fixture = BOUNDED_MULTI_BOX_INSPECTION_FIXTURE
    (
        membership_matrix,
        clearance_matrix,
        actual_domain,
        membership_mismatches,
        clearance_mismatches,
        domain_mismatches,
    ) = _bound_box_inference_inspection()
    total_cells = len(fixture.labels) * len(identity.row_labels)

    st.markdown("#### Class-bound BOX inference")
    st.markdown(
        f"Synthetic class binding · ordered IDs {identity.binding.trainable_class_ids} · "
        f"vocabulary SHA-256 `{identity.binding.class_manifest_sha256[:12]}…`"
    )
    row_associations = ", ".join(
        f"{label} → synthetic class ID {class_id}"
        for label, class_id in zip(identity.row_labels, identity.row_class_ids, strict=True)
    )
    st.markdown(f"Geometry row association: {row_associations}")
    st.markdown("Bound BOX membership matrix")
    st.dataframe(membership_matrix, width="stretch")
    st.markdown("Bound BOX signed-coordinate-clearance matrix")
    st.dataframe(clearance_matrix, width="stretch")
    st.markdown(
        "DomainThing vector (separate): "
        f"expected {identity.expected_domain_contains}; actual {actual_domain}"
    )

    if membership_mismatches:
        details = "; ".join(
            f"point {point}, box index {box_index}: expected {expected}, actual {actual}"
            for point, box_index, expected, actual in membership_mismatches
        )
        st.error(
            f"Bound BOX membership mismatches: {len(membership_mismatches)} of "
            f"{total_cells} ({details})"
        )
    else:
        st.success(f"Bound BOX membership mismatches: 0 of {total_cells}")

    if clearance_mismatches:
        details = "; ".join(
            f"point {point}, box index {box_index}: expected {expected:+.9g}, actual {actual:+.9g}"
            for point, box_index, expected, actual in clearance_mismatches
        )
        st.error(
            f"Bound BOX clearance mismatches: {len(clearance_mismatches)} of "
            f"{total_cells} ({details})"
        )
    else:
        st.success(f"Bound BOX clearance mismatches: 0 of {total_cells}")

    if domain_mismatches:
        details = "; ".join(
            f"point {point}: expected {expected}, actual {actual}"
            for point, expected, actual in domain_mismatches
        )
        st.error(
            f"DomainThing mismatches: {len(domain_mismatches)} of {len(fixture.labels)} "
            f"({details})"
        )
    else:
        st.success(f"DomainThing mismatches: 0 of {len(fixture.labels)}")


def _render_geometry_laboratory(st, runs_root: Path) -> None:
    del runs_root
    st.subheader("Geometry Laboratory")
    st.info(str(ClassifierBackendUnavailableError("historical synthetic geometry visualization")))


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Town Embeddings Research", layout="wide")
    st.title("Representing Ontologies Using Town Embeddings")
    st.caption(f"Dataset explorer remains available. Historical classifier execution, geometry interpretation, and image embedding operations are disconnected during classifier reconstruction · dataset wire schema {SCHEMA_VERSION}")
    page = st.sidebar.radio("Research workspace", ("Dataset overview", "Image explorer", "Generate dataset", "Geometry Laboratory", "Image Embeddings", "Mathematical Inspector"))
    if page == "Mathematical Inspector":
        _render_mathematical_inspector(st)
        return

    default_path = DEFAULT_EXAMPLE_DATASET if DEFAULT_EXAMPLE_DATASET.is_dir() else REPOSITORY_ROOT / "artifacts" / "datasets" / "pizza-48"
    dataset_path = Path(st.sidebar.text_input("Prepared dataset path", value=str(default_path))).expanduser()
    st.sidebar.caption("Generation is explicit. Opening or refreshing this dashboard never creates data.")
    runs_root = Path(st.sidebar.text_input("Synthetic run directory", value=str(REPOSITORY_ROOT / "artifacts" / "phase2" / "runs"))).expanduser()
    image_runs_root = Path(st.sidebar.text_input("Image embedding run directory", value=str(REPOSITORY_ROOT / "artifacts" / "phase3" / "runs"))).expanduser()

    if page == "Dataset overview":
        _render_overview(st, dataset_path)
    elif page == "Image explorer":
        _render_explorer(st, dataset_path)
    elif page == "Generate dataset":
        _generation_page(st)
    elif page == "Geometry Laboratory":
        _render_geometry_laboratory(st, runs_root)
    else:
        from town_model.dashboard.image_embeddings import render_image_embeddings

        render_image_embeddings(st, image_runs_root, dataset_path)


if __name__ == "__main__":
    main()
