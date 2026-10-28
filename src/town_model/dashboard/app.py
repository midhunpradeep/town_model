"""Permanent local Phase 1 dashboard built on shared application operations."""

from __future__ import annotations

import json
import queue
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw

from town_model.application import validate_dataset_operation, validate_generation_request
from town_model.config import DEFAULT_PIZZA_TYPES, DatasetConfig, load_config
from town_model.dataset.constants import SCHEMA_VERSION
from town_model.dashboard.data import class_names_for_targets, find_sample, load_dashboard_data
from town_model.dashboard.runtime import submit_generation

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_CONFIG = REPOSITORY_ROOT / "dataset.toml"
DEFAULT_EXAMPLE_DATASET = REPOSITORY_ROOT / "examples" / "pizza-48"


def _class_labels(vector: tuple[int, ...], classes: tuple[dict[str, Any], ...]) -> list[str]:
    return class_names_for_targets(vector, classes)


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
        manifest_path = summary.path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not verified and isinstance(manifest, dict):
                manifest.pop("statistics", None)
                st.caption("Manifest statistics are historical and withheld because current integrity did not pass.")
            st.json(manifest)
        except (OSError, json.JSONDecodeError) as error:
            st.error(f"Could not read manifest: {error}")
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
        with Image.open(explorer.summary.path / record.image_path) as image:
            image = image.convert("RGB")
            if show_boxes:
                image = _draw_annotations(image, record.components, selected_component.component_id if selected_component else None)
            st.image(image, caption=f"{record.sample_id} · split={record.split} · type={record.generator_type}", width="stretch")
    except OSError as error:
        st.error(f"Could not open image for sample {record.sample_id}: {error}")
        return

    st.markdown("#### Whole-pizza ontology labels")
    st.caption("The source-inferred labels below are retained as audit provenance; ontology targets include corrected class ancestry.")
    st.write("Source-inferred labels:", ", ".join(record.pizza_source_labels))
    st.write("Corrected ontology targets:", ", ".join(_class_labels(record.pizza_targets, explorer.classes)))

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


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Town Embeddings Research", layout="wide")
    st.title("Representing Ontologies Using Town Embeddings")
    st.caption(f"Phase 1 · dataset preparation and exploration · wire schema {SCHEMA_VERSION}")
    page = st.sidebar.radio("Research workspace", ("Dataset overview", "Image explorer", "Generate dataset"))
    default_path = DEFAULT_EXAMPLE_DATASET if DEFAULT_EXAMPLE_DATASET.is_dir() else REPOSITORY_ROOT / "artifacts" / "datasets" / "pizza-48"
    dataset_path = Path(st.sidebar.text_input("Prepared dataset path", value=str(default_path))).expanduser()
    st.sidebar.caption("Generation is explicit. Opening or refreshing this dashboard never creates data.")

    if page == "Dataset overview":
        _render_overview(st, dataset_path)
    elif page == "Image explorer":
        _render_explorer(st, dataset_path)
    else:
        _generation_page(st)


if __name__ == "__main__":
    main()
