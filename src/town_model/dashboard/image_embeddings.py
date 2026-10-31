"""Metadata-only dashboard inspection for historical image runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from town_model.classifier_backend import ClassifierBackendUnavailableError
from town_model.experiments.image_runs import RunLifecycleError, inspect_run
from town_model.inspection.image_snapshots import discover_image_snapshots, load_image_snapshot


def discover_image_runs(root: Path | str) -> list[Path]:
    """List persisted run manifests without loading models or checkpoints."""
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
        if (
            isinstance(manifest, dict)
            and isinstance(config, dict)
            and manifest.get("schema_version") == "image-run-manifest-v1"
            and manifest.get("state") in {"draft", "running", "completed", "stopped", "interrupted", "failed"}
            and config.get("identity_hash") == manifest.get("identity_hash")
        ):
            found.append(candidate)
    return found


def discover_snapshots(run_directory: Path | str) -> list[Path]:
    """Discover only checksummed historical snapshots."""
    return discover_image_snapshots(run_directory)


def render_image_embeddings(st, runs_root: Path, dataset_path: Path) -> None:
    """Show stored metadata only; never load checkpoints or interpret geometry."""
    del dataset_path  # Dataset images/overlays belong to a separate presentation decision.
    st.subheader("Image Embeddings")
    st.info(str(ClassifierBackendUnavailableError(
        "image classifier training, evaluation, resume, and geometry dashboard actions"
    )))
    runs = discover_image_runs(runs_root)
    if not runs:
        st.info(f"No persisted image-run metadata was found under `{runs_root}`.")
        return
    selected_run = st.selectbox("Historical image run metadata", runs, format_func=lambda path: path.name)
    try:
        manifest = inspect_run(selected_run)
        resolved_config = json.loads((selected_run / "resolved-config.json").read_text(encoding="utf-8"))
        metrics_path = selected_run / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    except (OSError, ValueError, RunLifecycleError, json.JSONDecodeError) as error:
        st.error(f"Could not read persisted historical run metadata: {error}")
        return
    st.caption(
        "Persisted values only: no classifier, checkpoint, optimizer, prediction, score, "
        "or geometric interpretation is recomputed. Lifecycle state is not reconciled or modified."
    )
    st.json({"run_manifest": manifest, "resolved_configuration": resolved_config, "stored_metrics": metrics})

    snapshots = discover_snapshots(selected_run)
    if not snapshots:
        st.info("No persisted image snapshots passed checksum-validated metadata inspection.")
        return
    selected_snapshot = st.selectbox(
        "Checksummed historical snapshot metadata", snapshots, format_func=lambda path: path.name
    )
    try:
        snapshot = load_image_snapshot(selected_snapshot)
    except (OSError, ValueError) as error:
        st.error(f"Snapshot failed its read-only integrity checks: {error}")
        return
    header: dict[str, Any] = snapshot.header
    st.json({key: header.get(key) for key in (
        "schema_version", "epoch", "optimizer_step", "identity_hash", "dataset_id",
        "manifest_id", "source_checkpoint_sha256", "arrays_sha256", "sample_ids",
        "source_splits", "class_ids", "class_names", "component_count_original",
        "component_count_traced",
    )})
