from __future__ import annotations

from pathlib import Path

import pytest

from town_model.experiments.image_config import (
    DatasetSubsetConfig,
    ExecutionConfig,
    ImageExperimentConfig,
    ImageModelConfig,
    TrainingConfig,
    VisionConfig,
)
from town_model.dashboard.image_embeddings import build_image_geometry_figure, point_rows
from town_model.inspection.image_snapshots import load_image_snapshot
from town_model.experiments.image_runs import (
    RunLifecycleError,
    RunWriterLock,
    inspect_run,
    resolve_image_experiment,
    run_image_experiment,
    run_image_matrix,
    transition_run,
)


def _config(**overrides):
    value = dict(
        dataset=DatasetSubsetConfig(dataset_path="examples/pizza-48", train_size=4, validation_size=2, selection_seed=31),
        vision=VisionConfig(visual_policy="frozen"),
        model=ImageModelConfig(family="baseline", pizza_mode="hybrid", latent_dimensions=2),
        training=TrainingConfig(seed=23, epochs=2, physical_batch_size=2, accumulation_steps=1, schedule="constant", warmup_steps=0),
        execution=ExecutionConfig(device="cpu", cpu_threads=2),
        purpose="smoke",
    )
    value.update(overrides)
    return ImageExperimentConfig(**value)


def test_run_resolution_persists_original_split_ids_and_ignores_visual_preferences_for_identity():
    first = resolve_image_experiment(_config())
    assert first.run_id.startswith("image-")
    assert {record.split for record in first.train_subset.records} == {"train"}
    assert {record.split for record in first.validation_subset.records} == {"validation"}
    assert not {record.sample_id for record in first.train_subset.records} & {record.sample_id for record in first.validation_subset.records}
    assert first.identity["dataset_id"] == first.dataset.dataset_id
    assert first.identity["class_manifest_sha256"] == first.dataset.class_manifest_sha256


def test_run_service_lifecycle_stop_and_faithful_resume(tmp_path):
    config = _config()
    calls = 0

    def stop_after_first_optimizer_update():
        nonlocal calls
        calls += 1
        return calls >= 2

    run_path, stopped = run_image_experiment(
        config,
        tmp_path / "runs",
        repository_root=Path(__file__).parents[1],
        stop_requested=stop_after_first_optimizer_update,
    )
    assert stopped.status == "stopped"
    assert inspect_run(run_path)["state"] == "stopped"
    assert (run_path / "resolved-config.json").is_file()
    assert (run_path / "checkpoints" / "latest.pt").is_file()

    resumed_path, completed = run_image_experiment(
        config,
        tmp_path / "runs",
        repository_root=Path(__file__).parents[1],
        resume=True,
    )
    assert resumed_path == run_path
    assert completed.status == "completed"
    manifest = inspect_run(run_path)
    assert manifest["state"] == "completed"
    assert manifest["optimizer_step"] == 4
    assert (run_path / "metrics.json").is_file()
    snapshot = load_image_snapshot(run_path / "snapshots" / "epoch-00002")
    assert set(snapshot.arrays["source_splits"].tolist()) <= {"train", "validation"}
    assert snapshot.header["roi_source"] == "ground_truth"
    assert "optimizer_state" not in snapshot.arrays
    assert "component_points" in snapshot.arrays and "pizza_points_final" in snapshot.arrays
    assert point_rows(snapshot, 0, level="pizza", class_index=0)
    geometry_figure = build_image_geometry_figure(snapshot, sample_index=0, level="pizza", class_index=0)
    import matplotlib.pyplot as plt
    plt.close(geometry_figure)
    with pytest.raises(RunLifecycleError, match="invalid image run state transition"):
        transition_run(run_path, "running")
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        run_image_experiment(config, tmp_path / "runs", repository_root=Path(__file__).parents[1])


def test_family_matrix_requires_a_strict_comparison_configuration(tmp_path):
    output = tmp_path / "matrix-runs"
    with pytest.raises(ValueError, match="purpose='comparison'"):
        run_image_matrix(_config(), output, repository_root=Path(__file__).parents[1])
    assert not output.exists()


def test_invalid_configuration_fails_before_any_run_directory_is_created(tmp_path):
    config = _config(vision=VisionConfig(roi_source="predicted"))
    output = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="GT ROIs only"):
        run_image_experiment(config, output, repository_root=Path(__file__).parents[1])
    assert not output.exists()


def test_one_writer_per_run_lock(tmp_path):
    path = tmp_path / ".writer.lock"
    first = RunWriterLock(path)
    second = RunWriterLock(path)
    first.acquire()
    try:
        with pytest.raises(RunLifecycleError, match="another process owns"):
            second.acquire()
    finally:
        first.release()
    assert not path.exists()
