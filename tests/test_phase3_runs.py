from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import os
import socket
import time

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
    launch_image_experiment,
    release_launch_reservation,
    request_stop,
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
    weights = first.resolved_config()["class_weight_vectors"]
    assert weights["policy"] == "none"
    assert set(weights["component_positive"]) == {1.0}
    assert set(weights["pizza_positive"]) == {1.0}
    assert "sum of known class weights" in weights["normalization"]


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


def test_dashboard_worker_launches_detached_cli_process_and_duplicate_launch_is_rejected(tmp_path):
    repository = Path(__file__).parents[1]
    config = _config(training=TrainingConfig(
        seed=23, epochs=10, physical_batch_size=2, accumulation_steps=1,
        schedule="constant", warmup_steps=0,
    ))
    output = tmp_path / "worker-runs"
    launch = launch_image_experiment(config, output, repository_root=repository)
    assert launch.pid != os.getpid()
    assert not hasattr(launch, "process")  # UI/session state retains no worker handle.
    with pytest.raises((FileExistsError, RunLifecycleError), match="already exists|launch is active"):
        launch_image_experiment(config, output, repository_root=repository)

    from streamlit.testing.v1 import AppTest

    app_path = repository / "src" / "town_model" / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Image Embeddings")
    app.sidebar.text_input[2].set_value(str(output))
    app.run()
    assert not app.exception
    next(button for button in app.button if button.label == "Refresh run state and logs").click()
    app.run()
    assert not app.exception
    # A Streamlit rerun has no worker handle to terminate; the detached process
    # must either still be alive or have completed successfully on its own.
    status_after_refresh = json.loads(launch.status_path.read_text(encoding="utf-8"))
    assert status_after_refresh["state"] in ("launching", "running", "completed")
    if status_after_refresh["state"] == "running":
        os.kill(launch.pid, 0)

    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            status = json.loads(launch.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(0.1)
            continue
        if status.get("state") in ("completed", "failed"):
            break
        time.sleep(0.1)
    else:
        raise AssertionError("detached image worker did not report a terminal subprocess status")
    assert status["state"] == "completed", _log(launch.log_path)
    assert status["exit_code"] == 0
    assert inspect_run(launch.run_directory)["state"] == "completed"
    progress = json.loads((launch.run_directory / "progress.json").read_text(encoding="utf-8"))
    assert progress["optimizer_step"] > 0
    assert progress["objective_identity"]["component_loss"]["name"] == "binary_cross_entropy_with_logits"
    assert "IMAGE_PROGRESS" in _log(launch.log_path)
    with pytest.raises(RunLifecycleError, match="cannot resume.*completed"):
        launch_image_experiment(config, output, repository_root=repository, resume=True)


def _log(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def test_detached_worker_graceful_stop_and_compatible_resume(tmp_path):
    repository = Path(__file__).parents[1]
    config = _config(training=TrainingConfig(
        seed=39, epochs=4, physical_batch_size=2, accumulation_steps=1,
        schedule="constant", warmup_steps=0,
    ))
    output = tmp_path / "detached-resume-runs"
    first = launch_image_experiment(config, output, repository_root=repository)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not (first.run_directory / "manifest.json").is_file():
        time.sleep(0.05)
    assert (first.run_directory / "manifest.json").is_file()
    assert inspect_run(first.run_directory)["state"] == "running"
    request_stop(first.run_directory)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        status = json.loads(first.status_path.read_text(encoding="utf-8"))
        if status.get("state") in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert status["state"] == "completed", _log(first.log_path)
    assert inspect_run(first.run_directory)["state"] == "stopped"
    assert (first.run_directory / "checkpoints" / "latest.pt").is_file()

    resumed = launch_image_experiment(config, output, repository_root=repository, resume=True)
    assert resumed.run_directory == first.run_directory
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        status = json.loads(resumed.status_path.read_text(encoding="utf-8"))
        if status.get("state") in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert status["state"] == "completed", _log(resumed.log_path)
    manifest = inspect_run(resumed.run_directory)
    assert manifest["state"] == "completed"
    assert manifest["optimizer_step"] > 0


def test_invalid_dashboard_launch_fails_preflight_without_creating_output(tmp_path):
    config = _config(vision=VisionConfig(roi_source="predicted"))
    output = tmp_path / "invalid-worker-runs"
    with pytest.raises(ValueError, match="GT ROIs only"):
        launch_image_experiment(config, output, repository_root=Path(__file__).parents[1])
    assert not output.exists()


def test_stop_api_rejects_nonrunning_state_and_writes_only_a_graceful_request(tmp_path):
    from town_model.experiments.image_runs import _atomic_json

    run = tmp_path / "running-run"
    run.mkdir()
    _atomic_json(run / "manifest.json", {
        "schema_version": "image-run-manifest-v1", "run_id": "running-run", "state": "running",
        "pid": os.getpid(), "hostname": socket.gethostname(),
    })
    request_stop(run)
    assert (run / "stop.request").is_file()
    transition_run(run, "completed")
    with pytest.raises(RunLifecycleError, match="cannot request graceful stop"):
        request_stop(run)


def test_dashboard_resume_uses_only_compatible_stopped_checkpoint_and_detached_cli(tmp_path, monkeypatch):
    import town_model.experiments.image_runs as image_runs

    repository = Path(__file__).parents[1]
    config = _config(training=TrainingConfig(
        seed=23, epochs=1, physical_batch_size=2, accumulation_steps=1,
        schedule="constant", warmup_steps=0,
    ))
    output = tmp_path / "resume-runs"
    run_path, result = run_image_experiment(
        config, output, repository_root=repository, stop_requested=lambda: True
    )
    assert result.status == "stopped"
    calls = {}

    class FakeProcess:
        pid = os.getpid()

    def fake_popen(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(image_runs.subprocess, "Popen", fake_popen)
    launch = launch_image_experiment(config, output, repository_root=repository, resume=True)
    assert launch.run_directory == run_path
    assert "--resume" in calls["command"]
    assert calls["kwargs"]["start_new_session"] is True
    assert calls["kwargs"]["stdin"] is image_runs.subprocess.DEVNULL
    reservation = output / ".image-launch-locks" / f"{run_path.name}.lock"
    owner = json.loads(reservation.read_text(encoding="utf-8"))
    release_launch_reservation(reservation, owner["token"])

    incompatible = _config(objective=replace(config.objective, geometry_loss="signed_bce"))
    with pytest.raises((FileNotFoundError, RunLifecycleError), match="exist|identity"):
        launch_image_experiment(incompatible, output, repository_root=repository, resume=True)
    assert calls["command"]  # incompatible request was rejected before spawning another worker.


def test_worker_persists_nonzero_subprocess_exit_code_and_command_context(tmp_path):
    from town_model.experiments.image_worker import run_worker

    status_path = tmp_path / "worker.json"
    status_path.write_text(json.dumps({"run_id": "failed-run", "log_path": "worker.log"}), encoding="utf-8")
    called = {}

    def failing_cli(command):
        called["command"] = list(command)
        return 17

    code = run_worker([
        "--config", str(tmp_path / "config.json"), "--output", str(tmp_path / "runs"),
        "--status", str(status_path), "--reservation", str(tmp_path / "reservation"),
        "--reservation-token", "token",
    ], command_main=failing_cli)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert code == 17
    assert status["state"] == "failed" and status["exit_code"] == 17
    assert called["command"][:3] == ["experiment", "image", "run"]


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
