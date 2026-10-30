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
    assert "sum of active class weights" in weights["normalization"]


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
    deadline = time.monotonic() + 30
    run_manifest = inspect_run(first.run_directory)
    while run_manifest["state"] == "draft" and time.monotonic() < deadline:
        time.sleep(0.02)
        run_manifest = inspect_run(first.run_directory)
    assert run_manifest["state"] == "running"
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


@pytest.mark.parametrize(
    "invalid_config, message",
    [
        (_config(vision=VisionConfig(roi_source="predicted")), "GT ROIs only"),
        (_config(model=ImageModelConfig(family="town", windows=0)), "positive integer counts"),
        (_config(vision=VisionConfig(visual_policy="trainable_fpn_last_blocks", trainable_backbone_blocks=(3.5,))), "integer feature indices"),
    ],
)
def test_invalid_dashboard_launch_fails_preflight_without_creating_output(tmp_path, invalid_config, message):
    output = tmp_path / "invalid-worker-runs"
    with pytest.raises(ValueError, match=message):
        launch_image_experiment(invalid_config, output, repository_root=Path(__file__).parents[1])
    assert not output.exists()
    assert not (output / ".image-workers").exists()
    assert not (output / ".image-launch-locks").exists()


def test_invalid_model_and_vision_preflight_has_no_run_cache_or_cli_side_effects(tmp_path):
    import json
    from dataclasses import asdict
    from town_model.cli import main

    repository = Path(__file__).parents[1]
    invalids = (
        _config(model=ImageModelConfig(family="town", windows=0)),
        _config(vision=VisionConfig(visual_policy="trainable_fpn_last_blocks", trainable_backbone_blocks=(3.5,))),
    )
    for index, config in enumerate(invalids):
        run_root = tmp_path / f"direct-{index}"
        with pytest.raises(ValueError):
            run_image_experiment(config, run_root, repository_root=repository)
        assert not run_root.exists()
        cache = tmp_path / f"cache-{index}"
        cached_config = replace(config, execution=ExecutionConfig(device="cpu", cpu_threads=2, feature_cache_directory=str(cache)))
        # Frozen cache is intentionally disabled by this model policy combination;
        # invalid config rejection must precede cache preparation.
        with pytest.raises(ValueError):
            run_image_experiment(cached_config, tmp_path / f"cached-run-{index}", repository_root=repository)
        assert not cache.exists()
        assert not (tmp_path / f"cached-run-{index}").exists()

        config_path = tmp_path / f"invalid-{index}.json"
        config_path.write_text(json.dumps(asdict(config)))
        cli_output = tmp_path / f"cli-{index}"
        with pytest.raises(SystemExit):
            main(["experiment", "image", "preflight", "--config", str(config_path)])
        with pytest.raises(SystemExit):
            main(["experiment", "image", "run", "--config", str(config_path), "--output", str(cli_output)])
        assert not cli_output.exists()


def test_forced_worker_death_reconciles_status_preserves_checkpoint_and_resumes(tmp_path):
    from town_model.experiments.image_runs import RunWriterLock, reconcile_worker_status

    repository = Path(__file__).parents[1]
    config = _config(training=TrainingConfig(
        seed=77, epochs=2, physical_batch_size=2, accumulation_steps=1,
        schedule="constant", warmup_steps=0,
    ))
    output = tmp_path / "forced-death-runs"
    launch = launch_image_experiment(config, output, repository_root=repository)
    latest = launch.run_directory / "checkpoints" / "latest.pt"
    deadline = time.monotonic() + 90
    payload = None
    while time.monotonic() < deadline:
        if latest.is_file():
            try:
                candidate = __import__("torch").load(latest, map_location="cpu", weights_only=False)
            except (OSError, EOFError, RuntimeError):
                candidate = None
            if candidate and candidate.get("optimizer_step", 0) >= 1:
                payload = candidate
                break
        time.sleep(0.02)
    assert payload is not None, _log(launch.log_path)
    checkpoint_before = __import__("hashlib").sha256(latest.read_bytes()).hexdigest()
    os.kill(launch.pid, 9)

    deadline = time.monotonic() + 30
    manifest = None
    while time.monotonic() < deadline:
        manifest = inspect_run(launch.run_directory, refresh_interrupted=True)
        if manifest["state"] == "interrupted":
            break
        time.sleep(0.05)
    assert manifest is not None and manifest["state"] == "interrupted"
    assert manifest["exit_code"] is None
    assert "unknown" in manifest["process_exit_status"]
    status = reconcile_worker_status(launch.run_directory)
    assert status is not None
    assert status["state"] == "interrupted"
    assert status["exit_code"] is None
    assert "unknown" in status["exit_status"]
    assert __import__("hashlib").sha256(latest.read_bytes()).hexdigest() == checkpoint_before
    recovered_checkpoint = __import__("torch").load(latest, map_location="cpu", weights_only=False)
    assert recovered_checkpoint["optimizer_step"] == payload["optimizer_step"]

    # Persistent orphan reservation files are safe: kernel flock ownership was
    # released by SIGKILL, so one compatible resumer may take it.
    reservation = output / ".image-launch-locks" / f"{launch.run_directory.name}.lock"
    reservation_lock = RunWriterLock(reservation)
    reservation_lock.acquire()
    reservation_lock.release()

    resumed = launch_image_experiment(config, output, repository_root=repository, resume=True)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        worker_status = json.loads(resumed.status_path.read_text(encoding="utf-8"))
        if worker_status.get("state") in ("completed", "failed", "stopped"):
            break
        time.sleep(0.1)
    assert worker_status["state"] == "completed", _log(resumed.log_path)
    final_manifest = inspect_run(resumed.run_directory)
    assert final_manifest["state"] == "completed"
    final_checkpoint = __import__("torch").load(latest, map_location="cpu", weights_only=False)
    assert final_checkpoint["optimizer_step"] > payload["optimizer_step"]


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
    assert reservation.exists()
    assert calls["kwargs"]["pass_fds"]

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
        "--status", str(status_path),
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
    assert path.exists()  # Persistent inode; flock, not pathname deletion, owns exclusivity.
    lock = RunWriterLock(path)
    lock.acquire()
    lock.release()


def test_worker_liveness_uses_host_and_pid_start_identity_conservatively():
    from town_model.experiments.image_runs import _process_identity_is_alive, _process_start_token

    pid = os.getpid()
    token = _process_start_token(pid)
    assert token is not None
    assert _process_identity_is_alive({
        "pid": pid, "hostname": socket.gethostname(), "process_start_token": token,
    })
    assert not _process_identity_is_alive({
        "pid": pid, "hostname": socket.gethostname(), "process_start_token": "different-start-time",
    })  # PID reuse is a dead owner, not proof of the recorded process.
    assert _process_identity_is_alive({
        "pid": 1, "hostname": "another-host", "process_start_token": None,
    })  # Remote owners remain uncertain and are never displaced by local PID probes.


def test_inspection_uses_live_kernel_lock_not_a_reused_or_stale_manifest_pid(tmp_path):
    from town_model.experiments.image_runs import _atomic_json

    run = tmp_path / "lock-authority-run"
    run.mkdir()
    _atomic_json(run / "manifest.json", {
        "schema_version": "image-run-manifest-v1", "run_id": run.name, "state": "running",
        "pid": 2**30, "hostname": socket.gethostname(), "process_start_token": "stale-start-token",
    })
    lock = RunWriterLock(run / ".writer.lock")
    lock.acquire()
    try:
        assert inspect_run(run)["state"] == "running"  # active lock is not displaced by stale metadata
    finally:
        lock.release()
    recovered = inspect_run(run)
    assert recovered["state"] == "interrupted"
    assert recovered["exit_code"] is None


def test_malformed_writer_lock_metadata_does_not_block_safe_recovery(tmp_path):
    path = tmp_path / ".writer.lock"
    path.write_text('{"pid":', encoding="utf-8")
    lock = RunWriterLock(path)
    lock.acquire()
    owner = json.loads(path.read_text(encoding="utf-8"))
    assert owner["protocol"] == "local-flock-v1"
    lock.release()
    # Crash-safe metadata may remain; an unlocked kernel flock is recoverable.
    lock.acquire()
    lock.release()


def test_live_writer_in_another_process_cannot_be_displaced_and_crash_releases_lock(tmp_path):
    import subprocess
    import sys

    path = tmp_path / ".writer.lock"
    script = (
        "import time; from pathlib import Path; "
        "from town_model.experiments.image_runs import RunWriterLock; "
        f"lock=RunWriterLock(Path({str(path)!r})); lock.acquire(); "
        "print('LOCKED', flush=True); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=Path(__file__).parents[1],
    )
    try:
        assert process.stdout is not None and process.stdout.readline().strip() == "LOCKED"
        with pytest.raises(RunLifecycleError, match="another process owns"):
            RunWriterLock(path).acquire()
        process.kill()
        process.wait(timeout=10)
        # The OS releases flock on forced process death, irrespective of its
        # persisted owner metadata; the next local writer may safely resume.
        recovered = RunWriterLock(path)
        recovered.acquire()
        recovered.release()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
