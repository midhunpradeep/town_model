from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from town_model.dataset.image_data import deterministic_subset, load_prepared_dataset
from town_model.experiments.image_config import (
    DatasetSubsetConfig,
    ExecutionConfig,
    ImageExperimentConfig,
    ImageModelConfig,
    TrainingConfig,
    VisionConfig,
)
from town_model.experiments.image_runs import (
    RunLifecycleError,
    RunWriterLock,
    _process_identity_is_alive,
    _process_start_token,
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


def test_run_config_identity_and_original_subset_order_remain_metadata_only():
    repository = Path(__file__).parents[1]
    config = _config().validate()
    dataset = load_prepared_dataset(repository / "examples" / "pizza-48")
    train = deterministic_subset(dataset, "train", size=config.dataset.train_size, seed=config.dataset.selection_seed)
    validation = deterministic_subset(dataset, "validation", size=config.dataset.validation_size, seed=config.dataset.selection_seed + 1)
    assert {record.split for record in train.records} == {"train"}
    assert {record.split for record in validation.records} == {"validation"}
    assert not {record.sample_id for record in train.records} & {record.sample_id for record in validation.records}
    presentation_variant = _config(visualization=replace(config.visualization, point_projection="pca"))
    assert config.identity_hash() == presentation_variant.identity_hash()


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
    assert path.exists()
    lock = RunWriterLock(path)
    lock.acquire()
    lock.release()


def test_worker_liveness_uses_host_and_pid_start_identity_conservatively():
    pid = os.getpid()
    token = _process_start_token(pid)
    assert token is not None
    assert _process_identity_is_alive({"pid": pid, "hostname": socket.gethostname(), "process_start_token": token})
    assert not _process_identity_is_alive({"pid": pid, "hostname": socket.gethostname(), "process_start_token": "different-start-time"})
    assert _process_identity_is_alive({"pid": 1, "hostname": "another-host", "process_start_token": None})


def test_malformed_writer_lock_metadata_does_not_block_safe_recovery(tmp_path):
    path = tmp_path / ".writer.lock"
    path.write_text('{"pid":', encoding="utf-8")
    lock = RunWriterLock(path)
    lock.acquire()
    owner = json.loads(path.read_text(encoding="utf-8"))
    assert owner["protocol"] == "local-flock-v1"
    lock.release()
    lock.acquire()
    lock.release()


def test_live_writer_in_another_process_cannot_be_displaced_and_crash_releases_lock(tmp_path):
    path = tmp_path / ".writer.lock"
    script = (
        "import time; from pathlib import Path; "
        "from town_model.experiments.image_runs import RunWriterLock; "
        f"lock=RunWriterLock(Path({str(path)!r})); lock.acquire(); "
        "print('LOCKED', flush=True); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=Path(__file__).parents[1],
    )
    try:
        assert process.stdout is not None and process.stdout.readline().strip() == "LOCKED"
        with pytest.raises(RunLifecycleError, match="another process owns"):
            RunWriterLock(path).acquire()
        process.kill()
        process.wait(timeout=10)
        recovered = RunWriterLock(path)
        recovered.acquire()
        recovered.release()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
