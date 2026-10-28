"""One local background generation operation; intentionally not a job scheduler."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..application import generate_dataset
from ..config import DatasetConfig

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="town-dataset-generation")
_generation_lock = threading.Lock()


def submit_generation(
    config: DatasetConfig,
    output_path: Path,
    progress_queue: queue.Queue[tuple[float, str]],
) -> Future[Path]:
    """Submit one explicit generation; reject concurrent local dashboard jobs."""

    if not _generation_lock.acquire(blocking=False):
        raise RuntimeError("A dataset generation is already running in this dashboard process")

    def run() -> Path:
        try:
            return generate_dataset(
                config,
                output_path,
                progress=lambda fraction, message: progress_queue.put((fraction, message)),
            )
        finally:
            _generation_lock.release()

    try:
        return _executor.submit(run)
    except Exception:
        _generation_lock.release()
        raise
