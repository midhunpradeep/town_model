"""Pinned upstream generator adapter; upstream files are never patched."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from ..config import DatasetConfig
from .constants import GENERATOR_COMMIT, GENERATOR_REPOSITORY

Progress = Callable[[float, str], None]


def _git(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=check,
        text=True,
        capture_output=True,
    )


def _assert_clean_pinned_checkout(path: Path) -> None:
    if not path.is_dir() or not (path / ".git").exists():
        raise ValueError(f"Pizzaïolo source path is not a Git checkout: {path}")
    origin = _git(path, "remote", "get-url", "origin").stdout.strip().rstrip("/")
    if origin != GENERATOR_REPOSITORY.rstrip("/"):
        raise ValueError(f"Unexpected Pizzaïolo source origin {origin!r} at {path}")
    revision = _git(path, "rev-parse", "HEAD").stdout.strip()
    if revision != GENERATOR_COMMIT:
        raise ValueError(
            f"Pizzaïolo source must be pinned at {GENERATOR_COMMIT}, found {revision}"
        )
    dirty = _git(path, "status", "--porcelain").stdout.strip()
    if dirty:
        raise ValueError(f"Pinned Pizzaïolo checkout must be clean; dirty files in {path}")


def _is_unchecked_out_cache(path: Path) -> bool:
    """Recognize the empty index left by a clone created with --no-checkout."""

    committed = _git(path, "ls-tree", "-r", "--name-only", "HEAD").stdout.splitlines()
    indexed = _git(path, "ls-files").stdout.splitlines()
    status = _git(path, "status", "--porcelain").stdout.splitlines()
    return (
        bool(committed)
        and not indexed
        and len(status) == len(committed)
        and all(line.startswith("D  ") for line in status)
    )


def ensure_pizzaiolo_repo(project_root: Path, source_repo: Path | None = None) -> Path:
    """Resolve source without changing user-supplied checkouts.

    A user-specified/env checkout must already be clean and pinned. Otherwise,
    a dedicated ignored cache is cloned and checked out at the immutable SHA.
    """

    if source_repo is None:
        configured = os.environ.get("PIZZAIOLO_SOURCE_DIR")
        source_repo = Path(configured).expanduser() if configured else None
    if source_repo is not None:
        source = source_repo.expanduser().resolve()
        _assert_clean_pinned_checkout(source)
        return source

    source = (project_root / "artifacts" / "sources" / "pizzaiolo").resolve()
    source.parent.mkdir(parents=True, exist_ok=True)
    new_clone = not source.exists()
    if new_clone:
        subprocess.run(
            ["git", "clone", "--no-checkout", GENERATOR_REPOSITORY, str(source)],
            check=True,
            text=True,
            capture_output=True,
        )
    else:
        origin = _git(source, "remote", "get-url", "origin").stdout.strip().rstrip("/")
        if origin != GENERATOR_REPOSITORY.rstrip("/"):
            raise ValueError(f"Unexpected cached Pizzaïolo origin {origin!r} at {source}")

    revision = _git(source, "rev-parse", "HEAD", check=False)
    current = revision.stdout.strip() if revision.returncode == 0 else ""
    needs_checkout = new_clone
    if current != GENERATOR_COMMIT:
        fetched = _git(source, "cat-file", "-e", f"{GENERATOR_COMMIT}^{{commit}}", check=False)
        if fetched.returncode != 0:
            _git(source, "fetch", "origin", GENERATOR_COMMIT)
        needs_checkout = True
    elif not new_clone and _is_unchecked_out_cache(source):
        needs_checkout = True
    if needs_checkout:
        _git(source, "checkout", "--detach", GENERATOR_COMMIT)
    _assert_clean_pinned_checkout(source)
    return source


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def seed_generation(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("generation_seed must be an integer from 0 through 2**31 - 1")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import cv2
    except ImportError:
        return
    cv2.setRNGSeed(seed)
    cv2.setNumThreads(1)


def _load_upstream_class(source: Path):
    module_path = source / "libs" / "pizzaiolo.py"
    spec = importlib.util.spec_from_file_location("_town_model_pinned_pizzaiolo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pinned Pizzaïolo generator at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module.Pizzaiolo


def generate_raw_dataset(
    dataset_dir: Path,
    config: DatasetConfig,
    *,
    project_root: Path,
    source_repo: Path | None = None,
    progress: Progress | None = None,
) -> dict[str, int]:
    """Generate exactly the configured raw dataset with the pinned revision."""

    config.validate(require_total=True)
    if dataset_dir.exists():
        raise FileExistsError(f"Raw dataset output already exists: {dataset_dir}")
    counts = config.allocate_type_counts()
    if progress:
        progress(0.04, "Resolving the pinned Pizzaïolo source")
    source = ensure_pizzaiolo_repo(project_root, source_repo)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config.resolved_dict(), sort_keys=True, separators=(",", ":"))
    environment = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = package_root + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = str(config.generation_seed)
    command = [
        sys.executable,
        "-m",
        "town_model.dataset.generator_worker",
        "--source",
        str(source),
        "--destination",
        str(dataset_dir.resolve()),
        "--config-json",
        payload,
    ]
    if progress:
        progress(0.08, "Cooking the configured generator types")
    process = subprocess.Popen(
        command,
        cwd=source,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        while process.poll() is None:
            if progress:
                image_dir = dataset_dir / "images"
                completed = len(list(image_dir.glob("*.png"))) if image_dir.exists() else 0
                fraction = min(completed / max(config.total_instances or 1, 1), 1.0)
                progress(0.08 + 0.66 * fraction, f"Rendering images: {completed}/{config.total_instances}")
            time.sleep(0.3)
        stdout, stderr = process.communicate()
    except BaseException:
        process.terminate()
        process.wait(timeout=5)
        raise
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or f"generator worker exited with status {process.returncode}"
        raise RuntimeError(f"Pinned Pizzaïolo generation failed: {detail}")

    _assert_clean_pinned_checkout(source)
    if progress:
        progress(0.76, "Raw images and annotations generated")
    return counts
