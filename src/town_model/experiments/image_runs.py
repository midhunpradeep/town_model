"""Read-only image-run metadata and classifier-independent lifecycle records.

Model-dependent run, worker, resume, and matrix operations reject through the
shared unavailable-backend error. Stored run identity/configuration is retained
as JSON metadata and is never used to reconstruct a historical model here.
"""

from __future__ import annotations

from datetime import datetime, timezone
import errno
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses byte-range locking below
    fcntl = None
import json
import os
from pathlib import Path
import socket
import tempfile
from typing import Any, NoReturn

if fcntl is None:  # pragma: no cover - exercised on Windows CI only
    import msvcrt

from town_model.classifier_backend import raise_classifier_backend_unavailable
from town_model.dataset.image_data import deterministic_subset, load_prepared_dataset
from town_model.experiments.image_config import ImageExperimentConfig
from town_model.experiments.image_validation import device_for_image_config, validate_image_training_data

RUN_STATES = ("draft", "running", "completed", "stopped", "interrupted", "failed")
_ALLOWED_TRANSITIONS = {
    "draft": {"running", "failed"},
    "running": {"completed", "stopped", "interrupted", "failed"},
    "stopped": {"running"},
    "interrupted": {"running"},
    "failed": {"running"},
    "completed": set(),
}


class RunLifecycleError(RuntimeError):
    """Invalid local image-run metadata state or writer ownership."""


def _atomic_json(path: Path, value: object) -> None:
    """Atomically write generic JSON metadata without model interpretation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=str)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def resolve_image_experiment(
    config: ImageExperimentConfig,
    *,
    repository_root: Path | str | None = None,
) -> NoReturn:
    """Validate persisted config/data inputs, then reject model execution."""
    config.validate()
    device_for_image_config(config)
    base = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[3]
    dataset = load_prepared_dataset(config.resolve_dataset_path(base), validate=True)
    train_subset = deterministic_subset(
        dataset, "train", size=config.dataset.train_size, seed=config.dataset.selection_seed
    )
    validation_subset = deterministic_subset(
        dataset, "validation", size=config.dataset.validation_size, seed=config.dataset.selection_seed + 1
    )
    train_ids = {record.sample_id for record in train_subset.records}
    validation_ids = {record.sample_id for record in validation_subset.records}
    if train_ids & validation_ids:
        raise ValueError("original train and validation partitions unexpectedly share sample IDs")
    class_ids = tuple(config.dataset.trainable_class_ids or dataset.canonical_class_ids)
    validate_image_training_data(
        dataset,
        train_subset,
        validation_subset,
        class_ids,
        require_training_coverage=config.purpose == "comparison",
        require_primary_monitor=config.purpose != "smoke",
    )
    raise_classifier_backend_unavailable("image classifier experiment preflight")


def _process_record(pid: int) -> tuple[str | None, str | None]:
    """Read Linux start-time token and state, safely handling spaces in comm."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields_after_comm = stat[stat.rfind(")") + 2 :].split()
        return fields_after_comm[19], fields_after_comm[0]
    except (OSError, IndexError, ValueError):
        return None, None


def _try_file_lock(descriptor: int) -> None:
    if fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    if os.fstat(descriptor).st_size == 0:
        os.write(descriptor, b"\\0")
    os.lseek(descriptor, 0, os.SEEK_SET)
    try:
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            raise BlockingIOError(error.errno, error.strerror) from error
        raise


def _unlock_file_lock(descriptor: int) -> None:
    if fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    else:
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)


def _process_start_token(pid: int) -> str | None:
    return _process_record(pid)[0]


def _process_identity_is_alive(owner: dict[str, Any]) -> bool:
    if owner.get("hostname") != socket.gethostname():
        return True
    try:
        pid = int(owner["pid"])
        if pid <= 0:
            return True
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return True
    observed, state = _process_record(pid)
    if state in ("Z", "X"):
        return False
    expected = owner.get("process_start_token")
    if expected is not None and observed is not None:
        return expected == observed
    return True


class RunWriterLock:
    """Crash-released local advisory lock for independent metadata lifecycle users."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.acquired = False
        self._descriptor: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            _try_file_lock(descriptor)
        except BlockingIOError as error:
            os.close(descriptor)
            if self.path.parent.name == ".image-launch-locks":
                message = f"another image worker launch is active for this run: {self.path}"
            else:
                message = f"another process owns this image run lock: {self.path}"
            raise RunLifecycleError(message) from error
        except OSError:
            os.close(descriptor)
            raise
        owner = {
            "protocol": "local-flock-v1",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "process_start_token": _process_start_token(os.getpid()),
            "acquired_at": _now(),
        }
        try:
            encoded = json.dumps(owner, sort_keys=True).encode("utf-8")
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            view = memoryview(encoded)
            while view:
                view = view[os.write(descriptor, view):]
            os.fsync(descriptor)
        except Exception:
            _unlock_file_lock(descriptor)
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        self.acquired = True

    def fileno(self) -> int:
        if not self.acquired or self._descriptor is None:
            raise RunLifecycleError("writer lock is not acquired")
        return self._descriptor

    def detach(self) -> None:
        if self.acquired and self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
            self.acquired = False

    def release(self) -> None:
        if self.acquired and self._descriptor is not None:
            _unlock_file_lock(self._descriptor)
            os.close(self._descriptor)
            self._descriptor = None
            self.acquired = False

    def __enter__(self) -> "RunWriterLock":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_manifest(run_directory: Path) -> dict[str, Any]:
    try:
        payload = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunLifecycleError(f"could not read image run manifest in {run_directory}: {error}") from error
    if not isinstance(payload, dict) or payload.get("state") not in RUN_STATES:
        raise RunLifecycleError(f"invalid image run manifest in {run_directory}")
    return payload


def transition_run(run_directory: Path | str, new_state: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Update a generic lifecycle record; this does not launch model execution."""
    if new_state not in RUN_STATES:
        raise RunLifecycleError(f"unknown image run state {new_state!r}")
    root = Path(run_directory).resolve()
    manifest = _read_manifest(root)
    old_state = manifest["state"]
    if new_state not in _ALLOWED_TRANSITIONS[old_state]:
        raise RunLifecycleError(f"invalid image run state transition {old_state!r} -> {new_state!r}")
    manifest["state"] = new_state
    manifest["updated_at"] = _now()
    if details:
        manifest.update(details)
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def _process_alive(manifest: dict[str, Any]) -> bool:
    return _process_identity_is_alive(manifest)


def _reconcile_worker_status_file(root: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    status_path = root.parent / ".image-workers" / f"{root.name}.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(status, dict):
        return None
    state = status.get("state")
    if state in ("launching", "running", "draft"):
        if manifest["state"] == "running":
            status.update({
                "state": "running", "pid": manifest.get("pid"),
                "hostname": manifest.get("hostname"),
                "process_start_token": manifest.get("process_start_token"),
                "reconciled_from_manifest": True,
            })
        elif manifest["state"] == "draft":
            reservation = root.parent / ".image-launch-locks" / f"{root.name}.lock"
            if _advisory_lock_is_held(reservation) or _process_identity_is_alive(status):
                status.update({"state": "launching", "reconciled_from_manifest": True})
            else:
                status.update({
                    "state": "interrupted", "exit_code": None,
                    "exit_status": "unknown; worker ended before run startup completed",
                    "finished_at": status.get("finished_at") or _now(),
                    "reconciled_from_manifest": True,
                })
        else:
            status.update({
                "state": manifest["state"], "exit_code": status.get("exit_code"),
                "exit_status": status.get("exit_status") or "unknown; no worker exit code was observed",
                "interruption_reason": manifest.get("interruption_reason"),
                "reconciled_from_manifest": True,
                "finished_at": status.get("finished_at") or _now(),
            })
    _atomic_json(status_path, status)
    return status


def _advisory_lock_is_held(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        descriptor = os.open(path, os.O_RDWR)
    except OSError:
        return True
    try:
        try:
            _try_file_lock(descriptor)
        except BlockingIOError:
            return True
        _unlock_file_lock(descriptor)
        return False
    except OSError:
        return True
    finally:
        os.close(descriptor)


def reconcile_worker_status(run_directory: Path | str) -> dict[str, Any] | None:
    """Explicitly reconcile generic worker metadata against process identity."""
    root = Path(run_directory).expanduser().resolve()
    if (root / "manifest.json").is_file():
        manifest = inspect_run(root)
        return _reconcile_worker_status_file(root, manifest)
    status_path = root.parent / ".image-workers" / f"{root.name}.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    reservation = root.parent / ".image-launch-locks" / f"{root.name}.lock"
    if status.get("state") in ("launching", "running") and not _advisory_lock_is_held(reservation) and not _process_identity_is_alive(status):
        status.update({
            "state": "interrupted", "exit_code": None,
            "exit_status": "unknown; worker died before publishing a run manifest",
            "finished_at": _now(), "reconciled_from_process_liveness": True,
        })
        _atomic_json(status_path, status)
    return status


def inspect_run(run_directory: Path | str, *, refresh_interrupted: bool = False) -> dict[str, Any]:
    """Read persisted lifecycle JSON without changing run or worker state."""
    del refresh_interrupted
    return _read_manifest(Path(run_directory).expanduser().resolve())


def request_stop(*args, **kwargs) -> NoReturn:
    raise_classifier_backend_unavailable("controlling a historical image classifier worker")


def launch_image_experiment(*args, **kwargs) -> NoReturn:
    raise_classifier_backend_unavailable("launching an image classifier background worker")


def run_image_experiment(*args, **kwargs) -> NoReturn:
    raise_classifier_backend_unavailable("image-to-ontology classifier training or resume")


def run_image_matrix(*args, **kwargs) -> NoReturn:
    raise_classifier_backend_unavailable("automatic image classifier matrix execution")
