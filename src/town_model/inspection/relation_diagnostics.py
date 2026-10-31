"""Read-only ontology metadata checks and generic immutable JSON publication.

Historical region construction, membership, margin and relation calculations
were removed. Persisted relation JSON can still be written generically, and
model-dependent diagnosis fails with the centralized unavailable-backend error.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import tempfile
from typing import Any, NoReturn

from town_model.classifier_backend import raise_classifier_backend_unavailable
from town_model.dataset.ontology import OntologySnapshot, build_class_vocabulary

RELATION_SCHEMA = "ontology-region-relation-diagnostic-v1"


class RelationDiagnosticPostPublicationError(RuntimeError):
    """A JSON record is visible, but a later durability or cleanup operation failed."""

    def __init__(
        self,
        destination: Path,
        *,
        published: bool,
        stage: str,
        durability_confirmed: bool,
        cleanup_completed: bool,
        cleanup_durability_confirmed: bool,
        cause: Exception,
    ) -> None:
        self.destination = destination
        self.published = published
        self.stage = stage
        self.durability_confirmed = durability_confirmed
        self.cleanup_completed = cleanup_completed
        self.cleanup_durability_confirmed = cleanup_durability_confirmed
        self.cause = cause
        message = (
            f"relation diagnostic was published at {destination}; complete JSON result is visible. "
            f"Post-publication {stage.replace('_', ' ')} failed: {cause}. "
            "Do not retry as if no result were published."
        )
        if not durability_confirmed:
            message += " Destination directory-entry durability is not confirmed."
        if not cleanup_completed:
            message += " The temporary hard link may remain."
        elif stage == "final_directory_sync":
            message += " Temporary cleanup completed, but its directory durability is not confirmed."
        super().__init__(message)


def validate_snapshot_ontology_mapping(snapshot, ontology: OntologySnapshot) -> None:
    """Check stored class IDs/names against the authoritative ontology vocabulary."""
    trainable = [item for item in build_class_vocabulary(ontology)["classes"] if item["trainable"]]
    expected = {item["class_id"]: item["name"] for item in trainable}
    for class_id, name in zip(snapshot.header.get("class_ids", ()), snapshot.header.get("class_names", ()), strict=True):
        if expected.get(class_id) != name:
            raise ValueError(
                f"snapshot class identity mismatch for ID {class_id}: "
                f"saved name {name!r}, canonical ontology name {expected.get(class_id)!r}"
            )


def diagnose_region_relation(*args, **kwargs) -> NoReturn:
    """Reject historical geometry relation computation; no regions are reconstructed."""
    raise_classifier_backend_unavailable("historical ontology-region relation diagnosis")


def _fsync_directory(directory: Path) -> bool:
    """Persist directory-entry changes and report whether fsync was available."""
    if os.name == "nt":
        return False
    unsupported = {
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as error:
        if error.errno in unsupported:
            return False
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in unsupported:
                raise
            return False
    finally:
        os.close(descriptor)
    return True


def save_relation_diagnostic(result: dict[str, Any], destination: Path | str) -> Path:
    """Atomically publish caller-supplied JSON with same-directory no-clobber linking.

    This generic storage helper does not compute or validate relation semantics.
    """
    path = Path(os.path.abspath(os.fspath(Path(destination).expanduser())))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    descriptor_open = True
    published = False

    def clean_up_before_publication() -> None:
        if descriptor_open:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.unlink(temporary)
        except OSError:
            pass

    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor_open = False
        with stream:
            json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        clean_up_before_publication()
        raise

    try:
        os.link(temporary, path)
        published = True
    except FileExistsError as error:
        clean_up_before_publication()
        raise FileExistsError(f"immutable JSON result already exists: {path}") from error
    except OSError as error:
        clean_up_before_publication()
        raise type(error)(f"atomic no-clobber hard-link publication failed for {path}: {error}") from error
    except BaseException:
        clean_up_before_publication()
        raise

    try:
        durability_confirmed = _fsync_directory(path.parent)
    except Exception as error:
        raise RelationDiagnosticPostPublicationError(
            path, published=published, stage="initial_directory_sync", durability_confirmed=False,
            cleanup_completed=False, cleanup_durability_confirmed=False, cause=error,
        ) from error
    try:
        os.unlink(temporary)
    except Exception as error:
        raise RelationDiagnosticPostPublicationError(
            path, published=published, stage="temporary_cleanup", durability_confirmed=durability_confirmed,
            cleanup_completed=False, cleanup_durability_confirmed=False, cause=error,
        ) from error
    try:
        _fsync_directory(path.parent)
    except Exception as error:
        raise RelationDiagnosticPostPublicationError(
            path, published=published, stage="final_directory_sync", durability_confirmed=durability_confirmed,
            cleanup_completed=True, cleanup_durability_confirmed=False, cause=error,
        ) from error
    return path
