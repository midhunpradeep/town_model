"""Shared Phase 1 operations used by both the CLI and Streamlit dashboard."""

from __future__ import annotations

import hashlib
import io
import json
import threading
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from itertools import count
from pathlib import Path, PurePosixPath
from typing import Any

from .config import DatasetConfig
from .dataset.constants import (
    CLASS_VOCABULARY_FILENAME,
    DETECTOR_VOCABULARY_FILENAME,
    MANIFEST_FILENAME,
    REPORT_DIR,
    SAMPLES_FILENAME,
    VALIDATION_REPORT_FILENAME,
    VOCABULARY_DIR,
)
from .dataset.generator import Progress
from .dataset.preparation import prepare_dataset, validate_output_path
from .dataset.schema import DatasetRecord, load_records_bytes
from .dataset.validation import Diagnostic, ValidationReport, validate_dataset
from .dataset.watch import DatasetFileWatcher


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    path: Path
    dataset_id: str
    schema_version: str
    sample_count: int | None
    split_sizes: dict[str, int]
    generator_type_counts: dict[str, int]
    generator_type_by_split: dict[str, dict[str, int]]
    class_support_by_split: dict[str, Any]
    provenance: dict[str, Any]
    ontology: dict[str, Any]
    vocabularies: dict[str, Any]
    integrity_status: str
    validation: dict[str, Any] | None
    saved_validation: dict[str, Any] | None
    validation_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "integrity_status": self.integrity_status,
            "split_sizes": self.split_sizes,
            "generator_type_counts": self.generator_type_counts,
            "generator_type_by_split": self.generator_type_by_split,
            "class_support_by_split": self.class_support_by_split,
            "provenance": self.provenance,
            "ontology": self.ontology,
            "vocabularies": self.vocabularies,
            "validation": self.validation,
            "validation_scope": "current_artifact_state" if self.validation is not None else None,
            "saved_validation": self.saved_validation,
            "saved_validation_scope": "historical_only" if self.saved_validation is not None else None,
            "validation_error": self.validation_error,
        }


@dataclass(frozen=True, slots=True)
class DatasetExplorer:
    summary: DatasetSummary
    records: tuple[DatasetRecord, ...]
    classes: tuple[dict[str, Any], ...]
    detector_categories: tuple[dict[str, Any], ...]
    manifest: dict[str, Any] = field(default_factory=dict)
    snapshot_token: int = 0


class DatasetSnapshotChangedError(RuntimeError):
    """The files selected for display no longer match the validated snapshot."""


@dataclass(slots=True)
class _DatasetCacheEntry:
    root: Path
    watcher: DatasetFileWatcher | None
    fallback_state: tuple[Any, ...] | None
    summary: DatasetSummary
    manifest: dict[str, Any]
    manifest_bytes: bytes | None
    report: ValidationReport | None
    token: int
    explorer: DatasetExplorer | None = None


_DATASET_CACHE: OrderedDict[Path, _DatasetCacheEntry] = OrderedDict()
_DATASET_CACHE_LIMIT = 3
_DATASET_CACHE_LOCK = threading.RLock()
_SNAPSHOT_TOKENS = count(1)
_MAX_SNAPSHOT_ATTEMPTS = 2


def _file_state(path: Path) -> tuple[Any, ...]:
    try:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError:
        return ("missing",)


def _validation_state_key(root: Path, manifest: dict[str, Any] | None) -> tuple[Any, ...]:
    """Key cached validation to metadata for the manifest and every declared asset."""

    state: list[tuple[Any, ...]] = [(MANIFEST_FILENAME, *_file_state(root / MANIFEST_FILENAME))]
    assets = manifest.get("asset_checksums") if isinstance(manifest, dict) else None
    if isinstance(assets, dict):
        for relative in sorted(assets, key=str):
            if not isinstance(relative, str) or "\\" in relative:
                state.append((repr(relative), "unsafe-path"))
                continue
            posix = PurePosixPath(relative)
            try:
                candidate = (root / Path(*posix.parts)).resolve()
            except (OSError, RuntimeError):
                state.append((relative, "unresolvable-path"))
                continue
            if posix.is_absolute() or ".." in posix.parts or root.resolve() not in candidate.parents:
                state.append((relative, "unsafe-path"))
                continue
            state.append((relative, *_file_state(candidate)))
    return (str(root), tuple(state))


def _json_object_from_bytes(data: bytes, path: Path) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_manifest(root: Path) -> tuple[dict[str, Any], bytes | None]:
    path = root / MANIFEST_FILENAME
    try:
        data = path.read_bytes()
        return _json_object_from_bytes(data, path), data
    except (OSError, json.JSONDecodeError, ValueError):
        return {}, None


def _saved_validation(root: Path) -> dict[str, Any] | None:
    path = root / REPORT_DIR / VALIDATION_REPORT_FILENAME
    try:
        return _load_json(path) if path.is_file() else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _summary_from_report(
    root: Path,
    manifest: dict[str, Any],
    report: ValidationReport | None,
    validation_error: str | None,
) -> DatasetSummary:
    status = "unavailable" if report is None else "current_pass" if report.valid else "current_fail"
    statistics = report.statistics if status == "current_pass" and report is not None else {}
    return DatasetSummary(
        path=root,
        dataset_id=str(manifest.get("dataset_id") or "unknown"),
        schema_version=str(manifest.get("schema_version") or "unknown"),
        sample_count=int(statistics["dataset_size"]) if "dataset_size" in statistics else None,
        split_sizes=dict(statistics.get("split_sizes", {})),
        generator_type_counts=dict(statistics.get("generator_type_counts", {})),
        generator_type_by_split=dict(statistics.get("generator_type_by_split", {})),
        class_support_by_split=dict(statistics.get("class_support_by_split", {})),
        provenance=dict(manifest.get("source", {})) if isinstance(manifest.get("source"), dict) else {},
        ontology=dict(manifest.get("ontology", {})) if isinstance(manifest.get("ontology"), dict) else {},
        vocabularies=dict(manifest.get("vocabularies", {})) if isinstance(manifest.get("vocabularies"), dict) else {},
        integrity_status=status,
        validation=deepcopy(report.to_dict()) if report is not None else None,
        saved_validation=_saved_validation(root),
        validation_error=validation_error,
    )


def _new_cache_entry(root: Path, *, write_report: bool = False) -> _DatasetCacheEntry:
    watcher: DatasetFileWatcher | None
    try:
        watcher = DatasetFileWatcher(root)
    except (NotImplementedError, OSError, ValueError):
        watcher = None

    manifest: dict[str, Any] = {}
    manifest_bytes: bytes | None = None
    report: ValidationReport | None = None
    validation_error: str | None = None
    stable = False
    fallback_state: tuple[Any, ...] | None = None

    for attempt in range(_MAX_SNAPSHOT_ATTEMPTS):
        manifest, before_manifest_bytes = _read_manifest(root)
        before_state = _validation_state_key(root, manifest) if watcher is None else None
        if watcher is not None:
            # Start the validation window after watcher installation and drain.
            watcher.consume_changes()
        try:
            report = validate_dataset(root, write_report=write_report)
            validation_error = None
        except Exception as error:  # integrity failures must never leave a previous PASS usable
            report = None
            validation_error = f"Current integrity validation could not complete: {error}"
        after_manifest, after_manifest_bytes = _read_manifest(root)
        changed = (
            watcher.consume_changes()
            if watcher is not None
            else before_state != _validation_state_key(root, after_manifest)
        )
        if before_manifest_bytes != after_manifest_bytes:
            changed = True
        if not changed:
            manifest = after_manifest
            manifest_bytes = after_manifest_bytes
            fallback_state = _validation_state_key(root, manifest) if watcher is None else None
            stable = True
            break
        if attempt + 1 == _MAX_SNAPSHOT_ATTEMPTS:
            validation_error = (
                "Dataset files changed repeatedly while integrity validation was running; "
                "inspect again after writes have stopped."
            )
            report = None

    if not stable and validation_error is None:
        validation_error = "Current integrity validation could not establish a stable artifact snapshot."
    summary = _summary_from_report(root, manifest, report, validation_error)
    return _DatasetCacheEntry(
        root=root,
        watcher=watcher,
        fallback_state=fallback_state,
        summary=summary,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        report=report,
        token=next(_SNAPSHOT_TOKENS),
    )


def _entry_changed(entry: _DatasetCacheEntry) -> bool:
    if entry.watcher is not None:
        return entry.watcher.consume_changes()
    manifest, _ = _read_manifest(entry.root)
    return entry.fallback_state != _validation_state_key(entry.root, manifest)


def _close_entry(entry: _DatasetCacheEntry) -> None:
    if entry.watcher is not None:
        entry.watcher.close()


def _discard_entry(root: Path) -> None:
    entry = _DATASET_CACHE.pop(root, None)
    if entry is not None:
        _close_entry(entry)


def _refresh_historical_report(entry: _DatasetCacheEntry) -> None:
    saved = _saved_validation(entry.root)
    if saved != entry.summary.saved_validation:
        entry.summary = replace(entry.summary, saved_validation=saved)
        if entry.explorer is not None:
            entry.explorer = replace(entry.explorer, summary=entry.summary)


def _get_dataset_entry(root: Path, *, write_report: bool = False, force: bool = False) -> _DatasetCacheEntry:
    with _DATASET_CACHE_LOCK:
        entry = _DATASET_CACHE.get(root)
        if entry is not None and not force and not _entry_changed(entry):
            _DATASET_CACHE.move_to_end(root)
            _refresh_historical_report(entry)
            return entry
        if entry is not None:
            _discard_entry(root)

        entry = _new_cache_entry(root, write_report=write_report)
        _DATASET_CACHE[root] = entry
        while len(_DATASET_CACHE) > _DATASET_CACHE_LIMIT:
            _, evicted = _DATASET_CACHE.popitem(last=False)
            _close_entry(evicted)
        return entry


def _invalidate_dataset(root: Path) -> None:
    with _DATASET_CACHE_LOCK:
        _discard_entry(root)


def _safe_snapshot_asset(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative:
        raise DatasetSnapshotChangedError(f"Unsafe dataset artifact path {relative!r}")
    path = (root / Path(*posix.parts)).resolve()
    if root not in path.parents:
        raise DatasetSnapshotChangedError(f"Dataset artifact path escapes its root: {relative!r}")
    return path


def _read_declared_artifact(entry: _DatasetCacheEntry, relative: str) -> bytes:
    checksums = entry.manifest.get("asset_checksums")
    expected = checksums.get(relative) if isinstance(checksums, dict) else None
    if not isinstance(expected, str):
        raise DatasetSnapshotChangedError(f"Artifact {relative!r} is not declared by the validated manifest")
    path = _safe_snapshot_asset(entry.root, relative)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise DatasetSnapshotChangedError(f"Validated artifact {relative!r} is no longer readable: {error}") from error
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise DatasetSnapshotChangedError(
            f"Dataset artifact {relative!r} changed after validation; reload and validate the dataset"
        )
    return data


def _load_explorer_snapshot(entry: _DatasetCacheEntry) -> DatasetExplorer:
    if entry.manifest_bytes is None:
        raise DatasetSnapshotChangedError("The validated dataset manifest is not available")
    try:
        if (entry.root / MANIFEST_FILENAME).read_bytes() != entry.manifest_bytes:
            raise DatasetSnapshotChangedError("manifest.json changed after validation; reload the dataset")
    except OSError as error:
        raise DatasetSnapshotChangedError(f"manifest.json is no longer readable: {error}") from error

    class_path = f"{VOCABULARY_DIR}/{CLASS_VOCABULARY_FILENAME}"
    category_path = f"{VOCABULARY_DIR}/{DETECTOR_VOCABULARY_FILENAME}"
    class_data = _read_declared_artifact(entry, class_path)
    category_data = _read_declared_artifact(entry, category_path)
    try:
        classes = _json_object_from_bytes(class_data, entry.root / class_path)["classes"]
        categories = _json_object_from_bytes(category_data, entry.root / category_path)["categories"]
    except (KeyError, TypeError) as error:
        raise DatasetSnapshotChangedError(f"Validated vocabulary structure changed: {error}") from error

    samples_data = _read_declared_artifact(entry, SAMPLES_FILENAME)
    records = load_records_bytes(samples_data, source=str(entry.root / SAMPLES_FILENAME))
    return DatasetExplorer(
        summary=entry.summary,
        records=tuple(records),
        classes=tuple(classes),
        detector_categories=tuple(categories),
        manifest=deepcopy(entry.manifest),
        snapshot_token=entry.token,
    )


def validate_generation_request(config: DatasetConfig, output_path: Path) -> dict[str, Any]:
    """Validate resolved settings and output policy without generating any files."""

    config.validate(require_total=True)
    counts = config.allocate_type_counts()
    resolved_path = validate_output_path(output_path)
    return {
        "resolved_config": config.resolved_dict(),
        "resolved_type_counts": counts,
        "output_path": str(resolved_path),
    }


def generate_dataset(
    config: DatasetConfig,
    output_path: Path,
    *,
    source_repo: Path | None = None,
    progress: Progress | None = None,
) -> Path:
    """Generate a prepared dataset through the one shared application operation."""

    validate_generation_request(config, output_path)
    return prepare_dataset(config, output_path, source_repo=source_repo, progress=progress)


def validate_dataset_operation(dataset_path: Path, *, write_report: bool = True) -> ValidationReport:
    """Force the full integrity gate and replace any previously cached snapshot."""

    root = dataset_path.expanduser().resolve()
    entry = _get_dataset_entry(root, write_report=write_report, force=True)
    if entry.report is not None:
        return entry.report
    return ValidationReport(
        dataset_path=str(root),
        dataset_id=entry.summary.dataset_id if entry.summary.dataset_id != "unknown" else None,
        errors=[Diagnostic("snapshot_unavailable", entry.summary.validation_error or "Could not validate a stable dataset snapshot")],
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def inspect_dataset(dataset_path: Path) -> DatasetSummary:
    """Return the cached, validated dataset snapshot or a fail-closed status."""

    root = dataset_path.expanduser().resolve()
    return _get_dataset_entry(root).summary


def _unavailable_explorer(root: Path, message: str) -> DatasetExplorer:
    manifest, _ = _read_manifest(root)
    summary = _summary_from_report(root, manifest, None, message)
    return DatasetExplorer(summary, (), (), ())


def load_explorer(dataset_path: Path) -> DatasetExplorer:
    """Reuse one validated, parsed explorer bundle until watched artifacts change."""

    root = dataset_path.expanduser().resolve()
    for attempt in range(_MAX_SNAPSHOT_ATTEMPTS):
        with _DATASET_CACHE_LOCK:
            entry = _get_dataset_entry(root)
            if entry.summary.integrity_status != "current_pass":
                return DatasetExplorer(
                    entry.summary, (), (), (), manifest=deepcopy(entry.manifest), snapshot_token=entry.token
                )
            if _entry_changed(entry):
                _discard_entry(root)
                continue
            if entry.explorer is not None:
                return entry.explorer
            try:
                explorer = _load_explorer_snapshot(entry)
            except (OSError, ValueError, KeyError, TypeError, DatasetSnapshotChangedError):
                _discard_entry(root)
                continue
            if _entry_changed(entry):
                _discard_entry(root)
                continue
            entry.explorer = explorer
            return explorer
    return _unavailable_explorer(
        root,
        "Dataset files changed while loading explorer data; no consistent validated snapshot could be loaded. Retry after writes have stopped.",
    )


def load_verified_sample_image(explorer: DatasetExplorer, record: DatasetRecord):
    """Open image bytes only when image and annotation assets match this snapshot."""

    root = explorer.summary.path.resolve()
    with _DATASET_CACHE_LOCK:
        entry = _DATASET_CACHE.get(root)
        if entry is None or entry.token != explorer.snapshot_token or entry.summary.integrity_status != "current_pass":
            raise DatasetSnapshotChangedError("The explorer snapshot has expired; reload and validate the dataset")
        if _entry_changed(entry):
            _discard_entry(root)
            raise DatasetSnapshotChangedError("Dataset files changed after validation; reload the explorer")
        try:
            # These source annotations supplied the visible component records.
            # Verify their immutable identities before displaying those records.
            for relative in record.source_annotations.values():
                _read_declared_artifact(entry, relative)
            image_data = _read_declared_artifact(entry, record.image_path)
            if _entry_changed(entry):
                _discard_entry(root)
                raise DatasetSnapshotChangedError("Dataset files changed while opening the selected sample")
            from PIL import Image

            with Image.open(io.BytesIO(image_data)) as image:
                image.load()
                if image.size != (record.image_width, record.image_height) or image.mode != "RGB":
                    raise DatasetSnapshotChangedError(
                        f"Selected image {record.image_path!r} no longer matches its validated dimensions/color space"
                    )
                return image.copy()
        except (OSError, DatasetSnapshotChangedError):
            _discard_entry(root)
            raise
