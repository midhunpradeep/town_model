"""Shared Phase 1 operations used by both the CLI and Streamlit dashboard."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
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
from .dataset.schema import DatasetRecord, load_records
from .dataset.validation import ValidationReport, validate_dataset


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


_VALIDATION_CACHE: OrderedDict[tuple[Any, ...], ValidationReport] = OrderedDict()
_VALIDATION_CACHE_LIMIT = 8
_VALIDATION_CACHE_LOCK = threading.Lock()


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


def _current_validation(root: Path) -> tuple[ValidationReport | None, str | None]:
    """Validate changed artifacts once; reuse only while every asset's stat state is stable."""

    for attempt in range(2):
        try:
            manifest = _load_json(root / MANIFEST_FILENAME)
        except (OSError, json.JSONDecodeError, ValueError):
            manifest = None
        key = _validation_state_key(root, manifest)
        with _VALIDATION_CACHE_LOCK:
            cached = _VALIDATION_CACHE.get(key)
            if cached is not None:
                _VALIDATION_CACHE.move_to_end(key)
                return cached, None
        try:
            report = validate_dataset(root, write_report=False)
        except Exception as error:
            return None, f"Current integrity validation could not complete: {error}"
        try:
            current_manifest = _load_json(root / MANIFEST_FILENAME)
        except (OSError, json.JSONDecodeError, ValueError):
            current_manifest = None
        if key == _validation_state_key(root, current_manifest):
            with _VALIDATION_CACHE_LOCK:
                _VALIDATION_CACHE[key] = report
                _VALIDATION_CACHE.move_to_end(key)
                while len(_VALIDATION_CACHE) > _VALIDATION_CACHE_LIMIT:
                    _VALIDATION_CACHE.popitem(last=False)
            return report, None
        if attempt == 1:
            return None, "Dataset files changed repeatedly while integrity validation was running; inspect again."
    return None, "Current integrity validation could not establish a stable artifact snapshot."


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
    """Run the shared dataset integrity gate used by CLI and dashboard."""

    return validate_dataset(dataset_path, write_report=write_report)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def inspect_dataset(dataset_path: Path) -> DatasetSummary:
    """Summarize only current validated support; preserve saved validation as historical."""

    root = dataset_path.expanduser().resolve()
    try:
        manifest = _load_json(root / MANIFEST_FILENAME)
    except (OSError, json.JSONDecodeError, ValueError):
        manifest = {}
    report, validation_error = _current_validation(root)
    integrity_status = (
        "unavailable" if report is None else "current_pass" if report.valid else "current_fail"
    )
    current = deepcopy(report.to_dict()) if report is not None else None

    validation_path = root / REPORT_DIR / VALIDATION_REPORT_FILENAME
    try:
        saved_validation = _load_json(validation_path) if validation_path.is_file() else None
    except (OSError, json.JSONDecodeError, ValueError):
        saved_validation = None

    statistics = deepcopy(report.statistics) if integrity_status == "current_pass" and report is not None else {}
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
        integrity_status=integrity_status,
        validation=current,
        saved_validation=saved_validation,
        validation_error=validation_error,
    )


def load_explorer(dataset_path: Path) -> DatasetExplorer:
    """Load only framework-neutral records/vocabularies for the image explorer."""

    root = dataset_path.expanduser().resolve()
    summary = inspect_dataset(root)
    if summary.integrity_status != "current_pass":
        return DatasetExplorer(summary, (), (), ())
    classes = _load_json(root / VOCABULARY_DIR / CLASS_VOCABULARY_FILENAME)
    categories = _load_json(root / VOCABULARY_DIR / DETECTOR_VOCABULARY_FILENAME)
    records = load_records(root / SAMPLES_FILENAME)
    return DatasetExplorer(
        summary=summary,
        records=tuple(records),
        classes=tuple(classes["classes"]),
        detector_categories=tuple(categories["categories"]),
    )
