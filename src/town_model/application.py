"""Shared Phase 1 operations used by both the CLI and Streamlit dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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
    sample_count: int
    split_sizes: dict[str, int]
    generator_type_counts: dict[str, int]
    generator_type_by_split: dict[str, dict[str, int]]
    class_support_by_split: dict[str, Any]
    provenance: dict[str, Any]
    ontology: dict[str, Any]
    vocabularies: dict[str, Any]
    validation: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "split_sizes": self.split_sizes,
            "generator_type_counts": self.generator_type_counts,
            "generator_type_by_split": self.generator_type_by_split,
            "class_support_by_split": self.class_support_by_split,
            "provenance": self.provenance,
            "ontology": self.ontology,
            "vocabularies": self.vocabularies,
            "validation": self.validation,
        }


@dataclass(frozen=True, slots=True)
class DatasetExplorer:
    summary: DatasetSummary
    records: tuple[DatasetRecord, ...]
    classes: tuple[dict[str, Any], ...]
    detector_categories: tuple[dict[str, Any], ...]


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
    """Load the saved dataset contract and its latest validation diagnostics."""

    root = dataset_path.expanduser().resolve()
    manifest = _load_json(root / MANIFEST_FILENAME)
    statistics = manifest.get("statistics", {})
    validation_path = root / REPORT_DIR / VALIDATION_REPORT_FILENAME
    validation = _load_json(validation_path) if validation_path.is_file() else None
    return DatasetSummary(
        path=root,
        dataset_id=str(manifest["dataset_id"]),
        schema_version=str(manifest["schema_version"]),
        sample_count=int(manifest["sample_record_count"]),
        split_sizes=dict(statistics.get("split_sizes", {})),
        generator_type_counts=dict(statistics.get("generator_type_counts", {})),
        generator_type_by_split=dict(statistics.get("generator_type_by_split", {})),
        class_support_by_split=dict(statistics.get("class_support_by_split", {})),
        provenance=dict(manifest.get("source", {})),
        ontology=dict(manifest.get("ontology", {})),
        vocabularies=dict(manifest.get("vocabularies", {})),
        validation=validation,
    )


def load_explorer(dataset_path: Path) -> DatasetExplorer:
    """Load only framework-neutral records/vocabularies for the image explorer."""

    root = dataset_path.expanduser().resolve()
    summary = inspect_dataset(root)
    classes = _load_json(root / VOCABULARY_DIR / CLASS_VOCABULARY_FILENAME)
    categories = _load_json(root / VOCABULARY_DIR / DETECTOR_VOCABULARY_FILENAME)
    records = load_records(root / SAMPLES_FILENAME)
    return DatasetExplorer(
        summary=summary,
        records=tuple(records),
        classes=tuple(classes["classes"]),
        detector_categories=tuple(categories["categories"]),
    )
