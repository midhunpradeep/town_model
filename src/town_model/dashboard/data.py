"""Streamlit-independent dashboard read helpers for framework-neutral datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..application import DatasetExplorer, load_explorer
from ..dataset.schema import DatasetRecord


@dataclass(frozen=True, slots=True)
class DashboardLoadResult:
    explorer: DatasetExplorer | None
    error: str | None


def load_dashboard_data(path: Path | str) -> DashboardLoadResult:
    try:
        return DashboardLoadResult(load_explorer(Path(path)), None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return DashboardLoadResult(None, f"Could not load dataset at {path}: {error}")


def find_sample(explorer: DatasetExplorer, sample_id: str) -> DatasetRecord | None:
    return next((record for record in explorer.records if record.sample_id == sample_id), None)


def class_names_for_targets(targets: tuple[int, ...], classes: tuple[dict, ...]) -> list[str]:
    return [
        item["name"]
        for item, target in zip(classes, targets)
        if target and item.get("entity_kind") == "owl_class"
    ]
