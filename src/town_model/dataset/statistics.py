"""Deterministic dataset-level and per-class supervision summaries."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .schema import DatasetRecord


def _support_counts(records: list[DatasetRecord], classes: list[dict[str, Any]]) -> dict[str, Any]:
    class_count = len(classes)
    output: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        split_records = [record for record in records if record.split == split]
        pizza_positive = [0] * class_count
        pizza_negative = [0] * class_count
        pizza_unknown = [0] * class_count
        component_positive = [0] * class_count
        component_negative = [0] * class_count
        component_unknown = [0] * class_count

        for record in split_records:
            for class_id, (target, known) in enumerate(zip(record.pizza_targets, record.pizza_known_mask, strict=True)):
                if not known:
                    pizza_unknown[class_id] += 1
                elif target:
                    pizza_positive[class_id] += 1
                else:
                    pizza_negative[class_id] += 1
            for component in record.components:
                for class_id, (target, known) in enumerate(zip(component.ontology_targets, component.ontology_known_mask, strict=True)):
                    if not known:
                        component_unknown[class_id] += 1
                    elif target:
                        component_positive[class_id] += 1
                    else:
                        component_negative[class_id] += 1

        output[split] = {
            "pizza_instances": len(split_records),
            "component_instances": sum(len(record.components) for record in split_records),
            "classes": [
                {
                    "class_id": item["class_id"],
                    "name": item["name"],
                    "trainable": item["trainable"],
                    "fixed_positive": item["fixed_positive"],
                    "unsupported": item["unsupported"],
                    "pizza": {
                        "positive": pizza_positive[item["class_id"]],
                        "known_negative": pizza_negative[item["class_id"]],
                        "unknown": pizza_unknown[item["class_id"]],
                    },
                    "component": {
                        "positive": component_positive[item["class_id"]],
                        "known_negative": component_negative[item["class_id"]],
                        "unknown": component_unknown[item["class_id"]],
                    },
                }
                for item in classes
            ],
        }
    return output


def build_statistics(records: list[DatasetRecord], classes: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts = Counter(record.generator_type for record in records)
    split_counts = Counter(record.split for record in records)
    generator_types_by_split = {
        split: dict(sorted(Counter(record.generator_type for record in records if record.split == split).items()))
        for split in ("train", "validation", "test")
    }
    return {
        "dataset_size": len(records),
        "split_sizes": {split: split_counts.get(split, 0) for split in ("train", "validation", "test")},
        "generator_type_counts": dict(sorted(type_counts.items())),
        "generator_type_by_split": generator_types_by_split,
        "class_support_by_split": _support_counts(records, classes),
    }


def support_warnings(statistics: dict[str, Any], classes: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for split, split_data in statistics["class_support_by_split"].items():
        for counts in split_data["classes"]:
            if counts["trainable"] and counts["pizza"]["positive"] == 0:
                warnings.append(
                    f"Pizza class {counts['name']} has no known positive example in {split}"
                )
            if counts["trainable"] and counts["component"]["positive"] == 0:
                warnings.append(
                    f"Component class {counts['name']} has no known positive example in {split}"
                )
    return warnings
