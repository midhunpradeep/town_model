"""Versioned, model-independent JSON Lines dataset wire schema."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .constants import SCHEMA_VERSION, SAMPLES_FILENAME, SPLIT_NAMES


@dataclass(frozen=True, slots=True)
class ComponentRecord:
    component_id: str
    detector_category_id: int
    detector_category: str
    bbox_xyxy_original: tuple[float, float, float, float]
    ontology_targets: tuple[int, ...]
    ontology_known_mask: tuple[int, ...]
    annotation_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "detector_category_id": self.detector_category_id,
            "detector_category": self.detector_category,
            "bbox_xyxy_original": list(self.bbox_xyxy_original),
            "ontology_targets": list(self.ontology_targets),
            "ontology_known_mask": list(self.ontology_known_mask),
            "annotation_path": self.annotation_path,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ComponentRecord":
        for key in ("ontology_targets", "ontology_known_mask"):
            vector = value.get(key)
            if not isinstance(vector, list) or any(type(item) is not int or item not in (0, 1) for item in vector):
                raise ValueError(f"{key} must be an integer 0/1 vector")
        bbox = value.get("bbox_xyxy_original")
        if not isinstance(bbox, list) or len(bbox) != 4 or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
            for item in bbox
        ):
            raise ValueError("bbox_xyxy_original must contain four numeric coordinates")
        category_id = value.get("detector_category_id")
        if type(category_id) is not int or category_id <= 0:
            raise ValueError("detector_category_id must be a positive integer")
        expected_keys = {
            "component_id", "detector_category_id", "detector_category", "bbox_xyxy_original",
            "ontology_targets", "ontology_known_mask", "annotation_path",
        }
        if set(value) != expected_keys:
            raise ValueError(f"component fields must be exactly {sorted(expected_keys)}")
        for key in ("component_id", "detector_category", "annotation_path"):
            if not isinstance(value.get(key), str):
                raise ValueError(f"{key} must be a string")
        return cls(
            component_id=value["component_id"],
            detector_category_id=category_id,
            detector_category=value["detector_category"],
            bbox_xyxy_original=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
            ontology_targets=tuple(value["ontology_targets"]),
            ontology_known_mask=tuple(value["ontology_known_mask"]),
            annotation_path=value["annotation_path"],
        )


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    sample_id: str
    split: str
    generator_type: str
    image_path: str
    image_width: int
    image_height: int
    pizza_targets: tuple[int, ...]
    pizza_known_mask: tuple[int, ...]
    pizza_source_labels: tuple[str, ...]
    components: tuple[ComponentRecord, ...]
    audit: dict[str, Any]
    source_annotations: dict[str, str]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "split": self.split,
            "generator_type": self.generator_type,
            "image": {
                "path": self.image_path,
                "width": self.image_width,
                "height": self.image_height,
                "color_space": "RGB",
            },
            "pizza": {
                "ontology_targets": list(self.pizza_targets),
                "ontology_known_mask": list(self.pizza_known_mask),
                "source_inferred_labels": list(self.pizza_source_labels),
            },
            "components": [component.to_dict() for component in self.components],
            "audit": self.audit,
            "source_annotations": self.source_annotations,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetRecord":
        image = value["image"]
        pizza = value["pizza"]
        expected_keys = {
            "schema_version", "sample_id", "split", "generator_type", "image", "pizza",
            "components", "audit", "source_annotations",
        }
        if set(value) != expected_keys:
            raise ValueError(f"sample fields must be exactly {sorted(expected_keys)}")
        if set(image) != {"path", "width", "height", "color_space"}:
            raise ValueError("image fields must be path, width, height, color_space")
        if set(pizza) != {"ontology_targets", "ontology_known_mask", "source_inferred_labels"}:
            raise ValueError("pizza fields must be ontology_targets, ontology_known_mask, source_inferred_labels")
        if image.get("color_space") != "RGB":
            raise ValueError("image.color_space must be RGB")
        for key in ("sample_id", "split", "generator_type", "schema_version"):
            if not isinstance(value.get(key), str):
                raise ValueError(f"{key} must be a string")
        if not isinstance(image.get("path"), str):
            raise ValueError("image.path must be a string")
        for key in ("width", "height"):
            if type(image.get(key)) is not int or image[key] <= 0:
                raise ValueError(f"image.{key} must be a positive integer")
        for key in ("ontology_targets", "ontology_known_mask"):
            vector = pizza.get(key)
            if not isinstance(vector, list) or any(type(item) is not int or item not in (0, 1) for item in vector):
                raise ValueError(f"pizza.{key} must be an integer 0/1 vector")
        if not isinstance(pizza.get("source_inferred_labels"), list) or not all(
            isinstance(item, str) for item in pizza["source_inferred_labels"]
        ):
            raise ValueError("pizza.source_inferred_labels must be an array of strings")
        annotations = value.get("source_annotations")
        if not isinstance(value.get("components"), list) or not isinstance(value.get("audit"), dict) or not isinstance(annotations, dict):
            raise ValueError("components, audit, and source_annotations have invalid wire types")
        if any(not isinstance(key, str) or not isinstance(path, str) for key, path in annotations.items()):
            raise ValueError("source_annotations keys and paths must be strings")
        return cls(
            schema_version=value["schema_version"],
            sample_id=value["sample_id"],
            split=value["split"],
            generator_type=value["generator_type"],
            image_path=image["path"],
            image_width=image["width"],
            image_height=image["height"],
            pizza_targets=tuple(pizza["ontology_targets"]),
            pizza_known_mask=tuple(pizza["ontology_known_mask"]),
            pizza_source_labels=tuple(str(v) for v in pizza["source_inferred_labels"]),
            components=tuple(ComponentRecord.from_dict(item) for item in value["components"]),
            audit=dict(value["audit"]),
            source_annotations=dict(annotations),
        )


def write_records(records: Iterable[DatasetRecord], path: Path) -> None:
    """Write deterministic UTF-8 JSONL records, one immutable sample per line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
    temporary.replace(path)


def load_records(path: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record must be a JSON object")
                record = DatasetRecord.from_dict(value)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid JSONL record at {path}:{line_number}: {error}") from error
            if record.schema_version != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported record schema {record.schema_version!r} at {path}:{line_number}; "
                    f"expected {SCHEMA_VERSION!r}"
                )
            if record.split not in SPLIT_NAMES:
                raise ValueError(f"Invalid split {record.split!r} at {path}:{line_number}")
            records.append(record)
    return records


def write_split_manifest(records: Iterable[DatasetRecord], path: Path) -> None:
    sample_ids = [record.sample_id for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            {"schema_version": SCHEMA_VERSION, "sample_ids": sample_ids},
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    temporary.replace(path)
