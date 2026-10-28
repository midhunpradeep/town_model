"""Dataset integrity gate and split-aware positive/negative/unknown reporting."""

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from ..config import DatasetConfig
from .constants import (
    AUDIT_DIR,
    BASE_CATEGORIES,
    CLASS_VOCABULARY_FILENAME,
    DETECTOR_CATEGORIES,
    DETECTOR_VOCABULARY_FILENAME,
    FIXED_POSITIVE_CLASS,
    FLAG_INDIVIDUALS,
    GENERATOR_COMMIT,
    GENERATOR_LICENSE,
    GENERATOR_REPOSITORY,
    IMAGE_DIR,
    LABEL_DIR,
    MANIFEST_FILENAME,
    ONTOLOGY_DIR,
    ONTOLOGY_SHA256,
    RAW_MASTER_CANONICALIZATION,
    RAW_MASTER_FILENAME,
    REPORT_DIR,
    SAMPLES_FILENAME,
    SCHEMA_VERSION,
    SPLIT_DIR,
    SPLIT_NAMES,
    TOPPING_COLUMNS,
    VALIDATION_REPORT_FILENAME,
    VALIDATOR_VERSION,
    VOCABULARY_DIR,
)
from .labels import infer_pizzaiolo_label
from .ontology import (
    OntologySnapshot,
    build_class_vocabulary,
    build_detector_vocabulary,
)
from .schema import DatasetRecord, load_records
from .statistics import build_statistics, support_warnings
from .splitting import assign_splits


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    message: str
    sample_id: str | None = None
    path: str | None = None
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "code": self.code,
                "message": self.message,
                "sample_id": self.sample_id,
                "path": self.path,
                "field": self.field,
            }.items()
            if value is not None
        }


@dataclass(slots=True)
class ValidationReport:
    dataset_path: str
    dataset_id: str | None
    errors: list[Diagnostic] = field(default_factory=list)
    warnings: list[Diagnostic] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)
    report_id: str | None = None

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        result = {
            "report_schema_version": "1.0.0",
            "validator_version": VALIDATOR_VERSION,
            "dataset_path": self.dataset_path,
            "dataset_id": self.dataset_id,
            "valid": self.valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "checks": self.checks,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "statistics": self.statistics,
        }
        if self.report_id:
            result["report_id"] = self.report_id
        return result


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_asset_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative:
        raise ValueError(f"Unsafe dataset-relative path: {relative!r}")
    candidate = (root / Path(*posix.parts)).resolve()
    if root.resolve() not in candidate.parents:
        raise ValueError(f"Path escapes dataset directory: {relative!r}")
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    temporary.replace(path)


def _vector_for(names: set[str], classes: list[dict[str, Any]]) -> list[int]:
    return [int(item["name"] in names) for item in classes]


def _known_mask(classes: list[dict[str, Any]]) -> list[int]:
    return [int(not item["unsupported"]) for item in classes]


def _target_names(vector: tuple[int, ...], classes: list[dict[str, Any]]) -> set[str]:
    return {item["name"] for item, value in zip(classes, vector) if value == 1}


def _add_issue(
    issues: list[Diagnostic], code: str, message: str, *, sample_id: str | None = None,
    path: str | None = None, field: str | None = None,
) -> None:
    issues.append(Diagnostic(code, message, sample_id, path, field))


def _load_samples(path: Path) -> list[DatasetRecord]:
    """Load each JSONL record strictly; preserve line-number context on errors."""

    records: list[DatasetRecord] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("sample must be a JSON object")
                if value.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
                image = value.get("image")
                pizza = value.get("pizza")
                if not isinstance(image, dict) or not isinstance(pizza, dict):
                    raise ValueError("image and pizza must be JSON objects")
                for label, sequence in (
                    ("pizza.ontology_targets", pizza.get("ontology_targets")),
                    ("pizza.ontology_known_mask", pizza.get("ontology_known_mask")),
                ):
                    if not isinstance(sequence, list) or any(type(item) is not int or item not in (0, 1) for item in sequence):
                        raise ValueError(f"{label} must be an array of integer 0/1 values")
                for key in ("width", "height"):
                    if type(image.get(key)) is not int or image[key] <= 0:
                        raise ValueError(f"image.{key} must be a positive integer")
                components = value.get("components")
                if not isinstance(components, list):
                    raise ValueError("components must be a JSON array")
                for index, component in enumerate(components):
                    if not isinstance(component, dict):
                        raise ValueError(f"components[{index}] must be a JSON object")
                    for key in ("ontology_targets", "ontology_known_mask"):
                        vector = component.get(key)
                        if not isinstance(vector, list) or any(type(item) is not int or item not in (0, 1) for item in vector):
                            raise ValueError(f"components[{index}].{key} must be an array of integer 0/1 values")
                    bbox = component.get("bbox_xyxy_original")
                    if not isinstance(bbox, list) or len(bbox) != 4 or any(
                        isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
                        for item in bbox
                    ):
                        raise ValueError(f"components[{index}].bbox_xyxy_original must contain four finite numbers")
                records.append(DatasetRecord.from_dict(value))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid sample record at {path}:{line_number}: {error}") from error
    return records


def _validate_manifest_identity(manifest: dict[str, Any], errors: list[Diagnostic], path: Path) -> None:
    manifest_id = manifest.get("manifest_id")
    body = {key: value for key, value in manifest.items() if key != "manifest_id"}
    expected = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if manifest_id != expected:
        _add_issue(errors, "manifest_hash", "Manifest ID does not match canonical manifest contents", path=str(path), field="manifest_id")

    assets = manifest.get("asset_checksums")
    config = manifest.get("resolved_config")
    source = manifest.get("source")
    ontology = manifest.get("ontology")
    if not isinstance(assets, dict) or not isinstance(config, dict) or not isinstance(source, dict) or not isinstance(ontology, dict):
        _add_issue(errors, "manifest_fields", "Manifest is missing asset_checksums, resolved_config, source, or ontology", path=str(path))
        return
    identity_payload = {
        "schema_version": manifest.get("schema_version"),
        "resolved_config": config,
        "generator_commit": source.get("generator_commit"),
        "ontology_sha256": ontology.get("sha256"),
        "asset_checksums": assets,
    }
    expected_dataset_id = "pizza-" + hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()[:20]
    if manifest.get("dataset_id") != expected_dataset_id:
        _add_issue(errors, "dataset_id", "Dataset ID does not match configuration, provenance, and asset hashes", path=str(path), field="dataset_id")


def _validate_vocabulary(
    root: Path,
    manifest: dict[str, Any],
    errors: list[Diagnostic],
) -> tuple[OntologySnapshot | None, dict[str, Any] | None, dict[str, Any] | None]:
    ontology_path = root / ONTOLOGY_DIR / "pizzaiolo.xml"
    try:
        ontology = OntologySnapshot.load(ontology_path, require_pinned_hash=True)
        ontology_info = manifest.get("ontology") if isinstance(manifest.get("ontology"), dict) else {}
        if ontology.sha256 != ontology_info.get("sha256"):
            _add_issue(errors, "ontology_hash", "Ontology hash differs from manifest provenance", path=str(ontology_path), field="ontology.sha256")
        expected_classes = build_class_vocabulary(ontology)
        expected_categories = build_detector_vocabulary(ontology)
        class_path = root / VOCABULARY_DIR / CLASS_VOCABULARY_FILENAME
        category_path = root / VOCABULARY_DIR / DETECTOR_VOCABULARY_FILENAME
        actual_classes = _read_json(class_path)
        actual_categories = _read_json(category_path)
        if actual_classes != expected_classes:
            _add_issue(errors, "class_vocabulary", "Canonical ontology vocabulary is nondeterministic or differs from the pinned ontology rules", path=str(class_path))
        if actual_categories != expected_categories:
            _add_issue(errors, "detector_vocabulary", "Detector categories differ from the pinned generator vocabulary", path=str(category_path))
        expected_classes_hash = sha256_file(class_path)
        expected_categories_hash = sha256_file(category_path)
        vocab_manifest = manifest.get("vocabularies", {})
        if vocab_manifest.get("ontology_classes_sha256") != expected_classes_hash:
            _add_issue(errors, "class_vocabulary_hash", "Class vocabulary hash differs from manifest", path=str(class_path))
        if vocab_manifest.get("detector_categories_sha256") != expected_categories_hash:
            _add_issue(errors, "detector_vocabulary_hash", "Detector vocabulary hash differs from manifest", path=str(category_path))
        if "Germany" in {item.get("name") for item in actual_categories.get("categories", [])}:
            _add_issue(errors, "unsupported_flag", "Germany is an ontology individual but is not a rendered detector category", path=str(category_path), field="categories")
        if any(item.get("entity_kind") != "owl_class" for item in actual_classes.get("classes", [])):
            _add_issue(errors, "class_individual_confusion", "Ontology vocabulary contains a non-class entity", path=str(class_path))
        return ontology, actual_classes, actual_categories
    except (OSError, ET.ParseError, KeyError, ValueError, json.JSONDecodeError) as error:
        _add_issue(errors, "ontology_or_vocabulary", str(error), path=str(ontology_path))
        return None, None, None


def _validate_assets(root: Path, manifest: dict[str, Any], errors: list[Diagnostic]) -> None:
    checksums = manifest.get("asset_checksums")
    if not isinstance(checksums, dict):
        _add_issue(errors, "asset_manifest", "asset_checksums must be a JSON object", path=str(root / MANIFEST_FILENAME))
        return
    for relative, expected_hash in checksums.items():
        try:
            asset = _safe_asset_path(root, str(relative))
        except ValueError as error:
            _add_issue(errors, "unsafe_asset_path", str(error), path=str(root / MANIFEST_FILENAME))
            continue
        if not asset.is_file():
            _add_issue(errors, "missing_asset", f"Referenced immutable dataset asset is missing: {relative}", path=relative)
        elif sha256_file(asset) != expected_hash:
            _add_issue(errors, "asset_checksum", f"SHA-256 mismatch for dataset asset {relative}", path=relative)


def _validate_record(
    root: Path,
    record: DatasetRecord,
    records_path: Path,
    ontology: OntologySnapshot,
    classes: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    sample_ids: set[str],
    component_ids: set[str],
    errors: list[Diagnostic],
) -> None:
    sample_id = record.sample_id
    if not sample_id or sample_id in sample_ids:
        _add_issue(errors, "duplicate_sample_id", f"Sample ID is empty or duplicated: {sample_id!r}", sample_id=sample_id, path=str(records_path), field="sample_id")
    sample_ids.add(sample_id)
    if record.schema_version != SCHEMA_VERSION:
        _add_issue(errors, "schema_version", f"Expected sample schema {SCHEMA_VERSION}, found {record.schema_version}", sample_id=sample_id, path=str(records_path), field="schema_version")
    if record.split not in SPLIT_NAMES:
        _add_issue(errors, "split_value", f"Unknown split {record.split!r}", sample_id=sample_id, path=str(records_path), field="split")
    raw_generator = record.audit.get("raw_generator", {})
    if not isinstance(raw_generator, dict):
        _add_issue(errors, "raw_metadata", "audit.raw_generator must be a JSON object", sample_id=sample_id, path=str(records_path), field="audit.raw_generator")
        raw_generator = {}

    if len(record.pizza_targets) != len(classes) or len(record.pizza_known_mask) != len(classes):
        _add_issue(errors, "target_shape", f"Pizza targets/mask must each have {len(classes)} entries", sample_id=sample_id, path=str(records_path), field="pizza.ontology_targets")
    else:
        inference_row = {
            "hasTopping": raw_generator.get("has_topping"),
            "hasBase": raw_generator.get("has_base"),
            "hasCountryOfOrigin": raw_generator.get("country_of_origin"),
            **raw_generator.get("topping_counts", {}),
        }
        try:
            expected_source_labels = infer_pizzaiolo_label(inference_row)
            if list(record.pizza_source_labels) != expected_source_labels:
                _add_issue(errors, "source_labels", "Stored source-inferred image labels differ from audited recipe predicates", sample_id=sample_id, path=str(records_path), field="pizza.source_inferred_labels")
            expected_names: set[str] = set()
            for name in expected_source_labels:
                expected_names.update(ontology.ancestors(name))
            expected_targets = _vector_for(expected_names, classes)
            expected_mask = _known_mask(classes)
            if list(record.pizza_targets) != expected_targets:
                _add_issue(errors, "pizza_targets", "Pizza ontology targets do not match the source labels and pinned superclass closure", sample_id=sample_id, path=str(records_path), field="pizza.ontology_targets")
            if list(record.pizza_known_mask) != expected_mask:
                _add_issue(errors, "pizza_known_mask", "Pizza known-mask violates the closed-world synthetic-row contract", sample_id=sample_id, path=str(records_path), field="pizza.ontology_known_mask")
            vegetarian = "VegetarianPizza" in expected_source_labels
            nonvegetarian = "NonVegetarianPizza" in expected_source_labels
            if vegetarian == nonvegetarian:
                _add_issue(errors, "vegetarian_exclusivity", "Exactly one of VegetarianPizza and NonVegetarianPizza must be positive", sample_id=sample_id, path=str(records_path), field="pizza.source_inferred_labels")
            named = {
                "American", "AmericanHot", "Cajun", "Capricciosa", "Fiorentina", "FourSeasons",
                "Giardiniera", "LaReine", "Mushroom", "Napoletana", "Parmense", "Siciliana",
                "SloppyGiuseppe", "Soho", "Veneziana",
            }
            if named.intersection(expected_source_labels) and "NamedPizza" not in expected_names:
                _add_issue(errors, "named_pizza_ancestry", "Named pizza target is missing NamedPizza ancestry", sample_id=sample_id, path=str(records_path), field="pizza.ontology_targets")
            if record.pizza_targets and record.pizza_targets[next(i for i, item in enumerate(classes) if item["name"] == FIXED_POSITIVE_CLASS)] != 1:
                _add_issue(errors, "domain_thing", "DomainThing must be a fixed positive pizza target", sample_id=sample_id, path=str(records_path), field="pizza.ontology_targets")
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, StopIteration) as error:
            _add_issue(errors, "raw_metadata", f"Cannot validate raw generator facts: {error}", sample_id=sample_id, path=str(records_path), field="audit.raw_generator")

    try:
        image_relative = record.image_path
        image_path = _safe_asset_path(root, image_relative)
        if not image_path.is_file():
            _add_issue(errors, "missing_image", f"Image file does not exist: {image_relative}", sample_id=sample_id, path=image_relative, field="image.path")
        else:
            with Image.open(image_path) as image:
                if image.mode != "RGB":
                    _add_issue(errors, "image_color_space", f"Expected RGB image, found mode {image.mode}", sample_id=sample_id, path=image_relative, field="image.color_space")
                if image.size != (record.image_width, record.image_height):
                    _add_issue(errors, "image_dimensions", f"Stored dimensions {(record.image_width, record.image_height)} differ from image {image.size}", sample_id=sample_id, path=image_relative, field="image.width,height")
    except (OSError, ValueError) as error:
        _add_issue(errors, "image_path", str(error), sample_id=sample_id, path=record.image_path, field="image.path")

    audit = raw_generator
    required_audit_fields = {
        "source_ref", "source_type", "generator_type", "has_topping", "topping_counts",
        "has_base", "country_of_origin", "original_annotation_filenames",
    }
    missing_audit_fields = required_audit_fields - set(audit)
    if missing_audit_fields:
        _add_issue(errors, "incomplete_raw_metadata", f"Required raw generator fields are missing: {sorted(missing_audit_fields)}", sample_id=sample_id, path=str(records_path), field="audit.raw_generator")
    if audit.get("source_ref") != sample_id:
        _add_issue(errors, "sample_id_scheme", "sample_id must preserve the original generator ref", sample_id=sample_id, path=str(records_path), field="audit.raw_generator.source_ref")
    source_type = audit.get("source_type")
    expected_generator_type = "random" if source_type in (None, "unspecified") else source_type
    if audit.get("generator_type") != expected_generator_type or record.generator_type != expected_generator_type:
        _add_issue(errors, "generator_type_metadata", "Sample generator_type does not match the recorded original type", sample_id=sample_id, path=str(records_path), field="generator_type")
    counts = audit.get("topping_counts", {})
    has_topping = audit.get("has_topping", [])
    if not isinstance(counts, dict) or not isinstance(has_topping, list):
        _add_issue(errors, "topping_metadata", "topping_counts and has_topping must be an object/list", sample_id=sample_id, path=str(records_path), field="audit.raw_generator")
        counts = {}
        has_topping = []
    if not all(isinstance(name, str) for name in has_topping):
        _add_issue(errors, "malformed_has_topping", "hasTopping entries must be strings", sample_id=sample_id, path=str(records_path), field="audit.raw_generator.has_topping")
        has_topping = [name for name in has_topping if isinstance(name, str)]
    unknown_count_names = set(counts) - set(TOPPING_COLUMNS)
    if unknown_count_names:
        _add_issue(errors, "unsupported_topping_count", f"Unknown topping-count fields {sorted(unknown_count_names)}", sample_id=sample_id, path=str(records_path), field="audit.raw_generator.topping_counts")
    if len(has_topping) != len(set(has_topping)):
        _add_issue(errors, "duplicate_has_topping", "hasTopping must not list a topping category more than once", sample_id=sample_id, path=str(records_path), field="audit.raw_generator.has_topping")
    if set(has_topping) - set(TOPPING_COLUMNS):
        _add_issue(errors, "unsupported_has_topping", f"hasTopping contains unsupported categories {sorted(set(has_topping) - set(TOPPING_COLUMNS))}", sample_id=sample_id, path=str(records_path), field="audit.raw_generator.has_topping")
    valid_counts = True
    for topping in TOPPING_COLUMNS:
        count = counts.get(topping)
        if type(count) is not int or count < 0:
            valid_counts = False
            _add_issue(errors, "invalid_topping_count", f"{topping} count must be a nonnegative integer, got {count!r}", sample_id=sample_id, path=str(records_path), field=f"audit.raw_generator.topping_counts.{topping}")
    if valid_counts:
        positive_counts = {name for name, count in counts.items() if count > 0}
        if positive_counts != set(has_topping):
            _add_issue(errors, "topping_count_mismatch", f"Positive topping counts {sorted(positive_counts)} disagree with hasTopping {sorted(set(has_topping))}", sample_id=sample_id, path=str(records_path), field="audit.raw_generator.has_topping")
    base = audit.get("has_base")
    if base not in BASE_CATEGORIES:
        _add_issue(errors, "invalid_base", f"Required hasBase must be one of {BASE_CATEGORIES}, got {base!r}", sample_id=sample_id, path=str(records_path), field="audit.raw_generator.has_base")
    origin = audit.get("country_of_origin")
    if origin is not None and origin not in FLAG_INDIVIDUALS:
        _add_issue(errors, "unsupported_origin", f"Origin {origin!r} is not one of the four rendered flags {FLAG_INDIVIDUALS}; Germany exists only as an ontology individual", sample_id=sample_id, path=str(records_path), field="audit.raw_generator.country_of_origin")

    by_category: dict[str, list[Any]] = {}
    category_ids = {item["name"]: item["category_id"] for item in categories}
    class_by_id = {item["class_id"]: item for item in classes}
    occurrence_counts: dict[str, int] = {}
    for component in record.components:
        component_id = component.component_id
        if not component_id or component_id in component_ids:
            _add_issue(errors, "duplicate_component_id", f"Component occurrence ID is empty or duplicated: {component_id!r}", sample_id=sample_id, path=str(records_path), field="components.component_id")
        component_ids.add(component_id)
        category = component.detector_category
        by_category.setdefault(category, []).append(component)
        if category not in category_ids or category not in DETECTOR_CATEGORIES:
            _add_issue(errors, "unsupported_detector_category", f"Unsupported detector category {category!r}; country individuals are restricted to America, England, France, Italy", sample_id=sample_id, path=str(records_path), field="components.detector_category")
            continue
        if component.detector_category_id != category_ids[category]:
            _add_issue(errors, "detector_category_id", f"Category ID {component.detector_category_id} does not match {category!r} ID {category_ids[category]}", sample_id=sample_id, path=str(records_path), field="components.detector_category_id")
        count = occurrence_counts.get(category, 0)
        expected_id = f"{sample_id}/{category}/occurrence_{count:03d}"
        occurrence_counts[category] = count + 1
        if component_id != expected_id:
            _add_issue(errors, "component_id_scheme", f"Expected stable occurrence ID {expected_id}, got {component_id}", sample_id=sample_id, path=str(records_path), field="components.component_id")
        x1, y1, x2, y2 = component.bbox_xyxy_original
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > record.image_width or y2 > record.image_height:
            _add_issue(errors, "invalid_bbox", f"Invalid original-image xyxy box {component.bbox_xyxy_original} for {record.image_width}x{record.image_height} image", sample_id=sample_id, path=component.annotation_path, field="components.bbox_xyxy_original")
        if len(component.ontology_targets) != len(classes) or len(component.ontology_known_mask) != len(classes):
            _add_issue(errors, "component_target_shape", f"Component targets/mask must each have {len(classes)} entries", sample_id=sample_id, path=str(records_path), field=f"components[{component_id}].ontology_targets")
            continue
        try:
            expected_names = set(ontology.component_class_targets(category))
            expected_targets = _vector_for(expected_names, classes)
            if list(component.ontology_targets) != expected_targets:
                actual_names = _target_names(component.ontology_targets, classes)
                _add_issue(errors, "component_targets", f"Ontology target mismatch for {category}: expected {sorted(expected_names)}, found {sorted(actual_names)}", sample_id=sample_id, path=str(records_path), field=f"components[{component_id}].ontology_targets")
            if list(component.ontology_known_mask) != _known_mask(classes):
                _add_issue(errors, "component_known_mask", "Component known-mask violates the validated complete visible-component contract", sample_id=sample_id, path=str(records_path), field=f"components[{component_id}].ontology_known_mask")
            if component.annotation_path not in record.source_annotations.values():
                _add_issue(errors, "annotation_reference", f"Component annotation path {component.annotation_path!r} is not referenced by its sample", sample_id=sample_id, path=component.annotation_path, field=f"components[{component_id}].annotation_path")
        except ValueError as error:
            _add_issue(errors, "component_identity", str(error), sample_id=sample_id, path=str(records_path), field=f"components[{component_id}].detector_category")

    expected_component_categories = set(has_topping) | ({base} if base in BASE_CATEGORIES else set()) | ({origin} if origin in FLAG_INDIVIDUALS else set())
    for topping in TOPPING_COLUMNS:
        expected_count = counts.get(topping)
        if type(expected_count) is int and expected_count >= 0 and len(by_category.get(topping, [])) != expected_count:
            _add_issue(errors, "component_count_mismatch", f"{topping} metadata count {expected_count} disagrees with {len(by_category.get(topping, []))} annotated occurrences", sample_id=sample_id, path=str(records_path), field=f"components.{topping}")
    if base in BASE_CATEGORIES and len(by_category.get(base, [])) != 1:
        _add_issue(errors, "base_annotation_count", f"Required base {base} must have exactly one annotation", sample_id=sample_id, path=str(records_path), field="components")
    if origin in FLAG_INDIVIDUALS and len(by_category.get(origin, [])) != 1:
        _add_issue(errors, "flag_annotation_count", f"Origin flag {origin} must have exactly one annotation", sample_id=sample_id, path=str(records_path), field="components")
    if set(by_category) != expected_component_categories:
        _add_issue(errors, "component_category_metadata", f"Annotated categories {sorted(by_category)} disagree with complete source metadata {sorted(expected_component_categories)}", sample_id=sample_id, path=str(records_path), field="components")
    if origin is None and any(flag in by_category for flag in FLAG_INDIVIDUALS):
        _add_issue(errors, "missing_origin_flag", "A blank origin is a known no-flag observation; no country flag annotation is allowed", sample_id=sample_id, path=str(records_path), field="audit.raw_generator.country_of_origin")
    if origin is not None and set(by_category).intersection(FLAG_INDIVIDUALS) != {origin}:
        _add_issue(errors, "origin_flag_mismatch", f"Origin {origin!r} must have exactly its matching flag annotation", sample_id=sample_id, path=str(records_path), field="audit.raw_generator.country_of_origin")

    required_annotation_refs = {"boxes", "contours", "segmentation"}
    if set(record.source_annotations) != required_annotation_refs:
        _add_issue(errors, "annotation_reference", f"source_annotations must reference exactly {sorted(required_annotation_refs)}", sample_id=sample_id, path=str(records_path), field="source_annotations")
    for name, relative in record.source_annotations.items():
        try:
            source_path = _safe_asset_path(root, relative)
            if not source_path.is_file():
                _add_issue(errors, "missing_annotation_asset", f"Annotation source file is missing: {relative}", sample_id=sample_id, path=relative, field=f"source_annotations.{name}")
        except ValueError as error:
            _add_issue(errors, "unsafe_annotation_path", str(error), sample_id=sample_id, path=relative, field=f"source_annotations.{name}")
    boxes_relative = record.source_annotations.get("boxes")
    contours_relative = record.source_annotations.get("contours")
    segmentation_relative = record.source_annotations.get("segmentation")
    if boxes_relative:
        try:
            box_file = _safe_asset_path(root, boxes_relative)
            if box_file.is_file():
                raw_boxes = _read_json(box_file)
                if set(raw_boxes) - set(DETECTOR_CATEGORIES):
                    _add_issue(errors, "unsupported_annotation_category", f"Bounding-box file contains unsupported categories {sorted(set(raw_boxes) - set(DETECTOR_CATEGORIES))}", sample_id=sample_id, path=boxes_relative, field="categories")
                for category, components in by_category.items():
                    source_xywh = raw_boxes.get(category, [])
                    converted = [[float(box[0]), float(box[1]), float(box[0]) + float(box[2]), float(box[1]) + float(box[3])] for box in source_xywh if isinstance(box, list) and len(box) == 4]
                    output_boxes = [list(component.bbox_xyxy_original) for component in components]
                    if len(converted) != len(source_xywh) or sorted(converted) != sorted(output_boxes):
                        _add_issue(errors, "bbox_annotation_mismatch", f"Prepared boxes for {category} do not match source xywh annotations", sample_id=sample_id, path=boxes_relative, field=f"components.{category}.bbox_xyxy_original")
                if set(raw_boxes) != set(by_category):
                    _add_issue(errors, "bbox_category_mismatch", f"Source box categories {sorted(raw_boxes)} differ from sample components {sorted(by_category)}", sample_id=sample_id, path=boxes_relative, field="categories")
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
            _add_issue(errors, "bbox_file", f"Cannot validate bounding-box annotations: {error}", sample_id=sample_id, path=boxes_relative, field="source_annotations.boxes")
    if contours_relative:
        try:
            contours_path = _safe_asset_path(root, contours_relative)
            if contours_path.is_file():
                contours = _read_json(contours_path)
                unsupported = set(contours) - set(DETECTOR_CATEGORIES)
                if unsupported:
                    _add_issue(errors, "unsupported_contour_category", f"Contour file contains unsupported categories {sorted(unsupported)}", sample_id=sample_id, path=contours_relative, field="categories")
        except (OSError, json.JSONDecodeError, ValueError) as error:
            _add_issue(errors, "contour_file", f"Cannot validate contour annotation: {error}", sample_id=sample_id, path=contours_relative, field="source_annotations.contours")
    if segmentation_relative:
        try:
            segmentation_path = _safe_asset_path(root, segmentation_relative)
            if segmentation_path.is_file() and segmentation_path.stat().st_size == 0:
                _add_issue(errors, "empty_segmentation", "Segmentation reference exists but is empty", sample_id=sample_id, path=segmentation_relative, field="source_annotations.segmentation")
        except (OSError, ValueError) as error:
            _add_issue(errors, "segmentation_file", str(error), sample_id=sample_id, path=segmentation_relative, field="source_annotations.segmentation")


def validate_dataset(
    dataset_path: Path,
    *,
    write_report: bool = True,
    report_path: Path | None = None,
) -> ValidationReport:
    """Validate one immutable dataset and return structured errors/warnings."""

    root = dataset_path.expanduser().resolve()
    errors: list[Diagnostic] = []
    warnings: list[Diagnostic] = []
    checks: dict[str, bool] = {}
    manifest_path = root / MANIFEST_FILENAME
    dataset_id: str | None = None
    try:
        manifest = _read_json(manifest_path)
        dataset_id = manifest.get("dataset_id") if isinstance(manifest.get("dataset_id"), str) else None
    except (OSError, json.JSONDecodeError, ValueError) as error:
        report = ValidationReport(str(root), None, [Diagnostic("manifest", str(error), path=str(manifest_path))])
        if write_report and root.is_dir():
            _store_report(root, report, report_path)
        return report

    if manifest.get("schema_version") != SCHEMA_VERSION:
        _add_issue(errors, "schema_version", f"Expected dataset schema {SCHEMA_VERSION}, found {manifest.get('schema_version')!r}", path=str(manifest_path), field="schema_version")
    source_info = manifest.get("source")
    if not isinstance(source_info, dict):
        source_info = {}
        _add_issue(errors, "manifest_source", "Manifest source provenance must be a JSON object", path=str(manifest_path), field="source")
    ontology_info = manifest.get("ontology")
    if not isinstance(ontology_info, dict):
        ontology_info = {}
        _add_issue(errors, "manifest_ontology", "Manifest ontology provenance must be a JSON object", path=str(manifest_path), field="ontology")
    if source_info.get("generator_repository") != GENERATOR_REPOSITORY:
        _add_issue(errors, "generator_repository", "Dataset source repository differs from the pinned Pizzaïolo project", path=str(manifest_path), field="source.generator_repository")
    if source_info.get("generator_commit") != GENERATOR_COMMIT:
        _add_issue(errors, "generator_provenance", f"Dataset must use pinned Pizzaïolo revision {GENERATOR_COMMIT}", path=str(manifest_path), field="source.generator_commit")
    if source_info.get("generator_license") != GENERATOR_LICENSE:
        _add_issue(errors, "generator_license", "Dataset license provenance differs from the pinned upstream declaration", path=str(manifest_path), field="source.generator_license")
    if source_info.get("raw_master_canonicalization") != RAW_MASTER_CANONICALIZATION:
        _add_issue(errors, "raw_master_canonicalization", "Raw generator metadata canonicalization policy is missing or changed", path=str(manifest_path), field="source.raw_master_canonicalization")
    raw_master_reference = source_info.get("raw_master_csv")
    expected_raw_master = f"{AUDIT_DIR}/{RAW_MASTER_FILENAME}"
    if raw_master_reference != expected_raw_master or not (root / expected_raw_master).is_file():
        _add_issue(errors, "raw_master_reference", f"Expected preserved generator metadata at {expected_raw_master}", path=str(manifest_path), field="source.raw_master_csv")
    if ontology_info.get("sha256") != ONTOLOGY_SHA256:
        _add_issue(errors, "ontology_provenance", "Manifest ontology hash is not the pinned ontology snapshot", path=str(manifest_path), field="ontology.sha256")
    _validate_manifest_identity(manifest, errors, manifest_path)
    _validate_assets(root, manifest, errors)
    ontology, class_vocabulary, detector_vocabulary = _validate_vocabulary(root, manifest, errors)

    records_path = root / SAMPLES_FILENAME
    try:
        records = _load_samples(records_path)
    except (OSError, ValueError) as error:
        _add_issue(errors, "samples_jsonl", str(error), path=str(records_path))
        records = []
    if not records:
        _add_issue(errors, "empty_dataset", "Dataset must contain at least one sample", path=str(records_path))

    if ontology and class_vocabulary and detector_vocabulary:
        classes = class_vocabulary.get("classes", [])
        categories = detector_vocabulary.get("categories", [])
        sample_ids: set[str] = set()
        component_ids: set[str] = set()
        for record in records:
            _validate_record(root, record, records_path, ontology, classes, categories, sample_ids, component_ids, errors)

        if sample_ids != set(manifest.get("sample_ids", [])):
            _add_issue(errors, "manifest_sample_ids", "Manifest sample_ids do not match JSONL records", path=str(manifest_path), field="sample_ids")

        # Splits must be disjoint, exhaustive, and exactly the established
        # generator-type-stratified two-stage split for the resolved seed.
        split_sets: dict[str, set[str]] = {}
        for split in SPLIT_NAMES:
            split_file = root / SPLIT_DIR / f"{split}.json"
            try:
                payload = _read_json(split_file)
                ids = payload.get("sample_ids")
                if payload.get("schema_version") != SCHEMA_VERSION or not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
                    raise ValueError("expected schema_version and string sample_ids array")
                if len(ids) != len(set(ids)):
                    _add_issue(errors, "duplicate_split_id", f"{split} split contains duplicate IDs", path=str(split_file), field="sample_ids")
                split_sets[split] = set(ids)
                record_ids = {record.sample_id for record in records if record.split == split}
                if split_sets[split] != record_ids:
                    _add_issue(errors, "split_reference", f"{split} manifest IDs differ from sample split assignments", path=str(split_file), field="sample_ids")
            except (OSError, json.JSONDecodeError, ValueError) as error:
                _add_issue(errors, "split_file", f"Cannot validate split {split}: {error}", path=str(split_file))
                split_sets[split] = set()
        if any(split_sets[left] & split_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            _add_issue(errors, "split_overlap", "Train, validation, and test sample IDs must be disjoint", path=str(root / SPLIT_DIR))
        if set.union(*split_sets.values()) != sample_ids:
            _add_issue(errors, "split_coverage", "Split manifests must cover every sample exactly once", path=str(root / SPLIT_DIR))

        generation = manifest.get("resolved_config", {})
        split_ratios = generation.get("split_ratios", {})
        try:
            expected_assignment = assign_splits(
                [{"sample_id": item.sample_id, "generator_type": item.generator_type} for item in records],
                train_size=float(split_ratios["train"]),
                valid_size=float(split_ratios["validation"]),
                test_size=float(split_ratios["test"]),
                seed=int(generation["split_seed"]),
            )
            actual_assignment = {item.sample_id: item.split for item in records}
            if expected_assignment != actual_assignment:
                _add_issue(errors, "split_determinism", "Split assignments differ from the documented deterministic type-stratified policy", path=str(records_path), field="split")
        except (KeyError, TypeError, ValueError) as error:
            _add_issue(errors, "split_config", f"Cannot reconstruct split policy from manifest: {error}", path=str(manifest_path), field="resolved_config.split_ratios")

        try:
            statistics = build_statistics(records, classes)
            recorded_statistics = manifest.get("statistics")
            if recorded_statistics != statistics:
                _add_issue(errors, "statistics_mismatch", "Manifest class-support or split statistics do not match sample targets", path=str(manifest_path), field="statistics")
            for message in support_warnings(statistics, classes):
                warnings.append(Diagnostic("low_class_support", message))
        except (IndexError, ValueError, TypeError) as error:
            statistics = {}
            _add_issue(errors, "statistics", f"Cannot compute class-support statistics: {error}", path=str(records_path))
    else:
        statistics = {}

    checks.update(
        {
            "manifest_schema": manifest.get("schema_version") == SCHEMA_VERSION,
            "pinned_generator": source_info.get("generator_commit") == GENERATOR_COMMIT,
            "pinned_ontology": ontology_info.get("sha256") == ONTOLOGY_SHA256,
            "asset_checksums": not any(issue.code in {"missing_asset", "asset_checksum", "unsafe_asset_path"} for issue in errors),
            "vocabulary": ontology is not None and class_vocabulary is not None and detector_vocabulary is not None,
            "records_and_annotations": not any(issue.code in {"missing_image", "invalid_bbox", "bbox_file", "bbox_annotation_mismatch", "missing_annotation_asset"} for issue in errors),
            "split_integrity": not any(issue.code.startswith("split_") or issue.code in {"split_overlap", "split_coverage", "split_reference"} for issue in errors),
            "semantic_targets": not any(issue.code in {"pizza_targets", "component_targets", "pizza_known_mask", "component_known_mask", "source_labels"} for issue in errors),
        }
    )
    report = ValidationReport(str(root), dataset_id, errors, warnings, statistics, checks)
    if write_report and root.is_dir():
        _store_report(root, report, report_path)
    return report


def _store_report(root: Path, report: ValidationReport, report_path: Path | None) -> None:
    destination = report_path or (root / REPORT_DIR / VALIDATION_REPORT_FILENAME)
    if not destination.is_absolute():
        destination = (root / destination).resolve()
    body = report.to_dict()
    body.pop("report_id", None)
    report.report_id = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    _atomic_json(destination, report.to_dict())


def report_markdown(report: ValidationReport) -> str:
    lines = [
        f"# Dataset validation: {'PASS' if report.valid else 'FAIL'}",
        "",
        f"- Dataset ID: `{report.dataset_id or 'unknown'}`",
        f"- Dataset path: `{report.dataset_path}`",
        f"- Errors: {len(report.errors)}",
        f"- Warnings: {len(report.warnings)}",
        "",
    ]
    for title, issues in (("Errors", report.errors), ("Warnings", report.warnings)):
        lines.extend([f"## {title}", ""])
        if not issues:
            lines.extend(["None.", ""])
        for issue in issues:
            context = ", ".join(
                f"{key}={value}" for key, value in (("sample", issue.sample_id), ("path", issue.path), ("field", issue.field)) if value
            )
            lines.append(f"- **{issue.code}:** {issue.message}" + (f" ({context})" if context else ""))
        lines.append("")
    return "\n".join(lines)
