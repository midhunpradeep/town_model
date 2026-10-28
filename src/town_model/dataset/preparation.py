"""Atomic dataset generation, ontology target expansion, and versioned export."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image

from .. import __version__
from ..config import DatasetConfig
from .constants import (
    AUDIT_DIR,
    BASE_CATEGORIES,
    CLASS_VOCABULARY_FILENAME,
    DETECTOR_CATEGORIES,
    DETECTOR_VOCABULARY_FILENAME,
    GENERATOR_COMMIT,
    GENERATOR_LICENSE,
    GENERATOR_REPOSITORY,
    IMAGE_DIR,
    LABEL_DIR,
    MANIFEST_FILENAME,
    ONTOLOGY_DIR,
    ONTOLOGY_FILENAME,
    ONTOLOGY_SHA256,
    RAW_MASTER_CANONICALIZATION,
    RAW_MASTER_FILENAME,
    REPORT_DIR,
    SAMPLES_FILENAME,
    SCHEMA_VERSION,
    SPLIT_DIR,
    SPLIT_NAMES,
    TOPPING_COLUMNS,
    VOCABULARY_DIR,
)
from .generator import Progress, generate_raw_dataset
from .labels import infer_pizzaiolo_label
from .ontology import (
    OntologySnapshot,
    build_class_vocabulary,
    build_detector_vocabulary,
)
from .schema import ComponentRecord, DatasetRecord, write_records, write_split_manifest
from .splitting import assign_splits
from .statistics import build_statistics
from .validation import (
    canonical_json_bytes,
    sha256_file,
    validate_dataset,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    temporary.replace(path)


def _git_provenance(project_root: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=no"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted", True


def _normalize_optional(value: Any) -> str | None:
    """Normalize only a genuinely absent CSV field; never treat text tokens as null."""

    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"Optional generator metadata must be text or blank, got {value!r}")
    return value


def _read_topping_list(value: Any, *, sample_id: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError) as error:
        raise ValueError(f"sample {sample_id}: malformed hasTopping field: {error}") from error
    if not isinstance(parsed, (list, tuple)) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"sample {sample_id}: hasTopping must be a serialized list of strings")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"sample {sample_id}: hasTopping contains duplicate categories")
    return list(parsed)


def _parse_count(value: Any, *, sample_id: str, topping: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"sample {sample_id}: {topping} must be a nonnegative integer, got {value!r}")
    if isinstance(value, int):
        count = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        count = int(value)
    else:
        raise ValueError(f"sample {sample_id}: {topping} must be a nonnegative integer, got {value!r}")
    if count < 0:
        raise ValueError(f"sample {sample_id}: {topping} count cannot be negative")
    return count


def _source_relative_filename(value: Any, *, sample_id: str, field: str) -> str:
    name = str(value).strip()
    if not name or Path(name).name != name or name in {".", ".."} or "\\" in name:
        raise ValueError(f"sample {sample_id}: invalid {field} filename {name!r}")
    return name


def _write_audit_labels(raw_frame: pd.DataFrame, labels: list[list[str]], destination: Path) -> None:
    frame = raw_frame.copy()
    frame["source_inferred_multilabel"] = ["|".join(items) for items in labels]
    frame.to_csv(destination, index=False)


def _build_record(
    row: dict[str, Any],
    *,
    root: Path,
    class_vocabulary: dict[str, Any],
    detector_vocabulary: dict[str, Any],
    ontology: OntologySnapshot,
) -> tuple[DatasetRecord, list[str]]:
    sample_id = str(row.get("ref", "")).strip()
    if not sample_id:
        raise ValueError("Generator row has an empty ref/sample ID")
    img_name = _source_relative_filename(row.get("img_name"), sample_id=sample_id, field="img_name")
    if img_name != f"{sample_id}.png":
        raise ValueError(f"sample {sample_id}: img_name must be {sample_id}.png, got {img_name!r}")

    counts = {name: _parse_count(row.get(name), sample_id=sample_id, topping=name) for name in TOPPING_COLUMNS}
    has_topping = _read_topping_list(row.get("hasTopping"), sample_id=sample_id)
    unknown_toppings = set(has_topping) - set(TOPPING_COLUMNS)
    if unknown_toppings:
        raise ValueError(f"sample {sample_id}: unsupported hasTopping values {sorted(unknown_toppings)}")
    positive_counts = {name for name, count in counts.items() if count > 0}
    if positive_counts != set(has_topping):
        raise ValueError(
            f"sample {sample_id}: positive topping counts {sorted(positive_counts)} "
            f"disagree with hasTopping {sorted(set(has_topping))}"
        )

    base = _normalize_optional(row.get("hasBase"))
    if base not in BASE_CATEGORIES:
        raise ValueError(f"sample {sample_id}: required hasBase is invalid: {base!r}")
    origin = _normalize_optional(row.get("hasCountryOfOrigin"))
    if origin is not None and origin not in {"America", "England", "France", "Italy"}:
        raise ValueError(
            f"sample {sample_id}: unsupported country flag/origin {origin!r}; "
            "the pinned generator renders America, England, France, and Italy only"
        )

    # The pinned generator writes Python None as the literal ``None`` only for
    # its type column; country-of-origin absence is serialized as an empty CSV
    # field. Keep this source-specific convention out of optional normalization.
    raw_source_type = row.get("type")
    source_type = None if raw_source_type == "None" else _normalize_optional(raw_source_type)
    generator_type = "random" if source_type in {None, "unspecified"} else source_type
    raw_boxes_name = _source_relative_filename(row.get("boxes_name"), sample_id=sample_id, field="boxes_name")
    raw_contours_name = _source_relative_filename(row.get("contours_name"), sample_id=sample_id, field="contours_name")
    raw_segmentation_name = _source_relative_filename(row.get("segmentation_name"), sample_id=sample_id, field="segmentation_name")
    expected_prefix = sample_id
    if not all(name.startswith(expected_prefix) for name in (raw_boxes_name, raw_contours_name, raw_segmentation_name)):
        raise ValueError(f"sample {sample_id}: annotation filenames must use the source ref prefix")

    boxes_path = root / LABEL_DIR / raw_boxes_name
    with boxes_path.open(encoding="utf-8") as file:
        raw_boxes = json.load(file)
    if not isinstance(raw_boxes, dict):
        raise ValueError(f"sample {sample_id}: bounding-box annotation must be a JSON object at {boxes_path}")
    detector_ids = {item["name"]: item["category_id"] for item in detector_vocabulary["categories"]}
    if set(raw_boxes) - set(detector_ids):
        raise ValueError(f"sample {sample_id}: bounding boxes contain unsupported categories {sorted(set(raw_boxes) - set(detector_ids))}")

    expected_categories = set(positive_counts) | {base} | ({origin} if origin else set())
    if set(raw_boxes) != expected_categories:
        raise ValueError(
            f"sample {sample_id}: bounding-box categories {sorted(raw_boxes)} disagree with "
            f"topping/base/origin metadata {sorted(expected_categories)}"
        )
    for topping, count in counts.items():
        if len(raw_boxes.get(topping, [])) != count:
            raise ValueError(
                f"sample {sample_id}: {topping} count {count} disagrees with "
                f"{len(raw_boxes.get(topping, []))} bounding boxes in {boxes_path}"
            )
    if len(raw_boxes.get(base, [])) != 1:
        raise ValueError(f"sample {sample_id}: required base {base} must have exactly one bounding box in {boxes_path}")
    if origin and len(raw_boxes.get(origin, [])) != 1:
        raise ValueError(f"sample {sample_id}: origin {origin} must have exactly one flag box in {boxes_path}")

    image_relative = f"{IMAGE_DIR}/{img_name}"
    image_path = root / image_relative
    with Image.open(image_path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
            image.save(image_path, format="PNG")
        width, height = image.size

    classes = class_vocabulary["classes"]
    class_ids = {item["name"]: item["class_id"] for item in classes}
    known_mask = tuple(int(not item["unsupported"]) for item in classes)
    source_labels = infer_pizzaiolo_label(
        {
            "hasTopping": has_topping,
            "hasBase": base,
            "hasCountryOfOrigin": origin,
            **counts,
        }
    )
    pizza_targets_set: set[str] = set()
    for label in source_labels:
        pizza_targets_set.update(ontology.ancestors(label))
    pizza_targets = tuple(int(item["name"] in pizza_targets_set) for item in classes)

    components: list[ComponentRecord] = []
    for category in sorted(raw_boxes, key=lambda name: detector_ids[name]):
        occurrences = raw_boxes[category]
        if not isinstance(occurrences, list):
            raise ValueError(f"sample {sample_id}: {category} annotations must be a list in {boxes_path}")
        ontology_target_names = ontology.component_class_targets(category)
        target_vector = tuple(int(item["name"] in ontology_target_names) for item in classes)
        for index, box in enumerate(occurrences):
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"sample {sample_id}: malformed {category} xywh box #{index} in {boxes_path}")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in box):
                raise ValueError(f"sample {sample_id}: nonnumeric {category} xywh box #{index} in {boxes_path}")
            x, y, box_width, box_height = (float(value) for value in box)
            if x < 0 or y < 0 or box_width <= 0 or box_height <= 0 or x + box_width > width or y + box_height > height:
                raise ValueError(f"sample {sample_id}: invalid {category} xywh box #{index} {box} for {width}x{height} image in {boxes_path}")
            component_id = f"{sample_id}/{category}/occurrence_{index:03d}"
            components.append(
                ComponentRecord(
                    component_id=component_id,
                    detector_category_id=detector_ids[category],
                    detector_category=category,
                    bbox_xyxy_original=(x, y, x + box_width, y + box_height),
                    ontology_targets=target_vector,
                    ontology_known_mask=known_mask,
                    annotation_path=f"{LABEL_DIR}/{raw_boxes_name}",
                )
            )

    audit = {
        "usage": "audit_only_not_predictive_input",
        "raw_generator": {
            "source_ref": sample_id,
            "source_type": source_type,
            "generator_type": generator_type,
            "has_topping": has_topping,
            "topping_counts": counts,
            "has_base": base,
            "country_of_origin": origin,
            "original_annotation_filenames": {
                "boxes": raw_boxes_name,
                "contours": raw_contours_name,
                "segmentation": raw_segmentation_name,
            },
        },
    }
    source_annotations = {
        "boxes": f"{LABEL_DIR}/{raw_boxes_name}",
        "contours": f"{LABEL_DIR}/{raw_contours_name}",
        "segmentation": f"{LABEL_DIR}/{raw_segmentation_name}",
    }
    record = DatasetRecord(
        sample_id=sample_id,
        split="train",
        generator_type=generator_type,
        image_path=image_relative,
        image_width=width,
        image_height=height,
        pizza_targets=pizza_targets,
        pizza_known_mask=known_mask,
        pizza_source_labels=tuple(source_labels),
        components=tuple(components),
        audit=audit,
        source_annotations=source_annotations,
    )
    return record, source_labels


def _write_jsonl_and_splits(root: Path, records: list[DatasetRecord]) -> None:
    write_records(records, root / SAMPLES_FILENAME)
    for split in SPLIT_NAMES:
        write_split_manifest(
            (record for record in records if record.split == split),
            root / SPLIT_DIR / f"{split}.json",
        )


def _asset_checksums(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    excluded = {MANIFEST_FILENAME}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded or relative.startswith(f"{REPORT_DIR}/") or relative.endswith(".tmp"):
            continue
        checksums[relative] = sha256_file(path)
    return checksums


def _dataset_root_allowed(destination: Path, project_root: Path) -> None:
    if destination == project_root or destination in project_root.parents:
        raise ValueError("Dataset output cannot replace the repository or one of its parent directories")
    source_tree = project_root / "src"
    git_tree = project_root / ".git"
    source_cache = project_root / "artifacts" / "sources"
    if destination == source_tree or source_tree in destination.parents:
        raise ValueError("Generated datasets must be stored outside the source-code tree")
    if destination == git_tree or git_tree in destination.parents:
        raise ValueError("Dataset output cannot be placed inside Git metadata")
    if destination == source_cache or source_cache in destination.parents or destination in source_cache.parents:
        raise ValueError("Dataset output cannot replace or be nested inside the pinned-source cache")


def validate_output_path(output_path: Path, *, project_root: Path | None = None) -> Path:
    """Resolve and validate a new output location without creating data."""

    root = (project_root or Path(__file__).resolve().parents[3]).expanduser().resolve()
    destination = output_path.expanduser().resolve()
    _dataset_root_allowed(destination, root)
    if destination.exists():
        raise FileExistsError(f"Dataset output already exists: {destination}; choose a new version path")
    return destination


def prepare_dataset(
    config: DatasetConfig,
    output_path: Path,
    *,
    project_root: Path | None = None,
    source_repo: Path | None = None,
    progress: Progress | None = None,
) -> Path:
    """Generate and atomically publish one immutable prepared dataset version.

    The output path must not exist. Failed runs leave no partial destination;
    raw source files, prepared records, vocabularies, splits, checksums, and a
    post-publication validation report are kept together as one version.
    """

    config.validate(require_total=True)
    requested_counts = config.allocate_type_counts()
    root = project_root or Path(__file__).resolve().parents[3]
    root = root.expanduser().resolve()
    destination = validate_output_path(output_path, project_root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)

    lock = destination.with_name(f".{destination.name}.generation.lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"Another generation is using output path {destination}") from error
    os.close(lock_fd)
    try:
        if destination.exists():
            raise FileExistsError(f"Dataset output already exists: {destination}")
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}.staging-", dir=destination.parent) as temporary:
            staged = Path(temporary) / "dataset"
            generate_raw_dataset(
                staged,
                config,
                project_root=root,
                source_repo=source_repo,
                progress=progress,
            )
            if progress:
                progress(0.78, "Checking ontology and expanding image/component targets")

            source_master = staged / "csv" / "pizzaiolo_dataset.csv"
            if not source_master.is_file():
                raise FileNotFoundError(f"Pinned generator did not produce {source_master}")
            audit_dir = staged / AUDIT_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            raw_frame = pd.read_csv(source_master, dtype=str, keep_default_na=False)
            required_columns = {
                "ref", "img_name", "type", "hasTopping", "hasBase", "hasCountryOfOrigin",
                "boxes_name", "contours_name", "segmentation_name", *TOPPING_COLUMNS,
            }
            missing_columns = required_columns - set(raw_frame.columns)
            if missing_columns:
                raise ValueError(f"Generator master CSV is missing required fields: {sorted(missing_columns)}")
            if raw_frame["ref"].duplicated().any():
                raise ValueError("Generator master CSV contains duplicate sample refs")
            # Pizzaïolo serializes a set as an ordered-looking list. Its order is
            # process-dependent, so canonicalize this semantically unordered field.
            raw_frame["hasTopping"] = [
                repr(sorted(_read_topping_list(value, sample_id=sample_id)))
                for sample_id, value in zip(raw_frame["ref"], raw_frame["hasTopping"], strict=True)
            ]
            raw_master = audit_dir / RAW_MASTER_FILENAME
            raw_frame.to_csv(raw_master, index=False)
            source_master.unlink()
            csv_dir = staged / "csv"
            if csv_dir.exists() and not any(csv_dir.iterdir()):
                csv_dir.rmdir()

            ontology_path = staged / ONTOLOGY_DIR / ONTOLOGY_FILENAME
            ontology = OntologySnapshot.load(ontology_path, require_pinned_hash=True)
            if ontology.sha256 != ONTOLOGY_SHA256:
                raise ValueError(
                    f"Pinned ontology changed: {ontology.sha256} != {ONTOLOGY_SHA256}"
                )
            class_vocabulary = build_class_vocabulary(ontology)
            detector_vocabulary = build_detector_vocabulary(ontology)
            class_path = staged / VOCABULARY_DIR / CLASS_VOCABULARY_FILENAME
            category_path = staged / VOCABULARY_DIR / DETECTOR_VOCABULARY_FILENAME
            _write_json(class_path, class_vocabulary)
            _write_json(category_path, detector_vocabulary)

            records: list[DatasetRecord] = []
            inferred_labels: list[list[str]] = []
            for row in raw_frame.to_dict(orient="records"):
                try:
                    record, labels = _build_record(
                        row,
                        root=staged,
                        class_vocabulary=class_vocabulary,
                        detector_vocabulary=detector_vocabulary,
                        ontology=ontology,
                    )
                except Exception as error:
                    sample_id = row.get("ref", "unknown")
                    raise ValueError(f"Invalid generator record {sample_id!r}: {error}") from error
                records.append(record)
                inferred_labels.append(labels)

            if len(records) != config.total_instances:
                raise RuntimeError(f"Generator produced {len(records)} samples; expected {config.total_instances}")
            actual_counts: dict[str, int] = {}
            for record in records:
                actual_counts[record.generator_type] = actual_counts.get(record.generator_type, 0) + 1
            expected_counts = {name: count for name, count in requested_counts.items() if count}
            if actual_counts != expected_counts:
                raise RuntimeError(f"Generator type counts differ from resolved config: {actual_counts} != {expected_counts}")
            _write_audit_labels(raw_frame, inferred_labels, audit_dir / "generator_master_with_source_labels.csv")

            assignment = assign_splits(
                [{"sample_id": record.sample_id, "generator_type": record.generator_type} for record in records],
                train_size=config.train_size,
                valid_size=config.valid_size,
                test_size=config.test_size,
                seed=config.split_seed,
            )
            records = [replace(record, split=assignment[record.sample_id]) for record in records]
            _write_jsonl_and_splits(staged, records)
            if progress:
                progress(0.88, "Writing immutable manifests and per-class support statistics")

            statistics = build_statistics(records, class_vocabulary["classes"])
            checksums = _asset_checksums(staged)
            resolved_config = config.resolved_dict()
            identity_payload = {
                "schema_version": SCHEMA_VERSION,
                "resolved_config": resolved_config,
                "generator_commit": GENERATOR_COMMIT,
                "ontology_sha256": ontology.sha256,
                "asset_checksums": checksums,
            }
            dataset_id = "pizza-" + hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()[:20]
            preparation_commit, preparation_dirty = _git_provenance(root)
            split_sizes = statistics["split_sizes"]
            manifest: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "manifest_id": "",
                "wire_format": "UTF-8 JSON Lines (one JSON object per sample)",
                "sample_record_count": len(records),
                "sample_ids": [record.sample_id for record in records],
                "sample_id_policy": "original pinned-generator ref, stable within the dataset version",
                "component_id_policy": "<sample_id>/<detector-category>/occurrence_<zero-based-index-padded-to-3>",
                "resolved_config": resolved_config,
                "source": {
                    "generator_repository": GENERATOR_REPOSITORY,
                    "generator_commit": GENERATOR_COMMIT,
                    "generator_license": GENERATOR_LICENSE,
                    "preparation_package_version": __version__,
                    "preparation_commit": preparation_commit,
                    "preparation_dirty": preparation_dirty,
                    "raw_master_csv": f"{AUDIT_DIR}/{raw_master.name}",
                    "raw_master_canonicalization": RAW_MASTER_CANONICALIZATION,
                    "image_label_inference": "audited fourth_attempt infer_pizzaiolo_label predicates; 32 overlapping image-level classes",
                },
                "ontology": {
                    "path": f"{ONTOLOGY_DIR}/{ONTOLOGY_FILENAME}",
                    "sha256": ontology.sha256,
                    "class_count": len(ontology.class_names),
                    "individual_count": len(ontology.individuals),
                    "country_individuals": sorted(ontology.individuals),
                },
                "vocabularies": {
                    "ontology_classes_path": f"{VOCABULARY_DIR}/{CLASS_VOCABULARY_FILENAME}",
                    "ontology_classes_sha256": sha256_file(class_path),
                    "ontology_class_count": len(class_vocabulary["classes"]),
                    "trainable_class_count": len(class_vocabulary["trainable_class_ids"]),
                    "detector_categories_path": f"{VOCABULARY_DIR}/{DETECTOR_VOCABULARY_FILENAME}",
                    "detector_categories_sha256": sha256_file(category_path),
                    "detector_foreground_category_count": len(detector_vocabulary["categories"]),
                    "background_category_id": detector_vocabulary["background_category_id"],
                },
                "coordinate_contract": {
                    "source": "pinned Pizzaïolo [x, y, width, height] integer boundingRect output",
                    "wire": "original-image pixel xyxy; x2=x+width and y2=y+height; finite pixel-edge extents, no normalization or resize",
                    "image_color_space": "RGB",
                },
                "supervision_contract": {
                    "target_order": "ontology class_id ascending in vocabularies/ontology_classes.json",
                    "encoding": "ontology_targets and ontology_known_mask are integer 0/1 vectors with one entry for every OWL class",
                    "known_negative": "Validated complete pinned-generator ingredient, base, origin, and entity-type facts use a closed-world mask for the 61 supported ontology classes only",
                    "unknown": "Five value-partition/spiciness classes lacking point supervision have target 0 and known mask 0; malformed metadata is rejected, never converted to negative",
                    "fixed_positive": "DomainThing is positive for all generated pizza and component entities and is not trainable",
                    "country_flags": "Four detector categories are OWL individuals; targets are Country and its class ancestors, never the individual names as classes",
                },
                "split_policy": {
                    "name": "generator-type-stratified two-stage seeded split, adapted from fourth_attempt",
                    "seed": config.split_seed,
                    "ratios": resolved_config["split_ratios"],
                    "files": {split: f"{SPLIT_DIR}/{split}.json" for split in SPLIT_NAMES},
                    "sizes": split_sizes,
                },
                "statistics": statistics,
                "asset_checksums": checksums,
                "validation_report_path": f"{REPORT_DIR}/validation.json",
                "compatibility": {
                    "schema_major": 1,
                    "rule": "Readers must reject unknown major versions and require matching ontology/class-vocabulary hashes; no implicit class-ID migration is performed",
                },
            }
            manifest["manifest_id"] = hashlib.sha256(canonical_json_bytes({key: value for key, value in manifest.items() if key != "manifest_id"})).hexdigest()
            _write_json(staged / MANIFEST_FILENAME, manifest)

            prepublish_report = validate_dataset(staged, write_report=False)
            if not prepublish_report.valid:
                details = "; ".join(issue.message for issue in prepublish_report.errors[:8])
                raise ValueError(f"Generated dataset failed integrity validation: {details}")
            if destination.exists():
                raise FileExistsError(f"Dataset output appeared during generation: {destination}")
            os.rename(staged, destination)

        final_report = validate_dataset(destination, write_report=True)
        if progress:
            progress(1.0, "Dataset published and integrity validation completed")
        if not final_report.valid:
            raise RuntimeError("Published dataset failed its post-publication validation")
        return destination
    finally:
        lock.unlink(missing_ok=True)
