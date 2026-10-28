from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from town_model.dataset.constants import FIXED_POSITIVE_CLASS, UNSUPPORTED_CLASS_NAMES
from town_model.dataset.ontology import build_class_vocabulary
from town_model.dataset.schema import load_records, write_records
from town_model.dataset.validation import canonical_json_bytes, sha256_file, validate_dataset
from town_model.dataset.preparation import prepare_dataset
from town_model.config import DatasetConfig
from conftest import fake_generate_raw_dataset


def _class_index(name: str) -> int:
    classes = build_class_vocabulary()["classes"]
    return next(item["class_id"] for item in classes if item["name"] == name)


def _rewrite_record(dataset: Path, sample_index: int, change) -> None:
    path = dataset / "samples.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    change(records[sample_index])
    path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records), encoding="utf-8")


def _refresh_manifest_asset(dataset: Path, relative_path: str) -> None:
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["asset_checksums"][relative_path] = sha256_file(dataset / relative_path)
    identity = {
        "schema_version": manifest["schema_version"],
        "resolved_config": manifest["resolved_config"],
        "generator_commit": manifest["source"]["generator_commit"],
        "ontology_sha256": manifest["ontology"]["sha256"],
        "asset_checksums": manifest["asset_checksums"],
    }
    manifest["dataset_id"] = "pizza-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:20]
    manifest["manifest_id"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in manifest.items() if key != "manifest_id"})
    ).hexdigest()
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_framework_neutral_jsonl_roundtrip_and_stable_ids(prepared_dataset: Path, tmp_path: Path) -> None:
    records = load_records(prepared_dataset / "samples.jsonl")
    roundtrip = tmp_path / "roundtrip.jsonl"
    write_records(records, roundtrip)
    assert [record.to_dict() for record in records] == [record.to_dict() for record in load_records(roundtrip)]
    assert records[0].sample_id == "img_00000"
    assert all(component.component_id.startswith(record.sample_id + "/") for record in records for component in record.components)


def test_duplicate_topping_occurrences_have_deterministic_distinct_ids(prepared_dataset: Path) -> None:
    random_sample = next(record for record in load_records(prepared_dataset / "samples.jsonl") if record.generator_type == "random" and record.audit["raw_generator"]["source_ref"] == "img_00003")
    repeated = [item for item in random_sample.components if item.detector_category == "AnchovyTopping"]
    assert len(repeated) == 2
    assert [item.component_id for item in repeated] == [
        "img_00003/AnchovyTopping/occurrence_000",
        "img_00003/AnchovyTopping/occurrence_001",
    ]


def test_positive_negative_unknown_masks_and_flag_semantics(prepared_dataset: Path) -> None:
    records = load_records(prepared_dataset / "samples.jsonl")
    no_origin = next(record for record in records if record.audit["raw_generator"]["country_of_origin"] is None)
    real_america = _class_index("RealAmericanPizza")
    domain = _class_index(FIXED_POSITIVE_CLASS)
    unsupported = _class_index("Hot")
    assert no_origin.pizza_targets[real_america] == 0
    assert no_origin.pizza_known_mask[real_america] == 1  # closed-world generated absence
    assert no_origin.pizza_targets[domain] == 1
    assert no_origin.pizza_known_mask[unsupported] == 0
    assert no_origin.pizza_targets[unsupported] == 0
    assert not any(item.detector_category in {"America", "England", "France", "Italy"} for item in no_origin.components)

    flagged = next(record for record in records if record.audit["raw_generator"]["country_of_origin"] == "America")
    real_america = _class_index("RealAmericanPizza")
    assert flagged.pizza_targets[real_america] == flagged.pizza_known_mask[real_america] == 1
    flag = next(item for item in flagged.components if item.detector_category == "America")
    country = _class_index("Country")
    assert flag.ontology_targets[country] == flag.ontology_known_mask[country] == 1
    assert "America" not in {item["name"] for item in build_class_vocabulary()["classes"]}


@pytest.mark.parametrize("origin", ["NaN", "nan", "None", "none", "Germany", " "])
def test_malformed_or_unsupported_origin_values_fail_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_config: DatasetConfig, origin: str
) -> None:
    from town_model.dataset import preparation

    def generate_with_origin(dataset_dir, config, *, project_root, source_repo=None, progress=None):
        result = fake_generate_raw_dataset(
            dataset_dir, config, project_root=project_root, source_repo=source_repo, progress=progress
        )
        source_csv = dataset_dir / "csv" / "pizzaiolo_dataset.csv"
        with source_csv.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
            fieldnames = list(rows[0])
        random_row = next(row for row in rows if row["type"] == "None")
        random_row["hasCountryOfOrigin"] = origin
        with source_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return result

    monkeypatch.setattr(preparation, "generate_raw_dataset", generate_with_origin)
    with pytest.raises(ValueError, match="unsupported country flag/origin"):
        prepare_dataset(small_config, tmp_path / "invalid-origin", project_root=tmp_path)


def test_blank_origin_remains_a_known_negative_and_random_generator_type_is_preserved(prepared_dataset: Path) -> None:
    records = load_records(prepared_dataset / "samples.jsonl")
    random_record = next(record for record in records if record.generator_type == "random")
    named_record = next(record for record in records if record.generator_type == "American")
    assert random_record.audit["raw_generator"]["source_type"] is None
    assert named_record.audit["raw_generator"]["source_type"] == "American"
    assert random_record.audit["raw_generator"]["country_of_origin"] is None
    with (prepared_dataset / "audit" / "generator_master.csv").open(encoding="utf-8", newline="") as file:
        raw_rows = list(csv.DictReader(file))
    assert next(row["type"] for row in raw_rows if row["ref"] == random_record.sample_id) == "None"
    report = validate_dataset(prepared_dataset, write_report=False)
    assert report.valid
    real_america = _class_index("RealAmericanPizza")
    assert random_record.pizza_targets[real_america] == 0
    assert random_record.pizza_known_mask[real_america] == 1


@pytest.mark.parametrize("source_type", ["", " ", "\\t", " None ", "none", "unspecified", "mystery"])
def test_preparation_rejects_blank_or_unsupported_generator_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_config: DatasetConfig,
    source_type: str,
) -> None:
    from town_model.dataset import preparation

    def generate_with_type(dataset_dir, config, *, project_root, source_repo=None, progress=None):
        result = fake_generate_raw_dataset(
            dataset_dir, config, project_root=project_root, source_repo=source_repo, progress=progress
        )
        source_csv = dataset_dir / "csv" / "pizzaiolo_dataset.csv"
        with source_csv.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
            fieldnames = list(rows[0])
        random_row = next(row for row in rows if row["type"] == "None")
        random_row["type"] = source_type
        with source_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return result

    monkeypatch.setattr(preparation, "generate_raw_dataset", generate_with_type)
    with pytest.raises(ValueError, match="unsupported generator type"):
        prepare_dataset(small_config, tmp_path / "invalid-type", project_root=tmp_path)


@pytest.mark.parametrize("source_type", ["", " ", "None", "none", "unspecified", "mystery"])
def test_validator_rejects_malformed_stored_generator_type(
    prepared_dataset: Path, source_type: str
) -> None:
    random_index = next(
        index for index, line in enumerate((prepared_dataset / "samples.jsonl").read_text().splitlines())
        if json.loads(line)["generator_type"] == "random"
    )

    def change(record: dict) -> None:
        record["audit"]["raw_generator"]["source_type"] = source_type

    _rewrite_record(prepared_dataset, random_index, change)
    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(
        issue.code == "generator_type_metadata" and issue.sample_id
        for issue in report.errors
    )


def test_validator_checks_literal_random_encoding_in_original_csv(prepared_dataset: Path) -> None:
    raw_master = prepared_dataset / "audit" / "generator_master.csv"
    with raw_master.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
        fieldnames = list(rows[0])
    random_row = next(row for row in rows if row["type"] == "None")
    random_row["type"] = ""
    with raw_master.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_manifest_asset(prepared_dataset, "audit/generator_master.csv")

    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(
        issue.code == "generator_type_source_metadata" and issue.sample_id
        for issue in report.errors
    )


@pytest.mark.parametrize("origin", ["NaN", "nan", "None", "none", "Germany", " "])
def test_validator_rejects_nonblank_origin_sentinels(prepared_dataset: Path, origin: str) -> None:
    random_index = next(
        index for index, line in enumerate((prepared_dataset / "samples.jsonl").read_text().splitlines())
        if json.loads(line)["generator_type"] == "random"
    )

    def change(record: dict) -> None:
        record["audit"]["raw_generator"]["country_of_origin"] = origin

    _rewrite_record(prepared_dataset, random_index, change)
    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code == "unsupported_origin" and item.sample_id for item in report.errors)


def test_blank_origin_with_a_rendered_flag_fails_semantic_validation(prepared_dataset: Path) -> None:
    america_index = next(
        index for index, line in enumerate((prepared_dataset / "samples.jsonl").read_text().splitlines())
        if json.loads(line)["audit"]["raw_generator"]["country_of_origin"] == "America"
    )

    def change(record: dict) -> None:
        record["audit"]["raw_generator"]["country_of_origin"] = None

    _rewrite_record(prepared_dataset, america_index, change)
    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code == "missing_origin_flag" and item.sample_id for item in report.errors)
    assert any(item.code == "component_category_metadata" and item.sample_id for item in report.errors)


def test_component_target_self_class_and_corrected_ancestors(prepared_dataset: Path) -> None:
    classes = build_class_vocabulary()["classes"]
    class_ids = {item["name"]: item["class_id"] for item in classes}
    records = load_records(prepared_dataset / "samples.jsonl")
    ham = next(component for record in records for component in record.components if component.detector_category == "HamTopping")
    assert {name for name, class_id in class_ids.items() if ham.ontology_targets[class_id]} == {
        "HamTopping", "MeatTopping", "PizzaTopping", "Food", "DomainThing"
    }
    deep_pan = next(component for record in records for component in record.components if component.detector_category == "DeepPanBase")
    assert {name for name, class_id in class_ids.items() if deep_pan.ontology_targets[class_id]} == {
        "DeepPanBase", "PizzaBase", "Food", "DomainThing"
    }


def test_class_support_reports_known_positives_negatives_unknowns_by_level_and_split(prepared_dataset: Path) -> None:
    report = validate_dataset(prepared_dataset, write_report=False)
    assert report.valid
    split = report.statistics["class_support_by_split"]["train"]
    assert split["pizza_instances"] == 2
    assert split["component_instances"] > 2
    for item in split["classes"]:
        assert set(item["pizza"]) == {"positive", "known_negative", "unknown"}
        assert set(item["component"]) == {"positive", "known_negative", "unknown"}
        assert sum(item["pizza"].values()) == split["pizza_instances"]
        assert sum(item["component"].values()) == split["component_instances"]
    unsupported = next(item for item in split["classes"] if item["name"] in UNSUPPORTED_CLASS_NAMES)
    assert unsupported["pizza"]["unknown"] == split["pizza_instances"]
    assert unsupported["component"]["unknown"] == split["component_instances"]


def test_repeated_generation_has_identical_ids_and_type_stratified_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_config: DatasetConfig) -> None:
    from town_model.dataset import preparation

    monkeypatch.setattr(preparation, "generate_raw_dataset", fake_generate_raw_dataset)
    first = prepare_dataset(small_config, tmp_path / "first", project_root=tmp_path)
    second = prepare_dataset(small_config, tmp_path / "second", project_root=tmp_path)
    first_records = load_records(first / "samples.jsonl")
    second_records = load_records(second / "samples.jsonl")
    assert first_records == second_records
    assert json.loads((first / "manifest.json").read_text())["dataset_id"] == json.loads((second / "manifest.json").read_text())["dataset_id"]
    assert {record.sample_id: record.split for record in first_records} == {record.sample_id: record.split for record in second_records}


def test_prepared_dataset_is_non_overwriting_and_has_original_source_audit(prepared_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_config: DatasetConfig) -> None:
    from town_model.dataset import preparation

    monkeypatch.setattr(preparation, "generate_raw_dataset", fake_generate_raw_dataset)
    before = (prepared_dataset / "audit" / "generator_master.csv").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_dataset(small_config, prepared_dataset, project_root=tmp_path)
    assert (prepared_dataset / "audit" / "generator_master.csv").read_bytes() == before
    assert (prepared_dataset / "reports" / "validation.json").is_file()
    manifest = json.loads((prepared_dataset / "manifest.json").read_text())
    assert manifest["source"]["generator_commit"] == "6c0bcb221a957c184012beaf103bebcde4b1f793"
    assert manifest["ontology"]["class_count"] == 66


def test_contour_occurrence_count_must_cover_each_source_box(prepared_dataset: Path) -> None:
    relative = "labels/img_00000_contours.json"
    path = prepared_dataset / relative
    contours = json.loads(path.read_text(encoding="utf-8"))
    category = next(iter(contours))
    contours[category] = []
    path.write_text(json.dumps(contours), encoding="utf-8")
    _refresh_manifest_asset(prepared_dataset, relative)

    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code == "contour_occurrence_count" and item.path == relative for item in report.errors)


def test_empty_contour_document_fails_even_with_coherent_manifest_hashes(prepared_dataset: Path) -> None:
    relative = "labels/img_00000_contours.json"
    (prepared_dataset / relative).write_text("{}\n", encoding="utf-8")
    _refresh_manifest_asset(prepared_dataset, relative)

    report = validate_dataset(prepared_dataset, write_report=False)
    assert not report.valid
    assert any(item.code == "contour_category_mismatch" and item.path == relative for item in report.errors)


def test_contour_points_require_pinned_shape_and_in_bounds_coordinates(prepared_dataset: Path) -> None:
    relative = "labels/img_00000_contours.json"
    path = prepared_dataset / relative
    contours = json.loads(path.read_text(encoding="utf-8"))
    category = next(iter(contours))
    contours[category][0] = [[[224, 10]], [[223, 10]]]
    path.write_text(json.dumps(contours), encoding="utf-8")
    _refresh_manifest_asset(prepared_dataset, relative)

    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code == "contour_coordinate_bounds" and item.path == relative for item in report.errors)


def test_contour_point_shape_is_validated(prepared_dataset: Path) -> None:
    relative = "labels/img_00000_contours.json"
    path = prepared_dataset / relative
    contours = json.loads(path.read_text(encoding="utf-8"))
    category = next(iter(contours))
    contours[category][0][0] = [1, 2]
    path.write_text(json.dumps(contours), encoding="utf-8")
    _refresh_manifest_asset(prepared_dataset, relative)

    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code == "contour_structure" and item.path == relative for item in report.errors)


def test_segmentation_categories_must_exist_in_sample_annotations(prepared_dataset: Path) -> None:
    relative = "labels/img_00000_segmentation.txt"
    concepts = json.loads((prepared_dataset / "labels" / "concepts.json").read_text(encoding="utf-8"))
    boxes = json.loads((prepared_dataset / "labels" / "img_00000_bboxes.json").read_text(encoding="utf-8"))
    unexpected_id = next(
        int(concept_id) for concept_id, name in concepts.items()
        if name != "None" and name not in boxes
    )
    path = prepared_dataset / relative
    lines = path.read_text(encoding="utf-8").splitlines()
    values = lines[0].split()
    values[0] = str(unexpected_id)
    lines[0] = " ".join(values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _refresh_manifest_asset(prepared_dataset, relative)

    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code == "segmentation_category_mismatch" and item.path == relative for item in report.errors)


def test_segmentation_dimensions_and_category_ids_are_validated(prepared_dataset: Path) -> None:
    relative = "labels/img_00000_segmentation.txt"
    path = prepared_dataset / relative
    lines = path.read_text(encoding="utf-8").splitlines()
    values = lines[0].split()
    values[0] = "999"
    lines[0] = " ".join(values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _refresh_manifest_asset(prepared_dataset, relative)

    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code == "segmentation_category_id" and item.path == relative for item in report.errors)

    lines = path.read_text(encoding="utf-8").splitlines()[:-1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _refresh_manifest_asset(prepared_dataset, relative)
    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code == "segmentation_dimensions" and item.path == relative for item in report.errors)


def test_pinned_generator_example_annotations_are_accepted() -> None:
    example = Path(__file__).parents[1] / "examples" / "pizza-48"
    report = validate_dataset(example, write_report=False)
    assert report.valid, [item.to_dict() for item in report.errors]


def test_has_topping_set_must_agree_with_positive_counts(prepared_dataset: Path) -> None:
    def change(record: dict) -> None:
        record["audit"]["raw_generator"]["has_topping"] = []
    _rewrite_record(prepared_dataset, 0, change)
    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code == "topping_count_mismatch" and item.sample_id == "img_00000" for item in report.errors)


def test_invalid_topping_count_is_reported_with_sample_and_field(prepared_dataset: Path) -> None:
    def change(record: dict) -> None:
        record["audit"]["raw_generator"]["topping_counts"]["OliveTopping"] = -1
    _rewrite_record(prepared_dataset, 0, change)
    report = validate_dataset(prepared_dataset, write_report=False)
    issue = next(item for item in report.errors if item.code == "invalid_topping_count")
    assert issue.sample_id == "img_00000"
    assert issue.field.endswith("OliveTopping")
    assert any(item.code == "asset_checksum" for item in report.errors)


def test_invalid_base_origin_and_origin_flag_mismatch_are_rejected(prepared_dataset: Path) -> None:
    def change(record: dict) -> None:
        record["audit"]["raw_generator"]["has_base"] = None
        record["audit"]["raw_generator"]["country_of_origin"] = "Germany"
    _rewrite_record(prepared_dataset, 0, change)
    report = validate_dataset(prepared_dataset, write_report=False)
    assert {item.code for item in report.errors}.intersection({"invalid_base", "unsupported_origin", "origin_flag_mismatch"}) == {
        "invalid_base", "unsupported_origin", "origin_flag_mismatch"
    }


def test_bbox_validity_and_annotation_file_references_are_checked(prepared_dataset: Path) -> None:
    def change(record: dict) -> None:
        record["components"][0]["bbox_xyxy_original"] = [-1, 2, 3, 4]
    _rewrite_record(prepared_dataset, 0, change)
    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code == "invalid_bbox" and item.sample_id == "img_00000" for item in report.errors)
    assert any(item.code == "bbox_annotation_mismatch" for item in report.errors)


def test_duplicate_component_occurrence_ids_are_rejected(prepared_dataset: Path) -> None:
    def change(record: dict) -> None:
        record["components"][1]["component_id"] = record["components"][0]["component_id"]
    _rewrite_record(prepared_dataset, 0, change)
    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code == "duplicate_component_id" for item in report.errors)


def test_missing_image_or_annotation_asset_is_actionable(prepared_dataset: Path) -> None:
    image = prepared_dataset / "images" / "img_00000.png"
    image.unlink()
    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code in {"missing_asset", "missing_image"} and "img_00000" in (item.path or "") for item in report.errors)


def test_missing_annotation_reference_is_reported(prepared_dataset: Path) -> None:
    annotation = prepared_dataset / "labels" / "img_00000_bboxes.json"
    annotation.unlink()
    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code in {"missing_asset", "missing_annotation_asset", "bbox_file"} and "img_00000_bboxes.json" in (item.path or "") for item in report.errors)


def test_germany_detector_identity_is_rejected(prepared_dataset: Path) -> None:
    def change(record: dict) -> None:
        record["components"][0]["detector_category"] = "Germany"
    _rewrite_record(prepared_dataset, 0, change)
    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code == "unsupported_detector_category" and item.sample_id == "img_00000" for item in report.errors)


def test_country_origin_without_flag_is_known_negative_but_germany_is_unsupported(prepared_dataset: Path) -> None:
    def change(record: dict) -> None:
        record["audit"]["raw_generator"]["country_of_origin"] = "Germany"
    random_index = next(i for i, line in enumerate((prepared_dataset / "samples.jsonl").read_text().splitlines()) if json.loads(line)["generator_type"] == "random")
    _rewrite_record(prepared_dataset, random_index, change)
    report = validate_dataset(prepared_dataset, write_report=False)
    assert any(item.code == "unsupported_origin" and item.sample_id for item in report.errors)
