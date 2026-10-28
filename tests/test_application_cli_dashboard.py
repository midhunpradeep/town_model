from __future__ import annotations

import json
from pathlib import Path

from town_model.application import inspect_dataset, load_explorer, validate_dataset_operation
from town_model.cli import main
from town_model.dashboard.data import find_sample, load_dashboard_data


def test_cli_and_dashboard_use_the_same_dataset_service(prepared_dataset: Path, capsys) -> None:
    service_summary = inspect_dataset(prepared_dataset).to_dict()
    assert service_summary["integrity_status"] == "current_pass"
    assert service_summary["sample_count"] == 6
    assert main(["inspect", str(prepared_dataset)]) == 0
    cli_summary = json.loads(capsys.readouterr().out)
    assert cli_summary == service_summary

    report = validate_dataset_operation(prepared_dataset, write_report=False)
    assert main(["validate", str(prepared_dataset), "--no-write-report", "--json"]) == 0
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["valid"] == report.valid
    assert cli_report["dataset_id"] == report.dataset_id

    loaded = load_dashboard_data(prepared_dataset)
    assert loaded.error is None
    assert loaded.explorer is not None
    assert loaded.explorer.summary.dataset_id == service_summary["dataset_id"]
    assert find_sample(loaded.explorer, "img_00000") is not None


def test_dashboard_data_access_reports_missing_or_corrupt_paths() -> None:
    missing = load_dashboard_data(Path("/definitely/not/a/dataset"))
    assert missing.explorer is None
    assert "Could not load dataset" in (missing.error or "")


def test_cli_config_overrides_are_validated_before_calling_shared_generation_service(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import town_model.cli as cli

    config_path = tmp_path / "dataset.toml"
    config_path.write_text('total_instances = 6\npizza_types = ["American", "random"]\n', encoding="utf-8")
    called: dict[str, object] = {}

    def fake_service(config, output, *, source_repo=None, progress=None):
        called["config"] = config.resolved_dict()
        called["output"] = output
        return output

    monkeypatch.setattr(cli, "generate_dataset", fake_service)
    result = main(["generate", "--config", str(config_path), "--output", str(tmp_path / "out"), "--generation-seed", "91"])
    assert result == 0
    capsys.readouterr()
    assert called["config"]["generation_seed"] == 91
    assert called["config"]["pizza_type_counts"] == {"American": 3, "random": 3}
    assert called["output"] == tmp_path / "out"


def test_dashboard_background_generation_reports_progress_and_errors(tmp_path: Path, small_config, monkeypatch) -> None:
    import queue
    import town_model.dashboard.runtime as runtime

    events: list[tuple[float, str]] = []

    def successful_generation(config, output, *, progress=None):
        progress(0.5, "halfway")
        return output

    monkeypatch.setattr(runtime, "generate_dataset", successful_generation)
    progress_queue: queue.Queue[tuple[float, str]] = queue.Queue()
    result = runtime.submit_generation(small_config, tmp_path / "result", progress_queue).result(timeout=2)
    while not progress_queue.empty():
        events.append(progress_queue.get_nowait())
    assert result == tmp_path / "result"
    assert events == [(0.5, "halfway")]

    def failed_generation(config, output, *, progress=None):
        raise RuntimeError("generator error fixture")

    monkeypatch.setattr(runtime, "generate_dataset", failed_generation)
    future = runtime.submit_generation(small_config, tmp_path / "failed", queue.Queue())
    try:
        future.result(timeout=2)
    except RuntimeError as error:
        assert "generator error fixture" in str(error)
    else:
        raise AssertionError("background generation failure was swallowed")


def test_inspection_and_dashboard_do_not_trust_saved_validation_after_sample_mutation(
    prepared_dataset: Path, capsys
) -> None:
    manifest_before = (prepared_dataset / "manifest.json").read_bytes()
    saved_report_before = (prepared_dataset / "reports" / "validation.json").read_bytes()
    samples_path = prepared_dataset / "samples.jsonl"
    records = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
    records[0]["pizza"]["ontology_targets"][0] = 1 - records[0]["pizza"]["ontology_targets"][0]
    samples_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    explicit_report = validate_dataset_operation(prepared_dataset, write_report=False)
    assert not explicit_report.valid
    assert any(issue.code in {"asset_checksum", "pizza_targets", "statistics_mismatch"} for issue in explicit_report.errors)

    summary = inspect_dataset(prepared_dataset)
    assert summary.integrity_status == "current_fail"
    assert summary.validation is not None and summary.validation["valid"] is False
    assert summary.validation["errors"]
    assert summary.validation["validator_version"] == "1.1.0"
    assert summary.saved_validation is not None and summary.saved_validation["valid"] is True
    assert summary.to_dict()["validation_scope"] == "current_artifact_state"
    assert summary.to_dict()["saved_validation_scope"] == "historical_only"
    assert summary.class_support_by_split == {}
    assert summary.sample_count is None
    assert (prepared_dataset / "manifest.json").read_bytes() == manifest_before
    assert (prepared_dataset / "reports" / "validation.json").read_bytes() == saved_report_before

    assert main(["inspect", str(prepared_dataset), "--json"]) == 1
    cli_summary = json.loads(capsys.readouterr().out)
    assert cli_summary["integrity_status"] == "current_fail"
    assert cli_summary["class_support_by_split"] == {}
    assert cli_summary["saved_validation"]["valid"] is True
    assert cli_summary["saved_validation_scope"] == "historical_only"
    assert cli_summary["validation"]["valid"] is False
    assert cli_summary["validation_scope"] == "current_artifact_state"

    loaded = load_dashboard_data(prepared_dataset)
    assert loaded.error is None
    assert loaded.explorer is not None
    assert loaded.explorer.summary.integrity_status == "current_fail"
    assert loaded.explorer.records == ()


def test_inspection_reuses_validation_only_while_declared_file_state_is_unchanged(
    prepared_dataset: Path, monkeypatch
) -> None:
    import town_model.application as application

    original = application.validate_dataset
    calls = 0

    def count_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(application, "validate_dataset", count_validation)
    assert inspect_dataset(prepared_dataset).integrity_status == "current_pass"
    assert inspect_dataset(prepared_dataset).integrity_status == "current_pass"
    assert calls == 1

    samples_path = prepared_dataset / "samples.jsonl"
    samples_path.write_text(samples_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert inspect_dataset(prepared_dataset).integrity_status == "current_fail"
    assert calls == 2


def test_change_before_validation_is_rejected(prepared_dataset: Path) -> None:
    samples = prepared_dataset / "samples.jsonl"
    records = [json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines()]
    records[0]["pizza"]["ontology_targets"][0] = 1 - records[0]["pizza"]["ontology_targets"][0]
    samples.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    explorer = load_explorer(prepared_dataset)
    assert explorer.summary.integrity_status == "current_fail"
    assert explorer.records == ()


def test_change_after_validation_before_explorer_read_is_rejected(
    prepared_dataset: Path, monkeypatch
) -> None:
    import town_model.application as application

    original = application._load_explorer_snapshot
    changed = False

    def mutate_after_validation(entry):
        nonlocal changed
        if not changed:
            path = prepared_dataset / "samples.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            records[0]["pizza"]["ontology_targets"][0] = 1 - records[0]["pizza"]["ontology_targets"][0]
            path.write_text(
                "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )
            changed = True
        return original(entry)

    monkeypatch.setattr(application, "_load_explorer_snapshot", mutate_after_validation)
    explorer = load_explorer(prepared_dataset)
    assert changed
    assert explorer.summary.integrity_status == "current_fail"
    assert explorer.records == ()


def test_manifest_change_after_validation_is_rejected(
    prepared_dataset: Path, monkeypatch
) -> None:
    import town_model.application as application

    original = application._load_explorer_snapshot
    changed = False

    def mutate_manifest_after_validation(entry):
        nonlocal changed
        if not changed:
            manifest = prepared_dataset / "manifest.json"
            manifest.write_bytes(b"{ malformed manifest")
            changed = True
        return original(entry)

    monkeypatch.setattr(application, "_load_explorer_snapshot", mutate_manifest_after_validation)
    explorer = load_explorer(prepared_dataset)
    assert changed
    assert explorer.summary.integrity_status == "current_fail"
    assert explorer.records == ()


def test_vocabulary_change_after_validation_is_rejected(
    prepared_dataset: Path, monkeypatch
) -> None:
    import town_model.application as application

    original = application._read_declared_artifact
    changed = False
    class_path = "vocabularies/ontology_classes.json"

    def mutate_vocabulary_after_validation(entry, relative):
        nonlocal changed
        if relative == class_path and not changed:
            vocabulary = prepared_dataset / relative
            vocabulary.write_bytes(b"{ malformed vocabulary")
            changed = True
        return original(entry, relative)

    monkeypatch.setattr(application, "_read_declared_artifact", mutate_vocabulary_after_validation)
    explorer = load_explorer(prepared_dataset)
    assert changed
    assert explorer.summary.integrity_status == "current_fail"
    assert explorer.records == ()


def test_change_between_vocabulary_and_sample_loading_is_rejected(
    prepared_dataset: Path, monkeypatch
) -> None:
    import town_model.application as application

    original = application._read_declared_artifact
    changed = False

    def mutate_before_samples(entry, relative):
        nonlocal changed
        if relative == "samples.jsonl" and not changed:
            path = prepared_dataset / relative
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            records[0]["pizza"]["ontology_targets"][0] = 1 - records[0]["pizza"]["ontology_targets"][0]
            path.write_text(
                "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )
            changed = True
        return original(entry, relative)

    monkeypatch.setattr(application, "_read_declared_artifact", mutate_before_samples)
    explorer = load_explorer(prepared_dataset)
    assert changed
    assert explorer.summary.integrity_status == "current_fail"
    assert explorer.records == ()


def test_change_before_selected_image_open_is_rejected(prepared_dataset: Path) -> None:
    import pytest
    from town_model.application import DatasetSnapshotChangedError, load_verified_sample_image

    explorer = load_explorer(prepared_dataset)
    record = explorer.records[0]
    image = prepared_dataset / record.image_path
    image.write_bytes(image.read_bytes() + b"changed")

    with pytest.raises(DatasetSnapshotChangedError, match="changed after validation|changed after"):
        load_verified_sample_image(explorer, record)
    assert inspect_dataset(prepared_dataset).integrity_status == "current_fail"


def test_failed_explicit_validation_discards_previously_successful_explorer(
    prepared_dataset: Path,
) -> None:
    from town_model.application import DatasetSnapshotChangedError, load_verified_sample_image
    import pytest

    previous = load_explorer(prepared_dataset)
    samples = prepared_dataset / "samples.jsonl"
    samples.write_text(samples.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = validate_dataset_operation(prepared_dataset, write_report=False)
    assert not report.valid
    failed = load_explorer(prepared_dataset)
    assert failed.summary.integrity_status == "current_fail"
    assert failed.records == ()
    assert failed.snapshot_token != previous.snapshot_token
    with pytest.raises(DatasetSnapshotChangedError, match="snapshot has expired"):
        load_verified_sample_image(previous, previous.records[0])


def test_unchanged_explorer_bundle_is_reused_but_selected_assets_are_verified(
    prepared_dataset: Path, monkeypatch
) -> None:
    import town_model.application as application

    first = load_explorer(prepared_dataset)
    original_validate = application.validate_dataset
    original_parse = application.load_records_bytes
    calls = {"validate": 0, "parse": 0, "asset_reads": 0}

    def count_validate(*args, **kwargs):
        calls["validate"] += 1
        return original_validate(*args, **kwargs)

    def count_parse(*args, **kwargs):
        calls["parse"] += 1
        return original_parse(*args, **kwargs)

    def count_asset_reads(entry, relative):
        calls["asset_reads"] += 1
        return original_read(entry, relative)

    original_read = application._read_declared_artifact
    monkeypatch.setattr(application, "validate_dataset", count_validate)
    monkeypatch.setattr(application, "load_records_bytes", count_parse)
    monkeypatch.setattr(application, "_read_declared_artifact", count_asset_reads)

    assert load_explorer(prepared_dataset) is first
    assert application.inspect_dataset(prepared_dataset).integrity_status == "current_pass"
    assert calls == {"validate": 0, "parse": 0, "asset_reads": 0}

    from town_model.application import load_verified_sample_image

    load_verified_sample_image(first, first.records[0])
    assert calls["validate"] == calls["parse"] == 0
    assert calls["asset_reads"] == 4  # three sample annotations and its image, not all dataset files


def test_inspection_distinguishes_current_and_saved_validation_on_disposable_fixture(
    prepared_dataset: Path,
) -> None:
    report_path = prepared_dataset / "reports" / "validation.json"
    historical = json.loads(report_path.read_text(encoding="utf-8"))
    historical["validator_version"] = "1.0.0"
    report_path.write_text(json.dumps(historical, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = inspect_dataset(prepared_dataset)
    assert summary.integrity_status == "current_pass"
    assert summary.validation is not None and summary.validation["validator_version"] == "1.1.0"
    assert summary.saved_validation is not None and summary.saved_validation["validator_version"] == "1.0.0"


def test_dashboard_inspection_is_framework_independent(prepared_dataset: Path) -> None:
    explorer = load_explorer(prepared_dataset)
    assert explorer.summary.schema_version == "1.0.0"
    assert len(explorer.records) == 6
    assert len(explorer.classes) == 66
    assert len(explorer.detector_categories) == 22
