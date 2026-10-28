from __future__ import annotations

import json
from pathlib import Path

from town_model.application import inspect_dataset, load_explorer, validate_dataset_operation
from town_model.cli import main
from town_model.dashboard.data import find_sample, load_dashboard_data


def test_cli_and_dashboard_use_the_same_dataset_service(prepared_dataset: Path, capsys) -> None:
    service_summary = inspect_dataset(prepared_dataset).to_dict()
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


def test_dashboard_inspection_is_framework_independent(prepared_dataset: Path) -> None:
    explorer = load_explorer(prepared_dataset)
    assert explorer.summary.schema_version == "1.0.0"
    assert len(explorer.records) == 6
    assert len(explorer.classes) == 66
    assert len(explorer.detector_categories) == 22
