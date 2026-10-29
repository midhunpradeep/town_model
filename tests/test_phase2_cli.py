from __future__ import annotations

import json
from pathlib import Path

from town_model.cli import main


def test_cli_runs_both_fixed_fixtures_and_evaluates_inspects_exports(tmp_path: Path, capsys) -> None:
    runs_root = tmp_path / "runs"
    assert main(["experiment", "fixed", "--output", str(runs_root), "--seed", "2", "--points", "30", "--steps", "1", "--snapshot-interval", "1"]) == 0
    capsys.readouterr()
    groups = sorted(path for path in runs_root.iterdir() if path.is_dir())
    assert len(groups) == 2
    for group in groups:
        assert (group / "baseline" / "checkpoint.pt").is_file()
        assert (group / "box" / "checkpoint.pt").is_file()
        assert (group / "town" / "checkpoint.pt").is_file()
    run = groups[0] / "town"
    assert main(["evaluate", str(run), "--split", "validation"]) == 0
    evaluated = json.loads(capsys.readouterr().out)
    assert evaluated["objective"] == "signed_distance"
    assert evaluated["split"] == "validation"
    assert main(["inspect-run", str(run)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["metrics"]["family"] == "town"
    export = tmp_path / "geometry.json"
    assert main(["export-geometry", str(run), "--output", str(export)]) == 0
    payload = json.loads(export.read_text())
    assert payload["family"] == "town"
    assert payload["geometry"]["window_mode"] == "contained"
    assert len(payload["geometry"]["frame_low"]) == 2


def test_cli_mapping_command_creates_all_model_family_runs(tmp_path: Path, capsys) -> None:
    output = tmp_path / "mapped"
    assert main(["experiment", "mapping", "--output", str(output), "--seed", "7", "--points", "30", "--steps", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    group = Path(result["run_groups"][0])
    assert (group / "baseline" / "checkpoint.pt").is_file()
    assert (group / "box" / "checkpoint.pt").is_file()
    assert (group / "town" / "checkpoint.pt").is_file()
