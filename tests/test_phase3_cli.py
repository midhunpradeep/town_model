from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from town_model.cli import main


def test_image_cli_runs_and_inspects_an_immutable_smoke_configuration(tmp_path, capsys):
    config = Path(__file__).parents[1] / "configs" / "phase3_image_smoke.json"
    output_root = tmp_path / "image-runs"
    assert main(["experiment", "image", "run", "--config", str(config), "--output", str(output_root)]) == 0
    result = json.loads(capsys.readouterr().out)
    run_path = Path(result["run_directory"])
    assert result["status"] == "completed"
    assert "pre-detector exploratory GT-ROI" in result["training_protocol"]
    assert (run_path / "resolved-config.json").is_file()
    assert (run_path / "snapshots" / "epoch-00002" / "arrays.npz").is_file()

    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    cpu_threads = torch.get_num_threads()
    assert main(["experiment", "image", "evaluate", str(run_path), "--split", "validation", "--checkpoint", "latest"]) == 0
    assert random.getstate() == python_rng
    current_numpy_rng = np.random.get_state()
    assert current_numpy_rng[0] == numpy_rng[0] and np.array_equal(current_numpy_rng[1], numpy_rng[1])
    assert current_numpy_rng[2:] == numpy_rng[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    assert torch.get_num_threads() == cpu_threads
    evaluated = json.loads(capsys.readouterr().out)
    assert evaluated["metrics"]["split"] == "validation"
    assert evaluated["checkpoint_optimizer_step"] == 4

    assert main(["experiment", "image", "inspect", str(run_path)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["manifest"]["state"] == "completed"
    assert inspected["metrics"]["status"] == "completed"
