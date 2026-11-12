from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from town_model.models.town_geometry import (
    house_signed_clearance,
    town_signed_clearance,
)

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "docs/reconstruction/TOWN03_CP2_CONFIRMATION_SET.json"
MANIFEST_SHA256 = "acb1f9c43110ac839530066ef2aa3311755003321b945b65345f3445b4548f3e"
ABSOLUTE_GRADIENT_BUDGET = 1e-6
FINITE_DIFFERENCE_BUDGET = 2e-5


def _tensor(values, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(
        values, dtype=torch.float32, device=device, requires_grad=requires_grad
    )


def _device(name: str) -> torch.device:
    if name == "cuda":
        assert (
            torch.cuda.is_available()
        ), "CP2 confirmation requires actual CUDA execution"
        return torch.device("cuda:0")
    return torch.device("cpu")


def test_frozen_confirmation_manifest_hash_and_inventory() -> None:
    payload = MANIFEST.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == MANIFEST_SHA256
    manifest = json.loads(payload)
    assert manifest["schema"] == "town03-cp2-confirmation-set-v1"
    assert len(manifest["cases"]) == 246
    assert manifest["group_counts"] == {
        "generated": 192,
        "literal": 18,
        "nextafter": 36,
    }
    assert set(manifest["total_configuration_counts"]) == {
        f"d{d}-h{h}-w{w}" for d in (1, 2) for h in (1, 2) for w in (0, 1, 2)
    }


def test_locked_confirmation_runner_passes_forward_and_boolean_budgets(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cp2-results.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/town03_cp2_verify_confirmation_set.py"),
            "--manifest",
            str(MANIFEST),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    results = json.loads(output.read_text(encoding="utf-8"))
    assert results["status"] == "PASS"
    assert results["manifest_sha256"] == MANIFEST_SHA256
    assert results["manifest_case_count"] == 246
    assert results["boolean_mismatches"] == 0
    assert results["failures"] == []


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_cp2_unique_and_native_tied_gradients_match_locked_expectations(
    device_name: str,
) -> None:
    device = _device(device_name)

    points = _tensor([[-0.25, 0.0]], device, True)
    lower = _tensor([-0.5, -1.0], device, True)
    upper = _tensor([1.0, 1.0], device, True)
    empty = torch.empty((0, 2), dtype=torch.float32, device=device)
    score = house_signed_clearance(points, lower, upper, empty, empty).sum()
    point_grad, lower_grad, upper_grad = torch.autograd.grad(
        score, (points, lower, upper)
    )
    assert abs(score.item() - 0.25) <= ABSOLUTE_GRADIENT_BUDGET
    assert torch.allclose(
        point_grad, _tensor([[1.0, 0.0]], device), atol=ABSOLUTE_GRADIENT_BUDGET, rtol=0
    )
    assert torch.allclose(
        lower_grad, _tensor([-1.0, 0.0], device), atol=ABSOLUTE_GRADIENT_BUDGET, rtol=0
    )
    assert torch.allclose(
        upper_grad, _tensor([0.0, 0.0], device), atol=ABSOLUTE_GRADIENT_BUDGET, rtol=0
    )

    points = _tensor([[0.4, 0.0]], device, True)
    frame_lower = _tensor([-1.0, -1.0], device, True)
    frame_upper = _tensor([1.0, 1.0], device, True)
    window_lower = _tensor([[-0.5, -0.5]], device, True)
    window_upper = _tensor([[0.5, 0.5]], device, True)
    score = house_signed_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper
    ).sum()
    grads = torch.autograd.grad(
        score, (points, frame_lower, frame_upper, window_lower, window_upper)
    )
    expected = (
        _tensor([[1.0, 0.0]], device),
        _tensor([0.0, 0.0], device),
        _tensor([0.0, 0.0], device),
        _tensor([[0.0, 0.0]], device),
        _tensor([[-1.0, 0.0]], device),
    )
    assert abs(score.item() - (-0.1)) <= ABSOLUTE_GRADIENT_BUDGET
    assert all(
        torch.allclose(actual, wanted, atol=ABSOLUTE_GRADIENT_BUDGET, rtol=0)
        for actual, wanted in zip(grads, expected, strict=True)
    )

    points = _tensor([[0.0]], device, True)
    lower = _tensor([-1.0], device, True)
    upper = _tensor([1.0], device, True)
    no_windows = torch.empty((0, 1), dtype=torch.float32, device=device)
    score = house_signed_clearance(points, lower, upper, no_windows, no_windows).sum()
    point_grad, lower_grad, upper_grad = torch.autograd.grad(
        score, (points, lower, upper)
    )
    assert torch.allclose(
        point_grad, _tensor([[0.0]], device), atol=ABSOLUTE_GRADIENT_BUDGET, rtol=0
    )
    assert torch.allclose(
        lower_grad, _tensor([-0.5], device), atol=ABSOLUTE_GRADIENT_BUDGET, rtol=0
    )
    assert torch.allclose(
        upper_grad, _tensor([0.5], device), atol=ABSOLUTE_GRADIENT_BUDGET, rtol=0
    )

    points = _tensor([[0.0]], device, True)
    frame_lower = _tensor([0.0], device, True)
    frame_upper = _tensor([1.0], device, True)
    window_lower = _tensor([[0.0]], device, True)
    window_upper = _tensor([[0.5]], device, True)
    score = house_signed_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper
    ).sum()
    point_grad, frame_lower_grad, window_lower_grad = torch.autograd.grad(
        score, (points, frame_lower, window_lower)
    )
    assert abs(score.item()) <= ABSOLUTE_GRADIENT_BUDGET
    assert torch.allclose(
        point_grad, _tensor([[0.0]], device), atol=ABSOLUTE_GRADIENT_BUDGET, rtol=0
    )
    assert torch.allclose(
        frame_lower_grad, _tensor([-0.5], device), atol=ABSOLUTE_GRADIENT_BUDGET, rtol=0
    )
    assert torch.allclose(
        window_lower_grad,
        _tensor([[0.5]], device),
        atol=ABSOLUTE_GRADIENT_BUDGET,
        rtol=0,
    )

    points = _tensor([[0.0]], device, True)
    lower = _tensor([[-1.0], [0.0]], device, True)
    upper = _tensor([[0.0], [1.0]], device, True)
    no_town_windows = torch.empty((2, 0, 1), dtype=torch.float32, device=device)
    score = town_signed_clearance(
        points, lower, upper, no_town_windows, no_town_windows
    ).sum()
    point_grad, lower_grad, upper_grad = torch.autograd.grad(
        score, (points, lower, upper)
    )
    assert torch.allclose(
        point_grad, _tensor([[0.0]], device), atol=ABSOLUTE_GRADIENT_BUDGET, rtol=0
    )
    assert torch.allclose(
        lower_grad,
        _tensor([[0.0], [-0.5]], device),
        atol=ABSOLUTE_GRADIENT_BUDGET,
        rtol=0,
    )
    assert torch.allclose(
        upper_grad,
        _tensor([[0.5], [0.0]], device),
        atol=ABSOLUTE_GRADIENT_BUDGET,
        rtol=0,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_cp2_non_tie_finite_differences_meet_locked_budget(device_name: str) -> None:
    device = _device(device_name)
    point_x = torch.tensor(
        -0.25, dtype=torch.float32, device=device, requires_grad=True
    )
    lower_x = torch.tensor(-0.5, dtype=torch.float32, device=device, requires_grad=True)
    points = torch.stack((point_x, torch.tensor(0.0, device=device))).reshape(1, 2)
    lower = torch.stack((lower_x, torch.tensor(-1.0, device=device)))
    upper = _tensor([1.0, 1.0], device)
    empty = torch.empty((0, 2), dtype=torch.float32, device=device)
    score = house_signed_clearance(points, lower, upper, empty, empty).sum()
    point_grad, lower_grad = torch.autograd.grad(score, (point_x, lower_x))
    step = torch.tensor(1e-3, dtype=torch.float32, device=device)

    def evaluate(point_value: torch.Tensor, lower_value: torch.Tensor) -> torch.Tensor:
        shifted_points = torch.stack(
            (point_value, torch.tensor(0.0, device=device))
        ).reshape(1, 2)
        shifted_lower = torch.stack((lower_value, torch.tensor(-1.0, device=device)))
        return house_signed_clearance(
            shifted_points, shifted_lower, upper, empty, empty
        ).sum()

    point_fd = (
        evaluate(point_x.detach() + step, lower_x.detach())
        - evaluate(point_x.detach() - step, lower_x.detach())
    ) / (2 * step)
    lower_fd = (
        evaluate(point_x.detach(), lower_x.detach() + step)
        - evaluate(point_x.detach(), lower_x.detach() - step)
    ) / (2 * step)
    assert abs(point_grad.item() - 1.0) <= ABSOLUTE_GRADIENT_BUDGET
    assert abs(lower_grad.item() + 1.0) <= ABSOLUTE_GRADIENT_BUDGET
    assert abs(point_fd.item() - 1.0) <= FINITE_DIFFERENCE_BUDGET
    assert abs(lower_fd.item() + 1.0) <= FINITE_DIFFERENCE_BUDGET
    if device.type == "cuda":
        torch.cuda.synchronize(device)
