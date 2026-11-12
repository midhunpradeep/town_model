#!/usr/bin/env python3
"""Reproduce bounded TOWN-03 CP1 numerical characterization (not acceptance)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, localcontext
import json
import platform
import random
import struct
import subprocess
from pathlib import Path
from typing import Any

import torch

from town_model.models.town_geometry import (
    house_signed_clearance,
    town_signed_clearance,
)

SEED = 20261112
CASE_COUNT = 128
NEAR_ZERO = Decimal("0.000001")


@dataclass(frozen=True)
class GeometryCase:
    name: str
    target: str
    points: list[list[float]]
    frame_lower: list[list[float]]
    frame_upper: list[list[float]]
    window_lower: list[list[list[float]]]
    window_upper: list[list[list[float]]]
    expected: list[float | str] | None = None
    source: str = "literal"

    @property
    def dimensions(self) -> int:
        return len(self.frame_lower[0])

    @property
    def houses(self) -> int:
        return len(self.frame_lower)

    @property
    def windows(self) -> int:
        return len(self.window_lower[0])


def _case(
    name: str,
    target: str,
    points: list[list[float]],
    frame_lower: list[list[float]],
    frame_upper: list[list[float]],
    window_lower: list[list[list[float]]],
    window_upper: list[list[list[float]]],
    expected: list[float | str] | None = None,
    source: str = "literal",
) -> GeometryCase:
    return GeometryCase(
        name,
        target,
        points,
        frame_lower,
        frame_upper,
        window_lower,
        window_upper,
        expected,
        source,
    )


def _exact_fixtures() -> tuple[GeometryCase, ...]:
    maximum = torch.finfo(torch.float32).max
    return (
        _case(
            "house-1d-no-window-interior",
            "house",
            [[0.25]],
            [[-1.0]],
            [[1.0]],
            [[]],
            [[]],
            [0.75],
        ),
        _case(
            "house-1d-closed-frame-boundary",
            "house",
            [[-1.0]],
            [[-1.0]],
            [[1.0]],
            [[]],
            [[]],
            [0.0],
        ),
        _case(
            "house-1d-finite-outside-query",
            "house",
            [[-1.5]],
            [[-1.0]],
            [[1.0]],
            [[]],
            [[]],
            [-0.5],
        ),
        _case(
            "house-1d-window-interior",
            "house",
            [[0.0]],
            [[-1.0]],
            [[1.0]],
            [[[-0.25]]],
            [[[0.25]]],
            [-0.25],
        ),
        _case(
            "house-1d-window-boundary",
            "house",
            [[0.25]],
            [[-1.0]],
            [[1.0]],
            [[[-0.25]]],
            [[[0.25]]],
            [0.0],
        ),
        _case(
            "house-1d-degenerate-window",
            "house",
            [[0.0]],
            [[-1.0]],
            [[1.0]],
            [[[0.0]]],
            [[[0.0]]],
            [0.0],
        ),
        _case(
            "house-2d-one-axis-degenerate-window",
            "house",
            [[0.0, 0.0]],
            [[-1.0, -1.0]],
            [[1.0, 1.0]],
            [[[0.0, -0.75]]],
            [[[0.0, 0.75]]],
            [0.0],
        ),
        _case(
            "house-2d-degenerate-frame",
            "house",
            [[0.0, 0.0]],
            [[0.0, 0.0]],
            [[0.0, 0.0]],
            [[]],
            [[]],
            [0.0],
        ),
        _case(
            "house-2d-h1-w0-box-equivalence",
            "house",
            [[0.0, 0.0]],
            [[-1.0, -0.5]],
            [[0.5, 1.0]],
            [[]],
            [[]],
            [0.5],
        ),
        _case(
            "house-1d-overlapping-windows",
            "house",
            [[0.5]],
            [[-1.0]],
            [[1.0]],
            [[[-0.5], [0.0]]],
            [[[0.25], [0.75]]],
            [-0.25],
        ),
        _case(
            "house-2d-protruding-window",
            "house",
            [[0.5, 0.5]],
            [[-0.5, -0.5]],
            [[0.5, 0.5]],
            [[[-0.75, -0.75]]],
            [[[0.75, 0.75]]],
            [-0.25],
        ),
        _case(
            "town-1d-two-house-union-positive",
            "town",
            [[0.25]],
            [[-1.0], [0.0]],
            [[0.0], [1.0]],
            [[], []],
            [[], []],
            [0.25],
        ),
        _case(
            "town-1d-touching-house-tie",
            "town",
            [[0.0]],
            [[-1.0], [0.0]],
            [[0.0], [1.0]],
            [[], []],
            [[], []],
            [0.0],
        ),
        _case(
            "town-1d-window-interior",
            "town",
            [[0.0]],
            [[-1.0]],
            [[1.0]],
            [[[-0.25]]],
            [[[0.25]]],
            [-0.25],
        ),
        _case(
            "town-1d-two-windowed-houses",
            "town",
            [[-0.375]],
            [[-1.0], [0.0]],
            [[0.0], [1.0]],
            [[[-0.5]], [[0.25]]],
            [[[-0.25]], [[0.5]]],
            [-0.125],
        ),
        _case(
            "town-2d-window-boundary-adjacent",
            "town",
            [[0.0, 0.75]],
            [[-1.0, -1.0]],
            [[1.0, 1.0]],
            [[[-0.5, -0.5]]],
            [[[0.5, 0.5]]],
            [0.25],
        ),
        _case(
            "house-1d-largest-finite-outside-query",
            "house",
            [[maximum]],
            [[-1.0]],
            [[1.0]],
            [[]],
            [[]],
            ["-340282346638528859811704183484516925439"],
        ),
        _case(
            "town-1d-largest-finite-outside-query",
            "town",
            [[-maximum]],
            [[-1.0]],
            [[1.0]],
            [[]],
            [[]],
            ["-340282346638528859811704183484516925439"],
        ),
    )


def _stored_tensors(
    case: GeometryCase, device: torch.device
) -> tuple[torch.Tensor, ...]:
    d = case.dimensions
    h = case.houses
    w = case.windows
    points = torch.tensor(case.points, dtype=torch.float32, device=device).reshape(
        -1, d
    )
    frame_lower = torch.tensor(
        case.frame_lower, dtype=torch.float32, device=device
    ).reshape(h, d)
    frame_upper = torch.tensor(
        case.frame_upper, dtype=torch.float32, device=device
    ).reshape(h, d)
    window_lower = torch.tensor(
        case.window_lower, dtype=torch.float32, device=device
    ).reshape(h, w, d)
    window_upper = torch.tensor(
        case.window_upper, dtype=torch.float32, device=device
    ).reshape(h, w, d)
    return points, frame_lower, frame_upper, window_lower, window_upper


def _dec(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def _box_oracle(point: list[float], lower: list[float], upper: list[float]) -> Decimal:
    faces = [_dec(x) - _dec(lo) for x, lo in zip(point, lower, strict=True)]
    faces.extend(_dec(hi) - _dec(x) for x, hi in zip(point, upper, strict=True))
    return min(faces)


def _house_oracle(
    point: list[float],
    frame_lower: list[float],
    frame_upper: list[float],
    window_lower: list[list[float]],
    window_upper: list[list[float]],
) -> Decimal:
    frame_score = _box_oracle(point, frame_lower, frame_upper)
    return min(
        [frame_score]
        + [
            -_box_oracle(point, lo, hi)
            for lo, hi in zip(window_lower, window_upper, strict=True)
        ]
    )


def _oracle(case: GeometryCase) -> tuple[list[Decimal], list[Decimal]]:
    with localcontext() as context:
        context.prec = 100
        points, frame_lower, frame_upper, window_lower, window_upper = _stored_tensors(
            case, torch.device("cpu")
        )
        p_rows = points.tolist()
        fl_rows = frame_lower.tolist()
        fu_rows = frame_upper.tolist()
        wl_rows = window_lower.tolist()
        wu_rows = window_upper.tolist()
        house_values: list[Decimal] = []
        town_values: list[Decimal] = []
        for point in p_rows:
            by_house = [
                _house_oracle(point, fl, fu, wl, wu)
                for fl, fu, wl, wu in zip(
                    fl_rows, fu_rows, wl_rows, wu_rows, strict=True
                )
            ]
            house_values.append(by_house[0])
            town_values.append(max(by_house))
        return house_values, town_values


def _f32(value: Decimal) -> float:
    return torch.tensor(float(value), dtype=torch.float32).item()


def _ordered_bits(value: float) -> int:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    if bits & 0x80000000:
        return 0x80000000 - (bits & 0x7FFFFFFF)
    return bits | 0x80000000


def _ulp_distance(left: float, right: float) -> int:
    return abs(_ordered_bits(left) - _ordered_bits(right))


def _sign(value: Decimal | float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _score(case: GeometryCase, device: torch.device) -> tuple[list[float], list[float]]:
    tensors = _stored_tensors(case, device)
    points, frame_lower, frame_upper, window_lower, window_upper = tensors
    house = house_signed_clearance(
        points,
        frame_lower[0],
        frame_upper[0],
        window_lower[0],
        window_upper[0],
    )
    town = town_signed_clearance(*tensors)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return house.detach().cpu().tolist(), town.detach().cpu().tolist()


def _summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
    abs_errors = [float(item["abs_error"]) for item in values]
    relative_errors = [float(item["relative_error"]) for item in values]
    ulps = [int(item["ulp_error"]) for item in values]
    crossings = sum(bool(item["sign_or_zero_crossing"]) for item in values)
    near = [item for item in values if item["near_zero"]]
    return {
        "outputs": len(values),
        "max_absolute_error": max(abs_errors, default=0.0),
        "mean_absolute_error": sum(abs_errors) / len(abs_errors) if abs_errors else 0.0,
        "max_relative_error": max(relative_errors, default=0.0),
        "max_ulp_error_vs_correctly_rounded_oracle": max(ulps, default=0),
        "sign_or_zero_crossings": crossings,
        "reference_outputs_with_abs_le_1e-6": len(near),
        "near_zero_max_absolute_error": max(
            (float(item["abs_error"]) for item in near), default=0.0
        ),
    }


def _measurement(actual: float, reference: Decimal) -> dict[str, Any]:
    with localcontext() as context:
        context.prec = 100
        error = abs(_dec(actual) - reference)
        relative_error = error / max(Decimal(1), abs(reference))
    rounded = _f32(reference)
    return {
        "actual_float32": actual,
        "oracle_decimal": str(reference),
        "rounded_oracle_float32": rounded,
        "abs_error": str(error),
        "relative_error": str(relative_error),
        "ulp_error": _ulp_distance(actual, rounded),
        "near_zero": abs(reference) <= NEAR_ZERO,
        "sign_or_zero_crossing": _sign(actual) != _sign(reference),
        "exact_zero_mismatch": (actual == 0.0) != (reference == 0),
    }


def _random_decimal(rng: random.Random, low: float, high: float) -> float:
    return float(f"{rng.uniform(low, high):.9f}")


def _boundary_case(name: str, d: int, h: int, w: int, side: int) -> GeometryCase:
    lo = [-0.6137, -0.3719][:d]
    hi = [0.8123, 0.7331][:d]
    lower_tensor = torch.tensor(lo, dtype=torch.float32)
    upper_tensor = torch.tensor(hi, dtype=torch.float32)
    points: list[float] = []
    for axis, (a, b) in enumerate(zip(lower_tensor, upper_tensor, strict=True)):
        if axis == 0 and side != 0:
            direction = torch.tensor(
                float("inf") if side > 0 else float("-inf"), dtype=torch.float32
            )
            points.append(torch.nextafter(a, direction).item())
        else:
            points.append(
                torch.tensor((a.item() + b.item()) / 2, dtype=torch.float32).item()
            )
    house_lowers = [lo.copy() for _ in range(h)]
    house_uppers = [hi.copy() for _ in range(h)]
    window_lowers: list[list[list[float]]] = []
    window_uppers: list[list[list[float]]] = []
    for _ in range(h):
        lows: list[list[float]] = []
        highs: list[list[float]] = []
        for window_index in range(w):
            bound_lo = 0.25 if window_index == 0 else 0.55
            bound_hi = 0.5 if window_index == 0 else 0.75
            lows.append([bound_lo] * d)
            highs.append([bound_hi] * d)
        window_lowers.append(lows)
        window_uppers.append(highs)
    return _case(
        name,
        "town",
        [points],
        house_lowers,
        house_uppers,
        window_lowers,
        window_uppers,
        source="nextafter-boundary",
    )


def _random_case(
    rng: random.Random, index: int, d: int, h: int, w: int
) -> GeometryCase:
    frame_lower: list[list[float]] = []
    frame_upper: list[list[float]] = []
    window_lower: list[list[list[float]]] = []
    window_upper: list[list[list[float]]] = []
    for _ in range(h):
        fl = [_random_decimal(rng, -0.9, 0.2) for _ in range(d)]
        fu = [_random_decimal(rng, lo + 0.04, 0.98) for lo in fl]
        frame_lower.append(fl)
        frame_upper.append(fu)
        lows: list[list[float]] = []
        highs: list[list[float]] = []
        for _ in range(w):
            wl = [_random_decimal(rng, -0.98, 0.75) for _ in range(d)]
            wu = [_random_decimal(rng, lo + 0.02, 0.99) for lo in wl]
            lows.append(wl)
            highs.append(wu)
        window_lower.append(lows)
        window_upper.append(highs)
    point = [_random_decimal(rng, -1.5, 1.5) for _ in range(d)]
    return _case(
        f"random-{index:03d}-d{d}-h{h}-w{w}",
        "town",
        [point],
        frame_lower,
        frame_upper,
        window_lower,
        window_upper,
        source="seeded-non-simple-float32",
    )


def _comparison_cases() -> tuple[GeometryCase, ...]:
    configs = [(d, h, w) for d in (1, 2) for h in (1, 2) for w in (0, 1, 2)]
    cases: list[GeometryCase] = []
    for d, h, w in configs:
        for side, label in ((-1, "below"), (0, "exact"), (1, "above")):
            cases.append(
                _boundary_case(f"boundary-{label}-d{d}-h{h}-w{w}", d, h, w, side)
            )
    rng = random.Random(SEED)
    random_count = CASE_COUNT - len(cases)
    for index in range(random_count):
        d, h, w = configs[index % len(configs)]
        cases.append(_random_case(rng, index, d, h, w))
    assert len(cases) == CASE_COUNT
    return tuple(cases)


def _gradient_run(device: torch.device) -> dict[str, Any]:
    def tensor(values: Any, requires_grad: bool = False) -> torch.Tensor:
        return torch.tensor(
            values, dtype=torch.float32, device=device, requires_grad=requires_grad
        )

    results: dict[str, Any] = {}

    p = tensor([[-0.25, 0.0]], True)
    lo = tensor([-0.5, -1.0], True)
    hi = tensor([1.0, 1.0], True)
    empty = torch.empty((0, 2), dtype=torch.float32, device=device)
    value = house_signed_clearance(p, lo, hi, empty, empty).sum()
    grads = torch.autograd.grad(value, (p, lo, hi))
    results["unique_frame_face"] = {
        "value": value.item(),
        "grads": [x.detach().cpu().tolist() for x in grads],
    }

    p = tensor([[0.4, 0.0]], True)
    flo, fhi = tensor([-1.0, -1.0], True), tensor([1.0, 1.0], True)
    wlo, whi = tensor([[-0.5, -0.5]], True), tensor([[0.5, 0.5]], True)
    value = house_signed_clearance(p, flo, fhi, wlo, whi).sum()
    grads = torch.autograd.grad(value, (p, flo, fhi, wlo, whi))
    results["unique_window_complement"] = {
        "value": value.item(),
        "grads": [x.detach().cpu().tolist() for x in grads],
    }

    p = tensor([[0.25]], True)
    flo, fhi = tensor([[-1.0], [0.0]], True), tensor([[0.0], [1.0]], True)
    no_windows = torch.empty((2, 0, 1), dtype=torch.float32, device=device)
    value = town_signed_clearance(p, flo, fhi, no_windows, no_windows).sum()
    grads = torch.autograd.grad(value, (p, flo, fhi))
    results["unique_town_house"] = {
        "value": value.item(),
        "grads": [x.detach().cpu().tolist() for x in grads],
    }

    p = tensor([[0.0]], True)
    lo, hi = tensor([-1.0], True), tensor([1.0], True)
    empty = torch.empty((0, 1), dtype=torch.float32, device=device)
    value = house_signed_clearance(p, lo, hi, empty, empty).sum()
    grads = torch.autograd.grad(value, (p, lo, hi))
    results["tied_box_amin"] = {
        "value": value.item(),
        "grads": [x.detach().cpu().tolist() for x in grads],
    }

    p = tensor([[0.0]], True)
    flo, fhi = tensor([0.0], True), tensor([1.0], True)
    wlo, whi = tensor([[0.0]], True), tensor([[0.5]], True)
    value = house_signed_clearance(p, flo, fhi, wlo, whi).sum()
    grads = torch.autograd.grad(value, (p, flo, wlo))
    results["tied_house_min_branch_cancellation"] = {
        "value": value.item(),
        "grads": [x.detach().cpu().tolist() for x in grads],
    }

    p = tensor([[0.0]], True)
    lo, hi = tensor([[-1.0], [0.0]], True), tensor([[0.0], [1.0]], True)
    no_windows = torch.empty((2, 0, 1), dtype=torch.float32, device=device)
    value = town_signed_clearance(p, lo, hi, no_windows, no_windows).sum()
    grads = torch.autograd.grad(value, (p, lo, hi))
    results["tied_town_amax_branch_cancellation"] = {
        "value": value.item(),
        "grads": [x.detach().cpu().tolist() for x in grads],
    }

    offset = tensor(0.0, True)
    p = torch.stack((offset - 0.25, offset + 0.25)).reshape(2, 1)
    no_windows = torch.empty((0, 1), dtype=torch.float32, device=device)
    values = house_signed_clearance(
        p, tensor([-1.0]), tensor([1.0]), no_windows, no_windows
    )
    individual = [
        torch.autograd.grad(values[i], offset, retain_graph=True)[0].item()
        for i in range(2)
    ]
    aggregate = torch.autograd.grad(values.sum(), offset)[0].item()
    results["aggregate_cancellation"] = {
        "values": values.detach().cpu().tolist(),
        "individual_grads": individual,
        "aggregate_grad": aggregate,
    }

    point_x = tensor(-0.25, True)
    lower_x = tensor(-0.5, True)
    points = torch.stack((point_x, tensor(0.0))).reshape(1, 2)
    lower = torch.stack((lower_x, tensor(-1.0)))
    upper = tensor([1.0, 1.0])
    empty = torch.empty((0, 2), dtype=torch.float32, device=device)
    score = house_signed_clearance(points, lower, upper, empty, empty).sum()
    point_grad, corner_grad = torch.autograd.grad(score, (point_x, lower_x))
    step = tensor(1e-3)

    def evaluate(point_value: torch.Tensor, lower_value: torch.Tensor) -> torch.Tensor:
        probe_points = torch.stack((point_value, tensor(0.0))).reshape(1, 2)
        probe_lower = torch.stack((lower_value, tensor(-1.0)))
        return house_signed_clearance(
            probe_points, probe_lower, upper, empty, empty
        ).sum()

    point_fd = (
        evaluate(point_x.detach() + step, lower_x.detach())
        - evaluate(point_x.detach() - step, lower_x.detach())
    ) / (2 * step)
    corner_fd = (
        evaluate(point_x.detach(), lower_x.detach() + step)
        - evaluate(point_x.detach(), lower_x.detach() - step)
    ) / (2 * step)
    results["finite_difference"] = {
        "score": score.item(),
        "step": step.item(),
        "analytic_point_gradient": point_grad.item(),
        "analytic_lower_corner_gradient": corner_grad.item(),
        "finite_difference_point_gradient": point_fd.item(),
        "finite_difference_lower_corner_gradient": corner_fd.item(),
    }

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return results


def _gradient_summary(
    cpu: dict[str, Any], cuda: dict[str, Any] | None
) -> dict[str, Any]:
    analytic = {
        "unique_frame_face": {
            "value": 0.25,
            "grads": [[[1.0, 0.0]], [-1.0, 0.0], [0.0, 0.0]],
        },
        "unique_window_complement": {
            "value": -0.1,
            "grads": [
                [[1.0, 0.0]],
                [0.0, 0.0],
                [0.0, 0.0],
                [[0.0, 0.0]],
                [[-1.0, 0.0]],
            ],
        },
        "unique_town_house": {
            "value": 0.25,
            "grads": [[[1.0]], [[0.0], [-1.0]], [[0.0], [0.0]]],
        },
        "tied_box_amin": {"value": 1.0, "grads": [[[0.0]], [-0.5], [0.5]]},
        "tied_house_min_branch_cancellation": {
            "value": 0.0,
            "grads": [[[0.0]], [-0.5], [[0.5]]],
        },
        "tied_town_amax_branch_cancellation": {
            "value": 0.0,
            "grads": [[[0.0]], [[0.0], [-0.5]], [[0.5], [0.0]]],
        },
    }
    max_unique_gradient_error = 0.0
    max_unique_score_error = 0.0
    max_tie_gradient_error = 0.0
    max_tie_score_error = 0.0
    device_max_difference = 0.0
    for name, expected in analytic.items():
        actual = cpu[name]
        gradient_error = max(
            _nested_abs_errors(actual["grads"], expected["grads"]), default=0.0
        )
        score_error = abs(actual["value"] - expected["value"])
        if name.startswith("unique_"):
            max_unique_gradient_error = max(max_unique_gradient_error, gradient_error)
            max_unique_score_error = max(max_unique_score_error, score_error)
        else:
            max_tie_gradient_error = max(max_tie_gradient_error, gradient_error)
            max_tie_score_error = max(max_tie_score_error, score_error)
        if cuda is not None:
            other = cuda[name]
            device_max_difference = max(
                device_max_difference,
                abs(actual["value"] - other["value"]),
                *_nested_abs_errors(actual["grads"], other["grads"]),
            )
    if cuda is not None:
        for key in ("aggregate_cancellation",):
            device_max_difference = max(
                device_max_difference,
                *_nested_abs_errors(cpu[key]["values"], cuda[key]["values"]),
                *_nested_abs_errors(
                    cpu[key]["individual_grads"], cuda[key]["individual_grads"]
                ),
                abs(cpu[key]["aggregate_grad"] - cuda[key]["aggregate_grad"]),
            )
        device_max_difference = max(
            device_max_difference,
            *(
                abs(cpu["finite_difference"][key] - cuda["finite_difference"][key])
                for key in (
                    "score",
                    "step",
                    "analytic_point_gradient",
                    "analytic_lower_corner_gradient",
                    "finite_difference_point_gradient",
                    "finite_difference_lower_corner_gradient",
                )
            ),
        )
    return {
        "cpu_observations": cpu,
        "cuda_observations": cuda,
        "max_unique_branch_gradient_abs_error_vs_analytic": max_unique_gradient_error,
        "max_unique_branch_score_abs_error_vs_analytic": max_unique_score_error,
        "max_tied_gradient_abs_error_vs_native_expected": max_tie_gradient_error,
        "max_tied_score_abs_error_vs_literal_expected": max_tie_score_error,
        "max_cpu_cuda_gradient_or_value_difference": device_max_difference,
        "finite_difference_probe": {
            "cpu": cpu["finite_difference"],
            "cuda": cuda["finite_difference"] if cuda is not None else None,
            "max_abs_discrepancy_from_analytic": max(
                abs(cpu["finite_difference"]["finite_difference_point_gradient"] - 1.0),
                abs(
                    cpu["finite_difference"]["finite_difference_lower_corner_gradient"]
                    + 1.0
                ),
                *(
                    [
                        abs(
                            cuda["finite_difference"][
                                "finite_difference_point_gradient"
                            ]
                            - 1.0
                        ),
                        abs(
                            cuda["finite_difference"][
                                "finite_difference_lower_corner_gradient"
                            ]
                            + 1.0
                        ),
                    ]
                    if cuda is not None
                    else []
                ),
            ),
        },
    }


def _nested_abs_errors(left: Any, right: Any) -> list[float]:
    if isinstance(left, list) and isinstance(right, list):
        errors: list[float] = []
        for a, b in zip(left, right, strict=True):
            errors.extend(_nested_abs_errors(a, b))
        return errors
    return [abs(float(left) - float(right))]


def _memory_probe(device: torch.device) -> dict[str, Any]:
    n, d, h, w = 128, 2, 2, 2
    points = (
        torch.linspace(-0.9, 0.9, n * d, dtype=torch.float32).reshape(n, d).to(device)
    )
    frame_lower = torch.tensor(
        [[-0.8, -0.7], [-0.6, -0.5]], dtype=torch.float32, device=device
    )
    frame_upper = torch.tensor(
        [[0.8, 0.7], [0.6, 0.5]], dtype=torch.float32, device=device
    )
    window_lower = torch.tensor(
        [[[-0.4, -0.3], [0.1, -0.2]], [[-0.2, -0.1], [0.2, 0.0]]],
        dtype=torch.float32,
        device=device,
    )
    window_upper = torch.tensor(
        [[[0.2, 0.3], [0.5, 0.2]], [[0.3, 0.25], [0.5, 0.3]]],
        dtype=torch.float32,
        device=device,
    )
    frame_faces_elements = n * h * 2 * d
    window_faces_elements = n * h * w * 2 * d
    record: dict[str, Any] = {
        "query_count": n,
        "d": d,
        "H": h,
        "W": w,
        "frame_face_temporary_elements": frame_faces_elements,
        "window_face_temporary_elements": window_faces_elements,
        "dominant_window_face_temporary_bytes_float32": window_faces_elements * 4,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        result = town_signed_clearance(
            points, frame_lower, frame_upper, window_lower, window_upper
        )
        torch.cuda.synchronize(device)
        record["cuda_incremental_peak_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(device) - baseline
        )
        record["output_shape"] = list(result.shape)
        del result
    else:
        result = town_signed_clearance(
            points, frame_lower, frame_upper, window_lower, window_upper
        )
        record["output_shape"] = list(result.shape)
        record["cuda_incremental_peak_allocated_bytes"] = None
    return record


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def _git_revision() -> dict[str, str | None]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
        return {"head": head, "working_tree": "dirty" if status else "clean"}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "working_tree": None}


def run() -> dict[str, Any]:
    exact_fixtures = _exact_fixtures()
    cases = _comparison_cases()
    cuda_available = torch.cuda.is_available()
    devices = [torch.device("cpu")]
    if cuda_available:
        devices.append(torch.device("cuda:0"))
    else:
        print("CUDA unavailable: CPU-only characterization; CUDA evidence is missing.")

    exact_records: list[dict[str, Any]] = []
    exact_measurements: dict[str, list[dict[str, Any]]] = {
        str(device): [] for device in devices
    }
    exact_oracle_mismatches: list[str] = []
    for fixture in exact_fixtures:
        house_ref, town_ref = _oracle(fixture)
        if fixture.expected is None or len(fixture.expected) != len(house_ref):
            raise AssertionError(
                f"literal expected result missing or malformed: {fixture.name}"
            )
        expected_decimal = [
            Decimal(value) if isinstance(value, str) else _dec(value)
            for value in fixture.expected
        ]
        wanted = house_ref if fixture.target == "house" else town_ref
        if wanted != expected_decimal:
            exact_oracle_mismatches.append(fixture.name)
        device_results: dict[str, Any] = {}
        device_measurements: dict[str, Any] = {}
        for device in devices:
            house_actual, town_actual = _score(fixture, device)
            values = house_actual if fixture.target == "house" else town_actual
            device_results[str(device)] = values
            device_measurements[str(device)] = [
                _measurement(actual, reference)
                for actual, reference in zip(values, wanted, strict=True)
            ]
            exact_measurements[str(device)].extend(device_measurements[str(device)])
        exact_records.append(
            {
                "name": fixture.name,
                "target": fixture.target,
                "d": fixture.dimensions,
                "H": fixture.houses,
                "W": fixture.windows,
                "expected_literal": fixture.expected,
                "oracle_decimal": [str(value) for value in wanted],
                "device_results": device_results,
                "device_oracle_measurements": device_measurements,
            }
        )
    if exact_oracle_mismatches:
        raise AssertionError(
            f"independent exact fixtures disagreed with hand literals: {exact_oracle_mismatches}"
        )

    metrics: dict[str, list[dict[str, Any]]] = {"house": [], "town": []}
    cpu_cuda: dict[str, list[dict[str, Any]]] = {"house": [], "town": []}
    raw_cases: list[dict[str, Any]] = []
    for case in cases:
        house_ref, town_ref = _oracle(case)
        cpu_house, cpu_town = _score(case, torch.device("cpu"))
        cuda_house: list[float] | None = None
        cuda_town: list[float] | None = None
        if cuda_available:
            cuda_house, cuda_town = _score(case, torch.device("cuda:0"))
        stored_cpu = _stored_tensors(case, torch.device("cpu"))
        case_row: dict[str, Any] = {
            "name": case.name,
            "source": case.source,
            "d": case.dimensions,
            "H": case.houses,
            "W": case.windows,
            "stored_float32_inputs": {
                "points": stored_cpu[0].tolist(),
                "frame_lower": stored_cpu[1].tolist(),
                "frame_upper": stored_cpu[2].tolist(),
                "window_lower": stored_cpu[3].tolist(),
                "window_upper": stored_cpu[4].tolist(),
            },
            "house": {},
            "town": {},
        }
        for name, refs, cpu_values, cuda_values in (
            ("house", house_ref, cpu_house, cuda_house),
            ("town", town_ref, cpu_town, cuda_town),
        ):
            for ref, cpu_value, cuda_value in zip(
                refs, cpu_values, cuda_values or [None] * len(refs), strict=True
            ):
                measurement = _measurement(cpu_value, ref)
                metrics[name].append(measurement)
                case_row[name].setdefault("cpu", []).append(measurement)
                if cuda_value is not None:
                    cuda_measurement = _measurement(cuda_value, ref)
                    case_row[name].setdefault("cuda", []).append(cuda_measurement)
                    cpu_cuda[name].append(
                        {
                            "abs_error": abs(cpu_value - cuda_value),
                            "ulp_error": _ulp_distance(cpu_value, cuda_value),
                            "sign_or_zero_crossing": _sign(cpu_value)
                            != _sign(cuda_value),
                            "exact_zero_mismatch": (cpu_value == 0.0)
                            != (cuda_value == 0.0),
                        }
                    )
        raw_cases.append(case_row)

    cuda_metrics: dict[str, Any] | None = None
    if cuda_available:
        cuda_metrics = {}
        for name in ("house", "town"):
            values = [
                entry for case in raw_cases for entry in case[name].get("cuda", [])
            ]
            cuda_metrics[name] = _summarize(values)
    cpu_cuda_summary = None
    if cuda_available:
        cpu_cuda_summary = {
            name: {
                "outputs": len(items),
                "max_absolute_difference": max(
                    (item["abs_error"] for item in items), default=0.0
                ),
                "max_ulp_difference": max(
                    (item["ulp_error"] for item in items), default=0
                ),
                "sign_or_zero_crossings": sum(
                    item["sign_or_zero_crossing"] for item in items
                ),
                "exact_zero_mismatches": sum(
                    item["exact_zero_mismatch"] for item in items
                ),
            }
            for name, items in cpu_cuda.items()
        }

    cpu_gradients = _gradient_run(torch.device("cpu"))
    cuda_gradients = _gradient_run(torch.device("cuda:0")) if cuda_available else None
    with localcontext() as context:
        context.prec = 80
        near_zero_count = sum(
            abs(_dec(float(value))) <= NEAR_ZERO
            for case in cases
            for value in _oracle(case)[1]
        )
    return {
        "classification": "exploratory only; not formal numerical acceptance",
        "seed": SEED,
        "comparison_case_count": len(cases),
        "case_config_counts": {
            f"d{d}-H{h}-W{w}": sum(
                (c.dimensions, c.houses, c.windows) == (d, h, w) for c in cases
            )
            for d in (1, 2)
            for h in (1, 2)
            for w in (0, 1, 2)
        },
        "near_zero_threshold_decimal": str(NEAR_ZERO),
        "near_zero_oracle_outputs_across_town_cases": near_zero_count,
        "exact_fixtures": {
            "count": len(exact_records),
            "oracle_metrics": {
                device: _summarize(values)
                for device, values in exact_measurements.items()
            },
            "records": exact_records,
        },
        "forward_metrics": {
            "cpu_house": _summarize(metrics["house"]),
            "cpu_town": _summarize(metrics["town"]),
            "cuda_house": cuda_metrics["house"] if cuda_metrics else None,
            "cuda_town": cuda_metrics["town"] if cuda_metrics else None,
            "cpu_cuda": cpu_cuda_summary,
        },
        "per_case_results": raw_cases,
        "gradient_characterization": _gradient_summary(cpu_gradients, cuda_gradients),
        "memory_observation": _memory_probe(
            torch.device("cuda:0") if cuda_available else torch.device("cpu")
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_processor": platform.processor(),
            "pytorch": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "cuda_driver": _nvidia_driver_version() if cuda_available else None,
            "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
            "cuda_capability": list(torch.cuda.get_device_capability(0))
            if cuda_available
            else None,
            "git": _git_revision(),
        },
        "limitations": [
            "Finite, hand-sized d/H/W cases only; no broad performance/capacity benchmark.",
            "The CPU/CUDA observations apply only to this software/device and deterministic fixtures.",
            "The exact sign/membership equivalence is mathematically implied by the approved equations; no disagreement is possible for valid finite inputs.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "exact_fixture_count": result["exact_fixtures"]["count"],
                "exact_fixture_oracle_metrics": result["exact_fixtures"][
                    "oracle_metrics"
                ],
                "comparison_case_count": result["comparison_case_count"],
                "forward_metrics": result["forward_metrics"],
                "gradient_metrics": {
                    "max_unique_branch_gradient_abs_error_vs_analytic": result[
                        "gradient_characterization"
                    ]["max_unique_branch_gradient_abs_error_vs_analytic"],
                    "max_unique_branch_score_abs_error_vs_analytic": result[
                        "gradient_characterization"
                    ]["max_unique_branch_score_abs_error_vs_analytic"],
                    "max_tied_gradient_abs_error_vs_native_expected": result[
                        "gradient_characterization"
                    ]["max_tied_gradient_abs_error_vs_native_expected"],
                    "max_cpu_cuda_gradient_or_value_difference": result[
                        "gradient_characterization"
                    ]["max_cpu_cuda_gradient_or_value_difference"],
                },
                "memory_observation": result["memory_observation"],
                "environment": result["environment"],
                "output": str(args.output) if args.output else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
