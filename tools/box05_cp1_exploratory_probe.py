#!/usr/bin/env python3
"""Reproducible CP1-only BOX-05 numerical characterization.

This is an exploratory diagnostic, not a formal acceptance suite. Its private
inverse arithmetic is limited to reporting inverse representability and error;
it is not imported by production code and is not a verified public inverse.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from town_model.models.bounded_trainable_box_geometry import (
    BoundedBoxGeometry,
    analytic_endpoint_jacobian,
    decode_bounded_box_raw,
    hosting_frame_minimum_full_width,
    ordinary_minimum_full_width,
    validate_bounded_box_geometry,
)

DIMENSIONS = (1, 2, 4, 8, 16, 32, 64)
CLASS_COUNTS = (0, 1, 2, 16, 60)
RAW_PROBES = (-40, -24, -20, -16, -12, -8, -4, -1, 0, 1, 4, 8, 12, 16, 20, 24, 40)
ORDINARY_EPSILONS = (1e-5, 1e-4, 1e-3, 1e-2)
HOSTING_CONFIGS = (
    (1e-4, 1e-4),
    (1e-5, 1e-4),
    (1e-4, 1e-5),
    (1e-3, 1e-3),
    (1e-3, 1e-2),
)
BASE_SEED = 20261112


def _ordered_float32_key(value: float) -> int:
    bits = int(np.asarray(np.float32(value)).view(np.uint32))
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000


def _float32_from_ordered_key(key: int) -> float:
    bits = (key & 0x7FFFFFFF) if key & 0x80000000 else (~key & 0xFFFFFFFF)
    return float(np.asarray(np.uint32(bits)).view(np.float32))


def _ulp_distance(left: float, right: float) -> int:
    if not math.isfinite(left) or not math.isfinite(right):
        return -1
    return abs(_ordered_float32_key(left) - _ordered_float32_key(right))


def _distribution(values: list[float] | list[int]) -> dict[str, float | int | str | None]:
    if not values:
        return {"count": 0, "nonfinite_count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    nonfinite_count = int(array.size - finite.size)
    if finite.size:
        minimum: float | str = float(np.min(finite))
        maximum: float | str = float(np.max(finite))
        p50 = float(np.quantile(finite, 0.50))
        p95 = float(np.quantile(finite, 0.95))
        p99 = float(np.quantile(finite, 0.99))
    else:
        minimum = maximum = None
        p50 = p95 = p99 = None
    if np.isneginf(array).any():
        minimum = "-Infinity"
    if np.isposinf(array).any():
        maximum = "Infinity"
    if np.isnan(array).any():
        nonfinite_count = int(np.count_nonzero(~np.isfinite(array)))
    return {
        "count": int(array.size),
        "nonfinite_count": nonfinite_count,
        "min": minimum,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "max": maximum,
    }


def _json_number(value: float) -> float | str:
    if math.isfinite(value):
        return float(value)
    if math.isnan(value):
        return "NaN"
    return "-Infinity" if value < 0.0 else "Infinity"


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    return _distribution([float(value) for value in values.reshape(-1)])


def _reference(raw_width: np.ndarray, raw_center: np.ndarray, width: float) -> dict[str, np.ndarray]:
    a = raw_width.astype(np.float64)
    b = raw_center.astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-a))
    half = (width + (2.0 - width) * p) / 2.0
    center = (1.0 - half) * np.tanh(b)
    return {"centers": center, "half_widths": half, "lower": center - half, "upper": center + half}


def _validation(geometry: BoundedBoxGeometry) -> dict[str, Any]:
    try:
        validate_bounded_box_geometry(geometry)
    except (TypeError, ValueError) as error:
        valid = False
        reason = str(error)
    else:
        valid = True
        reason = None
    row_validity: list[dict[str, Any]] = []
    for row in range(geometry.centers.shape[0]):
        try:
            validate_bounded_box_geometry(
                BoundedBoxGeometry(
                    centers=geometry.centers[row : row + 1],
                    half_widths=geometry.half_widths[row : row + 1],
                    lower=geometry.lower[row : row + 1],
                    upper=geometry.upper[row : row + 1],
                    minimum_full_width=geometry.minimum_full_width,
                )
            )
        except (TypeError, ValueError) as error:
            row_validity.append({"row": row, "valid": False, "reason": str(error)})
        else:
            row_validity.append({"row": row, "valid": True, "reason": None})
    return {"valid": valid, "reason": reason, "rows": row_validity}


def _jacobian_observation(a: torch.Tensor, b: torch.Tensor, width: float) -> dict[str, Any]:
    jacobian = analytic_endpoint_jacobian(a, b, minimum_full_width=width)
    aa = a.detach().clone().requires_grad_(True)
    bb = b.detach().clone().requires_grad_(True)
    geometry = decode_bounded_box_raw(aa, bb, minimum_full_width=width)
    lower_grads = torch.autograd.grad(
        geometry.lower.sum(), (aa, bb), retain_graph=True
    )
    upper_grads = torch.autograd.grad(geometry.upper.sum(), (aa, bb))
    autograd = torch.stack(
        (torch.stack(lower_grads, dim=-1), torch.stack(upper_grads, dim=-1)), dim=-2
    )
    jacobian_cpu = jacobian.detach().cpu().numpy().astype(np.float64)
    auto_cpu = autograd.detach().cpu().numpy().astype(np.float64)
    singular = np.linalg.svd(jacobian_cpu.reshape(-1, 2, 2), compute_uv=False)
    smallest = singular[:, -1]
    largest = singular[:, 0]
    condition = np.divide(
        largest,
        smallest,
        out=np.full_like(largest, np.inf),
        where=smallest != 0.0,
    )
    error = np.abs(jacobian_cpu - auto_cpu)
    flat_jacobian = jacobian.detach().cpu().numpy().reshape(-1, 2, 2)
    determinant = np.linalg.det(flat_jacobian.astype(np.float64))
    return {
        "analytic_vs_autograd_max_abs": float(np.max(error)) if error.size else 0.0,
        "analytic_vs_autograd_error_distribution": _summary(error),
        "analytic_jacobian_abs_max": float(np.max(np.abs(jacobian_cpu))) if error.size else 0.0,
        "jacobian_zero_entry_count": int(np.count_nonzero(jacobian_cpu == 0.0)),
        "jacobian_entry_count": int(jacobian_cpu.size),
        "sigma_min": _distribution(smallest.tolist()),
        "sigma_max": _distribution(largest.tolist()),
        "condition_2": _distribution(condition.tolist()),
        "log10_sigma_min": _distribution(
            [float(np.log10(value)) if value > 0.0 else -math.inf for value in smallest]
        ),
        "determinant_vs_formula_max_abs": float(
            np.max(
                np.abs(
                    determinant
                    - (
                        -2.0
                        * ((flat_jacobian[:, 1, 0] - flat_jacobian[:, 0, 0]) / 2.0)
                        * flat_jacobian[:, 0, 1]
                    )
                )
            )
        ) if determinant.size else 0.0,
        "sigma_min_samples": [_json_number(float(value)) for value in smallest],
        "condition_2_samples": [_json_number(float(value)) for value in condition],
        "autograd_abs_error_samples": [_json_number(float(value)) for value in error.reshape(-1)],
        "jacobian_abs_entry_samples": [_json_number(float(value)) for value in np.abs(jacobian_cpu).reshape(-1)],
        "analytic_jacobian_samples": [_json_number(float(value)) for value in jacobian_cpu.reshape(-1)],
        "autograd_jacobian_samples": [_json_number(float(value)) for value in auto_cpu.reshape(-1)],
    }


def _measure_device(
    raw_width_values: list[list[float]],
    raw_center_values: list[list[float]],
    width: float,
    device_name: str,
    dimension: int,
) -> dict[str, Any]:
    device = torch.device(device_name)
    a = torch.tensor(raw_width_values, dtype=torch.float32, device=device).reshape(-1, dimension)
    b = torch.tensor(raw_center_values, dtype=torch.float32, device=device).reshape(-1, dimension)
    geometry = decode_bounded_box_raw(a, b, minimum_full_width=width)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    validation = _validation(geometry)
    actual = {
        name: getattr(geometry, name).detach().cpu().numpy()
        for name in ("centers", "half_widths", "lower", "upper")
    }
    raw_a = np.asarray(raw_width_values, dtype=np.float32).reshape(-1, dimension)
    raw_b = np.asarray(raw_center_values, dtype=np.float32).reshape(-1, dimension)
    reference = _reference(raw_a, raw_b, width)
    errors: dict[str, Any] = {}
    for name in actual:
        ref64 = reference[name]
        ref32 = ref64.astype(np.float32)
        got = actual[name]
        abs_error = np.abs(got.astype(np.float64) - ref64)
        ulps = [
            _ulp_distance(float(x), float(y))
            for x, y in zip(got.reshape(-1), ref32.reshape(-1), strict=True)
        ]
        errors[name] = {
            "absolute_error": _summary(abs_error),
            "ulp_vs_float32_rounded_float64_reference": _distribution(ulps),
            "exact_zero_count": int(np.count_nonzero(got == 0.0)),
            "negative_zero_count": int(np.count_nonzero((got == 0.0) & np.signbit(got))),
            "reference_zero_count": int(np.count_nonzero(ref32 == 0.0)),
            "sign_mismatch_count": int(np.count_nonzero(np.signbit(got) != np.signbit(ref32))),
        }

    half = actual["half_widths"].astype(np.float64)
    lower = actual["lower"].astype(np.float64)
    upper = actual["upper"].astype(np.float64)
    widths = (actual["upper"] - actual["lower"]).astype(np.float64)
    domain_slack = np.minimum(lower + 1.0, 1.0 - upper)
    width_slack = widths - width
    half_slack = 2.0 * half - width
    jacobian = _jacobian_observation(a, b, width)
    result = {
        "device": device_name,
        "physical_validation": validation,
        "actual": {name: values.tolist() for name, values in actual.items()},
        "reference_errors": errors,
        "actual_endpoint_width": _summary(widths),
        "endpoint_width_reserve_slack": _summary(width_slack),
        "half_width_reserve_slack": _summary(half_slack),
        "domain_boundary_slack": _summary(domain_slack),
        "negative_endpoint_width_reserve_count": int(np.count_nonzero(width_slack < 0.0)),
        "negative_half_width_reserve_count": int(np.count_nonzero(half_slack < 0.0)),
        "negative_domain_slack_count": int(np.count_nonzero(domain_slack < 0.0)),
        "jacobian_and_autograd": jacobian,
    }
    return result


def _cross_device(cpu: dict[str, Any], cuda: dict[str, Any] | None) -> dict[str, Any] | None:
    if cuda is None:
        return None
    report: dict[str, Any] = {}
    for name in ("centers", "half_widths", "lower", "upper"):
        left = np.asarray(cpu["actual"][name], dtype=np.float32)
        right = np.asarray(cuda["actual"][name], dtype=np.float32)
        if left.size:
            ulps = [
                _ulp_distance(float(a), float(b))
                for a, b in zip(left.reshape(-1), right.reshape(-1), strict=True)
            ]
            differences = np.abs(left.astype(np.float64) - right.astype(np.float64))
            report[name] = {
                "absolute_difference": _summary(differences),
                "ulp_distance": _distribution(ulps),
                "zero_status_mismatch_count": int(np.count_nonzero((left == 0.0) != (right == 0.0))),
                "signbit_mismatch_count": int(np.count_nonzero(np.signbit(left) != np.signbit(right))),
            }
        else:
            report[name] = {"absolute_difference": _summary(np.array([])), "ulp_distance": _distribution([]), "zero_status_mismatch_count": 0, "signbit_mismatch_count": 0}
    return report


def _cross_device_jacobian(cpu: dict[str, Any], cuda: dict[str, Any] | None) -> dict[str, Any] | None:
    if cuda is None:
        return None
    left = cpu["jacobian_and_autograd"]
    right = cuda["jacobian_and_autograd"]
    result: dict[str, Any] = {}
    for name in ("analytic_jacobian_samples", "autograd_jacobian_samples"):
        lvals = np.asarray(left[name], dtype=np.float32)
        rvals = np.asarray(right[name], dtype=np.float32)
        if lvals.size:
            result[name] = {
                "absolute_difference": _summary(np.abs(lvals.astype(np.float64) - rvals.astype(np.float64))),
                "ulp_distance": _distribution([
                    _ulp_distance(float(a), float(b))
                    for a, b in zip(lvals.reshape(-1), rvals.reshape(-1), strict=True)
                ]),
                "zero_status_mismatch_count": int(np.count_nonzero((lvals == 0.0) != (rvals == 0.0))),
                "signbit_mismatch_count": int(np.count_nonzero(np.signbit(lvals) != np.signbit(rvals))),
            }
        else:
            result[name] = {"absolute_difference": _summary(np.array([])), "ulp_distance": _distribution([]), "zero_status_mismatch_count": 0, "signbit_mismatch_count": 0}
    result["analytic_vs_autograd_max_abs_cpu"] = left["analytic_vs_autograd_max_abs"]
    result["analytic_vs_autograd_max_abs_cuda"] = right["analytic_vs_autograd_max_abs"]
    result["sigma_min_cpu"] = left["sigma_min"]
    result["sigma_min_cuda"] = right["sigma_min"]
    return result


def _case(
    case_id: str,
    coverage: str,
    raw_width: list[list[float]],
    raw_center: list[list[float]],
    *,
    dimension: int,
    count: int,
    mode: str,
    epsilon: float,
    delta: float | None,
    minimum_width: float,
    devices: tuple[str, ...],
) -> dict[str, Any]:
    result_by_device = {
        device: _measure_device(raw_width, raw_center, minimum_width, device, dimension)
        for device in devices
    }
    cpu = result_by_device["cpu"]
    cuda = result_by_device.get("cuda")
    return {
        "case_id": case_id,
        "coverage": coverage,
        "dimension": dimension,
        "class_count": count,
        "mode": mode,
        "epsilon": epsilon,
        "delta": delta,
        "minimum_full_width": minimum_width,
        "raw_width": raw_width,
        "raw_center": raw_center,
        "device_results": result_by_device,
        "cpu_cuda_comparison": _cross_device(cpu, cuda),
        "cpu_cuda_jacobian_comparison": _cross_device_jacobian(cpu, cuda),
    }


def _pattern_matrix(count: int, dimension: int, sequence: tuple[float, ...], offset: int = 0) -> list[list[float]]:
    return [
        [float(sequence[(offset + row * dimension + col) % len(sequence)]) for col in range(dimension)]
        for row in range(count)
    ]


def _core_cases(devices: tuple[str, ...]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    canonical_raw = tuple(float(value) for value in RAW_PROBES)
    # Dimension sweep: every listed raw value reaches every coordinate of both
    # raw tensors across this deterministic 17-case sequence. C is fixed at 2.
    for dimension in DIMENSIONS:
        for mode, epsilon, delta, width in (
            ("ordinary", 1e-4, None, ordinary_minimum_full_width(1e-4)),
            ("hosting_frame", 1e-4, 1e-4, hosting_frame_minimum_full_width(1e-4, 1e-4)),
        ):
            for probe_index, probe in enumerate(canonical_raw):
                raw_width = [
                    [probe for _ in range(dimension)],
                    [canonical_raw[(probe_index + col + 7) % len(canonical_raw)] for col in range(dimension)],
                ]
                raw_center = [
                    [canonical_raw[(probe_index + col + 1) % len(canonical_raw)] for col in range(dimension)],
                    [canonical_raw[(probe_index + 5) % len(canonical_raw)] for _ in range(dimension)],
                ]
                cases.append(
                    _case(
                        f"dimension-{dimension}-{mode}-raw-{probe:g}",
                        "fixed_dimension_raw_sweep",
                        raw_width,
                        raw_center,
                        dimension=dimension,
                        count=2,
                        mode=mode,
                        epsilon=epsilon,
                        delta=delta,
                        minimum_width=width,
                        devices=devices,
                    )
                )

    # Representative C grid at d=2, not crossed with every dimension/config.
    for count in CLASS_COUNTS:
        for mode, epsilon, delta, width in (
            ("ordinary", 1e-4, None, ordinary_minimum_full_width(1e-4)),
            ("hosting_frame", 1e-4, 1e-4, hosting_frame_minimum_full_width(1e-4, 1e-4)),
        ):
            raw_width = _pattern_matrix(count, 2, (-8.0, -1.0, 0.0, 2.0, 8.0))
            raw_center = _pattern_matrix(count, 2, (-12.0, -2.0, 0.0, 1.0, 12.0), 2)
            cases.append(
                _case(
                    f"count-{count}-d2-{mode}",
                    "fixed_representative_class_count_sweep",
                    raw_width,
                    raw_center,
                    dimension=2,
                    count=count,
                    mode=mode,
                    epsilon=epsilon,
                    delta=delta,
                    minimum_width=width,
                    devices=devices,
                )
            )

    # Remaining approved epsilon/delta points are assigned to fixed, distinct
    # representative (d,C) pairs. This is deliberately sparse, not Cartesian.
    config_cases = (
        ("ordinary", 1e-5, None, 1, 1),
        ("ordinary", 1e-3, None, 4, 16),
        ("ordinary", 1e-2, None, 16, 60),
        ("hosting_frame", 1e-5, 1e-4, 2, 2),
        ("hosting_frame", 1e-4, 1e-5, 4, 2),
        ("hosting_frame", 1e-3, 1e-3, 8, 16),
        ("hosting_frame", 1e-3, 1e-2, 32, 60),
    )
    for index, (mode, epsilon, delta, dimension, count) in enumerate(config_cases):
        width = (
            ordinary_minimum_full_width(epsilon)
            if mode == "ordinary"
            else hosting_frame_minimum_full_width(epsilon, float(delta))
        )
        raw_width = _pattern_matrix(count, dimension, (-16.0, -4.0, -1.0, 0.0, 1.0, 4.0, 16.0), index)
        raw_center = _pattern_matrix(count, dimension, (-12.0, -2.0, -0.25, 0.5, 2.0, 8.0), index + 3)
        cases.append(
            _case(
                f"config-{mode}-eps{epsilon:g}-delta{delta if delta is not None else 'none'}-d{dimension}-c{count}",
                "fixed_sparse_width_configuration_sweep",
                raw_width,
                raw_center,
                dimension=dimension,
                count=count,
                mode=mode,
                epsilon=epsilon,
                delta=delta,
                minimum_width=width,
                devices=devices,
            )
        )

    # Capacity boundary and invalid policy checks are intentionally separate.
    near_two = float(np.nextafter(np.float64(2.0), np.float64(0.0)))
    cases.append(
        _case(
            "near-capacity-ordinary-w-nextafter-2",
            "near_capacity_boundary",
            [[-40.0, 0.0], [16.0, 40.0]],
            [[-16.0, 0.0], [8.0, 40.0]],
            dimension=2,
            count=2,
            mode="ordinary_near_capacity",
            epsilon=near_two,
            delta=None,
            minimum_width=ordinary_minimum_full_width(near_two),
            devices=devices,
        )
    )
    return cases


def _invalid_configuration_probes() -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for epsilon in (0.0, -1e-4, 2.0, 2.0001, float("nan")):
        try:
            value = ordinary_minimum_full_width(epsilon)
        except (TypeError, ValueError) as error:
            probes.append({"mode": "ordinary", "epsilon": str(epsilon), "accepted": False, "reason": str(error)})
        else:
            probes.append({"mode": "ordinary", "epsilon": str(epsilon), "accepted": True, "minimum_full_width": value})
    for epsilon, delta in ((0.0, 1e-4), (1e-4, 0.0), (1.0, 0.5), (1.5, 0.25), (1e-4, 1e-4)):
        try:
            value = hosting_frame_minimum_full_width(epsilon, delta)
        except (TypeError, ValueError) as error:
            probes.append({"mode": "hosting_frame", "epsilon": epsilon, "delta": delta, "accepted": False, "reason": str(error)})
        else:
            probes.append({"mode": "hosting_frame", "epsilon": epsilon, "delta": delta, "accepted": True, "minimum_full_width": value})
    return probes


def _first_true_float32(
    low: float, high: float, predicate: Any, device_name: str
) -> tuple[float, float]:
    low_key = _ordered_float32_key(low)
    high_key = _ordered_float32_key(high)
    if low_key >= high_key:
        raise ValueError("binary-search endpoints are not monotonically ordered")
    device = torch.device(device_name)

    def test(value: float) -> bool:
        scalar = torch.tensor(value, dtype=torch.float32, device=device)
        result = predicate(scalar)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return bool(result.item())

    if test(_float32_from_ordered_key(low_key)) or not test(_float32_from_ordered_key(high_key)):
        raise ValueError("binary-search endpoints do not bracket the transition")
    while high_key - low_key > 1:
        middle_key = (low_key + high_key) // 2
        middle = _float32_from_ordered_key(middle_key)
        if test(middle):
            high_key = middle_key
        else:
            low_key = middle_key
    return _float32_from_ordered_key(low_key), _float32_from_ordered_key(high_key)


def _scalar_activation(value: float, activation: str, device: str) -> float:
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    result = torch.sigmoid(tensor) if activation == "sigmoid" else torch.tanh(tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    return float(result.item())


def _boundary_thresholds(devices: tuple[str, ...]) -> dict[str, Any]:
    transitions: dict[str, Any] = {}
    predicates = {
        "sigmoid_first_nonzero": (-150.0, 0.0, lambda x: torch.sigmoid(x) > 0.0),
        "sigmoid_first_one": (0.0, 40.0, lambda x: torch.sigmoid(x) == 1.0),
        "tanh_first_above_minus_one": (-40.0, 0.0, lambda x: torch.tanh(x) > -1.0),
        "tanh_first_positive_one": (0.0, 40.0, lambda x: torch.tanh(x) == 1.0),
    }
    for device in devices:
        per_device: dict[str, Any] = {}
        for name, (low, high, predicate) in predicates.items():
            before, transition = _first_true_float32(low, high, predicate, device)
            direction = math.inf
            after_tensor = torch.nextafter(
                torch.tensor(transition, dtype=torch.float32, device=device),
                torch.tensor(direction, dtype=torch.float32, device=device),
            )
            if device == "cuda":
                torch.cuda.synchronize()
            after = float(after_tensor.item())
            activation = "sigmoid" if name.startswith("sigmoid") else "tanh"
            activation_values = {
                "predecessor": _scalar_activation(before, activation, device),
                "transition": _scalar_activation(transition, activation, device),
                "successor": _scalar_activation(after, activation, device),
            }
            def derivative(value: float) -> float:
                tensor = torch.tensor(value, dtype=torch.float32, device=device)
                output = torch.sigmoid(tensor) if activation == "sigmoid" else torch.tanh(tensor)
                derivative_value = output * (1.0 - output) if activation == "sigmoid" else 1.0 - output.square()
                if device == "cuda":
                    torch.cuda.synchronize()
                return float(derivative_value.item())
            per_device[name] = {
                "predecessor": before,
                "transition": transition,
                "successor": after,
                "activation_values": activation_values,
                "activation_derivatives": {
                    "predecessor": derivative(before),
                    "transition": derivative(transition),
                    "successor": derivative(after),
                },
            }
        transitions[device] = per_device
    return transitions


def _minimum_width_transition(width: float, device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)

    def actual_stored_width_meets_floor(raw: torch.Tensor) -> torch.Tensor:
        a = raw.reshape(1, 1)
        b = torch.zeros_like(a)
        geometry = decode_bounded_box_raw(a, b, minimum_full_width=width)
        half_ok = geometry.half_widths.to(torch.float64) >= width / 2.0
        endpoint_width = geometry.upper.to(torch.float64) - geometry.lower.to(torch.float64)
        return (half_ok & (endpoint_width >= width)).all()

    def valid_at(value: float) -> bool:
        result = actual_stored_width_meets_floor(torch.tensor(value, dtype=torch.float32, device=device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return bool(result.item())

    low_valid = valid_at(-40.0)
    high_valid = valid_at(0.0)
    if low_valid:
        return {"state": "already_valid_at_lower_bracket", "bracket": [-40.0, 0.0], "predecessor_invalid_candidate": None, "first_valid_candidate": None}
    if not high_valid:
        return {"state": "not_valid_at_upper_bracket", "bracket": [-40.0, 0.0], "predecessor_invalid_candidate": None, "first_valid_candidate": None}
    predecessor, transition = _first_true_float32(-40.0, 0.0, actual_stored_width_meets_floor, device_name)
    return {"state": "transition_found", "bracket": [-40.0, 0.0], "predecessor_invalid_candidate": predecessor, "first_valid_candidate": transition}


def _minimum_width_boundary_probes(devices: tuple[str, ...]) -> dict[str, Any]:
    configurations: list[tuple[str, float]] = [
        (f"ordinary-epsilon-{epsilon:g}", ordinary_minimum_full_width(epsilon))
        for epsilon in ORDINARY_EPSILONS
    ]
    configurations.extend(
        (
            f"hosting-epsilon-{epsilon:g}-delta-{delta:g}",
            hosting_frame_minimum_full_width(epsilon, delta),
        )
        for epsilon, delta in HOSTING_CONFIGS
    )
    thresholds: dict[str, Any] = {}
    probes: list[dict[str, Any]] = []
    for device in devices:
        thresholds[device] = {}
        for config_name, width in configurations:
            transition_result = _minimum_width_transition(width, device)
            predecessor = transition_result["predecessor_invalid_candidate"]
            transition = transition_result["first_valid_candidate"]
            thresholds[device][config_name] = {
                "minimum_full_width": width,
                **transition_result,
            }
            if transition is None:
                raw_candidates = (("lower_bracket", -40.0), ("upper_bracket", 0.0))
            else:
                transition_tensor = torch.tensor(transition, dtype=torch.float32, device=device)
                successor = float(
                    torch.nextafter(
                        transition_tensor,
                        torch.tensor(math.inf, dtype=torch.float32, device=device),
                    ).item()
                )
                thresholds[device][config_name]["successor"] = successor
                raw_candidates = (
                    ("predecessor", predecessor),
                    ("transition", transition),
                    ("successor", successor),
                )
            for adjacent, raw_a in raw_candidates:
                for raw_b in (0.0, -4.0, 4.0, -8.0, 8.0):
                    measurement = _measure_device([[raw_a]], [[raw_b]], width, device, 1)
                    probes.append(
                        {
                            "device": device,
                            "config": config_name,
                            "minimum_full_width": width,
                            "adjacent": adjacent,
                            "raw_width": raw_a,
                            "raw_center": raw_b,
                            "measurement": measurement,
                        }
                    )
    return {"transitions": thresholds, "probes": probes}


def _boundary_geometry_probes(
    devices: tuple[str, ...], thresholds: dict[str, Any]
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    configurations = (
        ("ordinary_default", ordinary_minimum_full_width(1e-4)),
        ("hosting_default", hosting_frame_minimum_full_width(1e-4, 1e-4)),
    )
    for device in devices:
        for config_name, width in configurations:
            for raw_name in ("sigmoid_first_nonzero", "sigmoid_first_one"):
                boundary = thresholds[device][raw_name]
                for label, raw_a in (("predecessor", boundary["predecessor"]), ("transition", boundary["transition"]), ("successor", boundary["successor"])):
                    for raw_b in (0.0, 4.0, -4.0):
                        measured = _measure_device([[raw_a]], [[raw_b]], width, device, 1)
                        probes.append({"device": device, "config": config_name, "transition": raw_name, "adjacent": label, "raw_width": raw_a, "raw_center": raw_b, "measurement": measured})
            for raw_name in ("tanh_first_above_minus_one", "tanh_first_positive_one"):
                boundary = thresholds[device][raw_name]
                for label, raw_b in (("predecessor", boundary["predecessor"]), ("transition", boundary["transition"]), ("successor", boundary["successor"])):
                    for raw_a in (-40.0, 0.0, 8.0):
                        measured = _measure_device([[raw_a]], [[raw_b]], width, device, 1)
                        probes.append({"device": device, "config": config_name, "transition": raw_name, "adjacent": label, "raw_width": raw_a, "raw_center": raw_b, "measurement": measured})
    return probes


def _seed_for(label: str) -> int:
    digest = hashlib.sha256(f"box05-cp1-v1:{BASE_SEED}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _occupancy_probes(devices: tuple[str, ...]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        for mode, epsilon, delta, width in (
            ("ordinary", 1e-4, None, ordinary_minimum_full_width(1e-4)),
            ("hosting_frame", 1e-4, 1e-4, hosting_frame_minimum_full_width(1e-4, 1e-4)),
        ):
            label = f"occupancy-d{dimension}-{mode}"
            seed = _seed_for(label)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            raw_width = torch.randn((60, dimension), generator=generator, dtype=torch.float32)
            raw_center = torch.randn((60, dimension), generator=generator, dtype=torch.float32)
            case = _case(
                label,
                "seeded_raw_occupancy_anisotropy_probe_not_initializer",
                raw_width.tolist(),
                raw_center.tolist(),
                dimension=dimension,
                count=60,
                mode=mode,
                epsilon=epsilon,
                delta=delta,
                minimum_width=width,
                devices=devices,
            )
            for device in devices:
                actual = case["device_results"][device]["actual"]
                half = np.asarray(actual["half_widths"], dtype=np.float64)
                log_q = np.log(half).sum(axis=1) if half.shape[0] else np.array([])
                anisotropy = np.max(half, axis=1) / np.min(half, axis=1) if half.shape[0] else np.array([])
                case["device_results"][device]["occupancy_anisotropy"] = {
                    "definition": "log_q=sum_j(log(h_j)); q=exp(log_q); anisotropy=max_j(h_j)/min_j(h_j)",
                    "log_q": _summary(log_q),
                    "q_float64_if_representable": _summary(np.exp(log_q)),
                    "anisotropy_ratio": _summary(anisotropy),
                    "seed": seed,
                    "seed_derivation": "first 64 little-endian bits of SHA256('box05-cp1-v1:<base-seed>:<case-id>'), modulo 2^63-1; private CPU torch.Generator",
                }
            cases.append(case)
    return cases


def _physical_target_valid(lower: np.float32, upper: np.float32, width: float) -> tuple[bool, str | None]:
    l = torch.tensor([[lower]], dtype=torch.float32)
    r = torch.tensor([[upper]], dtype=torch.float32)
    if not bool(torch.isfinite(l).all() and torch.isfinite(r).all()):
        return False, "nonfinite target endpoints"
    if not bool((l >= -1.0).all() and (r <= 1.0).all() and (l <= r).all()):
        return False, "target corners violate ordered bounded-domain invariant"
    exact_stored_width = r.to(torch.float64) - l.to(torch.float64)
    if not bool((exact_stored_width >= width).all()):
        return False, "actual float32 target endpoint width is below configured floor"
    return True, None


def _diagnostic_inverse_target(
    lower: np.float32, upper: np.float32, width: float, devices: tuple[str, ...], label: str
) -> dict[str, Any]:
    target_ok, invalid_reason = _physical_target_valid(lower, upper, width)
    base: dict[str, Any] = {
        "target_id": label,
        "lower": float(lower),
        "upper": float(upper),
        "minimum_full_width": width,
        "target_physical_valid": target_ok,
        "target_failure": invalid_reason,
        "diagnostic_status": None,
        "finite_real": None,
        "raw_width_f32": None,
        "raw_center_f32": None,
        "device_reconstructions": {},
    }
    if not target_ok:
        base["diagnostic_status"] = "invalid_target_geometry"
        return base

    # Diagnostic-only evaluation of the issue's inverse equations in float64.
    l64, r64 = float(lower), float(upper)
    half = (r64 - l64) / 2.0
    center = (r64 + l64) / 2.0
    t = (2.0 * half - width) / (2.0 - width)
    s = center / (1.0 - half) if half != 1.0 else math.copysign(math.inf, center)
    finite_real = 0.0 < t < 1.0 and abs(s) < 1.0 and math.isfinite(t) and math.isfinite(s)
    base["inverse_intermediates_float64"] = {
        "half_width": _json_number(half),
        "center": _json_number(center),
        "t": _json_number(t),
        "s": _json_number(s),
    }
    base["finite_real"] = finite_real
    if not finite_real:
        base["diagnostic_status"] = "finite_real_unattainable"
        return base

    a64 = math.log(t) - math.log1p(-t)
    b64 = math.atanh(s)
    a32, b32 = np.float32(a64), np.float32(b64)
    base["raw_width_float64"] = a64
    base["raw_center_float64"] = b64
    base["raw_width_f32"] = float(a32)
    base["raw_center_f32"] = float(b32)
    if not np.isfinite(a32) or not np.isfinite(b32):
        base["diagnostic_status"] = "float32_inverse_nonfinite"
        return base

    any_invariant_failure = False
    for device in devices:
        geometry = decode_bounded_box_raw(
            torch.tensor([[a32]], dtype=torch.float32, device=device),
            torch.tensor([[b32]], dtype=torch.float32, device=device),
            minimum_full_width=width,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        try:
            validate_bounded_box_geometry(geometry)
        except (TypeError, ValueError) as error:
            valid = False
            reason = str(error)
            any_invariant_failure = True
        else:
            valid = True
            reason = None
        out_l = float(geometry.lower.detach().cpu().item())
        out_r = float(geometry.upper.detach().cpu().item())
        endpoint_errors = (abs(out_l - l64), abs(out_r - r64))
        endpoint_ulps = (_ulp_distance(out_l, float(lower)), _ulp_distance(out_r, float(upper)))
        reconstructed_center = float(geometry.centers.detach().cpu().item())
        reconstructed_half = float(geometry.half_widths.detach().cpu().item())
        base["device_reconstructions"][device] = {
            "physical_valid": valid,
            "physical_failure": reason,
            "lower": out_l,
            "upper": out_r,
            "center": reconstructed_center,
            "half_width": reconstructed_half,
            "endpoint_abs_errors": endpoint_errors,
            "endpoint_max_abs_error": max(endpoint_errors),
            "endpoint_ulp_distances_including_near_zero": endpoint_ulps,
            "center_abs_error_diagnostic": abs(reconstructed_center - center),
            "half_width_abs_error_diagnostic": abs(reconstructed_half - half),
            "reserve_slack": (out_r - out_l) - width,
            "approved_budget_status": "not_assessed_no_author_approved_budget",
        }
    base["diagnostic_status"] = (
        "reconstructed_hard_invariant_failure" if any_invariant_failure else "finite_f32_candidate_measured"
    )
    return base


def _inverse_probes(devices: tuple[str, ...]) -> dict[str, Any]:
    rng = np.random.default_rng(BASE_SEED)
    configs: list[tuple[str, float, float | None]] = []
    configs.extend(("ordinary", epsilon, None) for epsilon in ORDINARY_EPSILONS)
    configs.extend(("hosting_frame", epsilon, delta) for epsilon, delta in HOSTING_CONFIGS)
    targets: list[dict[str, Any]] = []
    for config_index, (mode, epsilon, delta) in enumerate(configs):
        width = (
            ordinary_minimum_full_width(epsilon)
            if mode == "ordinary"
            else hosting_frame_minimum_full_width(epsilon, float(delta))
        )
        explicit: list[tuple[str, float, float]] = [
            ("minimum_width_boundary", -width / 2.0, width / 2.0),
            ("full_domain_capacity", -1.0, 1.0),
            ("upper_domain_touch", 0.5, 1.0),
            ("lower_domain_touch", -1.0, -0.5),
            ("zero_endpoint", 0.0, 0.5),
            ("interior_dyadic", -0.25, 0.25),
        ]
        for label, lower, upper in explicit:
            targets.append(
                _diagnostic_inverse_target(
                    np.float32(lower), np.float32(upper), width, devices,
                    f"{mode}-e{epsilon:g}-d{delta if delta is not None else 'none'}-{label}",
                )
            )
        for sample_index in range(128):
            half = float(rng.uniform(max(width / 2.0 + 0.025, 0.03), 0.95))
            center_fraction = float(rng.uniform(-0.98, 0.98))
            center = (1.0 - half) * center_fraction
            lower = np.float32(center - half)
            upper = np.float32(center + half)
            targets.append(
                _diagnostic_inverse_target(
                    lower,
                    upper,
                    width,
                    devices,
                    f"{mode}-e{epsilon:g}-d{delta if delta is not None else 'none'}-random-{sample_index:03d}",
                )
            )
    by_configuration: dict[str, Any] = {}
    for target in targets:
        prefix = target["target_id"].rsplit("-", 1)[0]
        if "random-" in target["target_id"]:
            prefix = target["target_id"].rsplit("-random-", 1)[0]
        bucket = by_configuration.setdefault(prefix, {"statuses": {}, "valid_random_endpoint_errors": {"cpu": [], "cuda": []}, "all_targets": []})
        status = target["diagnostic_status"]
        bucket["statuses"][status] = bucket["statuses"].get(status, 0) + 1
        bucket["all_targets"].append(target)
        if "-random-" in target["target_id"] and status == "finite_f32_candidate_measured":
            for device in devices:
                reconstruction = target["device_reconstructions"].get(device)
                if reconstruction and reconstruction["physical_valid"]:
                    bucket["valid_random_endpoint_errors"][device].append(reconstruction["endpoint_max_abs_error"])
    for bucket in by_configuration.values():
        bucket["valid_random_endpoint_error_distribution"] = {
            device: _distribution(values)
            for device, values in bucket["valid_random_endpoint_errors"].items()
        }
        del bucket["valid_random_endpoint_errors"]
    return {
        "method_boundary": "diagnostic-only float64 evaluation of the approved inverse equations in this script; no production inverse or budget decision",
        "random_seed": BASE_SEED,
        "random_target_sampling": "NumPy default_rng(20261112); 128 one-dimensional valid interior targets per each of nine exact width configurations, half-width uniform from max(w/2+0.025,0.03) to 0.95, center fraction uniform [-0.98,0.98], endpoints rounded to float32 then independently hard-validated",
        "configuration_summaries": by_configuration,
    }


def _finite_difference_probe() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    width = hosting_frame_minimum_full_width(1e-4, 1e-4)
    for a in (-4.0, -1.0, 0.0, 1.0, 4.0):
        for b in (-4.0, -1.0, 0.0, 1.0, 4.0):
            p = 1.0 / (1.0 + math.exp(-a))
            h = (width + (2.0 - width) * p) / 2.0
            u = math.tanh(b)
            h_a = (2.0 - width) / 2.0 * p * (1.0 - p)
            c_b = (1.0 - h) * (1.0 - u * u)
            analytic = ((-(1.0 + u) * h_a, c_b), ((1.0 - u) * h_a, c_b))
            for step in (1e-3, 1e-4):
                def endpoints(raw_a: float, raw_b: float) -> tuple[float, float]:
                    p_value = 1.0 / (1.0 + math.exp(-raw_a))
                    half_value = (width + (2.0 - width) * p_value) / 2.0
                    center_value = (1.0 - half_value) * math.tanh(raw_b)
                    return center_value - half_value, center_value + half_value

                at = endpoints(a, b)
                plus_a, minus_a = endpoints(a + step, b), endpoints(a - step, b)
                plus_b, minus_b = endpoints(a, b + step), endpoints(a, b - step)
                finite = (
                    ((plus_a[0] - minus_a[0]) / (2 * step), (plus_b[0] - minus_b[0]) / (2 * step)),
                    ((plus_a[1] - minus_a[1]) / (2 * step), (plus_b[1] - minus_b[1]) / (2 * step)),
                )
                abs_errors = [
                    abs(finite[row][col] - analytic[row][col])
                    for row in range(2)
                    for col in range(2)
                ]
                rows.append({"a": a, "b": b, "step": step, "endpoints": at, "analytic_jacobian": analytic, "central_difference_jacobian": finite, "max_abs_error": max(abs_errors)})
    return {
        "method": "independent float64 formula and symmetric central differences; moderate raw coordinates only, no saturation; diagnostic step sizes, not acceptance budgets",
        "rows": rows,
        "max_abs_error_by_step": {
            str(step): max(row["max_abs_error"] for row in rows if row["step"] == step)
            for step in (1e-3, 1e-4)
        },
    }


def _aggregate_jacobian(cases: list[dict[str, Any]], devices: tuple[str, ...]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for device in devices:
        values: dict[str, list[float]] = {
            "sigma_min": [],
            "condition_2": [],
            "autograd_abs_error": [],
            "jacobian_abs_entry": [],
        }
        by_dimension: dict[int, dict[str, list[float]]] = {}
        for case in cases:
            dimension = int(case["dimension"])
            bucket = by_dimension.setdefault(
                dimension,
                {name: [] for name in values},
            )
            jacobian = case["device_results"][device]["jacobian_and_autograd"]
            sample_mapping = {
                "sigma_min": "sigma_min_samples",
                "condition_2": "condition_2_samples",
                "autograd_abs_error": "autograd_abs_error_samples",
                "jacobian_abs_entry": "jacobian_abs_entry_samples",
            }
            for metric, field in sample_mapping.items():
                for sample in jacobian[field]:
                    if isinstance(sample, str):
                        parsed = -math.inf if sample == "-Infinity" else math.inf if sample == "Infinity" else math.nan
                    else:
                        parsed = float(sample)
                    values[metric].append(parsed)
                    bucket[metric].append(parsed)
        aggregate[device] = {
            "scope": "all fixed core cases and seeded occupancy probes; per-coordinate raw-to-endpoint Jacobian in unscaled raw and normalized-domain units",
            "overall": {name: _distribution(items) for name, items in values.items()},
            "by_dimension": {
                str(dimension): {name: _distribution(items) for name, items in metrics.items()}
                for dimension, metrics in sorted(by_dimension.items())
            },
        }
    return aggregate


def _runtime_environment(devices: tuple[str, ...]) -> dict[str, Any]:
    details: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pytorch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "devices_measured": devices,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        details["cuda_device"] = {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "cudnn": torch.backends.cudnn.version(),
        }
        try:
            details["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            details["nvidia_smi"] = "unavailable"
    return details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reconstruction/BOX05_CP1_PROBE_RESULTS.json.gz"),
    )
    parser.add_argument("--cpu-only", action="store_true", help="omit CUDA measurements")
    args = parser.parse_args()
    devices = ("cpu",) if args.cpu_only or not torch.cuda.is_available() else ("cpu", "cuda")

    thresholds = _boundary_thresholds(devices)
    minimum_width_refinement = _minimum_width_boundary_probes(devices)
    core_cases = _core_cases(devices)
    occupancy_cases = _occupancy_probes(devices)
    result = {
        "schema": "town-model.box05-cp1-exploratory-probes.v1",
        "scope": "CP1 exploratory evidence only; no formal feature acceptance",
        "environment": _runtime_environment(devices),
        "fixed_grid_plan": {
            "dimensions": DIMENSIONS,
            "representative_class_counts": CLASS_COUNTS,
            "ordinary_epsilon": ORDINARY_EPSILONS,
            "hosting_epsilon_delta_pairs": HOSTING_CONFIGS,
            "raw_probe_values": RAW_PROBES,
            "coverage_policy": "Sparse predeclared coverage, not the Cartesian product. Each d receives 17 mixed/asymmetric raw cases at C=2 under ordinary and hosting defaults; all C values receive d=2 default-width cases; each nondefault width configuration receives one stated representative (d,C) case. Additional seeded occupancy, inverse and nextafter refinements are separate fixed probes.",
            "seed": BASE_SEED,
        },
        "core_cases": core_cases,
        "invalid_configuration_probes": _invalid_configuration_probes(),
        "activation_saturation_boundaries_by_device": thresholds,
        "minimum_width_boundary_refinement": minimum_width_refinement,
        "nextafter_boundary_geometry": _boundary_geometry_probes(devices, thresholds),
        "occupancy_anisotropy_probes": occupancy_cases,
        "gradient_conditioning_summary": _aggregate_jacobian(core_cases + occupancy_cases, devices),
        "inverse_representability": _inverse_probes(devices),
        "independent_derivative_finite_difference_probe": _finite_difference_probe(),
        "interpretation_limits": [
            "Finite samples do not prove continuous validity or supported intervals.",
            "Float64 equations are diagnostic references, not production transforms.",
            "Inverse candidates are exploratory only; no approved reconstruction budget exists and no production inverse is implemented.",
            "RNG reproducibility is claimed only for the recorded software/runtime; cross-version bitwise identity is not claimed.",
            "CPU/CUDA differences are measurements on the recorded device/software, not universal guarantees.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if args.output.suffix == ".gz":
        with args.output.open("wb") as raw_output:
            with gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as compressed_output:
                compressed_output.write(payload)
    else:
        args.output.write_bytes(payload)
    print(f"wrote {args.output}")
    print(f"devices={devices}; core_cases={len(result['core_cases'])}; inverse_configurations={len(result['inverse_representability']['configuration_summaries'])}")


if __name__ == "__main__":
    main()
