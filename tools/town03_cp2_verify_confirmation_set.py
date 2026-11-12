#!/usr/bin/env python3
"""Verify the frozen CP2 confirmation set against author-locked budgets."""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
from typing import Any

import torch

from town_model.models.town_geometry import (
    house_contains,
    house_signed_clearance,
    town_contains,
    town_signed_clearance,
)

MANIFEST_PATH = Path("docs/reconstruction/TOWN03_CP2_CONFIRMATION_SET.json")
LOCKED_MANIFEST_SHA256 = (
    "acb1f9c43110ac839530066ef2aa3311755003321b945b65345f3445b4548f3e"
)
ABS_RELATIVE_BUDGET = Decimal("1e-7")
NEAR_ZERO = Decimal("1e-6")
NEAR_ZERO_ABS_BUDGET = Decimal("1e-7")
ULP_BUDGET = 1


def _ordered_bits(value: float) -> int:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    if bits & 0x80000000:
        return 0x80000000 - (bits & 0x7FFFFFFF)
    return bits | 0x80000000


def _ulp(left: float, right: float) -> int:
    return abs(_ordered_bits(left) - _ordered_bits(right))


def _sign(value: Decimal | float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def _box_oracle(point: list[float], lower: list[float], upper: list[float]) -> Decimal:
    faces = [_decimal(x) - _decimal(lo) for x, lo in zip(point, lower, strict=True)]
    faces.extend(_decimal(hi) - _decimal(x) for x, hi in zip(point, upper, strict=True))
    return min(faces)


def _house_oracle(
    point: list[float],
    lower: list[float],
    upper: list[float],
    window_lower: list[list[float]],
    window_upper: list[list[float]],
) -> Decimal:
    return min(
        [
            _box_oracle(point, lower, upper),
            *(
                -_box_oracle(point, lo, hi)
                for lo, hi in zip(window_lower, window_upper, strict=True)
            ),
        ]
    )


def _boolean_house(
    point: list[float],
    lower: list[float],
    upper: list[float],
    window_lower: list[list[float]],
    window_upper: list[list[float]],
) -> bool:
    return (
        all(-1.0 <= value <= 1.0 for value in point)
        and all(
            lo <= value <= hi for value, lo, hi in zip(point, lower, upper, strict=True)
        )
        and not any(
            all(
                lo < value < hi
                for value, lo, hi in zip(point, win_lo, win_hi, strict=True)
            )
            for win_lo, win_hi in zip(window_lower, window_upper, strict=True)
        )
    )


def _oracle(case: dict[str, Any]) -> dict[str, list[Any]]:
    with localcontext() as context:
        context.prec = 100
        house_scores: list[Decimal] = []
        town_scores: list[Decimal] = []
        house_bits: list[bool] = []
        town_bits: list[bool] = []
        for point in case["points"]:
            per_house_scores = [
                _house_oracle(point, frame_lo, frame_hi, win_lo, win_hi)
                for frame_lo, frame_hi, win_lo, win_hi in zip(
                    case["frame_lower"],
                    case["frame_upper"],
                    case["window_lower"],
                    case["window_upper"],
                    strict=True,
                )
            ]
            per_house_bits = [
                _boolean_house(point, frame_lo, frame_hi, win_lo, win_hi)
                for frame_lo, frame_hi, win_lo, win_hi in zip(
                    case["frame_lower"],
                    case["frame_upper"],
                    case["window_lower"],
                    case["window_upper"],
                    strict=True,
                )
            ]
            house_scores.append(per_house_scores[0])
            town_scores.append(max(per_house_scores))
            house_bits.append(per_house_bits[0])
            town_bits.append(any(per_house_bits))
        return {
            "house_decimal": house_scores,
            "town_decimal": town_scores,
            "house_membership": house_bits,
            "town_membership": town_bits,
        }


def _tensors(case: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    d, h, w = int(case["d"]), int(case["H"]), int(case["W"])
    return (
        torch.tensor(case["points"], dtype=torch.float32, device=device).reshape(-1, d),
        torch.tensor(case["frame_lower"], dtype=torch.float32, device=device).reshape(
            h, d
        ),
        torch.tensor(case["frame_upper"], dtype=torch.float32, device=device).reshape(
            h, d
        ),
        torch.tensor(case["window_lower"], dtype=torch.float32, device=device).reshape(
            h, w, d
        ),
        torch.tensor(case["window_upper"], dtype=torch.float32, device=device).reshape(
            h, w, d
        ),
    )


def _evaluate(case: dict[str, Any], device: torch.device) -> dict[str, list[Any]]:
    points, frame_lower, frame_upper, window_lower, window_upper = _tensors(
        case, device
    )
    house_scores = house_signed_clearance(
        points, frame_lower[0], frame_upper[0], window_lower[0], window_upper[0]
    )
    town_scores = town_signed_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper
    )
    house_bits = house_contains(
        points, frame_lower[0], frame_upper[0], window_lower[0], window_upper[0]
    )
    town_bits = town_contains(
        points, frame_lower, frame_upper, window_lower, window_upper
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "house_clearance": house_scores.detach().cpu().tolist(),
        "town_clearance": town_scores.detach().cpu().tolist(),
        "house_membership": house_bits.detach().cpu().tolist(),
        "town_membership": town_bits.detach().cpu().tolist(),
    }


def _gradient_probes(device: torch.device) -> dict[str, Any]:
    def tensor(values: Any, requires_grad: bool = False) -> torch.Tensor:
        return torch.tensor(
            values, dtype=torch.float32, device=device, requires_grad=requires_grad
        )

    result: dict[str, Any] = {}

    points = tensor([[-0.25, 0.0]], True)
    lower = tensor([-0.5, -1.0], True)
    upper = tensor([1.0, 1.0], True)
    empty = torch.empty((0, 2), dtype=torch.float32, device=device)
    score = house_signed_clearance(points, lower, upper, empty, empty).sum()
    grads = torch.autograd.grad(score, (points, lower, upper))
    result["unique_frame"] = {
        "score": score.item(),
        "grads": [value.detach().cpu().tolist() for value in grads],
    }

    points = tensor([[0.4, 0.0]], True)
    frame_lower = tensor([-1.0, -1.0], True)
    frame_upper = tensor([1.0, 1.0], True)
    window_lower = tensor([[-0.5, -0.5]], True)
    window_upper = tensor([[0.5, 0.5]], True)
    score = house_signed_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper
    ).sum()
    grads = torch.autograd.grad(
        score, (points, frame_lower, frame_upper, window_lower, window_upper)
    )
    result["unique_window"] = {
        "score": score.item(),
        "grads": [value.detach().cpu().tolist() for value in grads],
    }

    points = tensor([[0.25]], True)
    lower = tensor([[-1.0], [0.0]], True)
    upper = tensor([[0.0], [1.0]], True)
    empty = torch.empty((2, 0, 1), dtype=torch.float32, device=device)
    score = town_signed_clearance(points, lower, upper, empty, empty).sum()
    grads = torch.autograd.grad(score, (points, lower, upper))
    result["unique_town_frame"] = {
        "score": score.item(),
        "grads": [value.detach().cpu().tolist() for value in grads],
    }

    points = tensor([[0.4, 0.0]], True)
    frame_lower = tensor([[-1.0, -1.0]], True)
    frame_upper = tensor([[1.0, 1.0]], True)
    window_lower = tensor([[[-0.5, -0.5]]], True)
    window_upper = tensor([[[0.5, 0.5]]], True)
    score = town_signed_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper
    ).sum()
    grads = torch.autograd.grad(
        score, (points, frame_lower, frame_upper, window_lower, window_upper)
    )
    result["unique_town_window"] = {
        "score": score.item(),
        "grads": [value.detach().cpu().tolist() for value in grads],
    }

    points = tensor([[0.0]], True)
    lower = tensor([-1.0], True)
    upper = tensor([1.0], True)
    empty = torch.empty((0, 1), dtype=torch.float32, device=device)
    score = house_signed_clearance(points, lower, upper, empty, empty).sum()
    grads = torch.autograd.grad(score, (points, lower, upper))
    result["tied_amin_faces"] = {
        "score": score.item(),
        "grads": [value.detach().cpu().tolist() for value in grads],
    }

    points = tensor([[0.0]], True)
    frame_lower = tensor([0.0], True)
    frame_upper = tensor([1.0], True)
    window_lower = tensor([[0.0]], True)
    window_upper = tensor([[0.5]], True)
    score = house_signed_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper
    ).sum()
    grads = torch.autograd.grad(score, (points, frame_lower, window_lower))
    result["tied_house_branches"] = {
        "score": score.item(),
        "grads": [value.detach().cpu().tolist() for value in grads],
    }

    points = tensor([[0.0]], True)
    lower = tensor([[-1.0], [0.0]], True)
    upper = tensor([[0.0], [1.0]], True)
    empty = torch.empty((2, 0, 1), dtype=torch.float32, device=device)
    score = town_signed_clearance(points, lower, upper, empty, empty).sum()
    grads = torch.autograd.grad(score, (points, lower, upper))
    result["tied_town_amax"] = {
        "score": score.item(),
        "grads": [value.detach().cpu().tolist() for value in grads],
    }

    point_x = tensor(-0.25, True)
    lower_x = tensor(-0.5, True)
    points = torch.stack((point_x, tensor(0.0))).reshape(1, 2)
    lower = torch.stack((lower_x, tensor(-1.0)))
    upper = tensor([1.0, 1.0])
    empty = torch.empty((0, 2), dtype=torch.float32, device=device)
    score = house_signed_clearance(points, lower, upper, empty, empty).sum()
    point_grad, lower_grad = torch.autograd.grad(score, (point_x, lower_x))
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
    lower_fd = (
        evaluate(point_x.detach(), lower_x.detach() + step)
        - evaluate(point_x.detach(), lower_x.detach() - step)
    ) / (2 * step)
    result["finite_difference"] = {
        "score": score.item(),
        "step": step.item(),
        "analytic_gradients": [point_grad.item(), lower_grad.item()],
        "finite_difference_gradients": [point_fd.item(), lower_fd.item()],
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result


def _nested_values(value: Any) -> list[float]:
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_nested_values(item))
        return flattened
    return [float(value)]


def _metric(actual: float, oracle: Decimal) -> dict[str, Any]:
    with localcontext() as context:
        context.prec = 100
        error = abs(_decimal(actual) - oracle)
        scaled_error = error / max(Decimal(1), abs(oracle))
    rounded = torch.tensor(float(oracle), dtype=torch.float32).item()
    return {
        "actual": actual,
        "oracle_decimal": str(oracle),
        "correctly_rounded_oracle_f32": rounded,
        "absolute_error": str(error),
        "scaled_error": str(scaled_error),
        "ulp_error": _ulp(actual, rounded),
        "near_zero": abs(oracle) <= NEAR_ZERO,
        "sign_crossing": _sign(actual) != _sign(oracle),
        "zero_status_mismatch": (actual == 0.0) != (oracle == 0),
    }


def _summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    abs_errors = [float(value["absolute_error"]) for value in metrics]
    scaled_errors = [float(value["scaled_error"]) for value in metrics]
    near_errors = [
        float(value["absolute_error"]) for value in metrics if value["near_zero"]
    ]
    return {
        "outputs": len(metrics),
        "max_absolute_error": max(abs_errors, default=0.0),
        "max_scaled_error": max(scaled_errors, default=0.0),
        "max_ulp_error": max((int(value["ulp_error"]) for value in metrics), default=0),
        "near_zero_outputs": len(near_errors),
        "near_zero_max_absolute_error": max(near_errors, default=0.0),
        "sign_crossings": sum(bool(value["sign_crossing"]) for value in metrics),
        "zero_status_mismatches": sum(
            bool(value["zero_status_mismatch"]) for value in metrics
        ),
    }


def _device_environment() -> dict[str, Any]:
    driver = None
    if torch.cuda.is_available():
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            driver = result.stdout.strip().splitlines()[0]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver,
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "cuda_capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
    }


def run(manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    payload = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(payload).hexdigest()
    if manifest_sha != LOCKED_MANIFEST_SHA256:
        raise RuntimeError(
            f"frozen CP2 manifest hash mismatch: {manifest_sha} != {LOCKED_MANIFEST_SHA256}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CP2 approval requires actual CUDA; CUDA is unavailable")
    manifest = json.loads(payload)
    cases = manifest["cases"]
    if len(cases) != 246:
        raise RuntimeError(f"frozen CP2 set has {len(cases)} cases, expected 246")

    devices = (torch.device("cpu"), torch.device("cuda:0"))
    all_metrics: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(device): {"house": [], "town": []} for device in devices
    }
    device_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    boolean_mismatches = 0
    cpu_cuda_metrics: dict[str, list[dict[str, Any]]] = {"house": [], "town": []}

    for case in cases:
        expected = _oracle(case)
        for name, manifest_key, oracle_key in (
            ("house", "house_clearance_decimal", "house_decimal"),
            ("town", "town_clearance_decimal", "town_decimal"),
        ):
            serialized = [Decimal(value) for value in case[manifest_key]]
            if serialized != expected[oracle_key]:
                raise RuntimeError(
                    f"independent oracle disagrees with locked manifest for {case['id']} {name}"
                )
        if (
            expected["house_membership"] != case["house_membership"]
            or expected["town_membership"] != case["town_membership"]
        ):
            raise RuntimeError(
                f"independent Boolean oracle disagrees with locked manifest for {case['id']}"
            )

        case_results: dict[str, Any] = {
            "id": case["id"],
            "group": case["group"],
            "d": case["d"],
            "H": case["H"],
            "W": case["W"],
        }
        per_device: dict[str, Any] = {}
        for device in devices:
            actual = _evaluate(case, device)
            device_name = str(device)
            per_device[device_name] = actual
            case_measurements: dict[str, Any] = {}
            for name, decimal_key, membership_key in (
                ("house", "house_decimal", "house_membership"),
                ("town", "town_decimal", "town_membership"),
            ):
                metrics = [
                    _metric(value, oracle)
                    for value, oracle in zip(
                        actual[f"{name}_clearance"], expected[decimal_key], strict=True
                    )
                ]
                all_metrics[device_name][name].extend(metrics)
                expected_membership = expected[membership_key]
                if actual[f"{name}_membership"] != expected_membership:
                    boolean_mismatches += sum(
                        got != want
                        for got, want in zip(
                            actual[f"{name}_membership"],
                            expected_membership,
                            strict=True,
                        )
                    )
                    failures.append(
                        {
                            "id": case["id"],
                            "device": device_name,
                            "type": f"{name}_boolean_mismatch",
                        }
                    )
                for index, metric in enumerate(metrics):
                    oracle = expected[decimal_key][index]
                    allowed_scaled = ABS_RELATIVE_BUDGET
                    if (
                        Decimal(metric["scaled_error"]) > allowed_scaled
                        or int(metric["ulp_error"]) > ULP_BUDGET
                    ):
                        failures.append(
                            {
                                "id": case["id"],
                                "device": device_name,
                                "type": f"{name}_forward_budget",
                                "metric": metric,
                            }
                        )
                    if (
                        abs(oracle) <= NEAR_ZERO
                        and Decimal(metric["absolute_error"]) > NEAR_ZERO_ABS_BUDGET
                    ):
                        failures.append(
                            {
                                "id": case["id"],
                                "device": device_name,
                                "type": f"{name}_near_zero_budget",
                                "metric": metric,
                            }
                        )
                    if metric["sign_crossing"] or metric["zero_status_mismatch"]:
                        failures.append(
                            {
                                "id": case["id"],
                                "device": device_name,
                                "type": f"{name}_sign_or_zero_crossing",
                                "metric": metric,
                            }
                        )
                case_measurements[name] = metrics
            case_results[device_name] = {
                "actual": actual,
                "measurements": case_measurements,
            }
            if device == torch.device("cpu"):
                per_device[device_name] = actual
            else:
                for name in ("house", "town"):
                    cpu_values = per_device["cpu"][f"{name}_clearance"]
                    cuda_values = actual[f"{name}_clearance"]
                    for index, (cpu_value, cuda_value) in enumerate(
                        zip(cpu_values, cuda_values, strict=True)
                    ):
                        diff = {
                            "absolute_difference": abs(cpu_value - cuda_value),
                            "ulp_difference": _ulp(cpu_value, cuda_value),
                            "sign_mismatch": _sign(cpu_value) != _sign(cuda_value),
                            "zero_status_mismatch": (cpu_value == 0.0)
                            != (cuda_value == 0.0),
                        }
                        cpu_cuda_metrics[name].append(diff)
                        if (
                            diff["ulp_difference"] > ULP_BUDGET
                            or diff["sign_mismatch"]
                            or diff["zero_status_mismatch"]
                        ):
                            failures.append(
                                {
                                    "id": case["id"],
                                    "type": f"{name}_cpu_cuda_budget",
                                    "index": index,
                                    "metric": diff,
                                }
                            )
        device_results.append(case_results)

    if boolean_mismatches:
        failures.append(
            {"type": "boolean_membership_mismatches", "count": boolean_mismatches}
        )

    summaries = {
        device: {name: _summary(values) for name, values in by_kind.items()}
        for device, by_kind in all_metrics.items()
    }
    cross_device = {
        name: {
            "outputs": len(values),
            "max_absolute_difference": max(
                (item["absolute_difference"] for item in values), default=0.0
            ),
            "max_ulp_difference": max(
                (item["ulp_difference"] for item in values), default=0
            ),
            "sign_mismatches": sum(item["sign_mismatch"] for item in values),
            "zero_status_mismatches": sum(
                item["zero_status_mismatch"] for item in values
            ),
        }
        for name, values in cpu_cuda_metrics.items()
    }

    gradient_cpu = _gradient_probes(torch.device("cpu"))
    gradient_cuda = _gradient_probes(torch.device("cuda:0"))
    analytic_gradients = {
        "unique_frame": [[[1.0, 0.0]], [-1.0, 0.0], [0.0, 0.0]],
        "unique_window": [
            [[1.0, 0.0]],
            [0.0, 0.0],
            [0.0, 0.0],
            [[0.0, 0.0]],
            [[-1.0, 0.0]],
        ],
        "unique_town_frame": [[[1.0]], [[0.0], [-1.0]], [[0.0], [0.0]]],
        "unique_town_window": [
            [[1.0, 0.0]],
            [[0.0, 0.0]],
            [[0.0, 0.0]],
            [[[0.0, 0.0]]],
            [[[-1.0, 0.0]]],
        ],
        "tied_amin_faces": [[[0.0]], [-0.5], [0.5]],
        "tied_house_branches": [[[0.0]], [-0.5], [[0.5]]],
        "tied_town_amax": [[[0.0]], [[0.0], [-0.5]], [[0.5], [0.0]]],
    }
    max_unique_gradient_error = 0.0
    max_tied_gradient_error = 0.0
    max_cpu_cuda_gradient_error = 0.0
    for name, expected in analytic_gradients.items():
        cpu_flat = _nested_values(gradient_cpu[name]["grads"])
        cuda_flat = _nested_values(gradient_cuda[name]["grads"])
        expected_flat = _nested_values(expected)
        unique_error = max(
            (
                abs(actual - wanted)
                for actual, wanted in zip(cpu_flat, expected_flat, strict=True)
            ),
            default=0.0,
        )
        cuda_error = max(
            (
                abs(actual - wanted)
                for actual, wanted in zip(cuda_flat, expected_flat, strict=True)
            ),
            default=0.0,
        )
        cross_error = max(
            (abs(cpu - cuda) for cpu, cuda in zip(cpu_flat, cuda_flat, strict=True)),
            default=0.0,
        )
        if name.startswith("unique_"):
            max_unique_gradient_error = max(
                max_unique_gradient_error, unique_error, cuda_error
            )
        else:
            max_tied_gradient_error = max(
                max_tied_gradient_error, unique_error, cuda_error
            )
        max_cpu_cuda_gradient_error = max(max_cpu_cuda_gradient_error, cross_error)
    fd_analytic = [1.0, -1.0]
    finite_difference_error = max(
        abs(value - expected)
        for observations in (gradient_cpu, gradient_cuda)
        for value, expected in zip(
            observations["finite_difference"]["finite_difference_gradients"],
            fd_analytic,
            strict=True,
        )
    )
    if max_unique_gradient_error > 1e-6:
        failures.append(
            {
                "type": "unique_branch_gradient_budget",
                "max_error": max_unique_gradient_error,
            }
        )
    if max_tied_gradient_error > 1e-6:
        failures.append(
            {"type": "tied_gradient_budget", "max_error": max_tied_gradient_error}
        )
    if max_cpu_cuda_gradient_error > 1e-6:
        failures.append(
            {
                "type": "cpu_cuda_gradient_budget",
                "max_error": max_cpu_cuda_gradient_error,
            }
        )
    if finite_difference_error > 2e-5:
        failures.append(
            {"type": "finite_difference_budget", "max_error": finite_difference_error}
        )
    gradient_metrics = {
        "max_unique_gradient_abs_error": max_unique_gradient_error,
        "max_tied_gradient_abs_error": max_tied_gradient_error,
        "max_cpu_cuda_gradient_abs_difference": max_cpu_cuda_gradient_error,
        "max_finite_difference_abs_error": finite_difference_error,
        "cpu_observations": gradient_cpu,
        "cuda_observations": gradient_cuda,
    }
    return {
        "status": "PASS" if not failures else "FAIL",
        "classification": "CP2 numerical confirmation against author-locked budgets",
        "manifest_sha256": manifest_sha,
        "manifest_case_count": len(cases),
        "group_counts": manifest["group_counts"],
        "locked_limits": {
            "max_ulp": ULP_BUDGET,
            "scaled_error": str(ABS_RELATIVE_BUDGET),
            "near_zero_abs_error": str(NEAR_ZERO_ABS_BUDGET),
            "near_zero_threshold": str(NEAR_ZERO),
            "zero_and_sign_crossings": 0,
            "cpu_cuda_max_ulp": ULP_BUDGET,
        },
        "device_metrics": summaries,
        "cpu_cuda_metrics": cross_device,
        "gradient_metrics": gradient_metrics,
        "boolean_mismatches": boolean_mismatches,
        "environment": _device_environment(),
        "failures": failures,
        "per_case_results": device_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reconstruction/TOWN03_CP2_CONFIRMATION_RESULTS.json"),
    )
    args = parser.parse_args()
    result = run(args.manifest)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "manifest_sha256",
                    "manifest_case_count",
                    "group_counts",
                    "locked_limits",
                    "device_metrics",
                    "cpu_cuda_metrics",
                    "gradient_metrics",
                    "boolean_mismatches",
                    "environment",
                    "failures",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
