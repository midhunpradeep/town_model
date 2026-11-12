"""Independent exact-rational replay for the TOWN-03 CP2 review.

This review-only script is not imported by production code. It reads the frozen
manifest, derives clearance with Fraction arithmetic over stored binary32
values, derives Boolean membership from direct set inequalities, and compares
those independent expectations with actual CPU/CUDA evaluator executions.
"""

from __future__ import annotations

import json
import struct
from fractions import Fraction
from pathlib import Path

import torch

from town_model.models.bounded_box import box_signed_clearance
from town_model.models.town_geometry import (
    house_contains,
    house_signed_clearance,
    town_contains,
    town_signed_clearance,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/reconstruction/TOWN03_CP2_CONFIRMATION_SET.json"


def _q(value: float) -> Fraction:
    """Treat a stored binary32 value as its exact rational value."""
    return Fraction.from_float(float(value))


def _box_score(point, lower, upper) -> Fraction:
    faces = [_q(x) - _q(lo) for x, lo in zip(point, lower, strict=True)]
    faces.extend(_q(hi) - _q(x) for x, hi in zip(point, upper, strict=True))
    return min(faces)


def _house_score(point, lower, upper, window_lower, window_upper) -> Fraction:
    branches = [_box_score(point, lower, upper)]
    branches.extend(
        -_box_score(point, lo, hi)
        for lo, hi in zip(window_lower, window_upper, strict=True)
    )
    return min(branches)


def _house_member(point, lower, upper, window_lower, window_upper) -> bool:
    in_domain = all(-1.0 <= x <= 1.0 for x in point)
    in_frame = all(
        lo <= x <= hi
        for x, lo, hi in zip(point, lower, upper, strict=True)
    )
    in_window_interior = any(
        all(lo < x < hi for x, lo, hi in zip(point, lo_w, hi_w, strict=True))
        for lo_w, hi_w in zip(window_lower, window_upper, strict=True)
    )
    return in_domain and in_frame and not in_window_interior


def _oracles(case):
    rows = []
    for point in case["points"]:
        house_scores = [
            _house_score(point, lo, hi, win_lo, win_hi)
            for lo, hi, win_lo, win_hi in zip(
                case["frame_lower"],
                case["frame_upper"],
                case["window_lower"],
                case["window_upper"],
                strict=True,
            )
        ]
        house_members = [
            _house_member(point, lo, hi, win_lo, win_hi)
            for lo, hi, win_lo, win_hi in zip(
                case["frame_lower"],
                case["frame_upper"],
                case["window_lower"],
                case["window_upper"],
                strict=True,
            )
        ]
        rows.append((house_scores[0], max(house_scores), house_members[0], any(house_members)))
    return rows


def _ordered_f32_bits(value: float) -> int:
    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    if bits & 0x80000000:
        return 0x80000000 - (bits & 0x7FFFFFFF)
    return bits | 0x80000000


def _ulp_distance(left: float, right: float) -> int:
    return abs(_ordered_f32_bits(left) - _ordered_f32_bits(right))


def _f32_neighbors(value: Fraction) -> tuple[float, float, float]:
    rounded = torch.tensor(float(value), dtype=torch.float32)
    negative_inf = torch.tensor(float("-inf"), dtype=torch.float32)
    positive_inf = torch.tensor(float("inf"), dtype=torch.float32)
    return (
        float(torch.nextafter(rounded, negative_inf).item()),
        float(rounded.item()),
        float(torch.nextafter(rounded, positive_inf).item()),
    )


def _correctly_rounded_f32(value: Fraction) -> float:
    """Select the nearest adjacent float32 by exact rational distance."""
    candidates = _f32_neighbors(value)
    distances = [(abs(_q(candidate) - value), candidate) for candidate in candidates]
    minimum = min(distance for distance, _ in distances)
    tied = [candidate for distance, candidate in distances if distance == minimum]
    # IEEE-754 ties-to-even: the low significand bit is the representation LSB.
    return min(tied, key=lambda candidate: struct.unpack(">I", struct.pack(">f", candidate))[0] & 1)


def _inputs(case, device):
    d, houses, windows = case["d"], case["H"], case["W"]
    tensor = lambda values: torch.tensor(values, dtype=torch.float32, device=device)
    return (
        tensor(case["points"]).reshape(-1, d),
        tensor(case["frame_lower"]).reshape(houses, d),
        tensor(case["frame_upper"]).reshape(houses, d),
        tensor(case["window_lower"]).reshape(houses, windows, d),
        tensor(case["window_upper"]).reshape(houses, windows, d),
    )


def _run_manifest():
    cases = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    if len(cases) != 246:
        raise AssertionError(f"expected 246 manifest rows, got {len(cases)}")
    if not torch.cuda.is_available():
        raise RuntimeError("actual CUDA is required by the approved CP2 scope")

    devices = (torch.device("cpu"), torch.device("cuda:0"))
    observations = {
        str(device): {"house": [], "town": []} for device in devices
    }
    all_scores = {
        str(device): {"house": [], "town": []} for device in devices
    }
    boolean_mismatches = 0
    box_equivalence_calls = 0

    for case in cases:
        expected = _oracles(case)
        for device in devices:
            points, frame_lower, frame_upper, window_lower, window_upper = _inputs(case, device)
            house = house_signed_clearance(
                points, frame_lower[0], frame_upper[0], window_lower[0], window_upper[0]
            )
            town = town_signed_clearance(
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

            for kind, result, membership, score_index, member_index in (
                ("house", house, house_bits, 0, 2),
                ("town", town, town_bits, 1, 3),
            ):
                actual_values = result.detach().cpu().tolist()
                actual_bits = membership.detach().cpu().tolist()
                for index, (actual, row) in enumerate(zip(actual_values, expected, strict=True)):
                    oracle, wanted_member = row[score_index], row[member_index]
                    error = abs(_q(actual) - oracle)
                    scaled_error = error / max(Fraction(1), abs(oracle))
                    rounded_oracle = _correctly_rounded_f32(oracle)
                    near_zero = abs(oracle) <= Fraction(1, 1_000_000)
                    sign = (actual > 0) - (actual < 0)
                    oracle_sign = (oracle > 0) - (oracle < 0)
                    zero_mismatch = (actual == 0.0) != (oracle == 0)
                    if (
                        scaled_error > Fraction(1, 10_000_000)
                        or _ulp_distance(actual, rounded_oracle) > 1
                        or (near_zero and error > Fraction(1, 10_000_000))
                        or sign != oracle_sign
                        or zero_mismatch
                    ):
                        raise AssertionError((case["id"], str(device), kind, index))
                    if actual_bits[index] != wanted_member:
                        boolean_mismatches += 1
                observations[str(device)][kind].extend(
                    (oracle_rows[score_index],) for oracle_rows in expected
                )
                all_scores[str(device)][kind].extend(actual_values)

            if case["H"] == 1 and case["W"] == 0:
                box = box_signed_clearance(points, frame_lower[0], frame_upper[0])
                if not torch.equal(house, box) or not torch.equal(town, box):
                    raise AssertionError((case["id"], "H=1/W=0 BOX mismatch"))
                box_equivalence_calls += 1

    if boolean_mismatches:
        raise AssertionError(f"independent Boolean mismatches: {boolean_mismatches}")

    summaries = {}
    cross_device = {}
    for device in devices:
        name = str(device)
        summaries[name] = {}
        for kind in ("house", "town"):
            values = []
            for index, actual in enumerate(all_scores[name][kind]):
                oracle = observations[name][kind][index][0]
                error = abs(_q(actual) - oracle)
                rounded = _correctly_rounded_f32(oracle)
                values.append((error, error / max(Fraction(1), abs(oracle)), _ulp_distance(actual, rounded), abs(oracle) <= Fraction(1, 1_000_000)))
            near = [item[0] for item in values if item[3]]
            summaries[name][kind] = {
                "outputs": len(values),
                "max_absolute_error": float(max(item[0] for item in values)),
                "max_scaled_error": float(max(item[1] for item in values)),
                "max_ulp_error": max(item[2] for item in values),
                "near_zero_outputs": len(near),
                "near_zero_max_absolute_error": float(max(near, default=Fraction(0))),
            }

    for kind in ("house", "town"):
        pairs = zip(all_scores["cpu"][kind], all_scores["cuda:0"][kind], strict=True)
        pairs = list(pairs)
        cross_device[kind] = {
            "outputs": len(pairs),
            "max_absolute_difference": max((abs(a - b) for a, b in pairs), default=0.0),
            "max_ulp_difference": max((_ulp_distance(a, b) for a, b in pairs), default=0),
            "sign_mismatches": sum(((a > 0) - (a < 0)) != ((b > 0) - (b < 0)) for a, b in pairs),
            "zero_status_mismatches": sum((a == 0.0) != (b == 0.0) for a, b in pairs),
        }
    return {
        "cases": len(cases),
        "boolean_mismatches": boolean_mismatches,
        "H1_W0_BOX_equivalence_calls": box_equivalence_calls,
        "forward": summaries,
        "CPU_CUDA": cross_device,
    }


def _flatten(value):
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _flatten(child)]
    return [float(value)]


def _tensor(values, device, requires_grad=False):
    return torch.tensor(values, dtype=torch.float32, device=device, requires_grad=requires_grad)


def _gradient_probes(device):
    probes = {}

    points = _tensor([[-0.25, 0.0]], device, True)
    lower = _tensor([-0.5, -1.0], device, True)
    upper = _tensor([1.0, 1.0], device, True)
    empty = torch.empty((0, 2), dtype=torch.float32, device=device)
    score = house_signed_clearance(points, lower, upper, empty, empty).sum()
    probes["unique_frame"] = (score.item(), [x.detach().cpu().tolist() for x in torch.autograd.grad(score, (points, lower, upper))])

    points = _tensor([[0.4, 0.0]], device, True)
    frame_lower = _tensor([-1.0, -1.0], device, True)
    frame_upper = _tensor([1.0, 1.0], device, True)
    window_lower = _tensor([[-0.5, -0.5]], device, True)
    window_upper = _tensor([[0.5, 0.5]], device, True)
    score = house_signed_clearance(points, frame_lower, frame_upper, window_lower, window_upper).sum()
    probes["unique_window"] = (score.item(), [x.detach().cpu().tolist() for x in torch.autograd.grad(score, (points, frame_lower, frame_upper, window_lower, window_upper))])

    points = _tensor([[0.25]], device, True)
    frame_lower = _tensor([[-1.0], [0.0]], device, True)
    frame_upper = _tensor([[0.0], [1.0]], device, True)
    empty_town = torch.empty((2, 0, 1), dtype=torch.float32, device=device)
    score = town_signed_clearance(points, frame_lower, frame_upper, empty_town, empty_town).sum()
    probes["unique_town_frame"] = (score.item(), [x.detach().cpu().tolist() for x in torch.autograd.grad(score, (points, frame_lower, frame_upper))])

    points = _tensor([[0.4, 0.0]], device, True)
    frame_lower = _tensor([[-1.0, -1.0]], device, True)
    frame_upper = _tensor([[1.0, 1.0]], device, True)
    window_lower = _tensor([[[-0.5, -0.5]]], device, True)
    window_upper = _tensor([[[0.5, 0.5]]], device, True)
    score = town_signed_clearance(points, frame_lower, frame_upper, window_lower, window_upper).sum()
    probes["unique_town_window"] = (score.item(), [x.detach().cpu().tolist() for x in torch.autograd.grad(score, (points, frame_lower, frame_upper, window_lower, window_upper))])

    points = _tensor([[0.0]], device, True)
    lower = _tensor([-1.0], device, True)
    upper = _tensor([1.0], device, True)
    empty = torch.empty((0, 1), dtype=torch.float32, device=device)
    score = house_signed_clearance(points, lower, upper, empty, empty).sum()
    probes["tied_amin"] = (score.item(), [x.detach().cpu().tolist() for x in torch.autograd.grad(score, (points, lower, upper))])

    points = _tensor([[0.0]], device, True)
    frame_lower = _tensor([0.0], device, True)
    frame_upper = _tensor([1.0], device, True)
    window_lower = _tensor([[0.0]], device, True)
    window_upper = _tensor([[0.5]], device, True)
    score = house_signed_clearance(points, frame_lower, frame_upper, window_lower, window_upper).sum()
    probes["tied_house_min"] = (score.item(), [x.detach().cpu().tolist() for x in torch.autograd.grad(score, (points, frame_lower, window_lower))])

    points = _tensor([[0.0]], device, True)
    frame_lower = _tensor([[-1.0], [0.0]], device, True)
    frame_upper = _tensor([[0.0], [1.0]], device, True)
    empty_town = torch.empty((2, 0, 1), dtype=torch.float32, device=device)
    score = town_signed_clearance(points, frame_lower, frame_upper, empty_town, empty_town).sum()
    probes["tied_town_max"] = (score.item(), [x.detach().cpu().tolist() for x in torch.autograd.grad(score, (points, frame_lower, frame_upper))])

    point_x = torch.tensor(-0.25, dtype=torch.float32, device=device, requires_grad=True)
    lower_x = torch.tensor(-0.5, dtype=torch.float32, device=device, requires_grad=True)
    points = torch.stack((point_x, _tensor(0.0, device))).reshape(1, 2)
    lower = torch.stack((lower_x, _tensor(-1.0, device)))
    upper = _tensor([1.0, 1.0], device)
    empty = torch.empty((0, 2), dtype=torch.float32, device=device)
    score = house_signed_clearance(points, lower, upper, empty, empty).sum()
    point_grad, lower_grad = torch.autograd.grad(score, (point_x, lower_x))
    step = _tensor(1e-3, device)

    def evaluate(point_value, lower_value):
        moved_points = torch.stack((point_value, _tensor(0.0, device))).reshape(1, 2)
        moved_lower = torch.stack((lower_value, _tensor(-1.0, device)))
        return house_signed_clearance(moved_points, moved_lower, upper, empty, empty).sum()

    point_fd = ((evaluate(point_x.detach() + step, lower_x.detach()) - evaluate(point_x.detach() - step, lower_x.detach())) / (2 * step)).item()
    lower_fd = ((evaluate(point_x.detach(), lower_x.detach() + step) - evaluate(point_x.detach(), lower_x.detach() - step)) / (2 * step)).item()
    probes["finite_difference"] = (point_grad.item(), lower_grad.item(), point_fd, lower_fd)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return probes


def _verify_gradients():
    expected = {
        "unique_frame": (0.25, [[[1.0, 0.0]], [-1.0, 0.0], [0.0, 0.0]]),
        "unique_window": (-0.1, [[[1.0, 0.0]], [0.0, 0.0], [0.0, 0.0], [[0.0, 0.0]], [[-1.0, 0.0]]]),
        "unique_town_frame": (0.25, [[[1.0]], [[0.0], [-1.0]], [[0.0], [0.0]]]),
        "unique_town_window": (-0.1, [[[1.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[[0.0, 0.0]]], [[[-1.0, 0.0]]]]),
        "tied_amin": (1.0, [[[0.0]], [-0.5], [0.5]]),
        "tied_house_min": (0.0, [[[0.0]], [-0.5], [[0.5]]]),
        "tied_town_max": (0.0, [[[0.0]], [[0.0], [-0.5]], [[0.5], [0.0]]]),
    }
    runs = {str(device): _gradient_probes(device) for device in (torch.device("cpu"), torch.device("cuda:0"))}
    unique_gradient_error = tied_gradient_error = cross_device_gradient_error = 0.0
    unique_score_error = tied_score_error = cross_device_score_error = 0.0
    for name, (wanted_score, wanted_gradients) in expected.items():
        cpu_score, cpu_gradients = runs["cpu"][name]
        cuda_score, cuda_gradients = runs["cuda:0"][name]
        score_error = max(abs(cpu_score - wanted_score), abs(cuda_score - wanted_score))
        cross_device_score_error = max(cross_device_score_error, abs(cpu_score - cuda_score))
        gradient_errors = [
            abs(actual - wanted)
            for gradients in (cpu_gradients, cuda_gradients)
            for actual, wanted in zip(_flatten(gradients), _flatten(wanted_gradients), strict=True)
        ]
        cross_errors = [
            abs(cpu - cuda)
            for cpu, cuda in zip(_flatten(cpu_gradients), _flatten(cuda_gradients), strict=True)
        ]
        if name.startswith("unique_"):
            unique_gradient_error = max(unique_gradient_error, *gradient_errors)
            unique_score_error = max(unique_score_error, score_error)
        else:
            tied_gradient_error = max(tied_gradient_error, *gradient_errors)
            tied_score_error = max(tied_score_error, score_error)
        cross_device_gradient_error = max(cross_device_gradient_error, *cross_errors)
    fd_error = max(
        abs(value - expected_value)
        for device in ("cpu", "cuda:0")
        for value, expected_value in zip(runs[device]["finite_difference"], (1.0, -1.0, 1.0, -1.0), strict=True)
    )
    if (
        unique_gradient_error > 1e-6
        or tied_gradient_error > 1e-6
        or cross_device_gradient_error > 1e-6
        or fd_error > 2e-5
    ):
        raise AssertionError(
            (unique_gradient_error, tied_gradient_error, cross_device_gradient_error, fd_error)
        )
    return {
        "unique_gradient_max_error": unique_gradient_error,
        "tied_gradient_max_error": tied_gradient_error,
        "CPU_CUDA_gradient_max_difference": cross_device_gradient_error,
        "unique_probe_score_max_error": unique_score_error,
        "tied_probe_score_max_error": tied_score_error,
        "CPU_CUDA_probe_score_max_difference": cross_device_score_error,
        "finite_difference_max_error": fd_error,
    }


def main():
    print(json.dumps({"numerics": _run_manifest(), "gradients": _verify_gradients()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
