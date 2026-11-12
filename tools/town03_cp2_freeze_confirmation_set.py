#!/usr/bin/env python3
"""Freeze the author-approved CP2 input/oracle manifest without calling evaluators.

This generator deliberately imports no town_model evaluator. Its Decimal
oracle and direct Boolean set checks are independent of the production path.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
import json
import random
from pathlib import Path
import sys

import torch

SEED = 20261113
CONFIGS = [(d, h, w) for d in (1, 2) for h in (1, 2) for w in (0, 1, 2)]
LITERAL_FIXTURES = (
    {
        "id": "literal-house-1d-w0-interior",
        "points": [[0.25]],
        "frame_lower": [[-1.0]],
        "frame_upper": [[1.0]],
        "window_lower": [[]],
        "window_upper": [[]],
        "expected": [0.75],
        "membership": [True],
    },
    {
        "id": "literal-house-1d-frame-face",
        "points": [[-1.0]],
        "frame_lower": [[-1.0]],
        "frame_upper": [[1.0]],
        "window_lower": [[]],
        "window_upper": [[]],
        "expected": [0.0],
        "membership": [True],
    },
    {
        "id": "literal-house-1d-outside-domain",
        "points": [[1.5]],
        "frame_lower": [[-1.0]],
        "frame_upper": [[1.0]],
        "window_lower": [[]],
        "window_upper": [[]],
        "expected": [-0.5],
        "membership": [False],
    },
    {
        "id": "literal-house-1d-window-interior",
        "points": [[0.0]],
        "frame_lower": [[-1.0]],
        "frame_upper": [[1.0]],
        "window_lower": [[[-0.25]]],
        "window_upper": [[[0.25]]],
        "expected": [-0.25],
        "membership": [False],
    },
    {
        "id": "literal-house-1d-window-face",
        "points": [[0.25]],
        "frame_lower": [[-1.0]],
        "frame_upper": [[1.0]],
        "window_lower": [[[-0.25]]],
        "window_upper": [[[0.25]]],
        "expected": [0.0],
        "membership": [True],
    },
    {
        "id": "literal-house-1d-overlapping-windows",
        "points": [[0.5]],
        "frame_lower": [[-1.0]],
        "frame_upper": [[1.0]],
        "window_lower": [[[-0.5], [0.0]]],
        "window_upper": [[[0.25], [0.75]]],
        "expected": [-0.25],
        "membership": [False],
    },
    {
        "id": "literal-house-2d-box-equivalence",
        "points": [[0.0, 0.0]],
        "frame_lower": [[-1.0, -0.5]],
        "frame_upper": [[0.5, 1.0]],
        "window_lower": [[]],
        "window_upper": [[]],
        "expected": [0.5],
        "membership": [True],
    },
    {
        "id": "literal-house-2d-degenerate-window",
        "points": [[0.0, 0.0]],
        "frame_lower": [[-1.0, -1.0]],
        "frame_upper": [[1.0, 1.0]],
        "window_lower": [[[0.0, -0.75]]],
        "window_upper": [[[0.0, 0.75]]],
        "expected": [0.0],
        "membership": [True],
    },
    {
        "id": "literal-house-2d-degenerate-frame",
        "points": [[0.0, 0.0]],
        "frame_lower": [[0.0, 0.0]],
        "frame_upper": [[0.0, 0.0]],
        "window_lower": [[]],
        "window_upper": [[]],
        "expected": [0.0],
        "membership": [True],
    },
    {
        "id": "literal-house-2d-protruding-window",
        "points": [[0.5, 0.5]],
        "frame_lower": [[-0.5, -0.5]],
        "frame_upper": [[0.5, 0.5]],
        "window_lower": [[[-0.75, -0.75]]],
        "window_upper": [[[0.75, 0.75]]],
        "expected": [-0.25],
        "membership": [False],
    },
    {
        "id": "literal-town-1d-union-positive",
        "points": [[0.25]],
        "frame_lower": [[-1.0], [0.0]],
        "frame_upper": [[0.0], [1.0]],
        "window_lower": [[], []],
        "window_upper": [[], []],
        "expected": [0.25],
        "membership": [True],
    },
    {
        "id": "literal-town-1d-touching-house-tie",
        "points": [[0.0]],
        "frame_lower": [[-1.0], [0.0]],
        "frame_upper": [[0.0], [1.0]],
        "window_lower": [[], []],
        "window_upper": [[], []],
        "expected": [0.0],
        "membership": [True],
    },
    {
        "id": "literal-town-1d-hole-interior",
        "points": [[0.0]],
        "frame_lower": [[-1.0]],
        "frame_upper": [[1.0]],
        "window_lower": [[[-0.25]]],
        "window_upper": [[[0.25]]],
        "expected": [-0.25],
        "membership": [False],
    },
    {
        "id": "literal-town-1d-windowed-house-union",
        "points": [[-0.375]],
        "frame_lower": [[-1.0], [0.0]],
        "frame_upper": [[0.0], [1.0]],
        "window_lower": [[[-0.5]], [[0.25]]],
        "window_upper": [[[-0.25]], [[0.5]]],
        "expected": [-0.125],
        "membership": [False],
    },
    {
        "id": "literal-town-2d-window-boundary-adjacent",
        "points": [[0.0, 0.75]],
        "frame_lower": [[-1.0, -1.0]],
        "frame_upper": [[1.0, 1.0]],
        "window_lower": [[[-0.5, -0.5]]],
        "window_upper": [[[0.5, 0.5]]],
        "expected": [0.25],
        "membership": [True],
    },
    {
        "id": "literal-town-2d-two-houses-two-windows",
        "points": [[0.0, 0.0]],
        "frame_lower": [[-1.0, -1.0], [-0.5, -0.5]],
        "frame_upper": [[1.0, 1.0], [0.5, 0.5]],
        "window_lower": [[[-0.5, -0.5], [0.0, -0.25]], [[0.25, 0.25], [-0.75, -0.75]]],
        "window_upper": [[[0.5, 0.5], [0.75, 0.25]], [[0.75, 0.75], [-0.25, 0.25]]],
        "expected": [0.25],
        "membership": [True],
    },
    {
        "id": "literal-town-1d-two-window-boundaries",
        "points": [[0.0]],
        "frame_lower": [[-1.0], [0.0]],
        "frame_upper": [[0.0], [1.0]],
        "window_lower": [[[-0.5]], [[0.0]]],
        "window_upper": [[[0.0]], [[0.5]]],
        "expected": [0.0],
        "membership": [True],
    },
    {
        "id": "literal-town-1d-degenerate-house-and-face",
        "points": [[0.0]],
        "frame_lower": [[0.0], [-0.5]],
        "frame_upper": [[0.0], [0.0]],
        "window_lower": [[], []],
        "window_upper": [[], []],
        "expected": [0.0],
        "membership": [True],
    },
)


def _f32(value: float) -> float:
    return torch.tensor(value, dtype=torch.float32).item()


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def _box_score(point: list[float], lower: list[float], upper: list[float]) -> Decimal:
    faces = [_decimal(x) - _decimal(lo) for x, lo in zip(point, lower, strict=True)]
    faces.extend(_decimal(hi) - _decimal(x) for x, hi in zip(point, upper, strict=True))
    return min(faces)


def _house_score(
    point: list[float],
    lower: list[float],
    upper: list[float],
    window_lower: list[list[float]],
    window_upper: list[list[float]],
) -> Decimal:
    return min(
        [
            _box_score(point, lower, upper),
            *(
                -_box_score(point, lo, hi)
                for lo, hi in zip(window_lower, window_upper, strict=True)
            ),
        ]
    )


def _house_member(
    point: list[float],
    lower: list[float],
    upper: list[float],
    window_lower: list[list[float]],
    window_upper: list[list[float]],
) -> bool:
    in_domain = all(-1.0 <= x <= 1.0 for x in point)
    in_frame = all(lo <= x <= hi for x, lo, hi in zip(point, lower, upper, strict=True))
    in_window_interior = any(
        all(lo < x < hi for x, lo, hi in zip(point, win_lo, win_hi, strict=True))
        for win_lo, win_hi in zip(window_lower, window_upper, strict=True)
    )
    return in_domain and in_frame and not in_window_interior


def _oracle(case: dict[str, object]) -> dict[str, object]:
    with localcontext() as context:
        context.prec = 100
        points = case["points"]
        frame_lower = case["frame_lower"]
        frame_upper = case["frame_upper"]
        window_lower = case["window_lower"]
        window_upper = case["window_upper"]
        house_scores = []
        town_scores = []
        house_membership = []
        town_membership = []
        for point in points:
            house_scores.append(
                _house_score(
                    point,
                    frame_lower[0],
                    frame_upper[0],
                    window_lower[0],
                    window_upper[0],
                )
            )
            by_house_scores = [
                _house_score(point, lo, hi, wins_lo, wins_hi)
                for lo, hi, wins_lo, wins_hi in zip(
                    frame_lower, frame_upper, window_lower, window_upper, strict=True
                )
            ]
            by_house_membership = [
                _house_member(point, lo, hi, wins_lo, wins_hi)
                for lo, hi, wins_lo, wins_hi in zip(
                    frame_lower, frame_upper, window_lower, window_upper, strict=True
                )
            ]
            town_scores.append(max(by_house_scores))
            house_membership.append(
                _house_member(
                    point,
                    frame_lower[0],
                    frame_upper[0],
                    window_lower[0],
                    window_upper[0],
                )
            )
            town_membership.append(any(by_house_membership))
        return {
            "house_clearance_decimal": [str(x) for x in house_scores],
            "town_clearance_decimal": [str(x) for x in town_scores],
            "house_clearance_f32": [
                torch.tensor(float(x), dtype=torch.float32).item() for x in house_scores
            ],
            "town_clearance_f32": [
                torch.tensor(float(x), dtype=torch.float32).item() for x in town_scores
            ],
            "house_membership": house_membership,
            "town_membership": town_membership,
        }


def _case_record(
    case_id: str,
    group: str,
    d: int,
    h: int,
    w: int,
    points: list[list[float]],
    frame_lower: list[list[float]],
    frame_upper: list[list[float]],
    window_lower: list[list[list[float]]],
    window_upper: list[list[list[float]]],
    literal_expected: list[float] | None = None,
    literal_membership: list[bool] | None = None,
) -> dict[str, object]:
    stored = {
        "id": case_id,
        "group": group,
        "d": d,
        "H": h,
        "W": w,
        "points": [[_f32(v) for v in row] for row in points],
        "frame_lower": [[_f32(v) for v in row] for row in frame_lower],
        "frame_upper": [[_f32(v) for v in row] for row in frame_upper],
        "window_lower": [
            [[_f32(v) for v in row] for row in house] for house in window_lower
        ],
        "window_upper": [
            [[_f32(v) for v in row] for row in house] for house in window_upper
        ],
    }
    oracle = _oracle(stored)
    if literal_expected is not None:
        expected_decimal = [Decimal.from_float(_f32(v)) for v in literal_expected]
        if [Decimal(v) for v in oracle["town_clearance_decimal"]] != expected_decimal:
            raise AssertionError(
                f"independent literal clearance mismatch for {case_id}: {oracle}"
            )
        if oracle["town_membership"] != literal_membership:
            raise AssertionError(
                f"independent literal membership mismatch for {case_id}: {oracle}"
            )
        oracle["literal_expected_clearance"] = literal_expected
        oracle["literal_expected_membership"] = literal_membership
    stored.update(oracle)
    return stored


def _literal_cases() -> list[dict[str, object]]:
    records = []
    for fixture in LITERAL_FIXTURES:
        d = len(fixture["points"][0])
        h = len(fixture["frame_lower"])
        w = len(fixture["window_lower"][0])
        records.append(
            _case_record(
                fixture["id"],
                "literal",
                d,
                h,
                w,
                fixture["points"],
                fixture["frame_lower"],
                fixture["frame_upper"],
                fixture["window_lower"],
                fixture["window_upper"],
                fixture["expected"],
                fixture["membership"],
            )
        )
    return records


def _boundary_case(d: int, h: int, w: int, side: int) -> dict[str, object]:
    lower = [_f32(-0.6137), _f32(-0.3719)][:d]
    upper = [_f32(0.8123), _f32(0.7331)][:d]
    lower_t = torch.tensor(lower, dtype=torch.float32)
    upper_t = torch.tensor(upper, dtype=torch.float32)
    point = []
    for axis, (lo, hi) in enumerate(zip(lower_t, upper_t, strict=True)):
        if axis == 0 and side:
            direction = torch.tensor(
                float("inf") if side > 0 else float("-inf"), dtype=torch.float32
            )
            point.append(torch.nextafter(lo, direction).item())
        else:
            point.append(_f32((lo.item() + hi.item()) / 2))
    wins_lower = []
    wins_upper = []
    for _ in range(h):
        wins_lower.append([[0.25 if wi == 0 else 0.55] * d for wi in range(w)])
        wins_upper.append([[0.5 if wi == 0 else 0.75] * d for wi in range(w)])
    label = {-1: "below", 0: "exact", 1: "above"}[side]
    return _case_record(
        f"nextafter-{label}-d{d}-h{h}-w{w}",
        "nextafter",
        d,
        h,
        w,
        [point],
        [lower.copy() for _ in range(h)],
        [upper.copy() for _ in range(h)],
        wins_lower,
        wins_upper,
    )


def _random_decimal(rng: random.Random, lo: float, hi: float) -> float:
    return _f32(float(f"{rng.uniform(lo, hi):.9f}"))


def _random_case(
    rng: random.Random, d: int, h: int, w: int, index: int
) -> dict[str, object]:
    frame_lower = []
    frame_upper = []
    window_lower = []
    window_upper = []
    for _ in range(h):
        fl = [_random_decimal(rng, -0.9, 0.2) for _ in range(d)]
        fu = [_random_decimal(rng, lo + 0.04, 0.98) for lo in fl]
        frame_lower.append(fl)
        frame_upper.append(fu)
        wins_lo = []
        wins_hi = []
        for _ in range(w):
            wl = [_random_decimal(rng, -0.98, 0.75) for _ in range(d)]
            wu = [_random_decimal(rng, lo + 0.02, 0.99) for lo in wl]
            wins_lo.append(wl)
            wins_hi.append(wu)
        window_lower.append(wins_lo)
        window_upper.append(wins_hi)
    point = [_random_decimal(rng, -1.5, 1.5) for _ in range(d)]
    return _case_record(
        f"random-d{d}-h{h}-w{w}-{index:02d}",
        "generated",
        d,
        h,
        w,
        [point],
        frame_lower,
        frame_upper,
        window_lower,
        window_upper,
    )


def build_manifest() -> dict[str, object]:
    cases = []
    # 3 nextafter probes for each capacity combination: 36 cases.
    for d, h, w in CONFIGS:
        for side in (-1, 0, 1):
            cases.append(_boundary_case(d, h, w, side))
    # 16 independent seeded cases for each of the 12 configurations: 192 cases.
    rng = random.Random(SEED)
    for d, h, w in CONFIGS:
        for index in range(16):
            cases.append(_random_case(rng, d, h, w, index))
    cases.extend(_literal_cases())
    if len(cases) != 246:
        raise AssertionError(f"expected 246 cases, built {len(cases)}")
    total_configuration_counts = {
        f"d{d}-h{h}-w{w}": sum(
            (case["d"], case["H"], case["W"]) == (d, h, w) for case in cases
        )
        for d, h, w in CONFIGS
    }
    return {
        "schema": "town03-cp2-confirmation-set-v1",
        "authorization_comment_id": 6814259456,
        "cp1_implementation_sha": "b1cd7c5dee66576c6945c0a7d10a6429a47fa092",
        "cp1_evidence_sha": "9653f628a2163273896044c9549362701bf76052",
        "seed": SEED,
        "structured_configuration_counts": {
            f"d{d}-h{h}-w{w}": 19 for d, h, w in CONFIGS
        },
        "total_configuration_counts": total_configuration_counts,
        "group_counts": {"nextafter": 36, "generated": 192, "literal": 18},
        "independent_oracle": "100-digit Decimal over stored float32 endpoints and points; direct loop-based Boolean set equations",
        "cases": cases,
    }


def main() -> None:
    output = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("docs/reconstruction/TOWN03_CP2_CONFIRMATION_SET.json")
    )
    manifest = build_manifest()
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(f"wrote {len(manifest['cases'])} frozen confirmation cases to {output}")


if __name__ == "__main__":
    main()
