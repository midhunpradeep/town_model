#!/usr/bin/env python3
"""Freeze the separately versioned TOWN-03 R1 input supplement.

This input/oracle builder imports no TOWN model or evaluator. It uses exact
Decimal arithmetic over stored float32 values and direct set inequalities.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
import json
from pathlib import Path
import sys

import torch

CONFIGS = [(d, h, w) for d in (1, 2) for h in (1, 2) for w in (0, 1, 2)]
OUTPUT = Path("docs/reconstruction/TOWN03_CP2_INDEPENDENCE_SUPPLEMENT_V1.json")

# Each expectation is hand-calculated for the corresponding one-point fixture.
LITERAL_FIXTURES = (
    {
        "id": "r1-literal-d1-h1-w0-upper-frame-face",
        "d": 1,
        "H": 1,
        "W": 0,
        "points": [[0.6875]],
        "frame_lower": [[-0.9375]],
        "frame_upper": [[0.6875]],
        "window_lower": [[]],
        "window_upper": [[]],
        "expected": {"house": [0.0], "town": [0.0], "house_member": [True], "town_member": [True]},
        "coverage": ["closed-frame-upper-face", "exact-zero-clearance"],
    },
    {
        "id": "r1-literal-d1-h1-w0-collapsed-frame",
        "d": 1,
        "H": 1,
        "W": 0,
        "points": [[0.375]],
        "frame_lower": [[0.375]],
        "frame_upper": [[0.375]],
        "window_lower": [[]],
        "window_upper": [[]],
        "expected": {"house": [0.0], "town": [0.0], "house_member": [True], "town_member": [True]},
        "coverage": ["degenerate-frame", "exact-zero-clearance"],
    },
    {
        "id": "r1-literal-d1-h1-w1-window-upper-face",
        "d": 1,
        "H": 1,
        "W": 1,
        "points": [[0.375]],
        "frame_lower": [[-0.875]],
        "frame_upper": [[0.875]],
        "window_lower": [[[-0.25]]],
        "window_upper": [[[0.375]]],
        "expected": {"house": [0.0], "town": [0.0], "house_member": [True], "town_member": [True]},
        "coverage": ["closed-window-face", "strict-window-interior-exclusion", "exact-zero-clearance"],
    },
    {
        "id": "r1-literal-d1-h1-w1-window-interior",
        "d": 1,
        "H": 1,
        "W": 1,
        "points": [[0.125]],
        "frame_lower": [[-0.875]],
        "frame_upper": [[0.875]],
        "window_lower": [[[-0.375]]],
        "window_upper": [[[0.375]]],
        "expected": {"house": [-0.25], "town": [-0.25], "house_member": [False], "town_member": [False]},
        "coverage": ["strict-window-interior", "negative-clearance"],
    },
    {
        "id": "r1-literal-d1-h1-w2-overlapping-window-interiors",
        "d": 1,
        "H": 1,
        "W": 2,
        "points": [[0.125]],
        "frame_lower": [[-0.9375]],
        "frame_upper": [[0.9375]],
        "window_lower": [[[-0.5], [-0.125]]],
        "window_upper": [[[0.25], [0.5]]],
        "expected": {"house": [-0.25], "town": [-0.25], "house_member": [False], "town_member": [False]},
        "coverage": ["overlapping-windows", "strict-window-interior", "negative-clearance"],
    },
    {
        "id": "r1-literal-d1-h2-w0-touching-house-boundary",
        "d": 1,
        "H": 2,
        "W": 0,
        "points": [[0.125]],
        "frame_lower": [[-0.875], [0.125]],
        "frame_upper": [[0.125], [0.875]],
        "window_lower": [[], []],
        "window_upper": [[], []],
        "expected": {"house": [0.0], "town": [0.0], "house_member": [True], "town_member": [True]},
        "coverage": ["two-house-touching-faces", "closed-frame-boundary", "exact-zero-clearance"],
    },
    {
        "id": "r1-literal-d1-h2-w1-second-house-boundary-rescue",
        "d": 1,
        "H": 2,
        "W": 1,
        "points": [[0.0]],
        "frame_lower": [[-0.875], [0.0]],
        "frame_upper": [[0.875], [0.75]],
        "window_lower": [[[-0.25]], [[0.5]]],
        "window_upper": [[[0.25]], [[0.625]]],
        "expected": {"house": [-0.25], "town": [0.0], "house_member": [False], "town_member": [True]},
        "coverage": ["house-window-interior", "multi-house-boundary-rescue", "closed-frame-boundary"],
    },
    {
        "id": "r1-literal-d1-h1-w2-protruding-window-and-frame",
        "d": 1,
        "H": 1,
        "W": 2,
        "points": [[0.625]],
        "frame_lower": [[-0.5]],
        "frame_upper": [[0.5]],
        "window_lower": [[[0.375], [0.75]]],
        "window_upper": [[[0.75], [0.875]]],
        "expected": {"house": [-0.125], "town": [-0.125], "house_member": [False], "town_member": [False]},
        "coverage": ["protruding-window", "outside-frame", "overlapping-min-branches"],
    },
    {
        "id": "r1-literal-d2-h1-w0-outside-domain-query",
        "d": 2,
        "H": 1,
        "W": 0,
        "points": [[1.125, 0.25]],
        "frame_lower": [[-0.875, -0.625]],
        "frame_upper": [[0.75, 0.625]],
        "window_lower": [[]],
        "window_upper": [[]],
        "expected": {"house": [-0.375], "town": [-0.375], "house_member": [False], "town_member": [False]},
        "coverage": ["finite-outside-domain-query", "outside-frame", "negative-clearance"],
    },
    {
        "id": "r1-literal-d2-h1-w1-one-axis-degenerate-window-face",
        "d": 2,
        "H": 1,
        "W": 1,
        "points": [[0.25, 0.0]],
        "frame_lower": [[-0.875, -0.75]],
        "frame_upper": [[0.875, 0.75]],
        "window_lower": [[[0.25, -0.5]]],
        "window_upper": [[[0.25, 0.5]]],
        "expected": {"house": [0.0], "town": [0.0], "house_member": [True], "town_member": [True]},
        "coverage": ["one-axis-degenerate-window", "window-face", "exact-zero-clearance"],
    },
    {
        "id": "r1-literal-d2-h1-w2-overlapping-window-interiors",
        "d": 2,
        "H": 1,
        "W": 2,
        "points": [[0.0, 0.0]],
        "frame_lower": [[-0.9375, -0.9375]],
        "frame_upper": [[0.9375, 0.9375]],
        "window_lower": [[[-0.5, -0.25], [-0.25, -0.5]]],
        "window_upper": [[[0.25, 0.5], [0.5, 0.25]]],
        "expected": {"house": [-0.25], "town": [-0.25], "house_member": [False], "town_member": [False]},
        "coverage": ["overlapping-windows", "two-dimensional-interior", "negative-clearance"],
    },
    {
        "id": "r1-literal-d2-h2-w0-two-house-union-positive",
        "d": 2,
        "H": 2,
        "W": 0,
        "points": [[0.375, 0.375]],
        "frame_lower": [[-0.875, -0.875], [0.25, 0.25]],
        "frame_upper": [[-0.25, -0.25], [0.75, 0.75]],
        "window_lower": [[], []],
        "window_upper": [[], []],
        "expected": {"house": [-0.625], "town": [0.125], "house_member": [False], "town_member": [True]},
        "coverage": ["two-house-union", "positive-town-clearance", "house-town-distinction"],
    },
    {
        "id": "r1-literal-d2-h2-w1-second-house-window-rescue",
        "d": 2,
        "H": 2,
        "W": 1,
        "points": [[0.0, 0.0]],
        "frame_lower": [[-0.875, -0.875], [-0.5, -0.5]],
        "frame_upper": [[0.875, 0.875], [0.5, 0.5]],
        "window_lower": [[[-0.25, -0.25]], [[0.125, -0.125]]],
        "window_upper": [[[0.25, 0.25]], [[0.375, 0.125]]],
        "expected": {"house": [-0.25], "town": [0.125], "house_member": [False], "town_member": [True]},
        "coverage": ["multi-house-composition", "window-interior", "second-house-rescue"],
    },
    {
        "id": "r1-literal-d2-h2-w2-degenerate-house-and-two-window-house",
        "d": 2,
        "H": 2,
        "W": 2,
        "points": [[0.25, -0.125]],
        "frame_lower": [[0.25, -0.125], [-0.875, -0.875]],
        "frame_upper": [[0.25, -0.125], [0.875, 0.875]],
        "window_lower": [
            [[0.5, 0.5], [-0.625, 0.5]],
            [[-0.5, -0.5], [-0.25, -0.25]],
        ],
        "window_upper": [
            [[0.625, 0.625], [-0.5, 0.625]],
            [[0.5, 0.5], [0.375, 0.25]],
        ],
        "expected": {"house": [0.0], "town": [0.0], "house_member": [True], "town_member": [True]},
        "coverage": ["degenerate-frame", "two-house-two-window-composition", "zero-clearance"],
    },
)


def _f32(value: float) -> float:
    return torch.tensor(value, dtype=torch.float32).item()


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def _box_score(point: list[float], lower: list[float], upper: list[float]) -> Decimal:
    values = [_decimal(x) - _decimal(lo) for x, lo in zip(point, lower, strict=True)]
    values.extend(_decimal(hi) - _decimal(x) for x, hi in zip(point, upper, strict=True))
    return min(values)


def _house_score(
    point: list[float], lower: list[float], upper: list[float],
    window_lower: list[list[float]], window_upper: list[list[float]],
) -> Decimal:
    return min([
        _box_score(point, lower, upper),
        *(-_box_score(point, lo, hi) for lo, hi in zip(window_lower, window_upper, strict=True)),
    ])


def _house_member(
    point: list[float], lower: list[float], upper: list[float],
    window_lower: list[list[float]], window_upper: list[list[float]],
) -> bool:
    in_domain = all(-1.0 <= value <= 1.0 for value in point)
    in_frame = all(lo <= value <= hi for value, lo, hi in zip(point, lower, upper, strict=True))
    in_window_interior = any(
        all(lo < value < hi for value, lo, hi in zip(point, win_lo, win_hi, strict=True))
        for win_lo, win_hi in zip(window_lower, window_upper, strict=True)
    )
    return in_domain and in_frame and not in_window_interior


def _oracle(case: dict[str, object]) -> dict[str, list[object]]:
    with localcontext() as context:
        context.prec = 100
        house_scores: list[Decimal] = []
        town_scores: list[Decimal] = []
        house_members: list[bool] = []
        town_members: list[bool] = []
        for point in case["points"]:
            scores = [
                _house_score(point, lo, hi, win_lo, win_hi)
                for lo, hi, win_lo, win_hi in zip(
                    case["frame_lower"], case["frame_upper"],
                    case["window_lower"], case["window_upper"], strict=True,
                )
            ]
            members = [
                _house_member(point, lo, hi, win_lo, win_hi)
                for lo, hi, win_lo, win_hi in zip(
                    case["frame_lower"], case["frame_upper"],
                    case["window_lower"], case["window_upper"], strict=True,
                )
            ]
            house_scores.append(scores[0])
            town_scores.append(max(scores))
            house_members.append(members[0])
            town_members.append(any(members))
        return {
            "house_clearance_decimal": [str(value) for value in house_scores],
            "town_clearance_decimal": [str(value) for value in town_scores],
            "house_clearance_f32": [torch.tensor(float(value), dtype=torch.float32).item() for value in house_scores],
            "town_clearance_f32": [torch.tensor(float(value), dtype=torch.float32).item() for value in town_scores],
            "house_membership": house_members,
            "town_membership": town_members,
        }


def _record(
    case_id: str, group: str, d: int, h: int, w: int,
    points: list[list[float]], frame_lower: list[list[float]], frame_upper: list[list[float]],
    window_lower: list[list[list[float]]], window_upper: list[list[list[float]]],
    coverage: list[str], expected: dict[str, object] | None = None,
) -> dict[str, object]:
    case: dict[str, object] = {
        "id": case_id,
        "group": group,
        "d": d,
        "H": h,
        "W": w,
        "points": [[_f32(value) for value in row] for row in points],
        "frame_lower": [[_f32(value) for value in row] for row in frame_lower],
        "frame_upper": [[_f32(value) for value in row] for row in frame_upper],
        "window_lower": [[[ _f32(value) for value in row] for row in house] for house in window_lower],
        "window_upper": [[[ _f32(value) for value in row] for row in house] for house in window_upper],
        "coverage": coverage,
    }
    oracle = _oracle(case)
    if expected is not None:
        for key, actual_key in (
            ("house", "house_clearance_decimal"),
            ("town", "town_clearance_decimal"),
        ):
            expected_decimal = [Decimal.from_float(_f32(value)) for value in expected[key]]
            if [Decimal(value) for value in oracle[actual_key]] != expected_decimal:
                raise AssertionError(f"literal {case_id} {key} expectation mismatch: {oracle[actual_key]} != {expected_decimal}")
        if oracle["house_membership"] != expected["house_member"]:
            raise AssertionError(f"literal {case_id} house membership expectation mismatch")
        if oracle["town_membership"] != expected["town_member"]:
            raise AssertionError(f"literal {case_id} town membership expectation mismatch")
        case["literal_expected"] = expected
    case.update(oracle)
    return case


def _nextafter_case(d: int, h: int, w: int, side: int) -> dict[str, object]:
    lower = [_f32(-0.71875), _f32(-0.40625)][:d]
    upper = [_f32(0.84375), _f32(0.78125)][:d]
    point = [_f32((lo + hi) / 2) for lo, hi in zip(lower, upper, strict=True)]
    if side:
        direction = torch.tensor(float("-inf") if side < 0 else float("inf"), dtype=torch.float32)
        point[0] = torch.nextafter(torch.tensor(lower[0], dtype=torch.float32), direction).item()
    else:
        point[0] = lower[0]
    frame_lower = [lower.copy()]
    frame_upper = [upper.copy()]
    if h == 2:
        frame_lower.append([_f32(0.25)] * d)
        frame_upper.append([_f32(0.75)] * d)
    win_lo_one = [_f32(0.125)] * d
    win_hi_one = [_f32(0.375)] * d
    win_lo_two = [_f32(0.5)] * d
    win_hi_two = [_f32(0.75)] * d
    window_lower = []
    window_upper = []
    for _ in range(h):
        window_lower.append([win_lo_one.copy(), win_lo_two.copy()][:w])
        window_upper.append([win_hi_one.copy(), win_hi_two.copy()][:w])
    side_name = {-1: "below", 0: "on", 1: "above"}[side]
    return _record(
        f"r1-nextafter-{side_name}-d{d}-h{h}-w{w}",
        "nextafter",
        d,
        h,
        w,
        [point],
        frame_lower,
        frame_upper,
        window_lower,
        window_upper,
        ["frame-lower-nextafter", "boundary-below-on-above"],
    )


def build_supplement() -> dict[str, object]:
    cases = [
        _nextafter_case(d, h, w, side)
        for d, h, w in CONFIGS
        for side in (-1, 0, 1)
    ]
    for fixture in LITERAL_FIXTURES:
        cases.append(
            _record(
                fixture["id"],
                "literal",
                fixture["d"],
                fixture["H"],
                fixture["W"],
                fixture["points"],
                fixture["frame_lower"],
                fixture["frame_upper"],
                fixture["window_lower"],
                fixture["window_upper"],
                fixture["coverage"],
                fixture["expected"],
            )
        )
    if len(cases) != 50 or len({case["id"] for case in cases}) != 50:
        raise AssertionError("supplement must contain exactly 50 uniquely named cases")
    counts = {
        f"d{d}-h{h}-w{w}": sum(
            (case["d"], case["H"], case["W"]) == (d, h, w) for case in cases
        )
        for d, h, w in CONFIGS
    }
    return {
        "schema": "town03-cp2-independence-supplement-v1",
        "authorization_comment_id": 6830101628,
        "review_finding_comment_id": 6815233836,
        "preserved_original_manifest_sha256": "acb1f9c43110ac839530066ef2aa3311755003321b945b65345f3445b4548f3e",
        "cp1_exploratory_results_sha256": "7e31ffeaac15b704e3b582f84e4ef1fd5903591380a54dc98895f36a626457c2",
        "construction": {
            "nextafter": "36 cases; below/on/above the first frame lower face for each d/H/W configuration, with new dyadic frame bounds",
            "literal": "14 hand-specified one-point cases with independent Decimal score and direct Boolean expectations",
            "oracle": "100-digit Decimal over exact stored float32 inputs; direct closed-frame/strict-window-interior set equations",
            "production_evaluator_called": False,
        },
        "group_counts": {"nextafter": 36, "literal": 14},
        "configuration_counts": counts,
        "cases": cases,
    }


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT
    payload = json.dumps(build_supplement(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(f"wrote 50 frozen supplement cases to {output}")


if __name__ == "__main__":
    main()
