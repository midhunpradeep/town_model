#!/usr/bin/env python3
"""Byte-audit the TOWN-03 CP2 supplement against the CP1 input set.

This is a pre-evaluation audit. It loads stored input values and reconstructs
CP1's fixed literal inputs from the CP1 probe's fixture-data factory; it never
calls a clearance or Boolean evaluator. Equality compares every float32 input
bit plus (d,H,W,N) and tensor boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import struct
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CP1_RESULTS = ROOT / "docs/reconstruction/TOWN03_CP1_EXPLORATORY_RESULTS.json"
CP1_PROBE = ROOT / "tools/town03_cp1_exploratory_probe.py"
ORIGINAL_CP2 = ROOT / "docs/reconstruction/TOWN03_CP2_CONFIRMATION_SET.json"
SUPPLEMENT = ROOT / "docs/reconstruction/TOWN03_CP2_INDEPENDENCE_SUPPLEMENT_V1.json"
INPUT_KEYS = ("points", "frame_lower", "frame_upper", "window_lower", "window_upper")
CONFIGS = {(d, h, w) for d in (1, 2) for h in (1, 2) for w in (0, 1, 2)}
REQUIRED_LITERAL_COVERAGE = {
    "closed-frame-upper-face",
    "degenerate-frame",
    "closed-window-face",
    "strict-window-interior",
    "overlapping-windows",
    "protruding-window",
    "outside-frame",
    "finite-outside-domain-query",
    "one-axis-degenerate-window",
    "two-house-union",
    "multi-house-boundary-rescue",
}


def _float32_bytes(value: float) -> bytes:
    try:
        return struct.pack(">f", float(value))
    except (OverflowError, struct.error) as error:
        raise ValueError(f"input is not representable as finite float32: {value!r}") from error


def _flatten(values: Any) -> tuple[bytes, ...]:
    if isinstance(values, list):
        return tuple(part for item in values for part in _flatten(item))
    return (_float32_bytes(values),)


def _signature(case: dict[str, Any], values: dict[str, Any] | None = None) -> tuple[Any, ...]:
    inputs = case if values is None else values
    d, h, w = int(case["d"]), int(case["H"]), int(case["W"])
    points = inputs["points"]
    if not isinstance(points, list) or not points:
        raise ValueError(f"{case.get('id', '<unknown>')}: expected a nonempty point batch")
    n = len(points)
    expected_shapes = {
        "points": (n, d),
        "frame_lower": (h, d),
        "frame_upper": (h, d),
        "window_lower": (h, w, d),
        "window_upper": (h, w, d),
    }
    # Tensor shapes are implied by (d,H,W,N), including W=0, whose JSON arrays
    # have no innermost dimension to inspect. Check every nonempty axis too.
    if any(len(row) != d for row in points):
        raise ValueError(f"{case.get('id', '<unknown>')}: malformed point shape")
    for key in ("frame_lower", "frame_upper"):
        if len(inputs[key]) != h or any(len(row) != d for row in inputs[key]):
            raise ValueError(f"{case.get('id', '<unknown>')}: malformed {key} shape")
    for key in ("window_lower", "window_upper"):
        if len(inputs[key]) != h or any(len(house) != w for house in inputs[key]):
            raise ValueError(f"{case.get('id', '<unknown>')}: malformed {key} shape")
        if w and any(len(window) != d for house in inputs[key] for window in house):
            raise ValueError(f"{case.get('id', '<unknown>')}: malformed {key} coordinate shape")
    fields = tuple(
        (expected_shapes[key], _flatten(inputs[key])) for key in INPUT_KEYS
    )
    return d, h, w, n, fields


def _cp1_cases() -> list[dict[str, Any]]:
    results = json.loads(CP1_RESULTS.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for row in results["per_case_results"]:
        cases.append(
            {
                "id": row["name"],
                "d": row["d"],
                "H": row["H"],
                "W": row["W"],
                **row["stored_float32_inputs"],
            }
        )

    # CP1's exact-fixture JSON stores scores but not geometry. Reconstruct those
    # inputs from the original CP1 source's pure GeometryCase fixture factory.
    # run_path defines/imports functions only; this audit calls _exact_fixtures,
    # whose body constructs data and does not call any model evaluator.
    namespace = runpy.run_path(str(CP1_PROBE))
    for fixture in namespace["_exact_fixtures"]():
        cases.append(
            {
                "id": fixture.name,
                "d": fixture.dimensions,
                "H": fixture.houses,
                "W": fixture.windows,
                "points": fixture.points,
                "frame_lower": fixture.frame_lower,
                "frame_upper": fixture.frame_upper,
                "window_lower": fixture.window_lower,
                "window_upper": fixture.window_upper,
            }
        )
    if len(results["per_case_results"]) != 128 or len(cases) != 146:
        raise ValueError(f"unexpected CP1 input inventory size: {len(cases)}")
    return cases


def _index(cases: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[str]]:
    indexed: dict[tuple[Any, ...], list[str]] = {}
    for case in cases:
        indexed.setdefault(_signature(case), []).append(str(case["id"]))
    return indexed


def audit(supplement_path: Path = SUPPLEMENT) -> dict[str, Any]:
    cp1_payload = CP1_RESULTS.read_bytes()
    original_payload = ORIGINAL_CP2.read_bytes()
    supplement_payload = supplement_path.read_bytes()
    cp1 = _cp1_cases()
    original = json.loads(original_payload)["cases"]
    supplement_manifest = json.loads(supplement_payload)
    supplement = supplement_manifest["cases"]

    if len(original) != 246:
        raise ValueError(f"original CP2 manifest has {len(original)} cases, expected 246")
    if len(supplement) != 50:
        raise ValueError(f"supplement has {len(supplement)} cases, expected 50")
    if supplement_manifest.get("group_counts") != {"literal": 14, "nextafter": 36}:
        raise ValueError("supplement group counts are not 36 nextafter + 14 literal")
    if len({case["id"] for case in supplement}) != 50:
        raise ValueError("supplement case IDs are not unique")

    cp1_index = _index(cp1)
    original_index = _index(original)
    supplement_index = _index(supplement)
    cp1_signatures = set(cp1_index)
    original_signatures = set(original_index)
    supplement_signatures = set(supplement_index)

    original_overlap = [case for case in original if _signature(case) in cp1_signatures]
    original_independent = [case for case in original if _signature(case) not in cp1_signatures]
    if len(original_overlap) != 50:
        raise ValueError(f"expected the reviewed 50 original overlaps, found {len(original_overlap)}")
    overlap_groups = {
        group: sum(case["group"] == group for case in original_overlap)
        for group in ("generated", "literal", "nextafter")
    }
    if overlap_groups != {"generated": 0, "literal": 14, "nextafter": 36}:
        raise ValueError(f"original CP1 overlap categories changed: {overlap_groups}")
    if len(original_independent) != 196:
        raise ValueError(f"expected 196 retained independent original CP2 cases, got {len(original_independent)}")
    if sum(case["group"] == "generated" for case in original_independent) != 192:
        raise ValueError("the retained CP2 set does not contain all 192 generated cases")
    if sum(case["group"] == "literal" for case in original_independent) != 4:
        raise ValueError("the retained CP2 set does not contain exactly four independent literals")

    cp1_overlaps = sorted(
        (case["id"], sorted(cp1_index[_signature(case)]))
        for case in supplement
        if _signature(case) in cp1_signatures
    )
    original_kept_overlaps = sorted(
        (case["id"], sorted(original_index[_signature(case)]))
        for case in supplement
        if _signature(case) in {_signature(row) for row in original_independent}
    )
    if cp1_overlaps:
        raise ValueError(f"supplement overlaps CP1 full inputs: {cp1_overlaps}")
    if original_kept_overlaps:
        raise ValueError(f"supplement overlaps retained CP2 inputs: {original_kept_overlaps}")
    if len(supplement_signatures) != 50:
        raise ValueError("supplement contains duplicate full-input rows")

    combined = original_independent + supplement
    combined_signatures = [_signature(case) for case in combined]
    if len(combined) != 246 or len(set(combined_signatures)) != 246:
        raise ValueError("the retained original cases plus supplement are not 246 distinct full inputs")
    all_configs = {
        f"d{d}-h{h}-w{w}": sum(
            (case["d"], case["H"], case["W"]) == (d, h, w) for case in combined
        )
        for d, h, w in sorted(CONFIGS)
    }
    if not all(all_configs.values()):
        raise ValueError(f"combined set misses approved configuration(s): {all_configs}")

    supplement_by_group = {
        group: [case for case in supplement if case["group"] == group]
        for group in ("nextafter", "literal")
    }
    nextafter_counts = {
        f"d{d}-h{h}-w{w}": sum(
            (case["d"], case["H"], case["W"]) == (d, h, w)
            for case in supplement_by_group["nextafter"]
        )
        for d, h, w in sorted(CONFIGS)
    }
    if nextafter_counts != {key: 3 for key in nextafter_counts}:
        raise ValueError(f"nextafter probes do not preserve all 12 below/on/above configurations: {nextafter_counts}")
    literal_coverage = sorted(
        {tag for case in supplement_by_group["literal"] for tag in case.get("coverage", [])}
    )
    missing_coverage = sorted(REQUIRED_LITERAL_COVERAGE - set(literal_coverage))
    if missing_coverage:
        raise ValueError(f"supplement literals omit required boundary/geometry coverage: {missing_coverage}")

    return {
        "status": "PASS",
        "audit": "exact full-input float32 bit disjointness; no production evaluator called",
        "sha256": {
            "cp1_exploratory_results": hashlib.sha256(cp1_payload).hexdigest(),
            "original_cp2_manifest": hashlib.sha256(original_payload).hexdigest(),
            "supplement": hashlib.sha256(supplement_payload).hexdigest(),
        },
        "cp1_input_case_count": len(cp1),
        "original_cp2_case_count": len(original),
        "original_cp1_overlap": {
            "total": len(original_overlap),
            "by_group": overlap_groups,
        },
        "retained_original_cp2_independent_count": len(original_independent),
        "retained_original_groups": {"generated": 192, "literal": 4},
        "supplement_case_count": len(supplement),
        "supplement_groups": {"nextafter": 36, "literal": 14},
        "supplement_overlap_with_cp1": 0,
        "supplement_overlap_with_retained_original_cp2": 0,
        "supplement_unique_full_inputs": len(supplement_signatures),
        "combined_case_count": len(combined),
        "combined_distinct_full_inputs": len(set(combined_signatures)),
        "combined_configuration_counts": all_configs,
        "supplement_nextafter_configuration_counts": nextafter_counts,
        "supplement_literal_coverage": literal_coverage,
        "retained_original_case_ids": [case["id"] for case in original_independent],
        "supplement_case_ids": [case["id"] for case in supplement],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement", type=Path, default=SUPPLEMENT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.supplement)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
