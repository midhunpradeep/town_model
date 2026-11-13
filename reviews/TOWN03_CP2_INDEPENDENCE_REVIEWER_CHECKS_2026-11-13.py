#!/usr/bin/env python3
"""Independent full-input and numerical replay for TOWN-03 CP2 R1.

This review-only script does not import the implementation agent's audit or
numerical-verifier modules. It compares exact binary32 input tensors, derives
clearance with Decimal arithmetic, derives Boolean membership directly, and
executes the reviewed production functions on CPU and CUDA.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import runpy
import struct
from typing import Any

import torch

from town_model.models.bounded_box import box_signed_clearance
from town_model.models.town_geometry import (
    house_contains,
    house_signed_clearance,
    town_contains,
    town_signed_clearance,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/reconstruction"
CP1_RESULTS = DOCS / "TOWN03_CP1_EXPLORATORY_RESULTS.json"
CP1_PROBE = ROOT / "tools/town03_cp1_exploratory_probe.py"
CP2_MANIFEST = DOCS / "TOWN03_CP2_CONFIRMATION_SET.json"
SUPPLEMENT = DOCS / "TOWN03_CP2_INDEPENDENCE_SUPPLEMENT_V1.json"
CP2_RESULTS = DOCS / "TOWN03_CP2_CONFIRMATION_RESULTS.json"
CP2_REPORT = DOCS / "TOWN03_CP2_CONFIRMATION_REPORT.md"
CP1_REPORT = DOCS / "TOWN03_CP1_EXPLORATORY_NUMERICAL_REPORT.md"
SUPPLEMENT_AUDIT = DOCS / "TOWN03_CP2_INPUT_INDEPENDENCE_AUDIT_V1.json"
R1_RESULTS = DOCS / "TOWN03_CP2_INDEPENDENCE_RESULTS_V1.json"
RETIREMENT = DOCS / "T3_TEST_RETIREMENT_MANIFEST.md"
EXPECTED_HASHES = {
    "TOWN03_CP1_EXPLORATORY_RESULTS.json": "7e31ffeaac15b704e3b582f84e4ef1fd5903591380a54dc98895f36a626457c2",
    "TOWN03_CP1_EXPLORATORY_NUMERICAL_REPORT.md": "cad18fa6f42b4cd04ec09511fb9c83f1b4e52c08ea66f70edd0f647c501a9999",
    "TOWN03_CP2_CONFIRMATION_SET.json": "acb1f9c43110ac839530066ef2aa3311755003321b945b65345f3445b4548f3e",
    "TOWN03_CP2_CONFIRMATION_RESULTS.json": "2adc4a00ae12fc26a40306bf5dbe05a121813a93d38d8b25f438f959c7cd1188",
    "TOWN03_CP2_CONFIRMATION_REPORT.md": "1aa500326b7417d7de82e148f12d895ceef06d59f7f333c92fa38a465b22dc37",
    "TOWN03_CP2_INDEPENDENCE_SUPPLEMENT_V1.json": "b53625dcc7eb428ef02cca5dbba48831b0c8ca9fff5182e98df558d48867eafa",
    "TOWN03_CP2_INPUT_INDEPENDENCE_AUDIT_V1.json": "4bec527d803f3c8c017c4fee6a8d3d6e533cda2a965a8d418b2877bdef78b344",
    "TOWN03_CP2_INDEPENDENCE_RESULTS_V1.json": "fa77c9079c3c8a6bdcc53dab6049a820a7643e2a1420313c0728d8d72cf64ac9",
    "T3_TEST_RETIREMENT_MANIFEST.md": "b3b283a4b7849cd5dd5d08b9aa9ecf66888f8435954479a6f3a7823a0802ded3",
}

INPUT_KEYS = ("points", "frame_lower", "frame_upper", "window_lower", "window_upper")
EXPECTED_CONFIGS = {(d, h, w) for d in (1, 2) for h in (1, 2) for w in (0, 1, 2)}
REQUIRED_LITERAL_TAGS = {
    "closed-frame-upper-face", "degenerate-frame", "closed-window-face",
    "strict-window-interior", "overlapping-windows", "protruding-window",
    "outside-frame", "finite-outside-domain-query", "one-axis-degenerate-window",
    "two-house-union", "multi-house-boundary-rescue",
}


def f32(value: Any) -> float:
    return struct.unpack(">f", struct.pack(">f", float(value)))[0]


def f32_bytes(value: Any) -> bytes:
    return struct.pack(">f", float(value))


def flatten(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(part for item in value for part in flatten(item))
    return (value,)


def fixture_mapping(fixture: Any) -> dict[str, Any]:
    return {
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


def cp1_inventory() -> list[dict[str, Any]]:
    result = json.loads(CP1_RESULTS.read_text(encoding="utf-8"))
    rows = []
    for case in result["per_case_results"]:
        rows.append({
            "id": case["name"], "d": case["d"], "H": case["H"], "W": case["W"],
            **case["stored_float32_inputs"],
        })
    # The original CP1 factory contains the 18 fixed input fixtures. Calling
    # only this data constructor does not execute a clearance or Boolean model.
    namespace = runpy.run_path(str(CP1_PROBE))
    rows.extend(fixture_mapping(x) for x in namespace["_exact_fixtures"]())
    if len(result["per_case_results"]) != 128 or len(rows) != 146:
        raise AssertionError(f"unexpected CP1 inventory: {len(rows)}")
    return rows


def input_signature(case: dict[str, Any]) -> tuple[Any, ...]:
    d, h, w = int(case["d"]), int(case["H"]), int(case["W"])
    points = case["points"]
    if not isinstance(points, list) or not points:
        raise AssertionError(f"{case.get('id')}: invalid/empty point batch")
    n = len(points)
    shapes = {
        "points": (n, d),
        "frame_lower": (h, d),
        "frame_upper": (h, d),
        "window_lower": (h, w, d),
        "window_upper": (h, w, d),
    }
    if any(len(row) != d for row in points):
        raise AssertionError(f"{case.get('id')}: malformed point shape")
    for key in ("frame_lower", "frame_upper"):
        values = case[key]
        if len(values) != h or any(len(row) != d for row in values):
            raise AssertionError(f"{case.get('id')}: malformed {key}")
    for key in ("window_lower", "window_upper"):
        values = case[key]
        if len(values) != h or any(len(house) != w for house in values):
            raise AssertionError(f"{case.get('id')}: malformed {key}")
        if w and any(len(win) != d for house in values for win in house):
            raise AssertionError(f"{case.get('id')}: malformed {key} coordinate axis")
    fields = []
    for key in INPUT_KEYS:
        bits = tuple(f32_bytes(value) for value in flatten(case[key]))
        fields.append((key, shapes[key], bits))
    return d, h, w, n, tuple(fields)


def input_audit() -> dict[str, Any]:
    cp1 = cp1_inventory()
    original = json.loads(CP2_MANIFEST.read_text(encoding="utf-8"))["cases"]
    supplement_doc = json.loads(SUPPLEMENT.read_text(encoding="utf-8"))
    supplement = supplement_doc["cases"]
    if len(original) != 246 or len(supplement) != 50:
        raise AssertionError("wrong frozen CP2/supplement case count")
    if Counter(row["group"] for row in supplement) != {"nextafter": 36, "literal": 14}:
        raise AssertionError("supplement groups are not 36 nextafter + 14 literal")

    cp1_sig = [input_signature(row) for row in cp1]
    original_sig = [input_signature(row) for row in original]
    supplement_sig = [input_signature(row) for row in supplement]
    cp1_set = set(cp1_sig)
    cp1_duplicates = len(cp1_sig) - len(cp1_set)
    original_duplicates = len(original_sig) - len(set(original_sig))
    if len(set(supplement_sig)) != 50:
        raise AssertionError("supplement has duplicate complete tensors")

    overlap_by_group = Counter(
        row["group"] for row, sig in zip(original, original_sig, strict=True) if sig in cp1_set
    )
    if overlap_by_group != {"nextafter": 36, "literal": 14}:
        raise AssertionError(f"original CP2/CP1 overlap unexpected: {overlap_by_group}")
    retained = [row for row, sig in zip(original, original_sig, strict=True) if sig not in cp1_set]
    retained_sig = [input_signature(row) for row in retained]
    if len(retained) != 196 or len(set(retained_sig)) != 196 or Counter(row["group"] for row in retained) != {"generated": 192, "literal": 4}:
        raise AssertionError("retained original CP2 inventory is not 192 generated + 4 literals")

    supplement_cp1_overlap = sum(sig in cp1_set for sig in supplement_sig)
    retained_set = set(retained_sig)
    supplement_retained_overlap = sum(sig in retained_set for sig in supplement_sig)
    if supplement_cp1_overlap or supplement_retained_overlap:
        raise AssertionError("supplement is not disjoint from CP1 and retained CP2")
    combined = retained + supplement
    combined_sig = retained_sig + supplement_sig
    if len(combined) != 246 or len(set(combined_sig)) != 246:
        raise AssertionError("combined confirmation inputs are not 246 distinct full tensors")
    if set(combined_sig) & cp1_set:
        raise AssertionError("combined confirmation set intersects CP1")

    config_counts = Counter((int(c["d"]), int(c["H"]), int(c["W"])) for c in combined)
    if set(config_counts) != EXPECTED_CONFIGS:
        raise AssertionError(f"missing approved d/H/W configuration: {config_counts}")
    nextafter = [c for c in supplement if c["group"] == "nextafter"]
    next_counts = Counter((c["d"], c["H"], c["W"]) for c in nextafter)
    if len(nextafter) != 36 or set(next_counts.values()) != {3} or set(next_counts) != EXPECTED_CONFIGS:
        raise AssertionError("nextafter coverage is not below/on/above for all 12 configurations")
    for case in nextafter:
        boundary = f32(case["frame_lower"][0][0])
        point = f32(case["points"][0][0])
        side = case["id"].split("-")[2]
        if side == "below":
            expected = torch.nextafter(torch.tensor(boundary, dtype=torch.float32), torch.tensor(float("-inf"), dtype=torch.float32)).item()
            if f32_bytes(point) != f32_bytes(expected) or not point < boundary:
                raise AssertionError(f"{case['id']}: not the immediate below-face float32 probe")
        elif side == "on":
            if f32_bytes(point) != f32_bytes(boundary):
                raise AssertionError(f"{case['id']}: not on the frame face")
        elif side == "above":
            expected = torch.nextafter(torch.tensor(boundary, dtype=torch.float32), torch.tensor(float("inf"), dtype=torch.float32)).item()
            if f32_bytes(point) != f32_bytes(expected) or not point > boundary:
                raise AssertionError(f"{case['id']}: not the immediate above-face float32 probe")
        else:
            raise AssertionError(f"unrecognized nextafter case id: {case['id']}")

    literals = [c for c in supplement if c["group"] == "literal"]
    coverage = {tag for case in literals for tag in case.get("coverage", [])}
    missing_tags = REQUIRED_LITERAL_TAGS - coverage
    if len(literals) != 14 or missing_tags:
        raise AssertionError(f"literal coverage incomplete: missing {sorted(missing_tags)}")
    # Re-derive literal scores/membership; don't trust the stored expected fields.
    for case in literals:
        actual = exact_oracle(case)
        expected = case["literal_expected"]
        if actual["house"] != [Decimal(str(f32(x))) for x in expected["house"]]:
            raise AssertionError(f"{case['id']}: house literal score mismatch")
        if actual["town"] != [Decimal(str(f32(x))) for x in expected["town"]]:
            raise AssertionError(f"{case['id']}: town literal score mismatch")
        if actual["house_membership"] != expected["house_member"] or actual["town_membership"] != expected["town_member"]:
            raise AssertionError(f"{case['id']}: literal Boolean expectation mismatch")
        for tag in case.get("coverage", []):
            if not tag_is_demonstrated(tag, case):
                raise AssertionError(f"{case['id']}: literal coverage tag {tag!r} not demonstrated by inputs")

    return {
        "cp1_input_rows": len(cp1),
        "cp1_unique_full_inputs": len(cp1_set),
        "cp1_duplicate_input_rows": cp1_duplicates,
        "original_cp2_cases": len(original),
        "original_cp2_duplicate_input_rows": original_duplicates,
        "original_overlap_with_cp1": dict(sorted(overlap_by_group.items())),
        "retained_original_count": len(retained),
        "retained_original_groups": dict(sorted(Counter(c["group"] for c in retained).items())),
        "supplement_count": len(supplement),
        "supplement_groups": dict(sorted(Counter(c["group"] for c in supplement).items())),
        "supplement_full_input_overlap_cp1": supplement_cp1_overlap,
        "supplement_full_input_overlap_retained_original": supplement_retained_overlap,
        "combined_count": len(combined),
        "combined_unique_full_inputs": len(set(combined_sig)),
        "combined_overlap_cp1": len(set(combined_sig) & cp1_set),
        "configuration_counts": {f"d{d}-h{h}-w{w}": config_counts[(d,h,w)] for d,h,w in sorted(config_counts)},
        "nextafter_configuration_counts": {f"d{d}-h{h}-w{w}": next_counts[(d,h,w)] for d,h,w in sorted(next_counts)},
        "literal_coverage_tags": sorted(coverage),
        "retained_cases": retained,
        "supplement_cases": supplement,
    }


def dec(value: Any) -> Decimal:
    return Decimal.from_float(f32(value))


def box_oracle(point: list[float], lower: list[float], upper: list[float]) -> Decimal:
    values = [dec(x) - dec(lo) for x, lo in zip(point, lower, strict=True)]
    values.extend(dec(hi) - dec(x) for x, hi in zip(point, upper, strict=True))
    return min(values)


def house_oracle(point: list[float], lower: list[float], upper: list[float], win_lo: list[list[float]], win_hi: list[list[float]]) -> Decimal:
    return min([box_oracle(point, lower, upper)] + [-box_oracle(point, lo, hi) for lo, hi in zip(win_lo, win_hi, strict=True)])


def bool_house(point: list[float], lower: list[float], upper: list[float], win_lo: list[list[float]], win_hi: list[list[float]]) -> bool:
    in_domain = all(-1.0 <= f32(x) <= 1.0 for x in point)
    in_frame = all(f32(lo) <= f32(x) <= f32(hi) for x, lo, hi in zip(point, lower, upper, strict=True))
    in_window = any(all(f32(lo) < f32(x) < f32(hi) for x, lo, hi in zip(point, wl, wh, strict=True)) for wl, wh in zip(win_lo, win_hi, strict=True))
    return in_domain and in_frame and not in_window


def exact_oracle(case: dict[str, Any]) -> dict[str, list[Any]]:
    with localcontext() as context:
        context.prec = 250
        hs, ts, hm, tm = [], [], [], []
        for point in case["points"]:
            scores = [house_oracle(point, fl, fu, wl, wu) for fl, fu, wl, wu in zip(case["frame_lower"], case["frame_upper"], case["window_lower"], case["window_upper"], strict=True)]
            members = [bool_house(point, fl, fu, wl, wu) for fl, fu, wl, wu in zip(case["frame_lower"], case["frame_upper"], case["window_lower"], case["window_upper"], strict=True)]
            hs.append(scores[0]); ts.append(max(scores)); hm.append(members[0]); tm.append(any(members))
        return {"house": hs, "town": ts, "house_membership": hm, "town_membership": tm}


def per_house_memberships(case: dict[str, Any], point: list[float]) -> list[bool]:
    return [
        bool_house(point, fl, fu, wl, wu)
        for fl, fu, wl, wu in zip(case["frame_lower"], case["frame_upper"], case["window_lower"], case["window_upper"], strict=True)
    ]


def tag_is_demonstrated(tag: str, case: dict[str, Any]) -> bool:
    points = case["points"]
    frames_lo, frames_hi = case["frame_lower"], case["frame_upper"]
    windows_lo, windows_hi = case["window_lower"], case["window_upper"]
    d, h, w = int(case["d"]), int(case["H"]), int(case["W"])
    if tag == "closed-frame-upper-face":
        return any(any(f32(p[j]) == f32(frames_hi[i][j]) for j in range(d)) for p in points for i in range(h))
    if tag == "closed-frame-boundary":
        return any(any(f32(p[j]) in (f32(frames_lo[i][j]), f32(frames_hi[i][j])) for j in range(d)) for p in points for i in range(h))
    if tag == "degenerate-frame":
        return any(any(f32(lo[j]) == f32(hi[j]) for j in range(d)) for lo, hi in zip(frames_lo, frames_hi, strict=True))
    if tag in {"closed-window-face", "window-face"}:
        return any(
            f32(p[j]) in (f32(windows_lo[i][k][j]), f32(windows_hi[i][k][j]))
            for p in points for i in range(h) for k in range(w) for j in range(d)
        )
    if tag in {"strict-window-interior", "house-window-interior", "window-interior"}:
        return any(
            all(f32(windows_lo[i][k][j]) < f32(p[j]) < f32(windows_hi[i][k][j]) for j in range(d))
            for p in points for i in range(h) for k in range(w)
        )
    if tag == "strict-window-interior-exclusion":
        return any(
            f32(p[j]) in (f32(windows_lo[i][k][j]), f32(windows_hi[i][k][j]))
            and bool_house(p, frames_lo[i], frames_hi[i], windows_lo[i], windows_hi[i])
            for p in points for i in range(h) for k in range(w) for j in range(d)
        )
    if tag in {"overlapping-windows", "overlapping-min-branches"}:
        return any(
            all(max(f32(windows_lo[i][a][j]), f32(windows_lo[i][b][j])) <= min(f32(windows_hi[i][a][j]), f32(windows_hi[i][b][j])) for j in range(d))
            for i in range(h) for a in range(w) for b in range(a + 1, w)
        )
    if tag == "protruding-window":
        return any(
            f32(windows_lo[i][k][j]) < f32(frames_lo[i][j]) or f32(windows_hi[i][k][j]) > f32(frames_hi[i][j])
            for i in range(h) for k in range(w) for j in range(d)
        )
    if tag == "outside-frame":
        return any(
            any(f32(p[j]) < f32(frames_lo[i][j]) or f32(p[j]) > f32(frames_hi[i][j]) for j in range(d))
            for p in points for i in range(h)
        )
    if tag == "finite-outside-domain-query":
        return any(any(abs(f32(x)) > 1.0 for x in p) and all(torch.isfinite(torch.tensor(f32(x))) for x in p) for p in points)
    if tag == "one-axis-degenerate-window":
        return any(f32(windows_lo[i][k][j]) == f32(windows_hi[i][k][j]) for i in range(h) for k in range(w) for j in range(d))
    if tag in {"two-house-union", "multi-house-boundary-rescue", "second-house-rescue"}:
        return h >= 2 and any(
            (not membership[0]) and any(membership[1:])
            for p in points for membership in [per_house_memberships(case, p)]
        )
    if tag in {"exact-zero-clearance", "zero-clearance", "negative-clearance", "positive-town-clearance", "house-town-distinction"}:
        values = [exact_oracle(case)]
        if tag == "exact-zero-clearance" or tag == "zero-clearance":
            return any(value == 0 for key in ("house", "town") for value in values[0][key])
        if tag == "negative-clearance":
            return any(value < 0 for key in ("house", "town") for value in values[0][key])
        if tag == "positive-town-clearance":
            return any(value > 0 for value in values[0]["town"])
        return values[0]["house"] != values[0]["town"]
    if tag == "two-house-touching-faces":
        return h >= 2 and any(f32(frames_hi[i][j]) == f32(frames_lo[k][j]) for i in range(h) for k in range(i + 1, h) for j in range(d))
    if tag == "two-house-two-window-composition":
        return h >= 2 and w >= 2
    if tag == "multi-house-composition":
        return h >= 2
    if tag == "two-dimensional-interior":
        return d == 2 and any(all(f32(windows_lo[i][k][j]) < f32(points[0][j]) < f32(windows_hi[i][k][j]) for j in range(d)) for i in range(h) for k in range(w))
    return False


def tensor_inputs(case: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    d, h, w = int(case["d"]), int(case["H"]), int(case["W"])
    return (
        torch.tensor(case["points"], dtype=torch.float32, device=device).reshape(-1,d),
        torch.tensor(case["frame_lower"], dtype=torch.float32, device=device).reshape(h,d),
        torch.tensor(case["frame_upper"], dtype=torch.float32, device=device).reshape(h,d),
        torch.tensor(case["window_lower"], dtype=torch.float32, device=device).reshape(h,w,d),
        torch.tensor(case["window_upper"], dtype=torch.float32, device=device).reshape(h,w,d),
    )


def ordered_bits(value: float) -> int:
    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    return (0x80000000 - (bits & 0x7fffffff)) if bits & 0x80000000 else (bits | 0x80000000)


def ulp(a: float, b: float) -> int:
    return abs(ordered_bits(a) - ordered_bits(b))


def sign(value: Decimal | float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def numerical_replay(audit: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("actual CUDA is required by the approved CP2 plan")
    cases = audit["retained_cases"] + audit["supplement_cases"]
    devices = (torch.device("cpu"), torch.device("cuda:0"))
    summary: dict[str, Any] = {}
    evaluated: dict[str, dict[str, list[float]]] = {str(d): {"house": [], "town": []} for d in devices}
    membership_mismatches = 0
    box_calls = 0
    for device in devices:
        name = str(device)
        maxima = {kind: {"max_abs": Decimal(0), "max_scaled": Decimal(0), "max_ulp": 0, "near_zero_count": 0, "near_zero_max_abs": Decimal(0), "sign_crossings": 0, "zero_mismatches": 0, "outputs": 0} for kind in ("house", "town")}
        for case in cases:
            oracle = exact_oracle(case)
            points, fl, fu, wl, wu = tensor_inputs(case, device)
            actual_house = house_signed_clearance(points, fl[0], fu[0], wl[0], wu[0])
            actual_town = town_signed_clearance(points, fl, fu, wl, wu)
            bits_house = house_contains(points, fl[0], fu[0], wl[0], wu[0])
            bits_town = town_contains(points, fl, fu, wl, wu)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            actuals = {"house": actual_house.detach().cpu().tolist(), "town": actual_town.detach().cpu().tolist()}
            bools = {"house": bits_house.detach().cpu().tolist(), "town": bits_town.detach().cpu().tolist()}
            for kind in ("house", "town"):
                if bools[kind] != oracle[f"{kind}_membership"]:
                    membership_mismatches += sum(a != b for a,b in zip(bools[kind], oracle[f"{kind}_membership"], strict=True))
                for got, exact in zip(actuals[kind], oracle[kind], strict=True):
                    with localcontext() as context:
                        context.prec = 250
                        actual_decimal = Decimal.from_float(float(got))
                        error = abs(actual_decimal - exact)
                        scaled = error / max(Decimal(1), abs(exact))
                        rounded = torch.tensor(float(exact), dtype=torch.float32).item()
                        metric = maxima[kind]
                        metric["outputs"] += 1
                        metric["max_abs"] = max(metric["max_abs"], error)
                        metric["max_scaled"] = max(metric["max_scaled"], scaled)
                        metric["max_ulp"] = max(metric["max_ulp"], ulp(got, rounded))
                        if abs(exact) <= Decimal("1e-6"):
                            metric["near_zero_count"] += 1
                            metric["near_zero_max_abs"] = max(metric["near_zero_max_abs"], error)
                        metric["sign_crossings"] += sign(got) != sign(exact)
                        metric["zero_mismatches"] += (got == 0.0) != (exact == 0)
                        if scaled > Decimal("1e-7") or ulp(got, rounded) > 1:
                            raise AssertionError(f"{name} {case['id']} {kind}: forward budget failed")
                        if abs(exact) <= Decimal("1e-6") and error > Decimal("1e-7"):
                            raise AssertionError(f"{name} {case['id']} {kind}: near-zero budget failed")
                        if sign(got) != sign(exact) or (got == 0.0) != (exact == 0):
                            raise AssertionError(f"{name} {case['id']} {kind}: sign/zero mismatch")
                    evaluated[name][kind].append(float(got))
            if int(case["H"]) == 1 and int(case["W"]) == 0:
                box = box_signed_clearance(points, fl[0], fu[0])
                if device.type == "cuda": torch.cuda.synchronize(device)
                if not torch.equal(actual_house, box) or not torch.equal(actual_town, box):
                    raise AssertionError(f"{name} {case['id']}: H=1/W=0 != BOX")
                box_calls += 1
        summary[name] = maxima
    if membership_mismatches:
        raise AssertionError(f"direct Boolean mismatches: {membership_mismatches}")
    cross = {}
    for kind in ("house", "town"):
        pairs = list(zip(evaluated["cpu"][kind], evaluated["cuda:0"][kind], strict=True))
        cross[kind] = {"outputs": len(pairs), "max_abs": max((abs(a-b) for a,b in pairs), default=0.0), "max_ulp": max((ulp(a,b) for a,b in pairs), default=0), "sign_mismatches": sum(sign(a)!=sign(b) for a,b in pairs), "zero_mismatches": sum((a==0.0)!=(b==0.0) for a,b in pairs)}
        if cross[kind]["max_ulp"] > 1 or cross[kind]["sign_mismatches"] or cross[kind]["zero_mismatches"]:
            raise AssertionError(f"CPU/CUDA {kind} limit failed")
    for device_name, kinds in summary.items():
        for kind, m in kinds.items():
            if m["max_scaled"] > Decimal("1e-7") or m["max_ulp"] > 1 or m["near_zero_max_abs"] > Decimal("1e-7") or m["sign_crossings"] or m["zero_mismatches"]:
                raise AssertionError(f"{device_name} {kind}: approved budget failed")
            for key in ("max_abs", "max_scaled", "near_zero_max_abs"):
                m[key] = str(m[key])
    return {"device_metrics": summary, "cpu_cuda": cross, "boolean_mismatches": membership_mismatches, "H1_W0_BOX_equivalence_calls": box_calls}


def flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, list):
        return [x for part in value for x in flatten_numbers(part)]
    return [float(value)]


def gradient_replay() -> dict[str, Any]:
    if not torch.cuda.is_available(): raise RuntimeError("actual CUDA is required")
    expected = {
        "unique_frame": (0.25, [[[1.0,0.0]], [-1.0,0.0], [0.0,0.0]]),
        "unique_window": (-0.1, [[[1.0,0.0]], [0.0,0.0], [0.0,0.0], [[0.0,0.0]], [[-1.0,0.0]]]),
        "unique_town_frame": (0.25, [[[1.0]], [[0.0],[-1.0]], [[0.0],[0.0]]]),
        "unique_town_window": (-0.1, [[[1.0,0.0]], [[0.0,0.0]], [[0.0,0.0]], [[[0.0,0.0]]], [[[-1.0,0.0]]]]),
        "tied_amin_faces": (1.0, [[[0.0]], [-0.5], [0.5]]),
        "tied_house_branches": (0.0, [[[0.0]], [-0.5], [[0.5]]]),
        "tied_town_amax": (0.0, [[[0.0]], [[0.0],[-0.5]], [[0.5],[0.0]]]),
    }
    all_results: dict[str, Any] = {}
    def t(value: Any, device: torch.device, grad: bool = False) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float32, device=device, requires_grad=grad)
    for device in (torch.device("cpu"), torch.device("cuda:0")):
        rows = {}
        # Unique frame branch.
        p=t([[-.25,0]],device,True); lo=t([-.5,-1],device,True); hi=t([1,1],device,True); e=torch.empty((0,2),dtype=torch.float32,device=device)
        cases = {
            "unique_frame": lambda: (house_signed_clearance(p,lo,hi,e,e).sum(), (p,lo,hi)),
        }
        # Build each probe separately so each graph has only the requested leaves.
        score, leaves = cases["unique_frame"](); rows["unique_frame"]=(score,torch.autograd.grad(score,leaves))
        p=t([[.4,0]],device,True); fl=t([-1,-1],device,True); fu=t([1,1],device,True); wl=t([[-.5,-.5]],device,True); wu=t([[.5,.5]],device,True)
        score=house_signed_clearance(p,fl,fu,wl,wu).sum(); rows["unique_window"]=(score,torch.autograd.grad(score,(p,fl,fu,wl,wu)))
        p=t([[.25]],device,True); fl=t([[-1.],[0.]],device,True); fu=t([[0.],[1.]],device,True); ew=torch.empty((2,0,1),dtype=torch.float32,device=device)
        score=town_signed_clearance(p,fl,fu,ew,ew).sum(); rows["unique_town_frame"]=(score,torch.autograd.grad(score,(p,fl,fu)))
        p=t([[.4,0]],device,True); fl=t([[-1.,-1.]],device,True); fu=t([[1.,1.]],device,True); wl=t([[[-.5,-.5]]],device,True); wu=t([[[.5,.5]]],device,True)
        score=town_signed_clearance(p,fl,fu,wl,wu).sum(); rows["unique_town_window"]=(score,torch.autograd.grad(score,(p,fl,fu,wl,wu)))
        p=t([[0.]],device,True); lo=t([-1.],device,True); hi=t([1.],device,True); e=torch.empty((0,1),dtype=torch.float32,device=device)
        score=house_signed_clearance(p,lo,hi,e,e).sum(); rows["tied_amin_faces"]=(score,torch.autograd.grad(score,(p,lo,hi)))
        p=t([[0.]],device,True); fl=t([0.],device,True); fu=t([1.],device); wl=t([[0.]],device,True); wu=t([[.5]],device)
        score=house_signed_clearance(p,fl,fu,wl,wu).sum(); rows["tied_house_branches"]=(score,torch.autograd.grad(score,(p,fl,wl)))
        p=t([[0.]],device,True); fl=t([[-1.],[0.]],device,True); fu=t([[0.],[1.]],device,True); ew=torch.empty((2,0,1),dtype=torch.float32,device=device)
        score=town_signed_clearance(p,fl,fu,ew,ew).sum(); rows["tied_town_amax"]=(score,torch.autograd.grad(score,(p,fl,fu)))
        if device.type == "cuda": torch.cuda.synchronize(device)
        flattened = {}
        for name,(score,grads) in rows.items():
            actual = [part for g in grads for part in flatten_numbers(g.detach().cpu().tolist())]
            wanted = [part for group in expected[name][1] for part in flatten_numbers(group)]
            err=max((abs(a-b) for a,b in zip(actual,wanted,strict=True)),default=0.0)
            if abs(score.item()-expected[name][0])>1e-6 or err>1e-6:
                raise AssertionError(f"{device} {name}: analytic/tied gradient mismatch")
            flattened[name]={"score":score.item(),"gradients":actual,"max_expected_abs_error":err}
        # Central float32 finite differences at the authorized step.
        px=t(-.25,device,True); lx=t(-.5,device,True); upper=t([1.,1.],device); empty=torch.empty((0,2),dtype=torch.float32,device=device); step=t(1e-3,device)
        def fn(x: torch.Tensor, lower: torch.Tensor) -> torch.Tensor:
            pts=torch.stack((x,t(0.,device))).reshape(1,2); low=torch.stack((lower,t(-1.,device)))
            return house_signed_clearance(pts,low,upper,empty,empty).sum()
        base=fn(px,lx); analytic=torch.autograd.grad(base,(px,lx))
        fd_p=(fn(px.detach()+step,lx.detach())-fn(px.detach()-step,lx.detach()))/(2*step)
        fd_l=(fn(px.detach(),lx.detach()+step)-fn(px.detach(),lx.detach()-step))/(2*step)
        finite=[fd_p.item(),fd_l.item()]
        fd_error=max(abs(finite[0]-1.0),abs(finite[1]+1.0))
        if fd_error>2e-5: raise AssertionError(f"{device}: finite-difference budget failed")
        flattened["finite_difference"]={"step":step.item(),"analytic": [x.item() for x in analytic],"finite_difference":finite,"max_error":fd_error}
        all_results[str(device)]=flattened
    max_gradient_error=max(v["max_expected_abs_error"] for device in all_results.values() for v in device.values() if "max_expected_abs_error" in v)
    max_fd=max(v["max_error"] for device in all_results.values() for k,v in device.items() if k=="finite_difference")
    max_cross=0.0
    for name in expected:
        a=all_results["cpu"][name]["gradients"]; b=all_results["cuda:0"][name]["gradients"]
        max_cross=max(max_cross,max((abs(x-y) for x,y in zip(a,b,strict=True)),default=0.0))
    if max_cross>1e-6: raise AssertionError("CPU/CUDA gradient discrepancy exceeds 1e-6")
    return {"max_unique_or_tied_expected_abs_error":max_gradient_error,"max_cpu_cuda_gradient_abs_difference":max_cross,"max_finite_difference_abs_error":max_fd,"limits":{"unique":1e-6,"tied":1e-6,"cpu_cuda":1e-6,"finite_difference":2e-5,"step":1e-3},"probes":all_results}


def main() -> None:
    audit=input_audit()
    numerical=numerical_replay(audit)
    gradients=gradient_replay()
    hashes={}
    for path in (CP1_RESULTS,CP1_REPORT,CP2_MANIFEST,CP2_RESULTS,CP2_REPORT,SUPPLEMENT,SUPPLEMENT_AUDIT,R1_RESULTS,RETIREMENT):
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    if hashes != EXPECTED_HASHES:
        raise AssertionError(f"frozen review artifact hash mismatch: {hashes}")
    environment={"python":__import__("platform").python_version(),"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"cuda_device":torch.cuda.get_device_name(0),"cuda_capability":list(torch.cuda.get_device_capability(0))}
    print(json.dumps({"status":"PASS","input_audit":{k:v for k,v in audit.items() if k not in ("retained_cases","supplement_cases")},"numerical":numerical,"gradients":gradients,"environment":environment,"sha256":hashes},indent=2,sort_keys=True,default=str))

if __name__ == "__main__":
    main()
