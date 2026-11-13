#!/usr/bin/env python3
"""Run locked numerical checks on the R1 input-disjoint 246-case set.

The original CP2 artifacts and verifier remain unchanged. This runner selects
the 196 original CP2 inputs proven independent of CP1, appends the frozen
50-case supplement, rechecks provenance/independence before evaluation, then
applies the original author-approved budgets without modification.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

from town_model.models.bounded_box import box_signed_clearance

# The audited/original verifier modules provide only independent input/oracle
# helpers plus the existing production execution adapters. No clearance source
# is edited or reimplemented here.
import town03_cp2_audit_input_independence as independence
import town03_cp2_verify_confirmation_set as original

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MANIFEST = ROOT / "docs/reconstruction/TOWN03_CP2_CONFIRMATION_SET.json"
SUPPLEMENT = ROOT / "docs/reconstruction/TOWN03_CP2_INDEPENDENCE_SUPPLEMENT_V1.json"
AUDIT = ROOT / "docs/reconstruction/TOWN03_CP2_INPUT_INDEPENDENCE_AUDIT_V1.json"
OUTPUT = ROOT / "docs/reconstruction/TOWN03_CP2_INDEPENDENCE_RESULTS_V1.json"
LOCKED_SUPPLEMENT_SHA256 = "b53625dcc7eb428ef02cca5dbba48831b0c8ca9fff5182e98df558d48867eafa"
LOCKED_AUDIT_SHA256 = "4bec527d803f3c8c017c4fee6a8d3d6e533cda2a965a8d418b2877bdef78b344"
LOCKED_CP1_RESULTS_SHA256 = "7e31ffeaac15b704e3b582f84e4ef1fd5903591380a54dc98895f36a626457c2"
LOCKED_ORIGINAL_MANIFEST_SHA256 = original.LOCKED_MANIFEST_SHA256


def _preflight() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_bytes = ORIGINAL_MANIFEST.read_bytes()
    supplement_bytes = SUPPLEMENT.read_bytes()
    audit_bytes = AUDIT.read_bytes()
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    supplement_sha = hashlib.sha256(supplement_bytes).hexdigest()
    audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    if original_sha != LOCKED_ORIGINAL_MANIFEST_SHA256:
        raise RuntimeError(f"original CP2 manifest changed: {original_sha}")
    if supplement_sha != LOCKED_SUPPLEMENT_SHA256:
        raise RuntimeError(f"R1 supplement changed: {supplement_sha}")
    if audit_sha != LOCKED_AUDIT_SHA256:
        raise RuntimeError(f"pre-evaluation input audit changed: {audit_sha}")

    audit_document = json.loads(audit_bytes)
    regenerated_audit = independence.audit(SUPPLEMENT)
    if regenerated_audit != audit_document:
        raise RuntimeError("recomputed pre-evaluation independence audit differs from frozen audit artifact")
    if regenerated_audit["sha256"]["cp1_exploratory_results"] != LOCKED_CP1_RESULTS_SHA256:
        raise RuntimeError("CP1 exploratory source/results identity changed")

    original_manifest = json.loads(original_bytes)
    supplement_manifest = json.loads(supplement_bytes)
    retained_ids = set(regenerated_audit["retained_original_case_ids"])
    retained = [case for case in original_manifest["cases"] if case["id"] in retained_ids]
    if len(retained) != 196 or len(retained_ids) != 196:
        raise RuntimeError(f"expected 196 retained original cases, got {len(retained)}")
    combined = retained + supplement_manifest["cases"]
    if len(combined) != 246:
        raise RuntimeError(f"expected 246 input-disjoint cases, got {len(combined)}")

    # This canonical serialization identifies the exact confirmation inputs and
    # independent oracle records evaluated below; compute it before any model
    # evaluator is called.
    combined_payload = json.dumps(
        {
            "schema": "town03-cp2-input-independent-confirmation-v1",
            "original_manifest_sha256": original_sha,
            "supplement_sha256": supplement_sha,
            "audit_sha256": audit_sha,
            "cases": combined,
        },
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    metadata = {
        "original_manifest_sha256": original_sha,
        "supplement_sha256": supplement_sha,
        "audit_sha256": audit_sha,
        "cp1_exploratory_results_sha256": LOCKED_CP1_RESULTS_SHA256,
        "combined_input_record_sha256": hashlib.sha256(combined_payload).hexdigest(),
        "combined_case_count": len(combined),
        "group_counts": {
            "original_seeded_generated": sum(case["group"] == "generated" for case in retained),
            "original_independent_literals": sum(case["group"] == "literal" for case in retained),
            "supplement_nextafter": sum(case["group"] == "nextafter" for case in supplement_manifest["cases"]),
            "supplement_literals": sum(case["group"] == "literal" for case in supplement_manifest["cases"]),
        },
        "pre_evaluation_audit_status": regenerated_audit["status"],
        "pre_evaluation_combined_distinct_full_inputs": regenerated_audit["combined_distinct_full_inputs"],
    }
    if metadata["group_counts"] != {
        "original_seeded_generated": 192,
        "original_independent_literals": 4,
        "supplement_nextafter": 36,
        "supplement_literals": 14,
    }:
        raise RuntimeError(f"unexpected CP2 input composition: {metadata['group_counts']}")
    return combined, metadata


def _metric(actual: float, oracle: Decimal) -> dict[str, Any]:
    return original._metric(actual, oracle)


def run() -> dict[str, Any]:
    cases, frozen_metadata = _preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("approved CP2 remediation requires actual CUDA; CUDA is unavailable")

    # Evaluator execution begins only after the frozen file hashes and complete
    # full-input independence audit have passed above.
    devices = (torch.device("cpu"), torch.device("cuda:0"))
    all_metrics: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(device): {name: [] for name in ("house", "town")} for device in devices
    }
    cpu_cuda_metrics: dict[str, list[dict[str, Any]]] = {name: [] for name in ("house", "town")}
    per_case_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    boolean_mismatches = 0
    box_equivalence_calls = 0

    for case in cases:
        expected = original._oracle(case)
        for name, manifest_key, oracle_key in (
            ("house", "house_clearance_decimal", "house_decimal"),
            ("town", "town_clearance_decimal", "town_decimal"),
        ):
            if [Decimal(value) for value in case[manifest_key]] != expected[oracle_key]:
                raise RuntimeError(f"independent Decimal oracle mismatch: {case['id']} {name}")
        if (
            expected["house_membership"] != case["house_membership"]
            or expected["town_membership"] != case["town_membership"]
        ):
            raise RuntimeError(f"independent Boolean oracle mismatch: {case['id']}")

        row: dict[str, Any] = {
            "id": case["id"],
            "group": case["group"],
            "source_set": "supplement" if str(case["id"]).startswith("r1-") else "original-independent",
            "d": case["d"],
            "H": case["H"],
            "W": case["W"],
            "devices": {},
        }
        actual_by_device: dict[str, dict[str, Any]] = {}
        for device in devices:
            device_name = str(device)
            actual = original._evaluate(case, device)
            actual_by_device[device_name] = actual
            if case["H"] == 1 and case["W"] == 0:
                points, frame_lower, frame_upper, _, _ = original._tensors(case, device)
                box_values = box_signed_clearance(points, frame_lower[0], frame_upper[0])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                box_outputs = box_values.detach().cpu().tolist()
                if (
                    actual["house_clearance"] != box_outputs
                    or actual["town_clearance"] != box_outputs
                ):
                    failures.append({"id": case["id"], "device": device_name, "type": "H1_W0_BOX_equivalence"})
                box_equivalence_calls += 1
            measurements: dict[str, Any] = {}
            for name, decimal_key, membership_key in (
                ("house", "house_decimal", "house_membership"),
                ("town", "town_decimal", "town_membership"),
            ):
                values = actual[f"{name}_clearance"]
                metrics = [
                    _metric(value, oracle)
                    for value, oracle in zip(values, expected[decimal_key], strict=True)
                ]
                all_metrics[device_name][name].extend(metrics)
                if actual[f"{name}_membership"] != expected[membership_key]:
                    mismatch_count = sum(
                        got != want
                        for got, want in zip(
                            actual[f"{name}_membership"], expected[membership_key], strict=True
                        )
                    )
                    boolean_mismatches += mismatch_count
                    failures.append({"id": case["id"], "device": device_name, "type": f"{name}_boolean_mismatch", "count": mismatch_count})
                for metric in metrics:
                    oracle = Decimal(metric["oracle_decimal"])
                    if (
                        Decimal(metric["scaled_error"]) > original.ABS_RELATIVE_BUDGET
                        or int(metric["ulp_error"]) > original.ULP_BUDGET
                    ):
                        failures.append({"id": case["id"], "device": device_name, "type": f"{name}_forward_budget", "metric": metric})
                    if (
                        abs(oracle) <= original.NEAR_ZERO
                        and Decimal(metric["absolute_error"]) > original.NEAR_ZERO_ABS_BUDGET
                    ):
                        failures.append({"id": case["id"], "device": device_name, "type": f"{name}_near_zero_budget", "metric": metric})
                    if metric["sign_crossing"] or metric["zero_status_mismatch"]:
                        failures.append({"id": case["id"], "device": device_name, "type": f"{name}_sign_or_zero_crossing", "metric": metric})
                measurements[name] = metrics
            row["devices"][device_name] = {"actual": actual, "measurements": measurements}
            if device.type == "cuda":
                for name in ("house", "town"):
                    for index, (cpu_value, cuda_value) in enumerate(
                        zip(
                            actual_by_device["cpu"][f"{name}_clearance"],
                            actual[f"{name}_clearance"],
                            strict=True,
                        )
                    ):
                        diff = {
                            "absolute_difference": abs(cpu_value - cuda_value),
                            "ulp_difference": original._ulp(cpu_value, cuda_value),
                            "sign_mismatch": original._sign(cpu_value) != original._sign(cuda_value),
                            "zero_status_mismatch": (cpu_value == 0.0) != (cuda_value == 0.0),
                        }
                        cpu_cuda_metrics[name].append(diff)
                        if (
                            diff["ulp_difference"] > original.ULP_BUDGET
                            or diff["sign_mismatch"]
                            or diff["zero_status_mismatch"]
                        ):
                            failures.append({"id": case["id"], "type": f"{name}_cpu_cuda_budget", "index": index, "metric": diff})
        per_case_results.append(row)

    if boolean_mismatches:
        failures.append({"type": "boolean_membership_mismatches", "count": boolean_mismatches})
    device_summaries = {
        device: {name: original._summary(values) for name, values in by_kind.items()}
        for device, by_kind in all_metrics.items()
    }
    cpu_cuda_summaries = {
        name: {
            "outputs": len(values),
            "max_absolute_difference": max((item["absolute_difference"] for item in values), default=0.0),
            "max_ulp_difference": max((item["ulp_difference"] for item in values), default=0),
            "sign_mismatches": sum(item["sign_mismatch"] for item in values),
            "zero_status_mismatches": sum(item["zero_status_mismatch"] for item in values),
        }
        for name, values in cpu_cuda_metrics.items()
    }

    gradient_cpu = original._gradient_probes(torch.device("cpu"))
    gradient_cuda = original._gradient_probes(torch.device("cuda:0"))
    analytic_gradients = {
        "unique_frame": [[[1.0, 0.0]], [-1.0, 0.0], [0.0, 0.0]],
        "unique_window": [[[1.0, 0.0]], [0.0, 0.0], [0.0, 0.0], [[0.0, 0.0]], [[-1.0, 0.0]]],
        "unique_town_frame": [[[1.0]], [[0.0], [-1.0]], [[0.0], [0.0]]],
        "unique_town_window": [[[1.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[[0.0, 0.0]]], [[[-1.0, 0.0]]]],
        "tied_amin_faces": [[[0.0]], [-0.5], [0.5]],
        "tied_house_branches": [[[0.0]], [-0.5], [[0.5]]],
        "tied_town_amax": [[[0.0]], [[0.0], [-0.5]], [[0.5], [0.0]]],
    }
    unique_error = tied_error = cross_device_gradient_error = 0.0
    for name, wanted in analytic_gradients.items():
        expected_values = original._nested_values(wanted)
        cpu_values = original._nested_values(gradient_cpu[name]["grads"])
        cuda_values = original._nested_values(gradient_cuda[name]["grads"])
        cpu_error = max((abs(a - b) for a, b in zip(cpu_values, expected_values, strict=True)), default=0.0)
        cuda_error = max((abs(a - b) for a, b in zip(cuda_values, expected_values, strict=True)), default=0.0)
        cross_error = max((abs(a - b) for a, b in zip(cpu_values, cuda_values, strict=True)), default=0.0)
        if name.startswith("unique_"):
            unique_error = max(unique_error, cpu_error, cuda_error)
        else:
            tied_error = max(tied_error, cpu_error, cuda_error)
        cross_device_gradient_error = max(cross_device_gradient_error, cross_error)
    finite_difference_error = max(
        abs(value - wanted)
        for observations in (gradient_cpu, gradient_cuda)
        for value, wanted in zip(
            observations["finite_difference"]["finite_difference_gradients"],
            (1.0, -1.0),
            strict=True,
        )
    )
    for name, value, limit in (
        ("unique_branch_gradient", unique_error, 1e-6),
        ("tied_gradient", tied_error, 1e-6),
        ("cpu_cuda_gradient", cross_device_gradient_error, 1e-6),
        ("finite_difference", finite_difference_error, 2e-5),
    ):
        if value > limit:
            failures.append({"type": f"{name}_budget", "max_error": value, "limit": limit})
    gradient_summary = {
        "max_unique_gradient_abs_error": unique_error,
        "max_tied_gradient_abs_error": tied_error,
        "max_cpu_cuda_gradient_abs_difference": cross_device_gradient_error,
        "max_finite_difference_abs_error": finite_difference_error,
        "locked_limits": {
            "unique_gradient_abs": 1e-6,
            "tied_gradient_abs": 1e-6,
            "cpu_cuda_gradient_abs": 1e-6,
            "finite_difference_abs": 2e-5,
            "finite_difference_step_float32": 1e-3,
        },
        "cpu_observations": gradient_cpu,
        "cuda_observations": gradient_cuda,
    }
    return {
        "status": "PASS" if not failures else "FAIL",
        "classification": "R1 input-independent CP2 numerical confirmation against unchanged author-locked budgets",
        "frozen_inputs": frozen_metadata,
        "locked_limits": {
            "forward_max_ulp": original.ULP_BUDGET,
            "forward_scaled_abs_error": str(original.ABS_RELATIVE_BUDGET),
            "near_zero_threshold": str(original.NEAR_ZERO),
            "near_zero_abs_error": str(original.NEAR_ZERO_ABS_BUDGET),
            "zero_and_sign_crossings": 0,
            "cpu_cuda_max_ulp": original.ULP_BUDGET,
        },
        "device_metrics": device_summaries,
        "cpu_cuda_metrics": cpu_cuda_summaries,
        "gradient_metrics": gradient_summary,
        "boolean_mismatches": boolean_mismatches,
        "H1_W0_BOX_equivalence_calls": box_equivalence_calls,
        "environment": original._device_environment(),
        "failures": failures,
        "per_case_results": per_case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run()
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "frozen_inputs", "locked_limits", "device_metrics",
        "cpu_cuda_metrics", "gradient_metrics", "boolean_mismatches",
        "H1_W0_BOX_equivalence_calls", "environment", "failures"
    )}, indent=2, sort_keys=True))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
