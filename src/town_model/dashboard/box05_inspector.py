"""Fixed, read-only BOX-05 trainable-geometry Mathematical Inspector panel."""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np
import pandas as pd
import torch

from town_model.models.bound_box_inference import (
    ClassBinding,
)
from town_model.models.bounded_box_training_geometry import BoxWidthPolicy
from town_model.models.trainable_bounded_box import TrainableBoundedBoxModule


BOX05_SYNTHETIC_VOCABULARY_BYTES = (
    b'{"classes":[{"class_id":12,"trainable":true},'
    b'{"class_id":27,"trainable":true}],"trainable_class_ids":[12,27]}\n'
)
BOX05_INSPECTOR_SEED = 20261112


@dataclass(frozen=True, slots=True)
class Box05InspectionReport:
    physical_rows: pd.DataFrame
    membership: pd.DataFrame
    clearance: pd.DataFrame
    membership_mismatches: tuple[tuple[str, str, bool, bool], ...]
    clearance_mismatches: tuple[tuple[str, str, float, float], ...]
    reconstruction_budget_mismatches: tuple[tuple[int, str, float], ...]
    expected_domain_membership: tuple[bool, ...]
    actual_domain_membership: tuple[bool, ...]
    point_labels: tuple[str, ...]


def _ulp_distance(left: float, right: float) -> int:
    if left == 0.0 and right == 0.0:
        return 0

    def ordered(value: float) -> int:
        bits = struct.unpack("!I", struct.pack("!f", value))[0]
        return (~bits) & 0xFFFFFFFF if bits & 0x80000000 else bits | 0x80000000

    return abs(ordered(left) - ordered(right))


def _endpoint_budget_failures(
    record,
    policy: BoxWidthPolicy,
) -> tuple[tuple[str, float], ...]:
    failures: list[tuple[str, float]] = []
    reserve_limit = min(
        policy.epsilon / 32.0,
        policy.delta / 32.0 if policy.delta is not None else policy.epsilon / 32.0,
    )
    endpoints = (
        ("lower", record.target_lower, record.reconstructed_lower),
        ("upper", record.target_upper, record.reconstructed_upper),
    )
    for endpoint_name, target_row, actual_row in endpoints:
        for axis, (target, actual) in enumerate(zip(target_row, actual_row, strict=True)):
            error = abs(actual - target)
            if error > reserve_limit:
                failures.append((f"{endpoint_name}[{axis}] reserve", error))
            if abs(target) <= 0.01 or abs(actual) <= 0.01:
                if error > 2.0**-20:
                    failures.append((f"{endpoint_name}[{axis}] near-zero", error))
            elif _ulp_distance(target, actual) > 128:
                failures.append((f"{endpoint_name}[{axis}] ULP", error))
    return tuple(failures)


def _independent_clearance(point: tuple[float, float], lower, upper) -> float:
    faces = []
    for axis in range(2):
        faces.append(float(np.float32(np.float32(point[axis]) - np.float32(lower[axis]))))
        faces.append(float(np.float32(np.float32(upper[axis]) - np.float32(point[axis]))))
    return min(faces)


def _inspection_report() -> Box05InspectionReport:
    binding = ClassBinding.from_synthetic_vocabulary_bytes(
        BOX05_SYNTHETIC_VOCABULARY_BYTES
    )
    policies = (BoxWidthPolicy.ordinary(), BoxWidthPolicy.hosting_frame())
    module = TrainableBoundedBoxModule.initialize(
        binding,
        policies=policies,
        occupancy_targets=(0.01, 0.1),
        master_seed=BOX05_INSPECTOR_SEED,
        device="cpu",
    )
    physical = module.decode_physical_geometry()
    initialization_rows = module.initialization_records
    raw_width = module.raw_width.detach().cpu().tolist()
    raw_center = module.raw_center.detach().cpu().tolist()
    lower_actual = physical.lower.detach().cpu().tolist()
    upper_actual = physical.upper.detach().cpu().tolist()
    centers_actual = physical.centers.detach().cpu().tolist()
    halves_actual = physical.half_widths.detach().cpu().tolist()

    points: list[tuple[float, float]] = []
    point_labels: list[str] = []
    for index, record in enumerate(initialization_rows):
        points.append(record.target_centers)
        point_labels.append(f"{record.width_policy.name} center · class {record.class_id}")
        outside = (
            record.target_centers[0] + 1.5 * record.target_half_widths[0],
            record.target_centers[1],
        )
        points.append(outside)
        point_labels.append(f"outside {record.width_policy.name} · class {record.class_id}")
    points.append((1.0, 1.0))
    point_labels.append("domain corner")
    point_tensor = torch.tensor(points, dtype=torch.float32, device="cpu")
    inference = module(point_tensor)
    actual_membership = inference.contains.tolist()
    actual_clearance = inference.signed_coordinate_clearance.tolist()
    actual_domain = tuple(bool(value) for value in inference.domain_contains.tolist())

    expected_membership: list[list[bool]] = []
    expected_clearance: list[list[float]] = []
    expected_domain: list[bool] = []
    for point in points:
        in_domain = all(-1.0 <= coordinate <= 1.0 for coordinate in point)
        expected_domain.append(in_domain)
        row_membership: list[bool] = []
        row_clearance: list[float] = []
        for lower, upper in zip(lower_actual, upper_actual, strict=True):
            row_membership.append(
                in_domain
                and all(lower[axis] <= point[axis] <= upper[axis] for axis in range(2))
            )
            row_clearance.append(_independent_clearance(point, lower, upper))
        expected_membership.append(row_membership)
        expected_clearance.append(row_clearance)

    row_labels = tuple(
        f"{policy.name} · class {class_id}"
        for class_id, policy in zip(module.row_class_ids, module.policies, strict=True)
    )
    membership_frame = pd.DataFrame(
        actual_membership, index=point_labels, columns=row_labels
    )
    clearance_frame = pd.DataFrame(
        actual_clearance, index=point_labels, columns=row_labels
    )
    membership_mismatches = tuple(
        (point_labels[row], row_labels[column], expected, bool(actual_membership[row][column]))
        for row in range(len(points))
        for column in range(len(row_labels))
        for expected in (expected_membership[row][column],)
        if expected != bool(actual_membership[row][column])
    )
    clearance_mismatches = tuple(
        (
            point_labels[row],
            row_labels[column],
            expected_clearance[row][column],
            float(actual_clearance[row][column]),
        )
        for row in range(len(points))
        for column in range(len(row_labels))
        if expected_clearance[row][column] != float(actual_clearance[row][column])
    )
    reconstruction_budget_mismatches = tuple(
        (record.class_id, field, error)
        for record, policy in zip(initialization_rows, module.policies, strict=True)
        for field, error in _endpoint_budget_failures(record, policy)
    )

    physical_rows: list[dict[str, object]] = []
    for index, (record, policy) in enumerate(
        zip(initialization_rows, module.policies, strict=True)
    ):
        width_reserves = tuple(
            upper_actual[index][axis]
            - lower_actual[index][axis]
            - policy.minimum_full_width
            for axis in range(2)
        )
        domain_reserves = tuple(
            min(lower_actual[index][axis] + 1.0, 1.0 - upper_actual[index][axis])
            for axis in range(2)
        )
        physical_rows.append(
            {
                "Class ID": record.class_id,
                "Width policy": policy.name,
                "epsilon": policy.epsilon,
                "delta": policy.delta,
                "minimum full width": policy.minimum_full_width,
                "raw width a": tuple(raw_width[index]),
                "raw center b": tuple(raw_center[index]),
                "target centers": record.target_centers,
                "decoded centers": tuple(centers_actual[index]),
                "target half-widths": record.target_half_widths,
                "decoded half-widths": tuple(halves_actual[index]),
                "target lower": record.target_lower,
                "target upper": record.target_upper,
                "decoded lower": tuple(lower_actual[index]),
                "decoded upper": tuple(upper_actual[index]),
                "endpoint E_inf": record.endpoint_error,
                "endpoint max ULP": record.endpoint_ulp,
                "epsilon reserve ratio": record.epsilon_width_reserves,
                "delta reserve ratio": record.delta_width_reserves,
                "width reserves": width_reserves,
                "domain reserves": domain_reserves,
                "requested occupancy": record.requested_occupancy,
                "physical occupancy": physical.occupancy[index],
                "log occupancy": physical.log_occupancy[index],
                "sigma_min(J)": physical.conditioning_sigma_min[index],
                "attempts": record.attempt_count,
                "seed": module.initialization.master_seed,
            }
        )

    return Box05InspectionReport(
        physical_rows=pd.DataFrame(physical_rows),
        membership=membership_frame,
        clearance=clearance_frame,
        membership_mismatches=membership_mismatches,
        clearance_mismatches=clearance_mismatches,
        reconstruction_budget_mismatches=reconstruction_budget_mismatches,
        expected_domain_membership=tuple(expected_domain),
        actual_domain_membership=actual_domain,
        point_labels=tuple(point_labels),
    )


def _render_box05_inspector(st) -> None:
    report = _inspection_report()
    st.markdown("#### BOX-05 · Trainable bounded geometry (read-only)")
    st.caption(
        "Deterministic synthetic CPU fixture · ordered class IDs (12, 27) · vocabulary "
        "identity verified · ordinary boxes and hosting-frame width policy · physical-first "
        "seeded initialization · no editable parameters, optimizer, or training."
    )
    st.markdown("Raw parameters, physical corners, occupancy, reserves, and conditioning")
    st.dataframe(report.physical_rows, hide_index=True, width="stretch")
    st.markdown("BOX-04 class-bound membership (independent physical-target oracle)")
    st.dataframe(report.membership, width="stretch")
    st.markdown("BOX-04 signed coordinate clearance")
    st.dataframe(report.clearance, width="stretch")
    total_membership = report.membership.size
    total_clearance = report.clearance.size
    if report.membership_mismatches:
        st.error(
            f"BOX-05 membership mismatches: {len(report.membership_mismatches)} of "
            f"{total_membership} · {report.membership_mismatches}"
        )
    else:
        st.success(f"BOX-05 membership mismatches: 0 of {total_membership}")
    if report.clearance_mismatches:
        st.error(
            f"BOX-05 clearance mismatches: {len(report.clearance_mismatches)} of "
            f"{total_clearance} · {report.clearance_mismatches}"
        )
    else:
        st.success(f"BOX-05 clearance mismatches: 0 of {total_clearance}")
    if report.reconstruction_budget_mismatches:
        st.error(
            "BOX-05 endpoint reconstruction-budget mismatches: "
            f"{len(report.reconstruction_budget_mismatches)}"
        )
    else:
        st.success("BOX-05 endpoint reconstruction-budget mismatches: 0")
    domain_mismatches = sum(
        expected != actual
        for expected, actual in zip(
            report.expected_domain_membership,
            report.actual_domain_membership,
            strict=True,
        )
    )
    if domain_mismatches:
        st.error(f"BOX-05 DomainThing mismatches: {domain_mismatches} of {len(report.point_labels)}")
    else:
        st.success(f"BOX-05 DomainThing mismatches: 0 of {len(report.point_labels)}")
