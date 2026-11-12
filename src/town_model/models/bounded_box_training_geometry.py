"""Verified inverse and physical-first initialization for bounded BOX geometry.

The numerical operating envelope is intentionally narrower than the pure CP1
transform. This module does not implement losses, optimizer updates, or training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Literal, Sequence

import numpy as np
import torch
from torch import Tensor

from town_model.models.bound_box_inference import BoundBoxClassifier, ClassBinding
from town_model.models.bounded_trainable_box_geometry import (
    BoundedBoxGeometry,
    analytic_endpoint_jacobian,
    decode_bounded_box_raw,
    hosting_frame_minimum_full_width,
    ordinary_minimum_full_width,
    validate_bounded_box_geometry,
)


INITIAL_CERTIFIED_DIMENSION = 2
INITIAL_CERTIFIED_EPSILON = 1e-4
INITIAL_CERTIFIED_DELTA = 1e-4
INITIAL_CERTIFIED_RAW_MIN = -3.0
INITIAL_CERTIFIED_RAW_MAX = 3.0
ENDPOINT_NEAR_ZERO = 0.01
ENDPOINT_NEAR_ZERO_ABS_LIMIT = 2.0**-20
ENDPOINT_MAX_ULP = 128
RESERVE_RELATIVE_BUDGET = 1.0 / 32.0
INITIAL_CONDITIONING_MIN = 1e-5
INITIALIZATION_MAX_ATTEMPTS = 64
INITIALIZATION_MAX_ASPECT_RATIO = 4.0


class BoxGeometryNumericalError(ValueError):
    """Base class for explicitly classified inverse/configuration failures."""

    outcome = "box_geometry_numerical_error"

    def __init__(self, message: str, *, row_index: int | None = None) -> None:
        super().__init__(message)
        self.row_index = row_index


class InvalidPhysicalTargetError(BoxGeometryNumericalError):
    outcome = "invalid_physical_target"


class FiniteRealUnattainableError(BoxGeometryNumericalError):
    outcome = "finite_real_unattainability"


class Float32InverseRepresentabilityError(BoxGeometryNumericalError):
    outcome = "float32_inverse_representability_failure"


class BoundedBoxInverseBudgetError(BoxGeometryNumericalError):
    outcome = "reconstruction_budget_failure"


class InvalidReconstructedPhysicalGeometryError(BoxGeometryNumericalError):
    outcome = "invalid_reconstructed_physical_geometry"


class UnsupportedNumericalConfigurationError(BoxGeometryNumericalError):
    outcome = "unsupported_numerical_configuration"


@dataclass(frozen=True, slots=True)
class BoxWidthPolicy:
    """Immutable ordinary-box or hosting-frame minimum-width configuration."""

    kind: Literal["ordinary", "hosting_frame"]
    epsilon: float
    delta: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("ordinary", "hosting_frame"):
            raise ValueError("kind must be 'ordinary' or 'hosting_frame'")
        if self.kind == "ordinary":
            if self.delta is not None:
                raise ValueError("ordinary boxes do not have a hosting delta")
            ordinary_minimum_full_width(self.epsilon)
        else:
            if self.delta is None:
                raise ValueError("hosting frames require a positive delta")
            hosting_frame_minimum_full_width(self.epsilon, self.delta)

    @classmethod
    def ordinary(cls, *, epsilon: float = INITIAL_CERTIFIED_EPSILON) -> BoxWidthPolicy:
        return cls("ordinary", epsilon)

    @classmethod
    def hosting_frame(
        cls,
        *,
        epsilon: float = INITIAL_CERTIFIED_EPSILON,
        delta: float = INITIAL_CERTIFIED_DELTA,
    ) -> BoxWidthPolicy:
        return cls("hosting_frame", epsilon, delta)

    @property
    def minimum_full_width(self) -> float:
        if self.kind == "ordinary":
            return ordinary_minimum_full_width(self.epsilon)
        assert self.delta is not None
        return hosting_frame_minimum_full_width(self.epsilon, self.delta)

    @property
    def name(self) -> str:
        return self.kind

    def require_initial_certification(self, *, dimension: int) -> None:
        if dimension != INITIAL_CERTIFIED_DIMENSION:
            raise UnsupportedNumericalConfigurationError(
                f"initial BOX numerical certification supports d=2, not d={dimension}"
            )
        if self.kind == "ordinary" and self.epsilon != INITIAL_CERTIFIED_EPSILON:
            raise UnsupportedNumericalConfigurationError(
                "initial ordinary-box numerical certification requires epsilon=1e-4"
            )
        if self.kind == "hosting_frame" and (
            self.epsilon != INITIAL_CERTIFIED_EPSILON
            or self.delta != INITIAL_CERTIFIED_DELTA
        ):
            raise UnsupportedNumericalConfigurationError(
                "initial hosting-frame numerical certification requires "
                "epsilon=delta=1e-4"
            )


@dataclass(frozen=True, slots=True)
class BoxInverseRowDiagnostics:
    """Measured physical and numerical diagnostics for one target row."""

    row_index: int
    target_centers: tuple[float, ...]
    target_half_widths: tuple[float, ...]
    reconstructed_centers: tuple[float, ...]
    reconstructed_half_widths: tuple[float, ...]
    lower_endpoint_errors: tuple[float, ...]
    upper_endpoint_errors: tuple[float, ...]
    lower_endpoint_ulp: tuple[int, ...]
    upper_endpoint_ulp: tuple[int, ...]
    max_endpoint_error: float
    max_endpoint_ulp: int
    zero_status_mismatches: int
    sign_crossings: int
    exact_zero_signbit_mismatches: int
    epsilon_reserve: float
    delta_reserve: float | None
    conditioning_sigma_min: float


@dataclass(frozen=True, slots=True)
class EndpointBudgetDiagnostics:
    """Per-row endpoint errors, ULP distances, reserves, and zero crossings."""

    lower_endpoint_errors: tuple[float, ...]
    upper_endpoint_errors: tuple[float, ...]
    lower_endpoint_ulp: tuple[int, ...]
    upper_endpoint_ulp: tuple[int, ...]
    max_endpoint_error: float
    max_endpoint_ulp: int
    zero_status_mismatches: int
    sign_crossings: int
    exact_zero_signbit_mismatches: int


@dataclass(frozen=True, slots=True)
class BoundedBoxInverseResult:
    """Complete successful inverse for a batch of independent target boxes."""

    raw_width: Tensor
    raw_center: Tensor
    geometry: BoundedBoxGeometry
    row_diagnostics: tuple[BoxInverseRowDiagnostics, ...]

    @property
    def max_endpoint_error(self) -> float:
        return max((row.max_endpoint_error for row in self.row_diagnostics), default=0.0)


def _require_corner_batch(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if value.layout != torch.strided:
        raise TypeError(f"{name} must be a dense strided tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have exact shape [N, d]")
    if value.shape[1] < 1:
        raise ValueError("the point dimension d must be at least 1")
    if value.device.type not in ("cpu", "cuda"):
        raise UnsupportedNumericalConfigurationError(
            f"device {value.device.type!r} is outside the certified CPU/CUDA set"
        )
    if value.device.type == "cuda" and not torch.cuda.is_available():
        raise UnsupportedNumericalConfigurationError("CUDA is not available")
    return value


def _ordered_float32_integer(value: float) -> int:
    bits = struct.unpack("!I", struct.pack("!f", value))[0]
    if bits & 0x80000000:
        return (~bits) & 0xFFFFFFFF
    return bits | 0x80000000


def _float32_ulp_distance(left: float, right: float) -> int:
    if left == 0.0 and right == 0.0:
        return 0
    return abs(_ordered_float32_integer(left) - _ordered_float32_integer(right))


def validate_inverse_endpoint_budgets(
    target_lower: Tensor,
    target_upper: Tensor,
    reconstructed_lower: Tensor,
    reconstructed_upper: Tensor,
    *,
    policy: BoxWidthPolicy,
) -> tuple[EndpointBudgetDiagnostics, ...]:
    """Apply the locked endpoint, near-zero, and physical-reserve comparisons.

    Hard reconstructed physical invariants are checked separately before this
    numeric comparison. Exact signed zero is physically equal regardless of
    sign bit; zero-status and nonzero sign crossings are retained in the
    returned diagnostics.
    """
    if not isinstance(policy, BoxWidthPolicy):
        raise TypeError("policy must be a BoxWidthPolicy")
    tensors = (
        _require_corner_batch(target_lower, "target_lower"),
        _require_corner_batch(target_upper, "target_upper"),
        _require_corner_batch(reconstructed_lower, "reconstructed_lower"),
        _require_corner_batch(reconstructed_upper, "reconstructed_upper"),
    )
    first = tensors[0]
    if any(tensor.shape != first.shape for tensor in tensors[1:]):
        raise ValueError("all endpoint tensors must have identical [N, d] shapes")
    if any(tensor.device != first.device for tensor in tensors[1:]):
        raise ValueError("all endpoint tensors must use the same device")
    policy.require_initial_certification(dimension=first.shape[1])
    for name, tensor in zip(
        ("target_lower", "target_upper", "reconstructed_lower", "reconstructed_upper"),
        tensors,
        strict=True,
    ):
        if not bool(torch.isfinite(tensor).all().item()):
            raise BoundedBoxInverseBudgetError(
                f"{name} contains a nonfinite endpoint"
            )
    cpu_rows = tuple(tensor.detach().to(device="cpu").tolist() for tensor in tensors)
    lower_targets, upper_targets, lower_actual, upper_actual = cpu_rows
    rows: list[EndpointBudgetDiagnostics] = []
    for row_index in range(first.shape[0]):
        lower_errors = tuple(
            abs(lower_actual[row_index][axis] - lower_targets[row_index][axis])
            for axis in range(first.shape[1])
        )
        upper_errors = tuple(
            abs(upper_actual[row_index][axis] - upper_targets[row_index][axis])
            for axis in range(first.shape[1])
        )
        lower_ulps = tuple(
            _float32_ulp_distance(lower_actual[row_index][axis], lower_targets[row_index][axis])
            for axis in range(first.shape[1])
        )
        upper_ulps = tuple(
            _float32_ulp_distance(upper_actual[row_index][axis], upper_targets[row_index][axis])
            for axis in range(first.shape[1])
        )
        errors = (*lower_errors, *upper_errors)
        endpoint_error = max(errors, default=0.0)
        endpoint_ulps = max((*lower_ulps, *upper_ulps), default=0)
        epsilon_limit = RESERVE_RELATIVE_BUDGET * policy.epsilon
        delta_limit = (
            RESERVE_RELATIVE_BUDGET * policy.delta
            if policy.kind == "hosting_frame" and policy.delta is not None
            else epsilon_limit
        )
        reserve_limit = min(epsilon_limit, delta_limit)
        if endpoint_error > reserve_limit:
            raise BoundedBoxInverseBudgetError(
                f"target row {row_index} endpoint error {endpoint_error:.9g} exceeds "
                f"the reserve-relative limit {reserve_limit:.9g}",
                row_index=row_index,
            )
        for axis in range(first.shape[1]):
            for target_value, reconstructed_value, error, ulp in (
                (
                    lower_targets[row_index][axis],
                    lower_actual[row_index][axis],
                    lower_errors[axis],
                    lower_ulps[axis],
                ),
                (
                    upper_targets[row_index][axis],
                    upper_actual[row_index][axis],
                    upper_errors[axis],
                    upper_ulps[axis],
                ),
            ):
                if (
                    abs(target_value) <= ENDPOINT_NEAR_ZERO
                    or abs(reconstructed_value) <= ENDPOINT_NEAR_ZERO
                ):
                    if error > ENDPOINT_NEAR_ZERO_ABS_LIMIT:
                        raise BoundedBoxInverseBudgetError(
                            f"target row {row_index} near-zero endpoint error exceeds 2^-20",
                            row_index=row_index,
                        )
                elif ulp > ENDPOINT_MAX_ULP:
                    raise BoundedBoxInverseBudgetError(
                        f"target row {row_index} endpoint ULP distance exceeds 128",
                        row_index=row_index,
                    )
        zero_status_mismatches = 0
        sign_crossings = 0
        zero_signbit_mismatches = 0
        for axis in range(first.shape[1]):
            for target_value, reconstructed_value in (
                (lower_targets[row_index][axis], lower_actual[row_index][axis]),
                (upper_targets[row_index][axis], upper_actual[row_index][axis]),
            ):
                zero_status_mismatches += int((target_value == 0.0) != (reconstructed_value == 0.0))
                sign_crossings += int(target_value * reconstructed_value < 0.0)
                if target_value == 0.0 and reconstructed_value == 0.0:
                    target_bits = struct.unpack("!I", struct.pack("!f", target_value))[0]
                    reconstructed_bits = struct.unpack(
                        "!I", struct.pack("!f", reconstructed_value)
                    )[0]
                    zero_signbit_mismatches += int(target_bits != reconstructed_bits)
        rows.append(
            EndpointBudgetDiagnostics(
                lower_endpoint_errors=lower_errors,
                upper_endpoint_errors=upper_errors,
                lower_endpoint_ulp=lower_ulps,
                upper_endpoint_ulp=upper_ulps,
                max_endpoint_error=endpoint_error,
                max_endpoint_ulp=endpoint_ulps,
                zero_status_mismatches=zero_status_mismatches,
                sign_crossings=sign_crossings,
                exact_zero_signbit_mismatches=zero_signbit_mismatches,
            )
        )
    return tuple(rows)


def _target_physical_values(
    lower_rows: list[list[float]],
    upper_rows: list[list[float]],
    *,
    policy: BoxWidthPolicy,
) -> tuple[list[list[float]], list[list[float]]]:
    centers: list[list[float]] = []
    half_widths: list[list[float]] = []
    minimum_width = policy.minimum_full_width
    for row_index, (lower_row, upper_row) in enumerate(
        zip(lower_rows, upper_rows, strict=True)
    ):
        row_centers: list[float] = []
        row_half_widths: list[float] = []
        for axis, (lower, upper) in enumerate(zip(lower_row, upper_row, strict=True)):
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise InvalidPhysicalTargetError(
                    f"target row {row_index} axis {axis} has a nonfinite endpoint",
                    row_index=row_index,
                )
            if lower > upper:
                raise InvalidPhysicalTargetError(
                    f"target row {row_index} axis {axis} has reversed endpoints",
                    row_index=row_index,
                )
            if lower < -1.0 or upper > 1.0:
                raise InvalidPhysicalTargetError(
                    f"target row {row_index} axis {axis} lies outside [-1, 1]",
                    row_index=row_index,
                )
            full_width = upper - lower
            if full_width < minimum_width:
                raise InvalidPhysicalTargetError(
                    f"target row {row_index} axis {axis} violates its exact minimum width",
                    row_index=row_index,
                )
            half_width = full_width / 2.0
            center = (upper + lower) / 2.0
            row_centers.append(center)
            row_half_widths.append(half_width)
        centers.append(row_centers)
        half_widths.append(row_half_widths)
    return centers, half_widths


def invert_bounded_box_endpoints(
    target_lower: Tensor,
    target_upper: Tensor,
    *,
    policy: BoxWidthPolicy,
) -> BoundedBoxInverseResult:
    """Invert physical float32 target corners under the locked initial envelope.

    Inputs are independent `[N,2]` physical target corner batches. Any failing
    row rejects the whole call; no partial raw parameter batch is returned.
    Physical target checks, finite-real reachability, finite float32 raw
    representability, forward reconstruction, hard validation, endpoint
    budgets, and the `[-3,3]` operating envelope are separate gates.
    """
    if not isinstance(policy, BoxWidthPolicy):
        raise TypeError("policy must be a BoxWidthPolicy")
    lower = _require_corner_batch(target_lower, "target_lower")
    upper = _require_corner_batch(target_upper, "target_upper")
    if lower.shape != upper.shape:
        raise ValueError("target_lower and target_upper must have identical [N, d] shapes")
    if lower.device != upper.device:
        raise ValueError("target corners must use the same device")
    policy.require_initial_certification(dimension=lower.shape[1])
    if lower.shape[0] < 1:
        raise ValueError("inverse target batch must contain at least one box")

    # Copy only physical input values to host for deterministic float64 scalar
    # inverse calculations. The production decoder and validation remain float32.
    lower_rows = lower.detach().to(device="cpu").tolist()
    upper_rows = upper.detach().to(device="cpu").tolist()
    target_centers, target_half_widths = _target_physical_values(
        lower_rows, upper_rows, policy=policy
    )

    raw_width64: list[list[float]] = []
    raw_center64: list[list[float]] = []
    for row_index, (centers_row, half_row) in enumerate(
        zip(target_centers, target_half_widths, strict=True)
    ):
        width_values: list[float] = []
        center_values: list[float] = []
        for axis, (center, half_width) in enumerate(
            zip(centers_row, half_row, strict=True)
        ):
            t = (2.0 * half_width - policy.minimum_full_width) / (
                2.0 - policy.minimum_full_width
            )
            if half_width == 1.0:
                s = math.nan
            else:
                s = center / (1.0 - half_width)
            if not (0.0 < t < 1.0) or not math.isfinite(s) or not abs(s) < 1.0:
                raise FiniteRealUnattainableError(
                    f"target row {row_index} axis {axis} has no finite real inverse "
                    "(requires 0<t<1 and |s|<1)",
                    row_index=row_index,
                )
            try:
                raw_a = math.log(t) - math.log1p(-t)
                raw_b = math.atanh(s)
            except (OverflowError, ValueError) as error:
                raise Float32InverseRepresentabilityError(
                    f"target row {row_index} axis {axis} inverse calculation is not finite"
                , row_index=row_index) from error
            if not math.isfinite(raw_a) or not math.isfinite(raw_b):
                raise Float32InverseRepresentabilityError(
                    f"target row {row_index} axis {axis} inverse calculation is not finite",
                    row_index=row_index,
                )
            f32_max = torch.finfo(torch.float32).max
            if abs(raw_a) > f32_max or abs(raw_b) > f32_max:
                raise Float32InverseRepresentabilityError(
                    f"target row {row_index} axis {axis} raw inverse is outside finite float32",
                    row_index=row_index,
                )
            packed_a = struct.unpack("!f", struct.pack("!f", raw_a))[0]
            packed_b = struct.unpack("!f", struct.pack("!f", raw_b))[0]
            if not math.isfinite(packed_a) or not math.isfinite(packed_b):
                raise Float32InverseRepresentabilityError(
                    f"target row {row_index} axis {axis} raw inverse is outside finite float32",
                    row_index=row_index,
                )
            width_values.append(packed_a)
            center_values.append(packed_b)
        raw_width64.append(width_values)
        raw_center64.append(center_values)

    raw_width = torch.tensor(raw_width64, dtype=torch.float32, device=lower.device)
    raw_center = torch.tensor(raw_center64, dtype=torch.float32, device=lower.device)
    geometry = decode_bounded_box_raw(
        raw_width, raw_center, minimum_full_width=policy.minimum_full_width
    )
    try:
        validate_bounded_box_geometry(geometry)
    except (TypeError, ValueError) as batch_error:
        for row_index in range(geometry.lower.shape[0]):
            one_row = BoundedBoxGeometry(
                centers=geometry.centers[row_index : row_index + 1],
                half_widths=geometry.half_widths[row_index : row_index + 1],
                lower=geometry.lower[row_index : row_index + 1],
                upper=geometry.upper[row_index : row_index + 1],
                minimum_full_width=geometry.minimum_full_width,
            )
            try:
                validate_bounded_box_geometry(one_row)
            except (TypeError, ValueError) as row_error:
                raise InvalidReconstructedPhysicalGeometryError(
                    f"inverse reconstruction row {row_index} violates a hard physical "
                    f"invariant: {row_error}",
                    row_index=row_index,
                ) from row_error
        raise InvalidReconstructedPhysicalGeometryError(
            f"inverse reconstruction violates a hard physical invariant: {batch_error}",
            row_index=None,
        ) from batch_error

    centers_out = geometry.centers.detach().to(device="cpu").tolist()
    half_widths_out = geometry.half_widths.detach().to(device="cpu").tolist()
    budget_rows = validate_inverse_endpoint_budgets(
        lower, upper, geometry.lower, geometry.upper, policy=policy
    )
    jacobians = analytic_endpoint_jacobian(
        raw_width, raw_center, minimum_full_width=policy.minimum_full_width
    )
    singular_values = torch.linalg.svdvals(jacobians)
    conditioning_values = singular_values.min(dim=-1).values.min(dim=-1).values
    conditioning_values_cpu = conditioning_values.detach().to(device="cpu").tolist()
    row_diagnostics: list[BoxInverseRowDiagnostics] = []

    for row_index, budget in enumerate(budget_rows):
        conditioning_sigma_min = float(conditioning_values_cpu[row_index])
        row_diagnostics.append(
            BoxInverseRowDiagnostics(
                row_index=row_index,
                target_centers=tuple(target_centers[row_index]),
                target_half_widths=tuple(target_half_widths[row_index]),
                reconstructed_centers=tuple(centers_out[row_index]),
                reconstructed_half_widths=tuple(half_widths_out[row_index]),
                lower_endpoint_errors=budget.lower_endpoint_errors,
                upper_endpoint_errors=budget.upper_endpoint_errors,
                lower_endpoint_ulp=budget.lower_endpoint_ulp,
                upper_endpoint_ulp=budget.upper_endpoint_ulp,
                max_endpoint_error=budget.max_endpoint_error,
                max_endpoint_ulp=budget.max_endpoint_ulp,
                zero_status_mismatches=budget.zero_status_mismatches,
                sign_crossings=budget.sign_crossings,
                exact_zero_signbit_mismatches=budget.exact_zero_signbit_mismatches,
                epsilon_reserve=policy.epsilon,
                delta_reserve=policy.delta,
                conditioning_sigma_min=conditioning_sigma_min,
            )
        )

    unsupported = (raw_width < INITIAL_CERTIFIED_RAW_MIN) | (
        raw_width > INITIAL_CERTIFIED_RAW_MAX
    ) | (raw_center < INITIAL_CERTIFIED_RAW_MIN) | (
        raw_center > INITIAL_CERTIFIED_RAW_MAX
    )
    if bool(unsupported.any().item()):
        row_index = int(torch.nonzero(unsupported.any(dim=1), as_tuple=False)[0, 0].item())
        raise UnsupportedNumericalConfigurationError(
            f"target row {row_index} inverse raw parameters lie outside the certified "
            "raw operating range [-3,3]",
            row_index=row_index,
        )

    return BoundedBoxInverseResult(
        raw_width=raw_width,
        raw_center=raw_center,
        geometry=geometry,
        row_diagnostics=tuple(row_diagnostics),
    )


class InitializationInfeasibleError(BoxGeometryNumericalError):
    """A physical target occupancy is proven impossible before random search."""

    outcome = "proven_initialization_infeasibility"


class InitializationSearchExhaustedError(BoxGeometryNumericalError):
    """The feasible target exhausted its finite deterministic candidate budget."""

    outcome = "initialization_search_exhausted"

    def __init__(
        self,
        message: str,
        *,
        class_id: int,
        attempts: tuple[InitializationAttemptDiagnostic, ...],
        completed_class_ids: tuple[int, ...],
    ) -> None:
        super().__init__(message)
        self.class_id = class_id
        self.attempts = attempts
        self.completed_class_ids = completed_class_ids


@dataclass(frozen=True, slots=True)
class InitializationAttemptDiagnostic:
    """Audit record for one private-RNG physical candidate and its disposition."""

    class_id: int
    attempt: int
    field_seeds: tuple[tuple[str, int], ...]
    requested_occupancy: float
    sampled_center_fractions: tuple[float, float]
    target_centers: tuple[float, float]
    target_half_widths: tuple[float, float]
    target_lower: tuple[float, float]
    target_upper: tuple[float, float]
    candidate_raw_width: tuple[float, float] | None
    candidate_raw_center: tuple[float, float] | None
    actual_occupancy: float | None
    endpoint_error: float | None
    epsilon_width_reserves: tuple[float, float] | None
    delta_width_reserves: tuple[float, float] | None
    domain_reserves: tuple[float, float] | None
    aspect_ratio: float | None
    conditioning_sigma_min: float | None
    outcome: str
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class InitializationClassRecord:
    """Complete requested/actual physical provenance for one accepted class row."""

    class_id: int
    width_policy: BoxWidthPolicy
    requested_occupancy: float
    actual_occupancy: float
    reconstructed_occupancy: float
    target_centers: tuple[float, float]
    target_half_widths: tuple[float, float]
    reconstructed_centers: tuple[float, float]
    reconstructed_half_widths: tuple[float, float]
    target_lower: tuple[float, float]
    target_upper: tuple[float, float]
    reconstructed_lower: tuple[float, float]
    reconstructed_upper: tuple[float, float]
    endpoint_error: float
    endpoint_ulp: int
    epsilon_width_reserves: tuple[float, float]
    delta_width_reserves: tuple[float, float] | None
    domain_reserves: tuple[float, float]
    aspect_ratio: float
    conditioning_sigma_min: float
    accepted_attempt: int
    attempts: tuple[InitializationAttemptDiagnostic, ...]

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


@dataclass(frozen=True, slots=True)
class PhysicalBoxInitialization:
    """All-or-nothing initialized raw parameters and their physical audit trail."""

    class_ids: tuple[int, ...]
    class_manifest_sha256: str
    dimension: int
    master_seed: int
    raw_width: Tensor
    raw_center: Tensor
    row_geometries: tuple[BoundedBoxGeometry, ...]
    policies: tuple[BoxWidthPolicy, ...]
    class_records: tuple[InitializationClassRecord, ...]
    numpy_version: str
    rng_policy: str


def _initialization_seed(
    master_seed: int, class_id: int, attempt: int, field: str
) -> int:
    payload = json.dumps(
        ["town-model-box05-init-v1", master_seed, class_id, attempt, field],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rng_uniform(seed: int, low: float, high: float) -> float:
    generator = np.random.Generator(np.random.PCG64(seed))
    return float(generator.uniform(low, high))


def _preflight_occupancy(
    target_occupancy: float, policy: BoxWidthPolicy, *, class_id: int
) -> tuple[float, float]:
    if target_occupancy not in (0.01, 0.1):
        raise UnsupportedNumericalConfigurationError(
            f"class {class_id} occupancy target must be one of the approved values {{0.01, 0.1}}"
        )
    policy.require_initial_certification(dimension=INITIAL_CERTIFIED_DIMENSION)
    sigmoid_min = 1.0 / (1.0 + math.exp(-INITIAL_CERTIFIED_RAW_MIN))
    sigmoid_max = 1.0 / (1.0 + math.exp(-INITIAL_CERTIFIED_RAW_MAX))
    half_min = (
        policy.minimum_full_width
        + (2.0 - policy.minimum_full_width) * sigmoid_min
    ) / 2.0
    half_max = (
        policy.minimum_full_width
        + (2.0 - policy.minimum_full_width) * sigmoid_max
    ) / 2.0
    band_low = target_occupancy / 2.0
    band_high = min(2.0 * target_occupancy, 1.0)
    feasible_low = max(band_low, half_min * half_min)
    feasible_high = min(band_high, half_max * half_max)
    if feasible_low > feasible_high:
        raise InitializationInfeasibleError(
            f"class {class_id} occupancy band [{band_low:g}, {band_high:g}] has no "
            "intersection with the attainable 2D width-product interval",
        )
    # Equal half-widths establish mathematical feasibility for every product in
    # this intersection and satisfy the approved aspect-ratio bound.
    return feasible_low, feasible_high


def _device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise UnsupportedNumericalConfigurationError(
            f"invalid initialization device {value!r}"
        ) from error
    if device.type not in ("cpu", "cuda"):
        raise UnsupportedNumericalConfigurationError(
            f"device {device.type!r} is outside the certified CPU/CUDA set"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise UnsupportedNumericalConfigurationError("CUDA is not available")
    return device


def initialize_bounded_boxes(
    class_binding: ClassBinding,
    *,
    policies: Sequence[BoxWidthPolicy],
    occupancy_targets: Sequence[float],
    master_seed: int,
    device: str | torch.device = "cpu",
) -> PhysicalBoxInitialization:
    """Sample, invert, verify, then publish a complete deterministic 2D set.

    Each class row has independent SHA-256-derived NumPy PCG64 streams keyed
    by master seed, immutable class ID, attempt number, and sample field.
    Candidate geometry is sampled in physical coordinates without labels,
    inverted and forward-validated before any complete initialization object
    is returned. A proven-infeasible target and a finite search exhaustion are
    distinct failures; neither exposes partial parameter tensors.
    """
    if not isinstance(class_binding, ClassBinding):
        raise TypeError("class_binding must be a verified ClassBinding")
    # The public BOX-04 type performs the same verification used for inference.
    BoundBoxClassifier(class_binding)
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer (bool is not accepted)")
    device = _device(device)
    class_ids = class_binding.trainable_class_ids
    if len(class_ids) > 2:
        raise UnsupportedNumericalConfigurationError(
            "initial trainable BOX certification supports at most two class rows"
        )
    policy_rows = tuple(policies)
    target_rows = tuple(occupancy_targets)
    if len(policy_rows) != len(class_ids) or len(target_rows) != len(class_ids):
        raise ValueError("policies and occupancy_targets must match the exact class row count")
    if any(not isinstance(policy, BoxWidthPolicy) for policy in policy_rows):
        raise TypeError("every class row requires a BoxWidthPolicy")
    if any(type(target) not in (float, int) for target in target_rows):
        raise TypeError("occupancy targets must be real numeric scalars")
    target_rows = tuple(float(target) for target in target_rows)

    # Preflight the entire request before drawing or retaining any candidate.
    feasible_bands = tuple(
        _preflight_occupancy(target, policy, class_id=class_id)
        for class_id, target, policy in zip(
            class_ids, target_rows, policy_rows, strict=True
        )
    )
    if not class_ids:
        empty = torch.empty((0, INITIAL_CERTIFIED_DIMENSION), dtype=torch.float32, device=device)
        return PhysicalBoxInitialization(
            class_ids=(),
            class_manifest_sha256=class_binding.class_manifest_sha256,
            dimension=INITIAL_CERTIFIED_DIMENSION,
            master_seed=master_seed,
            raw_width=empty,
            raw_center=empty.clone(),
            row_geometries=(),
            policies=(),
            class_records=(),
            numpy_version=np.__version__,
            rng_policy="SHA256-64/NumPy-PCG64; keyed by (master_seed,class_id,attempt,field); NumPy "
            + np.__version__,
        )

    accepted_width_rows: list[Tensor] = []
    accepted_center_rows: list[Tensor] = []
    accepted_geometries: list[BoundedBoxGeometry] = []
    accepted_records: list[InitializationClassRecord] = []

    for class_id, policy, requested_q, feasible_band in zip(
        class_ids, policy_rows, target_rows, feasible_bands, strict=True
    ):
        attempt_records: list[InitializationAttemptDiagnostic] = []
        accepted = False
        for attempt in range(INITIALIZATION_MAX_ATTEMPTS):
            seed_fields = (
                ("occupancy", _initialization_seed(master_seed, class_id, attempt, "occupancy")),
                ("anisotropy", _initialization_seed(master_seed, class_id, attempt, "anisotropy")),
                ("center-0", _initialization_seed(master_seed, class_id, attempt, "center-0")),
                ("center-1", _initialization_seed(master_seed, class_id, attempt, "center-1")),
            )
            seeds = dict(seed_fields)
            # The preflight intersection is the mathematically attainable part
            # of the approved occupancy band, not an adaptive change to q.
            log_q_low = math.log(feasible_band[0])
            log_q_high = math.log(feasible_band[1])
            sampled_q = math.exp(
                _rng_uniform(seeds["occupancy"], log_q_low, log_q_high)
            )
            log_ratio = _rng_uniform(
                seeds["anisotropy"], -math.log(INITIALIZATION_MAX_ASPECT_RATIO),
                math.log(INITIALIZATION_MAX_ASPECT_RATIO),
            )
            ratio = math.exp(log_ratio)
            half_values = (
                math.sqrt(sampled_q * ratio),
                math.sqrt(sampled_q / ratio),
            )
            center_fractions = (
                _rng_uniform(seeds["center-0"], -0.5, 0.5),
                _rng_uniform(seeds["center-1"], -0.5, 0.5),
            )
            centers = tuple(
                (1.0 - half_values[axis]) * center_fractions[axis]
                for axis in range(INITIAL_CERTIFIED_DIMENSION)
            )
            lower_values = tuple(
                centers[axis] - half_values[axis]
                for axis in range(INITIAL_CERTIFIED_DIMENSION)
            )
            upper_values = tuple(
                centers[axis] + half_values[axis]
                for axis in range(INITIAL_CERTIFIED_DIMENSION)
            )
            target_lower = torch.tensor([lower_values], dtype=torch.float32, device=device)
            target_upper = torch.tensor([upper_values], dtype=torch.float32, device=device)
            actual_centers = tuple(
                (float(target_upper[0, axis].item()) + float(target_lower[0, axis].item())) / 2.0
                for axis in range(INITIAL_CERTIFIED_DIMENSION)
            )
            actual_half_widths = tuple(
                (float(target_upper[0, axis].item()) - float(target_lower[0, axis].item())) / 2.0
                for axis in range(INITIAL_CERTIFIED_DIMENSION)
            )
            actual_q = math.prod(actual_half_widths)
            aspect_ratio = max(actual_half_widths) / min(actual_half_widths)

            try:
                inverse = invert_bounded_box_endpoints(
                    target_lower, target_upper, policy=policy
                )
            except BoxGeometryNumericalError as error:
                attempt_records.append(
                    InitializationAttemptDiagnostic(
                        class_id=class_id,
                        attempt=attempt,
                        field_seeds=seed_fields,
                        requested_occupancy=requested_q,
                        sampled_center_fractions=center_fractions,
                        target_centers=actual_centers,
                        target_half_widths=actual_half_widths,
                        target_lower=tuple(float(value) for value in target_lower[0].tolist()),
                        target_upper=tuple(float(value) for value in target_upper[0].tolist()),
                        candidate_raw_width=None,
                        candidate_raw_center=None,
                        actual_occupancy=actual_q,
                        endpoint_error=None,
                        epsilon_width_reserves=None,
                        delta_width_reserves=None,
                        domain_reserves=None,
                        aspect_ratio=aspect_ratio,
                        conditioning_sigma_min=None,
                        outcome=error.outcome,
                        failure_reason=str(error),
                    )
                )
                continue

            row_diagnostic = inverse.row_diagnostics[0]
            reconstructed_centers = row_diagnostic.reconstructed_centers
            reconstructed_half_widths = row_diagnostic.reconstructed_half_widths
            reconstructed_lower = tuple(
                float(value) for value in inverse.geometry.lower[0].detach().cpu().tolist()
            )
            reconstructed_upper = tuple(
                float(value) for value in inverse.geometry.upper[0].detach().cpu().tolist()
            )
            width_reserves = tuple(
                reconstructed_upper[axis]
                - reconstructed_lower[axis]
                - policy.minimum_full_width
                for axis in range(INITIAL_CERTIFIED_DIMENSION)
            )
            epsilon_reserves = tuple(
                width_reserves[axis] / policy.epsilon
                for axis in range(INITIAL_CERTIFIED_DIMENSION)
            )
            delta_reserves = (
                tuple(width_reserves[axis] / policy.delta for axis in range(2))
                if policy.kind == "hosting_frame" and policy.delta is not None
                else None
            )
            domain_reserves = tuple(
                min(
                    reconstructed_lower[axis] + 1.0,
                    1.0 - reconstructed_upper[axis],
                )
                for axis in range(INITIAL_CERTIFIED_DIMENSION)
            )
            reconstructed_q = math.prod(reconstructed_half_widths)
            candidate_raw_width = tuple(float(value) for value in inverse.raw_width[0].tolist())
            candidate_raw_center = tuple(float(value) for value in inverse.raw_center[0].tolist())
            conditioning = row_diagnostic.conditioning_sigma_min
            failure_reason: str | None = None
            outcome = "accepted"
            if not requested_q / 2.0 <= actual_q <= min(requested_q * 2.0, 1.0):
                outcome = "occupancy_band_failure"
                failure_reason = "actual float32 target occupancy lies outside the approved band"
            elif aspect_ratio > INITIALIZATION_MAX_ASPECT_RATIO:
                outcome = "aspect_ratio_failure"
                failure_reason = "actual half-width aspect ratio exceeds 4"
            elif conditioning < INITIAL_CONDITIONING_MIN:
                outcome = "conditioning_gate_failure"
                failure_reason = "local raw-to-endpoint sigma_min is below 1e-5"

            attempt_records.append(
                InitializationAttemptDiagnostic(
                    class_id=class_id,
                    attempt=attempt,
                    field_seeds=seed_fields,
                    requested_occupancy=requested_q,
                    sampled_center_fractions=center_fractions,
                    target_centers=actual_centers,
                    target_half_widths=actual_half_widths,
                    target_lower=tuple(float(value) for value in target_lower[0].tolist()),
                    target_upper=tuple(float(value) for value in target_upper[0].tolist()),
                    candidate_raw_width=candidate_raw_width,
                    candidate_raw_center=candidate_raw_center,
                    actual_occupancy=actual_q,
                    endpoint_error=row_diagnostic.max_endpoint_error,
                    epsilon_width_reserves=epsilon_reserves,
                    delta_width_reserves=delta_reserves,
                    domain_reserves=domain_reserves,
                    aspect_ratio=aspect_ratio,
                    conditioning_sigma_min=conditioning,
                    outcome=outcome,
                    failure_reason=failure_reason,
                )
            )
            if failure_reason is not None:
                continue

            accepted_width_rows.append(inverse.raw_width)
            accepted_center_rows.append(inverse.raw_center)
            accepted_geometries.append(inverse.geometry)
            accepted_records.append(
                InitializationClassRecord(
                    class_id=class_id,
                    width_policy=policy,
                    requested_occupancy=requested_q,
                    actual_occupancy=actual_q,
                    reconstructed_occupancy=reconstructed_q,
                    target_centers=actual_centers,
                    target_half_widths=actual_half_widths,
                    reconstructed_centers=reconstructed_centers,
                    reconstructed_half_widths=reconstructed_half_widths,
                    target_lower=tuple(float(value) for value in target_lower[0].tolist()),
                    target_upper=tuple(float(value) for value in target_upper[0].tolist()),
                    reconstructed_lower=reconstructed_lower,
                    reconstructed_upper=reconstructed_upper,
                    endpoint_error=row_diagnostic.max_endpoint_error,
                    endpoint_ulp=row_diagnostic.max_endpoint_ulp,
                    epsilon_width_reserves=epsilon_reserves,
                    delta_width_reserves=delta_reserves,
                    domain_reserves=domain_reserves,
                    aspect_ratio=aspect_ratio,
                    conditioning_sigma_min=conditioning,
                    accepted_attempt=attempt,
                    attempts=tuple(attempt_records),
                )
            )
            accepted = True
            break

        if not accepted:
            raise InitializationSearchExhaustedError(
                f"class {class_id} exhausted {INITIALIZATION_MAX_ATTEMPTS} deterministic "
                "physical initialization attempts despite a feasible occupancy band",
                class_id=class_id,
                attempts=tuple(attempt_records),
                completed_class_ids=tuple(record.class_id for record in accepted_records),
            )

    return PhysicalBoxInitialization(
        class_ids=class_ids,
        class_manifest_sha256=class_binding.class_manifest_sha256,
        dimension=INITIAL_CERTIFIED_DIMENSION,
        master_seed=master_seed,
        raw_width=torch.cat(accepted_width_rows, dim=0),
        raw_center=torch.cat(accepted_center_rows, dim=0),
        row_geometries=tuple(accepted_geometries),
        policies=policy_rows,
        class_records=tuple(accepted_records),
        numpy_version=np.__version__,
        rng_policy="SHA256-64/NumPy-PCG64; keyed by (master_seed,class_id,attempt,field); NumPy "
        + np.__version__,
    )
