from __future__ import annotations

import math

import numpy as np
import pytest
import town_model.models.bounded_box_training_geometry as box_training_geometry
import torch

from town_model.models.bound_box_inference import ClassBinding
from town_model.models.bounded_box_training_geometry import (
    BoundedBoxInverseBudgetError,
    FiniteRealUnattainableError,
    InvalidPhysicalTargetError,
    InitializationSearchExhaustedError,
    UnsupportedNumericalConfigurationError,
    BoxWidthPolicy,
    initialize_bounded_boxes,
    invert_bounded_box_endpoints,
    validate_inverse_endpoint_budgets,
)
from town_model.dashboard import box05_inspector
from town_model.models.bound_box_inference import BoundBoxClassifier
from town_model.models.trainable_bounded_box import (
    InvalidTrainablePhysicalGeometryError,
    TrainableBoundedBoxModule,
)
from town_model.models.bounded_trainable_box_geometry import (
    analytic_endpoint_jacobian,
    decode_bounded_box_raw,
    validate_bounded_box_geometry,
)


def _ordinary_policy() -> BoxWidthPolicy:
    return BoxWidthPolicy.ordinary(epsilon=1e-4)


def test_inverse_reconstructs_independent_physical_target_with_locked_budgets() -> None:
    target_lower = torch.tensor([[-0.2, -0.1]], dtype=torch.float32)
    target_upper = torch.tensor([[0.2, 0.1]], dtype=torch.float32)

    result = invert_bounded_box_endpoints(
        target_lower, target_upper, policy=_ordinary_policy()
    )

    assert result.raw_width.shape == (1, 2)
    assert result.raw_center.shape == (1, 2)
    assert torch.isfinite(result.raw_width).all()
    assert torch.isfinite(result.raw_center).all()
    assert torch.all(result.raw_width >= -3.0) and torch.all(result.raw_width <= 3.0)
    assert torch.all(result.raw_center >= -3.0) and torch.all(result.raw_center <= 3.0)
    validate_bounded_box_geometry(result.geometry)
    assert result.max_endpoint_error <= 1e-4 / 32.0
    assert result.row_diagnostics[0].conditioning_sigma_min >= 1e-5
    assert result.geometry.lower.device == target_lower.device


def test_inverse_endpoint_budget_is_strict_and_records_zero_and_sign_diagnostics() -> None:
    policy = _ordinary_policy()
    lower = torch.tensor([[-0.2, -0.1]], dtype=torch.float32)
    upper = torch.tensor([[0.2, 0.1]], dtype=torch.float32)
    shifted_lower = lower + torch.tensor([[1e-5, 0.0]], dtype=torch.float32)
    with pytest.raises(BoundedBoxInverseBudgetError, match="reserve-relative"):
        validate_inverse_endpoint_budgets(
            lower, upper, shifted_lower, upper, policy=policy
        )

    negative_zero = torch.tensor([[-0.0, -0.1]], dtype=torch.float32)
    positive_zero = torch.tensor([[0.0, -0.1]], dtype=torch.float32)
    same_upper = torch.tensor([[0.2, 0.1]], dtype=torch.float32)
    diagnostics = validate_inverse_endpoint_budgets(
        negative_zero, same_upper, positive_zero, same_upper, policy=policy
    )[0]
    assert diagnostics.zero_status_mismatches == 0
    assert diagnostics.exact_zero_signbit_mismatches == 1

    crossed = validate_inverse_endpoint_budgets(
        torch.tensor([[-1e-7, -0.1]], dtype=torch.float32),
        same_upper,
        torch.tensor([[1e-7, -0.1]], dtype=torch.float32),
        same_upper,
        policy=policy,
    )[0]
    assert crossed.zero_status_mismatches == 0
    assert crossed.sign_crossings == 1


def test_inverse_preserves_distinct_target_and_operating_configuration_failures() -> None:
    policy = _ordinary_policy()
    with pytest.raises(InvalidPhysicalTargetError):
        invert_bounded_box_endpoints(
            torch.tensor([[-1e-5, -0.1]], dtype=torch.float32),
            torch.tensor([[1e-5, 0.1]], dtype=torch.float32),
            policy=policy,
        )

    # A valid box touching the domain face has no finite tanh inverse.
    with pytest.raises(FiniteRealUnattainableError):
        invert_bounded_box_endpoints(
            torch.tensor([[0.6, -0.1]], dtype=torch.float32),
            torch.tensor([[1.0, 0.1]], dtype=torch.float32),
            policy=policy,
        )

    with pytest.raises(UnsupportedNumericalConfigurationError):
        invert_bounded_box_endpoints(
            torch.tensor([[-0.2]], dtype=torch.float32),
            torch.tensor([[0.2]], dtype=torch.float32),
            policy=policy,
        )


def test_inverse_rejects_otherwise_valid_raw_state_outside_certified_range() -> None:
    raw_width = torch.tensor([[3.5, 0.0]], dtype=torch.float32)
    raw_center = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    h = (1e-4 + (2.0 - 1e-4) * torch.sigmoid(raw_width)) / 2.0
    c = (1.0 - h) * torch.tanh(raw_center)
    lower = c - h
    upper = c + h

    with pytest.raises(UnsupportedNumericalConfigurationError, match="raw operating"):
        invert_bounded_box_endpoints(lower, upper, policy=_ordinary_policy())


def test_physical_initializer_is_seeded_private_and_class_row_independent() -> None:
    binding = ClassBinding.from_synthetic_vocabulary_bytes(
        b'{"classes":[{"class_id":12,"trainable":true},'
        b'{"class_id":27,"trainable":true}],"trainable_class_ids":[12,27]}\n'
    )
    policies = (
        BoxWidthPolicy.ordinary(),
        BoxWidthPolicy.hosting_frame(),
    )
    numpy_state_before = np.random.get_state()
    first = initialize_bounded_boxes(
        binding,
        policies=policies,
        occupancy_targets=(0.01, 0.1),
        master_seed=20261112,
    )
    numpy_state_after = np.random.get_state()
    repeated = initialize_bounded_boxes(
        binding,
        policies=policies,
        occupancy_targets=(0.01, 0.1),
        master_seed=20261112,
    )

    assert first.class_ids == (12, 27)
    assert torch.equal(first.raw_width, repeated.raw_width)
    assert torch.equal(first.raw_center, repeated.raw_center)
    assert np.array_equal(numpy_state_before[1], numpy_state_after[1])
    assert numpy_state_before[0] == numpy_state_after[0]
    assert first.raw_width.shape == first.raw_center.shape == (2, 2)
    assert len(first.class_records) == 2
    for target, record in zip((0.01, 0.1), first.class_records, strict=True):
        assert record.requested_occupancy == target
        assert target / 2.0 <= record.actual_occupancy <= target * 2.0
        assert record.aspect_ratio <= 4.0
        assert record.accepted_attempt < 64
        assert record.conditioning_sigma_min >= 1e-5
        assert record.target_lower != record.reconstructed_lower or record.endpoint_error == 0.0
        assert len(record.attempts) == record.accepted_attempt + 1

    reversed_binding = ClassBinding.from_synthetic_vocabulary_bytes(
        b'{"classes":[{"class_id":27,"trainable":true},'
        b'{"class_id":12,"trainable":true}],"trainable_class_ids":[27,12]}\n'
    )
    reversed_rows = initialize_bounded_boxes(
        reversed_binding,
        policies=tuple(reversed(policies)),
        occupancy_targets=(0.1, 0.01),
        master_seed=20261112,
    )
    for class_id, raw_by_id in zip((12, 27), first.raw_width, strict=True):
        reverse_index = reversed_rows.class_ids.index(class_id)
        assert torch.equal(raw_by_id, reversed_rows.raw_width[reverse_index])
        assert torch.equal(
            first.raw_center[(12, 27).index(class_id)],
            reversed_rows.raw_center[reverse_index],
        )


def test_initializer_retries_are_deterministic_and_never_publish_partial_state(monkeypatch) -> None:
    binding = ClassBinding.from_synthetic_vocabulary_bytes(
        b'{"classes":[{"class_id":12,"trainable":true}],"trainable_class_ids":[12]}\n'
    )
    real_uniform = box_training_geometry._rng_uniform
    calls = 0

    def reject_first_occupancy(seed: int, low: float, high: float) -> float:
        nonlocal calls
        calls += 1
        if calls == 1:
            return low - 2.0
        return real_uniform(seed, low, high)

    monkeypatch.setattr(box_training_geometry, "_rng_uniform", reject_first_occupancy)
    first = initialize_bounded_boxes(
        binding,
        policies=(BoxWidthPolicy.ordinary(),),
        occupancy_targets=(0.01,),
        master_seed=227,
    )
    assert first.class_records[0].accepted_attempt == 1
    assert first.class_records[0].attempts[0].outcome == "unsupported_numerical_configuration"
    first_signature = tuple(
        (attempt.field_seeds, attempt.outcome, attempt.failure_reason)
        for attempt in first.class_records[0].attempts
    )

    calls = 0
    repeated = initialize_bounded_boxes(
        binding,
        policies=(BoxWidthPolicy.ordinary(),),
        occupancy_targets=(0.01,),
        master_seed=227,
    )
    assert repeated.class_records[0].accepted_attempt == 1
    assert tuple(
        (attempt.field_seeds, attempt.outcome, attempt.failure_reason)
        for attempt in repeated.class_records[0].attempts
    ) == first_signature
    assert torch.equal(first.raw_width, repeated.raw_width)
    assert torch.equal(first.raw_center, repeated.raw_center)

    def always_out_of_band(seed: int, low: float, high: float) -> float:
        nonlocal calls
        current_call = calls
        calls += 1
        return low - 2.0 if current_call % 4 == 0 else real_uniform(seed, low, high)

    calls = 0
    monkeypatch.setattr(box_training_geometry, "_rng_uniform", always_out_of_band)
    with pytest.raises(InitializationSearchExhaustedError) as failure:
        initialize_bounded_boxes(
            binding,
            policies=(BoxWidthPolicy.ordinary(),),
            occupancy_targets=(0.01,),
            master_seed=227,
        )
    assert len(failure.value.attempts) == 64
    assert failure.value.completed_class_ids == ()
    assert not hasattr(failure.value, "raw_width")


def test_trainable_module_preserves_binding_policy_and_box04_inference() -> None:
    binding = ClassBinding.from_synthetic_vocabulary_bytes(
        b'{"classes":[{"class_id":12,"trainable":true},'
        b'{"class_id":27,"trainable":true}],"trainable_class_ids":[12,27]}\n'
    )
    policies = (BoxWidthPolicy.ordinary(), BoxWidthPolicy.hosting_frame())
    module = TrainableBoundedBoxModule.initialize(
        binding,
        policies=policies,
        occupancy_targets=(0.01, 0.1),
        master_seed=19,
    )

    geometry = module.decode_physical_geometry()
    points = torch.tensor([[0.0, 0.0], [0.5, -0.25]], dtype=torch.float32)
    inferred = module(points)

    assert isinstance(module.raw_width, torch.nn.Parameter)
    assert isinstance(module.raw_center, torch.nn.Parameter)
    assert module.raw_width.shape == module.raw_center.shape == (2, 2)
    assert module.class_binding.trainable_class_ids == (12, 27)
    assert module.class_binding.class_manifest_sha256 == binding.class_manifest_sha256
    assert geometry.minimum_full_widths == (1e-4, 1e-4 + 2.0 * 1e-4)
    assert inferred.class_binding == binding
    assert inferred.contains.shape == inferred.signed_coordinate_clearance.shape == (2, 2)
    assert inferred.contains.device == points.device
    assert inferred.signed_coordinate_clearance.dtype == torch.float32
    assert tuple(record.class_id for record in module.initialization_records) == (12, 27)
    assert torch.equal(geometry.lower, torch.cat([row.lower for row in module.initialization.row_geometries]))


def test_trainable_clearance_keeps_point_and_raw_gradients_on_active_and_tied_faces() -> None:
    binding = ClassBinding.from_synthetic_vocabulary_bytes(
        b'{"classes":[{"class_id":12,"trainable":true}],"trainable_class_ids":[12]}\n'
    )
    module = TrainableBoundedBoxModule.initialize(
        binding,
        policies=(BoxWidthPolicy.ordinary(),),
        occupancy_targets=(0.1,),
        master_seed=73,
    )
    geometry = module.decode_physical_geometry()
    narrow_axis = min(range(2), key=lambda axis: float(geometry.half_widths[0, axis]))
    point = geometry.centers.detach().clone()
    point[0, narrow_axis] += 0.5 * geometry.half_widths[0, narrow_axis].detach()
    point.requires_grad_()
    active_clearance = module(point).signed_coordinate_clearance[0, 0]
    active_clearance.backward()
    assert point.grad is not None and float(point.grad.abs().sum()) > 0.0
    assert module.raw_width.grad is not None
    assert float(module.raw_width.grad.abs().sum()) > 0.0
    assert module.raw_center.grad is not None
    assert torch.isfinite(module.raw_center.grad).all()

    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        module.raw_center.zero_()
    tied_point = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
    tied_clearance = module(tied_point).signed_coordinate_clearance[0, 0]
    tied_clearance.backward()
    assert torch.equal(tied_point.grad, torch.zeros_like(tied_point))
    assert module.raw_width.grad is not None
    assert float(module.raw_width.grad.abs().sum()) > 0.0
    assert module.raw_center.grad is not None
    assert torch.equal(module.raw_center.grad, torch.zeros_like(module.raw_center))


def test_trainable_clearance_gradient_is_independent_between_class_rows() -> None:
    binding = ClassBinding.from_synthetic_vocabulary_bytes(
        b'{"classes":[{"class_id":12,"trainable":true},'
        b'{"class_id":27,"trainable":true}],"trainable_class_ids":[12,27]}\n'
    )
    module = TrainableBoundedBoxModule.initialize(
        binding,
        policies=(BoxWidthPolicy.ordinary(), BoxWidthPolicy.ordinary()),
        occupancy_targets=(0.01, 0.1),
        master_seed=481,
    )
    points = torch.tensor([[0.0, 0.0], [0.25, -0.1]], dtype=torch.float32)
    module(points).signed_coordinate_clearance[:, 0].sum().backward()

    assert module.raw_width.grad is not None and module.raw_center.grad is not None
    assert float(module.raw_width.grad[0].abs().sum()) > 0.0
    assert torch.equal(module.raw_width.grad[1], torch.zeros_like(module.raw_width.grad[1]))
    assert torch.equal(module.raw_center.grad[1], torch.zeros_like(module.raw_center.grad[1]))


def test_trainable_module_supports_empty_class_structure_and_rejects_unsupported_states() -> None:
    empty_binding = ClassBinding.from_synthetic_vocabulary_bytes(
        b'{"classes":[],"trainable_class_ids":[]}\n'
    )
    empty = TrainableBoundedBoxModule.initialize(
        empty_binding, policies=(), occupancy_targets=(), master_seed=7
    )
    empty_result = empty(torch.tensor([[0.0, 0.0]], dtype=torch.float32))
    assert empty_result.contains.shape == (1, 0)
    assert empty_result.signed_coordinate_clearance.shape == (1, 0)

    binding = ClassBinding.from_synthetic_vocabulary_bytes(
        b'{"classes":[{"class_id":12,"trainable":true}],"trainable_class_ids":[12]}\n'
    )
    module = TrainableBoundedBoxModule.initialize(
        binding,
        policies=(BoxWidthPolicy.ordinary(),),
        occupancy_targets=(0.1,),
        master_seed=11,
    )
    with torch.no_grad():
        module.raw_width[0, 0] = 3.25
    with pytest.raises(UnsupportedNumericalConfigurationError, match="operating range"):
        module(torch.tensor([[0.0, 0.0]], dtype=torch.float32))

    with torch.no_grad():
        module.raw_width[0, 0] = -40.0
    with pytest.raises(InvalidTrainablePhysicalGeometryError, match="minimum full width"):
        module.decode_physical_geometry()

    module = TrainableBoundedBoxModule.initialize(
        binding,
        policies=(BoxWidthPolicy.ordinary(),),
        occupancy_targets=(0.1,),
        master_seed=12,
    ).to(dtype=torch.float64)
    with pytest.raises(UnsupportedNumericalConfigurationError, match="float32"):
        module(torch.tensor([[0.0, 0.0]], dtype=torch.float32))


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_trainable_module_uses_float32_on_each_actual_device(device_name: str) -> None:
    if device_name == "cuda":
        assert torch.cuda.is_available(), "formal CP3 CUDA verification requires actual CUDA"
    device = torch.device(device_name)
    binding = ClassBinding.from_synthetic_vocabulary_bytes(
        b'{"classes":[{"class_id":12,"trainable":true}],"trainable_class_ids":[12]}\n'
    )
    module = TrainableBoundedBoxModule.initialize(
        binding,
        policies=(BoxWidthPolicy.hosting_frame(),),
        occupancy_targets=(0.1,),
        master_seed=882,
        device=device,
    )
    points = torch.tensor([[0.25, 0.25]], dtype=torch.float32, device=device, requires_grad=True)
    result = module(points)
    result.signed_coordinate_clearance.sum().backward()
    assert module.raw_width.device.type == module.raw_center.device.type == device.type
    if device.type == "cuda":
        assert module.raw_width.device.index == module.raw_center.device.index == 0
    assert module.raw_width.dtype == module.raw_center.dtype == torch.float32
    assert result.signed_coordinate_clearance.device.type == device.type
    assert points.grad is not None and torch.isfinite(points.grad).all()
    assert module.raw_width.grad is not None and torch.isfinite(module.raw_width.grad).all()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _formal_physical_targets(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = np.random.Generator(np.random.PCG64(seed))
    lower_rows: list[list[float]] = []
    upper_rows: list[list[float]] = []
    for index in range(1024):
        requested_q = 0.01 if index % 2 == 0 else 0.1
        q = math.exp(
            generator.uniform(math.log(requested_q / 2.0), math.log(requested_q * 2.0))
        )
        log_ratio = generator.uniform(-math.log(2.0), math.log(2.0))
        ratio = math.exp(log_ratio)
        half_widths = (math.sqrt(q * ratio), math.sqrt(q / ratio))
        center_fractions = generator.uniform(-0.5, 0.5, size=2)
        centers = tuple(
            (1.0 - half_widths[axis]) * float(center_fractions[axis])
            for axis in range(2)
        )
        lower_rows.append(
            [float(np.float32(centers[axis] - half_widths[axis])) for axis in range(2)]
        )
        upper_rows.append(
            [float(np.float32(centers[axis] + half_widths[axis])) for axis in range(2)]
        )
    return (
        torch.tensor(lower_rows, dtype=torch.float32),
        torch.tensor(upper_rows, dtype=torch.float32),
    )


def _ordered_ulp_distance(left: float, right: float) -> int:
    import struct

    if left == 0.0 and right == 0.0:
        return 0
    def ordered(value: float) -> int:
        bits = struct.unpack("!I", struct.pack("!f", value))[0]
        return (~bits) & 0xFFFFFFFF if bits & 0x80000000 else bits | 0x80000000
    return abs(ordered(left) - ordered(right))


@pytest.mark.parametrize(
    ("policy", "seed"),
    (
        (BoxWidthPolicy.ordinary(), 2026111201),
        (BoxWidthPolicy.hosting_frame(), 2026111202),
        (BoxWidthPolicy.ordinary(), 2026111203),
        (BoxWidthPolicy.hosting_frame(), 2026111204),
    ),
)
def test_formal_1024_target_inverse_cpu_cuda_and_locked_endpoint_budgets(
    policy: BoxWidthPolicy, seed: int, record_property
) -> None:
    target_lower_cpu, target_upper_cpu = _formal_physical_targets(seed)
    per_device = {}
    for device_name in ("cpu", "cuda"):
        if device_name == "cuda":
            assert torch.cuda.is_available(), "formal CP2 CUDA confirmation requires actual CUDA"
        device = torch.device(device_name)
        lower = target_lower_cpu.to(device)
        upper = target_upper_cpu.to(device)
        result = invert_bounded_box_endpoints(lower, upper, policy=policy)
        validate_bounded_box_geometry(result.geometry)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        assert result.raw_width.shape == result.raw_center.shape == (1024, 2)
        assert bool(torch.isfinite(result.raw_width).all())
        assert bool(torch.isfinite(result.raw_center).all())
        assert bool((result.raw_width >= -3.0).all() and (result.raw_width <= 3.0).all())
        assert bool((result.raw_center >= -3.0).all() and (result.raw_center <= 3.0).all())
        assert len(result.row_diagnostics) == 1024
        assert min(row.conditioning_sigma_min for row in result.row_diagnostics) >= 1e-5
        assert max(row.max_endpoint_error for row in result.row_diagnostics) <= min(
            policy.epsilon / 32.0,
            policy.delta / 32.0 if policy.delta is not None else policy.epsilon / 32.0,
        )
        per_device[device_name] = result

    cpu_result = per_device["cpu"]
    cuda_result = per_device["cuda"]
    assert torch.equal(cpu_result.raw_width, cuda_result.raw_width.cpu())
    assert torch.equal(cpu_result.raw_center, cuda_result.raw_center.cpu())
    cpu_lower = cpu_result.geometry.lower.tolist()
    cpu_upper = cpu_result.geometry.upper.tolist()
    cuda_lower = cuda_result.geometry.lower.cpu().tolist()
    cuda_upper = cuda_result.geometry.upper.cpu().tolist()
    maximum_cross_device_abs = 0.0
    zero_status_mismatches = 0
    sign_bit_mismatches = 0
    nonzero_sign_crossings = 0
    exact_zero_signbit_mismatches = 0
    near_zero_comparisons = 0
    outside_near_zero_comparisons = 0
    max_outside_near_zero_ulp = 0
    for row in range(1024):
        for axis in range(2):
            for cpu_value, cuda_value in (
                (cpu_lower[row][axis], cuda_lower[row][axis]),
                (cpu_upper[row][axis], cuda_upper[row][axis]),
            ):
                difference = abs(cpu_value - cuda_value)
                maximum_cross_device_abs = max(maximum_cross_device_abs, difference)
                assert difference <= 2.0**-21
                zero_status_diff = (cpu_value == 0.0) != (cuda_value == 0.0)
                zero_status_mismatches += int(zero_status_diff)
                cpu_sign = math.copysign(1.0, cpu_value)
                cuda_sign = math.copysign(1.0, cuda_value)
                sign_bit_diff = cpu_sign != cuda_sign
                sign_bit_mismatches += int(sign_bit_diff)
                nonzero_sign_crossings += int(cpu_value * cuda_value < 0.0)
                exact_zero_signbit_mismatches += int(
                    cpu_value == 0.0 and cuda_value == 0.0 and cpu_sign != cuda_sign
                )
                near_zero = abs(cpu_value) <= 0.01 or abs(cuda_value) <= 0.01
                if not near_zero:
                    outside_near_zero_comparisons += 1
                    distance = _ordered_ulp_distance(cpu_value, cuda_value)
                    max_outside_near_zero_ulp = max(max_outside_near_zero_ulp, distance)
                    assert distance <= 128
                    assert cpu_sign == cuda_sign
                else:
                    near_zero_comparisons += 1
                    # The signed-off absolute cross-device limit above is the
                    # near-zero rule; do not gate on ULP distance there.
                    if zero_status_diff or sign_bit_diff:
                        assert near_zero and difference <= 2.0**-21
    assert maximum_cross_device_abs <= 2.0**-21
    record_property("cpu_cuda_zero_status_mismatches", zero_status_mismatches)
    record_property("cpu_cuda_sign_bit_mismatches", sign_bit_mismatches)
    record_property("cpu_cuda_nonzero_sign_crossings", nonzero_sign_crossings)
    record_property("cpu_cuda_exact_zero_signbit_mismatches", exact_zero_signbit_mismatches)
    record_property("cpu_cuda_near_zero_comparisons", near_zero_comparisons)
    record_property("cpu_cuda_outside_near_zero_comparisons", outside_near_zero_comparisons)
    record_property("cpu_cuda_max_outside_near_zero_ulp", max_outside_near_zero_ulp)


def test_inverse_boundary_near_zero_nextafter_and_finite_real_statuses(record_property) -> None:
    policy = _ordinary_policy()
    zero = torch.tensor(0.0, dtype=torch.float32)
    positive = torch.tensor(float("inf"), dtype=torch.float32)
    tiny_positive = torch.nextafter(zero, positive).item()
    target_lower = torch.tensor([[0.0, -0.2]], dtype=torch.float32)
    target_upper = torch.tensor([[0.2, 0.2]], dtype=torch.float32)
    exact_zero = invert_bounded_box_endpoints(target_lower, target_upper, policy=policy)
    assert exact_zero.row_diagnostics[0].lower_endpoint_errors[0] <= 2.0**-20

    assert torch.cuda.is_available(), "near-zero CPU/CUDA confirmation requires actual CUDA"
    host_policy = BoxWidthPolicy.hosting_frame()
    host_cpu = invert_bounded_box_endpoints(
        target_lower, target_upper, policy=host_policy
    )
    host_cuda = invert_bounded_box_endpoints(
        target_lower.cuda(), target_upper.cuda(), policy=host_policy
    )
    torch.cuda.synchronize()
    near_zero_zero_status = 0
    near_zero_sign_differences = 0
    for cpu_side, cuda_side in (
        (host_cpu.geometry.lower, host_cuda.geometry.lower.cpu()),
        (host_cpu.geometry.upper, host_cuda.geometry.upper.cpu()),
    ):
        for cpu_value, cuda_value in zip(
            cpu_side.flatten().tolist(), cuda_side.flatten().tolist(), strict=True
        ):
            assert abs(cpu_value - cuda_value) <= 2.0**-21
            zero_status_diff = (cpu_value == 0.0) != (cuda_value == 0.0)
            sign_diff = math.copysign(1.0, cpu_value) != math.copysign(1.0, cuda_value)
            near_zero_zero_status += int(zero_status_diff)
            near_zero_sign_differences += int(sign_diff)
            if zero_status_diff or sign_diff:
                assert abs(cpu_value) <= 0.01 or abs(cuda_value) <= 0.01
                assert abs(cpu_value - cuda_value) <= 2.0**-21
            if abs(cpu_value) > 0.01 and abs(cuda_value) > 0.01:
                assert _ordered_ulp_distance(cpu_value, cuda_value) <= 128
    record_property("hosting_near_zero_zero_status_mismatches", near_zero_zero_status)
    record_property("hosting_near_zero_sign_differences", near_zero_sign_differences)

    nextafter_lower = torch.tensor([[tiny_positive, -0.2]], dtype=torch.float32)
    nextafter_results = {}
    for device_name in ("cpu", "cuda"):
        device = torch.device(device_name)
        result = invert_bounded_box_endpoints(
            nextafter_lower.to(device), target_upper.to(device), policy=policy
        )
        assert result.max_endpoint_error <= 1e-4 / 32.0
        validate_bounded_box_geometry(result.geometry)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        nextafter_results[device_name] = result
    nextafter_zero_status = 0
    nextafter_sign_differences = 0
    for cpu_side, cuda_side in (
        (
            nextafter_results["cpu"].geometry.lower,
            nextafter_results["cuda"].geometry.lower.cpu(),
        ),
        (
            nextafter_results["cpu"].geometry.upper,
            nextafter_results["cuda"].geometry.upper.cpu(),
        ),
    ):
        for cpu_value, cuda_value in zip(
            cpu_side.flatten().tolist(), cuda_side.flatten().tolist(), strict=True
        ):
            assert abs(cpu_value - cuda_value) <= 2.0**-21
            nextafter_zero_status += int((cpu_value == 0.0) != (cuda_value == 0.0))
            nextafter_sign_differences += int(
                math.copysign(1.0, cpu_value) != math.copysign(1.0, cuda_value)
            )
            if abs(cpu_value) > 0.01 and abs(cuda_value) > 0.01:
                assert _ordered_ulp_distance(cpu_value, cuda_value) <= 128
    record_property("nextafter_zero_status_mismatches", nextafter_zero_status)
    record_property("nextafter_sign_differences", nextafter_sign_differences)

    width_below_floor = torch.nextafter(
        torch.tensor(1e-4, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)
    ).item()
    face_predecessor = torch.nextafter(
        torch.tensor(1.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)
    ).item()
    for device_name in ("cpu", "cuda"):
        device = torch.device(device_name)
        if device.type == "cuda":
            assert torch.cuda.is_available(), "boundary CUDA confirmation requires actual CUDA"
        with pytest.raises(InvalidPhysicalTargetError):
            invert_bounded_box_endpoints(
                torch.tensor([[0.0, -0.2]], dtype=torch.float32, device=device),
                torch.tensor([[width_below_floor, 0.2]], dtype=torch.float32, device=device),
                policy=policy,
            )
        with pytest.raises(UnsupportedNumericalConfigurationError):
            invert_bounded_box_endpoints(
                torch.tensor([[0.6, -0.2]], dtype=torch.float32, device=device),
                torch.tensor([[face_predecessor, 0.2]], dtype=torch.float32, device=device),
                policy=policy,
            )
        with pytest.raises(FiniteRealUnattainableError):
            invert_bounded_box_endpoints(
                torch.tensor([[0.6, -0.2]], dtype=torch.float32, device=device),
                torch.tensor([[1.0, 0.2]], dtype=torch.float32, device=device),
                policy=policy,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def _independent_float64_endpoints(a: float, b: float, width: float) -> tuple[float, float]:
    sigmoid = 1.0 / (1.0 + math.exp(-a))
    half = (width + (2.0 - width) * sigmoid) / 2.0
    center = (1.0 - half) * math.tanh(b)
    return center - half, center + half


def _independent_float64_jacobian(
    a: float, b: float, width: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    p = 1.0 / (1.0 + math.exp(-a))
    u = math.tanh(b)
    half = (width + (2.0 - width) * p) / 2.0
    half_a = (2.0 - width) / 2.0 * p * (1.0 - p)
    center_b = (1.0 - half) * (1.0 - u * u)
    return (
        (-(1.0 + u) * half_a, center_b),
        ((1.0 - u) * half_a, center_b),
    )


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_approved_analytic_autograd_and_float64_finite_difference_budgets(
    device_name: str,
) -> None:
    if device_name == "cuda":
        assert torch.cuda.is_available(), "formal gradient CUDA verification requires actual CUDA"
    device = torch.device(device_name)
    policy = BoxWidthPolicy.hosting_frame()
    raw_width = torch.tensor(
        [[-2.5, -1.25], [-0.1, 1.75]], dtype=torch.float32, device=device, requires_grad=True
    )
    raw_center = torch.tensor(
        [[-0.75, 0.25], [1.0, -2.0]], dtype=torch.float32, device=device, requires_grad=True
    )
    analytic = analytic_endpoint_jacobian(
        raw_width, raw_center, minimum_full_width=policy.minimum_full_width
    )
    geometry = decode_bounded_box_raw(
        raw_width, raw_center, minimum_full_width=policy.minimum_full_width
    )
    lower_grad = torch.autograd.grad(
        geometry.lower.sum(), (raw_width, raw_center), retain_graph=True
    )
    upper_grad = torch.autograd.grad(geometry.upper.sum(), (raw_width, raw_center))
    autograd = torch.stack(
        (
            torch.stack(lower_grad, dim=-1),
            torch.stack(upper_grad, dim=-1),
        ),
        dim=-2,
    )
    max_gradient_difference = float((analytic - autograd).abs().max().item())
    assert max_gradient_difference <= 1e-6
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    step = 1e-4
    max_finite_difference_error = 0.0
    for row in range(raw_width.shape[0]):
        for axis in range(raw_width.shape[1]):
            a = float(raw_width.detach().cpu()[row, axis])
            b = float(raw_center.detach().cpu()[row, axis])
            for input_index, value in ((0, a), (1, b)):
                plus_a, plus_b = (a + step, b) if input_index == 0 else (a, b + step)
                minus_a, minus_b = (a - step, b) if input_index == 0 else (a, b - step)
                plus = _independent_float64_endpoints(plus_a, plus_b, policy.minimum_full_width)
                minus = _independent_float64_endpoints(minus_a, minus_b, policy.minimum_full_width)
                numerical = ((plus[0] - minus[0]) / (2.0 * step), (plus[1] - minus[1]) / (2.0 * step))
                reference = _independent_float64_jacobian(
                    a, b, policy.minimum_full_width
                )
                for endpoint_index in range(2):
                    max_finite_difference_error = max(
                        max_finite_difference_error,
                        abs(reference[endpoint_index][input_index] - numerical[endpoint_index]),
                    )
    assert max_finite_difference_error <= 1e-7


def test_cpu_cuda_local_gradient_difference_stays_within_locked_budget() -> None:
    assert torch.cuda.is_available(), "formal gradient CUDA confirmation requires actual CUDA"
    policy = BoxWidthPolicy.ordinary()
    raw_width_values = [[-2.0, -0.5], [0.75, 2.25]]
    raw_center_values = [[-1.5, 0.25], [1.25, -2.5]]
    jacobians = []
    for device_name in ("cpu", "cuda"):
        device = torch.device(device_name)
        raw_width = torch.tensor(
            raw_width_values, dtype=torch.float32, device=device, requires_grad=True
        )
        raw_center = torch.tensor(
            raw_center_values, dtype=torch.float32, device=device, requires_grad=True
        )
        geometry = decode_bounded_box_raw(
            raw_width, raw_center, minimum_full_width=policy.minimum_full_width
        )
        lower_grad = torch.autograd.grad(
            geometry.lower.sum(), (raw_width, raw_center), retain_graph=True
        )
        upper_grad = torch.autograd.grad(geometry.upper.sum(), (raw_width, raw_center))
        jacobians.append(
            torch.stack(
                (
                    torch.stack(lower_grad, dim=-1),
                    torch.stack(upper_grad, dim=-1),
                ),
                dim=-2,
            ).detach().cpu()
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    assert float((jacobians[0] - jacobians[1]).abs().max()) <= 1e-6


def test_read_only_box05_inspector_shows_geometry_and_independent_outputs() -> None:
    report = box05_inspector._inspection_report()

    assert report.physical_rows["Class ID"].tolist() == [12, 27]
    assert report.physical_rows["Width policy"].tolist() == ["ordinary", "hosting_frame"]
    required_columns = {
        "raw width a",
        "raw center b",
        "target lower",
        "target upper",
        "decoded lower",
        "decoded upper",
        "endpoint E_inf",
        "epsilon reserve ratio",
        "delta reserve ratio",
        "width reserves",
        "domain reserves",
        "physical occupancy",
        "sigma_min(J)",
    }
    assert required_columns <= set(report.physical_rows.columns)
    assert report.membership.shape == report.clearance.shape == (5, 2)
    assert report.membership_mismatches == ()
    assert report.clearance_mismatches == ()
    assert report.reconstruction_budget_mismatches == ()
    assert report.expected_domain_membership == report.actual_domain_membership


def test_box05_inspector_reports_independent_inference_mismatches(monkeypatch) -> None:
    from dataclasses import replace

    real_infer = BoundBoxClassifier.infer

    def deliberately_mismatched_inference(classifier, points, geometry):
        result = real_infer(classifier, points, geometry)
        membership = result.contains.clone()
        membership[0, 0] = ~membership[0, 0]
        clearance = result.signed_coordinate_clearance.clone()
        clearance[1, 1] += 0.125
        return replace(result, contains=membership, signed_coordinate_clearance=clearance)

    monkeypatch.setattr(BoundBoxClassifier, "infer", deliberately_mismatched_inference)
    report = box05_inspector._inspection_report()

    assert len(report.membership_mismatches) == 1
    assert report.membership_mismatches[0][:2] == (report.point_labels[0], "ordinary · class 12")
    assert len(report.clearance_mismatches) == 1
    assert report.clearance_mismatches[0][:2] == (report.point_labels[1], "hosting_frame · class 27")
