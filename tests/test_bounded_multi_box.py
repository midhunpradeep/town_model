from __future__ import annotations

import struct

import pytest
import torch

from town_model.models.bounded_box import bounded_box_contains, box_signed_clearance
from town_model.models.bounded_multi_box import (
    bounded_multi_box_contains,
    multi_box_signed_clearance,
)
from town_model.models.bounded_multi_box_fixture import (
    BOUNDED_MULTI_BOX_INSPECTION_FIXTURE,
)


def _fixture_tensors(device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
    fixture = BOUNDED_MULTI_BOX_INSPECTION_FIXTURE
    return (
        torch.tensor(fixture.points, dtype=torch.float32, device=device),
        torch.tensor(fixture.lower, dtype=torch.float32, device=device),
        torch.tensor(fixture.upper, dtype=torch.float32, device=device),
    )


def _device_for_test(device_name: str) -> torch.device:
    if device_name == "cuda":
        assert torch.cuda.is_available(), "BOX-03 CUDA acceptance requires a usable CUDA runtime"
        return torch.device("cuda:0")
    return torch.device("cpu")


def _ordered_float32_bits(value: float) -> int:
    bits = int.from_bytes(struct.pack(">f", value), byteorder="big")
    if bits & 0x80000000:
        return 0x80000000 - (bits & 0x7FFFFFFF)
    return bits | 0x80000000


def _max_float32_ulp_distance(left: torch.Tensor, right: torch.Tensor) -> int:
    left_values = left.detach().cpu().reshape(-1).tolist()
    right_values = right.detach().cpu().reshape(-1).tolist()
    assert len(left_values) == len(right_values)
    return max(
        (
            abs(_ordered_float32_bits(a) - _ordered_float32_bits(b))
            for a, b in zip(left_values, right_values, strict=True)
        ),
        default=0,
    )


def _assert_forward_budget(cpu: torch.Tensor, cuda: torch.Tensor) -> int:
    cuda_cpu = cuda.detach().cpu()
    assert torch.isfinite(cpu).all()
    assert torch.isfinite(cuda_cpu).all()
    assert torch.equal(cpu == 0, cuda_cpu == 0)
    nonzero = cpu != 0
    assert torch.equal(torch.signbit(cpu[nonzero]), torch.signbit(cuda_cpu[nonzero]))
    ulps = _max_float32_ulp_distance(cpu, cuda_cpu)
    assert ulps <= 1
    return ulps


def test_multibox_fixture_contains_independent_literal_oracles() -> None:
    fixture = BOUNDED_MULTI_BOX_INSPECTION_FIXTURE

    assert fixture.box_labels == ("Box 0", "Box 1")
    assert fixture.lower == ((-1.0, -0.5), (-0.25, -1.0))
    assert fixture.upper == ((0.5, 1.0), (1.0, 0.5))
    assert fixture.labels == ("A", "B", "C", "D", "E", "F")
    assert fixture.points == (
        (0.0, 0.0),
        (-0.75, 0.0),
        (0.75, 0.0),
        (-0.75, -0.75),
        (0.5, 0.5),
        (-1.25, 0.0),
    )
    assert fixture.expected_membership == (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (True, True),
        (False, False),
    )
    assert fixture.expected_clearances == (
        (0.50, 0.25),
        (0.25, -0.50),
        (-0.25, 0.25),
        (-0.25, -0.50),
        (0.0, 0.0),
        (-0.25, -1.0),
    )


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_fixture_membership_matches_literal_matrix_exactly(device_name: str) -> None:
    device = _device_for_test(device_name)
    points, lower, upper = _fixture_tensors(device)
    expected = torch.tensor(
        BOUNDED_MULTI_BOX_INSPECTION_FIXTURE.expected_membership,
        dtype=torch.bool,
        device=device,
    )

    actual = bounded_multi_box_contains(points, lower, upper)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    assert actual.dtype is torch.bool
    assert actual.shape == (6, 2)
    assert actual.device == device
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_fixture_clearance_matches_literal_matrix_exactly(device_name: str) -> None:
    device = _device_for_test(device_name)
    points, lower, upper = _fixture_tensors(device)
    expected = torch.tensor(
        BOUNDED_MULTI_BOX_INSPECTION_FIXTURE.expected_clearances,
        dtype=torch.float32,
        device=device,
    )

    actual = multi_box_signed_clearance(points, lower, upper)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    assert actual.dtype is torch.float32
    assert actual.shape == (6, 2)
    assert actual.device == device
    assert torch.equal(actual, expected)


def test_literal_fixture_exposes_overlap_boundaries_and_outside_domain_clearance() -> None:
    points, lower, upper = _fixture_tensors()

    membership = bounded_multi_box_contains(points, lower, upper)
    clearance = multi_box_signed_clearance(points, lower, upper)

    assert membership[0].tolist() == [True, True]
    assert membership[4].tolist() == [True, True]
    assert membership[3].tolist() == [False, False]
    assert membership[5].tolist() == [False, False]
    assert clearance[4].tolist() == [0.0, 0.0]
    assert clearance[5].tolist() == [-0.25, -1.0]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_M01_M05_closed_faces_overlap_order_and_outside_domain_raw_clearance(
    device_name: str,
) -> None:
    device = _device_for_test(device_name)
    points = torch.tensor(
        [[0.75, -0.75], [-2.0, 2.0], [0.5, 0.5]], dtype=torch.float32, device=device
    )
    lower = torch.tensor(((-1.0, -0.5), (-0.25, -1.0)), dtype=torch.float32, device=device)
    upper = torch.tensor(((0.5, 1.0), (1.0, 0.5)), dtype=torch.float32, device=device)

    membership = bounded_multi_box_contains(points, lower, upper)
    clearance = multi_box_signed_clearance(points, lower, upper)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    assert membership.tolist() == [[False, True], [False, False], [True, True]]
    assert clearance.tolist() == [[-0.25, 0.25], [-1.0, -1.75], [0.0, 0.0]]


@pytest.mark.parametrize(
    ("point_count", "box_count"),
    ((3, 2), (0, 2), (3, 0), (0, 0)),
)
@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_E01_all_empty_and_nonempty_axis_combinations(
    point_count: int, box_count: int, device_name: str
) -> None:
    device = _device_for_test(device_name)
    points = torch.zeros((point_count, 3), dtype=torch.float32, device=device)
    lower = torch.full((box_count, 3), -0.5, dtype=torch.float32, device=device)
    upper = torch.full((box_count, 3), 0.5, dtype=torch.float32, device=device)

    membership = bounded_multi_box_contains(points, lower, upper)
    clearance = multi_box_signed_clearance(points, lower, upper)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    assert membership.dtype is torch.bool
    assert membership.shape == (point_count, box_count)
    assert membership.device == device
    assert clearance.dtype is torch.float32
    assert clearance.shape == (point_count, box_count)
    assert clearance.device == device
    assert membership.tolist() == [[True] * box_count for _ in range(point_count)]
    assert clearance.tolist() == [[0.5] * box_count for _ in range(point_count)]


def test_E02_E04_empty_axes_still_validate_all_inputs_and_dimensions() -> None:
    empty_lower = torch.empty((0, 2), dtype=torch.float32)
    empty_upper = torch.empty((0, 2), dtype=torch.float32)
    invalid_empty_points = torch.tensor([[float("nan"), 0.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="finite"):
        bounded_multi_box_contains(invalid_empty_points, empty_lower, empty_upper)
    with pytest.raises(ValueError, match="finite"):
        multi_box_signed_clearance(invalid_empty_points, empty_lower, empty_upper)

    empty_points = torch.empty((0, 2), dtype=torch.float32)
    invalid_lower = torch.tensor([[-1.0, 0.0]], dtype=torch.float32)
    invalid_upper = torch.tensor([[0.5, 1.01]], dtype=torch.float32)
    with pytest.raises(ValueError, match="domain"):
        bounded_multi_box_contains(empty_points, invalid_lower, invalid_upper)
    with pytest.raises(ValueError, match="domain"):
        multi_box_signed_clearance(empty_points, invalid_lower, invalid_upper)

    valid_points = torch.zeros((2, 2), dtype=torch.float32)
    mismatched_lower = torch.empty((0, 2), dtype=torch.float32)
    mismatched_upper = torch.empty((1, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="same box count"):
        bounded_multi_box_contains(valid_points, mismatched_lower, mismatched_upper)
    with pytest.raises(ValueError, match="same box count"):
        multi_box_signed_clearance(valid_points, mismatched_lower, mismatched_upper)

    zero_dimensional_points = torch.empty((2, 0), dtype=torch.float32)
    zero_dimensional_lower = torch.empty((1, 0), dtype=torch.float32)
    zero_dimensional_upper = torch.empty((1, 0), dtype=torch.float32)
    with pytest.raises(ValueError, match="d must be at least 1"):
        bounded_multi_box_contains(
            zero_dimensional_points, zero_dimensional_lower, zero_dimensional_upper
        )
    with pytest.raises(ValueError, match="d must be at least 1"):
        multi_box_signed_clearance(
            zero_dimensional_points, zero_dimensional_lower, zero_dimensional_upper
        )


@pytest.mark.parametrize("evaluator", (bounded_multi_box_contains, multi_box_signed_clearance))
def test_V01_rejects_wrong_input_types_dtypes_layouts_ranks_and_shapes(evaluator) -> None:
    points = torch.zeros((2, 2), dtype=torch.float32)
    lower = torch.full((1, 2), -0.5, dtype=torch.float32)
    upper = torch.full((1, 2), 0.5, dtype=torch.float32)

    invalid_arguments = []
    for argument_index in range(3):
        arguments: list[object] = [points, lower, upper]
        arguments[argument_index] = [[0.0]]
        invalid_arguments.append((arguments, TypeError))
        arguments = [points, lower, upper]
        arguments[argument_index] = arguments[argument_index].to(torch.float64)
        invalid_arguments.append((arguments, TypeError))

    for argument_index, shape in enumerate(((2, 2), (1, 2), (1, 2))):
        sparse_value = torch.sparse_coo_tensor(
            torch.tensor([[0], [0]]), torch.tensor([0.0]), shape
        )
        arguments = [points, lower, upper]
        arguments[argument_index] = sparse_value
        invalid_arguments.append((arguments, TypeError))
    invalid_arguments.extend(
        (
            ([torch.zeros((2, 2, 1)), lower, upper], ValueError),
            ([torch.zeros((2,)), lower, upper], ValueError),
            ([points, torch.zeros((2,)), upper], ValueError),
            ([points, lower, torch.zeros((2,))], ValueError),
            ([points, torch.zeros((1, 1)), upper], ValueError),
            ([points, lower, torch.zeros((2, 2))], ValueError),
            ([points, lower, torch.zeros((1, 3))], ValueError),
        )
    )
    for arguments, error_type in invalid_arguments:
        with pytest.raises(error_type):
            evaluator(*arguments)


@pytest.mark.parametrize("evaluator", (bounded_multi_box_contains, multi_box_signed_clearance))
def test_V02_rejects_nonfinite_values_unordered_and_out_of_domain_corners(evaluator) -> None:
    valid_points = torch.zeros((1, 2), dtype=torch.float32)
    valid_lower = torch.tensor([[-1.0, -0.5]], dtype=torch.float32)
    valid_upper = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
    invalid_inputs = (
        (torch.tensor([[float("inf"), 0.0]]), valid_lower, valid_upper, "points"),
        (valid_points, torch.tensor([[float("nan"), -0.5]]), valid_upper, "corners"),
        (valid_points, valid_lower, torch.tensor([[0.5, float("-inf")]]), "corners"),
        (valid_points, torch.tensor([[0.0, -0.5]]), torch.tensor([[-0.1, 1.0]]), "less than or equal"),
        (valid_points, torch.tensor([[-1.01, -0.5]]), valid_upper, "domain"),
        (valid_points, valid_lower, torch.tensor([[0.5, 1.01]]), "domain"),
    )
    for points, lower, upper, message in invalid_inputs:
        with pytest.raises(ValueError, match=message):
            evaluator(points, lower, upper)


@pytest.mark.parametrize("evaluator", (bounded_multi_box_contains, multi_box_signed_clearance))
def test_V03_rejects_mixed_and_unsupported_devices(evaluator) -> None:
    assert torch.cuda.is_available(), "BOX-03 CUDA validation requires a usable CUDA runtime"
    points = torch.zeros((1, 2), dtype=torch.float32, device="cuda:0")
    lower = torch.tensor([[-1.0, -0.5]], dtype=torch.float32)
    upper = torch.tensor([[0.5, 1.0]], dtype=torch.float32, device="cuda:0")
    with pytest.raises(ValueError, match="same device"):
        evaluator(points, lower, upper)

    meta_points = torch.empty((1, 2), dtype=torch.float32, device="meta")
    meta_lower = torch.empty((1, 2), dtype=torch.float32, device="meta")
    meta_upper = torch.empty((1, 2), dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="supported CPU or CUDA"):
        evaluator(meta_points, meta_lower, meta_upper)


@pytest.mark.parametrize("evaluator", (bounded_multi_box_contains, multi_box_signed_clearance))
def test_V04_validation_does_not_mutate_inputs(evaluator) -> None:
    points, lower, upper = _fixture_tensors()
    originals = (points.clone(), lower.clone(), upper.clone())
    evaluator(points, lower, upper)
    assert torch.equal(points, originals[0])
    assert torch.equal(lower, originals[1])
    assert torch.equal(upper, originals[2])

    bad_upper = upper.clone()
    bad_upper[0, 0] = 2.0
    original_bad_upper = bad_upper.clone()
    with pytest.raises(ValueError):
        evaluator(points, lower, bad_upper)
    assert torch.equal(points, originals[0])
    assert torch.equal(lower, originals[1])
    assert torch.equal(bad_upper, original_bad_upper)


def _representative_case(dimensions: int, box_count: int) -> tuple[torch.Tensor, ...]:
    values = (-1.25, -1.0, -0.5, 0.0, 0.125, 0.25, 1.0, 1.25)
    points = torch.tensor(
        [[values[(row + 3 * coordinate) % len(values)] for coordinate in range(dimensions)]
         for row in range(len(values))],
        dtype=torch.float32,
    )
    lower_values: list[list[float]] = []
    upper_values: list[list[float]] = []
    for box in range(box_count):
        lower_row: list[float] = []
        upper_row: list[float] = []
        for coordinate in range(dimensions):
            kind = (box + coordinate) % 4
            if kind == 0:
                lower_row.append(-1.0)
                upper_row.append(1.0)
            elif kind == 1:
                lower_row.append(-0.5)
                upper_row.append(0.25)
            elif kind == 2:
                lower_row.append(0.125)
                upper_row.append(0.125)
            else:
                lower_row.append(0.0)
                upper_row.append(2.0**-20)
        lower_values.append(lower_row)
        upper_values.append(upper_row)
    return (
        points,
        torch.tensor(lower_values, dtype=torch.float32).reshape(box_count, dimensions),
        torch.tensor(upper_values, dtype=torch.float32).reshape(box_count, dimensions),
    )


@pytest.mark.parametrize(
    ("dimensions", "box_count"),
    ((1, 1), (2, 2), (4, 4), (1, 4), (2, 3), (4, 1)),
)
@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_M03_M06_each_box_column_matches_independent_box01_box02_oracles(
    dimensions: int, box_count: int, device_name: str
) -> None:
    device = _device_for_test(device_name)
    points, lower, upper = _representative_case(dimensions, box_count)
    points, lower, upper = (value.to(device) for value in (points, lower, upper))

    membership = bounded_multi_box_contains(points, lower, upper)
    clearance = multi_box_signed_clearance(points, lower, upper)
    for box_index in range(box_count):
        expected_membership = bounded_box_contains(
            points, lower[box_index], upper[box_index]
        )
        expected_clearance = box_signed_clearance(
            points, lower[box_index], upper[box_index]
        )
        assert torch.equal(membership[:, box_index], expected_membership)
        assert torch.equal(clearance[:, box_index], expected_clearance)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def test_M04_membership_and_clearance_keep_point_and_box_axis_order() -> None:
    points, lower, upper = _fixture_tensors()
    actual_membership = bounded_multi_box_contains(points, lower, upper)
    actual_clearance = multi_box_signed_clearance(points, lower, upper)
    expected_membership = torch.tensor(
        BOUNDED_MULTI_BOX_INSPECTION_FIXTURE.expected_membership, dtype=torch.bool
    )
    expected_clearance = torch.tensor(
        BOUNDED_MULTI_BOX_INSPECTION_FIXTURE.expected_clearances, dtype=torch.float32
    )

    assert torch.equal(actual_membership, expected_membership)
    assert torch.equal(actual_clearance, expected_clearance)


def test_M02_clearance_builds_all_faces_then_uses_one_amin_on_face_axis(monkeypatch) -> None:
    original_amin = torch.amin
    calls: list[tuple[tuple[int, ...], int | None]] = []

    def recording_amin(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        dimension = kwargs.get("dim", args[0] if args else None)
        calls.append((tuple(input.shape), dimension))
        return original_amin(input, *args, **kwargs)

    monkeypatch.setattr(torch, "amin", recording_amin)
    points = torch.tensor([[0.0, 0.0], [0.75, -0.75]], dtype=torch.float32)
    lower = torch.tensor(((-1.0, -0.5), (-0.25, -1.0)), dtype=torch.float32)
    upper = torch.tensor(((0.5, 1.0), (1.0, 0.5)), dtype=torch.float32)

    result = multi_box_signed_clearance(points, lower, upper)

    assert result.tolist() == [[0.5, 0.25], [-0.25, 0.25]]
    assert calls == [((2, 2, 4), -1)]


def test_E03_zero_box_clearance_reduces_only_nonempty_face_axis(monkeypatch) -> None:
    original_amin = torch.amin
    calls: list[tuple[tuple[int, ...], int | None]] = []

    def recording_amin(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        dimension = kwargs.get("dim", args[0] if args else None)
        calls.append((tuple(input.shape), dimension))
        return original_amin(input, *args, **kwargs)

    monkeypatch.setattr(torch, "amin", recording_amin)
    result = multi_box_signed_clearance(
        torch.zeros((3, 2), dtype=torch.float32),
        torch.empty((0, 2), dtype=torch.float32),
        torch.empty((0, 2), dtype=torch.float32),
    )
    assert result.shape == (3, 0)
    assert calls == [((3, 0, 4), -1)]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_G01_box_zero_fixture_clearance_has_isolated_amin_tie_gradients(
    device_name: str,
) -> None:
    device = _device_for_test(device_name)
    points, lower, upper = _fixture_tensors(device)
    points.requires_grad_()
    lower.requires_grad_()
    upper.requires_grad_()

    clearance = multi_box_signed_clearance(points, lower, upper)
    gradients = torch.autograd.grad(clearance[0, 0], (points, lower, upper))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    assert clearance[0, 0].item() == 0.5
    assert gradients[0][0].tolist() == [-0.5, 0.5]
    assert gradients[1].tolist() == [[0.0, -0.5], [0.0, 0.0]]
    assert gradients[2].tolist() == [[0.5, 0.0], [0.0, 0.0]]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_G02_box_one_fixture_clearance_has_isolated_unique_face_gradients(
    device_name: str,
) -> None:
    device = _device_for_test(device_name)
    points, lower, upper = _fixture_tensors(device)
    points.requires_grad_()
    lower.requires_grad_()
    upper.requires_grad_()

    clearance = multi_box_signed_clearance(points, lower, upper)
    gradients = torch.autograd.grad(clearance[0, 1], (points, lower, upper))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    assert clearance[0, 1].item() == 0.25
    assert gradients[0][0].tolist() == [1.0, 0.0]
    assert gradients[1].tolist() == [[0.0, 0.0], [-1.0, 0.0]]
    assert gradients[2].tolist() == [[0.0, 0.0], [0.0, 0.0]]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_G03_one_point_box_output_has_no_unrelated_point_or_box_gradients(
    device_name: str,
) -> None:
    device = _device_for_test(device_name)
    points, lower, upper = _fixture_tensors(device)
    points.requires_grad_()
    lower.requires_grad_()
    upper.requires_grad_()

    clearance = multi_box_signed_clearance(points, lower, upper)
    gradients = torch.autograd.grad(clearance[0, 1], (points, lower, upper))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    assert gradients[0][0].tolist() == [1.0, 0.0]
    assert torch.equal(gradients[0][1:], torch.zeros_like(gradients[0][1:]))
    assert torch.equal(gradients[1][0], torch.zeros_like(gradients[1][0]))
    assert torch.equal(gradients[2], torch.zeros_like(gradients[2]))


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_G04_selected_clearance_sum_accumulates_point_and_shared_corner_gradients(
    device_name: str,
) -> None:
    device = _device_for_test(device_name)
    points = torch.tensor([[0.0], [0.25]], dtype=torch.float32, device=device, requires_grad=True)
    lower = torch.tensor([[-1.0], [-0.5]], dtype=torch.float32, device=device, requires_grad=True)
    upper = torch.tensor([[1.0], [1.0]], dtype=torch.float32, device=device, requires_grad=True)

    clearance = multi_box_signed_clearance(points, lower, upper)
    selected_sum = clearance[0, 0] + clearance[0, 1] + clearance[1, 0]
    gradients = torch.autograd.grad(selected_sum, (points, lower, upper))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    assert clearance.tolist() == [[1.0, 0.5], [0.75, 0.75]]
    assert gradients[0].tolist() == [[1.0], [-1.0]]
    assert gradients[1].tolist() == [[-0.5], [-1.0]]
    assert gradients[2].tolist() == [[1.5], [0.0]]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_G05_amin_tie_and_degenerate_box_gradients_are_framework_defined_and_finite(
    device_name: str,
) -> None:
    device = _device_for_test(device_name)
    points = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device, requires_grad=True)
    lower = torch.zeros((1, 2), dtype=torch.float32, device=device, requires_grad=True)
    upper = torch.zeros((1, 2), dtype=torch.float32, device=device, requires_grad=True)

    clearance = multi_box_signed_clearance(points, lower, upper)
    gradients = torch.autograd.grad(clearance.sum(), (points, lower, upper))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    assert clearance.tolist() == [[0.0]]
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert gradients[0].tolist() == [[0.0, 0.0]]
    assert gradients[1].tolist() == [[-0.25, -0.25]]
    assert gradients[2].tolist() == [[0.25, 0.25]]


def _selected_output_gradients(
    device: torch.device, box_index: int, degenerate: bool = False
) -> tuple[torch.Tensor, ...]:
    if degenerate:
        points = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device)
        lower = torch.zeros((2, 2), dtype=torch.float32, device=device)
        upper = torch.zeros((2, 2), dtype=torch.float32, device=device)
        output_row = 0
    else:
        points = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device)
        lower = torch.tensor(((-1.0, -0.5), (-0.25, -1.0)), dtype=torch.float32, device=device)
        upper = torch.tensor(((0.5, 1.0), (1.0, 0.5)), dtype=torch.float32, device=device)
        output_row = 0
    points.requires_grad_()
    lower.requires_grad_()
    upper.requires_grad_()
    value = multi_box_signed_clearance(points, lower, upper)[output_row, box_index]
    return torch.autograd.grad(value, (points, lower, upper))


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_G06_multi_box_gradients_match_independent_single_box_oracle(
    device_name: str,
) -> None:
    device = _device_for_test(device_name)
    points = torch.tensor(
        [[0.0, 0.0], [0.25, -0.25], [-0.75, 0.2]], dtype=torch.float32, device=device
    )
    lower = torch.tensor(((-1.0, -0.5), (-0.25, -1.0)), dtype=torch.float32, device=device)
    upper = torch.tensor(((0.5, 1.0), (1.0, 0.5)), dtype=torch.float32, device=device)
    points_multi = points.clone().requires_grad_()
    lower_multi = lower.clone().requires_grad_()
    upper_multi = upper.clone().requires_grad_()
    multi = multi_box_signed_clearance(points_multi, lower_multi, upper_multi)

    for box_index in range(2):
        multi_gradients = torch.autograd.grad(
            multi[:, box_index].sum(),
            (points_multi, lower_multi, upper_multi),
            retain_graph=box_index == 0,
        )
        points_single = points.clone().requires_grad_()
        lower_single = lower[box_index].clone().requires_grad_()
        upper_single = upper[box_index].clone().requires_grad_()
        single = box_signed_clearance(points_single, lower_single, upper_single)
        single_gradients = torch.autograd.grad(
            single.sum(), (points_single, lower_single, upper_single)
        )

        assert torch.equal(multi[:, box_index], single)
        assert torch.equal(multi_gradients[0], single_gradients[0])
        assert torch.equal(multi_gradients[1][box_index], single_gradients[1])
        assert torch.equal(multi_gradients[2][box_index], single_gradients[2])
        for other_box in range(2):
            if other_box != box_index:
                assert torch.equal(
                    multi_gradients[1][other_box], torch.zeros_like(multi_gradients[1][other_box])
                )
                assert torch.equal(
                    multi_gradients[2][other_box], torch.zeros_like(multi_gradients[2][other_box])
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def test_G05_N02_cpu_cuda_tie_and_degenerate_gradient_differences_within_budget() -> None:
    assert torch.cuda.is_available(), "BOX-03 CUDA gradient acceptance requires a usable CUDA runtime"
    cpu = torch.device("cpu")
    cuda = torch.device("cuda:0")
    maximum_absolute_difference = 0.0
    for box_index, degenerate in ((0, False), (1, False), (0, True)):
        cpu_gradients = _selected_output_gradients(cpu, box_index, degenerate)
        cuda_gradients = _selected_output_gradients(cuda, box_index, degenerate)
        torch.cuda.synchronize(cuda)
        for cpu_gradient, cuda_gradient in zip(cpu_gradients, cuda_gradients, strict=True):
            assert torch.isfinite(cpu_gradient).all()
            assert torch.isfinite(cuda_gradient).all()
            difference = (cpu_gradient - cuda_gradient.cpu()).abs().max().item()
            maximum_absolute_difference = max(maximum_absolute_difference, difference)
    assert maximum_absolute_difference <= 1e-6


def _face_nextafter_case(box_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fixture = BOUNDED_MULTI_BOX_INSPECTION_FIXTURE
    lower = torch.tensor(fixture.lower[box_index], dtype=torch.float32)
    upper = torch.tensor(fixture.upper[box_index], dtype=torch.float32)
    negative_infinity = torch.tensor(float("-inf"), dtype=torch.float32)
    positive_infinity = torch.tensor(float("inf"), dtype=torch.float32)
    zeros = torch.tensor(0.0, dtype=torch.float32)
    points = torch.stack(
        (
            torch.stack((torch.nextafter(lower[0], negative_infinity), zeros)),
            torch.stack((torch.nextafter(lower[0], positive_infinity), zeros)),
            torch.stack((torch.nextafter(upper[0], negative_infinity), zeros)),
            torch.stack((torch.nextafter(upper[0], positive_infinity), zeros)),
            torch.stack((zeros, torch.nextafter(lower[1], negative_infinity))),
            torch.stack((zeros, torch.nextafter(lower[1], positive_infinity))),
            torch.stack((zeros, torch.nextafter(upper[1], negative_infinity))),
            torch.stack((zeros, torch.nextafter(upper[1], positive_infinity))),
        )
    )
    if box_index == 0:
        expected_values = (
            -1.1920928955078125e-7,
            5.960464477539063e-8,
            2.9802322387695312e-8,
            -5.960464477539063e-8,
            -5.960464477539063e-8,
            2.9802322387695312e-8,
            5.960464477539063e-8,
            -1.1920928955078125e-7,
        )
    else:
        expected_values = (
            -2.9802322387695312e-8,
            1.4901161193847656e-8,
            5.960464477539063e-8,
            -1.1920928955078125e-7,
            -1.1920928955078125e-7,
            5.960464477539063e-8,
            2.9802322387695312e-8,
            -5.960464477539063e-8,
        )
    return (
        points,
        lower.unsqueeze(0),
        upper.unsqueeze(0),
        torch.tensor(expected_values, dtype=torch.float32).unsqueeze(1),
    )


@pytest.mark.parametrize("box_index", (0, 1))
def test_N01_immediate_nextafter_neighbors_of_every_box_face_are_exact_on_cpu_cuda(
    box_index: int,
) -> None:
    assert torch.cuda.is_available(), "BOX-03 CUDA numerical acceptance requires a usable CUDA runtime"
    points, lower, upper, expected = _face_nextafter_case(box_index)
    cpu_actual = multi_box_signed_clearance(points, lower, upper)
    cpu_membership = bounded_multi_box_contains(points, lower, upper)
    cuda_device = torch.device("cuda:0")
    cuda_points, cuda_lower, cuda_upper = (
        points.to(cuda_device), lower.to(cuda_device), upper.to(cuda_device)
    )
    cuda_actual = multi_box_signed_clearance(cuda_points, cuda_lower, cuda_upper)
    cuda_membership = bounded_multi_box_contains(cuda_points, cuda_lower, cuda_upper)
    torch.cuda.synchronize(cuda_device)

    assert torch.equal(cpu_actual, expected)
    assert torch.equal(cuda_actual.cpu(), expected)
    assert _assert_forward_budget(cpu_actual, cuda_actual) == 0
    expected_membership = torch.tensor(
        [[False], [True], [True], [False], [False], [True], [True], [False]],
        dtype=torch.bool,
    )
    assert torch.equal(cpu_membership, expected_membership)
    assert torch.equal(cuda_membership.cpu(), expected_membership)


def test_N02_small_large_extreme_narrow_and_degenerate_forward_values_are_finite() -> None:
    assert torch.cuda.is_available(), "BOX-03 CUDA numerical acceptance requires a usable CUDA runtime"
    points, lower, upper = _fixture_tensors()
    smallest_positive = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    largest_finite = torch.tensor(torch.finfo(torch.float32).max, dtype=torch.float32)
    probes = torch.stack(
        (
            torch.stack((smallest_positive, torch.tensor(0.0))),
            torch.stack((-smallest_positive, torch.tensor(0.0))),
            torch.stack((largest_finite, torch.tensor(0.0))),
            torch.stack((-largest_finite, torch.tensor(0.0))),
            torch.tensor([-2.0, 0.0]),
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.0, -2.0]),
            torch.tensor([0.0, 2.0]),
        )
    )
    all_points = torch.cat((points, probes), dim=0)
    cpu_actual = multi_box_signed_clearance(all_points, lower, upper)
    cpu_membership = bounded_multi_box_contains(all_points, lower, upper)
    cuda_device = torch.device("cuda:0")
    cuda_points, cuda_lower, cuda_upper = (
        all_points.to(cuda_device), lower.to(cuda_device), upper.to(cuda_device)
    )
    cuda_actual = multi_box_signed_clearance(cuda_points, cuda_lower, cuda_upper)
    cuda_membership = bounded_multi_box_contains(cuda_points, cuda_lower, cuda_upper)
    torch.cuda.synchronize(cuda_device)
    largest_negative = -torch.finfo(torch.float32).max
    expected_probe_values = torch.tensor(
        [
            [0.5, 0.25],
            [0.5, 0.25],
            [largest_negative, largest_negative],
            [largest_negative, largest_negative],
            [-1.0, -1.75],
            [-1.5, -1.0],
            [-1.5, -1.0],
            [-1.0, -1.5],
        ],
        dtype=torch.float32,
    )

    assert torch.isfinite(cpu_actual).all()
    assert torch.isfinite(cuda_actual).all()
    assert torch.equal(
        cpu_actual[:6],
        torch.tensor(BOUNDED_MULTI_BOX_INSPECTION_FIXTURE.expected_clearances),
    )
    assert torch.equal(cpu_actual[6:], expected_probe_values)
    assert torch.equal(cuda_actual.cpu(), cpu_actual)
    expected_probe_membership = torch.tensor(
        [
            [True, True], [True, True], [False, False], [False, False],
            [False, False], [False, False], [False, False], [False, False],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(
        cpu_membership[6:], expected_probe_membership
    )
    assert torch.equal(cuda_membership.cpu(), cpu_membership)
    assert _assert_forward_budget(cpu_actual, cuda_actual) == 0

    degenerate_lower = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    degenerate_upper = degenerate_lower.clone()
    degenerate_points = torch.tensor([[0.25, -0.5], [0.25, 0.0]], dtype=torch.float32)
    narrow_lower = torch.tensor([[0.0]], dtype=torch.float32)
    narrow_upper = torch.tensor([[2.0**-20]], dtype=torch.float32)
    narrow_midpoint = torch.tensor([[2.0**-21]], dtype=torch.float32)
    for case_points, case_lower, case_upper, expected in (
        (degenerate_points, degenerate_lower, degenerate_upper, torch.tensor([[0.0], [-0.5]])),
        (narrow_midpoint, narrow_lower, narrow_upper, torch.tensor([[2.0**-21]])),
    ):
        cpu_case = multi_box_signed_clearance(case_points, case_lower, case_upper)
        cuda_case = multi_box_signed_clearance(
            case_points.to(cuda_device), case_lower.to(cuda_device), case_upper.to(cuda_device)
        )
        torch.cuda.synchronize(cuda_device)
        assert torch.equal(cpu_case, expected)
        assert torch.equal(cuda_case.cpu(), expected)
        assert _assert_forward_budget(cpu_case, cuda_case) == 0


def test_N03_cpu_cuda_forward_comparison_obeys_one_ulp_and_sign_budget() -> None:
    assert torch.cuda.is_available(), "BOX-03 CUDA numerical acceptance requires a usable CUDA runtime"
    cuda_device = torch.device("cuda:0")
    maximum_ulp_difference = 0
    for dimensions, box_count in ((1, 1), (2, 2), (4, 4), (1, 4), (2, 3), (4, 1)):
        cpu_points, cpu_lower, cpu_upper = _representative_case(dimensions, box_count)
        cpu_result = multi_box_signed_clearance(cpu_points, cpu_lower, cpu_upper)
        cuda_result = multi_box_signed_clearance(
            cpu_points.to(cuda_device),
            cpu_lower.to(cuda_device),
            cpu_upper.to(cuda_device),
        )
        torch.cuda.synchronize(cuda_device)
        maximum_ulp_difference = max(
            maximum_ulp_difference, _assert_forward_budget(cpu_result, cuda_result)
        )
    assert maximum_ulp_difference <= 1


def test_N03_cuda_rejects_invalid_finite_and_empty_axis_inputs() -> None:
    assert torch.cuda.is_available(), "BOX-03 CUDA validation requires a usable CUDA runtime"
    device = torch.device("cuda:0")
    evaluators = (bounded_multi_box_contains, multi_box_signed_clearance)
    for evaluator in evaluators:
        with pytest.raises(ValueError, match="finite"):
            evaluator(
                torch.tensor([[float("nan"), 0.0]], dtype=torch.float32, device=device),
                torch.empty((0, 2), dtype=torch.float32, device=device),
                torch.empty((0, 2), dtype=torch.float32, device=device),
            )
        with pytest.raises(ValueError, match="domain"):
            evaluator(
                torch.empty((0, 2), dtype=torch.float32, device=device),
                torch.tensor([[-1.0, -0.5]], dtype=torch.float32, device=device),
                torch.tensor([[0.5, 1.01]], dtype=torch.float32, device=device),
            )
        with pytest.raises(ValueError, match="same box count"):
            evaluator(
                torch.zeros((2, 2), dtype=torch.float32, device=device),
                torch.empty((0, 2), dtype=torch.float32, device=device),
                torch.empty((1, 2), dtype=torch.float32, device=device),
            )
        with pytest.raises(ValueError, match="d must be at least 1"):
            evaluator(
                torch.empty((2, 0), dtype=torch.float32, device=device),
                torch.empty((1, 0), dtype=torch.float32, device=device),
                torch.empty((1, 0), dtype=torch.float32, device=device),
            )
    torch.cuda.synchronize(device)
