from __future__ import annotations

from itertools import product

import pytest
import torch

from town_model.models.bounded_box import bounded_box_contains, box_signed_clearance
from town_model.models.bounded_box_fixture import BOUNDED_BOX_INSPECTION_FIXTURE


def _fixture_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fixture = BOUNDED_BOX_INSPECTION_FIXTURE
    return (
        torch.tensor(fixture.points, dtype=torch.float32),
        torch.tensor(fixture.lower, dtype=torch.float32),
        torch.tensor(fixture.upper, dtype=torch.float32),
    )


def _cuda_device() -> torch.device:
    assert torch.cuda.is_available(), "BOX-01 CUDA acceptance requires a usable CUDA runtime"
    return torch.device("cuda:0")


def test_shared_inspection_fixture_has_independent_literal_oracle() -> None:
    fixture = BOUNDED_BOX_INSPECTION_FIXTURE
    assert fixture.lower == (-1.0, -0.5)
    assert fixture.upper == (0.5, 1.0)
    assert fixture.labels == ("A", "B", "C", "D", "E", "F")
    assert fixture.points == (
        (0.0, 0.0),
        (-1.0, 0.0),
        (-1.0, 1.0),
        (0.5, 0.0),
        (0.75, 0.0),
        (-1.25, 0.0),
    )
    assert fixture.expected_contains == (True, True, True, True, False, False)
    assert fixture.expected_clearances == (0.50, 0.00, 0.00, 0.00, -0.25, -0.25)


def test_fixed_fixture_membership_matches_literal_expected_values() -> None:
    points, lower, upper = _fixture_tensors()

    actual = bounded_box_contains(points, lower, upper)

    assert actual.dtype is torch.bool
    assert actual.shape == (6,)
    assert actual.device == points.device
    assert actual.tolist() == [True, True, True, True, False, False]


def test_M01_interior_points_are_members() -> None:
    points = torch.tensor([[0.0, 0.0], [-0.5, 0.25]], dtype=torch.float32)
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32)

    assert bounded_box_contains(points, lower, upper).tolist() == [True, True]


def test_M02_exterior_points_inside_domain_are_not_members() -> None:
    points = torch.tensor([[0.75, 0.0], [0.0, -0.75], [0.75, -0.75]], dtype=torch.float32)
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32)

    assert bounded_box_contains(points, lower, upper).tolist() == [False, False, False]


def test_M03_exact_box_faces_are_included() -> None:
    points = torch.tensor([[-1.0, 0.0], [0.5, 0.0], [0.0, -0.5], [0.0, 1.0]], dtype=torch.float32)
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32)

    assert bounded_box_contains(points, lower, upper).tolist() == [True, True, True, True]


def test_M04_all_physical_box_corners_are_included() -> None:
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32)
    points = torch.tensor(
        [[-1.0, -0.5], [-1.0, 1.0], [0.5, -0.5], [0.5, 1.0]], dtype=torch.float32
    )

    assert bounded_box_contains(points, lower, upper).tolist() == [True, True, True, True]


@pytest.mark.parametrize("dimensions", (1, 2, 4))
def test_M04_all_valid_bounded_domain_corners_are_included(dimensions: int) -> None:
    lower = torch.full((dimensions,), -1.0, dtype=torch.float32)
    upper = torch.full((dimensions,), 1.0, dtype=torch.float32)
    points = torch.tensor(tuple(product((-1.0, 1.0), repeat=dimensions)), dtype=torch.float32)

    assert bounded_box_contains(points, lower, upper).tolist() == [True] * (2**dimensions)


def test_M05_adjacent_float32_values_on_both_sides_of_box_faces() -> None:
    negative_infinity = torch.tensor(float("-inf"), dtype=torch.float32)
    positive_infinity = torch.tensor(float("inf"), dtype=torch.float32)
    lower_x = torch.tensor(-1.0, dtype=torch.float32)
    upper_x = torch.tensor(0.5, dtype=torch.float32)
    lower_y = torch.tensor(-0.5, dtype=torch.float32)
    upper_y = torch.tensor(1.0, dtype=torch.float32)
    points = torch.stack(
        (
            torch.stack((torch.nextafter(lower_x, negative_infinity), torch.tensor(0.0))),
            torch.stack((torch.nextafter(lower_x, positive_infinity), torch.tensor(0.0))),
            torch.stack((torch.nextafter(upper_x, negative_infinity), torch.tensor(0.0))),
            torch.stack((torch.nextafter(upper_x, positive_infinity), torch.tensor(0.0))),
            torch.stack((torch.tensor(0.0), torch.nextafter(lower_y, negative_infinity))),
            torch.stack((torch.tensor(0.0), torch.nextafter(lower_y, positive_infinity))),
            torch.stack((torch.tensor(0.0), torch.nextafter(upper_y, negative_infinity))),
            torch.stack((torch.tensor(0.0), torch.nextafter(upper_y, positive_infinity))),
        )
    )
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32)

    assert bounded_box_contains(points, lower, upper).tolist() == [
        False, True, True, False, False, True, True, False
    ]


def test_M06_boxes_touching_domain_endpoints_include_exact_endpoints() -> None:
    lower = torch.tensor([-1.0, -1.0], dtype=torch.float32)
    upper = torch.tensor([1.0, 1.0], dtype=torch.float32)
    domain_corners = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=torch.float32
    )

    assert bounded_box_contains(domain_corners, lower, upper).tolist() == [True] * 4


def test_M07_finite_outside_domain_queries_are_false_without_magnitude_cutoff() -> None:
    smallest_positive = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    largest_finite = torch.tensor(torch.finfo(torch.float32).max, dtype=torch.float32)
    points = torch.stack(
        (
            torch.tensor([-2.0, 0.0]),
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.0, -2.0]),
            torch.tensor([0.0, 2.0]),
            torch.stack((smallest_positive, torch.tensor(0.0))),
            torch.stack((-smallest_positive, torch.tensor(0.0))),
            torch.stack((largest_finite, torch.tensor(0.0))),
            torch.stack((-largest_finite, torch.tensor(0.0))),
        )
    )
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32)

    assert torch.isfinite(points).all()
    assert bounded_box_contains(points, lower, upper).tolist() == [
        False, False, False, False, True, True, False, False
    ]


def test_M08_zero_width_coordinate_is_a_valid_fixed_box() -> None:
    zero = torch.tensor(0.0, dtype=torch.float32)
    below_zero = torch.nextafter(zero, torch.tensor(float("-inf")))
    points = torch.stack(
        (torch.tensor([0.0, 0.0]), torch.stack((below_zero, zero)), torch.tensor([0.0, 1.0]))
    )
    lower = torch.tensor([0.0, -1.0], dtype=torch.float32)
    upper = torch.tensor([0.0, 1.0], dtype=torch.float32)

    assert bounded_box_contains(points, lower, upper).tolist() == [True, False, True]


def test_M09_fully_collapsed_box_contains_only_its_fixed_point() -> None:
    corner = torch.tensor([0.25, -0.5], dtype=torch.float32)
    left = torch.nextafter(corner[0], torch.tensor(float("-inf")))
    right = torch.nextafter(corner[0], torch.tensor(float("inf")))
    points = torch.stack((corner, torch.stack((left, corner[1])), torch.stack((right, corner[1]))))

    assert bounded_box_contains(points, corner, corner).tolist() == [True, False, False]


@pytest.mark.parametrize("dimensions", (1, 2, 4))
def test_M10_dimensions_one_two_and_greater_than_two(dimensions: int) -> None:
    lower = torch.full((dimensions,), -0.25, dtype=torch.float32)
    upper = torch.full((dimensions,), 0.25, dtype=torch.float32)
    points = torch.stack((torch.zeros(dimensions), torch.full((dimensions,), 0.5)))

    assert bounded_box_contains(points, lower, upper).tolist() == [True, False]


def test_M11_empty_singleton_and_multiple_point_batches() -> None:
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32)

    empty = bounded_box_contains(torch.empty((0, 2), dtype=torch.float32), lower, upper)
    singleton = bounded_box_contains(torch.tensor([[0.0, 0.0]], dtype=torch.float32), lower, upper)
    multiple = bounded_box_contains(torch.tensor([[0.0, 0.0], [0.75, 0.0]], dtype=torch.float32), lower, upper)

    assert empty.dtype is torch.bool and empty.shape == (0,)
    assert singleton.tolist() == [True]
    assert multiple.tolist() == [True, False]


@pytest.mark.parametrize(
    ("points", "lower", "upper"),
    (
        (torch.empty((3, 0), dtype=torch.float32), torch.empty((0,), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)),
        (torch.tensor([0.0, 0.0], dtype=torch.float32), torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])),
        (torch.zeros((2, 2, 1), dtype=torch.float32), torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])),
        (torch.zeros((2, 2), dtype=torch.float32), torch.tensor([0.0]), torch.tensor([1.0])),
        (torch.zeros((2, 2), dtype=torch.float32), torch.tensor([-1.0, -1.0]), torch.tensor([1.0])),
        (torch.zeros((2, 2), dtype=torch.float32), torch.tensor([[-1.0, -1.0]]), torch.tensor([[1.0, 1.0]])),
    ),
)
def test_V01_rejects_zero_dimensions_wrong_ranks_and_shape_mismatches(
    points: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        bounded_box_contains(points, lower, upper)


@pytest.mark.parametrize(
    ("lower", "upper"),
    (
        ((0.5, -0.5), (-0.5, 0.5)),
        ((-1.01, -0.5), (0.5, 1.0)),
        ((-1.0, -0.5), (0.5, 1.01)),
    ),
)
def test_V02_rejects_reversed_or_out_of_domain_corners(
    lower: tuple[float, float], upper: tuple[float, float]
) -> None:
    points = torch.zeros((1, 2), dtype=torch.float32)
    with pytest.raises(ValueError):
        bounded_box_contains(
            points,
            torch.tensor(lower, dtype=torch.float32),
            torch.tensor(upper, dtype=torch.float32),
        )


@pytest.mark.parametrize(
    ("argument", "values"),
    (
        ("points", ((float("nan"), 0.0),)),
        ("points", ((float("inf"), 0.0),)),
        ("lower", (float("nan"), -0.5)),
        ("lower", (float("-inf"), -0.5)),
        ("upper", (0.5, float("nan"))),
        ("upper", (0.5, float("inf"))),
    ),
)
def test_V03_rejects_nan_and_infinity(argument: str, values: tuple[object, ...]) -> None:
    inputs: list[torch.Tensor] = [
        torch.zeros((1, 2), dtype=torch.float32),
        torch.tensor([-1.0, -0.5], dtype=torch.float32),
        torch.tensor([0.5, 1.0], dtype=torch.float32),
    ]
    index = {"points": 0, "lower": 1, "upper": 2}[argument]
    inputs[index] = torch.tensor(values, dtype=torch.float32)

    with pytest.raises(ValueError):
        bounded_box_contains(*inputs)


@pytest.mark.parametrize("argument_index", (0, 1, 2))
def test_V04_rejects_non_float32_tensors(argument_index: int) -> None:
    inputs = [
        torch.zeros((1, 2), dtype=torch.float32),
        torch.tensor([-1.0, -0.5], dtype=torch.float32),
        torch.tensor([0.5, 1.0], dtype=torch.float32),
    ]
    inputs[argument_index] = inputs[argument_index].to(torch.float64)

    with pytest.raises(TypeError):
        bounded_box_contains(*inputs)


@pytest.mark.parametrize("argument_index", (0, 1, 2))
def test_V04_rejects_non_tensor_inputs(argument_index: int) -> None:
    inputs: list[object] = [
        torch.zeros((1, 2), dtype=torch.float32),
        torch.tensor([-1.0, -0.5], dtype=torch.float32),
        torch.tensor([0.5, 1.0], dtype=torch.float32),
    ]
    inputs[argument_index] = [0.0, 0.0]

    with pytest.raises(TypeError):
        bounded_box_contains(*inputs)  # type: ignore[arg-type]


def test_V04_rejects_non_strided_tensor_layout() -> None:
    sparse_points = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]), torch.tensor([0.0], dtype=torch.float32), (1, 2)
    )

    with pytest.raises(TypeError):
        bounded_box_contains(
            sparse_points,
            torch.tensor([-1.0, -0.5], dtype=torch.float32),
            torch.tensor([0.5, 1.0], dtype=torch.float32),
        )


def test_V05_rejects_mixed_cpu_and_cuda_devices() -> None:
    device = _cuda_device()
    points = torch.zeros((1, 2), dtype=torch.float32, device=device)
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="same device"):
        bounded_box_contains(points, lower, upper)


def test_V05_rejects_unsupported_common_device() -> None:
    device = torch.device("meta")
    points = torch.empty((2, 2), dtype=torch.float32, device=device)
    lower = torch.empty((2,), dtype=torch.float32, device=device)
    upper = torch.empty((2,), dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="supported CPU or CUDA"):
        bounded_box_contains(points, lower, upper)


def test_V06_preserves_inputs_and_returns_no_partial_result_on_failure() -> None:
    points, lower, upper = _fixture_tensors()
    originals = (points.clone(), lower.clone(), upper.clone())

    result = bounded_box_contains(points, lower, upper)
    assert result.tolist() == [True, True, True, True, False, False]
    assert torch.equal(points, originals[0])
    assert torch.equal(lower, originals[1])
    assert torch.equal(upper, originals[2])

    invalid_upper = upper.clone()
    invalid_upper[0] = 2.0
    invalid_original = invalid_upper.clone()
    no_result: object = "no result returned"
    with pytest.raises(ValueError):
        no_result = bounded_box_contains(points, lower, invalid_upper)
    assert no_result == "no result returned"
    assert torch.equal(points, originals[0])
    assert torch.equal(lower, originals[1])
    assert torch.equal(invalid_upper, invalid_original)


def test_G01_cpu_and_cuda_results_match_for_ordinary_inputs() -> None:
    device = _cuda_device()
    points, lower, upper = _fixture_tensors()
    cpu_result = bounded_box_contains(points, lower, upper)
    cuda_result = bounded_box_contains(
        points.to(device), lower.to(device), upper.to(device)
    )
    torch.cuda.synchronize(device)

    assert cuda_result.device == device
    assert torch.equal(cuda_result.cpu(), cpu_result)


def test_G02_cpu_and_cuda_match_for_nextafter_boundaries_and_degenerate_boxes() -> None:
    device = _cuda_device()
    negative_infinity = torch.tensor(float("-inf"), dtype=torch.float32)
    positive_infinity = torch.tensor(float("inf"), dtype=torch.float32)
    lower_x = torch.tensor(-1.0, dtype=torch.float32)
    upper_x = torch.tensor(0.5, dtype=torch.float32)
    lower_y = torch.tensor(-0.5, dtype=torch.float32)
    upper_y = torch.tensor(1.0, dtype=torch.float32)
    points = torch.stack(
        (
            torch.stack((torch.nextafter(lower_x, negative_infinity), torch.tensor(0.0))),
            torch.stack((torch.nextafter(lower_x, positive_infinity), torch.tensor(0.0))),
            torch.stack((torch.nextafter(upper_x, negative_infinity), torch.tensor(0.0))),
            torch.stack((torch.nextafter(upper_x, positive_infinity), torch.tensor(0.0))),
            torch.stack((torch.tensor(0.0), torch.nextafter(lower_y, negative_infinity))),
            torch.stack((torch.tensor(0.0), torch.nextafter(lower_y, positive_infinity))),
            torch.stack((torch.tensor(0.0), torch.nextafter(upper_y, negative_infinity))),
            torch.stack((torch.tensor(0.0), torch.nextafter(upper_y, positive_infinity))),
        )
    )
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32)
    cpu_boundary = bounded_box_contains(points, lower, upper)
    cuda_boundary = bounded_box_contains(points.to(device), lower.to(device), upper.to(device))

    smallest_positive = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    largest_finite = torch.tensor(torch.finfo(torch.float32).max, dtype=torch.float32)
    extreme_points = torch.stack(
        (
            torch.tensor([-2.0, 0.0]),
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.0, -2.0]),
            torch.tensor([0.0, 2.0]),
            torch.stack((smallest_positive, torch.tensor(0.0))),
            torch.stack((-smallest_positive, torch.tensor(0.0))),
            torch.stack((largest_finite, torch.tensor(0.0))),
            torch.stack((-largest_finite, torch.tensor(0.0))),
        )
    )
    cpu_extremes = bounded_box_contains(extreme_points, lower, upper)
    cuda_extremes = bounded_box_contains(
        extreme_points.to(device), lower.to(device), upper.to(device)
    )

    collapsed = torch.tensor([0.25, -0.5], dtype=torch.float32)
    collapsed_points = torch.tensor([[0.25, -0.5], [0.25, 0.0]], dtype=torch.float32)
    cpu_degenerate = bounded_box_contains(collapsed_points, collapsed, collapsed)
    cuda_degenerate = bounded_box_contains(
        collapsed_points.to(device), collapsed.to(device), collapsed.to(device)
    )
    torch.cuda.synchronize(device)

    assert cpu_boundary.tolist() == [False, True, True, False, False, True, True, False]
    assert torch.equal(cuda_boundary.cpu(), cpu_boundary)
    assert cpu_extremes.tolist() == [False, False, False, False, True, True, False, False]
    assert torch.equal(cuda_extremes.cpu(), cpu_extremes)
    assert cpu_degenerate.tolist() == [True, False]
    assert torch.equal(cuda_degenerate.cpu(), cpu_degenerate)


def test_G03_cuda_empty_batch_and_invalid_input_behavior() -> None:
    device = _cuda_device()
    empty = bounded_box_contains(
        torch.empty((0, 2), dtype=torch.float32, device=device),
        torch.tensor([-1.0, -0.5], dtype=torch.float32, device=device),
        torch.tensor([0.5, 1.0], dtype=torch.float32, device=device),
    )
    assert empty.dtype is torch.bool and empty.shape == (0,) and empty.device == device

    with pytest.raises(ValueError, match="finite"):
        bounded_box_contains(
            torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device),
            torch.tensor([-1.0, float("nan")], dtype=torch.float32, device=device),
            torch.tensor([0.5, 1.0], dtype=torch.float32, device=device),
        )


def test_G04_project_cuda_runtime_and_actual_device_are_available() -> None:
    device = _cuda_device()
    assert torch.version.cuda is not None
    assert torch.cuda.device_count() >= 1
    assert torch.cuda.get_device_name(device)
    value = torch.tensor([2.0], dtype=torch.float32, device=device)
    assert (value.square() == 4.0).item()
    torch.cuda.synchronize(device)


def _device_for_box02(device_name: str) -> torch.device:
    if device_name == "cuda":
        return _cuda_device()
    return torch.device("cpu")


def _ordered_float32_bits(value: float) -> int:
    import struct

    bits = int.from_bytes(struct.pack(">f", value), byteorder="big")
    if bits & 0x80000000:
        return 0x80000000 - (bits & 0x7FFFFFFF)
    return bits | 0x80000000


def _max_float32_ulp_distance(left: torch.Tensor, right: torch.Tensor) -> int:
    left_values = left.detach().cpu().reshape(-1).tolist()
    right_values = right.detach().cpu().reshape(-1).tolist()
    assert len(left_values) == len(right_values)
    return max(
        (abs(_ordered_float32_bits(a) - _ordered_float32_bits(b)) for a, b in zip(left_values, right_values, strict=True)),
        default=0,
    )


def _assert_box02_forward_budget(cpu: torch.Tensor, cuda: torch.Tensor) -> None:
    cuda_cpu = cuda.detach().cpu()
    assert torch.equal(cpu == 0, cuda_cpu == 0)
    nonzero = cpu != 0
    assert torch.equal(torch.signbit(cpu[nonzero]), torch.signbit(cuda_cpu[nonzero]))
    assert _max_float32_ulp_distance(cpu, cuda_cpu) <= 1


def test_box02_C01_formula_constructs_all_faces_and_uses_one_amin(monkeypatch) -> None:
    original_amin = torch.amin
    calls: list[tuple[tuple[int, ...], int | None]] = []

    def recording_amin(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        dimension = kwargs.get("dim", args[0] if args else None)
        calls.append((tuple(input.shape), dimension))
        return original_amin(input, *args, **kwargs)

    monkeypatch.setattr(torch, "amin", recording_amin)
    points = torch.tensor([[0.0, 0.0], [0.75, -0.75]], dtype=torch.float32)
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32)

    actual = box_signed_clearance(points, lower, upper)

    assert actual.tolist() == [0.5, -0.25]
    assert calls == [((2, 4), 1)]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_box02_C02_N01_fixture_clearances_are_exact_on_cpu_and_cuda(device_name: str) -> None:
    device = _device_for_box02(device_name)
    points, lower, upper = _fixture_tensors()
    expected = torch.tensor(
        BOUNDED_BOX_INSPECTION_FIXTURE.expected_clearances,
        dtype=torch.float32,
        device=device,
    )

    actual = box_signed_clearance(points.to(device), lower.to(device), upper.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    assert actual.dtype is torch.float32
    assert actual.shape == (6,)
    assert actual.device == device
    assert torch.equal(actual, expected)
    assert actual.tolist() == [0.5, 0.0, 0.0, 0.0, -0.25, -0.25]


def test_box02_C03_interior_faces_exterior_multiple_violations_and_domain_queries() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [0.5, 0.0],
            [0.75, -0.75],
            [2.0, 0.0],
            [-2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32)

    actual = box_signed_clearance(points, lower, upper)

    assert actual.tolist() == [0.5, 0.0, 0.0, -0.25, -1.5, -1.0]


def test_box02_C04_degenerate_and_positive_width_narrow_boxes() -> None:
    fixed = torch.tensor([0.25, -0.5], dtype=torch.float32)
    fixed_points = torch.tensor([[0.25, -0.5], [0.25, -0.25]], dtype=torch.float32)
    fixed_clearances = box_signed_clearance(fixed_points, fixed, fixed)

    narrow_lower = torch.tensor([0.0], dtype=torch.float32)
    narrow_upper = torch.tensor([2.0**-20], dtype=torch.float32)
    narrow_midpoint = torch.tensor([[2.0**-21]], dtype=torch.float32)
    narrow_clearance = box_signed_clearance(narrow_midpoint, narrow_lower, narrow_upper)

    assert fixed_clearances.tolist() == [0.0, -0.25]
    assert narrow_clearance.tolist() == [2.0**-21]


@pytest.mark.parametrize("dimensions", (1, 2, 4))
def test_box02_C05_dimensions_empty_batches_and_input_order(dimensions: int) -> None:
    lower = torch.full((dimensions,), -0.25, dtype=torch.float32)
    upper = torch.full((dimensions,), 0.25, dtype=torch.float32)
    points = torch.stack(
        (
            torch.full((dimensions,), 0.125),
            torch.zeros(dimensions),
            torch.full((dimensions,), 0.5),
            torch.full((dimensions,), -0.5),
        )
    )

    actual = box_signed_clearance(points, lower, upper)
    empty = box_signed_clearance(torch.empty((0, dimensions)), lower, upper)

    assert actual.shape == (4,)
    assert actual.dtype is torch.float32
    assert actual.tolist() == [0.125, 0.25, -0.25, -0.25]
    assert empty.shape == (0,)
    assert empty.dtype is torch.float32
    assert empty.device == points.device


def test_box02_V01_V02_boolean_membership_and_shared_validation_contract() -> None:
    points = torch.tensor([[0.0, 0.0], [0.75, 0.0]], dtype=torch.float32)
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32)
    expected_membership = torch.tensor([True, False])
    expected_clearance = torch.tensor([0.5, -0.25], dtype=torch.float32)

    assert torch.equal(bounded_box_contains(points, lower, upper), expected_membership)
    assert torch.equal(box_signed_clearance(points, lower, upper), expected_clearance)

    invalid_cases = (
        (
            torch.zeros((1, 2), dtype=torch.float64),
            lower,
            upper,
            TypeError,
            "dtype",
        ),
        (
            torch.zeros((1, 2), dtype=torch.float32),
            lower,
            torch.tensor([0.5, 1.01], dtype=torch.float32),
            ValueError,
            "domain",
        ),
        (
            torch.tensor([[float("nan"), 0.0]], dtype=torch.float32),
            lower,
            upper,
            ValueError,
            "finite",
        ),
    )
    for invalid_points, invalid_lower, invalid_upper, error_type, message in invalid_cases:
        for evaluator in (bounded_box_contains, box_signed_clearance):
            with pytest.raises(error_type, match=message):
                evaluator(invalid_points, invalid_lower, invalid_upper)


def test_box02_V03_signed_validation_does_not_mutate_inputs() -> None:
    points, lower, upper = _fixture_tensors()
    originals = (points.clone(), lower.clone(), upper.clone())

    result = box_signed_clearance(points, lower, upper)
    assert result.tolist() == list(BOUNDED_BOX_INSPECTION_FIXTURE.expected_clearances)
    assert torch.equal(points, originals[0])
    assert torch.equal(lower, originals[1])
    assert torch.equal(upper, originals[2])

    invalid_upper = upper.clone()
    invalid_upper[0] = 2.0
    invalid_original = invalid_upper.clone()
    with pytest.raises(ValueError, match="domain"):
        box_signed_clearance(points, lower, invalid_upper)
    assert torch.equal(points, originals[0])
    assert torch.equal(lower, originals[1])
    assert torch.equal(invalid_upper, invalid_original)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_box02_G01_unique_upper_face_gradients(device_name: str) -> None:
    device = _device_for_box02(device_name)
    points = torch.tensor([[0.25]], dtype=torch.float32, device=device, requires_grad=True)
    lower = torch.tensor([-0.5], dtype=torch.float32, device=device, requires_grad=True)
    upper = torch.tensor([0.5], dtype=torch.float32, device=device, requires_grad=True)

    clearance = box_signed_clearance(points, lower, upper)
    grad_points, grad_lower, grad_upper = torch.autograd.grad(clearance.sum(), (points, lower, upper))

    assert clearance.item() == 0.25
    assert grad_points.tolist() == [[-1.0]]
    assert grad_lower.tolist() == [0.0]
    assert grad_upper.tolist() == [1.0]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_box02_G02_unique_lower_face_gradients(device_name: str) -> None:
    device = _device_for_box02(device_name)
    points = torch.tensor([[-0.25]], dtype=torch.float32, device=device, requires_grad=True)
    lower = torch.tensor([-0.5], dtype=torch.float32, device=device, requires_grad=True)
    upper = torch.tensor([0.5], dtype=torch.float32, device=device, requires_grad=True)

    clearance = box_signed_clearance(points, lower, upper)
    grad_points, grad_lower, grad_upper = torch.autograd.grad(clearance.sum(), (points, lower, upper))

    assert clearance.item() == 0.25
    assert grad_points.tolist() == [[1.0]]
    assert grad_lower.tolist() == [-1.0]
    assert grad_upper.tolist() == [0.0]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_box02_G03_exterior_point_gradients(device_name: str) -> None:
    device = _device_for_box02(device_name)
    points = torch.tensor([[0.75]], dtype=torch.float32, device=device, requires_grad=True)
    lower = torch.tensor([-0.5], dtype=torch.float32, device=device, requires_grad=True)
    upper = torch.tensor([0.5], dtype=torch.float32, device=device, requires_grad=True)

    clearance = box_signed_clearance(points, lower, upper)
    grad_points, grad_lower, grad_upper = torch.autograd.grad(clearance.sum(), (points, lower, upper))

    assert clearance.item() == -0.25
    assert grad_points.tolist() == [[-1.0]]
    assert grad_lower.tolist() == [0.0]
    assert grad_upper.tolist() == [1.0]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_box02_G04_one_dimensional_tie_uses_amin_subgradient(device_name: str) -> None:
    device = _device_for_box02(device_name)
    points = torch.tensor([[0.0]], dtype=torch.float32, device=device, requires_grad=True)
    lower = torch.tensor([-0.5], dtype=torch.float32, device=device, requires_grad=True)
    upper = torch.tensor([0.5], dtype=torch.float32, device=device, requires_grad=True)

    clearance = box_signed_clearance(points, lower, upper)
    grad_points, grad_lower, grad_upper = torch.autograd.grad(clearance.sum(), (points, lower, upper))

    assert clearance.item() == 0.5
    assert grad_points.tolist() == [[0.0]]
    assert grad_lower.tolist() == [-0.5]
    assert grad_upper.tolist() == [0.5]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_box02_G05_multidimensional_fixture_tie_uses_amin_subgradient(device_name: str) -> None:
    device = _device_for_box02(device_name)
    points = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device, requires_grad=True)
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32, device=device, requires_grad=True)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32, device=device, requires_grad=True)

    clearance = box_signed_clearance(points, lower, upper)
    grad_points, grad_lower, grad_upper = torch.autograd.grad(clearance.sum(), (points, lower, upper))

    assert clearance.item() == 0.5
    assert grad_points.tolist() == [[-0.5, 0.5]]
    assert grad_lower.tolist() == [0.0, -0.5]
    assert grad_upper.tolist() == [0.5, 0.0]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_box02_G06_degenerate_and_empty_gradients_are_finite(device_name: str) -> None:
    device = _device_for_box02(device_name)
    fixed_points = torch.tensor([[0.25, -0.5]], dtype=torch.float32, device=device, requires_grad=True)
    fixed_lower = torch.tensor([0.25, -0.5], dtype=torch.float32, device=device, requires_grad=True)
    fixed_upper = torch.tensor([0.25, -0.5], dtype=torch.float32, device=device, requires_grad=True)

    fixed_clearance = box_signed_clearance(fixed_points, fixed_lower, fixed_upper)
    fixed_gradients = torch.autograd.grad(
        fixed_clearance.sum(), (fixed_points, fixed_lower, fixed_upper)
    )
    assert fixed_clearance.tolist() == [0.0]
    assert all(torch.isfinite(gradient).all() for gradient in fixed_gradients)
    assert fixed_gradients[0].tolist() == [[0.0, 0.0]]
    assert fixed_gradients[1].tolist() == [-0.25, -0.25]
    assert fixed_gradients[2].tolist() == [0.25, 0.25]

    empty_points = torch.empty((0, 2), dtype=torch.float32, device=device, requires_grad=True)
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32, device=device, requires_grad=True)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32, device=device, requires_grad=True)
    empty_clearance = box_signed_clearance(empty_points, lower, upper)
    empty_gradients = torch.autograd.grad(empty_clearance.sum(), (empty_points, lower, upper))

    assert empty_clearance.shape == (0,)
    assert all(torch.isfinite(gradient).all() for gradient in empty_gradients)
    assert empty_gradients[0].shape == (0, 2)
    assert empty_gradients[1].tolist() == [0.0, 0.0]
    assert empty_gradients[2].tolist() == [0.0, 0.0]


def test_box02_N02_cuda_forward_gradient_and_invalid_input_checks() -> None:
    device = _cuda_device()
    points = torch.tensor([[0.25]], dtype=torch.float32, device=device, requires_grad=True)
    lower = torch.tensor([-0.5], dtype=torch.float32, device=device, requires_grad=True)
    upper = torch.tensor([0.5], dtype=torch.float32, device=device, requires_grad=True)
    clearance = box_signed_clearance(points, lower, upper)
    gradients = torch.autograd.grad(clearance.sum(), (points, lower, upper))
    torch.cuda.synchronize(device)

    assert clearance.device == device and clearance.item() == 0.25
    assert [gradient.item() for gradient in gradients] == [-1.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="finite"):
        box_signed_clearance(
            torch.tensor([[float("nan")]], dtype=torch.float32, device=device),
            lower.detach(),
            upper.detach(),
        )
    with pytest.raises(ValueError, match="domain"):
        box_signed_clearance(
            torch.tensor([[0.0]], dtype=torch.float32, device=device),
            lower.detach(),
            torch.tensor([1.01], dtype=torch.float32, device=device),
        )


def test_box02_N03_nextafter_faces_small_large_extreme_degenerate_and_empty() -> None:
    device = _cuda_device()
    negative_infinity = torch.tensor(float("-inf"), dtype=torch.float32)
    positive_infinity = torch.tensor(float("inf"), dtype=torch.float32)
    lower_x = torch.tensor(-1.0, dtype=torch.float32)
    upper_x = torch.tensor(0.5, dtype=torch.float32)
    lower_y = torch.tensor(-0.5, dtype=torch.float32)
    upper_y = torch.tensor(1.0, dtype=torch.float32)
    boundary_points = torch.stack(
        (
            torch.stack((torch.nextafter(lower_x, negative_infinity), torch.tensor(0.0))),
            torch.stack((torch.nextafter(lower_x, positive_infinity), torch.tensor(0.0))),
            torch.stack((torch.nextafter(upper_x, negative_infinity), torch.tensor(0.0))),
            torch.stack((torch.nextafter(upper_x, positive_infinity), torch.tensor(0.0))),
            torch.stack((torch.tensor(0.0), torch.nextafter(lower_y, negative_infinity))),
            torch.stack((torch.tensor(0.0), torch.nextafter(lower_y, positive_infinity))),
            torch.stack((torch.tensor(0.0), torch.nextafter(upper_y, negative_infinity))),
            torch.stack((torch.tensor(0.0), torch.nextafter(upper_y, positive_infinity))),
        )
    )
    lower = torch.tensor([-1.0, -0.5], dtype=torch.float32)
    upper = torch.tensor([0.5, 1.0], dtype=torch.float32)
    expected_boundary = torch.tensor(
        [
            -1.1920928955078125e-7,
            5.960464477539063e-8,
            2.9802322387695312e-8,
            -5.960464477539063e-8,
            -5.960464477539063e-8,
            2.9802322387695312e-8,
            5.960464477539063e-8,
            -1.1920928955078125e-7,
        ],
        dtype=torch.float32,
    )
    cpu_boundary = box_signed_clearance(boundary_points, lower, upper)
    cuda_boundary = box_signed_clearance(
        boundary_points.to(device), lower.to(device), upper.to(device)
    )
    torch.cuda.synchronize(device)
    assert torch.equal(cpu_boundary, expected_boundary)
    assert torch.equal(cuda_boundary.cpu(), expected_boundary)
    _assert_box02_forward_budget(cpu_boundary, cuda_boundary)

    smallest_positive = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    largest_finite = torch.tensor(torch.finfo(torch.float32).max, dtype=torch.float32)
    extreme_points = torch.stack(
        (
            torch.stack((smallest_positive, torch.tensor(0.0))),
            torch.stack((-smallest_positive, torch.tensor(0.0))),
            torch.tensor([-2.0, 0.0]),
            torch.tensor([2.0, 0.0]),
            torch.stack((largest_finite, torch.tensor(0.0))),
            torch.stack((-largest_finite, torch.tensor(0.0))),
        )
    )
    expected_extremes = torch.tensor(
        [0.5, 0.5, -1.0, -1.5, -torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max],
        dtype=torch.float32,
    )
    cpu_extremes = box_signed_clearance(extreme_points, lower, upper)
    cuda_extremes = box_signed_clearance(
        extreme_points.to(device), lower.to(device), upper.to(device)
    )
    torch.cuda.synchronize(device)
    assert torch.isfinite(cpu_extremes).all()
    assert torch.isfinite(cuda_extremes).all()
    assert torch.equal(cpu_extremes, expected_extremes)
    assert torch.equal(cuda_extremes.cpu(), expected_extremes)
    _assert_box02_forward_budget(cpu_extremes, cuda_extremes)

    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    tiny_box_lower = torch.tensor([0.0], dtype=torch.float32)
    tiny_box_upper = torch.tensor([2.0 * tiny], dtype=torch.float32)
    tiny_midpoint = torch.tensor([[tiny]], dtype=torch.float32)
    tiny_expected = torch.tensor([tiny], dtype=torch.float32)
    cpu_tiny = box_signed_clearance(tiny_midpoint, tiny_box_lower, tiny_box_upper)
    cuda_tiny = box_signed_clearance(
        tiny_midpoint.to(device), tiny_box_lower.to(device), tiny_box_upper.to(device)
    )
    torch.cuda.synchronize(device)
    assert torch.equal(cpu_tiny, tiny_expected)
    assert torch.equal(cuda_tiny.cpu(), tiny_expected)
    _assert_box02_forward_budget(cpu_tiny, cuda_tiny)

    fixed = torch.tensor([0.25, -0.5], dtype=torch.float32)
    fixed_points = torch.tensor([[0.25, -0.5], [0.25, 0.0]], dtype=torch.float32)
    cpu_fixed = box_signed_clearance(fixed_points, fixed, fixed)
    cuda_fixed = box_signed_clearance(
        fixed_points.to(device), fixed.to(device), fixed.to(device)
    )
    empty = box_signed_clearance(
        torch.empty((0, 2), dtype=torch.float32, device=device),
        lower.to(device),
        upper.to(device),
    )
    torch.cuda.synchronize(device)
    assert cpu_fixed.tolist() == [0.0, -0.5]
    assert cuda_fixed.cpu().tolist() == [0.0, -0.5]
    _assert_box02_forward_budget(cpu_fixed, cuda_fixed)
    assert empty.dtype is torch.float32 and empty.shape == (0,) and empty.device == device
