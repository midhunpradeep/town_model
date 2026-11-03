from __future__ import annotations

from itertools import product

import pytest
import torch

from town_model.models.bounded_box import bounded_box_contains
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
