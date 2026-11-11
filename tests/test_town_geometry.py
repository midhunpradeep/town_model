from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from town_model.models.bounded_box import bounded_box_contains
from town_model.models.town_geometry import box_strict_interior
from town_model.models.town_geometry_fixture import TOWN_STRICT_INTERIOR_FIXTURE


def _device(device_name: str) -> torch.device:
    if device_name == "cuda":
        assert torch.cuda.is_available(), "TOWN-01 CUDA acceptance requires a usable CUDA runtime"
        return torch.device("cuda:0")
    return torch.device("cpu")


def _fixture_tensors(device: torch.device = torch.device("cpu")) -> tuple[torch.Tensor, ...]:
    fixture = TOWN_STRICT_INTERIOR_FIXTURE
    return (
        torch.tensor(fixture.points, dtype=torch.float32, device=device),
        torch.tensor(fixture.lower, dtype=torch.float32, device=device),
        torch.tensor(fixture.upper, dtype=torch.float32, device=device),
    )


def _nextafter_probes(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lower = torch.tensor([-1.0, -0.25], dtype=torch.float32, device=device)
    upper = torch.tensor([0.5, 0.25], dtype=torch.float32, device=device)
    negative_infinity = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
    positive_infinity = torch.tensor(float("inf"), dtype=torch.float32, device=device)
    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    probes = torch.stack(
        (
            torch.stack((torch.nextafter(lower[0], negative_infinity), zero)),
            torch.stack((lower[0], zero)),
            torch.stack((torch.nextafter(lower[0], positive_infinity), zero)),
            torch.stack((torch.nextafter(upper[0], negative_infinity), zero)),
            torch.stack((upper[0], zero)),
            torch.stack((torch.nextafter(upper[0], positive_infinity), zero)),
            torch.stack((zero, torch.nextafter(lower[1], negative_infinity))),
            torch.stack((zero, lower[1])),
            torch.stack((zero, torch.nextafter(lower[1], positive_infinity))),
            torch.stack((zero, torch.nextafter(upper[1], negative_infinity))),
            torch.stack((zero, upper[1])),
            torch.stack((zero, torch.nextafter(upper[1], positive_infinity))),
        )
    )
    expected = torch.tensor(
        [False, False, True, True, False, False, False, False, True, True, False, False],
        dtype=torch.bool,
        device=device,
    )
    return probes, lower, upper, expected


def test_fixture_is_immutable_and_contains_independent_literal_expectations() -> None:
    fixture = TOWN_STRICT_INTERIOR_FIXTURE
    assert fixture.lower == (-1.0, -0.25)
    assert fixture.upper == (0.5, 0.25)
    assert fixture.labels == tuple("ABCDEFGHI")
    assert fixture.points == (
        (0.0, 0.0),
        (-1.0, 0.0),
        (0.5, 0.0),
        (0.0, -0.25),
        (0.0, 0.25),
        (-1.0, -0.25),
        (-0.5, 0.125),
        (0.75, 0.0),
        (-1.25, 0.0),
    )
    assert fixture.expected_contains == (
        True, False, False, False, False, False, True, False, False
    )
    with pytest.raises(FrozenInstanceError):
        fixture.lower = (0.0, 0.0)  # type: ignore[misc]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_fixture_results_match_literal_expectations_on_cpu_and_cuda(device_name: str) -> None:
    device = _device(device_name)
    points, lower, upper = _fixture_tensors(device)

    actual = box_strict_interior(points, lower, upper)

    assert actual.dtype is torch.bool
    assert actual.shape == (9,)
    assert actual.device == device
    assert actual.tolist() == list(TOWN_STRICT_INTERIOR_FIXTURE.expected_contains)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_every_face_and_corner_is_excluded(device_name: str) -> None:
    device = _device(device_name)
    lower = torch.tensor([-1.0, -0.25], dtype=torch.float32, device=device)
    upper = torch.tensor([0.5, 0.25], dtype=torch.float32, device=device)
    points = torch.tensor(
        [
            [-1.0, 0.0], [0.5, 0.0], [0.0, -0.25], [0.0, 0.25],
            [-1.0, -0.25], [-1.0, 0.25], [0.5, -0.25], [0.5, 0.25],
        ],
        dtype=torch.float32,
        device=device,
    )

    assert box_strict_interior(points, lower, upper).tolist() == [False] * 8


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_nextafter_neighbors_on_both_sides_of_every_face(device_name: str) -> None:
    device = _device(device_name)
    points, lower, upper, expected = _nextafter_probes(device)

    actual = box_strict_interior(points, lower, upper)

    assert torch.equal(actual, expected)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_ambient_interior_and_boxes_touching_domain_endpoints(device_name: str) -> None:
    device = _device(device_name)
    lower = torch.tensor([-1.0, -1.0], dtype=torch.float32, device=device)
    upper = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    points = torch.tensor(
        [
            [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0],
            [-1.0, -1.0], [1.0, 1.0], [0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    # The domain boundary is not relative interior: every physical face stays excluded.
    assert box_strict_interior(points, lower, upper).tolist() == [False] * 6 + [True]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_degenerate_single_axis_and_fully_collapsed_boxes_have_empty_ambient_interior(
    device_name: str,
) -> None:
    device = _device(device_name)
    points = torch.tensor(
        [[0.0, 0.0], [0.25, -0.25], [-0.5, 0.5]], dtype=torch.float32, device=device
    )
    single_axis_lower = torch.tensor([0.0, -1.0], dtype=torch.float32, device=device)
    single_axis_upper = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)
    collapsed = torch.tensor([0.25, -0.25], dtype=torch.float32, device=device)

    assert not box_strict_interior(points, single_axis_lower, single_axis_upper).any().item()
    assert not box_strict_interior(points, collapsed, collapsed).any().item()


@pytest.mark.parametrize("dimensions", (1, 2, 4))
def test_arbitrary_positive_dimensions_and_point_order(dimensions: int) -> None:
    lower = torch.full((dimensions,), -0.5, dtype=torch.float32)
    upper = torch.full((dimensions,), 0.5, dtype=torch.float32)
    points = torch.stack(
        (
            torch.zeros(dimensions),
            torch.full((dimensions,), 0.25),
            torch.full((dimensions,), 0.5),
            torch.full((dimensions,), -0.75),
        )
    )

    assert box_strict_interior(points, lower, upper).tolist() == [True, True, False, False]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_empty_singleton_and_multiple_batches_preserve_shape_order_and_device(
    device_name: str,
) -> None:
    device = _device(device_name)
    lower = torch.tensor([-1.0, -0.25], dtype=torch.float32, device=device)
    upper = torch.tensor([0.5, 0.25], dtype=torch.float32, device=device)

    empty = box_strict_interior(torch.empty((0, 2), dtype=torch.float32, device=device), lower, upper)
    singleton = box_strict_interior(torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device), lower, upper)
    multiple = box_strict_interior(
        torch.tensor([[0.0, 0.0], [0.5, 0.0], [-0.5, 0.125]], dtype=torch.float32, device=device),
        lower,
        upper,
    )

    assert empty.dtype is torch.bool and empty.shape == (0,) and empty.device == device
    assert singleton.tolist() == [True]
    assert multiple.tolist() == [True, False, True]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_finite_outside_domain_and_extreme_float32_queries_are_false(device_name: str) -> None:
    device = _device(device_name)
    largest = torch.finfo(torch.float32).max
    smallest_positive = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float32, device=device),
        torch.tensor(1.0, dtype=torch.float32, device=device),
    )
    points = torch.stack(
        (
            torch.tensor([-2.0, 0.0], dtype=torch.float32, device=device),
            torch.tensor([2.0, 0.0], dtype=torch.float32, device=device),
            torch.tensor([largest, 0.0], dtype=torch.float32, device=device),
            torch.tensor([-largest, 0.0], dtype=torch.float32, device=device),
            torch.stack((smallest_positive, torch.tensor(0.0, dtype=torch.float32, device=device))),
        )
    )
    lower = torch.tensor([-1.0, -0.25], dtype=torch.float32, device=device)
    upper = torch.tensor([0.5, 0.25], dtype=torch.float32, device=device)

    assert torch.isfinite(points).all().item()
    assert box_strict_interior(points, lower, upper).tolist() == [False, False, False, False, True]


def test_strict_membership_is_a_subset_of_public_closed_box_membership() -> None:
    points, lower, upper = _fixture_tensors()

    strict = box_strict_interior(points, lower, upper)
    closed = bounded_box_contains(points, lower, upper)

    assert torch.all(~strict | closed).item()
    assert strict.tolist() == [True, False, False, False, False, False, True, False, False]
    assert closed.tolist() == [True, True, True, True, True, True, True, False, False]
    assert not torch.equal(strict, closed)


def test_validation_error_categories_match_public_box_evaluator() -> None:
    valid = [
        torch.zeros((1, 2), dtype=torch.float32),
        torch.tensor([-1.0, -0.25], dtype=torch.float32),
        torch.tensor([0.5, 0.25], dtype=torch.float32),
    ]
    invalid_cases: list[tuple[object, object, object]] = [
        (None, valid[1], valid[2]),
        (valid[0].to(torch.float64), valid[1], valid[2]),
        (torch.zeros((1, 0), dtype=torch.float32), torch.empty((0,), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)),
        (valid[0], torch.tensor([-1.01, -0.25]), valid[2]),
        (valid[0], torch.tensor([0.6, -0.25]), valid[2]),
        (torch.tensor([[float("nan"), 0.0]], dtype=torch.float32), valid[1], valid[2]),
    ]

    for inputs in invalid_cases:
        with pytest.raises((TypeError, ValueError)) as town_error:
            box_strict_interior(*inputs)  # type: ignore[arg-type]
        with pytest.raises(type(town_error.value)):
            bounded_box_contains(*inputs)  # type: ignore[arg-type]


def test_rejects_invalid_rank_dimension_and_exact_shape() -> None:
    valid_points = torch.zeros((2, 2), dtype=torch.float32)
    lower = torch.tensor([-1.0, -0.25], dtype=torch.float32)
    upper = torch.tensor([0.5, 0.25], dtype=torch.float32)
    invalid_cases = (
        (torch.empty((0, 0), dtype=torch.float32), torch.empty((0,), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)),
        (torch.zeros((2,), dtype=torch.float32), lower, upper),
        (torch.zeros((2, 2, 1), dtype=torch.float32), lower, upper),
        (valid_points, torch.tensor([-1.0]), upper),
        (valid_points, lower, torch.tensor([0.25])),
        (valid_points, lower.unsqueeze(0), upper),
    )
    for points, bad_lower, bad_upper in invalid_cases:
        with pytest.raises(ValueError):
            box_strict_interior(points, bad_lower, bad_upper)


def test_rejects_non_tensor_and_wrong_dtype_inputs() -> None:
    valid = [
        torch.zeros((1, 2), dtype=torch.float32),
        torch.tensor([-1.0, -0.25], dtype=torch.float32),
        torch.tensor([0.5, 0.25], dtype=torch.float32),
    ]
    for index in range(3):
        non_tensor = list(valid)
        non_tensor[index] = ((0.0, 0.0),)
        with pytest.raises(TypeError):
            box_strict_interior(*non_tensor)  # type: ignore[arg-type]
        wrong_dtype = list(valid)
        wrong_dtype[index] = wrong_dtype[index].to(torch.float64)
        with pytest.raises(TypeError):
            box_strict_interior(*wrong_dtype)  # type: ignore[arg-type]


@pytest.mark.parametrize("argument_index", (0, 1, 2))
def test_rejects_unsupported_tensor_layout_for_every_input(argument_index: int) -> None:
    valid = [
        torch.zeros((1, 2), dtype=torch.float32),
        torch.tensor([-1.0, -0.25], dtype=torch.float32),
        torch.tensor([0.5, 0.25], dtype=torch.float32),
    ]
    shape = (1, 2) if argument_index == 0 else (2,)
    indices = torch.zeros((len(shape), 1), dtype=torch.int64)
    sparse = torch.sparse_coo_tensor(indices, torch.tensor([0.0]), shape)
    valid[argument_index] = sparse
    with pytest.raises(TypeError):
        box_strict_interior(*valid)


@pytest.mark.parametrize(
    ("argument_index", "values"),
    (
        (0, [[float("nan"), 0.0]]),
        (0, [[float("inf"), 0.0]]),
        (1, [float("nan"), -0.25]),
        (1, [float("-inf"), -0.25]),
        (2, [0.5, float("nan")]),
        (2, [0.5, float("inf")]),
    ),
)
def test_rejects_nonfinite_points_and_corners(argument_index: int, values: list[object]) -> None:
    inputs: list[torch.Tensor] = [
        torch.zeros((1, 2), dtype=torch.float32),
        torch.tensor([-1.0, -0.25], dtype=torch.float32),
        torch.tensor([0.5, 0.25], dtype=torch.float32),
    ]
    shape = (1, 2) if argument_index == 0 else (2,)
    inputs[argument_index] = torch.tensor(values, dtype=torch.float32).reshape(shape)
    with pytest.raises(ValueError, match="finite"):
        box_strict_interior(*inputs)


@pytest.mark.parametrize(
    ("lower_values", "upper_values"),
    (
        ((-1.01, -0.25), (0.5, 0.25)),
        ((-1.0, -0.25), (0.5, 1.01)),
        ((0.25, -0.25), (-0.25, 0.25)),
    ),
)
def test_rejects_corners_outside_domain_or_out_of_order(
    lower_values: tuple[float, float], upper_values: tuple[float, float]
) -> None:
    with pytest.raises(ValueError):
        box_strict_interior(
            torch.zeros((1, 2), dtype=torch.float32),
            torch.tensor(lower_values, dtype=torch.float32),
            torch.tensor(upper_values, dtype=torch.float32),
        )


def test_validation_error_categories_match_box01_for_layout_and_mixed_devices() -> None:
    sparse_points = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]), torch.tensor([0.0]), (1, 2)
    )
    lower = torch.tensor([-1.0, -0.25], dtype=torch.float32)
    upper = torch.tensor([0.5, 0.25], dtype=torch.float32)
    with pytest.raises(TypeError) as town_layout_error:
        box_strict_interior(sparse_points, lower, upper)
    with pytest.raises(type(town_layout_error.value)):
        bounded_box_contains(sparse_points, lower, upper)

    device = _device("cuda")
    points = torch.zeros((1, 2), dtype=torch.float32, device=device)
    cuda_lower = lower.to(device)
    cpu_upper = upper
    with pytest.raises(ValueError, match="same device") as town_device_error:
        box_strict_interior(points, cuda_lower, cpu_upper)
    with pytest.raises(type(town_device_error.value), match="same device"):
        bounded_box_contains(points, cuda_lower, cpu_upper)


def test_rejects_mixed_cpu_cuda_inputs_and_unsupported_common_device() -> None:
    device = _device("cuda")
    points = torch.zeros((1, 2), dtype=torch.float32, device=device)
    lower = torch.tensor([-1.0, -0.25], dtype=torch.float32)
    upper = torch.tensor([0.5, 0.25], dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="same device"):
        box_strict_interior(points, lower, upper)

    meta = torch.empty((1, 2), dtype=torch.float32, device="meta")
    meta_corner = torch.empty((2,), dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="supported CPU or CUDA"):
        box_strict_interior(meta, meta_corner, meta_corner)


def test_valid_and_invalid_calls_preserve_all_input_values() -> None:
    points, lower, upper = _fixture_tensors()
    originals = tuple(value.clone() for value in (points, lower, upper))
    result = box_strict_interior(points, lower, upper)
    assert result.tolist() == list(TOWN_STRICT_INTERIOR_FIXTURE.expected_contains)
    assert all(torch.equal(value, original) for value, original in zip((points, lower, upper), originals, strict=True))

    invalid_upper = upper.clone()
    invalid_upper[0] = 2.0
    before_invalid = invalid_upper.clone()
    with pytest.raises(ValueError):
        box_strict_interior(points, lower, invalid_upper)
    assert torch.equal(points, originals[0])
    assert torch.equal(lower, originals[1])
    assert torch.equal(invalid_upper, before_invalid)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_cuda_and_cpu_agree_exactly_for_fixture_boundaries_degeneracy_and_empty(
    device_name: str,
) -> None:
    device = _device(device_name)
    cpu_points, cpu_lower, cpu_upper = _fixture_tensors()
    points, lower, upper = _fixture_tensors(device)
    actual = box_strict_interior(points, lower, upper)
    expected = box_strict_interior(cpu_points, cpu_lower, cpu_upper)
    assert torch.equal(actual.cpu(), expected)

    probe_cpu_points, probe_cpu_lower, probe_cpu_upper, _ = _nextafter_probes(
        torch.device("cpu")
    )
    probe_points = probe_cpu_points.to(device)
    probe_lower = probe_cpu_lower.to(device)
    probe_upper = probe_cpu_upper.to(device)
    assert torch.equal(
        box_strict_interior(probe_points, probe_lower, probe_upper).cpu(),
        box_strict_interior(probe_cpu_points, probe_cpu_lower, probe_cpu_upper),
    )

    collapsed = torch.tensor([0.25, -0.25], dtype=torch.float32, device=device)
    collapsed_points = torch.tensor([[0.25, -0.25], [0.25, 0.0]], dtype=torch.float32, device=device)
    collapsed_result = box_strict_interior(collapsed_points, collapsed, collapsed)
    empty_result = box_strict_interior(torch.empty((0, 2), dtype=torch.float32, device=device), probe_lower, probe_upper)
    assert torch.equal(collapsed_result.cpu(), torch.tensor([False, False]))
    assert empty_result.shape == (0,) and empty_result.dtype is torch.bool
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def test_cuda_runtime_and_device_are_available() -> None:
    device = _device("cuda")
    assert torch.version.cuda is not None
    assert torch.cuda.device_count() >= 1
    assert torch.cuda.get_device_name(device)
    probe = torch.tensor([1.0], dtype=torch.float32, device=device)
    assert probe.item() == 1.0
    torch.cuda.synchronize(device)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_invalid_corner_is_rejected_on_cpu_and_cuda(device_name: str) -> None:
    device = _device(device_name)
    with pytest.raises(ValueError, match="domain"):
        box_strict_interior(
            torch.zeros((0, 2), dtype=torch.float32, device=device),
            torch.tensor([-1.0, -0.25], dtype=torch.float32, device=device),
            torch.tensor([0.5, 1.01], dtype=torch.float32, device=device),
        )


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_invalid_nonfinite_query_is_rejected_on_cpu_and_cuda(device_name: str) -> None:
    device = _device(device_name)
    with pytest.raises(ValueError, match="finite"):
        box_strict_interior(
            torch.tensor([[float("nan"), 0.0]], dtype=torch.float32, device=device),
            torch.tensor([-1.0, -0.25], dtype=torch.float32, device=device),
            torch.tensor([0.5, 0.25], dtype=torch.float32, device=device),
        )
