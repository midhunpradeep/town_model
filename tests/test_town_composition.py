from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import Tensor

from town_model.models.bounded_box import bounded_box_contains
from town_model.models.town_composition_fixture import (
    DEGENERATE_FRAME_FIXTURE,
    DEGENERATE_WINDOW_FIXTURES,
    TOWN_COMPOSITION_FIXTURES,
    TownCompositionFixture,
)
from town_model.models.town_geometry import box_strict_interior, house_contains, town_contains


def _device(device_name: str) -> torch.device:
    if device_name == "cuda":
        assert torch.cuda.is_available(), "TOWN-02 CUDA acceptance requires a usable CUDA runtime"
        return torch.device("cuda:0")
    return torch.device("cpu")


def _house_tensors(
    fixture: TownCompositionFixture, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert len(fixture.frame_lower) == 1
    dimensions = len(fixture.frame_lower[0])
    window_count = len(fixture.window_lower[0])
    points = torch.tensor(fixture.points, dtype=torch.float32, device=device)
    frame_lower = torch.tensor(fixture.frame_lower[0], dtype=torch.float32, device=device)
    frame_upper = torch.tensor(fixture.frame_upper[0], dtype=torch.float32, device=device)
    if window_count:
        window_lower = torch.tensor(fixture.window_lower[0], dtype=torch.float32, device=device)
        window_upper = torch.tensor(fixture.window_upper[0], dtype=torch.float32, device=device)
    else:
        window_lower = torch.empty((0, dimensions), dtype=torch.float32, device=device)
        window_upper = torch.empty((0, dimensions), dtype=torch.float32, device=device)
    return points, frame_lower, frame_upper, window_lower, window_upper


def _house_oracle(
    points: Tensor,
    frame_lower: Tensor,
    frame_upper: Tensor,
    window_lower: Tensor,
    window_upper: Tensor,
) -> Tensor:
    """Compose the independently accepted public closed/open primitives."""
    in_frame = bounded_box_contains(points, frame_lower, frame_upper)
    excluded = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    for index in range(window_lower.shape[0]):
        excluded |= box_strict_interior(points, window_lower[index], window_upper[index])
    return in_frame & ~excluded


def _valid_house_inputs(device: torch.device = torch.device("cpu")) -> list[Tensor]:
    return [
        torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device),
        torch.tensor([-1.0, -1.0], dtype=torch.float32, device=device),
        torch.tensor([1.0, 1.0], dtype=torch.float32, device=device),
        torch.tensor([[-0.5, -0.5]], dtype=torch.float32, device=device),
        torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device),
    ]


def test_primary_fixtures_are_immutable_and_store_literal_results() -> None:
    f1, f2, f3, f4, f5 = TOWN_COMPOSITION_FIXTURES
    assert f1.expected_contains == (False, True, True, True, False)
    assert f2.expected_contains == (False, False, True, True, False, True, True, False)
    assert f3.expected_contains == (False, False, True, True, False, True)
    assert f4.expected_contains == (True, True, False, False)
    assert f5.expected_contains == (True, False, False, True, True, False)
    assert f1.points[1] == (0.5, 0.0)
    assert f2.points[1] == (0.5, 0.0)
    assert f2.points[2] == (0.5, 0.25)
    assert f3.window_upper[0][0] == (1.0, 0.25)
    assert f4.window_lower == ((),) and f4.window_upper == ((),)
    with pytest.raises(FrozenInstanceError):
        f1.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
@pytest.mark.parametrize("fixture", TOWN_COMPOSITION_FIXTURES[:4], ids=lambda fixture: fixture.name)
def test_literal_single_house_fixtures_match_exact_boolean_results(
    device_name: str, fixture: TownCompositionFixture
) -> None:
    device = _device(device_name)
    points, frame_lower, frame_upper, window_lower, window_upper = _house_tensors(
        fixture, device
    )
    actual = house_contains(points, frame_lower, frame_upper, window_lower, window_upper)

    assert actual.dtype is torch.bool
    assert actual.shape == (len(fixture.points),)
    assert actual.device == device
    assert actual.tolist() == list(fixture.expected_contains)
    assert torch.equal(
        actual,
        _house_oracle(points, frame_lower, frame_upper, window_lower, window_upper),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
@pytest.mark.parametrize("fixture", DEGENERATE_WINDOW_FIXTURES, ids=lambda fixture: fixture.name)
def test_degenerate_fixed_windows_have_empty_ambient_interior(
    device_name: str, fixture: TownCompositionFixture
) -> None:
    device = _device(device_name)
    points, frame_lower, frame_upper, window_lower, window_upper = _house_tensors(
        fixture, device
    )
    actual = house_contains(points, frame_lower, frame_upper, window_lower, window_upper)
    assert actual.tolist() == list(fixture.expected_contains)
    assert torch.equal(
        actual,
        _house_oracle(points, frame_lower, frame_upper, window_lower, window_upper),
    )


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_degenerate_fixed_frame_retains_its_closed_lower_dimensional_set(
    device_name: str,
) -> None:
    device = _device(device_name)
    fixture = DEGENERATE_FRAME_FIXTURE
    points, frame_lower, frame_upper, window_lower, window_upper = _house_tensors(
        fixture, device
    )
    actual = house_contains(points, frame_lower, frame_upper, window_lower, window_upper)
    assert actual.tolist() == list(fixture.expected_contains)
    assert torch.equal(
        actual,
        _house_oracle(points, frame_lower, frame_upper, window_lower, window_upper),
    )


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_nested_coincident_disjoint_and_permuted_windows_preserve_union_semantics(
    device_name: str,
) -> None:
    device = _device(device_name)
    points = torch.tensor(
        [
            [-0.75, 0.0], [-0.5, 0.0], [-0.25, 0.0], [0.0, 0.0],
            [0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0],
            [0.0, 0.5], [0.0, -0.5],
        ],
        dtype=torch.float32,
        device=device,
    )
    frame_lower = torch.tensor([-1.0, -1.0], dtype=torch.float32, device=device)
    frame_upper = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    outer_lower = torch.tensor([-0.5, -0.5], dtype=torch.float32, device=device)
    outer_upper = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)
    inner_lower = torch.tensor([-0.25, -0.25], dtype=torch.float32, device=device)
    inner_upper = torch.tensor([0.25, 0.25], dtype=torch.float32, device=device)
    other_lower = torch.tensor([0.5, -0.25], dtype=torch.float32, device=device)
    other_upper = torch.tensor([0.75, 0.25], dtype=torch.float32, device=device)

    no_windows = torch.empty((0, 2), dtype=torch.float32, device=device)
    baseline = house_contains(points, frame_lower, frame_upper, no_windows, no_windows)
    outer = house_contains(
        points, frame_lower, frame_upper, outer_lower.unsqueeze(0), outer_upper.unsqueeze(0)
    )
    nested = house_contains(
        points,
        frame_lower,
        frame_upper,
        torch.stack((outer_lower, inner_lower)),
        torch.stack((outer_upper, inner_upper)),
    )
    coincident = house_contains(
        points,
        frame_lower,
        frame_upper,
        torch.stack((outer_lower, outer_lower)),
        torch.stack((outer_upper, outer_upper)),
    )
    disjoint_and_overlapping = house_contains(
        points,
        frame_lower,
        frame_upper,
        torch.stack((outer_lower, other_lower)),
        torch.stack((outer_upper, other_upper)),
    )
    permuted = house_contains(
        points,
        frame_lower,
        frame_upper,
        torch.stack((other_lower, outer_lower)),
        torch.stack((other_upper, outer_upper)),
    )

    assert torch.all(~outer | baseline).item()
    assert torch.equal(outer, nested)
    assert torch.equal(outer, coincident)
    assert torch.equal(disjoint_and_overlapping, permuted)
    assert torch.equal(
        disjoint_and_overlapping,
        _house_oracle(
            points,
            frame_lower,
            frame_upper,
            torch.stack((outer_lower, other_lower)),
            torch.stack((outer_upper, other_upper)),
        ),
    )


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_every_house_member_is_outside_each_excluded_strict_window(device_name: str) -> None:
    device = _device(device_name)
    fixture = TOWN_COMPOSITION_FIXTURES[1]
    points, frame_lower, frame_upper, window_lower, window_upper = _house_tensors(
        fixture, device
    )
    members = house_contains(points, frame_lower, frame_upper, window_lower, window_upper)
    for index in range(window_lower.shape[0]):
        excluded = box_strict_interior(points, window_lower[index], window_upper[index])
        assert not torch.any(members & excluded).item()


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_protruding_window_does_not_extend_membership_and_can_empty_a_house(
    device_name: str,
) -> None:
    device = _device(device_name)
    points = torch.tensor(
        [
            [-0.5, 0.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.5],
            [-0.5, -0.5], [0.5, 0.5],
        ],
        dtype=torch.float32,
        device=device,
    )
    frame_lower = torch.tensor([-0.5, -0.5], dtype=torch.float32, device=device)
    frame_upper = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)
    no_windows = torch.empty((0, 2), dtype=torch.float32, device=device)
    closed_frame = house_contains(points, frame_lower, frame_upper, no_windows, no_windows)
    covering_lower = torch.tensor([-0.75, -0.75], dtype=torch.float32, device=device)
    covering_upper = torch.tensor([0.75, 0.75], dtype=torch.float32, device=device)
    empty_house = house_contains(
        points,
        frame_lower,
        frame_upper,
        covering_lower.unsqueeze(0),
        covering_upper.unsqueeze(0),
    )

    assert closed_frame.tolist() == [True, True, True, True, True, True]
    assert empty_house.tolist() == [False] * len(points)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_window_and_frame_faces_use_direct_comparisons_at_nextafter_neighbors(
    device_name: str,
) -> None:
    device = _device(device_name)
    negative_infinity = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
    positive_infinity = torch.tensor(float("inf"), dtype=torch.float32, device=device)
    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    window_lower = torch.tensor(-0.5, dtype=torch.float32, device=device)
    window_upper = torch.tensor(0.5, dtype=torch.float32, device=device)
    frame_lower = torch.tensor(-1.0, dtype=torch.float32, device=device)
    frame_upper = torch.tensor(1.0, dtype=torch.float32, device=device)

    def point(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.stack((x, y))

    points = torch.stack(
        (
            point(window_lower, zero),
            point(torch.nextafter(window_lower, positive_infinity), zero),
            point(torch.nextafter(window_lower, negative_infinity), zero),
            point(window_upper, zero),
            point(torch.nextafter(window_upper, negative_infinity), zero),
            point(torch.nextafter(window_upper, positive_infinity), zero),
            point(zero, window_lower),
            point(zero, torch.nextafter(window_lower, positive_infinity)),
            point(zero, torch.nextafter(window_lower, negative_infinity)),
            point(zero, window_upper),
            point(zero, torch.nextafter(window_upper, negative_infinity)),
            point(zero, torch.nextafter(window_upper, positive_infinity)),
            point(frame_lower, zero),
            point(torch.nextafter(frame_lower, positive_infinity), zero),
            point(torch.nextafter(frame_lower, negative_infinity), zero),
            point(frame_upper, zero),
            point(torch.nextafter(frame_upper, negative_infinity), zero),
            point(torch.nextafter(frame_upper, positive_infinity), zero),
            point(zero, frame_lower),
            point(zero, torch.nextafter(frame_lower, positive_infinity)),
            point(zero, torch.nextafter(frame_lower, negative_infinity)),
            point(zero, frame_upper),
            point(zero, torch.nextafter(frame_upper, negative_infinity)),
            point(zero, torch.nextafter(frame_upper, positive_infinity)),
        )
    )
    actual = house_contains(
        points,
        torch.tensor([-1.0, -1.0], dtype=torch.float32, device=device),
        torch.tensor([1.0, 1.0], dtype=torch.float32, device=device),
        torch.tensor([-0.5, -0.5], dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor([0.5, 0.5], dtype=torch.float32, device=device).unsqueeze(0),
    )
    assert actual.tolist() == [
        True, False, True, True, False, True,
        True, False, True, True, False, True,
        True, True, False, True, True, False,
        True, True, False, True, True, False,
    ]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_zero_windows_house_is_exactly_closed_box_membership(device_name: str) -> None:
    device = _device(device_name)
    fixture = TOWN_COMPOSITION_FIXTURES[3]
    points, frame_lower, frame_upper, window_lower, window_upper = _house_tensors(
        fixture, device
    )
    actual = house_contains(points, frame_lower, frame_upper, window_lower, window_upper)
    expected = bounded_box_contains(points, frame_lower, frame_upper)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
@pytest.mark.parametrize("dimensions", (1, 2, 4))
def test_empty_singleton_and_multiple_batches_work_for_arbitrary_dimensions(
    device_name: str, dimensions: int
) -> None:
    device = _device(device_name)
    frame_lower = torch.full((dimensions,), -0.5, dtype=torch.float32, device=device)
    frame_upper = torch.full((dimensions,), 0.5, dtype=torch.float32, device=device)
    window_lower = torch.empty((0, dimensions), dtype=torch.float32, device=device)
    window_upper = torch.empty((0, dimensions), dtype=torch.float32, device=device)
    empty = house_contains(
        torch.empty((0, dimensions), dtype=torch.float32, device=device),
        frame_lower,
        frame_upper,
        window_lower,
        window_upper,
    )
    singleton = house_contains(
        torch.zeros((1, dimensions), dtype=torch.float32, device=device),
        frame_lower,
        frame_upper,
        window_lower,
        window_upper,
    )
    points = torch.stack(
        (
            torch.zeros((dimensions,), dtype=torch.float32, device=device),
            frame_upper,
            torch.full((dimensions,), 0.75, dtype=torch.float32, device=device),
        )
    )
    multiple = house_contains(points, frame_lower, frame_upper, window_lower, window_upper)

    assert empty.dtype is torch.bool and empty.shape == (0,) and empty.device == device
    assert singleton.tolist() == [True]
    assert multiple.tolist() == [True, True, False]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_finite_extreme_outside_domain_queries_are_valid_and_false(device_name: str) -> None:
    device = _device(device_name)
    largest = torch.finfo(torch.float32).max
    points = torch.tensor(
        [[-2.0, 0.0], [2.0, 0.0], [-largest, 0.0], [largest, 0.0], [0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    no_windows = torch.empty((0, 2), dtype=torch.float32, device=device)
    actual = house_contains(
        points,
        torch.tensor([-1.0, -1.0], dtype=torch.float32, device=device),
        torch.tensor([1.0, 1.0], dtype=torch.float32, device=device),
        no_windows,
        no_windows,
    )
    assert torch.isfinite(points).all().item()
    assert actual.tolist() == [False, False, False, False, True]


@pytest.mark.parametrize("argument_index", range(5))
def test_rejects_non_tensor_wrong_dtype_and_unsupported_layout_per_argument(
    argument_index: int,
) -> None:
    valid = _valid_house_inputs()
    non_tensor = list(valid)
    non_tensor[argument_index] = None  # type: ignore[assignment]
    with pytest.raises(TypeError):
        house_contains(*non_tensor)  # type: ignore[arg-type]

    wrong_dtype = list(valid)
    wrong_dtype[argument_index] = wrong_dtype[argument_index].to(torch.float64)
    with pytest.raises(TypeError):
        house_contains(*wrong_dtype)

    sparse_inputs = list(valid)
    shape = tuple(sparse_inputs[argument_index].shape)
    indices = torch.empty((len(shape), 0), dtype=torch.int64)
    sparse_inputs[argument_index] = torch.sparse_coo_tensor(
        indices, torch.empty((0,), dtype=torch.float32), size=shape
    )
    with pytest.raises(TypeError):
        house_contains(*sparse_inputs)


@pytest.mark.parametrize(
    "inputs",
    (
        (torch.empty((0, 0)), torch.empty((0,)), torch.empty((0,)), torch.empty((0, 0)), torch.empty((0, 0))),
        (torch.zeros((2,)), torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]), torch.empty((0, 2)), torch.empty((0, 2))),
        (torch.zeros((1, 2, 1)), torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]), torch.empty((0, 2)), torch.empty((0, 2))),
        (torch.zeros((1, 2)), torch.tensor([-1.0]), torch.tensor([1.0, 1.0]), torch.empty((0, 2)), torch.empty((0, 2))),
        (torch.zeros((1, 2)), torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]), torch.empty((0,)), torch.empty((0, 2))),
        (torch.zeros((1, 2)), torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]), torch.empty((0, 2)), torch.empty((1, 2))),
        (torch.zeros((1, 2)), torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]), torch.empty((1, 3)), torch.empty((1, 3))),
    ),
)
def test_rejects_invalid_rank_dimension_and_exact_shape(inputs: tuple[Tensor, ...]) -> None:
    with pytest.raises(ValueError):
        house_contains(*inputs)


def test_malformed_zero_window_tensors_reject_even_for_empty_point_batches() -> None:
    points = torch.empty((0, 2), dtype=torch.float32)
    frame_lower = torch.tensor([-1.0, -1.0], dtype=torch.float32)
    frame_upper = torch.tensor([1.0, 1.0], dtype=torch.float32)
    valid_empty = torch.empty((0, 2), dtype=torch.float32)
    with pytest.raises(ValueError):
        house_contains(points, frame_lower, frame_upper, torch.empty((0,)), valid_empty)
    with pytest.raises(ValueError):
        house_contains(points, frame_lower, frame_upper, valid_empty, torch.empty((0,)))
    with pytest.raises(TypeError):
        house_contains(points, frame_lower, frame_upper, None, valid_empty)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("argument_index", "values", "shape", "message"),
    (
        (0, [[float("nan"), 0.0]], (1, 2), "finite"),
        (0, [[float("inf"), 0.0]], (1, 2), "finite"),
        (1, [float("nan"), -1.0], (2,), "finite"),
        (2, [1.0, float("inf")], (2,), "finite"),
        (3, [[float("nan"), 0.0]], (1, 2), "finite"),
        (4, [[0.0, float("inf")]], (1, 2), "finite"),
        (1, [-1.01, -1.0], (2,), "domain"),
        (2, [1.0, 1.01], (2,), "domain"),
        (3, [[-1.01, -0.5]], (1, 2), "domain"),
        (4, [[0.5, 1.01]], (1, 2), "domain"),
    ),
)
def test_rejects_nonfinite_or_out_of_domain_coordinates(
    argument_index: int, values: list[object], shape: tuple[int, ...], message: str
) -> None:
    inputs = _valid_house_inputs()
    inputs[argument_index] = torch.tensor(values, dtype=torch.float32).reshape(shape)
    with pytest.raises(ValueError, match=message):
        house_contains(*inputs)


@pytest.mark.parametrize("argument_index", (1, 2, 3, 4))
def test_rejects_reversed_frame_or_window_corners(argument_index: int) -> None:
    inputs = _valid_house_inputs()
    if argument_index == 1:
        inputs[1] = torch.tensor([0.25, -1.0], dtype=torch.float32)
        inputs[2] = torch.tensor([-0.25, 1.0], dtype=torch.float32)
    elif argument_index == 2:
        inputs[1] = torch.tensor([0.25, -1.0], dtype=torch.float32)
        inputs[2] = torch.tensor([-0.25, 1.0], dtype=torch.float32)
    elif argument_index == 3:
        inputs[3] = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
        inputs[4] = torch.tensor([[-0.25, 0.5]], dtype=torch.float32)
    else:
        inputs[3] = torch.tensor([[-0.5, -0.5]], dtype=torch.float32)
        inputs[4] = torch.tensor([[-0.75, 0.5]], dtype=torch.float32)
    with pytest.raises(ValueError, match="less than or equal"):
        house_contains(*inputs)


def test_validates_geometry_with_empty_points_and_accepts_noncontiguous_strided_points() -> None:
    valid = _valid_house_inputs()
    empty_points = torch.empty((0, 2), dtype=torch.float32)
    invalid_frame_lower = torch.tensor([-1.01, -1.0], dtype=torch.float32)
    with pytest.raises(ValueError):
        house_contains(empty_points, invalid_frame_lower, valid[2], valid[3], valid[4])

    noncontiguous = torch.zeros((2, 3), dtype=torch.float32).transpose(0, 1)
    assert noncontiguous.shape == (3, 2) and not noncontiguous.is_contiguous()
    result = house_contains(noncontiguous, valid[1], valid[2], valid[3], valid[4])
    assert result.tolist() == [False, False, False]


def test_rejects_mixed_devices_and_unsupported_device_type() -> None:
    if torch.cuda.is_available():
        inputs = _valid_house_inputs()
        inputs[3] = inputs[3].cuda()
        with pytest.raises(ValueError, match="same device"):
            house_contains(*inputs)

    meta_inputs = [
        torch.empty((1, 2), dtype=torch.float32, device="meta"),
        torch.empty((2,), dtype=torch.float32, device="meta"),
        torch.empty((2,), dtype=torch.float32, device="meta"),
        torch.empty((0, 2), dtype=torch.float32, device="meta"),
        torch.empty((0, 2), dtype=torch.float32, device="meta"),
    ]
    with pytest.raises(ValueError, match="CPU or CUDA"):
        house_contains(*meta_inputs)


def test_house_evaluation_does_not_mutate_any_input() -> None:
    inputs = _valid_house_inputs()
    before = [value.clone() for value in inputs]
    house_contains(*inputs)
    for actual, expected in zip(inputs, before, strict=True):
        assert torch.equal(actual, expected)


def _town_tensors(
    fixture: TownCompositionFixture, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    house_count = len(fixture.frame_lower)
    dimensions = len(fixture.frame_lower[0])
    window_count = len(fixture.window_lower[0])
    assert all(len(windows) == window_count for windows in fixture.window_lower)
    assert all(len(windows) == window_count for windows in fixture.window_upper)
    points = torch.tensor(fixture.points, dtype=torch.float32, device=device)
    frame_lower = torch.tensor(fixture.frame_lower, dtype=torch.float32, device=device)
    frame_upper = torch.tensor(fixture.frame_upper, dtype=torch.float32, device=device)
    if window_count:
        window_lower = torch.tensor(fixture.window_lower, dtype=torch.float32, device=device)
        window_upper = torch.tensor(fixture.window_upper, dtype=torch.float32, device=device)
    else:
        window_lower = torch.empty(
            (house_count, 0, dimensions), dtype=torch.float32, device=device
        )
        window_upper = torch.empty_like(window_lower)
    return points, frame_lower, frame_upper, window_lower, window_upper


def _town_oracle(
    points: Tensor,
    frame_lower: Tensor,
    frame_upper: Tensor,
    window_lower: Tensor,
    window_upper: Tensor,
) -> Tensor:
    """Compose accepted public single-box primitives with explicit loops."""
    house_results = [
        _house_oracle(
            points,
            frame_lower[house],
            frame_upper[house],
            window_lower[house],
            window_upper[house],
        )
        for house in range(frame_lower.shape[0])
    ]
    return torch.stack(house_results, dim=0).any(dim=0)


def _valid_town_inputs(device: torch.device = torch.device("cpu")) -> list[Tensor]:
    return [
        torch.tensor([[0.0, 0.0], [0.75, 0.0]], dtype=torch.float32, device=device),
        torch.tensor([[-1.0, -1.0], [0.0, -1.0]], dtype=torch.float32, device=device),
        torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float32, device=device),
        torch.tensor([[[-0.5, -0.5]], [[0.5, -0.5]]], dtype=torch.float32, device=device),
        torch.tensor([[[0.0, 0.5]], [[0.75, 0.5]]], dtype=torch.float32, device=device),
    ]


def _generated_geometry(dimensions: int, device: torch.device) -> tuple[Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(20261111 + dimensions)
    points = torch.rand((37, dimensions), generator=generator, dtype=torch.float32) * 2.5 - 1.25
    frame_corners_a = torch.rand((3, dimensions), generator=generator) * 2.0 - 1.0
    frame_corners_b = torch.rand((3, dimensions), generator=generator) * 2.0 - 1.0
    frame_lower = torch.minimum(frame_corners_a, frame_corners_b)
    frame_upper = torch.maximum(frame_corners_a, frame_corners_b)
    window_corners_a = torch.rand((3, 4, dimensions), generator=generator) * 2.0 - 1.0
    window_corners_b = torch.rand((3, 4, dimensions), generator=generator) * 2.0 - 1.0
    window_lower = torch.minimum(window_corners_a, window_corners_b)
    window_upper = torch.maximum(window_corners_a, window_corners_b)
    return tuple(
        tensor.to(device)
        for tensor in (points, frame_lower, frame_upper, window_lower, window_upper)
    )


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
@pytest.mark.parametrize("fixture", TOWN_COMPOSITION_FIXTURES, ids=lambda fixture: fixture.name)
def test_all_literal_house_and_town_fixtures_match_expected_values(
    device_name: str, fixture: TownCompositionFixture
) -> None:
    device = _device(device_name)
    points, frame_lower, frame_upper, window_lower, window_upper = _town_tensors(
        fixture, device
    )
    actual = town_contains(points, frame_lower, frame_upper, window_lower, window_upper)

    assert actual.dtype is torch.bool
    assert actual.shape == (len(fixture.points),)
    assert actual.device == device
    assert actual.tolist() == list(fixture.expected_contains)
    assert torch.equal(
        actual,
        _town_oracle(points, frame_lower, frame_upper, window_lower, window_upper),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
@pytest.mark.parametrize("dimensions", (1, 2, 4))
def test_vectorized_town_matches_independent_oracle_on_generated_inputs(
    device_name: str, dimensions: int
) -> None:
    device = _device(device_name)
    inputs = _generated_geometry(dimensions, device)
    points, frame_lower, frame_upper, window_lower, window_upper = inputs

    actual = town_contains(*inputs)

    assert actual.shape == (points.shape[0],)
    assert actual.dtype is torch.bool and actual.device == device
    assert torch.equal(
        actual,
        _town_oracle(points, frame_lower, frame_upper, window_lower, window_upper),
    )


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_empty_window_axis_town_is_exactly_box01_for_one_house(device_name: str) -> None:
    device = _device(device_name)
    fixture = TOWN_COMPOSITION_FIXTURES[3]
    points, frame_lower, frame_upper, window_lower, window_upper = _town_tensors(
        fixture, device
    )
    actual = town_contains(points, frame_lower, frame_upper, window_lower, window_upper)
    expected = bounded_box_contains(points, frame_lower[0], frame_upper[0])
    assert torch.equal(actual, expected)
    assert actual.tolist() == list(fixture.expected_contains)

    generated = _generated_geometry(4, device)
    points, frame_lower, frame_upper, _, _ = generated
    no_windows = torch.empty((3, 0, 4), dtype=torch.float32, device=device)
    multi_house = town_contains(points, frame_lower, frame_upper, no_windows, no_windows)
    per_house = torch.stack(
        [bounded_box_contains(points, frame_lower[h], frame_upper[h]) for h in range(3)]
    ).any(dim=0)
    assert torch.equal(multi_house, per_house)


def test_town_oracle_and_required_set_invariants_on_generated_geometry() -> None:
    points, frame_lower, frame_upper, window_lower, window_upper = _generated_geometry(
        3, torch.device("cpu")
    )
    town = town_contains(points, frame_lower, frame_upper, window_lower, window_upper)
    oracle = _town_oracle(points, frame_lower, frame_upper, window_lower, window_upper)
    assert torch.equal(town, oracle)

    inside_some_frame = torch.stack(
        [bounded_box_contains(points, frame_lower[h], frame_upper[h]) for h in range(3)]
    ).any(dim=0)
    assert torch.all(~town | inside_some_frame).item()

    first_house = town_contains(
        points,
        frame_lower[:1],
        frame_upper[:1],
        window_lower[:1],
        window_upper[:1],
    )
    assert torch.all(~first_house | town).item()

    house_order = torch.tensor([2, 0, 1])
    permuted_houses = town_contains(
        points,
        frame_lower[house_order],
        frame_upper[house_order],
        window_lower[house_order],
        window_upper[house_order],
    )
    assert torch.equal(permuted_houses, town)

    duplicated_houses = town_contains(
        points,
        torch.cat((frame_lower, frame_lower[:1]), dim=0),
        torch.cat((frame_upper, frame_upper[:1]), dim=0),
        torch.cat((window_lower, window_lower[:1]), dim=0),
        torch.cat((window_upper, window_upper[:1]), dim=0),
    )
    assert torch.equal(duplicated_houses, town)

    duplicated_windows = town_contains(
        points,
        frame_lower,
        frame_upper,
        torch.cat((window_lower, window_lower[:, :1]), dim=1),
        torch.cat((window_upper, window_upper[:, :1]), dim=1),
    )
    assert torch.equal(duplicated_windows, town)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_valid_town_may_be_empty_when_every_house_is_fully_excluded(device_name: str) -> None:
    device = _device(device_name)
    points = torch.tensor(
        [[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]], dtype=torch.float32, device=device
    )
    frame_lower = torch.tensor([[-0.5, -0.5]], dtype=torch.float32, device=device)
    frame_upper = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
    window_lower = torch.tensor([[[-0.75, -0.75]]], dtype=torch.float32, device=device)
    window_upper = torch.tensor([[[0.75, 0.75]]], dtype=torch.float32, device=device)
    result = town_contains(points, frame_lower, frame_upper, window_lower, window_upper)
    assert result.tolist() == [False, False, False]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_adding_a_house_cannot_remove_membership(device_name: str) -> None:
    device = _device(device_name)
    points = torch.tensor(
        [[-0.75, 0.0], [-0.25, 0.0], [0.0, 0.0], [0.25, 0.0], [0.75, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    frame_lower = torch.tensor([[-1.0, -1.0], [0.0, -1.0]], dtype=torch.float32, device=device)
    frame_upper = torch.tensor([[-0.1, 1.0], [1.0, 1.0]], dtype=torch.float32, device=device)
    window_lower = torch.tensor([[[-0.8, -0.5]], [[0.5, -0.5]]], dtype=torch.float32, device=device)
    window_upper = torch.tensor([[[-0.4, 0.5]], [[0.9, 0.5]]], dtype=torch.float32, device=device)
    one_house = town_contains(
        points, frame_lower[:1], frame_upper[:1], window_lower[:1], window_upper[:1]
    )
    two_houses = town_contains(points, frame_lower, frame_upper, window_lower, window_upper)
    assert torch.all(~one_house | two_houses).item()


def test_town_cpu_and_cuda_results_are_exactly_equal() -> None:
    assert torch.cuda.is_available(), "TOWN-02 parity acceptance requires actual CUDA execution"
    cuda = torch.device("cuda:0")
    for fixture in TOWN_COMPOSITION_FIXTURES:
        cpu_inputs = _town_tensors(fixture, torch.device("cpu"))
        cuda_inputs = tuple(tensor.to(cuda) for tensor in cpu_inputs)
        cpu_result = town_contains(*cpu_inputs)
        cuda_result = town_contains(*cuda_inputs)
        torch.cuda.synchronize(cuda)
        assert torch.equal(cpu_result, cuda_result.cpu())
        assert cpu_result.tolist() == list(fixture.expected_contains)

    points = _generated_geometry(2, torch.device("cpu"))[0]
    lower_face = torch.tensor(-0.5, dtype=torch.float32)
    upper_face = torch.tensor(0.5, dtype=torch.float32)
    points = torch.cat(
        (
            points,
            torch.tensor(
                [
                    [-0.5, 0.0],
                    [torch.nextafter(lower_face, torch.tensor(float("inf"))).item(), 0.0],
                    [torch.nextafter(lower_face, torch.tensor(float("-inf"))).item(), 0.0],
                    [0.5, 0.0],
                    [torch.nextafter(upper_face, torch.tensor(float("-inf"))).item(), 0.0],
                    [torch.nextafter(upper_face, torch.tensor(float("inf"))).item(), 0.0],
                ],
                dtype=torch.float32,
            ),
        ),
        dim=0,
    )
    frame_lower = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float32)
    frame_upper = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    window_lower = torch.tensor([[[-0.5, -0.5]], [[1.0, 1.0]]], dtype=torch.float32)
    window_upper = torch.tensor([[[0.5, 0.5]], [[1.0, 1.0]]], dtype=torch.float32)
    cpu_inputs = (points, frame_lower, frame_upper, window_lower, window_upper)
    cuda_inputs = tuple(tensor.to(cuda) for tensor in cpu_inputs)
    cpu_result = town_contains(*cpu_inputs)
    cuda_result = town_contains(*cuda_inputs)
    torch.cuda.synchronize(cuda)
    assert torch.equal(cpu_result, _town_oracle(*cpu_inputs))
    assert cpu_result[-6:].tolist() == [True, False, True, True, False, True]
    assert torch.equal(cpu_result, cuda_result.cpu())


def test_empty_point_batch_returns_bool_empty_after_full_town_validation() -> None:
    points = torch.empty((0, 2), dtype=torch.float32)
    frame_lower = torch.tensor([[-1.0, -1.0], [0.0, -1.0]], dtype=torch.float32)
    frame_upper = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    window_lower = torch.empty((2, 0, 2), dtype=torch.float32)
    window_upper = torch.empty((2, 0, 2), dtype=torch.float32)
    result = town_contains(points, frame_lower, frame_upper, window_lower, window_upper)
    assert result.dtype is torch.bool and result.shape == (0,) and result.device.type == "cpu"

    invalid_frame_lower = frame_lower.clone()
    invalid_frame_lower[0, 0] = -1.01
    with pytest.raises(ValueError):
        town_contains(points, invalid_frame_lower, frame_upper, window_lower, window_upper)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_finite_extreme_outside_domain_town_queries_are_false(device_name: str) -> None:
    device = _device(device_name)
    largest = torch.finfo(torch.float32).max
    points = torch.tensor(
        [[-largest, 0.0], [largest, 0.0], [-2.0, 0.0], [0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    frame_lower = torch.tensor([[-1.0, -1.0]], dtype=torch.float32, device=device)
    frame_upper = torch.tensor([[1.0, 1.0]], dtype=torch.float32, device=device)
    no_windows = torch.empty((1, 0, 2), dtype=torch.float32, device=device)
    result = town_contains(points, frame_lower, frame_upper, no_windows, no_windows)
    assert torch.isfinite(points).all().item()
    assert result.tolist() == [False, False, False, True]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_empty_town_query_and_empty_window_axis_on_cpu_and_cuda(device_name: str) -> None:
    device = _device(device_name)
    points = torch.empty((0, 3), dtype=torch.float32, device=device)
    frame_lower = torch.full((2, 3), -1.0, dtype=torch.float32, device=device)
    frame_upper = torch.full((2, 3), 1.0, dtype=torch.float32, device=device)
    window_lower = torch.empty((2, 0, 3), dtype=torch.float32, device=device)
    window_upper = torch.empty((2, 0, 3), dtype=torch.float32, device=device)
    result = town_contains(points, frame_lower, frame_upper, window_lower, window_upper)
    assert result.dtype is torch.bool and result.shape == (0,) and result.device == device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@pytest.mark.parametrize("argument_index", range(5))
def test_town_rejects_non_tensor_wrong_dtype_and_unsupported_layout_per_argument(
    argument_index: int,
) -> None:
    valid = _valid_town_inputs()
    non_tensor = list(valid)
    non_tensor[argument_index] = None  # type: ignore[assignment]
    with pytest.raises(TypeError):
        town_contains(*non_tensor)  # type: ignore[arg-type]

    wrong_dtype = list(valid)
    wrong_dtype[argument_index] = wrong_dtype[argument_index].to(torch.float64)
    with pytest.raises(TypeError):
        town_contains(*wrong_dtype)

    sparse_inputs = list(valid)
    shape = tuple(sparse_inputs[argument_index].shape)
    indices = torch.zeros((len(shape), 1), dtype=torch.int64)
    sparse_inputs[argument_index] = torch.sparse_coo_tensor(
        indices, torch.tensor([0.0]), size=shape
    )
    with pytest.raises(TypeError):
        town_contains(*sparse_inputs)


@pytest.mark.parametrize(
    "inputs",
    (
        (torch.empty((0, 0)), torch.empty((1, 0)), torch.empty((1, 0)), torch.empty((1, 0, 0)), torch.empty((1, 0, 0))),
        (torch.zeros((2,)), torch.zeros((1, 2)), torch.ones((1, 2)), torch.empty((1, 0, 2)), torch.empty((1, 0, 2))),
        (torch.zeros((1, 2)), torch.zeros((0, 2)), torch.zeros((0, 2)), torch.empty((0, 0, 2)), torch.empty((0, 0, 2))),
        (torch.zeros((1, 2)), torch.zeros((1, 2)), torch.ones((2, 2)), torch.empty((1, 0, 2)), torch.empty((1, 0, 2))),
        (torch.zeros((1, 2)), torch.zeros((1, 2)), torch.ones((1, 2)), torch.empty((1, 0, 2)), torch.empty((2, 0, 2))),
        (torch.zeros((1, 2)), torch.zeros((1, 2)), torch.ones((1, 2)), torch.empty((1, 1)), torch.empty((1, 1))),
        (torch.zeros((1, 2)), torch.zeros((1, 2)), torch.ones((1, 2)), torch.empty((1, 0, 2)), torch.empty((1, 1, 2))),
        (torch.zeros((1, 2)), torch.zeros((1, 2)), torch.ones((1, 2)), torch.empty((1, 1, 3)), torch.empty((1, 1, 3))),
    ),
)
def test_town_rejects_invalid_ranks_dimensions_house_counts_and_window_shapes(
    inputs: tuple[Tensor, ...],
) -> None:
    with pytest.raises(ValueError):
        town_contains(*inputs)


def test_town_rejects_malformed_zero_window_axes_even_for_empty_queries() -> None:
    points = torch.empty((0, 2), dtype=torch.float32)
    frame_lower = torch.tensor([[-1.0, -1.0], [0.0, 0.0]], dtype=torch.float32)
    frame_upper = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    valid_empty = torch.empty((2, 0, 2), dtype=torch.float32)
    with pytest.raises(ValueError):
        town_contains(points, frame_lower, frame_upper, torch.empty((2, 0)), valid_empty)
    with pytest.raises(ValueError):
        town_contains(points, frame_lower, frame_upper, valid_empty, torch.empty((2, 0)))
    with pytest.raises(ValueError):
        town_contains(points, frame_lower, frame_upper, valid_empty[:1], valid_empty)


@pytest.mark.parametrize(
    ("argument_index", "values", "shape", "message"),
    (
        (0, [[float("nan"), 0.0]], (1, 2), "finite"),
        (0, [[float("inf"), 0.0]], (1, 2), "finite"),
        (1, [[float("nan"), -1.0], [0.0, -1.0]], (2, 2), "finite"),
        (2, [[0.0, 1.0], [1.0, float("inf")]], (2, 2), "finite"),
        (3, [[[-1.01, 0.0]], [[0.5, 0.5]]], (2, 1, 2), "domain"),
        (4, [[[0.0, 0.5]], [[0.75, 1.01]]], (2, 1, 2), "domain"),
    ),
)
def test_town_rejects_nonfinite_and_out_of_domain_coordinates(
    argument_index: int, values: list[object], shape: tuple[int, ...], message: str
) -> None:
    inputs = _valid_town_inputs()
    inputs[argument_index] = torch.tensor(values, dtype=torch.float32).reshape(shape)
    with pytest.raises(ValueError, match=message):
        town_contains(*inputs)


@pytest.mark.parametrize("argument_index", (1, 2, 3, 4))
def test_town_rejects_reversed_frame_and_window_corners(argument_index: int) -> None:
    inputs = _valid_town_inputs()
    if argument_index in (1, 2):
        inputs[1][0] = torch.tensor([0.25, -1.0])
        inputs[2][0] = torch.tensor([-0.25, 1.0])
    else:
        inputs[3][0, 0] = torch.tensor([0.25, -0.5])
        inputs[4][0, 0] = torch.tensor([-0.25, 0.5])
    with pytest.raises(ValueError, match="less than or equal"):
        town_contains(*inputs)


def test_town_rejects_invalid_geometry_even_when_points_or_window_axis_is_empty() -> None:
    points = torch.empty((0, 2), dtype=torch.float32)
    frame_lower = torch.tensor([[-1.01, -1.0]], dtype=torch.float32)
    frame_upper = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    no_windows = torch.empty((1, 0, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="domain"):
        town_contains(points, frame_lower, frame_upper, no_windows, no_windows)

    valid_frame_lower = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)
    valid_frame_upper = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    invalid_window_lower = torch.tensor([[[-1.01, 0.0]]], dtype=torch.float32)
    invalid_window_upper = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)
    with pytest.raises(ValueError, match="domain"):
        town_contains(
            points,
            valid_frame_lower,
            valid_frame_upper,
            invalid_window_lower,
            invalid_window_upper,
        )

    finite_points = torch.tensor([[float("nan"), 0.0]], dtype=torch.float32)
    valid_frame_lower = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)
    valid_frame_upper = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="finite"):
        town_contains(finite_points, valid_frame_lower, valid_frame_upper, no_windows, no_windows)


def test_town_rejects_mixed_devices_and_unsupported_device_type() -> None:
    if torch.cuda.is_available():
        inputs = _valid_town_inputs()
        inputs[4] = inputs[4].cuda()
        with pytest.raises(ValueError, match="same device"):
            town_contains(*inputs)

    meta_inputs = [
        torch.empty((1, 2), dtype=torch.float32, device="meta"),
        torch.empty((1, 2), dtype=torch.float32, device="meta"),
        torch.empty((1, 2), dtype=torch.float32, device="meta"),
        torch.empty((1, 0, 2), dtype=torch.float32, device="meta"),
        torch.empty((1, 0, 2), dtype=torch.float32, device="meta"),
    ]
    with pytest.raises(ValueError, match="CPU or CUDA"):
        town_contains(*meta_inputs)


def test_town_evaluation_does_not_mutate_any_input() -> None:
    inputs = _valid_town_inputs()
    before = [value.clone() for value in inputs]
    town_contains(*inputs)
    for actual, expected in zip(inputs, before, strict=True):
        assert torch.equal(actual, expected)
