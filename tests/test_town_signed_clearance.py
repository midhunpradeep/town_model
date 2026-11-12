from __future__ import annotations

import pytest
import torch
from torch import Tensor

from town_model.models.bounded_box import box_signed_clearance
from town_model.models.town_geometry import (
    house_contains,
    house_signed_clearance,
    town_contains,
    town_signed_clearance,
)


def _device(name: str) -> torch.device:
    if name == "cuda":
        assert (
            torch.cuda.is_available()
        ), "TOWN-03 CUDA checks require an actual CUDA runtime"
        return torch.device("cuda:0")
    return torch.device("cpu")


def _tensor(values: object, device: torch.device = torch.device("cpu")) -> Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


def _house_inputs(device: torch.device = torch.device("cpu")) -> tuple[Tensor, ...]:
    return (
        _tensor([[0.0]], device),
        _tensor([-1.0], device),
        _tensor([1.0], device),
        _tensor([[-0.25]], device),
        _tensor([[0.25]], device),
    )


def _town_inputs(device: torch.device = torch.device("cpu")) -> tuple[Tensor, ...]:
    return (
        _tensor([[0.0]], device),
        _tensor([[-1.0], [0.0]], device),
        _tensor([[0.0], [1.0]], device),
        torch.empty((2, 0, 1), dtype=torch.float32, device=device),
        torch.empty((2, 0, 1), dtype=torch.float32, device=device),
    )


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_house_signed_clearance_matches_independent_one_dimensional_literals(
    device_name: str,
) -> None:
    device = _device(device_name)
    points = _tensor([[-1.5], [-1.0], [-0.25], [0.0], [0.5], [1.0]], device)
    frame_lower = _tensor([-1.0], device)
    frame_upper = _tensor([1.0], device)
    window_lower = _tensor([[-0.25]], device)
    window_upper = _tensor([[0.25]], device)

    actual = house_signed_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper
    )

    assert actual.dtype is torch.float32
    assert actual.shape == (6,)
    assert actual.device == device
    assert actual.tolist() == [-0.5, 0.0, 0.0, -0.25, 0.25, 0.0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_overlapping_windows_have_literal_minimum_composition(
    device_name: str,
) -> None:
    device = _device(device_name)
    points = _tensor(
        [[-0.75], [-0.5], [-0.25], [0.0], [0.25], [0.5], [0.75], [1.0]], device
    )
    frame_lower = _tensor([-1.0], device)
    frame_upper = _tensor([1.0], device)
    window_lower = _tensor([[-0.5], [0.0]], device)
    window_upper = _tensor([[0.25], [0.75]], device)

    actual = house_signed_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper
    )

    assert actual.tolist() == [0.25, 0.0, -0.25, -0.25, -0.25, -0.25, 0.0, 0.0]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_h1_w0_is_literal_box_clearance_and_box_kernel_equivalence(
    device_name: str,
) -> None:
    device = _device(device_name)
    points = _tensor([[0.0, 0.0], [-1.0, 0.0], [0.75, 0.0], [2.0, -2.0]], device)
    lower = _tensor([-1.0, -0.5], device)
    upper = _tensor([0.5, 1.0], device)
    empty_windows = torch.empty((0, 2), dtype=torch.float32, device=device)
    expected = torch.tensor([0.5, 0.0, -0.25, -1.5], dtype=torch.float32, device=device)

    house = house_signed_clearance(points, lower, upper, empty_windows, empty_windows)
    town = town_signed_clearance(
        points,
        lower.unsqueeze(0),
        upper.unsqueeze(0),
        empty_windows.unsqueeze(0),
        empty_windows.unsqueeze(0),
    )

    assert torch.equal(house, expected)
    assert torch.equal(town, expected)
    assert torch.equal(house, box_signed_clearance(points, lower, upper))


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_w0_equals_frame_clearance_for_multiple_houses(device_name: str) -> None:
    device = _device(device_name)
    points = _tensor([[-1.5], [-0.75], [0.0], [0.25], [0.75], [1.5]], device)
    frame_lower = _tensor([[-1.0], [0.0]], device)
    frame_upper = _tensor([[0.0], [1.0]], device)
    empty_lower = torch.empty((2, 0, 1), dtype=torch.float32, device=device)
    empty_upper = torch.empty((2, 0, 1), dtype=torch.float32, device=device)
    expected = torch.tensor(
        [-0.5, 0.25, 0.0, 0.25, 0.25, -0.5], dtype=torch.float32, device=device
    )

    actual = town_signed_clearance(
        points, frame_lower, frame_upper, empty_lower, empty_upper
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_two_house_town_with_windows_matches_independent_literals(
    device_name: str,
) -> None:
    device = _device(device_name)
    points = _tensor(
        [
            [-0.75],
            [-0.5],
            [-0.375],
            [-0.25],
            [0.0],
            [0.25],
            [0.5],
            [0.75],
            [1.0],
            [1.25],
        ],
        device,
    )
    frame_lower = _tensor([[-1.0], [0.0]], device)
    frame_upper = _tensor([[0.0], [1.0]], device)
    window_lower = _tensor([[[-0.5]], [[0.25]]], device)
    window_upper = _tensor([[[-0.25]], [[0.5]]], device)
    expected = torch.tensor(
        [0.25, 0.0, -0.125, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, -0.25],
        dtype=torch.float32,
        device=device,
    )

    actual = town_signed_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_degenerate_fixed_frames_and_windows_have_literal_scores(
    device_name: str,
) -> None:
    device = _device(device_name)
    points = _tensor([[0.0, 0.0], [0.0, 0.75], [0.25, 0.0], [-1.0, 0.0]], device)
    frame_lower = _tensor([-1.0, -1.0], device)
    frame_upper = _tensor([1.0, 1.0], device)
    degenerate_window_lower = _tensor([[0.0, -0.75]], device)
    degenerate_window_upper = _tensor([[0.0, 0.75]], device)
    degenerate_window_scores = house_signed_clearance(
        points,
        frame_lower,
        frame_upper,
        degenerate_window_lower,
        degenerate_window_upper,
    )
    assert degenerate_window_scores.tolist() == [0.0, 0.0, 0.25, 0.0]

    collapsed_points = _tensor([[0.0, 0.0], [0.0, 0.25], [0.25, 0.0]], device)
    collapsed_frame = house_signed_clearance(
        collapsed_points,
        _tensor([0.0, 0.0], device),
        _tensor([0.0, 0.0], device),
        torch.empty((0, 2), dtype=torch.float32, device=device),
        torch.empty((0, 2), dtype=torch.float32, device=device),
    )
    assert collapsed_frame.tolist() == [0.0, -0.25, -0.25]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_protruding_window_does_not_extend_frame_and_can_remove_its_boundary(
    device_name: str,
) -> None:
    device = _device(device_name)
    points = _tensor([[-0.5, 0.0], [0.0, 0.0], [0.5, 0.5], [0.625, 0.0]], device)
    frame_lower = _tensor([-0.5, -0.5], device)
    frame_upper = _tensor([0.5, 0.5], device)
    window_lower = _tensor([[-0.75, -0.75]], device)
    window_upper = _tensor([[0.75, 0.75]], device)

    scores = house_signed_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper
    )
    membership = house_contains(
        points, frame_lower, frame_upper, window_lower, window_upper
    )

    assert scores.tolist() == [-0.25, -0.75, -0.25, -0.125]
    assert membership.tolist() == [False, False, False, False]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_zero_and_positive_clearance_boundary_degeneracy_diagnostic_is_independent(
    device_name: str,
) -> None:
    """Scores remain measurements; Boolean APIs are evaluated separately.

    Under the approved equations their signs agree mathematically with set
    membership. This fixture records a zero-score included frame boundary, a
    zero-score retained window boundary, a positive-score member, and a
    negative-score hole interior without implementing membership by threshold.
    """
    device = _device(device_name)
    points = _tensor([[-1.0, 0.0], [0.5, 0.0], [0.0, 0.0], [0.75, 0.75]], device)
    frame_lower = _tensor([-1.0, -1.0], device)
    frame_upper = _tensor([1.0, 1.0], device)
    window_lower = _tensor([[-0.5, -0.5]], device)
    window_upper = _tensor([[0.5, 0.5]], device)

    scores = house_signed_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper
    )
    membership = house_contains(
        points, frame_lower, frame_upper, window_lower, window_upper
    )

    assert scores.tolist() == [0.0, 0.0, -0.5, 0.25]
    assert membership.tolist() == [True, True, False, True]
    assert not torch.equal(scores, membership.to(dtype=scores.dtype))
    assert membership[0].item() and not (scores[0] > 0.0).item()
    assert torch.equal(membership, scores >= 0.0)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_empty_query_batches_return_float32_empty_vector(device_name: str) -> None:
    device = _device(device_name)
    points = torch.empty((0, 2), dtype=torch.float32, device=device)
    frame_lower = _tensor([-1.0, -1.0], device)
    frame_upper = _tensor([1.0, 1.0], device)
    no_windows = torch.empty((0, 2), dtype=torch.float32, device=device)
    with_window = _tensor([[-0.5, -0.5]], device)
    with_window_upper = _tensor([[0.5, 0.5]], device)

    house_empty = house_signed_clearance(
        points, frame_lower, frame_upper, no_windows, no_windows
    )
    house_nonempty_w = house_signed_clearance(
        points, frame_lower, frame_upper, with_window, with_window_upper
    )
    town_empty = town_signed_clearance(
        points,
        torch.stack((frame_lower,)),
        torch.stack((frame_upper,)),
        no_windows.unsqueeze(0),
        no_windows.unsqueeze(0),
    )
    town_empty_with_window = town_signed_clearance(
        points,
        torch.stack((frame_lower,)),
        torch.stack((frame_upper,)),
        with_window.unsqueeze(0),
        with_window_upper.unsqueeze(0),
    )

    for result in (house_empty, house_nonempty_w, town_empty, town_empty_with_window):
        assert result.dtype is torch.float32
        assert result.shape == (0,)
        assert result.device == device


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_finite_outside_domain_queries_preserve_raw_clearance(device_name: str) -> None:
    device = _device(device_name)
    points = _tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [0.0, 2.0]], device)
    lower = _tensor([-1.0, -0.5], device)
    upper = _tensor([0.5, 1.0], device)
    no_windows = torch.empty((0, 2), dtype=torch.float32, device=device)

    actual = house_signed_clearance(points, lower, upper, no_windows, no_windows)

    assert actual.tolist() == [-1.0, -1.5, -1.5, -1.0]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_finite_extreme_outside_queries_are_accepted_without_magnitude_cutoff(
    device_name: str,
) -> None:
    device = _device(device_name)
    maximum = torch.finfo(torch.float32).max
    points = _tensor([[-maximum], [maximum]], device)
    lower = _tensor([-1.0], device)
    upper = _tensor([1.0], device)
    empty_house = torch.empty((0, 1), dtype=torch.float32, device=device)
    empty_town = torch.empty((1, 0, 1), dtype=torch.float32, device=device)

    house = house_signed_clearance(points, lower, upper, empty_house, empty_house)
    town = town_signed_clearance(
        points, lower.unsqueeze(0), upper.unsqueeze(0), empty_town, empty_town
    )

    expected = torch.full((2,), -maximum, dtype=torch.float32, device=device)
    assert torch.isfinite(house).all()
    assert torch.equal(house, expected)
    assert torch.equal(town, expected)


def test_house_validation_rejects_wrong_rank_dtype_nonfinite_and_invalid_geometry() -> (
    None
):
    valid = _house_inputs()
    with pytest.raises(ValueError, match="points must have exact shape"):
        house_signed_clearance(valid[0].squeeze(0), *valid[1:])
    with pytest.raises(TypeError, match="dtype torch.float32"):
        house_signed_clearance(valid[0].double(), *valid[1:])
    with pytest.raises(ValueError, match="finite"):
        house_signed_clearance(torch.tensor([[float("nan")]]), *valid[1:])
    with pytest.raises(ValueError, match="less than or equal"):
        house_signed_clearance(
            valid[0], _tensor([0.5]), _tensor([-0.5]), valid[3], valid[4]
        )
    with pytest.raises(ValueError, match="inside the fixed domain"):
        house_signed_clearance(valid[0], _tensor([-1.1]), valid[2], valid[3], valid[4])
    with pytest.raises(ValueError, match="same window count"):
        house_signed_clearance(
            valid[0], valid[1], valid[2], _tensor([[-0.5], [0.0]]), valid[4]
        )
    with pytest.raises(ValueError, match="coordinate dimension"):
        house_signed_clearance(
            valid[0], valid[1], valid[2], torch.empty((0, 2)), torch.empty((0, 2))
        )
    with pytest.raises(TypeError, match="dtype torch.float32"):
        house_signed_clearance(
            valid[0], valid[1], valid[2], valid[3].double(), valid[4]
        )
    with pytest.raises(ValueError, match="inside the fixed domain"):
        house_signed_clearance(
            valid[0], valid[1], valid[2], _tensor([[-1.1]]), valid[4]
        )
    with pytest.raises(ValueError, match="less than or equal"):
        house_signed_clearance(
            valid[0], valid[1], valid[2], _tensor([[0.5]]), _tensor([[-0.5]])
        )
    with pytest.raises(ValueError, match="dimension d must be at least 1"):
        house_signed_clearance(
            torch.empty((2, 0)),
            torch.empty((0,)),
            torch.empty((0,)),
            torch.empty((0, 0)),
            torch.empty((0, 0)),
        )


def test_town_validation_rejects_empty_house_axis_bad_shapes_and_geometry() -> None:
    valid = _town_inputs()
    with pytest.raises(ValueError, match="at least one house"):
        town_signed_clearance(
            valid[0],
            torch.empty((0, 1)),
            torch.empty((0, 1)),
            torch.empty((0, 0, 1)),
            torch.empty((0, 0, 1)),
        )
    with pytest.raises(ValueError, match=r"identical \[H, d\]"):
        town_signed_clearance(valid[0], valid[1], valid[2][:1], valid[3], valid[4])
    with pytest.raises(ValueError, match="same house count"):
        town_signed_clearance(valid[0], valid[1], valid[2], valid[3][:1], valid[4][:1])
    with pytest.raises(ValueError, match="less than or equal"):
        town_signed_clearance(
            valid[0], _tensor([[0.5], [0.0]]), valid[2], valid[3], valid[4]
        )
    with pytest.raises(TypeError, match="dtype torch.float32"):
        town_signed_clearance(valid[0], valid[1], valid[2], valid[3].double(), valid[4])
    with pytest.raises(ValueError, match="coordinate dimension"):
        town_signed_clearance(
            valid[0], valid[1], valid[2], torch.empty((2, 0, 2)), torch.empty((2, 0, 2))
        )
    with pytest.raises(ValueError, match="inside the fixed domain"):
        town_signed_clearance(
            valid[0],
            valid[1],
            valid[2],
            _tensor([[[-1.1]], [[-0.5]]]),
            _tensor([[[-0.5]], [[0.5]]]),
        )
    with pytest.raises(ValueError, match="dimension d must be at least 1"):
        town_signed_clearance(
            torch.empty((2, 0)),
            torch.empty((1, 0)),
            torch.empty((1, 0)),
            torch.empty((1, 0, 0)),
            torch.empty((1, 0, 0)),
        )


def test_validation_checks_invalid_geometry_even_for_empty_point_batch() -> None:
    points = torch.empty((0, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="finite"):
        house_signed_clearance(
            points,
            torch.tensor([float("inf")]),
            _tensor([1.0]),
            torch.empty((0, 1)),
            torch.empty((0, 1)),
        )
    with pytest.raises(ValueError, match="finite"):
        town_signed_clearance(
            points,
            _tensor([[-1.0]]),
            _tensor([[1.0]]),
            torch.tensor([[[float("nan")]]]),
            _tensor([[[0.5]]]),
        )


def test_wrong_layout_and_mixed_device_inputs_reject() -> None:
    valid = _house_inputs()
    sparse_points = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]), torch.tensor([0.0]), size=(1, 1)
    )
    with pytest.raises(TypeError, match="dense strided"):
        house_signed_clearance(sparse_points, *valid[1:])
    with pytest.raises(ValueError, match="supported CPU or CUDA"):
        meta_points = torch.empty((1, 1), dtype=torch.float32, device="meta")
        meta_corners = torch.empty((1,), dtype=torch.float32, device="meta")
        meta_windows = torch.empty((0, 1), dtype=torch.float32, device="meta")
        house_signed_clearance(
            meta_points, meta_corners, meta_corners, meta_windows, meta_windows
        )
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="same device"):
            house_signed_clearance(valid[0].cuda(), *valid[1:])


def test_evaluators_do_not_mutate_points_or_physical_corners() -> None:
    house_args = _house_inputs()
    house_before = tuple(value.clone() for value in house_args)
    house_signed_clearance(*house_args)
    assert all(
        torch.equal(before, after)
        for before, after in zip(house_before, house_args, strict=True)
    )

    town_args = _town_inputs()
    town_before = tuple(value.clone() for value in town_args)
    town_signed_clearance(*town_args)
    assert all(
        torch.equal(before, after)
        for before, after in zip(town_before, town_args, strict=True)
    )


def test_unique_house_frame_branch_gradients_reach_points_and_corners() -> None:
    points = _tensor([[-0.75, 0.0]]).requires_grad_()
    lower = _tensor([-1.0, -1.0]).requires_grad_()
    upper = _tensor([1.0, 1.0]).requires_grad_()
    empty = torch.empty((0, 2), dtype=torch.float32)

    score = house_signed_clearance(points, lower, upper, empty, empty).sum()
    point_grad, lower_grad, upper_grad = torch.autograd.grad(
        score, (points, lower, upper)
    )

    assert score.item() == pytest.approx(0.25)
    assert point_grad.tolist() == [[1.0, 0.0]]
    assert lower_grad.tolist() == [-1.0, 0.0]
    assert upper_grad.tolist() == [0.0, 0.0]


def test_unique_house_window_complement_branch_gradients_reach_window_corners() -> None:
    points = _tensor([[0.4, 0.0]]).requires_grad_()
    frame_lower = _tensor([-1.0, -1.0]).requires_grad_()
    frame_upper = _tensor([1.0, 1.0]).requires_grad_()
    window_lower = _tensor([[-0.5, -0.5]]).requires_grad_()
    window_upper = _tensor([[0.5, 0.5]]).requires_grad_()

    score = house_signed_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper
    ).sum()
    grads = torch.autograd.grad(
        score, (points, frame_lower, frame_upper, window_lower, window_upper)
    )

    assert score.item() == pytest.approx(-0.1, abs=1e-7)
    assert grads[0].tolist() == [[1.0, 0.0]]
    assert grads[1].tolist() == [0.0, 0.0]
    assert grads[2].tolist() == [0.0, 0.0]
    assert grads[3].tolist() == [[0.0, 0.0]]
    assert grads[4].tolist() == [[-1.0, 0.0]]


def test_unique_town_branch_gradients_reach_query_and_active_house_corners() -> None:
    points = _tensor([[0.25]]).requires_grad_()
    frame_lower = _tensor([[-1.0], [0.0]]).requires_grad_()
    frame_upper = _tensor([[0.0], [1.0]]).requires_grad_()
    empty_lower = torch.empty((2, 0, 1), dtype=torch.float32)
    empty_upper = torch.empty((2, 0, 1), dtype=torch.float32)

    score = town_signed_clearance(
        points, frame_lower, frame_upper, empty_lower, empty_upper
    ).sum()
    point_grad, lower_grad, upper_grad = torch.autograd.grad(
        score, (points, frame_lower, frame_upper)
    )

    assert score.item() == pytest.approx(0.25)
    assert point_grad.tolist() == [[1.0]]
    assert lower_grad.tolist() == [[0.0], [-1.0]]
    assert upper_grad.tolist() == [[0.0], [0.0]]


def test_unique_town_window_branch_gradients_reach_window_corners() -> None:
    points = _tensor([[0.4, 0.0]]).requires_grad_()
    frame_lower = _tensor([[-1.0, -1.0]]).requires_grad_()
    frame_upper = _tensor([[1.0, 1.0]]).requires_grad_()
    window_lower = _tensor([[[-0.5, -0.5]]]).requires_grad_()
    window_upper = _tensor([[[0.5, 0.5]]]).requires_grad_()

    score = town_signed_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper
    ).sum()
    grads = torch.autograd.grad(
        score, (points, frame_lower, frame_upper, window_lower, window_upper)
    )

    assert score.item() == pytest.approx(-0.1, abs=1e-7)
    assert grads[0].tolist() == [[1.0, 0.0]]
    assert grads[3].tolist() == [[[0.0, 0.0]]]
    assert grads[4].tolist() == [[[-1.0, 0.0]]]


def test_native_amin_tie_uses_torch_equal_split_gradient() -> None:
    points = _tensor([[0.0]]).requires_grad_()
    lower = _tensor([-1.0]).requires_grad_()
    upper = _tensor([1.0]).requires_grad_()
    empty = torch.empty((0, 1), dtype=torch.float32)

    score = house_signed_clearance(points, lower, upper, empty, empty).sum()
    grads = torch.autograd.grad(score, (points, lower, upper))

    assert score.item() == 1.0
    assert grads[0].tolist() == [[0.0]]
    assert grads[1].tolist() == [-0.5]
    assert grads[2].tolist() == [0.5]


def test_house_amin_tie_cancels_point_gradient_without_custom_backward() -> None:
    points = _tensor([[0.0]]).requires_grad_()
    frame_lower = _tensor([0.0]).requires_grad_()
    frame_upper = _tensor([1.0]).requires_grad_()
    window_lower = _tensor([[0.0]]).requires_grad_()
    window_upper = _tensor([[0.5]]).requires_grad_()

    score = house_signed_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper
    ).sum()
    grads = torch.autograd.grad(score, (points, frame_lower, window_lower))

    assert score.item() == 0.0
    assert grads[0].tolist() == [[0.0]]
    assert grads[1].tolist() == [-0.5]
    assert grads[2].tolist() == [[0.5]]


def test_town_amax_tie_uses_native_equal_split_and_cancels_point_gradient() -> None:
    points = _tensor([[0.0]]).requires_grad_()
    lower = _tensor([[-1.0], [0.0]]).requires_grad_()
    upper = _tensor([[0.0], [1.0]]).requires_grad_()
    no_windows = torch.empty((2, 0, 1), dtype=torch.float32)

    score = town_signed_clearance(points, lower, upper, no_windows, no_windows).sum()
    point_grad, lower_grad, upper_grad = torch.autograd.grad(
        score, (points, lower, upper)
    )

    assert score.item() == 0.0
    assert point_grad.tolist() == [[0.0]]
    assert lower_grad.tolist() == [[0.0], [-0.5]]
    assert upper_grad.tolist() == [[0.5], [0.0]]


def test_aggregate_point_gradient_cancellation_keeps_nonzero_per_query_gradients() -> (
    None
):
    shared_offset = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    points = torch.stack((shared_offset - 0.25, shared_offset + 0.25)).reshape(2, 1)
    lower = _tensor([-1.0])
    upper = _tensor([1.0])
    no_windows = torch.empty((0, 1), dtype=torch.float32)

    scores = house_signed_clearance(points, lower, upper, no_windows, no_windows)
    individual_grads = (
        torch.autograd.grad(scores[0], shared_offset, retain_graph=True)[0],
        torch.autograd.grad(scores[1], shared_offset, retain_graph=True)[0],
    )
    aggregate_grad = torch.autograd.grad(scores.sum(), shared_offset)[0]

    assert scores.tolist() == [0.75, 0.75]
    assert [value.item() for value in individual_grads] == [1.0, -1.0]
    assert aggregate_grad.item() == 0.0


def test_unique_branch_finite_difference_matches_analytic_point_and_corner_gradients() -> (
    None
):
    point_x = torch.tensor(-0.25, dtype=torch.float32, requires_grad=True)
    lower_x = torch.tensor(-0.5, dtype=torch.float32, requires_grad=True)
    points = torch.stack((point_x, torch.tensor(0.0))).reshape(1, 2)
    lower = torch.stack((lower_x, torch.tensor(-1.0)))
    upper = _tensor([1.0, 1.0])
    empty = torch.empty((0, 2), dtype=torch.float32)
    score = house_signed_clearance(points, lower, upper, empty, empty).sum()
    point_grad, corner_grad = torch.autograd.grad(score, (point_x, lower_x))

    step = torch.tensor(1e-3, dtype=torch.float32)

    def evaluate(point_value: Tensor, lower_value: Tensor) -> Tensor:
        return house_signed_clearance(
            torch.stack((point_value, torch.tensor(0.0))).reshape(1, 2),
            torch.stack((lower_value, torch.tensor(-1.0))),
            upper,
            empty,
            empty,
        ).sum()

    point_fd = (
        evaluate(point_x.detach() + step, lower_x.detach())
        - evaluate(point_x.detach() - step, lower_x.detach())
    ) / (2 * step)
    corner_fd = (
        evaluate(point_x.detach(), lower_x.detach() + step)
        - evaluate(point_x.detach(), lower_x.detach() - step)
    ) / (2 * step)

    assert point_grad.item() == 1.0
    assert corner_grad.item() == -1.0
    assert point_fd.item() == pytest.approx(1.0, abs=1e-4)
    assert corner_fd.item() == pytest.approx(-1.0, abs=1e-4)
