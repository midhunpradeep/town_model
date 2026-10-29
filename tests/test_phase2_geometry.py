from __future__ import annotations

import pytest
import torch

from town_model.models.geometry import BoxGeometry, TownGeometry, evaluate_town_corners


def test_box_closed_boundaries_margin_and_class_shapes() -> None:
    geometry = BoxGeometry.from_physical(
        torch.tensor([[[0.0, 0.0]], [[-0.5, -0.5]]]),
        torch.tensor([[[1.0, 1.0]], [[0.5, 0.5]]]),
    )
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.01, 0.0], [-0.5, -0.5]], dtype=torch.float32)
    result = geometry.evaluate(points)
    assert result.contains.dtype == torch.bool
    assert result.signed_margin.dtype == torch.float32
    assert result.contains.shape == result.signed_margin.shape == (4, 2)
    assert result.contains.tolist() == [[True, True], [True, False], [False, False], [False, True]]
    assert result.signed_margin[0, 0].item() == pytest.approx(0.0)
    assert result.signed_margin[1, 0].item() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="chunk_size"):
        geometry.evaluate(points, class_chunk_size=0)


def test_window_boundary_exclusion_and_other_house_reinclusion() -> None:
    frames_low = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    frames_high = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    windows_low = torch.tensor([[[[0.25, 0.25]], [[0.4, 0.4]]]])
    windows_high = torch.tensor([[[[0.75, 0.75]], [[0.6, 0.6]]]])
    points = torch.tensor([[0.25, 0.5], [0.5, 0.5], [0.1, 0.1]], dtype=torch.float32)
    result = evaluate_town_corners(points, frames_low, frames_high, windows_low, windows_high, diagnostics=True)
    assert result.contains[:, 0].tolist() == [True, False, True]
    # Window boundary is excluded by house 0; house 1 independently re-includes it.
    assert result.diagnostics["house_contains"][:, 0, 0].tolist() == [False, False, True]
    assert result.diagnostics["house_contains"][:, 0, 1].tolist() == [True, False, True]
    assert result.signed_margin[1, 0].item() == pytest.approx(-0.1)


def test_closed_window_boundary_is_excluded_even_when_margin_is_zero() -> None:
    frame_low = torch.tensor([[[0.0, 0.0]]])
    frame_high = torch.tensor([[[1.0, 1.0]]])
    window_low = torch.tensor([[[[0.25, 0.25]]]])
    window_high = torch.tensor([[[[0.75, 0.75]]]])
    point = torch.tensor([[0.25, 0.5]], dtype=torch.float32)
    result = evaluate_town_corners(point, frame_low, frame_high, window_low, window_high)
    assert result.signed_margin.item() == pytest.approx(0.0)
    assert result.contains.item() is False


def test_multiple_windows_and_classes_follow_literal_boolean_set_operations() -> None:
    frame_low = torch.zeros((2, 1, 2), dtype=torch.float32)
    frame_high = torch.ones((2, 1, 2), dtype=torch.float32)
    window_low = torch.tensor([
        [[[0.1, 0.1], [0.6, 0.6]]],
        [[[0.2, 0.2], [0.4, 0.4]]],
    ], dtype=torch.float32)
    window_high = torch.tensor([
        [[[0.3, 0.3], [0.8, 0.8]]],
        [[[0.35, 0.35], [0.5, 0.5]]],
    ], dtype=torch.float32)
    points = torch.tensor([[0.2, 0.2], [0.7, 0.7], [0.45, 0.45], [0.95, 0.95]], dtype=torch.float32)
    result = evaluate_town_corners(points, frame_low, frame_high, window_low, window_high)
    assert result.contains.shape == (4, 2)
    assert result.contains.tolist() == [[False, False], [False, True], [True, False], [True, True]]


def test_extreme_float32_raw_parameters_remain_valid_and_seeded() -> None:
    first = BoxGeometry(3, 2, seed=91, min_width=1e-5)
    second = BoxGeometry(3, 2, seed=91, min_width=1e-5)
    with torch.no_grad():
        first.raw_low.fill_(1e30)
        first.raw_width.fill_(-1e30)
    low, high = first.physical_boxes()
    assert low.dtype == high.dtype == torch.float32
    assert torch.all(low >= -1) and torch.all(high <= 1)
    assert torch.all(high - low >= 1e-5)
    assert torch.all(low <= 1) and torch.all(high >= -1)
    assert torch.equal(second.raw_low, BoxGeometry(3, 2, seed=91).raw_low)


def test_contained_town_geometry_extremes_and_gradients() -> None:
    geometry = TownGeometry(2, 3, houses=2, windows=1, seed=8, min_width=1e-5)
    with torch.no_grad():
        geometry.raw_frame_low.fill_(1e30)
        geometry.raw_frame_width.fill_(-1e30)
        geometry.raw_window_position.fill_(-1e30)
        geometry.raw_window_width.fill_(1e30)
    fl, fr, wl, wr = geometry.physical_geometry()
    assert torch.all(fl >= -1) and torch.all(fr <= 1)
    assert torch.all(wl - fl[:, :, None, :] >= 1e-5)
    assert torch.all(fr[:, :, None, :] - wr >= 1e-5)
    assert torch.all(wr - wl >= 1e-5)
    assert torch.all(fr - fl >= 3e-5)

    regular = TownGeometry(2, 1, houses=2, windows=1, seed=3)
    points = torch.tensor([[0.95, 0.95], [0.0, 0.0]], requires_grad=True)
    result = regular.evaluate(points)
    loss = torch.relu(0.1 - result.signed_margin[0, 0]) + result.signed_margin[1, 0]
    loss.backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()
    assert regular.raw_frame_low.grad is not None
    assert regular.raw_window_width.grad is not None


def test_empty_batches_chunked_equivalence_and_validation() -> None:
    geometry = TownGeometry(4, 3, houses=2, windows=2, seed=2)
    empty = geometry.evaluate(torch.empty((0, 4), dtype=torch.float32))
    assert empty.contains.shape == empty.signed_margin.shape == (0, 3)
    points = torch.rand((13, 4), dtype=torch.float32) * 2 - 1
    direct = geometry.evaluate(points)
    chunked = geometry.evaluate(points, point_chunk_size=3, class_chunk_size=1)
    assert torch.equal(direct.contains, chunked.contains)
    assert torch.allclose(direct.signed_margin, chunked.signed_margin)
    assert direct.diagnostics is None
    detailed = geometry.evaluate(points, diagnostics=True)
    chunked_detailed = geometry.evaluate(points, diagnostics=True, point_chunk_size=3, class_chunk_size=1)
    for name, values in detailed.diagnostics.items():
        assert torch.equal(values, chunked_detailed.diagnostics[name])
    direct.signed_margin.sum().backward()
    direct_grad = geometry.raw_frame_low.grad.clone()
    geometry.zero_grad(set_to_none=True)
    chunked = geometry.evaluate(points, point_chunk_size=3, class_chunk_size=1)
    chunked.signed_margin.sum().backward()
    assert torch.allclose(direct_grad, geometry.raw_frame_low.grad)
    with pytest.raises(ValueError, match="shape"):
        geometry.evaluate(torch.zeros((3, 3), dtype=torch.float32))
    with pytest.raises(ValueError, match="finite"):
        geometry.evaluate(torch.tensor([[float("nan"), 0, 0, 0]], dtype=torch.float32))
    with pytest.raises(ValueError, match="float32"):
        geometry.evaluate(torch.zeros((1, 4), dtype=torch.float64))
    with pytest.raises(ValueError, match="chunk_size"):
        geometry.evaluate(points, point_chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
        geometry.evaluate(points, class_chunk_size=0)


def test_geometry_initialization_is_label_agnostic_and_houses_differ() -> None:
    geometry = TownGeometry(16, 4, houses=2, windows=1, seed=73)
    fl, fr, wl, wr = geometry.physical_geometry()
    centers = (fl + fr) / 2
    assert not torch.equal(centers[:, 0], centers[:, 1])
    assert torch.all((fr - fl) > 0)
    assert torch.all((wr - wl) > 0)
    assert geometry.initialization_info["seed"] == 73
    assert len(geometry.initialization_info["frame_centers"]) == 4
    assert len(geometry.initialization_info["frame_centers"][0]) == 2
