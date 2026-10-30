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
    assert geometry.initialization_info["window_target_volume_fraction"] == pytest.approx(0.25)
    assert geometry.initialization_info["window_axis_fraction"] ** 16 == pytest.approx(0.25)
    assert len(geometry.initialization_info["frame_centers"]) == 4
    assert len(geometry.initialization_info["frame_centers"][0]) == 2


def test_dimension_aware_initialization_is_seeded_and_diverse() -> None:
    first = TownGeometry(16, 2, seed=73, window_target_volume=0.25)
    repeated = TownGeometry(16, 2, seed=73, window_target_volume=0.25)
    alternate = TownGeometry(16, 2, seed=74, window_target_volume=0.25)
    assert all(torch.equal(a, b) for a, b in zip(first.physical_geometry(), repeated.physical_geometry()))
    assert any(not torch.equal(a, b) for a, b in zip(first.physical_geometry(), alternate.physical_geometry()))
    frame_low, frame_high, window_low, window_high = first.physical_geometry()
    assert not torch.equal((frame_low[:, 0] + frame_high[:, 0]), (frame_low[:, 1] + frame_high[:, 1]))
    assert not torch.equal(window_low[:, 0], window_low[:, 1])
    assert torch.all((window_high - window_low) < (frame_high - frame_low)[:, :, None, :])


def test_minimum_width_feasibility_and_float32_physical_invariants() -> None:
    reserve = 16.0 * torch.finfo(torch.float32).eps
    box_maximum = 2.0 - 2.0 * reserve
    town_maximum = (2.0 - 5.0 * reserve) / 3.0
    assert BoxGeometry(2, 1).min_width == pytest.approx(1e-5)
    assert TownGeometry(2, 1).min_width == pytest.approx(1e-5)
    near_max_box = BoxGeometry(2, 1, min_width=box_maximum - 1e-6)
    with torch.no_grad():
        near_max_box.raw_low.fill_(-1e30)
        near_max_box.raw_width.fill_(1e30)
    near_low, near_high = near_max_box.physical_boxes()
    assert torch.all(near_low >= -1) and torch.all(near_high <= 1)
    assert torch.all(near_high - near_low >= box_maximum - 1e-6)
    with pytest.raises(ValueError, match="float32 width reserve"):
        BoxGeometry(2, 1, min_width=box_maximum)
    with pytest.raises(ValueError, match="float32 width reserve"):
        BoxGeometry(2, 1, min_width=box_maximum + 1e-6)
    near_max_town = TownGeometry(2, 1, min_width=town_maximum - 1e-6)
    with torch.no_grad():
        near_max_town.raw_frame_low.fill_(1e30)
        near_max_town.raw_frame_width.fill_(-1e30)
        near_max_town.raw_window_position.fill_(1e30)
        near_max_town.raw_window_width.fill_(-1e30)
    near_frame_low, near_frame_high, near_window_low, near_window_high = near_max_town.physical_geometry()
    assert torch.all(near_frame_low >= -1) and torch.all(near_frame_high <= 1)
    assert torch.all(near_frame_high - near_frame_low >= town_maximum - 1e-6)
    assert torch.all(near_window_high - near_window_low >= town_maximum - 1e-6)
    assert torch.all(near_window_low - near_frame_low[:, :, None, :] >= town_maximum - 1e-6)
    assert torch.all(near_frame_high[:, :, None, :] - near_window_high >= town_maximum - 1e-6)
    with pytest.raises(ValueError, match="contained town frames"):
        TownGeometry(2, 1, min_width=town_maximum + 1e-6)

    for invalid in (0.0, -1e-5, float("inf"), float("nan"), 2.0):
        with pytest.raises(ValueError, match="min_width"):
            BoxGeometry(2, 1, min_width=invalid)
        with pytest.raises(ValueError, match="min_width"):
            TownGeometry(2, 1, min_width=invalid)

    requested = 1e-5
    box = BoxGeometry(2, 2, min_width=requested)
    town = TownGeometry(2, 2, min_width=requested)
    with torch.no_grad():
        box.raw_low.fill_(1e30)
        box.raw_width.fill_(-1e30)
        town.raw_frame_low.fill_(1e30)
        town.raw_frame_width.fill_(-1e30)
        town.raw_window_position.fill_(-1e30)
        town.raw_window_width.fill_(1e30)
    box_low, box_high = box.physical_boxes()
    frame_low, frame_high, window_low, window_high = town.physical_geometry()
    assert torch.all(torch.isfinite(box_low)) and torch.all(torch.isfinite(box_high))
    assert torch.all(box_low >= -1) and torch.all(box_high <= 1)
    assert torch.all(box_high - box_low >= requested)
    assert torch.all(frame_low >= -1) and torch.all(frame_high <= 1)
    assert torch.all(window_low >= -1) and torch.all(window_high <= 1)
    assert torch.all(frame_high - frame_low >= 3.0 * town.clearance)
    assert torch.all(window_high - window_low >= requested)
    assert torch.all(window_low - frame_low[:, :, None, :] >= requested)
    assert torch.all(frame_high[:, :, None, :] - window_high >= requested)
    assert torch.all(window_low > frame_low[:, :, None, :])
    assert torch.all(window_high < frame_high[:, :, None, :])


def test_physical_box_helper_rejects_width_below_configured_minimum() -> None:
    with pytest.raises(ValueError, match="minimum axis width"):
        BoxGeometry.from_physical(
            torch.tensor([[[0.0, 0.0]]]),
            torch.tensor([[[1e-6, 0.5]]]),
            min_width=1e-5,
        )


@pytest.mark.parametrize("dimensions", [2, 8, 16])
def test_target_volume_initialization_materially_improves_high_dimensional_window_activity(dimensions: int) -> None:
    old_volume = 0.72**dimensions
    new_window_counts = torch.zeros((2, 2), dtype=torch.int64)
    old_window_counts = torch.zeros((2, 2), dtype=torch.int64)
    new_effective = 0
    old_effective = 0
    new_volume_fractions = []
    old_volume_fractions = []
    new_grad_norms = []
    new_zero_fractions = []

    for seed in range(6):
        new = TownGeometry(dimensions, 2, seed=seed, window_target_volume=0.25)
        old = TownGeometry(dimensions, 2, seed=seed, window_target_volume=old_volume)
        generator = torch.Generator(device="cpu").manual_seed(10_000 + seed)
        points = torch.rand((2048, dimensions), generator=generator) * 2 - 1
        new_result = new.evaluate(points, diagnostics=True)
        old_result = old.evaluate(points, diagnostics=True)
        new_window_counts += new_result.diagnostics["window_contains"].sum(dim=0).squeeze(-1)
        old_window_counts += old_result.diagnostics["window_contains"].sum(dim=0).squeeze(-1)
        new_effective += int((new_result.diagnostics["frame_contains"].any(dim=-1) & ~new_result.contains).sum().item())
        old_effective += int((old_result.diagnostics["frame_contains"].any(dim=-1) & ~old_result.contains).sum().item())
        for geometry, result, volume_rows in (
            (new, new_result, new_volume_fractions),
            (old, old_result, old_volume_fractions),
        ):
            frame_low, frame_high, window_low, window_high = geometry.physical_geometry()
            frame_volume = (frame_high - frame_low).prod(dim=-1)
            window_volume = (window_high - window_low).prod(dim=-1)
            volume_rows.extend((window_volume / frame_volume[:, :, None]).flatten().tolist())

        labels = torch.stack((
            (points[:, 0].abs() <= 0.82) & (points[:, 1].abs() <= 0.82) & ~((points[:, 0].abs() <= 0.30) & (points[:, 1].abs() <= 0.30)),
            (points[:, 0].abs() >= 0.55) & (points[:, 1].abs() <= 0.65),
        ), dim=1).float()
        signs = labels * 2 - 1
        loss = torch.relu(0.1 - signs * new_result.signed_margin).mean()
        loss.backward()
        grad = new.raw_window_width.grad
        new_grad_norms.append(float(torch.linalg.vector_norm(grad).item()))
        new_zero_fractions.append(float((grad == 0).float().mean().item()))

    assert all(0.0 < value < 1.0 for value in new_volume_fractions + old_volume_fractions)
    assert torch.all(new_window_counts > 0)
    assert new_effective > 0 and old_effective > 0
    assert all(value > 0.0 for value in new_grad_norms)
    assert all(0.0 <= value < 1.0 for value in new_zero_fractions)
    if dimensions == 2:
        assert new_window_counts.float().mean().item() > 50
    else:
        assert new_window_counts.sum().item() > old_window_counts.sum().item() * (2 if dimensions == 8 else 10)
        assert new_effective > old_effective
