from __future__ import annotations

import torch

from town_model.models import BoxClassifier, TownClassifier, reduce_supervision, unreduced_penalties


def _reduced_hinge(model, points, targets):
    output = model(points)
    penalty = unreduced_penalties(output, targets, mode="signed_distance", margin=0.1)
    return output, reduce_supervision(penalty, targets, torch.ones_like(targets, dtype=torch.bool)).loss


def test_positive_outside_and_negative_inside_box_have_point_and_geometry_gradients() -> None:
    model = BoxClassifier(2, ("A",), seed=1)
    with torch.no_grad():
        model.geometry.raw_low.zero_()
        model.geometry.raw_width.zero_()
    positive_outside = torch.tensor([[0.95, 0.95]], dtype=torch.float32, requires_grad=True)
    _, loss = _reduced_hinge(model, positive_outside, torch.tensor([[1.0]]))
    loss.backward()
    assert loss.item() > 0
    assert positive_outside.grad is not None and positive_outside.grad.abs().sum() > 0
    assert model.geometry.raw_low.grad is not None and model.geometry.raw_low.grad.abs().sum() > 0
    assert model.geometry.raw_width.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    negative_inside = torch.tensor([[0.45, 0.2]], dtype=torch.float32, requires_grad=True)
    output, loss = _reduced_hinge(model, negative_inside, torch.tensor([[0.0]]))
    assert output.contains.item()
    assert loss.item() > 0
    loss.backward()
    assert negative_inside.grad is not None and negative_inside.grad.abs().sum() > 0
    assert model.geometry.raw_width.grad is not None and model.geometry.raw_width.grad.abs().sum() > 0


def test_positive_inside_window_activates_window_branch_and_negative_inside_house_activates_frame() -> None:
    model = TownClassifier(2, ("A",), seed=21, houses=1, windows=1)
    frame_low, frame_high, window_low, window_high = model.geometry.physical_geometry()
    window_width = window_high[0, 0, 0] - window_low[0, 0, 0]
    inside_window = (window_low[0, 0, 0] + window_width * torch.tensor([0.12, 0.37])).detach().reshape(1, 2).requires_grad_(True)
    output, loss = _reduced_hinge(model, inside_window, torch.tensor([[1.0]]))
    assert not output.contains.item()
    assert output.signed_margin.item() < 0
    loss.backward()
    assert inside_window.grad is not None and inside_window.grad.abs().sum() > 0
    assert model.geometry.raw_window_width.grad is not None
    assert model.geometry.raw_window_width.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    frame_low, frame_high, window_low, window_high = model.geometry.physical_geometry()
    near_frame = torch.stack((frame_low[0, 0, 0] + 0.01, (frame_low[0, 0, 1] + frame_high[0, 0, 1]) / 2)).detach().reshape(1, 2).requires_grad_(True)
    output, loss = _reduced_hinge(model, near_frame, torch.tensor([[0.0]]))
    assert output.contains.item()
    assert output.signed_margin.item() > 0
    assert loss.item() > 0
    loss.backward()
    assert near_frame.grad is not None and near_frame.grad.abs().sum() > 0
    assert model.geometry.raw_frame_low.grad is not None
    assert model.geometry.raw_frame_low.grad.abs().sum() > 0


def test_high_dimensional_geometry_hinge_has_finite_loss_and_gradients() -> None:
    model = TownClassifier(64, ("A", "B"), seed=113, houses=2, windows=1)
    points = torch.linspace(-0.95, 0.95, 64 * 5, dtype=torch.float32).reshape(5, 64).requires_grad_(True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    _, loss = _reduced_hinge(model, points, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert points.grad is not None and torch.isfinite(points.grad).all()
    for parameter in model.geometry.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
