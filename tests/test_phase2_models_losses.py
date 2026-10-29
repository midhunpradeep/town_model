from __future__ import annotations

import pytest
import torch

from town_model.models import (
    BoxClassifier,
    NeuralBaseline,
    SyntheticPointEncoder,
    TownClassifier,
    reduce_supervision,
    unreduced_penalties,
)


@pytest.mark.parametrize("model_type", [NeuralBaseline, BoxClassifier, TownClassifier])
def test_classifiers_share_order_shapes_fixed_domainthing_and_dtype(model_type) -> None:
    classes = ("A", "B")
    model = model_type(3, classes, seed=7)
    output = model(torch.zeros((5, 3), dtype=torch.float32))
    assert output.contains.shape == (5, 2)
    assert output.contains.dtype == torch.bool
    assert output.fixed_positive["DomainThing"].tolist() == [True] * 5
    assert model.class_names == classes
    if model_type is NeuralBaseline:
        assert output.logits.shape == (5, 2)
        assert output.signed_margin is None
    else:
        assert output.signed_margin.shape == (5, 2)
        assert output.signed_margin.dtype == torch.float32


def test_baseline_zero_logit_threshold_is_inclusive_and_bce_is_unreduced() -> None:
    model = NeuralBaseline(2, ("left",), seed=1)
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
    output = model(torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32))
    assert output.logits.shape == (2, 1)
    assert output.contains[:, 0].tolist() == [True, True]
    target = torch.tensor([[1.0], [0.0]])
    penalties = unreduced_penalties(output, target, mode="bce")
    assert penalties.shape == (2, 1)
    assert penalties.flatten().tolist() == pytest.approx([0.6931472, 0.6931472])


def test_signed_distance_hinge_shared_for_box_and_town_and_validates_margin() -> None:
    for model in (BoxClassifier(2, ("A",), seed=3), TownClassifier(2, ("A",), seed=3)):
        output = model(torch.tensor([[0.99, 0.99], [-0.99, -0.99]], dtype=torch.float32))
        targets = torch.tensor([[1.0], [0.0]])
        penalty = unreduced_penalties(output, targets, mode="signed_distance", margin=0.1)
        expected = torch.relu(0.1 - (2 * targets - 1) * output.signed_margin)
        assert torch.equal(penalty, expected)
    with pytest.raises(ValueError, match="nonnegative"):
        unreduced_penalties(output, targets, mode="signed_distance", margin=-0.1)
    with pytest.raises(ValueError, match="only signed_distance"):
        unreduced_penalties(output, targets, mode="bce")


def test_common_reducer_masks_unknowns_weights_positives_and_normalizes_instances() -> None:
    penalties = torch.tensor([[2.0, 4.0, 100.0], [8.0, 6.0, 1.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    known = torch.tensor([[True, True, False], [True, True, True]])
    result = reduce_supervision(
        penalties,
        targets,
        known,
        positive_weights=torch.tensor([3.0, 2.0, 4.0]),
        trainable_mask=torch.tensor([True, True, False]),
    )
    # instance 0: (2*3 + 4)/4 = 2.5; instance 1: (8 + 6*2)/3 = 20/3
    assert result.per_instance.tolist() == pytest.approx([2.5, 20 / 3])
    assert result.loss.item() == pytest.approx((2.5 + 20 / 3) / 2)
    assert result.known_count == 4
    assert result.positive_count == 2
    assert result.negative_count == 2
    assert result.unknown_count == 1
    result.loss.backward()
    assert penalties.grad is not None
    assert penalties.grad[0, 2].item() == 0.0
    assert penalties.grad[0, 0].item() > 0


def test_all_unknown_instances_are_absent_from_objective_and_have_zero_gradient() -> None:
    penalties = torch.tensor([[2.0, 1.0], [3.0, 4.0]], requires_grad=True)
    targets = torch.zeros_like(penalties)
    known = torch.zeros_like(penalties, dtype=torch.bool)
    result = reduce_supervision(penalties, targets, known)
    assert result.loss.item() == 0
    assert result.supervised_instances.tolist() == [False, False]
    result.loss.backward()
    assert torch.equal(penalties.grad, torch.zeros_like(penalties))


def test_encoder_gradients_flow_and_points_remain_bounded() -> None:
    encoder = SyntheticPointEncoder(4, 3, seed=11)
    features = torch.randn((8, 4), dtype=torch.float32)
    points = encoder(features)
    BoxClassifier(3, ("A",), seed=9)(points).signed_margin.sum().backward()
    assert points.shape == (8, 3)
    assert torch.all(points.abs() <= 1)
    assert encoder.network[0].weight.grad is not None
    assert torch.isfinite(encoder.network[0].weight.grad).all()


def test_reducer_rejects_invalid_target_and_mask_shapes() -> None:
    losses = torch.ones((2, 2))
    with pytest.raises(ValueError, match="binary"):
        reduce_supervision(losses, torch.tensor([[2.0, 0.0], [0.0, 0.0]]), torch.ones((2, 2), dtype=torch.bool))
    with pytest.raises(ValueError, match="bool"):
        reduce_supervision(losses, torch.zeros((2, 2)), torch.ones((2, 2)))
