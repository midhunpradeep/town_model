from __future__ import annotations

import pytest
import torch

from town_model.models import reduce_supervision


def test_common_reducer_masks_unknowns_weights_positives_and_normalizes_instances() -> None:
    values = torch.tensor([[2.0, 4.0, 100.0], [8.0, 6.0, 1.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    known = torch.tensor([[True, True, False], [True, True, True]])
    result = reduce_supervision(
        values,
        targets,
        known,
        positive_weights=torch.tensor([3.0, 2.0, 4.0]),
        trainable_mask=torch.tensor([True, True, False]),
    )
    assert result.per_instance.tolist() == pytest.approx([2.5, 20 / 3])
    assert result.loss.item() == pytest.approx((2.5 + 20 / 3) / 2)
    assert result.known_count == 4
    assert result.positive_count == 2
    assert result.negative_count == 2
    assert result.unknown_count == 1
    result.loss.backward()
    assert values.grad is not None
    assert values.grad[0, 2].item() == 0.0
    assert values.grad[0, 0].item() > 0


def test_all_unknown_instances_are_absent_from_reduction_and_have_zero_gradient() -> None:
    values = torch.tensor([[2.0, 1.0], [3.0, 4.0]], requires_grad=True)
    targets = torch.zeros_like(values)
    known = torch.zeros_like(values, dtype=torch.bool)
    result = reduce_supervision(values, targets, known)
    assert result.loss.item() == 0
    assert result.supervised_instances.tolist() == [False, False]
    result.loss.backward()
    assert torch.equal(values.grad, torch.zeros_like(values))


def test_reducer_rejects_invalid_target_and_mask_shapes() -> None:
    values = torch.ones((2, 2))
    with pytest.raises(ValueError, match="binary"):
        reduce_supervision(values, torch.tensor([[2.0, 0.0], [0.0, 0.0]]), torch.ones((2, 2), dtype=torch.bool))
    with pytest.raises(ValueError, match="bool"):
        reduce_supervision(values, torch.zeros((2, 2)), torch.ones((2, 2)))
