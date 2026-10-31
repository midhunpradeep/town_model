from __future__ import annotations

import pytest
import torch

from town_model.contracts import FeatureDiagnostics, VisionFeatureBatch
from town_model.models import DeepSetsComposition, ImagePointModel
from town_model.training.supervision import (
    ImageObjectiveTerms,
    UndefinedImageObjective,
    aggregate_effective_batch,
    reduce_weighted_instances,
)


def _features(batch: int = 2, components: int = 3) -> VisionFeatureBatch:
    generator = torch.Generator().manual_seed(784)
    valid = torch.ones((batch, components), dtype=torch.bool)
    if components:
        valid[-1, -1] = False
    ids = tuple(
        tuple(f"component-{row}-{column}" if valid[row, column] else None for column in range(components))
        for row in range(batch)
    )
    return VisionFeatureBatch(
        component_vectors=torch.randn((batch, components, 260), generator=generator),
        component_valid_mask=valid,
        global_spatial_features=torch.randn((batch, 256, 7, 7), generator=generator),
        component_ids=ids,
        diagnostics=FeatureDiagnostics(),
    )


@pytest.mark.parametrize("dimensions", [2, 8, 16, 5])
def test_model_projection_shapes_and_bounded_point_contract(dimensions: int) -> None:
    features = _features()
    model = ImagePointModel(dimensions, pizza_mode="hybrid", seed=17)
    result = model(features, ("sample-1", "sample-2"))
    assert result.component_points.shape == (2, 3, dimensions)
    assert result.global_pizza_points.shape == (2, dimensions)
    assert result.composed_pizza_points.shape == (2, dimensions)
    assert result.final_pizza_points.shape == (2, dimensions)
    assert result.sample_ids == ("sample-1", "sample-2")
    assert result.component_ids == features.component_ids
    assert result.component_valid_mask.equal(features.component_valid_mask)
    for points in (result.component_points, result.global_pizza_points, result.composed_pizza_points, result.final_pizza_points):
        assert torch.all(points >= -1) and torch.all(points <= 1)
    assert not hasattr(result, "contains")
    assert not hasattr(result, "logits")


def test_deepsets_is_permutation_invariant_multiplicity_sensitive_and_empty_safe() -> None:
    torch.manual_seed(3)
    model = DeepSetsComposition(8).eval()
    points = torch.randn(2, 4, 8).tanh()
    valid = torch.tensor([[True, True, True, False], [False, False, False, False]])
    value = model(points, valid)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = model(points[:, permutation], valid[:, permutation])
    assert torch.allclose(value, permuted, atol=1e-7)
    empty = model(points[:1, :0], valid[:1, :0])
    expected_empty = model.rho(torch.zeros((1, 128)))
    assert torch.equal(empty, expected_empty)
    one_sum = model.phi(points[:, :1]).sum(dim=1)
    repeated_sum = model.phi(torch.cat((points[:, :1], points[:, :1]), dim=1)).sum(dim=1)
    assert not torch.equal(one_sum, repeated_sum)
    assert torch.all(value <= 1) and torch.all(value >= -1)


def test_zero_roi_image_points_produce_the_empty_deepsets_point() -> None:
    features = _features(batch=2, components=0)
    model = ImagePointModel(2, pizza_mode="compositional").eval()
    result = model(features, ("empty-1", "empty-2"))
    assert result.component_points.shape == (2, 0, 2)
    assert torch.equal(result.composed_pizza_points, model.composition.rho(torch.zeros((2, 128))))
    assert torch.equal(result.final_pizza_points, result.composed_pizza_points)


def test_effective_batch_aggregation_matches_full_batch_gradient_with_heterogeneous_eligibility() -> None:
    component = torch.tensor([0.2, 1.0, 5.0, 8.0, 3.0], requires_grad=True)
    pizza = torch.tensor([1.0, 4.0, 2.0, 9.0, 6.0], requires_grad=True)
    component_eligible = torch.tensor([True, False, True, True, False])
    pizza_eligible = torch.tensor([True, True, False, True, True])
    full = ImageObjectiveTerms(component, component_eligible, pizza, pizza_eligible, 3, 4)
    full_loss = aggregate_effective_batch([full]).total
    full_grad = torch.autograd.grad(full_loss, (component, pizza))

    component_split = component.detach().clone().requires_grad_()
    pizza_split = pizza.detach().clone().requires_grad_()
    micros = []
    for start, end in ((0, 2), (2, 4), (4, 5)):
        micros.append(ImageObjectiveTerms(
            component_split[start:end], component_eligible[start:end],
            pizza_split[start:end], pizza_eligible[start:end],
            int(component_eligible[start:end].sum()), int(pizza_eligible[start:end].sum()),
        ))
    accumulated = aggregate_effective_batch(micros).total
    accumulated_grad = torch.autograd.grad(accumulated, (component_split, pizza_split))
    assert torch.allclose(full_loss, accumulated)
    assert torch.allclose(full_grad[0], accumulated_grad[0])
    assert torch.allclose(full_grad[1], accumulated_grad[1])


def test_weighted_instance_reduction_masks_unknown_values_and_gradients() -> None:
    values = torch.tensor([[2.0, 10.0, 4.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    known = torch.tensor([[True, True, False]])
    reduced, eligible = reduce_weighted_instances(
        values, targets, known, positive_weights=torch.tensor([9.0, 1.0, 100.0])
    )
    assert reduced.item() == pytest.approx(2.8)
    assert eligible.tolist() == [True]
    reduced.sum().backward()
    assert torch.allclose(values.grad, torch.tensor([[0.9, 0.1, 0.0]]))


def test_weighted_effective_batch_accumulation_matches_full_batch_gradients() -> None:
    weights = torch.tensor([9.0, 1.0, 3.0])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    known = torch.tensor([[True, True, False], [True, True, True], [True, False, True], [False, False, False]])
    full_values = torch.tensor([[2.0, 10.0, 7.0], [4.0, 1.0, 9.0], [3.0, 6.0, 2.0], [50.0, 40.0, 30.0]], requires_grad=True)
    values, eligible = reduce_weighted_instances(full_values, targets, known, positive_weights=weights)
    full_terms = ImageObjectiveTerms(values, eligible, values.detach(), torch.zeros(4, dtype=torch.bool), int(eligible.sum()), 0)
    full_loss = aggregate_effective_batch([full_terms], component_coefficient=1.0, pizza_coefficient=0.0).total
    full_gradient = torch.autograd.grad(full_loss, full_values)[0]

    split_values = full_values.detach().clone().requires_grad_()
    windows = []
    for indices in ((0,), (1, 2), (3,)):
        index = torch.tensor(indices)
        part, mask = reduce_weighted_instances(
            split_values.index_select(0, index), targets.index_select(0, index),
            known.index_select(0, index), positive_weights=weights,
        )
        windows.append(ImageObjectiveTerms(part, mask, part.detach(), torch.zeros(len(indices), dtype=torch.bool), int(mask.sum()), 0))
    split_loss = aggregate_effective_batch(windows, component_coefficient=1.0, pizza_coefficient=0.0).total
    split_gradient = torch.autograd.grad(split_loss, split_values)[0]
    assert torch.allclose(full_loss, split_loss)
    assert torch.allclose(full_gradient, split_gradient)
    assert torch.equal(full_gradient[3], torch.zeros_like(full_gradient[3]))


def test_known_mask_values_reduce_per_instance_then_per_image_without_score_conversion() -> None:
    component_values = torch.tensor([[[2.0, 10.0], [3.0, 4.0]], [[8.0, 9.0], [0.0, 0.0]]], requires_grad=True)
    component_targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]])
    component_known = torch.tensor([[[True, True], [True, False]], [[False, False], [False, False]]])
    flattened, eligible = reduce_weighted_instances(
        component_values.reshape(-1, 2), component_targets.reshape(-1, 2),
        component_known.reshape(-1, 2), positive_weights=torch.tensor([3.0, 2.0]),
    )
    per_component = flattened.reshape(2, 2)
    component_eligible = eligible.reshape(2, 2)
    counts = component_eligible.sum(dim=1)
    component_per_image = (per_component * component_eligible).sum(dim=1) / counts.clamp_min(1)
    component_images = counts > 0
    assert component_per_image.tolist() == pytest.approx([3.5, 0.0])
    assert component_images.tolist() == [True, False]

    pizza_values = torch.tensor([[4.0, 99.0], [99.0, -99.0]], requires_grad=True)
    pizza_targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    pizza_known = torch.tensor([[True, False], [False, False]])
    pizza_per_image, pizza_eligible = reduce_weighted_instances(
        pizza_values, pizza_targets, pizza_known,
        positive_weights=torch.tensor([3.0, 2.0]),
    )
    terms = ImageObjectiveTerms(component_per_image, component_images, pizza_per_image, pizza_eligible, 2, 1)
    aggregate = aggregate_effective_batch([terms])
    assert pizza_per_image.tolist() == [4.0, 0.0]
    assert pizza_eligible.tolist() == [True, False]
    assert aggregate.component_loss.item() == pytest.approx(3.5)
    assert aggregate.pizza_loss.item() == 4.0
    assert aggregate.total.item() == pytest.approx(7.5)
    aggregate.total.backward()
    assert component_values.grad[1].abs().sum() == 0
    assert component_values.grad[0, 0, 1] != 0
    assert component_values.grad[0, 1, 1] == 0
    assert pizza_values.grad[0, 1] == 0
    assert pizza_values.grad[1].abs().sum() == 0


def test_unit_weights_reproduce_unweighted_known_value_mean() -> None:
    values = torch.tensor([[1.0, 3.0, 99.0], [4.0, 8.0, 10.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    known = torch.tensor([[True, True, False], [True, False, True]])
    unweighted, _ = reduce_weighted_instances(values, targets, known)
    unit_weighted, _ = reduce_weighted_instances(values, targets, known, positive_weights=torch.ones(3))
    assert torch.equal(unweighted, unit_weighted)
    assert unweighted.tolist() == pytest.approx([2.0, 7.0])


def test_undefined_effective_batch_is_reported_without_fabricated_zero_objective() -> None:
    values = torch.zeros(2)
    mask = torch.zeros(2, dtype=torch.bool)
    terms = ImageObjectiveTerms(values, mask, values, mask, 0, 0)
    with pytest.raises(UndefinedImageObjective, match="no eligible"):
        aggregate_effective_batch([terms])
