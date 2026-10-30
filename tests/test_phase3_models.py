from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from town_model.contracts import ImageModelResult, VisionFeatureBatch
from town_model.dataset.image_data import deterministic_subset, load_prepared_dataset
from town_model.models import ClassifierOutput, DeepSetsComposition, ImageOntologyModel
from town_model.training.image_objectives import (
    ImageObjectiveTerms,
    UndefinedImageObjective,
    aggregate_effective_batch,
    image_objective_terms,
)
from town_model.vision.backbone import MobileNetV3LargeFPN
from town_model.vision.batching import collate_image_records
from town_model.vision.features import SharedVisualFeatureExtractor
from town_model.vision.roi import ExplicitROIProvider


@pytest.fixture(scope="module")
def image_examples():
    torch.set_num_threads(2)
    dataset = load_prepared_dataset(Path(__file__).parents[1] / "examples" / "pizza-48")
    subset = deterministic_subset(dataset, "train", size=2, seed=117)
    batch = collate_image_records(dataset, subset.records)
    extractor = SharedVisualFeatureExtractor(MobileNetV3LargeFPN(pretrained=True))
    extractor.backbone.set_trainable(trainable_fpn=False)
    extractor.eval()
    with torch.no_grad():
        features = extractor(batch)
    return dataset, batch, features, extractor


@pytest.mark.parametrize("dimensions", [2, 8, 16, 5])
def test_model_projection_shapes_and_bounded_point_contract(dimensions, image_examples):
    dataset, batch, features, _ = image_examples
    model = ImageOntologyModel(dimensions, dataset.class_names, family="baseline", pizza_mode="hybrid")
    result = model(features, batch.sample_ids)
    assert result.component_points.shape == (*features.component_valid_mask.shape, dimensions)
    assert result.global_pizza_points.shape == (len(batch.sample_ids), dimensions)
    assert result.composed_pizza_points.shape == (len(batch.sample_ids), dimensions)
    assert result.final_pizza_points.shape == (len(batch.sample_ids), dimensions)
    assert torch.all(result.component_points >= -1) and torch.all(result.component_points <= 1)
    assert torch.all(result.final_pizza_points >= -1) and torch.all(result.final_pizza_points <= 1)


@pytest.mark.parametrize("family", ["baseline", "box", "town"])
@pytest.mark.parametrize("mode", ["global", "compositional", "hybrid"])
def test_all_nine_image_model_configurations_forward_and_backpropagate(family, mode, image_examples):
    dataset, batch, features, _ = image_examples
    model = ImageOntologyModel(8, dataset.class_names, family=family, pizza_mode=mode, seed=17, classifier_seed=29)
    result = model(features, batch.sample_ids)
    assert result.sample_ids == batch.sample_ids
    assert result.component_ids == features.component_ids
    assert result.component_valid_mask.equal(features.component_valid_mask)
    assert result.component_classifier.contains.shape == (*result.component_valid_mask.shape, len(dataset.class_names))
    assert result.pizza_classifier.contains.shape == (len(batch.sample_ids), len(dataset.class_names))
    terms = image_objective_terms(result, batch, family=family, geometry_loss="signed_bce")
    objective = aggregate_effective_batch([terms])
    assert torch.isfinite(objective.total)
    objective.total.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.classifier.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.component_projection.parameters())
    if mode in ("compositional", "hybrid"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.composition.parameters())
    if mode in ("global", "hybrid"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.global_projection.parameters())


def test_hybrid_pizza_supervision_alone_reaches_deepsets_and_component_projection(image_examples):
    dataset, batch, features, _ = image_examples
    for family in ("baseline", "box", "town"):
        model = ImageOntologyModel(8, dataset.class_names, family=family, pizza_mode="hybrid", seed=7, classifier_seed=31)
        result = model(features, batch.sample_ids)
        terms = image_objective_terms(result, batch, family=family, geometry_loss="signed_bce")
        pizza_only = aggregate_effective_batch([terms], component_coefficient=0.0, pizza_coefficient=1.0)
        pizza_only.total.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.composition.parameters()), family
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.component_projection.parameters()), family


def test_common_neural_modules_are_byte_identical_and_classifiers_are_independent(image_examples):
    dataset, _, _, _ = image_examples
    models = [ImageOntologyModel(2, dataset.class_names, family=family, pizza_mode="hybrid", seed=88, classifier_seed=101) for family in ("baseline", "box", "town")]
    states = [model.common_state_dict() for model in models]
    assert states[0].keys() == states[1].keys() == states[2].keys()
    assert all(torch.equal(states[0][key], states[i][key]) for i in (1, 2) for key in states[0])
    assert len({id(model.classifier) for model in models}) == 3
    assert [model.classifier.family for model in models] == ["baseline", "box", "town"]


def test_deepsets_is_permutation_invariant_multiplicity_sensitive_and_empty_safe():
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


def test_zero_roi_models_produce_rho_zero_pizza_point(image_examples):
    dataset, batch, _, extractor = image_examples
    boxes = tuple(torch.empty((0, 4), dtype=torch.float32) for _ in batch.sample_ids)
    identities = tuple(() for _ in batch.sample_ids)
    with torch.no_grad():
        features = extractor(batch, ExplicitROIProvider(boxes, identities))
    for family in ("baseline", "box", "town"):
        model = ImageOntologyModel(2, dataset.class_names, family=family, pizza_mode="compositional")
        result = model(features, batch.sample_ids)
        assert result.component_points.shape == (len(batch.sample_ids), 0, 2)
        assert result.component_classifier.contains.shape == (len(batch.sample_ids), 0, len(dataset.class_names))
        assert torch.equal(result.composed_pizza_points, model.composition.rho(torch.zeros((len(batch.sample_ids), 128))))


def test_effective_batch_aggregation_matches_full_batch_gradient_with_heterogeneous_eligibility():
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
    # The last microbatch has one image, exercising a partial final window.
    for start, end in ((0, 2), (2, 4), (4, 5)):
        micros.append(ImageObjectiveTerms(
            component_split[start:end], component_eligible[start:end],
            pizza_split[start:end], pizza_eligible[start:end],
            int(component_eligible[start:end].sum()), int(pizza_eligible[start:end].sum()),
        ))
    accumulated_loss = aggregate_effective_batch(micros).total
    accumulated_grad = torch.autograd.grad(accumulated_loss, (component_split, pizza_split))
    assert torch.allclose(full_loss, accumulated_loss)
    assert torch.allclose(full_grad[0], accumulated_grad[0])
    assert torch.allclose(full_grad[1], accumulated_grad[1])


def test_image_objective_masks_unknowns_uses_positive_weights_and_omits_empty_components(image_examples):
    dataset, batch, features, _ = image_examples
    # A real model result supplies class scores; the reducer must honor the
    # actual dataset masks, including no fabricated loss for empty/unknown rows.
    model = ImageOntologyModel(2, dataset.class_names, family="baseline", pizza_mode="global")
    result = model(features, batch.sample_ids)
    terms = image_objective_terms(
        result, batch, family="baseline", geometry_loss="signed_bce",
        component_positive_weights=torch.full((len(dataset.class_names),), 3.0),
        pizza_positive_weights=torch.ones((len(dataset.class_names),)),
    )
    assert terms.component_eligible_images.sum().item() <= len(batch.sample_ids)
    assert terms.pizza_eligible_images.shape == (len(batch.sample_ids),)
    assert terms.component_instance_count <= sum(len(row) for row in batch.component_ids)
    result2 = ImageModelResult(
        sample_ids=result.sample_ids,
        component_ids=result.component_ids,
        component_valid_mask=torch.zeros_like(result.component_valid_mask),
        component_points=result.component_points,
        global_pizza_points=result.global_pizza_points,
        composed_pizza_points=result.composed_pizza_points,
        final_pizza_points=result.final_pizza_points,
        component_classifier=ClassifierOutput(contains=result.component_classifier.contains, logits=result.component_classifier.logits),
        pizza_classifier=result.pizza_classifier,
        pizza_mode=result.pizza_mode,
    )
    empty_terms = image_objective_terms(result2, batch, family="baseline", geometry_loss="signed_bce")
    assert empty_terms.component_instance_count == 0
    reduced = aggregate_effective_batch([empty_terms])
    assert reduced.component_loss is None
    assert "component" in reduced.undefined_tasks


def test_objective_undefined_for_entire_effective_batch_is_actionable():
    values = torch.zeros(2)
    mask = torch.zeros(2, dtype=torch.bool)
    terms = ImageObjectiveTerms(values, mask, values, mask, 0, 0)
    with pytest.raises(UndefinedImageObjective, match="no supervised"):
        aggregate_effective_batch([terms])


def test_hand_computed_known_mask_positive_weight_and_per_image_component_reduction():
    component_logits = torch.tensor([[[2.0, 2.0], [-2.0, 0.0]], [[-9.0, 9.0], [0.0, 0.0]]], requires_grad=True)
    pizza_logits = torch.tensor([[2.0, 99.0], [99.0, -99.0]], requires_grad=True)
    component_targets = (
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.0, 0.0]]),
    )
    component_known = (
        torch.tensor([[True, True], [True, False]]),
        torch.tensor([[False, False]]),
    )
    batch = SimpleNamespace(
        sample_ids=("many-components", "no-known-components"),
        pizza_targets=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        pizza_known_masks=torch.tensor([[True, False], [False, False]]),
        component_targets=component_targets,
        component_known_masks=component_known,
    )
    result = ImageModelResult(
        sample_ids=batch.sample_ids,
        component_ids=(("a", "b"), ("c", None)),
        component_valid_mask=torch.tensor([[True, True], [True, False]]),
        component_points=torch.zeros((2, 2, 2)),
        global_pizza_points=torch.zeros((2, 2)),
        composed_pizza_points=torch.zeros((2, 2)),
        final_pizza_points=torch.zeros((2, 2)),
        component_classifier=ClassifierOutput(
            contains=component_logits >= 0,
            logits=component_logits,
        ),
        pizza_classifier=ClassifierOutput(contains=pizza_logits >= 0, logits=pizza_logits),
        pizza_mode="global",
    )
    terms = image_objective_terms(
        result,
        batch,
        family="baseline",
        geometry_loss="signed_bce",
        component_positive_weights=torch.tensor([3.0, 2.0]),
        pizza_positive_weights=torch.tensor([3.0, 2.0]),
    )
    expected_positive = torch.nn.functional.softplus(torch.tensor(-2.0))
    expected_negative_inside = torch.nn.functional.softplus(torch.tensor(2.0))
    first_component = (3 * expected_positive + expected_negative_inside) / 2
    second_component = expected_positive
    assert terms.component_eligible_images.tolist() == [True, False]
    assert terms.component_per_image[0].item() == pytest.approx(((first_component + second_component) / 2).item())
    assert terms.component_per_image[1].item() == 0  # not eligible and excluded from the mean
    assert terms.pizza_eligible_images.tolist() == [True, False]
    assert terms.pizza_per_image[0].item() == pytest.approx((3 * expected_positive).item())
    objective = aggregate_effective_batch([terms])
    assert objective.component_eligible_images == 1
    assert objective.pizza_eligible_images == 1
    assert objective.total.item() == pytest.approx((terms.component_per_image[0] + terms.pizza_per_image[0]).item())
    objective.total.backward()
    assert component_logits.grad[0, 0, 0].abs() > 0
    assert component_logits.grad[0, 0, 1].abs() > 0
    assert component_logits.grad[0, 1, 1] == 0  # unknown second class
    assert component_logits.grad[1].abs().sum() == 0  # no eligible component for this image
    assert pizza_logits.grad[0, 1] == 0  # unknown target has no gradient
    assert pizza_logits.grad[1].abs().sum() == 0  # unknown pizza row is omitted
