from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from town_model.contracts import ImageModelResult, VisionFeatureBatch
from town_model.dataset.image_data import deterministic_subset, load_prepared_dataset
from town_model.models import BoxClassifier, ClassifierOutput, DeepSetsComposition, ImageOntologyModel, TownClassifier
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


@pytest.mark.parametrize("family", ["baseline", "box", "town"])
@pytest.mark.parametrize("mode", ["global", "compositional", "hybrid"])
def test_pizza_objective_has_gradient_paths_to_mode_branches_and_shared_classifier(family, mode, image_examples):
    dataset, batch, features, _ = image_examples
    model = ImageOntologyModel(8, dataset.class_names, family=family, pizza_mode=mode, seed=7, classifier_seed=31)
    result = model(features, batch.sample_ids)
    scores = result.pizza_classifier.logits if family == "baseline" else result.pizza_classifier.signed_margin
    assert scores is not None
    # Select known targets opposite the current classifier decision so every
    # supported class has a nonzero task penalty; unknowns remain masked.
    targets = (scores.detach() < 0).to(dtype=batch.pizza_targets.dtype)
    pizza_only_batch = replace(batch, pizza_targets=targets)
    terms = image_objective_terms(
        result, pizza_only_batch, family=family, geometry_loss="signed_distance"
    )
    loss = aggregate_effective_batch([terms], component_coefficient=0.0, pizza_coefficient=1.0)
    assert loss.pizza_loss is not None and loss.pizza_loss.item() > 0
    loss.total.backward()

    def active(module):
        return any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in module.parameters())

    assert active(model.classifier), family
    if mode in ("global", "hybrid"):
        assert active(model.global_projection), (family, mode)
    else:
        assert not active(model.global_projection), (family, mode)
    if mode in ("compositional", "hybrid"):
        assert active(model.composition), (family, mode)
        assert active(model.component_projection), (family, mode)
    else:
        assert not active(model.composition), (family, mode)
        assert not active(model.component_projection), (family, mode)
    assert active(model.fusion) == (mode == "hybrid"), (family, mode)


def test_component_classifier_reshaping_preserves_box_parameters_and_reshapes_town_activity():
    from town_model.models.image_embeddings import _reshape_classifier_output

    points = torch.zeros((6, 2), dtype=torch.float32)
    box = _reshape_classifier_output(BoxClassifier(2, ("A", "B"), seed=7)(points, diagnostics=True), 2, 3)
    assert box.contains.shape == (2, 3, 2)
    assert box.diagnostics is not None
    assert box.diagnostics["box_low"].shape == (2, 2)
    assert box.diagnostics["box_high"].shape == (2, 2)

    town = _reshape_classifier_output(TownClassifier(2, ("A", "B"), seed=7)(points, diagnostics=True), 2, 3)
    assert town.contains.shape == (2, 3, 2)
    assert town.diagnostics is not None
    assert town.diagnostics["frame_low"].shape == (2, 2, 2)
    assert town.diagnostics["house_margin"].shape == (2, 3, 2, 2)
    assert town.diagnostics["window_contains"].shape == (2, 3, 2, 2, 1)


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


def test_default_family_objective_penalties_and_signed_distance_hinge_are_numerically_exact():
    from town_model.training.image_objectives import _penalties

    targets = torch.tensor([[1.0, 0.0]])
    margins = torch.tensor([[0.02, 0.05]], requires_grad=True)
    hinge = _penalties(
        margins, targets, family="box", geometry_loss="signed_distance", temperature=0.1, hinge_margin=0.1
    )
    assert hinge.flatten().tolist() == pytest.approx([0.08, 0.15])
    hinge.sum().backward()
    assert torch.allclose(margins.grad, torch.tensor([[-1.0, 1.0]]))

    zero_margin = _penalties(
        torch.zeros((1, 2)), targets, family="town", geometry_loss="signed_distance", temperature=0.1, hinge_margin=0.0
    )
    assert zero_margin.tolist() == [[0.0, 0.0]]
    baseline = _penalties(
        torch.zeros((1, 2)), targets, family="baseline", geometry_loss="signed_distance", temperature=0.1, hinge_margin=0.1
    )
    assert baseline.flatten().tolist() == pytest.approx([0.6931472, 0.6931472])


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


def test_weighted_instance_reduction_uses_sum_of_known_class_weights_and_masks_unknown_gradients():
    from town_model.training.image_objectives import _instance_terms

    penalties = torch.tensor([[2.0, 10.0, 4.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    known = torch.tensor([[True, True, False]])
    values, eligible = _instance_terms(
        penalties, targets, known, positive_weights=torch.tensor([9.0, 1.0, 100.0])
    )
    # Contract: (9*2 + 1*10) / (9+1) = 2.8. The former count
    # denominator gave 14, a five-fold scale change for this known negative.
    assert values.item() == pytest.approx(2.8)
    assert values.item() != pytest.approx((9 * 2 + 10) / 2)
    assert eligible.tolist() == [True]
    values.sum().backward()
    assert torch.allclose(penalties.grad, torch.tensor([[0.9, 0.1, 0.0]]))

    changed_unknown = torch.tensor([[2.0, 10.0, 1_000_000.0]], requires_grad=True)
    changed_target = torch.tensor([[1.0, 0.0, 1.0]])
    changed_values, _ = _instance_terms(
        changed_unknown, changed_target, known, positive_weights=torch.tensor([9.0, 1.0, 100.0])
    )
    changed_values.sum().backward()
    assert changed_values.item() == pytest.approx(values.item())
    assert torch.allclose(changed_unknown.grad, torch.tensor([[0.9, 0.1, 0.0]]))


def test_weighted_effective_batch_accumulation_matches_full_batch_gradients():
    from town_model.training.image_objectives import _instance_terms

    weights = torch.tensor([9.0, 1.0, 3.0])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    known = torch.tensor([[True, True, False], [True, True, True], [True, False, True], [False, False, False]])
    full_penalties = torch.tensor([[2.0, 10.0, 7.0], [4.0, 1.0, 9.0], [3.0, 6.0, 2.0], [50.0, 40.0, 30.0]], requires_grad=True)
    full_values, full_eligible = _instance_terms(full_penalties, targets, known, positive_weights=weights)
    full_terms = ImageObjectiveTerms(full_values, full_eligible, full_values.detach(), torch.zeros(4, dtype=torch.bool), 3, 0)
    full_loss = aggregate_effective_batch([full_terms], component_coefficient=1.0, pizza_coefficient=0.0).total
    full_gradient = torch.autograd.grad(full_loss, full_penalties)[0]

    split_penalties = full_penalties.detach().clone().requires_grad_()
    windows = []
    for indices in ((0,), (1, 2), (3,)):
        index = torch.tensor(indices)
        values, eligible = _instance_terms(
            split_penalties.index_select(0, index),
            targets.index_select(0, index),
            known.index_select(0, index),
            positive_weights=weights,
        )
        windows.append(ImageObjectiveTerms(values, eligible, values.detach(), torch.zeros(len(indices), dtype=torch.bool), int(eligible.sum()), 0))
    split_loss = aggregate_effective_batch(windows, component_coefficient=1.0, pizza_coefficient=0.0).total
    split_gradient = torch.autograd.grad(split_loss, split_penalties)[0]
    assert torch.allclose(full_loss, split_loss)
    assert torch.allclose(full_gradient, split_gradient)
    assert torch.equal(full_gradient[3], torch.zeros_like(full_gradient[3]))


def test_all_unit_image_weights_reproduce_unweighted_known_label_mean():
    from town_model.training.image_objectives import _instance_terms

    penalties = torch.tensor([[1.0, 3.0, 99.0], [4.0, 8.0, 10.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    known = torch.tensor([[True, True, False], [True, False, True]])
    unweighted, _ = _instance_terms(penalties, targets, known, positive_weights=None)
    unit_weighted, _ = _instance_terms(penalties, targets, known, positive_weights=torch.ones(3))
    assert torch.equal(unweighted, unit_weighted)
    assert unweighted.tolist() == pytest.approx([2.0, 7.0])


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
    first_component = (3 * expected_positive + expected_negative_inside) / 4
    second_component = expected_positive
    assert terms.component_eligible_images.tolist() == [True, False]
    assert terms.component_per_image[0].item() == pytest.approx(((first_component + second_component) / 2).item())
    assert terms.component_per_image[1].item() == 0  # not eligible and excluded from the mean
    assert terms.pizza_eligible_images.tolist() == [True, False]
    assert terms.pizza_per_image[0].item() == pytest.approx(expected_positive.item())
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
