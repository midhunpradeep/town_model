"""Image-aware ontology losses and exact effective-batch aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from town_model.contracts import ImageBatch, ImageModelResult


class UndefinedImageObjective(ValueError):
    """The effective batch has no supervised observations for either task."""


@dataclass(frozen=True, slots=True)
class ImageObjectiveTerms:
    component_per_image: Tensor
    component_eligible_images: Tensor
    pizza_per_image: Tensor
    pizza_eligible_images: Tensor
    component_instance_count: int
    pizza_instance_count: int


@dataclass(frozen=True, slots=True)
class ImageObjectiveResult:
    total: Tensor
    component_loss: Tensor | None
    pizza_loss: Tensor | None
    component_eligible_images: int
    pizza_eligible_images: int
    undefined_tasks: tuple[str, ...]


def _penalties(scores: Tensor, targets: Tensor, *, family: str, geometry_loss: str, temperature: float, hinge_margin: float) -> Tensor:
    if family == "baseline":
        return F.binary_cross_entropy_with_logits(scores, targets.float(), reduction="none")
    signs = targets.float() * 2.0 - 1.0
    if geometry_loss == "signed_bce":
        return F.softplus(-signs * scores.float() / temperature)
    if geometry_loss == "signed_distance":
        return torch.relu(hinge_margin - signs * scores.float())
    raise ValueError(f"unsupported geometric objective {geometry_loss!r}")


def _instance_terms(
    penalties: Tensor,
    targets: Tensor,
    known: Tensor,
    *,
    positive_weights: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if penalties.shape != targets.shape or known.shape != targets.shape or known.dtype != torch.bool:
        raise ValueError("penalties, targets, and known masks must have matching [N,C] shape")
    target_values = targets.to(device=penalties.device, dtype=torch.float32)
    if torch.any(known & ((target_values != 0) & (target_values != 1))):
        raise ValueError("known ontology targets must be binary")
    if positive_weights is None:
        pos_weights = penalties.new_ones((penalties.shape[-1],))
    else:
        if positive_weights.shape != (penalties.shape[-1],):
            raise ValueError("positive weight vector must have one value per model class")
        pos_weights = positive_weights.to(device=penalties.device, dtype=torch.float32)
        if not torch.isfinite(pos_weights).all() or torch.any(pos_weights <= 0):
            raise ValueError("positive weights must be finite and positive")
    weights = torch.where(target_values == 1, pos_weights.unsqueeze(0), 1.0)
    weights = weights * known.to(dtype=torch.float32)
    denominator = weights.sum(dim=-1)
    eligible = denominator > 0
    values = (penalties * weights).sum(dim=-1) / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    return values, eligible


def image_objective_terms(
    result: ImageModelResult,
    batch: ImageBatch,
    *,
    family: str,
    geometry_loss: str,
    temperature: float = 0.1,
    hinge_margin: float = 0.1,
    class_column_indices: tuple[int, ...] | None = None,
    component_positive_weights: Tensor | None = None,
    pizza_positive_weights: Tensor | None = None,
) -> ImageObjectiveTerms:
    """Return graph-preserving per-image terms before effective-window means."""
    batch_size, padded_count = result.component_valid_mask.shape
    if batch_size != len(batch.sample_ids) or result.sample_ids != batch.sample_ids:
        raise ValueError("model result and target batch sample identities differ")
    indices = torch.tensor(class_column_indices, dtype=torch.long, device=batch.pizza_targets.device) if class_column_indices is not None else None
    pizza_targets = batch.pizza_targets if indices is None else batch.pizza_targets.index_select(1, indices)
    pizza_known = batch.pizza_known_masks if indices is None else batch.pizza_known_masks.index_select(1, indices)
    component_targets = batch.pizza_targets.new_zeros((batch_size, padded_count, pizza_targets.shape[1]))
    component_known = torch.zeros_like(component_targets, dtype=torch.bool)
    for image_index, (target, mask) in enumerate(zip(batch.component_targets, batch.component_known_masks, strict=True)):
        count = target.shape[0]
        if count > padded_count:
            raise ValueError("target component count exceeds model ROI count")
        if count:
            target = target.to(batch.pizza_targets.device)
            mask = mask.to(batch.pizza_known_masks.device)
            if indices is not None:
                target = target.index_select(1, indices)
                mask = mask.index_select(1, indices)
            component_targets[image_index, :count] = target
            component_known[image_index, :count] = mask
    component_known = component_known & result.component_valid_mask.unsqueeze(-1)

    component_output = result.component_classifier
    component_scores = component_output.logits if component_output.logits is not None else component_output.signed_margin
    pizza_output = result.pizza_classifier
    pizza_scores = pizza_output.logits if pizza_output.logits is not None else pizza_output.signed_margin
    if component_scores is None or pizza_scores is None:
        raise ValueError("classifier result must expose logits or signed margins")
    if component_scores.shape != component_targets.shape or pizza_scores.shape != pizza_targets.shape:
        raise ValueError("classifier output and selected ontology target vocabulary disagree")
    component_penalties = _penalties(
        component_scores, component_targets, family=family, geometry_loss=geometry_loss,
        temperature=temperature, hinge_margin=hinge_margin,
    )
    pizza_penalties = _penalties(
        pizza_scores, pizza_targets, family=family, geometry_loss=geometry_loss,
        temperature=temperature, hinge_margin=hinge_margin,
    )
    component_instances, component_eligible = _instance_terms(
        component_penalties.reshape(-1, component_penalties.shape[-1]),
        component_targets.reshape(-1, component_targets.shape[-1]),
        component_known.reshape(-1, component_known.shape[-1]),
        positive_weights=component_positive_weights,
    )
    component_instances = component_instances.reshape(batch_size, padded_count)
    component_eligible = component_eligible.reshape(batch_size, padded_count)
    per_image_counts = component_eligible.sum(dim=1)
    component_image_eligible = per_image_counts > 0
    component_per_image = (component_instances * component_eligible.to(component_instances.dtype)).sum(dim=1) / per_image_counts.clamp_min(1)
    pizza_per_image, pizza_eligible = _instance_terms(
        pizza_penalties,
        pizza_targets,
        pizza_known,
        positive_weights=pizza_positive_weights,
    )
    return ImageObjectiveTerms(
        component_per_image=component_per_image,
        component_eligible_images=component_image_eligible,
        pizza_per_image=pizza_per_image,
        pizza_eligible_images=pizza_eligible,
        component_instance_count=int(component_eligible.sum().item()),
        pizza_instance_count=int(pizza_eligible.sum().item()),
    )


def aggregate_effective_batch(
    windows: list[ImageObjectiveTerms] | tuple[ImageObjectiveTerms, ...],
    *,
    component_coefficient: float = 1.0,
    pizza_coefficient: float = 1.0,
) -> ImageObjectiveResult:
    """Aggregate once over all images in an accumulation window.

    Each defined task is averaged over its eligible images in the full window;
    no independently normalized microbatch means are averaged. A task with no
    effective-batch support is omitted and reported. If both tasks are
    undefined, the caller must diagnose/abort rather than train on zero loss.
    """
    if not windows:
        raise ValueError("effective batch must contain at least one microbatch")
    component_values = torch.cat([part.component_per_image for part in windows], dim=0)
    component_mask = torch.cat([part.component_eligible_images for part in windows], dim=0)
    pizza_values = torch.cat([part.pizza_per_image for part in windows], dim=0)
    pizza_mask = torch.cat([part.pizza_eligible_images for part in windows], dim=0)
    component_loss = component_values[component_mask].mean() if component_mask.any() else None
    pizza_loss = pizza_values[pizza_mask].mean() if pizza_mask.any() else None
    contributions = []
    undefined = []
    if component_coefficient > 0:
        if component_loss is None:
            undefined.append("component")
        else:
            contributions.append(component_loss * component_coefficient)
    if pizza_coefficient > 0:
        if pizza_loss is None:
            undefined.append("pizza")
        else:
            contributions.append(pizza_loss * pizza_coefficient)
    if not contributions:
        raise UndefinedImageObjective(
            "no supervised component or pizza observations in the effective accumulation batch"
        )
    total = torch.stack(contributions).sum()
    return ImageObjectiveResult(
        total=total,
        component_loss=component_loss,
        pizza_loss=pizza_loss,
        component_eligible_images=int(component_mask.sum().item()),
        pizza_eligible_images=int(pizza_mask.sum().item()),
        undefined_tasks=tuple(undefined),
    )
