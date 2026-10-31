"""Generic mask-aware reductions and effective-batch value aggregation.

This module accepts caller-supplied per-class or per-image values. It does not
convert classifier scores, define a loss, or approve a final training objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


class UndefinedImageObjective(ValueError):
    """The supplied effective batch has no eligible observations."""


@dataclass(frozen=True, slots=True)
class ImageObjectiveTerms:
    """Caller-supplied per-image values and eligibility masks for aggregation only."""

    component_per_image: Tensor
    component_eligible_images: Tensor
    pizza_per_image: Tensor
    pizza_eligible_images: Tensor
    component_instance_count: int
    pizza_instance_count: int


@dataclass(frozen=True, slots=True)
class ImageObjectiveResult:
    """Aggregate of caller-supplied values; not a classifier objective contract."""

    total: Tensor
    component_loss: Tensor | None
    pizza_loss: Tensor | None
    component_eligible_images: int
    pizza_eligible_images: int
    undefined_tasks: tuple[str, ...]


def reduce_weighted_instances(
    values: Tensor,
    targets: Tensor,
    known: Tensor,
    *,
    positive_weights: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Reduce caller-supplied [N,C] values using known masks and weights."""
    if values.shape != targets.shape or known.shape != targets.shape or known.dtype != torch.bool:
        raise ValueError("values, targets, and known masks must have matching [N,C] shape")
    target_values = targets.to(device=values.device, dtype=torch.float32)
    if torch.any(known & ((target_values != 0) & (target_values != 1))):
        raise ValueError("known ontology targets must be binary")
    if positive_weights is None:
        pos_weights = values.new_ones((values.shape[-1],))
    else:
        if positive_weights.shape != (values.shape[-1],):
            raise ValueError("positive weight vector must have one value per class")
        pos_weights = positive_weights.to(device=values.device, dtype=torch.float32)
        if not torch.isfinite(pos_weights).all() or torch.any(pos_weights <= 0):
            raise ValueError("positive weights must be finite and positive")
    weights = torch.where(target_values == 1, pos_weights.unsqueeze(0), 1.0)
    weights = weights * known.to(dtype=torch.float32)
    denominator = weights.sum(dim=-1)
    eligible = denominator > 0
    reduced = (values * weights).sum(dim=-1) / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    return reduced, eligible


def aggregate_effective_batch(
    windows: list[ImageObjectiveTerms] | tuple[ImageObjectiveTerms, ...],
    *,
    component_coefficient: float = 1.0,
    pizza_coefficient: float = 1.0,
) -> ImageObjectiveResult:
    """Aggregate caller-supplied values once across a full accumulation window.

    Keeping this reducer does not approve the task split, coefficients, or loss
    semantics for any future classifier.
    """
    if not windows:
        raise ValueError("effective batch must contain at least one microbatch")
    component_values = torch.cat([part.component_per_image for part in windows], dim=0)
    component_mask = torch.cat([part.component_eligible_images for part in windows], dim=0)
    pizza_values = torch.cat([part.pizza_per_image for part in windows], dim=0)
    pizza_mask = torch.cat([part.pizza_eligible_images for part in windows], dim=0)
    component_value = component_values[component_mask].mean() if component_mask.any() else None
    pizza_value = pizza_values[pizza_mask].mean() if pizza_mask.any() else None
    contributions = []
    undefined = []
    if component_coefficient > 0:
        if component_value is None:
            undefined.append("component")
        else:
            contributions.append(component_value * component_coefficient)
    if pizza_coefficient > 0:
        if pizza_value is None:
            undefined.append("pizza")
        else:
            contributions.append(pizza_value * pizza_coefficient)
    if not contributions:
        raise UndefinedImageObjective("no eligible component or pizza observations in the effective batch")
    total = torch.stack(contributions).sum()
    return ImageObjectiveResult(
        total=total,
        component_loss=component_value,
        pizza_loss=pizza_value,
        component_eligible_images=int(component_mask.sum().item()),
        pizza_eligible_images=int(pizza_mask.sum().item()),
        undefined_tasks=tuple(undefined),
    )
