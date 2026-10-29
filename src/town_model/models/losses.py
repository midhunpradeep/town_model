"""Shared unreduced classifier penalties and masked per-instance reduction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class SupervisionReduction:
    loss: Tensor
    per_instance: Tensor
    supervised_instances: Tensor
    known_count: int
    positive_count: int
    negative_count: int
    unknown_count: int


def reduce_supervision(
    penalties: Tensor,
    targets: Tensor,
    known_mask: Tensor,
    *,
    positive_weights: Tensor | None = None,
    trainable_mask: Tensor | None = None,
) -> SupervisionReduction:
    """Normalize each instance over its known trainable classes, then average.

    Positive weights affect known positives only; a known negative always has
    weight one. Unknown labels and fixed/unsupported classes contribute neither
    numerator nor denominator. Instances without active targets are excluded.
    """
    if not isinstance(penalties, Tensor) or penalties.ndim != 2:
        raise ValueError("penalties must have shape [N,C]")
    if not torch.isfinite(penalties).all():
        raise ValueError("penalties must be finite")
    shape = penalties.shape
    if targets.shape != shape or known_mask.shape != shape:
        raise ValueError("targets and known_mask must match penalties [N,C]")
    if known_mask.dtype != torch.bool:
        raise ValueError("known_mask must have bool dtype")
    target_values = targets.to(device=penalties.device, dtype=torch.float32)
    mask = known_mask.to(device=penalties.device)
    if torch.any(mask & ((target_values != 0.0) & (target_values != 1.0))):
        raise ValueError("known targets must be binary 0/1")
    classes = shape[1]
    if trainable_mask is None:
        active_class = torch.ones((classes,), dtype=torch.bool, device=penalties.device)
    else:
        if trainable_mask.shape != (classes,):
            raise ValueError("trainable_mask must have shape [C]")
        active_class = trainable_mask.to(device=penalties.device, dtype=torch.bool)
    mask = mask & active_class.unsqueeze(0)

    if positive_weights is None:
        positive_weight = penalties.new_ones((classes,))
    else:
        if positive_weights.shape != (classes,):
            raise ValueError("positive_weights must have shape [C]")
        positive_weight = positive_weights.to(device=penalties.device, dtype=torch.float32)
        if not torch.isfinite(positive_weight).all() or torch.any(positive_weight <= 0):
            raise ValueError("positive_weights must be finite and strictly positive")

    weights = torch.where(target_values == 1.0, positive_weight.unsqueeze(0), 1.0)
    weights = weights * mask.to(dtype=torch.float32)
    denominator = weights.sum(dim=1)
    numerator = (penalties * weights).sum(dim=1)
    supervised = denominator > 0
    per_instance = torch.where(supervised, numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny), 0.0)
    if supervised.any():
        loss = per_instance[supervised].mean()
    else:
        # Connected zero makes backward safe while carrying no supervised gradient.
        loss = penalties.sum() * 0.0
    known_count = int(mask.sum().item())
    positive_count = int(((target_values == 1.0) & mask).sum().item())
    negative_count = int(((target_values == 0.0) & mask).sum().item())
    unknown_count = int((~known_mask.to(device=penalties.device)).sum().item())
    return SupervisionReduction(loss, per_instance, supervised, known_count, positive_count, negative_count, unknown_count)


def validate_signed_distance_margin(margin: float) -> float:
    """Validate and return the shared signed-distance hinge margin."""
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise ValueError("signed-distance margin must be a finite nonnegative number")
    return float(margin)
