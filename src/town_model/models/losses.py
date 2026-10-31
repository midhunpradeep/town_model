"""Generic known-mask reductions for caller-supplied values.

These reducers do not define score conversion, a classifier loss, or an approved
training objective. Their inputs are values supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class SupervisionReduction:
    """Result of reducing caller-supplied values, not a selected loss."""

    loss: Tensor
    per_instance: Tensor
    supervised_instances: Tensor
    known_count: int
    positive_count: int
    negative_count: int
    unknown_count: int


def reduce_supervision(
    values: Tensor,
    targets: Tensor,
    known_mask: Tensor,
    *,
    positive_weights: Tensor | None = None,
    trainable_mask: Tensor | None = None,
) -> SupervisionReduction:
    """Normalize supplied values over known trainable classes, then average.

    Positive weights affect known positives only; a known negative always has
    weight one. Unknown labels and fixed/unsupported classes contribute neither
    numerator nor denominator. This reduction does not define scoring or a
    training objective.
    """
    if not isinstance(values, Tensor) or values.ndim != 2:
        raise ValueError("values must have shape [N,C]")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    shape = values.shape
    if targets.shape != shape or known_mask.shape != shape:
        raise ValueError("targets and known_mask must match values [N,C]")
    if known_mask.dtype != torch.bool:
        raise ValueError("known_mask must have bool dtype")
    target_values = targets.to(device=values.device, dtype=torch.float32)
    mask = known_mask.to(device=values.device)
    if torch.any(mask & ((target_values != 0.0) & (target_values != 1.0))):
        raise ValueError("known targets must be binary 0/1")
    classes = shape[1]
    if trainable_mask is None:
        active_class = torch.ones((classes,), dtype=torch.bool, device=values.device)
    else:
        if trainable_mask.shape != (classes,):
            raise ValueError("trainable_mask must have shape [C]")
        active_class = trainable_mask.to(device=values.device, dtype=torch.bool)
    mask = mask & active_class.unsqueeze(0)

    if positive_weights is None:
        positive_weight = values.new_ones((classes,))
    else:
        if positive_weights.shape != (classes,):
            raise ValueError("positive_weights must have shape [C]")
        positive_weight = positive_weights.to(device=values.device, dtype=torch.float32)
        if not torch.isfinite(positive_weight).all() or torch.any(positive_weight <= 0):
            raise ValueError("positive_weights must be finite and strictly positive")

    weights = torch.where(target_values == 1.0, positive_weight.unsqueeze(0), 1.0)
    weights = weights * mask.to(dtype=torch.float32)
    denominator = weights.sum(dim=1)
    numerator = (values * weights).sum(dim=1)
    supervised = denominator > 0
    per_instance = torch.where(supervised, numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny), 0.0)
    if supervised.any():
        loss = per_instance[supervised].mean()
    else:
        # Connected zero makes backward safe while carrying no supervised gradient.
        loss = values.sum() * 0.0
    known_count = int(mask.sum().item())
    positive_count = int(((target_values == 1.0) & mask).sum().item())
    negative_count = int(((target_values == 0.0) & mask).sum().item())
    unknown_count = int((~known_mask.to(device=values.device)).sum().item())
    return SupervisionReduction(loss, per_instance, supervised, known_count, positive_count, negative_count, unknown_count)
