"""Legacy classifier result transport, separate from classifier implementations.

This record preserves the current adapters' output shape during teardown. Its
optional logits and signed-margin fields describe historical implementations;
they are not a required scoring representation for future classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class ClassifierOutput:
    """Historical predictions and optional family-specific diagnostic values."""

    contains: Tensor
    logits: Tensor | None = None
    signed_margin: Tensor | None = None
    diagnostics: dict[str, Tensor] | None = None
    fixed_positive: dict[str, Tensor] | None = None
