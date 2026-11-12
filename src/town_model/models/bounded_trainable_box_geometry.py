"""CP1 pure bounded width-first geometry and physical invariant checks.

This module does not implement inversion, initialization, trainable parameters,
losses, or optimizer behavior. Finite-precision saturation and conditioning are
diagnostics distinct from physical validity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class BoundedBoxGeometry:
    """Float32 physical values decoded from aligned raw ``[C, d]`` tensors."""

    centers: Tensor
    half_widths: Tensor
    lower: Tensor
    upper: Tensor
    minimum_full_width: float


def ordinary_minimum_full_width(epsilon: float) -> float:
    """Return the ordinary-box minimum full width ``epsilon``."""
    epsilon = _positive_finite_scalar(epsilon, "epsilon")
    if epsilon >= 2.0:
        raise ValueError("epsilon must be less than the domain width 2")
    return epsilon


def hosting_frame_minimum_full_width(epsilon: float, delta: float) -> float:
    """Return the hosting-frame width floor ``epsilon + 2 * delta``."""
    epsilon = _positive_finite_scalar(epsilon, "epsilon")
    delta = _positive_finite_scalar(delta, "delta")
    minimum_full_width = epsilon + 2.0 * delta
    if minimum_full_width >= 2.0:
        raise ValueError("epsilon + 2 * delta must be less than the domain width 2")
    return minimum_full_width


def _positive_finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _minimum_width(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("minimum_full_width must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 2.0:
        raise ValueError("minimum_full_width must satisfy 0 < w < 2")
    return result


def _require_raw_pair(raw_width: object, raw_center: object) -> tuple[Tensor, Tensor]:
    if not isinstance(raw_width, Tensor) or not isinstance(raw_center, Tensor):
        raise TypeError("raw_width and raw_center must be torch.Tensor values")
    if raw_width.dtype != torch.float32 or raw_center.dtype != torch.float32:
        raise TypeError("raw_width and raw_center must both have dtype torch.float32")
    if raw_width.layout != torch.strided or raw_center.layout != torch.strided:
        raise TypeError("raw_width and raw_center must be dense strided tensors")
    if raw_width.ndim != 2 or raw_center.ndim != 2:
        raise ValueError("raw_width and raw_center must have exact shape [C, d]")
    if raw_width.shape != raw_center.shape:
        raise ValueError("raw_width and raw_center must have identical [C, d] shapes")
    if raw_width.shape[1] < 1:
        raise ValueError("the point dimension d must be at least 1")
    if raw_width.device != raw_center.device:
        raise ValueError("raw_width and raw_center must be on the same device")
    if raw_width.device.type not in ("cpu", "cuda"):
        raise ValueError("raw tensors must use a supported CPU or CUDA device")
    if raw_width.device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if not bool(torch.isfinite(raw_width).all().item()):
        raise ValueError("raw_width must contain only finite values")
    if not bool(torch.isfinite(raw_center).all().item()):
        raise ValueError("raw_center must contain only finite values")
    return raw_width, raw_center


def decode_bounded_box_raw(
    raw_width: Tensor,
    raw_center: Tensor,
    *,
    minimum_full_width: float,
) -> BoundedBoxGeometry:
    """Apply the approved width-first map without clipping or repairing output.

    Inputs are aligned float32 tensors of shape ``[C, d]`` with ``d >= 1``.
    ``minimum_full_width`` is ``epsilon`` for ordinary boxes or
    ``epsilon + 2 * delta`` for hosting frames. This function deliberately
    returns actual computed values without silently repairing finite-precision
    invariant failures; callers independently invoke
    :func:`validate_bounded_box_geometry`.
    """
    minimum_full_width = _minimum_width(minimum_full_width)
    raw_width, raw_center = _require_raw_pair(raw_width, raw_center)

    half_widths = (
        minimum_full_width
        + (2.0 - minimum_full_width) * torch.sigmoid(raw_width)
    ) / 2.0
    centers = (1.0 - half_widths) * torch.tanh(raw_center)
    lower = centers - half_widths
    upper = centers + half_widths
    return BoundedBoxGeometry(
        centers=centers,
        half_widths=half_widths,
        lower=lower,
        upper=upper,
        minimum_full_width=minimum_full_width,
    )


def validate_bounded_box_geometry(geometry: BoundedBoxGeometry) -> None:
    """Validate actual stored float32 centers, half-widths, and corners exactly.

    No tolerance is used for domain, ordering, endpoint-width, or half-width
    inequalities. No value is cast, clamped, or repaired. Raises
    ``TypeError``/``ValueError`` on invalid physical state; poor gradients or
    saturated raw coordinates do not by themselves make physical geometry
    invalid.
    """
    if not isinstance(geometry, BoundedBoxGeometry):
        raise TypeError("geometry must be a BoundedBoxGeometry")
    minimum_full_width = _minimum_width(geometry.minimum_full_width)
    centers, half_widths, lower, upper = _require_physical_fields(geometry)

    for name, value in (
        ("centers", centers),
        ("half_widths", half_widths),
        ("lower", lower),
        ("upper", upper),
    ):
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain only finite values")

    if not bool((half_widths > 0.0).all().item()):
        raise ValueError("half_widths must be strictly positive")
    # Compare the exact real values represented by stored float32 tensors to
    # the configured Python scalar; a float32-rounded scalar comparison could
    # silently admit a sub-floor physical width.
    if not bool((half_widths.to(torch.float64) >= minimum_full_width / 2.0).all().item()):
        raise ValueError("half_widths violate the configured minimum full width")

    if not bool((lower <= upper).all().item()):
        raise ValueError("lower corners must not exceed upper corners")
    if not bool((lower >= -1.0).all().item()) or not bool((upper <= 1.0).all().item()):
        raise ValueError("physical corners must lie inside [-1, 1]^d")

    expected_lower = centers - half_widths
    expected_upper = centers + half_widths
    if not torch.equal(lower, expected_lower) or not torch.equal(upper, expected_upper):
        raise ValueError("corners must exactly equal centers minus/plus half_widths")

    actual_full_widths = upper.to(torch.float64) - lower.to(torch.float64)
    if not bool((actual_full_widths >= minimum_full_width).all().item()):
        raise ValueError("stored endpoint widths violate the configured minimum full width")


def _require_physical_fields(
    geometry: BoundedBoxGeometry,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    fields = {
        "centers": geometry.centers,
        "half_widths": geometry.half_widths,
        "lower": geometry.lower,
        "upper": geometry.upper,
    }
    for name, value in fields.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32")
        if value.layout != torch.strided:
            raise TypeError(f"{name} must be a dense strided tensor")
        if value.ndim != 2:
            raise ValueError(f"{name} must have exact shape [C, d]")
        if value.shape[1] < 1:
            raise ValueError("the point dimension d must be at least 1")

    shape = geometry.centers.shape
    device = geometry.centers.device
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have the same [C, d] shape as centers")
        if value.device != device:
            raise ValueError("all physical fields must be on the same device")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("physical fields must use a supported CPU or CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    return geometry.centers, geometry.half_widths, geometry.lower, geometry.upper


def analytic_endpoint_jacobian(
    raw_width: Tensor,
    raw_center: Tensor,
    *,
    minimum_full_width: float,
) -> Tensor:
    """Return the analytic local ``(lower, upper)`` by ``(a, b)`` Jacobian.

    Output shape is ``[C, d, 2, 2]``. The final axes are ordered as
    ``(lower, upper)`` and ``(raw_width, raw_center)`` respectively.
    """
    minimum_full_width = _minimum_width(minimum_full_width)
    raw_width, raw_center = _require_raw_pair(raw_width, raw_center)
    k = (2.0 - minimum_full_width) / 2.0
    p = torch.sigmoid(raw_width)
    u = torch.tanh(raw_center)
    half_widths = (minimum_full_width + (2.0 - minimum_full_width) * p) / 2.0
    half_width_raw = k * p * (1.0 - p)
    center_raw_width = -u * half_width_raw
    center_raw_center = (1.0 - half_widths) * (1.0 - u.square())

    lower_raw_width = center_raw_width - half_width_raw
    upper_raw_width = center_raw_width + half_width_raw
    return torch.stack(
        (
            torch.stack((lower_raw_width, center_raw_center), dim=-1),
            torch.stack((upper_raw_width, center_raw_center), dim=-1),
        ),
        dim=-2,
    )
