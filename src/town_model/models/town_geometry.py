"""Pure strict ambient interior membership for one bounded physical box."""

from __future__ import annotations

import torch
from torch import Tensor


def _require_float32_tensor(value: object, name: str) -> Tensor:
    """Apply the public tensor-type conventions used by BOX-01 locally."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if value.layout != torch.strided:
        raise TypeError(f"{name} must be a dense strided tensor")
    return value


def _all_true(value: Tensor) -> bool:
    return bool(value.all().item())


def _validate_town_box_inputs(
    points: Tensor, lower: Tensor, upper: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate a single physical box without depending on BOX-01 internals."""
    points = _require_float32_tensor(points, "points")
    lower = _require_float32_tensor(lower, "lower")
    upper = _require_float32_tensor(upper, "upper")

    if points.ndim != 2:
        raise ValueError("points must have exact shape [N, d]")
    if lower.ndim != 1 or upper.ndim != 1:
        raise ValueError("lower and upper must have exact shape [d]")

    dimensions = points.shape[1]
    if dimensions < 1:
        raise ValueError("the point dimension d must be at least 1")
    if lower.shape != (dimensions,) or upper.shape != (dimensions,):
        raise ValueError("lower and upper must each have shape [d] matching points")

    if points.device != lower.device or points.device != upper.device:
        raise ValueError("points, lower, and upper must be on the same device")
    if points.device.type not in ("cpu", "cuda"):
        raise ValueError("inputs must use a supported CPU or CUDA device")
    if points.device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    if not _all_true(torch.isfinite(points)):
        raise ValueError("points must contain only finite values")
    if not _all_true(torch.isfinite(lower)) or not _all_true(torch.isfinite(upper)):
        raise ValueError("box corners must contain only finite values")
    if not _all_true(lower >= -1.0) or not _all_true(upper <= 1.0):
        raise ValueError("box corners must lie inside the fixed domain [-1, 1]^d")
    if not _all_true(lower <= upper):
        raise ValueError("lower must be less than or equal to upper in every coordinate")

    return points, lower, upper


def box_strict_interior(points: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Return exact ambient strict-interior membership for one physical box.

    ``points`` has shape ``[N, d]`` and ``lower``/``upper`` have shape ``[d]``.
    All tensors must be finite float32 dense-strided values on the same CPU or
    CUDA device. Physical corners must be ordered and lie in ``[-1, 1]^d``.
    Finite points outside that domain remain valid pure queries. The result is
    Boolean ``[N]`` and is true exactly when every coordinate is strictly
    between its corresponding physical corners. Faces and degenerate boxes
    therefore have empty ambient interior. Comparisons are direct: no
    clearance, tolerance, subtraction, casting, or repair is used.
    """
    points, lower, upper = _validate_town_box_inputs(points, lower, upper)
    return ((points > lower) & (points < upper)).all(dim=1)
