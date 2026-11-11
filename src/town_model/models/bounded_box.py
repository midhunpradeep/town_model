"""Pure exact Boolean membership for one fixed, bounded axis-aligned box."""

from __future__ import annotations

import torch
from torch import Tensor


def _require_float32_tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if value.layout != torch.strided:
        raise TypeError(f"{name} must be a dense strided tensor")
    return value


def _all_true(value: Tensor) -> bool:
    return bool(value.all().item())


def _validate_bounded_box_inputs(
    points: Tensor, lower: Tensor, upper: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate the shared inputs for the fixed bounded-box evaluators."""
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


def bounded_box_contains(points: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Return exact closed-box membership for float32 points in ``[-1, 1]^d``.

    ``points`` has shape ``[N, d]`` and both corners have shape ``[d]``.
    All inputs must be finite float32 dense tensors on the same CPU or CUDA
    device. The physical corners must be ordered and lie inside the fixed
    domain. Every box face is included, including coincident faces.
    """
    points, lower, upper = _validate_bounded_box_inputs(points, lower, upper)

    in_domain = (points >= -1.0) & (points <= 1.0)
    in_box = (points >= lower) & (points <= upper)
    return (in_domain & in_box).all(dim=1)


def box_signed_clearance(points: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Return signed coordinate clearance to one fixed bounded physical box.

    The result is ``min_j(points[:, j] - lower[j], upper[j] - points[:, j])``
    in raw physical units, not Euclidean distance. Queries may be any finite
    float32 values, including points outside ``[-1, 1]^d``. The minimum uses
    ``torch.amin``; PyTorch distributes its backward gradient equally among
    all exactly tied minima. This framework-defined tie behavior is retained.
    """
    points, lower, upper = _validate_bounded_box_inputs(points, lower, upper)

    face_clearances = torch.cat((points - lower, upper - points), dim=1)
    return torch.amin(face_clearances, dim=1)
