"""Pure bounded multi-box membership and signed-clearance evaluators."""

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


def _validate_bounded_multi_box_inputs(
    points: Tensor, lower: Tensor, upper: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate exact multi-box input shapes and physical geometry."""
    points = _require_float32_tensor(points, "points")
    lower = _require_float32_tensor(lower, "lower")
    upper = _require_float32_tensor(upper, "upper")

    if points.ndim != 2:
        raise ValueError("points must have exact shape [N, d]")
    if lower.ndim != 2 or upper.ndim != 2:
        raise ValueError("lower and upper must have exact shape [C, d]")

    dimensions = points.shape[1]
    if dimensions < 1:
        raise ValueError("the point dimension d must be at least 1")
    if lower.shape[1] != dimensions or upper.shape[1] != dimensions:
        raise ValueError("corner matrices must have coordinate dimension d matching points")
    if lower.shape[0] != upper.shape[0]:
        raise ValueError("lower and upper must have the same box count C")

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


def bounded_multi_box_contains(points: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Return exact closed membership for each point and bounded physical box.

    ``points`` has shape ``[N, d]`` and ``lower``/``upper`` have shape
    ``[C, d]``. Inputs are finite float32 dense-strided tensors on one CPU or
    CUDA device; ``d >= 1`` and physical corners are ordered within
    ``[-1, 1]^d``. Finite query points outside the domain are accepted and
    have false membership in every box column.
    """
    points, lower, upper = _validate_bounded_multi_box_inputs(points, lower, upper)

    in_domain = ((points >= -1.0) & (points <= 1.0)).unsqueeze(1)
    points_by_box = points.unsqueeze(1)
    lower_by_box = lower.unsqueeze(0)
    upper_by_box = upper.unsqueeze(0)
    in_physical_boxes = (points_by_box >= lower_by_box) & (points_by_box <= upper_by_box)
    return (in_domain & in_physical_boxes).all(dim=-1)


def multi_box_signed_clearance(points: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Return raw signed coordinate clearance for every point/box pair.

    The result is ``min_j(points[i, j] - lower[c, j], upper[c, j] -
    points[i, j])`` with shape ``[N, C]``. It is a coordinate clearance in
    physical units, not Euclidean distance. Finite queries outside
    ``[-1, 1]^d`` are accepted without clipping. A single ``torch.amin``
    reduces the ``2d`` faces for each pair independently; PyTorch's framework
    tie-gradient behavior is preserved.
    """
    points, lower, upper = _validate_bounded_multi_box_inputs(points, lower, upper)

    points_by_box = points.unsqueeze(1)
    lower_by_box = lower.unsqueeze(0)
    upper_by_box = upper.unsqueeze(0)
    face_clearances = torch.cat(
        (points_by_box - lower_by_box, upper_by_box - points_by_box), dim=-1
    )
    return torch.amin(face_clearances, dim=-1)
