"""Pure Boolean membership for bounded boxes, houses, and towns."""

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


def _validate_house_inputs(
    points: Tensor,
    frame_lower: Tensor,
    frame_upper: Tensor,
    window_lower: Tensor,
    window_upper: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Validate exact shapes and physical geometry for one fixed house."""
    points = _require_float32_tensor(points, "points")
    frame_lower = _require_float32_tensor(frame_lower, "frame_lower")
    frame_upper = _require_float32_tensor(frame_upper, "frame_upper")
    window_lower = _require_float32_tensor(window_lower, "window_lower")
    window_upper = _require_float32_tensor(window_upper, "window_upper")

    if points.ndim != 2:
        raise ValueError("points must have exact shape [N, d]")
    if frame_lower.ndim != 1 or frame_upper.ndim != 1:
        raise ValueError("frame corners must have exact shape [d]")
    if window_lower.ndim != 2 or window_upper.ndim != 2:
        raise ValueError("window corners must have exact shape [W, d]")

    dimensions = points.shape[1]
    if dimensions < 1:
        raise ValueError("the point dimension d must be at least 1")
    if frame_lower.shape != (dimensions,) or frame_upper.shape != (dimensions,):
        raise ValueError("frame corners must each have shape [d] matching points")
    if window_lower.shape[1] != dimensions or window_upper.shape[1] != dimensions:
        raise ValueError("window corner matrices must have coordinate dimension d matching points")
    if window_lower.shape[0] != window_upper.shape[0]:
        raise ValueError("window corners must have the same window count W")

    tensors = (frame_lower, frame_upper, window_lower, window_upper)
    if any(tensor.device != points.device for tensor in tensors):
        raise ValueError("points and all corners must be on the same device")
    if points.device.type not in ("cpu", "cuda"):
        raise ValueError("inputs must use a supported CPU or CUDA device")
    if points.device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    if not _all_true(torch.isfinite(points)):
        raise ValueError("points must contain only finite values")
    corners = (frame_lower, frame_upper, window_lower, window_upper)
    if any(not _all_true(torch.isfinite(corner)) for corner in corners):
        raise ValueError("frame and window corners must contain only finite values")
    if any(not _all_true(lower >= -1.0) for lower in (frame_lower, window_lower)) or any(
        not _all_true(upper <= 1.0) for upper in (frame_upper, window_upper)
    ):
        raise ValueError("frame and window corners must lie inside the fixed domain [-1, 1]^d")
    if not _all_true(frame_lower <= frame_upper):
        raise ValueError("frame_lower must be less than or equal to frame_upper in every coordinate")
    if not _all_true(window_lower <= window_upper):
        raise ValueError("window_lower must be less than or equal to window_upper in every coordinate")

    return points, frame_lower, frame_upper, window_lower, window_upper


def house_contains(
    points: Tensor,
    frame_lower: Tensor,
    frame_upper: Tensor,
    window_lower: Tensor,
    window_upper: Tensor,
) -> Tensor:
    """Return exact closed-frame membership minus strict window interiors.

    ``points`` has shape ``[N, d]``; frame corners have shape ``[d]`` and
    window corners have shape ``[W, d]``. All inputs are finite float32
    dense-strided tensors on the same CPU or CUDA device. Physical corners
    must be ordered and bounded by ``[-1, 1]^d``. Finite outside-domain query
    points are valid and return false. Frame faces are included. A point is
    excluded only when it lies strictly inside at least one window in the
    ambient coordinate space, so window boundaries are retained unless
    another window's strict interior contains the point. ``W=0`` is an exact
    empty window axis. Comparisons are direct Boolean comparisons without
    clearance, tolerance, casting, clipping, or mutation.
    """
    points, frame_lower, frame_upper, window_lower, window_upper = _validate_house_inputs(
        points, frame_lower, frame_upper, window_lower, window_upper
    )

    in_domain = ((points >= -1.0) & (points <= 1.0)).all(dim=1)
    in_frame = ((points >= frame_lower) & (points <= frame_upper)).all(dim=1)
    points_by_window = points.unsqueeze(1)
    inside_windows = (
        (points_by_window > window_lower.unsqueeze(0))
        & (points_by_window < window_upper.unsqueeze(0))
    ).all(dim=-1)
    excluded = inside_windows.any(dim=1)
    return in_domain & in_frame & ~excluded


def _validate_town_inputs(
    points: Tensor,
    frame_lower: Tensor,
    frame_upper: Tensor,
    window_lower: Tensor,
    window_upper: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Validate exact stacked shapes and physical geometry for one town."""
    points = _require_float32_tensor(points, "points")
    frame_lower = _require_float32_tensor(frame_lower, "frame_lower")
    frame_upper = _require_float32_tensor(frame_upper, "frame_upper")
    window_lower = _require_float32_tensor(window_lower, "window_lower")
    window_upper = _require_float32_tensor(window_upper, "window_upper")

    if points.ndim != 2:
        raise ValueError("points must have exact shape [N, d]")
    if frame_lower.ndim != 2 or frame_upper.ndim != 2:
        raise ValueError("town frame corners must have exact shape [H, d]")
    if window_lower.ndim != 3 or window_upper.ndim != 3:
        raise ValueError("town window corners must have exact shape [H, W, d]")

    dimensions = points.shape[1]
    if dimensions < 1:
        raise ValueError("the point dimension d must be at least 1")
    if frame_lower.shape[0] < 1:
        raise ValueError("a town must contain at least one house (H >= 1)")
    if frame_lower.shape != frame_upper.shape:
        raise ValueError("frame_lower and frame_upper must have identical [H, d] shapes")
    if frame_lower.shape[1] != dimensions:
        raise ValueError("frame corners must have coordinate dimension d matching points")
    if window_lower.shape != window_upper.shape:
        raise ValueError("window_lower and window_upper must have identical [H, W, d] shapes")
    if window_lower.shape[0] != frame_lower.shape[0]:
        raise ValueError("window tensors and frames must have the same house count H")
    if window_lower.shape[2] != dimensions:
        raise ValueError("window corners must have coordinate dimension d matching points")

    tensors = (frame_lower, frame_upper, window_lower, window_upper)
    if any(tensor.device != points.device for tensor in tensors):
        raise ValueError("points and all corners must be on the same device")
    if points.device.type not in ("cpu", "cuda"):
        raise ValueError("inputs must use a supported CPU or CUDA device")
    if points.device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    if not _all_true(torch.isfinite(points)):
        raise ValueError("points must contain only finite values")
    corners = (frame_lower, frame_upper, window_lower, window_upper)
    if any(not _all_true(torch.isfinite(corner)) for corner in corners):
        raise ValueError("frame and window corners must contain only finite values")
    if any(not _all_true(lower >= -1.0) for lower in (frame_lower, window_lower)) or any(
        not _all_true(upper <= 1.0) for upper in (frame_upper, window_upper)
    ):
        raise ValueError("frame and window corners must lie inside the fixed domain [-1, 1]^d")
    if not _all_true(frame_lower <= frame_upper):
        raise ValueError("frame_lower must be less than or equal to frame_upper in every coordinate")
    if not _all_true(window_lower <= window_upper):
        raise ValueError("window_lower must be less than or equal to window_upper in every coordinate")

    return points, frame_lower, frame_upper, window_lower, window_upper


def town_contains(
    points: Tensor,
    frame_lower: Tensor,
    frame_upper: Tensor,
    window_lower: Tensor,
    window_upper: Tensor,
) -> Tensor:
    """Return exact Boolean membership in a union of bounded houses.

    ``points`` has shape ``[N, d]``; frame corners have shape ``[H, d]``;
    and window corners have shape ``[H, W, d]``. A town has ``H >= 1`` and
    permits ``W=0`` as a real empty window axis. All inputs are finite float32
    dense-strided tensors on the same CPU or CUDA device. Physical corners
    must be ordered and lie inside ``[-1, 1]^d``. Finite outside-domain
    queries are valid and false. Each closed frame is reduced against the
    union of its windows' ambient strict interiors, then houses are combined
    by union. This is vectorized across points, houses, and windows and uses
    direct comparisons only; it does not evaluate clearances, tolerances, or
    trainable geometry.
    """
    points, frame_lower, frame_upper, window_lower, window_upper = _validate_town_inputs(
        points, frame_lower, frame_upper, window_lower, window_upper
    )

    in_domain = ((points >= -1.0) & (points <= 1.0)).all(dim=1)
    points_by_house = points.unsqueeze(1)
    in_frames = (
        (points_by_house >= frame_lower.unsqueeze(0))
        & (points_by_house <= frame_upper.unsqueeze(0))
    ).all(dim=-1)
    closed_frames = in_domain.unsqueeze(1) & in_frames

    points_by_window = points[:, None, None, :]
    inside_windows = (
        (points_by_window > window_lower.unsqueeze(0))
        & (points_by_window < window_upper.unsqueeze(0))
    ).all(dim=-1)
    excluded = inside_windows.any(dim=-1)
    house_membership = closed_frames & ~excluded
    return house_membership.any(dim=1)
