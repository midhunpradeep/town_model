"""Exact bounded box/town geometry and trainable float32 parameterizations.

The physical parameterizations use ordered bounded corners. Raw logits are
bounded to [-12, 12] before sigmoid so extreme optimizer values remain finite
and leave a representable amount of slack. A small float32 safety reserve is
added to the requested minimum width/clearance; it is not a membership epsilon.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable

import torch
from torch import Tensor, nn

# Re-exported for callers using the historical geometry module path.
from .geometry_diagnostics import GEOMETRY_DIAGNOSTIC_AXES

_RAW_LIMIT = 12.0
_FLOAT32_RESERVE = 16.0 * torch.finfo(torch.float32).eps



@dataclass(frozen=True, slots=True)
class GeometryEvaluation:
    """Hard set membership, signed coordinate margin, and opt-in diagnostics."""

    contains: Tensor
    signed_margin: Tensor
    diagnostics: dict[str, Tensor] | None = None


def _as_float32_tensor(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must use float32")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def validate_min_width(
    min_width: object,
    *,
    model_families: Iterable[str],
    window_mode: str = "contained",
) -> float:
    """Validate width feasibility for the geometric families in a composed run.

    Return the requested width plus the float32 safety reserve used by the
    geometry transforms. Keeping the family-specific upper bounds here lets
    configuration preflight and direct constructors enforce the same rules.
    """
    if isinstance(min_width, bool) or not isinstance(min_width, Real):
        raise ValueError("min_width must be a finite numeric value and positive")
    try:
        requested = float(min_width)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("min_width must be a finite numeric value and positive") from error
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError("min_width must be a finite numeric value and positive")

    limits: list[tuple[float, str]] = []
    families = tuple(model_families)
    if not families:
        raise ValueError("model_families must include at least one classifier family")
    for family in families:
        if family == "baseline":
            continue
        if family == "box":
            limits.append(
                (2.0 - 2.0 * _FLOAT32_RESERVE, "float32 width reserve inside [-1,1] for pure boxes")
            )
        elif family == "town":
            if window_mode != "contained":
                raise ValueError("Phase 2 supports only the contained town window parameterization")
            limits.append(
                ((2.0 - 5.0 * _FLOAT32_RESERVE) / 3.0, "contained town frames and clearance feasibility")
            )
        else:
            raise ValueError(f"unknown model family for min_width validation: {family!r}")

    if limits:
        maximum, constraint = min(limits, key=lambda limit: limit[0])
        if requested >= maximum:
            raise ValueError(
                f"min_width must be finite and positive and less than {maximum:.9g} "
                f"to satisfy {constraint}"
            )
    return requested + _FLOAT32_RESERVE


def _bounded_corners(raw_low: Tensor, raw_width: Tensor, floor: float) -> tuple[Tensor, Tensor]:
    """Map raw values to ordered corners in [-1,1], with width >= floor."""
    low_fraction = torch.sigmoid(raw_low.float().clamp(-_RAW_LIMIT, _RAW_LIMIT))
    width_fraction = torch.sigmoid(raw_width.float().clamp(-_RAW_LIMIT, _RAW_LIMIT))
    low = -1.0 + (2.0 - floor) * low_fraction
    remaining = (1.0 - low - floor).clamp_min(0.0)
    high = 1.0 - remaining * (1.0 - width_fraction)
    return low, high


def _logit_fraction(fraction: Tensor) -> Tensor:
    # The same finite bound used by the forward transform gives stable inverses.
    fraction = fraction.float().clamp(torch.sigmoid(torch.tensor(-_RAW_LIMIT)).item(), torch.sigmoid(torch.tensor(_RAW_LIMIT)).item())
    return torch.logit(fraction).clamp(-_RAW_LIMIT, _RAW_LIMIT)


def _set_bounded_corners(raw_low: Tensor, raw_width: Tensor, low: Tensor, high: Tensor, floor: float) -> None:
    """Initialize raw corner logits from valid physical corners."""
    low_fraction = (low + 1.0) / (2.0 - floor)
    remaining = 1.0 - low - floor
    width_fraction = (high - low - floor) / remaining.clamp_min(torch.finfo(torch.float32).tiny)
    with torch.no_grad():
        raw_low.copy_(_logit_fraction(low_fraction))
        raw_width.copy_(_logit_fraction(width_fraction))


def _check_corner_shapes(points: Tensor, lows: Tensor, highs: Tensor, *, dims: int) -> None:
    _as_float32_tensor(points, "points")
    _as_float32_tensor(lows, "lows")
    _as_float32_tensor(highs, "highs")
    if points.ndim != 2 or points.shape[1] != dims:
        raise ValueError(f"points must have shape [N,{dims}]")
    if lows.ndim < 2 or lows.shape != highs.shape or lows.shape[-1] != dims:
        raise ValueError("corner tensors must have matching [...,d] shapes")
    if torch.any(lows < -1) or torch.any(highs > 1) or torch.any(lows > highs):
        raise ValueError("physical corners must be ordered and lie in [-1,1]")


def evaluate_town_corners(
    points: Tensor,
    frame_low: Tensor,
    frame_high: Tensor,
    window_low: Tensor,
    window_high: Tensor,
    *,
    diagnostics: bool = False,
) -> GeometryEvaluation:
    r"""Evaluate exact set operations for supplied corners.

    Shapes are points [N,d], frames [C,H,d], windows [C,H,W,d]. Windows may
    extend outside their frame; subtraction is the literal frame minus the union
    of windows. Trainable Phase 2 towns use strictly-contained windows.
    Per-house/per-window diagnostic arrays are built only when requested.
    """
    if frame_low.ndim != 3 or frame_high.shape != frame_low.shape:
        raise ValueError("frame corners must have matching [C,H,d] shapes")
    if window_low.ndim != 4 or window_high.shape != window_low.shape:
        raise ValueError("window corners must have matching [C,H,W,d] shapes")
    classes, houses, dims = frame_low.shape
    if window_low.shape[:2] != (classes, houses) or window_low.shape[-1] != dims:
        raise ValueError("window corner shape is incompatible with frame geometry")
    _check_corner_shapes(points, frame_low, frame_high, dims=dims)
    _check_corner_shapes(points, window_low, window_high, dims=dims)
    if points.device != frame_low.device or points.device != window_low.device:
        raise ValueError("points and corners must be on the same device")

    n, window_count = points.shape[0], window_low.shape[2]
    town_margin: Tensor | None = None
    town_contains = torch.zeros((n, classes), dtype=torch.bool, device=points.device)
    frame_margins: list[Tensor] = []
    frame_memberships: list[Tensor] = []
    window_margins: list[Tensor] = []
    window_memberships: list[Tensor] = []
    house_margins: list[Tensor] = []
    house_memberships: list[Tensor] = []
    p = points[:, None, :]

    for house_index in range(houses):
        low = frame_low[:, house_index, :].unsqueeze(0)
        high = frame_high[:, house_index, :].unsqueeze(0)
        frame_margin = torch.minimum(p - low, high - p).amin(dim=-1)
        frame_contains = ((p >= low) & (p <= high)).all(dim=-1)
        house_margin = frame_margin
        excluded = torch.zeros_like(frame_contains)
        house_window_margins: list[Tensor] = []
        house_window_memberships: list[Tensor] = []
        for window_index in range(window_count):
            wlow = window_low[:, house_index, window_index, :].unsqueeze(0)
            whigh = window_high[:, house_index, window_index, :].unsqueeze(0)
            window_margin = torch.minimum(p - wlow, whigh - p).amin(dim=-1)
            window_contains = ((p >= wlow) & (p <= whigh)).all(dim=-1)
            house_margin = torch.minimum(house_margin, -window_margin)
            excluded = excluded | window_contains
            if diagnostics:
                house_window_margins.append(window_margin)
                house_window_memberships.append(window_contains)
        house_contains = frame_contains & ~excluded
        town_margin = house_margin if town_margin is None else torch.maximum(town_margin, house_margin)
        town_contains = town_contains | house_contains
        if diagnostics:
            frame_margins.append(frame_margin)
            frame_memberships.append(frame_contains)
            house_margins.append(house_margin)
            house_memberships.append(house_contains)
            if window_count:
                window_margins.append(torch.stack(house_window_margins, dim=-1))
                window_memberships.append(torch.stack(house_window_memberships, dim=-1))

    if town_margin is None:
        town_margin = points.new_full((n, classes), -float("inf"))
    diag: dict[str, Tensor] | None = None
    if diagnostics:
        empty_margin = points.new_empty((n, classes, 0))
        empty_bool = torch.empty((n, classes, 0), dtype=torch.bool, device=points.device)
        frame_margin_tensor = torch.stack(frame_margins, dim=-1) if houses else empty_margin
        frame_contains_tensor = torch.stack(frame_memberships, dim=-1) if houses else empty_bool
        house_margin_tensor = torch.stack(house_margins, dim=-1) if houses else empty_margin
        house_contains_tensor = torch.stack(house_memberships, dim=-1) if houses else empty_bool
        window_margin_tensor = torch.stack(window_margins, dim=2) if houses and window_count else points.new_empty((n, classes, houses, window_count))
        window_contains_tensor = torch.stack(window_memberships, dim=2) if houses and window_count else torch.empty((n, classes, houses, window_count), dtype=torch.bool, device=points.device)
        diag = {
            "frame_margin": frame_margin_tensor,
            "frame_contains": frame_contains_tensor,
            "window_margin": window_margin_tensor,
            "window_contains": window_contains_tensor,
            "house_margin": house_margin_tensor,
            "house_contains": house_contains_tensor,
        }
    return GeometryEvaluation(town_contains, town_margin.float(), diag)


class BoxGeometry(nn.Module):
    """One bounded closed axis-aligned trainable box per class."""

    def __init__(self, dimensions: int, classes: int, *, seed: int = 42, min_width: float = 1e-5):
        super().__init__()
        if type(dimensions) is not int or type(classes) is not int or dimensions < 1 or classes < 1:
            raise ValueError("dimensions and classes must be positive integers")
        self.dimensions = int(dimensions)
        self.classes = int(classes)
        self.seed = int(seed)
        self.width_floor = validate_min_width(min_width, model_families=("box",))
        self.min_width = float(min_width)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        center = torch.randn((classes, dimensions), generator=generator) * 0.025
        widths = 1.82 + torch.rand((classes, dimensions), generator=generator) * 0.12
        widths = torch.minimum(widths, 2.0 - 2.0 * center.abs())
        low = (center - widths / 2).clamp(-1.0, 1.0 - self.width_floor)
        high = torch.minimum(torch.maximum(center + widths / 2, low + self.width_floor), torch.ones_like(low))
        self.raw_low = nn.Parameter(torch.empty((classes, dimensions), dtype=torch.float32))
        self.raw_width = nn.Parameter(torch.empty((classes, dimensions), dtype=torch.float32))
        _set_bounded_corners(self.raw_low, self.raw_width, low, high, self.width_floor)
        self.initialization_info = {
            "seed": self.seed,
            "dimensions": self.dimensions,
            "width_floor": self.width_floor,
            "centers": ((low + high) / 2).detach().tolist(),
            "widths": (high - low).detach().tolist(),
        }

    @classmethod
    def from_physical(cls, lows: Tensor, highs: Tensor, *, min_width: float = 1e-5) -> "BoxGeometry":
        """Build a fixed-corner geometry for boundary tests and exact diagnostics."""
        _as_float32_tensor(lows, "lows")
        _as_float32_tensor(highs, "highs")
        if lows.ndim != 3 or highs.shape != lows.shape:
            raise ValueError("physical box corners must have shape [C,1,d] or [C,H,d]")
        if torch.any(lows < -1) or torch.any(highs > 1) or torch.any(highs < lows):
            raise ValueError("physical corners must be ordered and lie in [-1,1]")
        result = cls(lows.shape[-1], lows.shape[0], min_width=min_width)
        if lows.shape[1] != 1:
            raise ValueError("a pure box has one physical box per class; expected [C,1,d]")
        if torch.any(highs - lows < result.min_width):
            raise ValueError("physical boxes must meet the configured minimum axis width")
        result.register_buffer("fixed_low", lows[:, 0, :].clone())
        result.register_buffer("fixed_high", highs[:, 0, :].clone())
        result.raw_low.requires_grad_(False)
        result.raw_width.requires_grad_(False)
        return result

    def physical_boxes(self) -> tuple[Tensor, Tensor]:
        if hasattr(self, "fixed_low"):
            return self.fixed_low, self.fixed_high
        return _bounded_corners(self.raw_low, self.raw_width, self.width_floor)

    def evaluate(
        self,
        points: Tensor,
        *,
        diagnostics: bool = False,
        point_chunk_size: int | None = None,
        class_chunk_size: int | None = None,
    ) -> GeometryEvaluation:
        _as_float32_tensor(points, "points")
        if points.ndim != 2 or points.shape[1] != self.dimensions:
            raise ValueError(f"points must have shape [N,{self.dimensions}]")
        for name, size in (("point_chunk_size", point_chunk_size), ("class_chunk_size", class_chunk_size)):
            if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1):
                raise ValueError(f"{name} must be a positive integer")
        lows, highs = self.physical_boxes()
        if points.shape[0] == 0:
            empty = torch.empty((0, self.classes), dtype=torch.bool, device=points.device)
            margin = points.new_empty((0, self.classes))
            diag = {"box_low": lows, "box_high": highs} if diagnostics else None
            return GeometryEvaluation(empty, margin, diag)
        psize = point_chunk_size if point_chunk_size is not None else points.shape[0]
        csize = class_chunk_size if class_chunk_size is not None else self.classes
        contains_rows: list[Tensor] = []
        margin_rows: list[Tensor] = []
        for p_start in range(0, points.shape[0], psize):
            point_part = points[p_start : p_start + psize]
            contains_classes: list[Tensor] = []
            margin_classes: list[Tensor] = []
            for c_start in range(0, self.classes, csize):
                low = lows[c_start : c_start + csize]
                high = highs[c_start : c_start + csize]
                delta = torch.minimum(point_part[:, None, :] - low[None, :, :], high[None, :, :] - point_part[:, None, :])
                margin_classes.append(delta.amin(dim=-1))
                contains_classes.append(((point_part[:, None, :] >= low[None, :, :]) & (point_part[:, None, :] <= high[None, :, :])).all(dim=-1))
            contains_rows.append(torch.cat(contains_classes, dim=1))
            margin_rows.append(torch.cat(margin_classes, dim=1))
        result = GeometryEvaluation(torch.cat(contains_rows), torch.cat(margin_rows).float())
        if diagnostics:
            result = GeometryEvaluation(result.contains, result.signed_margin, {"box_low": lows, "box_high": highs})
        return result


class TownGeometry(nn.Module):
    """Fixed-capacity houses with seeded windows sized by target volume.

    ``window_target_volume`` is a provisional fraction of available interior
    span above the minimum window floor. Its per-axis fraction is
    ``volume ** (1/dimensions)``; centers remain seeded, label-agnostic, and
    distinct across houses.
    """

    def __init__(
        self,
        dimensions: int,
        classes: int,
        *,
        houses: int = 2,
        windows: int = 1,
        seed: int = 42,
        min_width: float = 1e-5,
        window_mode: str = "contained",
        window_target_volume: float = 0.25,
    ):
        super().__init__()
        if any(type(value) is not int for value in (dimensions, classes, houses, windows)) or dimensions < 1 or classes < 1 or houses < 1 or windows < 1:
            raise ValueError("dimensions, classes, houses, and windows must be positive integers")
        if window_mode != "contained":
            raise ValueError("Phase 2 supports only the genuine 'contained' window parameterization")
        self.dimensions = int(dimensions)
        self.classes = int(classes)
        self.houses = int(houses)
        self.windows = int(windows)
        self.seed = int(seed)
        self.clearance = validate_min_width(
            min_width,
            model_families=("town",),
            window_mode=window_mode,
        )
        self.min_width = float(min_width)
        if (
            isinstance(window_target_volume, bool)
            or not isinstance(window_target_volume, (int, float))
            or not math.isfinite(window_target_volume)
            or not 0.0 < window_target_volume < 1.0
        ):
            raise ValueError("window_target_volume must be finite and in (0,1)")
        self.window_target_volume = float(window_target_volume)
        self.window_axis_fraction = self.window_target_volume ** (1.0 / self.dimensions)
        self.window_floor = self.clearance
        self.frame_floor = 3.0 * self.clearance + _FLOAT32_RESERVE
        self.window_mode = window_mode
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        centers = torch.randn((classes, houses, dimensions), generator=generator) * 0.03
        for house_index in range(houses):
            # Distinct deterministic offsets prevent coincident initial houses.
            centers[:, house_index, :] += (house_index - (houses - 1) / 2) * 0.055
        widths = 1.82 + torch.rand((classes, houses, dimensions), generator=generator) * 0.10
        widths = torch.minimum(widths, 2.0 - 2.0 * centers.abs())
        frame_low = (centers - widths / 2).clamp(-1.0, 1.0 - self.frame_floor)
        frame_high = torch.minimum(torch.maximum(centers + widths / 2, frame_low + self.frame_floor), torch.ones_like(frame_low))
        shape = (classes, houses, dimensions)
        self.raw_frame_low = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        self.raw_frame_width = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        _set_bounded_corners(self.raw_frame_low, self.raw_frame_width, frame_low, frame_high, self.frame_floor)

        # Windows start as substantial central exclusions, with distinct offsets
        # between houses/classes. No target or positive point is consulted.
        frame_width = frame_high - frame_low
        interior = frame_width - 2 * self.clearance
        window_width = self.window_floor + self.window_axis_fraction * (interior - self.window_floor)
        positions = torch.full((classes, houses, windows, dimensions), 0.5)
        for house_index in range(houses):
            positions[:, house_index, :, :] = min(0.85, max(0.15, 0.5 + (house_index - (houses - 1) / 2) * 0.06))
        positions += torch.randn(positions.shape, generator=generator) * 0.025
        positions = positions.clamp(0.15, 0.85)
        repeated_frame_low = frame_low[:, :, None, :]
        repeated_frame_high = frame_high[:, :, None, :]
        slack = interior[:, :, None, :] - window_width[:, :, None, :]
        window_low = repeated_frame_low + self.clearance + slack * positions
        window_high = repeated_frame_high - self.clearance - slack * (1.0 - positions)
        self.raw_window_width = nn.Parameter(torch.empty((classes, houses, windows, dimensions), dtype=torch.float32))
        self.raw_window_position = nn.Parameter(torch.empty((classes, houses, windows, dimensions), dtype=torch.float32))
        self._set_contained_window_raw(frame_low, frame_high, window_low, window_high)
        self.initialization_info = {
            "seed": self.seed,
            "dimensions": self.dimensions,
            "houses": self.houses,
            "windows_per_house": self.windows,
            "window_mode": self.window_mode,
            "minimum_axis_width": self.min_width,
            "numerical_clearance_floor": self.clearance,
            "window_target_volume_fraction": self.window_target_volume,
            "window_target_volume_basis": "available interior span above minimum window floor",
            "window_axis_fraction": self.window_axis_fraction,
            "frame_centers": ((frame_low + frame_high) / 2).detach().tolist(),
            "frame_widths": (frame_high - frame_low).detach().tolist(),
            "window_centers": ((window_low + window_high) / 2).detach().tolist(),
            "window_widths": (window_high - window_low).detach().tolist(),
        }

    def _set_contained_window_raw(self, frame_low: Tensor, frame_high: Tensor, window_low: Tensor, window_high: Tensor) -> None:
        frame_width = frame_high[:, :, None, :] - frame_low[:, :, None, :]
        interior = frame_width - 2 * self.clearance
        target_width = window_high - window_low
        width_fraction = (target_width - self.window_floor) / (interior - self.window_floor)
        slack = interior - target_width
        position_fraction = (window_low - frame_low[:, :, None, :] - self.clearance) / slack.clamp_min(torch.finfo(torch.float32).tiny)
        with torch.no_grad():
            self.raw_window_width.copy_(_logit_fraction(width_fraction))
            self.raw_window_position.copy_(_logit_fraction(position_fraction))

    def physical_geometry(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        frame_low, frame_high = _bounded_corners(self.raw_frame_low, self.raw_frame_width, self.frame_floor)
        frame_width = frame_high - frame_low
        interior = frame_width - 2.0 * self.clearance
        fraction_width = torch.sigmoid(self.raw_window_width.float().clamp(-_RAW_LIMIT, _RAW_LIMIT))
        window_width = self.window_floor + (interior[:, :, None, :] - self.window_floor) * fraction_width
        position = torch.sigmoid(self.raw_window_position.float().clamp(-_RAW_LIMIT, _RAW_LIMIT))
        slack = interior[:, :, None, :] - window_width
        window_low = frame_low[:, :, None, :] + self.clearance + slack * position
        window_high = frame_high[:, :, None, :] - self.clearance - slack * (1.0 - position)
        return frame_low, frame_high, window_low, window_high

    def evaluate(
        self,
        points: Tensor,
        *,
        diagnostics: bool = False,
        point_chunk_size: int | None = None,
        class_chunk_size: int | None = None,
    ) -> GeometryEvaluation:
        _as_float32_tensor(points, "points")
        if points.ndim != 2 or points.shape[1] != self.dimensions:
            raise ValueError(f"points must have shape [N,{self.dimensions}]")
        for name, size in (("point_chunk_size", point_chunk_size), ("class_chunk_size", class_chunk_size)):
            if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1):
                raise ValueError(f"{name} must be a positive integer")
        frame_low, frame_high, window_low, window_high = self.physical_geometry()
        if points.shape[0] == 0:
            empty = torch.empty((0, self.classes), dtype=torch.bool, device=points.device)
            margin = points.new_empty((0, self.classes))
            if not diagnostics:
                return GeometryEvaluation(empty, margin)
            return GeometryEvaluation(empty, margin, {
                "frame_low": frame_low,
                "frame_high": frame_high,
                "window_low": window_low,
                "window_high": window_high,
                "frame_margin": points.new_empty((0, self.classes, self.houses)),
                "frame_contains": torch.empty((0, self.classes, self.houses), dtype=torch.bool, device=points.device),
                "window_margin": points.new_empty((0, self.classes, self.houses, self.windows)),
                "window_contains": torch.empty((0, self.classes, self.houses, self.windows), dtype=torch.bool, device=points.device),
                "house_margin": points.new_empty((0, self.classes, self.houses)),
                "house_contains": torch.empty((0, self.classes, self.houses), dtype=torch.bool, device=points.device),
            })
        psize = point_chunk_size if point_chunk_size is not None else points.shape[0]
        csize = class_chunk_size if class_chunk_size is not None else self.classes
        contains_batches: list[Tensor] = []
        margin_batches: list[Tensor] = []
        diagnostic_batches: dict[str, list[Tensor]] = {
            key: [] for key in ("frame_margin", "frame_contains", "window_margin", "window_contains", "house_margin", "house_contains")
        }
        for p_start in range(0, points.shape[0], psize):
            point_chunk = points[p_start : p_start + psize]
            contains_chunks: list[Tensor] = []
            margin_chunks: list[Tensor] = []
            diagnostic_class_chunks: dict[str, list[Tensor]] = {key: [] for key in diagnostic_batches}
            for c_start in range(0, self.classes, csize):
                c_stop = min(c_start + csize, self.classes)
                result = evaluate_town_corners(
                    point_chunk,
                    frame_low[c_start:c_stop],
                    frame_high[c_start:c_stop],
                    window_low[c_start:c_stop],
                    window_high[c_start:c_stop],
                    diagnostics=diagnostics,
                )
                contains_chunks.append(result.contains)
                margin_chunks.append(result.signed_margin)
                if diagnostics:
                    assert result.diagnostics is not None
                    for key in diagnostic_class_chunks:
                        diagnostic_class_chunks[key].append(result.diagnostics[key])
            contains_batches.append(torch.cat(contains_chunks, dim=1))
            margin_batches.append(torch.cat(margin_chunks, dim=1))
            if diagnostics:
                for key, class_chunks in diagnostic_class_chunks.items():
                    diagnostic_batches[key].append(torch.cat(class_chunks, dim=1))
        result = GeometryEvaluation(torch.cat(contains_batches), torch.cat(margin_batches).float())
        if diagnostics:
            result = GeometryEvaluation(
                result.contains,
                result.signed_margin,
                {
                    "frame_low": frame_low,
                    "frame_high": frame_high,
                    "window_low": window_low,
                    "window_high": window_high,
                    **{key: torch.cat(values, dim=0) for key, values in diagnostic_batches.items()},
                },
            )
        return result
