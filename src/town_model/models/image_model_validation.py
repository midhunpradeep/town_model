"""Constructor-equivalent validation for the temporary image-model adapter."""

from __future__ import annotations

import math
from numbers import Real


def validate_image_model_parameters(
    dimensions: object,
    class_names: object,
    *,
    family: object,
    pizza_mode: object,
    houses: object,
    windows: object,
    min_width: object,
    window_mode: object,
    window_target_volume: object,
) -> None:
    """Validate legacy model settings without importing concrete classifiers."""
    if type(dimensions) is not int or dimensions < 1:
        raise ValueError("latent dimensions must be a positive integer")
    if family not in ("baseline", "box", "town"):
        raise ValueError(f"unsupported classifier family {family!r}")
    if pizza_mode not in ("global", "compositional", "hybrid"):
        raise ValueError(f"unsupported pizza composition mode {pizza_mode!r}")
    if class_names is not None:
        if not isinstance(class_names, tuple) or not class_names or any(not isinstance(name, str) or not name for name in class_names):
            raise ValueError("class_names must be a nonempty ordered tuple of names")
        if len(set(class_names)) != len(class_names):
            raise ValueError("class_names must be unique")
    if type(houses) is not int or type(windows) is not int or houses < 1 or windows < 1:
        raise ValueError("town houses and windows must be positive integer counts")
    if window_mode != "contained":
        raise ValueError("Phase 3 currently supports the contained town window parameterization")
    if isinstance(window_target_volume, bool) or not isinstance(window_target_volume, Real):
        raise ValueError("window_target_volume must be finite and between zero and one")
    try:
        target_volume = float(window_target_volume)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("window_target_volume must be finite and between zero and one") from error
    if not math.isfinite(target_volume) or not 0 < target_volume < 1:
        raise ValueError("window_target_volume must be finite and between zero and one")
    if family != "town" and (houses != 2 or windows != 1 or window_mode != "contained" or window_target_volume != 0.25):
        raise ValueError("town geometry settings are incompatible with non-town classifier families")
    from .geometry import validate_min_width

    validate_min_width(min_width, model_families=(family,), window_mode=window_mode)
