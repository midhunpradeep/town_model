"""Classifier-independent point and supervision infrastructure."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ComponentPointProjection": ".image_points",
    "DeepSetsComposition": ".image_points",
    "GlobalPointProjection": ".image_points",
    "HybridPointFusion": ".image_points",
    "ImagePointModel": ".image_points",
    "PizzaMode": ".image_points",
    "SupervisionReduction": ".losses",
    "reduce_supervision": ".losses",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
