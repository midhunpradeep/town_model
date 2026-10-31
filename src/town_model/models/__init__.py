"""Lazy public model exports.

The package namespace preserves historical imports without importing concrete
classifier implementations when callers only need vision-to-point components or
transport contracts.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BoxClassifier": ".classifiers",
    "ComponentPointProjection": ".image_points",
    "DeepSetsComposition": ".image_points",
    "BoxGeometry": ".geometry",
    "ClassifierOutput": ".classifier_transport",
    "GeometryEvaluation": ".geometry",
    "GlobalPointProjection": ".image_points",
    "HybridPointFusion": ".image_points",
    "ImageOntologyModel": ".image_embeddings",
    "ImagePointModel": ".image_points",
    "NeuralBaseline": ".classifiers",
    "PointClassifier": ".classifiers",
    "PizzaMode": ".image_points",
    "SupervisionReduction": ".losses",
    "SyntheticPointEncoder": ".classifiers",
    "TownClassifier": ".classifiers",
    "TownGeometry": ".geometry",
    "evaluate_town_corners": ".geometry",
    "make_classifier": ".classifiers",
    "reduce_supervision": ".losses",
    "unreduced_penalties": ".classifiers",
    "validate_min_width": ".geometry",
    "validate_signed_distance_margin": ".losses",
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
