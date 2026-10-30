"""Geometry and shared classifier interfaces for Phase 2."""

from .classifiers import (
    BoxClassifier,
    ClassifierOutput,
    NeuralBaseline,
    PointClassifier,
    SyntheticPointEncoder,
    TownClassifier,
    make_classifier,
    unreduced_penalties,
)
from .geometry import (
    BoxGeometry,
    GeometryEvaluation,
    TownGeometry,
    evaluate_town_corners,
    validate_min_width,
)
from .losses import SupervisionReduction, reduce_supervision, validate_signed_distance_margin

__all__ = [
    "BoxClassifier",
    "BoxGeometry",
    "ClassifierOutput",
    "GeometryEvaluation",
    "NeuralBaseline",
    "PointClassifier",
    "SupervisionReduction",
    "SyntheticPointEncoder",
    "TownClassifier",
    "TownGeometry",
    "evaluate_town_corners",
    "make_classifier",
    "reduce_supervision",
    "unreduced_penalties",
    "validate_min_width",
    "validate_signed_distance_margin",
]
