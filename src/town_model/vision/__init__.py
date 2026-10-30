"""Torchvision-only visual execution adapter for ontology image models."""

from .backbone import MobileNetV3LargeFPN
from .features import SharedVisualFeatureExtractor
from .preprocessing import ImageNetLetterbox, inverse_boxes_xyxy, transform_boxes_xyxy
from .roi import GroundTruthROIProvider, ROIProvider

__all__ = [
    "GroundTruthROIProvider",
    "ImageNetLetterbox",
    "MobileNetV3LargeFPN",
    "ROIProvider",
    "SharedVisualFeatureExtractor",
    "inverse_boxes_xyxy",
    "transform_boxes_xyxy",
]
