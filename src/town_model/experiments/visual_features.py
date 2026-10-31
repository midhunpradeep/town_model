"""Construction of the classifier-independent visual feature pathway."""

from __future__ import annotations

import random

import numpy as np
import torch

from town_model.experiments.image_config import ImageExperimentConfig
from town_model.vision.backbone import MobileNetV3LargeFPN
from town_model.vision.features import SharedVisualFeatureExtractor


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_visual_feature_extractor(config: ImageExperimentConfig) -> SharedVisualFeatureExtractor:
    """Construct the retained MobileNetV3/FPN path without a classifier."""
    config.vision.validate()
    _set_seed(config.training.seed)
    backbone = MobileNetV3LargeFPN(
        pretrained=config.vision.pretrained_imagenet,
        trainable_backbone_blocks=config.vision.trainable_backbone_blocks,
    )
    policy = config.vision.visual_policy
    backbone.set_trainable(
        trainable_fpn=policy != "frozen",
        trainable_backbone_blocks=(
            config.vision.trainable_backbone_blocks
            if policy == "trainable_fpn_last_blocks"
            else ()
        ),
    )
    return SharedVisualFeatureExtractor(backbone)
