"""Torchvision MobileNetV3-Large and shared 256-channel FPN adapter."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
from torch import Tensor, nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool


class MobileNetV3LargeFPN(nn.Module):
    """ImageNet MobileNetV3-Large stages with four detector-compatible FPN maps.

    Backbone feature indices (Torchvision 0.20.1) are 3, 6, 12, and 16 with
    channels 24, 40, 112, 960 and strides 4, 8, 16, 32 respectively. The
    256-channel FPN maps are named ``0`` through ``3`` at those same strides;
    ``pool`` is a 256-channel stride-64 max-pooled map for later detector use.
    """

    feature_indices = (3, 6, 12, 16)
    feature_names = ("0", "1", "2", "3")
    input_channels = (24, 40, 112, 960)
    spatial_strides = (4, 8, 16, 32)
    output_channels = 256
    torchvision_version = "0.20.1"

    def __init__(self, *, pretrained: bool = True, trainable_backbone_blocks: Iterable[int] = ()) -> None:
        super().__init__()
        if torch.__version__.split("+")[0] != "2.5.1":
            raise RuntimeError(
                f"this adapter is validated with torch 2.5.1 and torchvision 0.20.1; found torch {torch.__version__}"
            )
        if not str(__import__("torchvision").__version__).startswith("0.20.1"):
            raise RuntimeError(f"expected torchvision 0.20.1, found {__import__('torchvision').__version__}")
        blocks = tuple(sorted(set(trainable_backbone_blocks)))
        if any(type(index) is not int or index < 0 or index >= 17 for index in blocks):
            raise ValueError("trainable MobileNet blocks must be feature indices in [0,16]")
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        classification_model = mobilenet_v3_large(weights=weights)
        self.mobile_features = classification_model.features
        del classification_model
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(self.input_channels),
            out_channels=self.output_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.pretrained = bool(pretrained)
        self.initialization_source = "imagenet_mobilenet_v3_large" if pretrained else "random"
        self.trainable_backbone_blocks = blocks
        self.set_trainable(trainable_fpn=True, trainable_backbone_blocks=blocks)

    def set_trainable(self, *, trainable_fpn: bool, trainable_backbone_blocks: Iterable[int] = ()) -> None:
        blocks = tuple(sorted(set(trainable_backbone_blocks)))
        if any(type(index) is not int or index < 0 or index >= len(self.mobile_features) for index in blocks):
            raise ValueError("trainable MobileNet blocks must be valid feature indices [0,16]")
        self.trainable_backbone_blocks = blocks
        for parameter in self.mobile_features.parameters():
            parameter.requires_grad_(False)
        for index in blocks:
            for parameter in self.mobile_features[index].parameters():
                parameter.requires_grad_(True)
        for parameter in self.fpn.parameters():
            parameter.requires_grad_(trainable_fpn)
        self._trainable_fpn = bool(trainable_fpn)
        self.train(self.training)

    def train(self, mode: bool = True) -> "MobileNetV3LargeFPN":
        super().train(mode)
        for index, module in enumerate(self.mobile_features):
            if index not in self.trainable_backbone_blocks:
                module.eval()
        if not self._trainable_fpn:
            self.fpn.eval()
        return self

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("MobileNetV3LargeFPN input must have shape [B,3,H,W]")
        if images.dtype != torch.float32 or not torch.isfinite(images).all():
            raise ValueError("backbone input must be finite float32 ImageNet-normalized pixels")
        outputs: OrderedDict[str, Tensor] = OrderedDict()
        value = images
        index_to_name = dict(zip(self.feature_indices, self.feature_names, strict=True))
        for index, layer in enumerate(self.mobile_features):
            value = layer(value)
            if index in index_to_name:
                outputs[index_to_name[index]] = value
        fpn_features = self.fpn(outputs)
        if tuple(fpn_features.keys()) != ("0", "1", "2", "3", "pool"):
            raise RuntimeError(f"Torchvision FPN returned unexpected feature names: {tuple(fpn_features)}")
        if any(feature.shape[1] != self.output_channels for feature in fpn_features.values()):
            raise RuntimeError("all FPN feature maps, including extra levels, must have 256 channels")
        return fpn_features
