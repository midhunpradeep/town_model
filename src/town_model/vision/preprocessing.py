"""Authoritative RGB, ImageNet normalization, resize/pad, and box coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from town_model.contracts import ImageGeometry


@dataclass(frozen=True, slots=True)
class ImageNetLetterbox:
    """Convert RGB pixels once to a fixed square ImageNet-normalized tensor."""

    resolution: int = 224

    def __post_init__(self) -> None:
        if type(self.resolution) is not int or self.resolution < 32:
            raise ValueError("image resolution must be an integer >= 32")

    def __call__(self, image: Image.Image) -> tuple[Tensor, ImageGeometry]:
        if not isinstance(image, Image.Image):
            raise TypeError("ImageNetLetterbox accepts a PIL image; pass raw RGB pixels, not a normalized tensor")
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width < 1 or height < 1:
            raise ValueError("input image dimensions must be positive")
        scale = min(self.resolution / width, self.resolution / height)
        resized_width = min(self.resolution, max(1, int(round(width * scale))))
        resized_height = min(self.resolution, max(1, int(round(height * scale))))
        resized = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        pad_left = (self.resolution - resized_width) // 2
        pad_top = (self.resolution - resized_height) // 2
        canvas = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))
        canvas.paste(resized, (pad_left, pad_top))
        array = np.asarray(canvas, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        normalized = (tensor - mean) / std
        geometry = ImageGeometry(
            original_size_hw=(height, width),
            processed_size_hw=(self.resolution, self.resolution),
            resized_size_hw=(resized_height, resized_width),
            scale_xy=(resized_width / width, resized_height / height),
            pad_left_top=(pad_left, pad_top),
        )
        return normalized, geometry


def _validate_boxes(boxes: Tensor, geometry: ImageGeometry) -> None:
    if not isinstance(boxes, Tensor) or boxes.dtype != torch.float32 or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must be a float32 tensor with shape [N,4]")
    if not torch.isfinite(boxes).all():
        raise ValueError("boxes must contain only finite values")
    height, width = geometry.original_size_hw
    if boxes.shape[0] and (
        torch.any(boxes[:, 0] < 0) or torch.any(boxes[:, 1] < 0)
        or torch.any(boxes[:, 2] > width) or torch.any(boxes[:, 3] > height)
        or torch.any(boxes[:, 2] <= boxes[:, 0]) or torch.any(boxes[:, 3] <= boxes[:, 1])
    ):
        raise ValueError("original-image boxes must have positive area and lie within pixel-edge bounds")


def transform_boxes_xyxy(boxes: Tensor, geometry: ImageGeometry) -> Tensor:
    """Map original pixel-edge xyxy boxes through recorded resize and padding."""
    _validate_boxes(boxes, geometry)
    sx, sy = geometry.scale_xy
    left, top = geometry.pad_left_top
    transformed = boxes.clone()
    transformed[:, (0, 2)] = transformed[:, (0, 2)] * sx + left
    transformed[:, (1, 3)] = transformed[:, (1, 3)] * sy + top
    return transformed


def inverse_boxes_xyxy(boxes: Tensor, geometry: ImageGeometry) -> Tensor:
    """Map processed pixel-edge xyxy boxes back into original coordinates."""
    if not isinstance(boxes, Tensor) or boxes.dtype != torch.float32 or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must be a float32 tensor with shape [N,4]")
    if not torch.isfinite(boxes).all():
        raise ValueError("boxes must contain only finite values")
    sx, sy = geometry.scale_xy
    left, top = geometry.pad_left_top
    result = boxes.clone()
    result[:, (0, 2)] = (result[:, (0, 2)] - left) / sx
    result[:, (1, 3)] = (result[:, (1, 3)] - top) / sy
    height, width = geometry.original_size_hw
    if result.shape[0] and (
        torch.any(result[:, 0] < -1e-4) or torch.any(result[:, 1] < -1e-4)
        or torch.any(result[:, 2] > width + 1e-4) or torch.any(result[:, 3] > height + 1e-4)
        or torch.any(result[:, 2] <= result[:, 0]) or torch.any(result[:, 3] <= result[:, 1])
    ):
        raise ValueError("processed boxes map outside the original image or have nonpositive area")
    return result


def normalized_roi_coordinates(boxes_xyxy: Tensor, image_size_hw: tuple[int, int]) -> Tensor:
    """Return (cx/W, cy/H, w/W, h/H) from original-image xyxy boxes."""
    if boxes_xyxy.dtype != torch.float32 or boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError("boxes_xyxy must be float32 [N,4]")
    height, width = image_size_hw
    if height <= 0 or width <= 0:
        raise ValueError("original image dimensions must be positive")
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    return torch.stack(
        ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height),
        dim=-1,
    )
