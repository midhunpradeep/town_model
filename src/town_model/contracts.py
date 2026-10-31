"""Shared typed contracts that sit below dataset, vision, model, and training code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ImageGeometry:
    """Exact resize/pad record; image sizes use (height, width) ordering."""

    original_size_hw: tuple[int, int]
    processed_size_hw: tuple[int, int]
    resized_size_hw: tuple[int, int]
    scale_xy: tuple[float, float]
    pad_left_top: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ImageBatch:
    """One padded image batch with ragged, per-image semantic components."""

    images: Tensor
    original_image_sizes: tuple[tuple[int, int], ...]
    processed_image_sizes: tuple[tuple[int, int], ...]
    image_geometries: tuple[ImageGeometry, ...]
    sample_ids: tuple[str, ...]
    gt_component_boxes_xyxy: tuple[Tensor, ...]
    component_ids: tuple[tuple[str, ...], ...]
    component_detector_category_ids: tuple[tuple[int, ...], ...]
    pizza_targets: Tensor
    pizza_known_masks: Tensor
    component_targets: tuple[Tensor, ...]
    component_known_masks: tuple[Tensor, ...]
    split_identity: tuple[str, ...]

    def validate(self) -> None:
        if self.images.ndim != 4 or self.images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        batch = self.images.shape[0]
        if batch < 1 or not torch.isfinite(self.images).all():
            raise ValueError("images must be a nonempty finite batch")
        batch_fields = (
            self.original_image_sizes, self.processed_image_sizes, self.image_geometries,
            self.sample_ids, self.gt_component_boxes_xyxy, self.component_ids,
            self.component_detector_category_ids, self.component_targets,
            self.component_known_masks, self.split_identity,
        )
        if any(len(field) != batch for field in batch_fields):
            raise ValueError("all ragged ImageBatch fields must have one entry per image")
        if self.pizza_targets.ndim != 2 or self.pizza_targets.shape[0] != batch:
            raise ValueError("pizza_targets must have shape [B,C]")
        if self.pizza_known_masks.shape != self.pizza_targets.shape or self.pizza_known_masks.dtype != torch.bool:
            raise ValueError("pizza_known_masks must be bool [B,C] matching pizza_targets")
        if torch.any((self.pizza_targets[self.pizza_known_masks] != 0) & (self.pizza_targets[self.pizza_known_masks] != 1)):
            raise ValueError("known pizza targets must be binary")
        if not torch.isfinite(self.pizza_targets).all():
            raise ValueError("pizza targets must be finite")
        for index in range(batch):
            boxes = self.gt_component_boxes_xyxy[index]
            ids = self.component_ids[index]
            categories = self.component_detector_category_ids[index]
            targets = self.component_targets[index]
            known = self.component_known_masks[index]
            if boxes.dtype != torch.float32 or boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError(f"component boxes for image {self.sample_ids[index]} must be float32 [N,4]")
            if not torch.isfinite(boxes).all():
                raise ValueError(f"component boxes for image {self.sample_ids[index]} must be finite")
            count = boxes.shape[0]
            if len(ids) != count or len(categories) != count:
                raise ValueError(f"component identities/categories disagree with boxes for {self.sample_ids[index]}")
            if targets.ndim != 2 or targets.shape[0] != count or known.shape != targets.shape or known.dtype != torch.bool:
                raise ValueError(f"component targets/masks for {self.sample_ids[index]} must be [N,C]")
            if targets.shape[1] != self.pizza_targets.shape[1]:
                raise ValueError("pizza and component target vocabularies must have identical widths")
            if not torch.isfinite(targets).all():
                raise ValueError(f"component targets for {self.sample_ids[index]} must be finite")
            if torch.any((targets[known] != 0) & (targets[known] != 1)):
                raise ValueError(f"known component targets for {self.sample_ids[index]} must be binary")
            if count and (torch.any(boxes[:, 2] <= boxes[:, 0]) or torch.any(boxes[:, 3] <= boxes[:, 1])):
                raise ValueError(f"component boxes for {self.sample_ids[index]} must have positive area")


@dataclass(frozen=True, slots=True)
class ROIIdentity:
    """Provider identity metadata; detector labels/scores are never point inputs."""

    occurrence_id: str
    detector_category_id: int | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ProvidedROIs:
    boxes_xyxy_original: tuple[Tensor, ...]
    identities: tuple[tuple[ROIIdentity, ...], ...]
    source: str

    def validate(self, batch: ImageBatch) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("ROI provider source must be a nonempty identity")
        if len(self.boxes_xyxy_original) != len(batch.sample_ids) or len(self.identities) != len(batch.sample_ids):
            raise ValueError("ROI provider must return one box/identity list per image")
        for sample_id, boxes, identities, (height, width) in zip(
            batch.sample_ids, self.boxes_xyxy_original, self.identities, batch.original_image_sizes, strict=True
        ):
            if boxes.dtype != torch.float32 or boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError(f"ROIs for {sample_id} must be float32 [N,4] original-image boxes")
            if boxes.shape[0] != len(identities) or not torch.isfinite(boxes).all():
                raise ValueError(f"ROI boxes and identities are inconsistent for {sample_id}")
            if boxes.shape[0] and (
                torch.any(boxes[:, 0] < 0) or torch.any(boxes[:, 1] < 0)
                or torch.any(boxes[:, 2] > width) or torch.any(boxes[:, 3] > height)
                or torch.any(boxes[:, 2] <= boxes[:, 0]) or torch.any(boxes[:, 3] <= boxes[:, 1])
            ):
                raise ValueError(f"ROIs for {sample_id} must be valid original-image pixel-edge boxes")


@dataclass(frozen=True, slots=True)
class FeatureDiagnostics:
    roi_features: Tensor | None = None
    global_features: Tensor | None = None
    roi_source: str | None = None
    normalized_roi_coordinates: Tensor | None = None


@dataclass(frozen=True, slots=True)
class VisionFeatureBatch:
    component_vectors: Tensor
    component_valid_mask: Tensor
    global_spatial_features: Tensor
    component_ids: tuple[tuple[str | None, ...], ...]
    diagnostics: FeatureDiagnostics | None = None


class FrozenImageFeatureCache(Protocol):
    """Read-only, producer-identified cached features for frozen visual runs."""

    identity: dict[str, Any]
    cache_id: str

    def get(self, batch: ImageBatch, *, diagnostics: bool = False) -> VisionFeatureBatch: ...


@dataclass(frozen=True, slots=True)
class ImagePointResult:
    """Classifier-free transport from visual features to point tensors."""

    sample_ids: tuple[str, ...]
    component_ids: tuple[tuple[str | None, ...], ...]
    component_valid_mask: Tensor
    component_points: Tensor
    global_pizza_points: Tensor
    composed_pizza_points: Tensor
    final_pizza_points: Tensor
    pizza_mode: str
    vision_diagnostics: FeatureDiagnostics | None = None
