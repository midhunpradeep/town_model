"""Replaceable source of component regions for the shared visual pathway."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from town_model.contracts import ImageBatch, ProvidedROIs, ROIIdentity


class ROIProvider(Protocol):
    """Supply original-image ROIs and occurrence identities, without labels."""

    def provide(self, batch: ImageBatch) -> ProvidedROIs: ...


class GroundTruthROIProvider:
    source = "ground_truth"

    def provide(self, batch: ImageBatch) -> ProvidedROIs:
        provided = ProvidedROIs(
            boxes_xyxy_original=tuple(boxes.detach() for boxes in batch.gt_component_boxes_xyxy),
            identities=tuple(
                tuple(
                    ROIIdentity(
                        occurrence_id=component_id,
                        detector_category_id=category_id,
                    )
                    for component_id, category_id in zip(ids, categories, strict=True)
                )
                for ids, categories in zip(
                    batch.component_ids,
                    batch.component_detector_category_ids,
                    strict=True,
                )
            ),
            source=self.source,
        )
        provided.validate(batch)
        return provided


class ExplicitROIProvider:
    """Small controlled provider useful for substitutability/integration tests."""

    def __init__(self, boxes: tuple[Tensor, ...], identities: tuple[tuple[ROIIdentity, ...], ...], source: str = "test_provider"):
        self._boxes = tuple(box.detach().to(dtype=torch.float32) for box in boxes)
        self._identities = identities
        self.source = source

    def provide(self, batch: ImageBatch) -> ProvidedROIs:
        if len(self._boxes) != len(batch.sample_ids):
            raise ValueError("explicit ROI provider batch length differs from ImageBatch")
        result = ProvidedROIs(self._boxes, self._identities, self.source)
        result.validate(batch)
        return result
