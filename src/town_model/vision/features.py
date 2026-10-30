"""FPN ROIAlign and independent whole-image spatial feature extraction."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torchvision.ops import MultiScaleRoIAlign

from town_model.contracts import FeatureDiagnostics, ImageBatch, ProvidedROIs, VisionFeatureBatch

from .backbone import MobileNetV3LargeFPN
from .preprocessing import normalized_roi_coordinates, transform_boxes_xyxy
from .roi import GroundTruthROIProvider, ROIProvider


class SharedVisualFeatureExtractor(nn.Module):
    """One shared FPN pass for replaceable component ROIs and global image ROIs."""

    def __init__(self, backbone: MobileNetV3LargeFPN | None = None) -> None:
        super().__init__()
        self.backbone = backbone or MobileNetV3LargeFPN(pretrained=True)
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        )
        self.global_roi_align = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        )

    @staticmethod
    def _feature_shapes(batch: ImageBatch) -> list[tuple[int, int]]:
        return list(batch.processed_image_sizes)

    def forward(
        self,
        batch: ImageBatch,
        roi_provider: ROIProvider | None = None,
        *,
        diagnostics: bool = False,
    ) -> VisionFeatureBatch:
        batch.validate()
        provider = roi_provider or GroundTruthROIProvider()
        rois = provider.provide(batch)
        rois.validate(batch)
        maps = self.backbone(batch.images)
        image_shapes = self._feature_shapes(batch)
        processed_boxes: list[Tensor] = []
        for boxes, geometry in zip(rois.boxes_xyxy_original, batch.image_geometries, strict=True):
            processed_boxes.append(transform_boxes_xyxy(boxes.to(batch.images.device), geometry))

        roi_counts = [boxes.shape[0] for boxes in processed_boxes]
        total_rois = sum(roi_counts)
        if total_rois:
            pooled = self.roi_align(maps, processed_boxes, image_shapes)
            if pooled.shape != (total_rois, 256, 7, 7):
                raise RuntimeError(f"ROIAlign returned unexpected component shape {tuple(pooled.shape)}")
        else:
            pooled = maps["0"].new_empty((0, 256, 7, 7))

        # Map the full original-image ROI through the same authoritative
        # resize/pad transform as component boxes. A letterboxed canvas includes
        # padding, so its processed rectangle is not the original image extent.
        full_image_boxes = []
        for (height, width), geometry in zip(batch.original_image_sizes, batch.image_geometries, strict=True):
            original_extent = batch.images.new_tensor([[0.0, 0.0, float(width), float(height)]])
            full_image_boxes.append(transform_boxes_xyxy(original_extent, geometry))
        global_spatial = self.global_roi_align(maps, full_image_boxes, image_shapes)
        if global_spatial.shape != (len(batch.sample_ids), 256, 7, 7):
            raise RuntimeError(f"global ROIAlign returned unexpected shape {tuple(global_spatial.shape)}")

        batch_size = len(batch.sample_ids)
        max_rois = max(roi_counts, default=0)
        component_vectors = batch.images.new_zeros((batch_size, max_rois, 260))
        valid_mask = torch.zeros((batch_size, max_rois), dtype=torch.bool, device=batch.images.device)
        pooled_features = batch.images.new_zeros((batch_size, max_rois, 256)) if diagnostics else None
        normalized_coordinates = batch.images.new_zeros((batch_size, max_rois, 4)) if diagnostics else None
        ids_out: list[tuple[str | None, ...]] = []
        start = 0
        for image_index, count in enumerate(roi_counts):
            identities = rois.identities[image_index]
            ids_out.append(tuple(identity.occurrence_id for identity in identities) + (None,) * (max_rois - count))
            if count:
                roi_features = pooled[start : start + count]
                visual_vectors = roi_features.mean(dim=(-2, -1))
                source_boxes = rois.boxes_xyxy_original[image_index].to(batch.images.device)
                coordinates = normalized_roi_coordinates(source_boxes, batch.original_image_sizes[image_index])
                component_vectors[image_index, :count] = torch.cat((visual_vectors, coordinates), dim=-1)
                valid_mask[image_index, :count] = True
                if diagnostics:
                    assert pooled_features is not None and normalized_coordinates is not None
                    pooled_features[image_index, :count] = visual_vectors
                    normalized_coordinates[image_index, :count] = coordinates
            start += count
        if not torch.isfinite(component_vectors).all() or not torch.isfinite(global_spatial).all():
            raise FloatingPointError("visual feature extraction produced nonfinite component/global features")
        diagnostic = None
        if diagnostics:
            diagnostic = FeatureDiagnostics(
                roi_features=pooled_features,
                global_features=global_spatial,
                roi_source=rois.source,
                normalized_roi_coordinates=normalized_coordinates,
            )
        return VisionFeatureBatch(
            component_vectors=component_vectors,
            component_valid_mask=valid_mask,
            global_spatial_features=global_spatial,
            component_ids=tuple(ids_out),
            diagnostics=diagnostic,
        )
