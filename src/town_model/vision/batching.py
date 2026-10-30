"""Convert validated dataset records into the shared image-batch contract."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from town_model.contracts import ImageBatch
from town_model.dataset.image_data import PreparedDataset, targets_in_trainable_order
from town_model.dataset.schema import DatasetRecord

from .preprocessing import ImageNetLetterbox


def collate_image_records(
    dataset: PreparedDataset,
    records: Sequence[DatasetRecord],
    *,
    transform: ImageNetLetterbox | None = None,
) -> ImageBatch:
    if not records:
        raise ValueError("cannot collate an empty image batch")
    transform = transform or ImageNetLetterbox()
    images = []
    geometries = []
    boxes = []
    component_ids = []
    detector_ids = []
    pizza_targets = []
    pizza_known = []
    component_targets = []
    component_known = []
    root = dataset.root.resolve()
    for record in records:
        image_path = (root / record.image_path).resolve()
        if root not in image_path.parents:
            raise ValueError(f"dataset image path escapes dataset root: {record.image_path!r}")
        with Image.open(image_path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (record.image_width, record.image_height):
                raise ValueError(f"image {record.sample_id} no longer matches validated RGB dimensions")
            tensor, geometry = transform(image)
        images.append(tensor)
        geometries.append(geometry)
        targets, known, component_targets_for_record, component_known_for_record = targets_in_trainable_order(dataset, record)
        pizza_targets.append(targets)
        pizza_known.append(known)
        boxes.append(torch.tensor([component.bbox_xyxy_original for component in record.components], dtype=torch.float32).reshape(-1, 4))
        component_ids.append(tuple(component.component_id for component in record.components))
        detector_ids.append(tuple(component.detector_category_id for component in record.components))
        component_targets.append(torch.tensor(component_targets_for_record, dtype=torch.float32).reshape(-1, len(dataset.class_names)))
        component_known.append(torch.tensor(component_known_for_record, dtype=torch.bool).reshape(-1, len(dataset.class_names)))
    batch = ImageBatch(
        images=torch.stack(images),
        original_image_sizes=tuple(geometry.original_size_hw for geometry in geometries),
        processed_image_sizes=tuple(geometry.processed_size_hw for geometry in geometries),
        image_geometries=tuple(geometries),
        sample_ids=tuple(record.sample_id for record in records),
        gt_component_boxes_xyxy=tuple(boxes),
        component_ids=tuple(component_ids),
        component_detector_category_ids=tuple(detector_ids),
        pizza_targets=torch.tensor(pizza_targets, dtype=torch.float32),
        pizza_known_masks=torch.tensor(pizza_known, dtype=torch.bool),
        component_targets=tuple(component_targets),
        component_known_masks=tuple(component_known),
        split_identity=tuple(record.split for record in records),
    )
    batch.validate()
    return batch
