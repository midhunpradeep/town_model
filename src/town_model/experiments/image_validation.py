"""Classifier-independent validation for persisted image experiment inputs."""

from __future__ import annotations

import torch

from town_model.dataset.image_data import DatasetSubset, PreparedDataset
from town_model.experiments.image_config import ImageExperimentConfig


def device_for_image_config(config: ImageExperimentConfig) -> torch.device:
    """Resolve the stored device request without constructing model modules."""
    requested = config.execution.device
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but this PyTorch runtime has no available CUDA device")
    device = torch.device(
        "cuda" if requested == "auto" and torch.cuda.is_available()
        else "cpu" if requested == "auto"
        else requested
    )
    if device.type == "cpu" and config.purpose == "smoke" and not config.execution.allow_cpu_smoke:
        raise ValueError("CPU smoke runs are disabled by execution.allow_cpu_smoke")
    return device


def validate_image_training_data(
    dataset: PreparedDataset,
    train_subset: DatasetSubset,
    validation_subset: DatasetSubset,
    class_ids: tuple[int, ...],
    *,
    require_training_coverage: bool,
    require_primary_monitor: bool,
) -> None:
    """Validate class-order and supervision support in selected data subsets."""
    if train_subset.dataset.dataset_id != dataset.dataset_id or validation_subset.dataset.dataset_id != dataset.dataset_id:
        raise ValueError("train/validation subsets must reference the loaded canonical dataset")
    if train_subset.split != "train" or validation_subset.split != "validation":
        raise ValueError("image data validation requires original train and validation partitions")
    if not class_ids or tuple(sorted(class_ids)) != class_ids or any(class_id not in dataset.canonical_class_ids for class_id in class_ids):
        raise ValueError("selected class IDs must preserve the canonical trainable class order")
    class_ids_for_level = {
        "pizza": set(dataset.pizza_supervision_class_ids),
        "component": set(dataset.component_supervision_class_ids),
    }
    support_index = {class_id: index for index, class_id in enumerate(train_subset.class_support["class_ids"])}
    validation_index = {class_id: index for index, class_id in enumerate(validation_subset.class_support["class_ids"])}
    for level, scoped_ids in class_ids_for_level.items():
        required = set(class_ids) & scoped_ids
        train_positive = train_subset.class_support[level]["known_positive"]
        validation_positive = validation_subset.class_support[level]["known_positive"]
        if require_training_coverage:
            missing = [class_id for class_id in sorted(required) if train_positive[support_index[class_id]] == 0]
            if missing:
                names = [dataset.class_names[dataset.canonical_class_ids.index(class_id)] for class_id in missing]
                raise ValueError(f"comparison input lacks known-positive {level} support in the training subset: {names}")
        if require_primary_monitor and not any(
            validation_positive[validation_index[class_id]] > 0 for class_id in required
        ):
            raise ValueError(
                f"primary validation support is undefined: no {level} class has known-positive validation support"
            )
