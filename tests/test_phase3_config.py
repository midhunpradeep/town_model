from __future__ import annotations

import pytest

from town_model.experiments.image_config import (
    DatasetSubsetConfig,
    ImageExperimentConfig,
    ImageModelConfig,
    VisionConfig,
    VisualizationPreferences,
)


def test_resolved_config_is_immutable_validated_and_excludes_visualization_identity():
    first = ImageExperimentConfig().validate()
    second = ImageExperimentConfig(visualization=VisualizationPreferences(point_projection="pca", image_overlay_alpha=0.2)).validate()
    assert first.identity_hash() == second.identity_hash()
    assert first.model.family == "baseline" and first.model.pizza_mode == "hybrid"
    assert first.vision.pretrained_imagenet is True


def test_invalid_model_vision_and_split_combinations_fail_before_execution():
    invalid = [
        ImageExperimentConfig(vision=VisionConfig(roi_source="predicted")),
        ImageExperimentConfig(vision=VisionConfig(roi_size=5)),
        ImageExperimentConfig(dataset=DatasetSubsetConfig(train_split="test")),
        ImageExperimentConfig(model=ImageModelConfig(family="town", min_width=0.7)),
        ImageExperimentConfig(model=ImageModelConfig(family="box", window_mode="independent")),
        ImageExperimentConfig(model=ImageModelConfig(latent_dimensions=0)),
    ]
    for config in invalid:
        with pytest.raises(ValueError):
            config.validate()


def test_explicit_trainable_ids_must_be_nonempty_unique_and_valid():
    with pytest.raises(ValueError, match="nonempty and unique"):
        DatasetSubsetConfig(trainable_class_ids=()).validate()
    with pytest.raises(ValueError, match="nonempty and unique"):
        DatasetSubsetConfig(trainable_class_ids=(2, 2)).validate()
    with pytest.raises(ValueError, match="nonnegative"):
        DatasetSubsetConfig(trainable_class_ids=(-1,)).validate()
