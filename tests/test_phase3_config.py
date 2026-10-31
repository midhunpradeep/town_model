from __future__ import annotations

import json
from pathlib import Path

import pytest

from town_model.experiments.image_config import (
    DatasetSubsetConfig,
    ImageExperimentConfig,
    ImageModelConfig,
    ObjectiveConfig,
    VisionConfig,
    DiagnosticsConfig,
    ExecutionConfig,
    TrainingConfig,
    VisualizationPreferences,
    image_config_from_mapping,
)


def test_resolved_config_is_immutable_validated_and_excludes_visualization_identity():
    first = ImageExperimentConfig().validate()
    second = ImageExperimentConfig(visualization=VisualizationPreferences(point_projection="pca", image_overlay_alpha=0.2)).validate()
    assert first.identity_hash() == second.identity_hash()
    assert first.model.family == "baseline" and first.model.pizza_mode == "hybrid"
    assert first.vision.pretrained_imagenet is True
    operational = ImageExperimentConfig(
        execution=ExecutionConfig(feature_cache_directory="/tmp/cache"),
        diagnostics=DiagnosticsConfig(snapshot_every_epochs=7, include_roi_feature_means=False),
    ).validate()
    assert first.identity_hash() == operational.identity_hash()


def test_invalid_model_vision_and_split_combinations_fail_before_execution():
    invalid = [
        ImageExperimentConfig(vision=VisionConfig(roi_source="predicted")),
        ImageExperimentConfig(vision=VisionConfig(roi_size=5)),
        ImageExperimentConfig(dataset=DatasetSubsetConfig(train_split="test")),
        ImageExperimentConfig(model=ImageModelConfig(family="box", window_mode="independent")),
        ImageExperimentConfig(model=ImageModelConfig(latent_dimensions=0)),
    ]
    for config in invalid:
        with pytest.raises(ValueError):
            config.validate()


def test_comparison_requires_best_checkpoints_and_consumed_resource_settings():
    with pytest.raises(ValueError, match="best-validation"):
        ImageExperimentConfig(purpose="comparison").validate()
    with pytest.raises(ValueError, match="num_workers must be 0"):
        ImageExperimentConfig(execution=ExecutionConfig(num_workers=1)).validate()
    with pytest.raises(ValueError, match="every optimizer step"):
        ImageExperimentConfig(training=TrainingConfig(checkpoint_latest_every_steps=5)).validate()


def test_config_mapping_restores_sequence_fields():
    config = image_config_from_mapping({
        "dataset": {"trainable_class_ids": [0, 2]},
        "vision": {"visual_policy": "trainable_fpn_last_blocks", "trainable_backbone_blocks": [12, 16]},
    })
    assert config.dataset.trainable_class_ids == (0, 2)
    assert config.vision.trainable_backbone_blocks == (12, 16)


def test_historical_objective_metadata_and_run_identity_compatibility_are_stable():
    expected = {
        "baseline": "binary_cross_entropy_with_logits",
        "box": "signed_distance_hinge",
        "town": "signed_distance_hinge",
    }
    configs = {}
    for family, loss_name in expected.items():
        config = ImageExperimentConfig(model=ImageModelConfig(family=family)).validate()
        configs[family] = config
        assert config.objective.geometry_loss == "signed_distance"
        assert config.objective_identity()["component_loss"]["name"] == loss_name
        assert config.objective_identity()["pizza_loss"]["name"] == loss_name
        assert config.objective_identity()["reduction"] == "known-weight-sum-normalized-per-instance-v2"
        assert config.identity_dict()["objective_semantics"] == "known-weight-sum-normalized-per-instance-v2"
        assert config.identity_dict()["vision_feature_contract"] == "original_extent_roi_letterbox_padding_v2"
        assert config.objective_identity()["positive_weighting_semantics"] == "positive-count-proportional-v2-all-in-scope-most-supported-class-unit-scaled"

    hinge = ImageExperimentConfig(model=ImageModelConfig(family="box"), objective=ObjectiveConfig(hinge_margin=0.0)).validate()
    assert hinge.objective_identity()["component_loss"]["parameters"] == {"margin": 0.0}
    signed_bce = ImageExperimentConfig(
        model=ImageModelConfig(family="box"), objective=ObjectiveConfig(geometry_loss="signed_bce")
    ).validate()
    assert signed_bce.objective_identity()["component_loss"] == {
        "name": "signed_margin_bce",
        "formula": "softplus(-(2 * target - 1) * signed_margin / temperature)",
        "parameters": {"temperature": 0.1},
    }
    assert signed_bce.identity_hash() != configs["box"].identity_hash()
    smoke_mapping = json.loads((Path(__file__).parents[1] / "configs" / "phase3_image_smoke.json").read_text())
    assert image_config_from_mapping(smoke_mapping).identity_hash() == "394452225dc600041b25818f9edc3fcb0cccb7b8d79771b898ce88a4191109ff"
    changed_margin = ImageExperimentConfig(
        model=ImageModelConfig(family="box"), objective=ObjectiveConfig(hinge_margin=0.2)
    ).validate()
    assert changed_margin.identity_hash() != configs["box"].identity_hash()
    with pytest.raises(ValueError, match="legacy inverse_frequency is unsupported"):
        ImageExperimentConfig(objective=ObjectiveConfig(positive_weighting="inverse_frequency")).validate()


def test_explicit_trainable_ids_must_be_nonempty_unique_and_valid():
    with pytest.raises(ValueError, match="nonempty"):
        DatasetSubsetConfig(trainable_class_ids=()).validate()
    with pytest.raises(ValueError, match="unique"):
        DatasetSubsetConfig(trainable_class_ids=(2, 2)).validate()
    with pytest.raises(ValueError, match="nonnegative"):
        DatasetSubsetConfig(trainable_class_ids=(-1,)).validate()
