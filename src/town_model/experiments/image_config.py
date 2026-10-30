"""Immutable, validated image-experiment configuration authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from town_model.models.geometry import validate_min_width


@dataclass(frozen=True, slots=True)
class DatasetSubsetConfig:
    dataset_path: str = "examples/pizza-48"
    train_size: int | None = 32
    validation_size: int | None = 16
    selection_seed: int = 42
    train_split: str = "train"
    validation_split: str = "validation"
    trainable_class_ids: tuple[int, ...] | None = None

    def validate(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must not be empty")
        for name, value in (("train_size", self.train_size), ("validation_size", self.validation_size)):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if type(self.selection_seed) is not int or self.selection_seed < 0:
            raise ValueError("selection_seed must be a nonnegative integer")
        if (self.train_split, self.validation_split) != ("train", "validation"):
            raise ValueError("image experiments must select from original train and validation partitions")
        if self.trainable_class_ids is not None:
            if not self.trainable_class_ids or len(set(self.trainable_class_ids)) != len(self.trainable_class_ids):
                raise ValueError("explicit trainable class IDs must be nonempty and unique")
            if any(type(class_id) is not int or class_id < 0 for class_id in self.trainable_class_ids):
                raise ValueError("explicit trainable class IDs must be nonnegative integers")


@dataclass(frozen=True, slots=True)
class VisionConfig:
    backbone: str = "mobilenet_v3_large_fpn"
    pretrained_imagenet: bool = True
    resolution: int = 224
    roi_size: int = 7
    fpn_channels: int = 256
    roi_source: str = "ground_truth"
    visual_policy: Literal["frozen", "trainable_fpn", "trainable_fpn_last_blocks"] = "frozen"
    trainable_backbone_blocks: tuple[int, ...] = ()

    def validate(self) -> None:
        if self.backbone != "mobilenet_v3_large_fpn":
            raise ValueError("Phase 3 currently supports only the shared MobileNetV3-Large/FPN backbone")
        if type(self.resolution) is not int or self.resolution < 32:
            raise ValueError("resolution must be an integer >=32")
        if self.roi_size != 7 or self.fpn_channels != 256:
            raise ValueError("the accepted model requires 7x7 ROIs and 256-channel FPN maps")
        if self.roi_source != "ground_truth":
            raise ValueError("Phase 3 permits GT ROIs only; predicted ROI integration belongs to the future detector stage")
        if self.visual_policy not in ("frozen", "trainable_fpn", "trainable_fpn_last_blocks"):
            raise ValueError(f"unsupported visual feature policy {self.visual_policy!r}")
        if self.visual_policy == "trainable_fpn_last_blocks":
            if not self.trainable_backbone_blocks or any(i < 0 or i > 16 for i in self.trainable_backbone_blocks):
                raise ValueError("trainable_fpn_last_blocks requires explicit MobileNet block indices in [0,16]")
        elif self.trainable_backbone_blocks:
            raise ValueError("backbone block selections require visual_policy='trainable_fpn_last_blocks'")


@dataclass(frozen=True, slots=True)
class ImageModelConfig:
    family: Literal["baseline", "box", "town"] = "baseline"
    pizza_mode: Literal["global", "compositional", "hybrid"] = "hybrid"
    latent_dimensions: int = 16
    houses: int = 2
    windows: int = 1
    window_mode: str = "contained"
    window_target_volume: float = 0.25
    min_width: float = 1e-5
    classifier_seed: int = 42

    def validate(self) -> None:
        if self.family not in ("baseline", "box", "town"):
            raise ValueError(f"unsupported classifier family {self.family!r}")
        if self.pizza_mode not in ("global", "compositional", "hybrid"):
            raise ValueError(f"unsupported pizza composition mode {self.pizza_mode!r}")
        if type(self.latent_dimensions) is not int or self.latent_dimensions < 1:
            raise ValueError("latent_dimensions must be a positive integer")
        if type(self.houses) is not int or self.houses < 1 or type(self.windows) is not int or self.windows < 0:
            raise ValueError("houses must be positive and windows nonnegative integers")
        if self.window_mode != "contained":
            raise ValueError("Phase 3 uses the accepted contained town parameterization")
        if not math.isfinite(self.window_target_volume) or not 0 < self.window_target_volume < 1:
            raise ValueError("window_target_volume must be finite and between zero and one")
        if type(self.classifier_seed) is not int or self.classifier_seed < 0:
            raise ValueError("classifier_seed must be a nonnegative integer")
        validate_min_width(self.min_width, model_families=(self.family,), window_mode=self.window_mode)


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    baseline_loss: str = "bce"
    geometry_loss: str = "signed_bce"
    temperature: float = 0.1
    hinge_margin: float = 0.1
    positive_weighting: Literal["none", "inverse_sqrt", "inverse_frequency", "effective_number"] = "none"
    positive_weight_cap: float = 10.0
    effective_number_beta: float = 0.99
    component_coefficient: float = 1.0
    pizza_coefficient: float = 1.0

    def validate(self, family: str) -> None:
        if self.baseline_loss != "bce":
            raise ValueError("baseline classifier supports unreduced BCEWithLogits only")
        if family == "baseline" and self.geometry_loss not in ("signed_bce", "signed_distance"):
            raise ValueError("baseline ignores geometric losses; select signed_bce or signed_distance as the configured geometric reference")
        if family != "baseline" and self.geometry_loss not in ("signed_bce", "signed_distance"):
            raise ValueError("Phase 3 geometric classifiers support signed_bce or signed_distance")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("signed BCE temperature must be finite and positive")
        if not math.isfinite(self.hinge_margin) or self.hinge_margin < 0:
            raise ValueError("hinge margin must be finite and nonnegative")
        if self.positive_weighting not in ("none", "inverse_sqrt", "inverse_frequency", "effective_number"):
            raise ValueError("unsupported positive class-weight policy")
        if not math.isfinite(self.positive_weight_cap) or self.positive_weight_cap < 1:
            raise ValueError("positive_weight_cap must be finite and at least one")
        if not math.isfinite(self.effective_number_beta) or not 0 <= self.effective_number_beta < 1:
            raise ValueError("effective_number_beta must be finite in [0,1)")
        for name, value in (("component_coefficient", self.component_coefficient), ("pizza_coefficient", self.pizza_coefficient)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.component_coefficient == 0 and self.pizza_coefficient == 0:
            raise ValueError("at least one task coefficient must be positive")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 50
    physical_batch_size: int = 4
    accumulation_steps: int = 4
    learning_rate_pretrained: float = 1e-5
    learning_rate_fpn: float = 1e-4
    learning_rate_projection: float = 1e-3
    learning_rate_classifier: float = 1e-3
    weight_decay_pretrained: float = 1e-4
    weight_decay_fpn: float = 1e-4
    weight_decay_projection: float = 1e-4
    weight_decay_classifier: float = 1e-4
    schedule: Literal["warmup_cosine", "constant"] = "warmup_cosine"
    warmup_steps: int = 2
    gradient_clip_norm: float = 1.0
    precision: str = "float32"
    checkpoint_latest_every_steps: int = 1
    early_stopping_patience: int = 10
    exploratory_latest_only: bool = True

    def validate(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("training seed must be a nonnegative integer")
        for name, value in (("epochs", self.epochs), ("physical_batch_size", self.physical_batch_size), ("accumulation_steps", self.accumulation_steps), ("checkpoint_latest_every_steps", self.checkpoint_latest_every_steps), ("early_stopping_patience", self.early_stopping_patience)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.checkpoint_latest_every_steps != 1:
            raise ValueError("latest checkpoint writes are currently restricted to every optimizer step for faithful resume")
        for name, value in (("learning_rate_pretrained", self.learning_rate_pretrained), ("learning_rate_fpn", self.learning_rate_fpn), ("learning_rate_projection", self.learning_rate_projection), ("learning_rate_classifier", self.learning_rate_classifier), ("gradient_clip_norm", self.gradient_clip_norm)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (("weight_decay_pretrained", self.weight_decay_pretrained), ("weight_decay_fpn", self.weight_decay_fpn), ("weight_decay_projection", self.weight_decay_projection), ("weight_decay_classifier", self.weight_decay_classifier)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.schedule not in ("warmup_cosine", "constant"):
            raise ValueError("unsupported learning-rate schedule")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        if self.precision != "float32":
            raise ValueError("Phase 3 currently supports float32 only")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    device: str = "cpu"
    num_workers: int = 0
    cpu_threads: int = 2
    allow_cpu_smoke: bool = True
    feature_cache_directory: str | None = None

    def validate(self) -> None:
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda, or auto")
        if type(self.num_workers) is not int or self.num_workers != 0:
            raise ValueError("num_workers must be 0; Phase 3 currently uses deterministic synchronous image loading")
        if type(self.allow_cpu_smoke) is not bool:
            raise ValueError("allow_cpu_smoke must be a boolean")
        if type(self.cpu_threads) is not int or self.cpu_threads < 1:
            raise ValueError("cpu_threads must be a positive integer")
        if self.feature_cache_directory is not None and not self.feature_cache_directory:
            raise ValueError("feature_cache_directory must be a nonempty path or None")


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    snapshot_every_epochs: int = 1
    max_reference_samples: int = 16
    max_component_traces_per_sample: int = 64
    include_roi_feature_means: bool = True

    def validate(self) -> None:
        if self.snapshot_every_epochs < 1 or self.max_reference_samples < 1 or self.max_component_traces_per_sample < 0:
            raise ValueError("snapshot frequencies/caps are invalid")

    @property
    def include_visual_features(self) -> bool:
        return self.include_roi_feature_means


@dataclass(frozen=True, slots=True)
class VisualizationPreferences:
    point_projection: Literal["none", "pca"] = "none"
    image_overlay_alpha: float = 0.6

    def validate(self) -> None:
        if self.point_projection not in ("none", "pca"):
            raise ValueError("unsupported point visualization projection")
        if not math.isfinite(self.image_overlay_alpha) or not 0 <= self.image_overlay_alpha <= 1:
            raise ValueError("image_overlay_alpha must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class ImageExperimentConfig:
    dataset: DatasetSubsetConfig = field(default_factory=DatasetSubsetConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    model: ImageModelConfig = field(default_factory=ImageModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    visualization: VisualizationPreferences = field(default_factory=VisualizationPreferences)
    purpose: Literal["smoke", "exploratory", "comparison"] = "smoke"
    schema_version: str = "image-run-config-v1"

    def validate(self) -> "ImageExperimentConfig":
        self.dataset.validate()
        self.vision.validate()
        self.model.validate()
        self.objective.validate(self.model.family)
        self.training.validate()
        self.execution.validate()
        self.diagnostics.validate()
        self.visualization.validate()
        if self.purpose not in ("smoke", "exploratory", "comparison"):
            raise ValueError("purpose must be smoke, exploratory, or comparison")
        if self.purpose == "comparison" and self.training.exploratory_latest_only:
            raise ValueError("comparison runs require best-validation checkpoints; latest-only is for labeled smoke runs")
        if self.execution.feature_cache_directory and self.vision.visual_policy != "frozen":
            raise ValueError("feature caching is valid only for frozen visual features")
        if self.model.family != "town" and (self.model.houses != 2 or self.model.windows != 1 or self.model.window_mode != "contained"):
            raise ValueError("town geometry settings are invalid for non-town classifier families")
        return self

    def identity_dict(self) -> dict[str, object]:
        """Scientific/training identity deliberately excludes presentation preferences."""
        self.validate()
        value = asdict(self)
        value.pop("visualization", None)
        value.pop("diagnostics", None)
        value["execution"].pop("feature_cache_directory", None)
        return value

    def identity_hash(self) -> str:
        encoded = json.dumps(self.identity_dict(), sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    def resolve_dataset_path(self, repository_root: Path | None = None) -> Path:
        path = Path(self.dataset.dataset_path).expanduser()
        if not path.is_absolute() and repository_root is not None:
            path = repository_root / path
        return path.resolve()


def image_config_from_mapping(value: dict[str, object]) -> ImageExperimentConfig:
    """Strict JSON/TOML-compatible constructor shared by CLI and dashboard."""
    expected = {"dataset", "vision", "model", "objective", "training", "execution", "diagnostics", "visualization", "purpose", "schema_version"}
    unknown = set(value) - expected
    if unknown:
        raise ValueError(f"unknown image experiment configuration fields: {sorted(unknown)}")
    sections = {
        "dataset": DatasetSubsetConfig,
        "vision": VisionConfig,
        "model": ImageModelConfig,
        "objective": ObjectiveConfig,
        "training": TrainingConfig,
        "execution": ExecutionConfig,
        "diagnostics": DiagnosticsConfig,
        "visualization": VisualizationPreferences,
    }
    kwargs: dict[str, object] = {}
    for section, cls in sections.items():
        raw = value.get(section, {})
        if not isinstance(raw, dict):
            raise ValueError(f"configuration section {section!r} must be an object")
        fields = dict(raw)
        if section == "dataset" and fields.get("trainable_class_ids") is not None:
            fields["trainable_class_ids"] = tuple(fields["trainable_class_ids"])
        if section == "vision" and fields.get("trainable_backbone_blocks") is not None:
            fields["trainable_backbone_blocks"] = tuple(fields["trainable_backbone_blocks"])
        kwargs[section] = cls(**fields)
    if "purpose" in value:
        kwargs["purpose"] = value["purpose"]
    if "schema_version" in value:
        kwargs["schema_version"] = value["schema_version"]
    config = ImageExperimentConfig(**kwargs)
    return config.validate()
