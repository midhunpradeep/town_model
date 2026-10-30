"""Immutable, validated image-experiment configuration authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Literal

from town_model.models.image_embeddings import validate_image_model_parameters
from town_model.vision_validation import validate_mobilenet_block_indices, validate_mobilenet_runtime


IMAGE_OBJECTIVE_REDUCTION = "known-weight-sum-normalized-per-instance-v2"
IMAGE_FEATURE_EXTRACTION_CONTRACT = "original_extent_roi_letterbox_padding_v2"
IMAGE_POSITIVE_WEIGHTING_POLICY = "positive-count-proportional-v2-all-in-scope-most-supported-class-unit-scaled"


def _finite_real(name: str, value: object, *, minimum: float, inclusive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric value")
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric value") from error
    if not math.isfinite(resolved) or (resolved < minimum if inclusive else resolved <= minimum):
        comparator = "at least" if inclusive else "greater than"
        raise ValueError(f"{name} must be finite and {comparator} {minimum}")
    return resolved


def _loss_identity(name: str, *, temperature: float, hinge_margin: float) -> dict[str, object]:
    if name == "bce":
        return {
            "name": "binary_cross_entropy_with_logits",
            "formula": "softplus(logit) - target * logit",
            "parameters": {},
        }
    if name == "signed_distance":
        return {
            "name": "signed_distance_hinge",
            "formula": "max(0, margin - (2 * target - 1) * signed_margin)",
            "parameters": {"margin": hinge_margin},
        }
    if name == "signed_bce":
        return {
            "name": "signed_margin_bce",
            "formula": "softplus(-(2 * target - 1) * signed_margin / temperature)",
            "parameters": {"temperature": temperature},
        }
    raise ValueError(f"unsupported image loss identity {name!r}")


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
        if not isinstance(self.dataset_path, str) or not self.dataset_path:
            raise ValueError("dataset_path must be a nonempty string")
        for name, value in (("train_size", self.train_size), ("validation_size", self.validation_size)):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if type(self.selection_seed) is not int or self.selection_seed < 0:
            raise ValueError("selection_seed must be a nonnegative integer")
        if (self.train_split, self.validation_split) != ("train", "validation"):
            raise ValueError("image experiments must select from original train and validation partitions")
        if self.trainable_class_ids is not None:
            if not isinstance(self.trainable_class_ids, tuple) or not self.trainable_class_ids:
                raise ValueError("explicit trainable class IDs must be a nonempty ordered tuple")
            if any(type(class_id) is not int or class_id < 0 for class_id in self.trainable_class_ids):
                raise ValueError("explicit trainable class IDs must be nonnegative integers")
            if len(set(self.trainable_class_ids)) != len(self.trainable_class_ids):
                raise ValueError("explicit trainable class IDs must be unique")


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
        validate_mobilenet_runtime()
        if type(self.pretrained_imagenet) is not bool:
            raise ValueError("pretrained_imagenet must be a boolean")
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
        blocks = validate_mobilenet_block_indices(self.trainable_backbone_blocks)
        if self.visual_policy == "trainable_fpn_last_blocks":
            if not blocks:
                raise ValueError("trainable_fpn_last_blocks requires explicit MobileNet block indices in [0,16]")
        elif blocks:
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
        if type(self.classifier_seed) is not int or self.classifier_seed < 0:
            raise ValueError("classifier_seed must be a nonnegative integer")
        validate_image_model_parameters(
            self.latent_dimensions,
            None,
            family=self.family,
            pizza_mode=self.pizza_mode,
            houses=self.houses,
            windows=self.windows,
            min_width=self.min_width,
            window_mode=self.window_mode,
            window_target_volume=self.window_target_volume,
        )


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    baseline_loss: str = "bce"
    geometry_loss: str = "signed_distance"
    temperature: float = 0.1
    hinge_margin: float = 0.1
    positive_weighting: Literal["none", "inverse_sqrt", "experimental_inverse_frequency", "effective_number"] = "none"
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
        _finite_real("signed BCE temperature", self.temperature, minimum=0.0, inclusive=False)
        _finite_real("hinge margin", self.hinge_margin, minimum=0.0, inclusive=True)
        if self.positive_weighting == "inverse_frequency":
            raise ValueError("legacy inverse_frequency is unsupported; use the explicitly experimental experimental_inverse_frequency policy")
        if self.positive_weighting not in ("none", "inverse_sqrt", "experimental_inverse_frequency", "effective_number"):
            raise ValueError("unsupported positive class-weight policy")
        _finite_real("positive_weight_cap", self.positive_weight_cap, minimum=1.0, inclusive=True)
        beta = _finite_real("effective_number_beta", self.effective_number_beta, minimum=0.0, inclusive=True)
        if beta >= 1.0:
            raise ValueError("effective_number_beta must be finite in [0,1)")
        for name, value in (("component_coefficient", self.component_coefficient), ("pizza_coefficient", self.pizza_coefficient)):
            _finite_real(name, value, minimum=0.0, inclusive=True)
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
            _finite_real(name, value, minimum=0.0, inclusive=False)
        for name, value in (("weight_decay_pretrained", self.weight_decay_pretrained), ("weight_decay_fpn", self.weight_decay_fpn), ("weight_decay_projection", self.weight_decay_projection), ("weight_decay_classifier", self.weight_decay_classifier)):
            _finite_real(name, value, minimum=0.0, inclusive=True)
        if self.schedule not in ("warmup_cosine", "constant"):
            raise ValueError("unsupported learning-rate schedule")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        if self.precision != "float32":
            raise ValueError("Phase 3 currently supports float32 only")
        if type(self.exploratory_latest_only) is not bool:
            raise ValueError("exploratory_latest_only must be a boolean")


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
        if self.feature_cache_directory is not None and (not isinstance(self.feature_cache_directory, str) or not self.feature_cache_directory):
            raise ValueError("feature_cache_directory must be a nonempty path string or None")


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    snapshot_every_epochs: int = 1
    max_reference_samples: int = 16
    max_component_traces_per_sample: int = 64
    include_roi_feature_means: bool = True

    def validate(self) -> None:
        if type(self.include_roi_feature_means) is not bool:
            raise ValueError("include_roi_feature_means must be a boolean")
        if any(type(value) is not int for value in (self.snapshot_every_epochs, self.max_reference_samples, self.max_component_traces_per_sample)):
            raise ValueError("snapshot frequencies and caps must be integers")
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
        alpha = _finite_real("image_overlay_alpha", self.image_overlay_alpha, minimum=0.0, inclusive=True)
        if alpha > 1:
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
        if self.schema_version != "image-run-config-v1":
            raise ValueError(f"unsupported image configuration schema {self.schema_version!r}")
        if self.purpose not in ("smoke", "exploratory", "comparison"):
            raise ValueError("purpose must be smoke, exploratory, or comparison")
        if self.purpose == "comparison" and self.training.exploratory_latest_only:
            raise ValueError("comparison runs require best-validation checkpoints; latest-only is for labeled smoke runs")
        if self.execution.feature_cache_directory and self.vision.visual_policy != "frozen":
            raise ValueError("feature caching is valid only for frozen visual features")
        if self.model.family != "town" and (self.model.houses != 2 or self.model.windows != 1 or self.model.window_mode != "contained"):
            raise ValueError("town geometry settings are invalid for non-town classifier families")
        return self

    def objective_identity(self) -> dict[str, object]:
        """Return the effective family loss formulas and their resolved parameters."""
        if self.model.family == "baseline":
            component_loss = pizza_loss = _loss_identity("bce", temperature=self.objective.temperature, hinge_margin=self.objective.hinge_margin)
        else:
            component_loss = pizza_loss = _loss_identity(
                self.objective.geometry_loss,
                temperature=self.objective.temperature,
                hinge_margin=self.objective.hinge_margin,
            )
        return {
            "family": self.model.family,
            "component_loss": component_loss,
            "pizza_loss": pizza_loss,
            "positive_weighting": self.objective.positive_weighting,
            "positive_weighting_semantics": IMAGE_POSITIVE_WEIGHTING_POLICY,
            "positive_weighting_formula": {
                "inverse_sqrt": "r_c=1/sqrt(n_c^+)",
                "effective_number": "r_c=(1-beta)/(1-beta^n_c^+)",
                "experimental_inverse_frequency": "r_c=1/n_c^+ (explicit experimental alternative)",
                "normalization": "scale raw values across all positive-supported canonical classes in this supervision scope so the most-supported class has weight 1; then cap each positive weight at positive_weight_cap and project to selected classes",
                "negative_weight": 1.0,
                "zero_positive_support": "weight 1 for exploratory runs; comparison coverage fails before training",
            },
            "positive_weight_cap": self.objective.positive_weight_cap,
            "effective_number_beta": self.objective.effective_number_beta,
            "component_coefficient": self.objective.component_coefficient,
            "pizza_coefficient": self.objective.pizza_coefficient,
            "reduction": IMAGE_OBJECTIVE_REDUCTION,
        }

    def identity_dict(self) -> dict[str, object]:
        """Scientific/training identity deliberately excludes presentation preferences."""
        self.validate()
        value = asdict(self)
        value.pop("visualization", None)
        value.pop("diagnostics", None)
        value["execution"].pop("feature_cache_directory", None)
        value["objective_semantics"] = IMAGE_OBJECTIVE_REDUCTION
        value["vision_feature_contract"] = IMAGE_FEATURE_EXTRACTION_CONTRACT
        value["resolved_objective"] = self.objective_identity()
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
            raw_ids = fields["trainable_class_ids"]
            if not isinstance(raw_ids, list):
                raise ValueError("trainable_class_ids must be a JSON array of integers")
            fields["trainable_class_ids"] = tuple(raw_ids)
        if section == "vision" and fields.get("trainable_backbone_blocks") is not None:
            raw_blocks = fields["trainable_backbone_blocks"]
            if not isinstance(raw_blocks, list):
                raise ValueError("trainable_backbone_blocks must be a JSON array of integers")
            fields["trainable_backbone_blocks"] = tuple(raw_blocks)
        kwargs[section] = cls(**fields)
    if "purpose" in value:
        kwargs["purpose"] = value["purpose"]
    if "schema_version" in value:
        kwargs["schema_version"] = value["schema_version"]
    config = ImageExperimentConfig(**kwargs)
    return config.validate()
