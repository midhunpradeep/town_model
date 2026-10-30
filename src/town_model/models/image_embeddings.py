"""Image-derived ontology point projections and pizza composition mathematics."""

from __future__ import annotations

from typing import Literal
import math
from numbers import Real

import torch
from torch import Tensor, nn

from town_model.contracts import ImageModelResult, VisionFeatureBatch

from .classifiers import ClassifierOutput, PointClassifier, make_classifier
from .geometry import GEOMETRY_DIAGNOSTIC_AXES, validate_min_width

PizzaMode = Literal["global", "compositional", "hybrid"]


def validate_image_model_parameters(
    dimensions: object,
    class_names: object,
    *,
    family: object,
    pizza_mode: object,
    houses: object,
    windows: object,
    min_width: object,
    window_mode: object,
    window_target_volume: object,
) -> None:
    """Pure constructor-equivalent model validation shared with config preflight."""
    if type(dimensions) is not int or dimensions < 1:
        raise ValueError("latent dimensions must be a positive integer")
    if family not in ("baseline", "box", "town"):
        raise ValueError(f"unsupported classifier family {family!r}")
    if pizza_mode not in ("global", "compositional", "hybrid"):
        raise ValueError(f"unsupported pizza composition mode {pizza_mode!r}")
    if class_names is not None:
        if not isinstance(class_names, tuple) or not class_names or any(not isinstance(name, str) or not name for name in class_names):
            raise ValueError("class_names must be a nonempty ordered tuple of names")
        if len(set(class_names)) != len(class_names):
            raise ValueError("class_names must be unique")
    if type(houses) is not int or type(windows) is not int or houses < 1 or windows < 1:
        raise ValueError("town houses and windows must be positive integer counts")
    if window_mode != "contained":
        raise ValueError("Phase 3 currently supports the contained town window parameterization")
    if isinstance(window_target_volume, bool) or not isinstance(window_target_volume, Real):
        raise ValueError("window_target_volume must be finite and between zero and one")
    try:
        target_volume = float(window_target_volume)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("window_target_volume must be finite and between zero and one") from error
    if not math.isfinite(target_volume) or not 0 < target_volume < 1:
        raise ValueError("window_target_volume must be finite and between zero and one")
    if family != "town" and (houses != 2 or windows != 1 or window_mode != "contained" or window_target_volume != 0.25):
        raise ValueError("town geometry settings are incompatible with non-town classifier families")
    validate_min_width(min_width, model_families=(family,), window_mode=window_mode)


class ComponentPointProjection(nn.Module):
    """Specified 260→256→128→d bounded component point head."""

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(260, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, dimensions),
            nn.Tanh(),
        )

    def forward(self, component_features: Tensor) -> Tensor:
        if component_features.ndim != 3 or component_features.shape[-1] != 260:
            raise ValueError("component visual features must have shape [B,K,260]")
        return self.layers(component_features)


class GlobalPointProjection(nn.Module):
    """Specified spatial 256×7×7→32→1568→256→d image point head."""

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(256, 32, kernel_size=1)
        self.hidden = nn.Linear(32 * 7 * 7, 256)
        self.output = nn.Linear(256, dimensions)

    def forward(self, spatial_features: Tensor) -> Tensor:
        if spatial_features.ndim != 4 or spatial_features.shape[1:] != (256, 7, 7):
            raise ValueError("global spatial features must have shape [B,256,7,7]")
        value = self.spatial(spatial_features).flatten(start_dim=1)
        # ReLU after the 1568→256 projection is the documented activation choice.
        value = torch.relu(self.hidden(value))
        return torch.tanh(self.output(value))


class DeepSetsComposition(nn.Module):
    """Sum-based permutation-invariant component-occurrence composition."""

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(dimensions, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dimensions),
            nn.Tanh(),
        )

    def forward(self, component_points: Tensor, valid_mask: Tensor) -> Tensor:
        if component_points.ndim != 3:
            raise ValueError("component_points must have shape [B,K,d]")
        if valid_mask.shape != component_points.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B,K]")
        transformed = self.phi(component_points)
        transformed = transformed * valid_mask.unsqueeze(-1).to(transformed.dtype)
        summed = transformed.sum(dim=1)
        return self.rho(summed)


class HybridPointFusion(nn.Module):
    """Bounded 2d→128→d fusion, with no auxiliary intermediate pizza losses."""

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(2 * dimensions, 128)
        self.output = nn.Linear(128, dimensions)

    def forward(self, global_points: Tensor, composed_points: Tensor) -> Tensor:
        if global_points.shape != composed_points.shape or global_points.ndim != 2:
            raise ValueError("global and composed points must have matching [B,d] shapes")
        value = torch.relu(self.hidden(torch.cat((global_points, composed_points), dim=-1)))
        return torch.tanh(self.output(value))


def _reshape_classifier_output(output: ClassifierOutput, batch: int, count: int) -> ClassifierOutput:
    def reshape_optional(value: Tensor | None) -> Tensor | None:
        return value.reshape(batch, count, *value.shape[1:]) if value is not None else None

    fixed = {
        name: value.reshape(batch, count, *value.shape[1:])
        for name, value in (output.fixed_positive or {}).items()
    }
    diagnostics = None
    if output.diagnostics:
        diagnostics = {}
        point_count, class_count = output.contains.shape
        for name, value in output.diagnostics.items():
            axes = GEOMETRY_DIAGNOSTIC_AXES.get(name)
            if axes is None:
                raise ValueError(f"classifier diagnostic {name!r} has no declared semantic axes")
            if axes[0] == "point":
                if value.shape[:2] != (point_count, class_count):
                    raise ValueError(f"point diagnostic {name!r} does not match [point,class,...] contract")
                diagnostics[name] = value.reshape(batch, count, *value.shape[1:])
            else:
                if value.shape[0] != class_count:
                    raise ValueError(f"class-level diagnostic {name!r} does not match classifier class count")
                diagnostics[name] = value
    return ClassifierOutput(
        contains=output.contains.reshape(batch, count, output.contains.shape[-1]),
        logits=reshape_optional(output.logits),
        signed_margin=reshape_optional(output.signed_margin),
        diagnostics=diagnostics,
        fixed_positive=fixed or None,
    )


class ImageOntologyModel(nn.Module):
    """Trainable point/composition/classifier model over shared visual features.

    Torchvision execution is intentionally outside this module. The caller
    supplies feature tensors produced by the vision adapter, and this model
    contains only point mathematics, composition, and the accepted classifier.
    """

    def __init__(
        self,
        dimensions: int,
        class_names: tuple[str, ...],
        *,
        family: Literal["baseline", "box", "town"],
        pizza_mode: PizzaMode,
        seed: int = 42,
        classifier_seed: int | None = None,
        houses: int = 2,
        windows: int = 1,
        min_width: float = 1e-5,
        window_mode: str = "contained",
        window_target_volume: float = 0.25,
    ) -> None:
        super().__init__()
        validate_image_model_parameters(
            dimensions,
            class_names,
            family=family,
            pizza_mode=pizza_mode,
            houses=houses,
            windows=windows,
            min_width=min_width,
            window_mode=window_mode,
            window_target_volume=window_target_volume,
        )
        # Common point/composition module states are seed-controlled and can be
        # copied exactly across classifier families. Geometry initialization is
        # separately seeded/documented by classifier_seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.component_projection = ComponentPointProjection(dimensions)
            self.global_projection = GlobalPointProjection(dimensions)
            self.composition = DeepSetsComposition(dimensions)
            self.fusion = HybridPointFusion(dimensions)
        self.classifier: PointClassifier = make_classifier(
            family,
            dimensions,
            class_names,
            seed=seed if classifier_seed is None else classifier_seed,
            houses=houses,
            windows=windows,
            min_width=min_width,
            window_mode=window_mode,
            window_target_volume=window_target_volume,
        )
        self.dimensions = dimensions
        self.class_names = tuple(class_names)
        self.family = family
        self.pizza_mode = pizza_mode
        self.common_initialization_seed = int(seed)
        self.classifier_initialization_seed = int(seed if classifier_seed is None else classifier_seed)

    def common_state_dict(self) -> dict[str, Tensor]:
        """State of modules that must start byte-identically across families."""
        return {
            **{f"component_projection.{name}": value for name, value in self.component_projection.state_dict().items()},
            **{f"global_projection.{name}": value for name, value in self.global_projection.state_dict().items()},
            **{f"composition.{name}": value for name, value in self.composition.state_dict().items()},
            **{f"fusion.{name}": value for name, value in self.fusion.state_dict().items()},
        }

    def forward(
        self,
        features: VisionFeatureBatch,
        sample_ids: tuple[str, ...],
        *,
        diagnostics: bool = False,
    ) -> ImageModelResult:
        vectors = features.component_vectors
        valid = features.component_valid_mask
        global_spatial = features.global_spatial_features
        if vectors.ndim != 3 or vectors.shape[-1] != 260:
            raise ValueError("component feature contract must be [B,K,260]")
        batch, count, _ = vectors.shape
        if len(sample_ids) != batch or valid.shape != (batch, count):
            raise ValueError("sample IDs/component valid mask disagree with visual features")
        if global_spatial.shape != (batch, 256, 7, 7):
            raise ValueError("global spatial features must be [B,256,7,7]")
        component_points = self.component_projection(vectors)
        global_points = self.global_projection(global_spatial)
        composed_points = self.composition(component_points, valid)
        if self.pizza_mode == "global":
            final_points = global_points
        elif self.pizza_mode == "compositional":
            final_points = composed_points
        else:
            final_points = self.fusion(global_points, composed_points)
        flat_components = component_points.reshape(batch * count, self.dimensions)
        component_output = self.classifier(flat_components, diagnostics=diagnostics)
        component_output = _reshape_classifier_output(component_output, batch, count)
        pizza_output = self.classifier(final_points, diagnostics=diagnostics)
        return ImageModelResult(
            sample_ids=sample_ids,
            component_ids=features.component_ids,
            component_valid_mask=valid,
            component_points=component_points,
            global_pizza_points=global_points,
            composed_pizza_points=composed_points,
            final_pizza_points=final_points,
            component_classifier=component_output,
            pizza_classifier=pizza_output,
            pizza_mode=self.pizza_mode,
            vision_diagnostics=features.diagnostics if diagnostics else None,
            geometry_diagnostics=(
                {
                    "component": component_output.diagnostics,
                    "pizza": pizza_output.diagnostics,
                }
                if diagnostics
                else None
            ),
        )
