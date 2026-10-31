"""Temporary compatibility adapter applying the historical classifier to points."""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from town_model.contracts import ImageModelResult, VisionFeatureBatch

from .classifier_transport import ClassifierOutput
from .geometry_diagnostics import GEOMETRY_DIAGNOSTIC_AXES
from .image_model_validation import validate_image_model_parameters
from .image_points import (
    ComponentPointProjection,
    DeepSetsComposition,
    GlobalPointProjection,
    HybridPointFusion,
    ImagePointModel,
    PizzaMode,
)


def _reshape_classifier_output(output: ClassifierOutput, batch: int, count: int) -> ClassifierOutput:
    """Adapt the current classifier DTO to padded component/image axes."""
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


class ImageOntologyModel(ImagePointModel):
    """Compatibility adapter for the currently active classifier families.

    Point extraction/composition lives in ``ImagePointModel`` and does not
    construct or require a classifier. This adapter preserves the historical
    model behavior until model execution is disconnected in a later pass.
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
        super().__init__(dimensions, pizza_mode=pizza_mode, seed=seed)
        # Keep the historical factory inside this explicitly compatibility-only
        # adapter. Importing point extraction, config validation, or transport
        # contracts does not import concrete classifier implementations.
        from .classifiers import make_classifier

        self.classifier: nn.Module = make_classifier(
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
        self.class_names = tuple(class_names)
        self.family = family
        self.classifier_initialization_seed = int(seed if classifier_seed is None else classifier_seed)

    def forward(
        self,
        features: VisionFeatureBatch,
        sample_ids: tuple[str, ...],
        *,
        diagnostics: bool = False,
    ) -> ImageModelResult:
        points = super().forward(features, sample_ids, diagnostics=diagnostics)
        batch, count = points.component_valid_mask.shape
        flat_components = points.component_points.reshape(batch * count, self.dimensions)
        component_output = self.classifier(flat_components, diagnostics=diagnostics)
        component_output = _reshape_classifier_output(component_output, batch, count)
        pizza_output = self.classifier(points.final_pizza_points, diagnostics=diagnostics)
        return ImageModelResult(
            sample_ids=points.sample_ids,
            component_ids=points.component_ids,
            component_valid_mask=points.component_valid_mask,
            component_points=points.component_points,
            global_pizza_points=points.global_pizza_points,
            composed_pizza_points=points.composed_pizza_points,
            final_pizza_points=points.final_pizza_points,
            component_classifier=component_output,
            pizza_classifier=pizza_output,
            pizza_mode=points.pizza_mode,
            vision_diagnostics=points.vision_diagnostics,
            geometry_diagnostics=(
                {
                    "component": component_output.diagnostics,
                    "pizza": pizza_output.diagnostics,
                }
                if diagnostics
                else None
            ),
        )
