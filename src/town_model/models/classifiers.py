"""Shared point-to-ontology classifier interface for Phase 2 models."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from town_model.classifier_backend import raise_classifier_backend_unavailable

from .classifier_transport import ClassifierOutput
from .geometry import BoxGeometry, TownGeometry
from .losses import validate_signed_distance_margin


class PointClassifier(nn.Module):
    """Base contract: flat float32 points, stable trainable class ordering."""

    family: str

    def __init__(self, dimensions: int, class_names: tuple[str, ...], *, fixed_classes: tuple[str, ...] = ("DomainThing",)):
        raise_classifier_backend_unavailable("historical point classifier construction")
        super().__init__()
        if type(dimensions) is not int or dimensions < 1:
            raise ValueError("dimensions must be a positive integer")
        if not class_names or len(set(class_names)) != len(class_names):
            raise ValueError("class_names must be nonempty and unique")
        if set(class_names) & set(fixed_classes):
            raise ValueError("fixed-positive classes cannot be trainable classifier outputs")
        self.dimensions = int(dimensions)
        self.class_names = tuple(class_names)
        self.fixed_classes = tuple(fixed_classes)

    def _validate_points(self, points: Tensor) -> None:
        if not isinstance(points, Tensor) or points.dtype != torch.float32:
            raise ValueError("points must be a torch.float32 tensor")
        if points.ndim != 2 or points.shape[1] != self.dimensions:
            raise ValueError(f"points must have shape [N,{self.dimensions}]")
        if not torch.isfinite(points).all():
            raise ValueError("points must contain only finite values")

    def _fixed_outputs(self, count: int, device: torch.device) -> dict[str, Tensor]:
        return {
            name: torch.ones((count,), dtype=torch.bool, device=device)
            for name in self.fixed_classes
        }

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, points: Tensor, *, diagnostics: bool = False) -> ClassifierOutput:
        raise_classifier_backend_unavailable("historical point classifier inference")


class NeuralBaseline(PointClassifier):
    """One shared Linear(d,C) multilabel head with zero-logit threshold."""

    family = "baseline"

    def __init__(self, dimensions: int, class_names: tuple[str, ...], *, seed: int = 42, fixed_classes: tuple[str, ...] = ("DomainThing",)):
        super().__init__(dimensions, class_names, fixed_classes=fixed_classes)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.linear = nn.Linear(dimensions, len(class_names), bias=True)
        self.seed = int(seed)

    def forward(self, points: Tensor, *, diagnostics: bool = False) -> ClassifierOutput:
        raise_classifier_backend_unavailable("historical neural classifier inference")
        self._validate_points(points)
        logits = self.linear(points)
        return ClassifierOutput(
            contains=logits >= 0.0,
            logits=logits,
            fixed_positive=self._fixed_outputs(points.shape[0], points.device),
        )


class BoxClassifier(PointClassifier):
    """One trainable closed box per trainable ontology class."""

    family = "box"

    def __init__(
        self,
        dimensions: int,
        class_names: tuple[str, ...],
        *,
        seed: int = 42,
        min_width: float = 1e-5,
        fixed_classes: tuple[str, ...] = ("DomainThing",),
    ):
        super().__init__(dimensions, class_names, fixed_classes=fixed_classes)
        self.geometry = BoxGeometry(dimensions, len(class_names), seed=seed, min_width=min_width)
        self.seed = int(seed)

    def forward(self, points: Tensor, *, diagnostics: bool = False) -> ClassifierOutput:
        raise_classifier_backend_unavailable("historical box classifier inference")
        self._validate_points(points)
        result = self.geometry.evaluate(points, diagnostics=diagnostics)
        return ClassifierOutput(
            contains=result.contains,
            signed_margin=result.signed_margin,
            diagnostics=result.diagnostics,
            fixed_positive=self._fixed_outputs(points.shape[0], points.device),
        )


class TownClassifier(PointClassifier):
    """One trainable union of perforated houses per class."""

    family = "town"

    def __init__(
        self,
        dimensions: int,
        class_names: tuple[str, ...],
        *,
        seed: int = 42,
        houses: int = 2,
        windows: int = 1,
        min_width: float = 1e-5,
        window_mode: str = "contained",
        window_target_volume: float = 0.25,
        fixed_classes: tuple[str, ...] = ("DomainThing",),
    ):
        super().__init__(dimensions, class_names, fixed_classes=fixed_classes)
        self.geometry = TownGeometry(
            dimensions,
            len(class_names),
            houses=houses,
            windows=windows,
            seed=seed,
            min_width=min_width,
            window_mode=window_mode,
            window_target_volume=window_target_volume,
        )
        self.seed = int(seed)

    def forward(self, points: Tensor, *, diagnostics: bool = False) -> ClassifierOutput:
        raise_classifier_backend_unavailable("historical town classifier inference")
        self._validate_points(points)
        result = self.geometry.evaluate(points, diagnostics=diagnostics)
        return ClassifierOutput(
            contains=result.contains,
            signed_margin=result.signed_margin,
            diagnostics=result.diagnostics,
            fixed_positive=self._fixed_outputs(points.shape[0], points.device),
        )


def make_classifier(
    family: Literal["baseline", "box", "town"],
    dimensions: int,
    class_names: tuple[str, ...],
    *,
    seed: int = 42,
    houses: int = 2,
    windows: int = 1,
    min_width: float = 1e-5,
    window_mode: str = "contained",
    window_target_volume: float = 0.25,
) -> PointClassifier:
    """Historical factory retained only as source during the T2 transition."""
    raise_classifier_backend_unavailable(f"historical {family} classifier construction")
    if family == "baseline":
        return NeuralBaseline(dimensions, class_names, seed=seed)
    if family == "box":
        return BoxClassifier(dimensions, class_names, seed=seed, min_width=min_width)
    if family == "town":
        return TownClassifier(
            dimensions,
            class_names,
            seed=seed,
            houses=houses,
            windows=windows,
            min_width=min_width,
            window_mode=window_mode,
            window_target_volume=window_target_volume,
        )
    raise ValueError(f"unsupported classifier family: {family}")


class SyntheticPointEncoder(nn.Module):
    """Small synthetic-only feature-to-point mapper; labels are never inputs."""

    def __init__(self, input_dimensions: int, latent_dimensions: int, *, seed: int, hidden_dimensions: int = 16):
        super().__init__()
        if any(type(value) is not int for value in (input_dimensions, latent_dimensions, hidden_dimensions)) or input_dimensions < 1 or latent_dimensions < 1 or hidden_dimensions < 1:
            raise ValueError("encoder dimensions must be positive integers")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(input_dimensions, hidden_dimensions),
                nn.Tanh(),
                nn.Linear(hidden_dimensions, latent_dimensions),
                nn.Tanh(),
            )
        self.input_dimensions = input_dimensions
        self.latent_dimensions = latent_dimensions
        self.seed = int(seed)

    def forward(self, features: Tensor) -> Tensor:
        if features.dtype != torch.float32 or features.ndim != 2 or features.shape[1] != self.input_dimensions:
            raise ValueError(f"features must be float32 [N,{self.input_dimensions}]")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        return self.network(features)


def unreduced_penalties(
    output: ClassifierOutput,
    targets: Tensor,
    *,
    mode: str,
    margin: float = 0.1,
) -> Tensor:
    """Return historical penalties (unavailable during classifier reconstruction)."""
    raise_classifier_backend_unavailable("historical classifier loss computation")
    margin = validate_signed_distance_margin(margin)
    if output.logits is not None:
        if mode != "bce":
            raise ValueError("the neural baseline supports only standard BCE")
        return F.binary_cross_entropy_with_logits(output.logits, targets.to(dtype=torch.float32), reduction="none")
    if output.signed_margin is None:
        raise ValueError("geometric classifier output requires signed_margin")
    if mode != "signed_distance":
        raise ValueError("Phase 2 geometric classifiers support only signed_distance")
    signs = targets.to(dtype=torch.float32) * 2.0 - 1.0
    return torch.relu(margin - signs * output.signed_margin)
