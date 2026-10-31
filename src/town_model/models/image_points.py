"""Classifier-independent image-feature to ontology-point infrastructure."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from town_model.contracts import ImagePointResult, VisionFeatureBatch

PizzaMode = Literal["global", "compositional", "hybrid"]


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


class ImagePointModel(nn.Module):
    """Trainable image-to-point path with no classifier or scoring semantics.

    The caller supplies vision feature tensors and the authoritative sample and
    occurrence identities. This module returns points only; a future classifier
    integration can consume those points without adopting the historical
    logits, margins, membership, or loss representation.
    """

    def __init__(self, dimensions: int, *, pizza_mode: PizzaMode, seed: int = 42) -> None:
        super().__init__()
        if type(dimensions) is not int or dimensions < 1:
            raise ValueError("latent dimensions must be a positive integer")
        if pizza_mode not in ("global", "compositional", "hybrid"):
            raise ValueError(f"unsupported pizza composition mode {pizza_mode!r}")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.component_projection = ComponentPointProjection(dimensions)
            self.global_projection = GlobalPointProjection(dimensions)
            self.composition = DeepSetsComposition(dimensions)
            self.fusion = HybridPointFusion(dimensions)
        self.dimensions = dimensions
        self.pizza_mode = pizza_mode
        self.common_initialization_seed = int(seed)

    def common_state_dict(self) -> dict[str, Tensor]:
        """State of the shared point/composition modules."""
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
    ) -> ImagePointResult:
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
        return ImagePointResult(
            sample_ids=sample_ids,
            component_ids=features.component_ids,
            component_valid_mask=valid,
            component_points=component_points,
            global_pizza_points=global_points,
            composed_pizza_points=composed_points,
            final_pizza_points=final_points,
            pizza_mode=self.pizza_mode,
            vision_diagnostics=features.diagnostics if diagnostics else None,
        )
