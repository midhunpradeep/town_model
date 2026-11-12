"""Minimal trainable raw geometry bound to BOX-04 class identity."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from town_model.models.bound_box_inference import (
    BoundBoxClassifier,
    BoundBoxGeometry,
    BoxInferenceResult,
    ClassBinding,
)
from town_model.models.bounded_box_training_geometry import (
    INITIAL_CERTIFIED_DIMENSION,
    INITIAL_CERTIFIED_RAW_MAX,
    INITIAL_CERTIFIED_RAW_MIN,
    BoxWidthPolicy,
    PhysicalBoxInitialization,
    UnsupportedNumericalConfigurationError,
    initialize_bounded_boxes,
)
from town_model.models.bounded_trainable_box_geometry import (
    BoundedBoxGeometry,
    analytic_endpoint_jacobian,
    decode_bounded_box_raw,
    validate_bounded_box_geometry,
)


class InvalidTrainablePhysicalGeometryError(ValueError):
    """Actual decoded float32 geometry fails a hard physical invariant."""


@dataclass(frozen=True, slots=True)
class TrainableBoxPhysicalGeometry:
    """Float32 geometry plus immutable row identity and scalar diagnostics."""

    centers: Tensor
    half_widths: Tensor
    lower: Tensor
    upper: Tensor
    minimum_full_widths: tuple[float, ...]
    row_class_ids: tuple[int, ...]
    class_manifest_sha256: str
    occupancy: tuple[float, ...]
    log_occupancy: tuple[float, ...]
    conditioning_sigma_min: tuple[float, ...]


class TrainableBoundedBoxModule(nn.Module):
    """Own bounded `[C,2]` raw Parameters and expose BOX-04 inference only.

    Initialization is physical-first and all-or-nothing. The module owns no
    optimizer, loss, scheduler, or training lifecycle; its sole forward route
    returns the existing identity-bearing BOX-04 result.
    """

    def __init__(
        self,
        class_binding: ClassBinding,
        *,
        policies: Sequence[BoxWidthPolicy],
        occupancy_targets: Sequence[float],
        master_seed: int,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        initialization = initialize_bounded_boxes(
            class_binding,
            policies=policies,
            occupancy_targets=occupancy_targets,
            master_seed=master_seed,
            device=device,
        )
        self._class_binding = class_binding
        self._policies = initialization.policies
        self._initialization = initialization
        self.raw_width = nn.Parameter(initialization.raw_width.clone())
        self.raw_center = nn.Parameter(initialization.raw_center.clone())
        self._classifier = BoundBoxClassifier(class_binding)
        # Validate the actual Parameters that this module owns, not only the
        # temporary initializer tensors from which they were cloned.
        self.decode_physical_geometry()

    @classmethod
    def initialize(
        cls,
        class_binding: ClassBinding,
        *,
        policies: Sequence[BoxWidthPolicy],
        occupancy_targets: Sequence[float],
        master_seed: int,
        device: str | torch.device = "cpu",
    ) -> TrainableBoundedBoxModule:
        """Named constructor documenting physical-first initialization."""
        return cls(
            class_binding,
            policies=policies,
            occupancy_targets=occupancy_targets,
            master_seed=master_seed,
            device=device,
        )

    @property
    def class_binding(self) -> ClassBinding:
        return self._class_binding

    @property
    def row_class_ids(self) -> tuple[int, ...]:
        return self._class_binding.trainable_class_ids

    @property
    def policies(self) -> tuple[BoxWidthPolicy, ...]:
        return self._policies

    @property
    def initialization(self) -> PhysicalBoxInitialization:
        return self._initialization

    @property
    def initialization_records(self):
        return self._initialization.class_records

    def decode_physical_geometry(self) -> TrainableBoxPhysicalGeometry:
        """Decode and independently validate current float32 raw Parameters."""
        if self.raw_width.dtype is not torch.float32 or self.raw_center.dtype is not torch.float32:
            raise UnsupportedNumericalConfigurationError(
                "trainable bounded BOX geometry is certified only in float32"
            )
        if self.raw_width.layout != torch.strided or self.raw_center.layout != torch.strided:
            raise UnsupportedNumericalConfigurationError(
                "trainable bounded BOX Parameters must use dense strided tensors"
            )
        expected_shape = (len(self.row_class_ids), INITIAL_CERTIFIED_DIMENSION)
        if tuple(self.raw_width.shape) != expected_shape or tuple(self.raw_center.shape) != expected_shape:
            raise UnsupportedNumericalConfigurationError(
                f"raw Parameters must retain the bound [C,2] shape {expected_shape}"
            )
        if self.raw_width.device != self.raw_center.device:
            raise UnsupportedNumericalConfigurationError(
                "raw width and center Parameters must remain on the same device"
            )
        if self.raw_width.device.type not in ("cpu", "cuda"):
            raise UnsupportedNumericalConfigurationError(
                "trainable bounded BOX supports only CPU and CUDA devices"
            )
        if self.raw_width.device.type == "cuda" and not torch.cuda.is_available():
            raise UnsupportedNumericalConfigurationError("CUDA is not available")
        if len(self._policies) != len(self.row_class_ids):
            raise UnsupportedNumericalConfigurationError(
                "immutable width-policy rows no longer match class binding"
            )
        for policy in self._policies:
            policy.require_initial_certification(dimension=INITIAL_CERTIFIED_DIMENSION)

        row_geometries: list[BoundedBoxGeometry] = []
        for row_index, policy in enumerate(self._policies):
            try:
                row = decode_bounded_box_raw(
                    self.raw_width[row_index : row_index + 1],
                    self.raw_center[row_index : row_index + 1],
                    minimum_full_width=policy.minimum_full_width,
                )
                validate_bounded_box_geometry(row)
            except (TypeError, ValueError) as error:
                raise InvalidTrainablePhysicalGeometryError(
                    f"class row {row_index} ({self.row_class_ids[row_index]}) has invalid "
                    f"float32 physical geometry: {error}"
                ) from error
            row_geometries.append(row)

        if not row_geometries:
            empty_geometry = decode_bounded_box_raw(
                self.raw_width,
                self.raw_center,
                minimum_full_width=BoxWidthPolicy.ordinary().minimum_full_width,
            )
            validate_bounded_box_geometry(empty_geometry)
            return TrainableBoxPhysicalGeometry(
                centers=empty_geometry.centers,
                half_widths=empty_geometry.half_widths,
                lower=empty_geometry.lower,
                upper=empty_geometry.upper,
                minimum_full_widths=(),
                row_class_ids=(),
                class_manifest_sha256=self._class_binding.class_manifest_sha256,
                occupancy=(),
                log_occupancy=(),
                conditioning_sigma_min=(),
            )

        # Actual physical validity is checked before classifying unsupported raw
        # values, so a bad physical state is never mislabeled as only out of range.
        raw_in_range = (
            (self.raw_width >= INITIAL_CERTIFIED_RAW_MIN)
            & (self.raw_width <= INITIAL_CERTIFIED_RAW_MAX)
            & (self.raw_center >= INITIAL_CERTIFIED_RAW_MIN)
            & (self.raw_center <= INITIAL_CERTIFIED_RAW_MAX)
        )
        if not bool(raw_in_range.all().item()):
            bad_rows = torch.nonzero(~raw_in_range.all(dim=1), as_tuple=False)
            row_index = int(bad_rows[0, 0].item())
            raise UnsupportedNumericalConfigurationError(
                f"class row {row_index} ({self.row_class_ids[row_index]}) raw values "
                "are outside the certified operating range [-3,3]"
            )

        centers = torch.cat([geometry.centers for geometry in row_geometries], dim=0)
        half_widths = torch.cat([geometry.half_widths for geometry in row_geometries], dim=0)
        lower = torch.cat([geometry.lower for geometry in row_geometries], dim=0)
        upper = torch.cat([geometry.upper for geometry in row_geometries], dim=0)
        occupancies: list[float] = []
        log_occupancies: list[float] = []
        conditioning: list[float] = []
        for row_index, (geometry, policy) in enumerate(
            zip(row_geometries, self._policies, strict=True)
        ):
            halves = [float(value) for value in geometry.half_widths[0].detach().cpu().tolist()]
            log_q = sum(math.log(value) for value in halves)
            log_occupancies.append(log_q)
            occupancies.append(math.exp(log_q))
            with torch.no_grad():
                jacobian = analytic_endpoint_jacobian(
                    self.raw_width[row_index : row_index + 1].detach(),
                    self.raw_center[row_index : row_index + 1].detach(),
                    minimum_full_width=policy.minimum_full_width,
                )
                sigma_min = float(torch.linalg.svdvals(jacobian).min().item())
            conditioning.append(sigma_min)

        return TrainableBoxPhysicalGeometry(
            centers=centers,
            half_widths=half_widths,
            lower=lower,
            upper=upper,
            minimum_full_widths=tuple(
                policy.minimum_full_width for policy in self._policies
            ),
            row_class_ids=self.row_class_ids,
            class_manifest_sha256=self._class_binding.class_manifest_sha256,
            occupancy=tuple(occupancies),
            log_occupancy=tuple(log_occupancies),
            conditioning_sigma_min=tuple(conditioning),
        )

    def forward(self, points: Tensor) -> BoxInferenceResult:
        """Return BOX-04 membership/clearance while retaining autograd history."""
        physical = self.decode_physical_geometry()
        bound_geometry = BoundBoxGeometry.from_ordered_rows(
            physical.lower,
            physical.upper,
            row_class_ids=physical.row_class_ids,
            class_manifest_sha256=physical.class_manifest_sha256,
            expected_binding=self._class_binding,
        )
        return self._classifier.infer(points, bound_geometry)
