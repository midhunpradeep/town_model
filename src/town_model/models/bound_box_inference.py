"""Identity-bound, point-only inference over the accepted BOX-03 kernels.

Class IDs and manifest hashes identify columns/rows; they cannot prove that an
upstream caller has attached the intended semantic meaning to anonymous
numerical geometry tensors. Frozen records are shallowly immutable: tensor
storage can still be mutated, so physical invariants are rechecked per query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import ClassVar, Literal, Mapping, Sequence

import torch
from torch import Tensor

from town_model.dataset.image_data import PreparedDataset, load_prepared_dataset
from town_model.models.bounded_multi_box import (
    bounded_multi_box_contains,
    multi_box_signed_clearance,
)

_CLASS_BINDING_SCHEMA = "town-model.class-binding.v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CLASS_BINDING_FIELDS = frozenset(
    {"schema", "trainable_class_ids", "class_manifest_sha256"}
)
_BINDING_VERIFICATION_TOKEN = object()


def _validated_class_ids(
    values: object, *, field_name: str = "trainable_class_ids"
) -> tuple[int, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field_name} must be an ordered tuple or list of integers")
    ids = tuple(values)
    if any(type(class_id) is not int for class_id in ids):
        raise TypeError(
            f"{field_name} must contain nonnegative integers (bool is not accepted)"
        )
    if any(class_id < 0 for class_id in ids):
        raise ValueError(f"{field_name} must contain only nonnegative class IDs")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{field_name} must not contain duplicate IDs")
    return ids


def _validated_sha256(
    value: object, *, field_name: str = "class_manifest_sha256"
) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest"
        )
    return value


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True, init=False)
class ClassBinding:
    """Immutable exact class-column identity from a verified vocabulary.

    Use :meth:`from_prepared_dataset` for production dataset identities. The
    explicitly named synthetic factory is intended only for fixed fixtures
    and tests; it derives IDs from the supplied vocabulary bytes and hashes
    those exact bytes rather than trusting a separate caller-supplied tuple or
    digest.
    """

    trainable_class_ids: tuple[int, ...]
    class_manifest_sha256: str
    schema_id: str
    _verification_source: str = field(repr=False, compare=False)
    _verification_token: object = field(repr=False, compare=False)

    SCHEMA_ID: ClassVar[str] = _CLASS_BINDING_SCHEMA

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("use a verified ClassBinding factory")

    @classmethod
    def _create(
        cls,
        class_ids: object,
        class_manifest_sha256: object,
        verification_source: str,
        *,
        _token: object,
    ) -> ClassBinding:
        if _token is not _BINDING_VERIFICATION_TOKEN:
            raise TypeError(
                "ClassBinding creation requires a verified vocabulary factory"
            )
        ids = _validated_class_ids(class_ids)
        digest = _validated_sha256(class_manifest_sha256)
        instance = object.__new__(cls)
        object.__setattr__(instance, "trainable_class_ids", ids)
        object.__setattr__(instance, "class_manifest_sha256", digest)
        object.__setattr__(instance, "schema_id", _CLASS_BINDING_SCHEMA)
        object.__setattr__(instance, "_verification_source", verification_source)
        object.__setattr__(instance, "_verification_token", _BINDING_VERIFICATION_TOKEN)
        return instance

    @classmethod
    def from_prepared_dataset(cls, dataset: PreparedDataset) -> ClassBinding:
        """Revalidate a prepared dataset at its root before binding its columns."""
        if not isinstance(dataset, PreparedDataset):
            raise TypeError(
                "dataset must be a PreparedDataset loaded by the dataset API"
            )
        verified = load_prepared_dataset(dataset.root, validate=True)
        reported_ids = _validated_class_ids(dataset.canonical_class_ids)
        verified_ids = _validated_class_ids(verified.canonical_class_ids)
        if reported_ids != verified_ids:
            raise ValueError(
                "PreparedDataset class IDs disagree with its revalidated vocabulary"
            )
        reported_digest = _validated_sha256(dataset.class_manifest_sha256)
        if reported_digest != verified.class_manifest_sha256:
            raise ValueError(
                "PreparedDataset vocabulary hash disagrees with its revalidated bytes"
            )
        return cls._create(
            verified_ids,
            verified.class_manifest_sha256,
            "prepared-dataset",
            _token=_BINDING_VERIFICATION_TOKEN,
        )

    @classmethod
    def from_synthetic_vocabulary_bytes(
        cls, canonical_vocabulary_bytes: bytes
    ) -> ClassBinding:
        """Create a fixture-only identity derived from explicit vocabulary bytes.

        The accepted synthetic schema is intentionally small: an object with
        ``classes`` (objects containing exactly ``class_id`` and ``trainable``)
        and ``trainable_class_ids``. IDs are derived from the class rows and
        checked against the declared order; the digest is computed over the
        original bytes. This is not a substitute for a validated prepared
        dataset identity.
        """
        if type(canonical_vocabulary_bytes) is not bytes:
            raise TypeError("canonical_vocabulary_bytes must be bytes")
        try:
            vocabulary = json.loads(
                canonical_vocabulary_bytes,
                object_pairs_hook=_strict_json_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "synthetic canonical vocabulary must be valid UTF-8 JSON"
            ) from error
        if type(vocabulary) is not dict or set(vocabulary) != {
            "classes",
            "trainable_class_ids",
        }:
            raise ValueError(
                "synthetic vocabulary must contain exactly classes and trainable_class_ids"
            )
        classes = vocabulary["classes"]
        if type(classes) is not list:
            raise ValueError("synthetic vocabulary classes must be a list")
        all_ids: list[int] = []
        derived_trainable: list[int] = []
        for index, row in enumerate(classes):
            if type(row) is not dict or set(row) != {"class_id", "trainable"}:
                raise ValueError(
                    f"synthetic vocabulary class row {index} has an invalid schema"
                )
            class_id = row["class_id"]
            trainable = row["trainable"]
            if type(class_id) is not int or class_id < 0:
                raise ValueError(
                    f"synthetic vocabulary class row {index} has an invalid class_id"
                )
            if type(trainable) is not bool:
                raise ValueError(
                    f"synthetic vocabulary class row {index} has an invalid trainable flag"
                )
            all_ids.append(class_id)
            if trainable:
                derived_trainable.append(class_id)
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("synthetic vocabulary class IDs must be unique")
        declared_trainable = _validated_class_ids(vocabulary["trainable_class_ids"])
        derived_ids = tuple(derived_trainable)
        if declared_trainable != derived_ids:
            raise ValueError(
                "synthetic trainable_class_ids disagree with ordered class rows"
            )
        digest = hashlib.sha256(canonical_vocabulary_bytes).hexdigest()
        return cls._create(
            derived_ids,
            digest,
            "synthetic-vocabulary",
            _token=_BINDING_VERIFICATION_TOKEN,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a fresh strict JSON-compatible representation."""
        return {
            "schema": self.schema_id,
            "trainable_class_ids": list(self.trainable_class_ids),
            "class_manifest_sha256": self.class_manifest_sha256,
        }

    def to_json(self) -> str:
        """Serialize deterministically without whitespace or nondeterministic key order."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(
        cls, serialized: Mapping[str, object], *, expected_binding: ClassBinding
    ) -> ClassBinding:
        """Deserialize only when an independent verified identity is supplied."""
        _require_verified_binding(expected_binding, "expected_binding")
        if not isinstance(serialized, Mapping):
            raise TypeError("serialized binding must be a mapping")
        if set(serialized) != _CLASS_BINDING_FIELDS:
            raise ValueError("serialized binding has missing or extra fields")
        schema = serialized["schema"]
        if type(schema) is not str or schema != _CLASS_BINDING_SCHEMA:
            raise ValueError(f"unsupported class-binding schema {schema!r}")
        raw_ids = serialized["trainable_class_ids"]
        if type(raw_ids) is not list:
            raise TypeError("serialized trainable_class_ids must be a JSON array")
        ids = _validated_class_ids(raw_ids)
        digest = _validated_sha256(serialized["class_manifest_sha256"])
        if ids != expected_binding.trainable_class_ids:
            raise ValueError(
                "serialized ordered class IDs do not match the expected binding"
            )
        if digest != expected_binding.class_manifest_sha256:
            raise ValueError(
                "serialized vocabulary hash does not match the expected binding"
            )
        return cls._create(
            ids,
            digest,
            expected_binding._verification_source,
            _token=_BINDING_VERIFICATION_TOKEN,
        )

    @classmethod
    def from_json(
        cls, serialized: str, *, expected_binding: ClassBinding
    ) -> ClassBinding:
        """Strictly parse and verify one serialized binding against authority."""
        _require_verified_binding(expected_binding, "expected_binding")
        if type(serialized) is not str:
            raise TypeError("serialized binding must be a JSON string")
        try:
            value = json.loads(serialized, object_pairs_hook=_strict_json_object)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                "serialized binding must be valid JSON without duplicate fields"
            ) from error
        if type(value) is not dict:
            raise ValueError("serialized binding must be a JSON object")
        return cls.from_dict(value, expected_binding=expected_binding)


@dataclass(frozen=True, slots=True, init=False)
class BoundBoxGeometry:
    """Shallowly immutable physical corners with explicit class-to-row identity.

    ``row_class_ids[i]`` names rows ``lower[i]`` and ``upper[i]``. The record
    deliberately retains the caller's tensors (and autograd graph) without
    sorting, copying, casting, or detaching them. A vocabulary hash and row-ID
    match cannot detect an upstream caller who falsely labels anonymous
    numerical rows; that provenance limit is not a claim of semantic proof.
    """

    lower: Tensor
    upper: Tensor
    row_class_ids: tuple[int, ...]
    class_manifest_sha256: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("use BoundBoxGeometry.from_ordered_rows")

    @classmethod
    def from_ordered_rows(
        cls,
        lower: Tensor,
        upper: Tensor,
        *,
        row_class_ids: Sequence[int],
        class_manifest_sha256: str,
        expected_binding: ClassBinding,
    ) -> BoundBoxGeometry:
        """Validate explicit ordered row associations against independent identity."""
        _require_verified_binding(expected_binding, "expected_binding")
        ids = _validated_class_ids(row_class_ids, field_name="row_class_ids")
        digest = _validated_sha256(class_manifest_sha256)
        if ids != expected_binding.trainable_class_ids:
            raise ValueError(
                "geometry row class IDs do not match the expected ordered binding"
            )
        if digest != expected_binding.class_manifest_sha256:
            raise ValueError(
                "geometry vocabulary hash does not match the expected binding"
            )
        _validate_corner_tensors(lower, upper)
        if lower.shape[0] != len(ids):
            raise ValueError(
                "geometry row count must equal the number of associated class IDs"
            )
        instance = object.__new__(cls)
        object.__setattr__(instance, "lower", lower)
        object.__setattr__(instance, "upper", upper)
        object.__setattr__(instance, "row_class_ids", ids)
        object.__setattr__(instance, "class_manifest_sha256", digest)
        return instance


def _require_verified_binding(value: object, name: str) -> ClassBinding:
    if not isinstance(value, ClassBinding):
        raise TypeError(f"{name} must be a verified ClassBinding")
    if (
        value.schema_id != _CLASS_BINDING_SCHEMA
        or value._verification_source
        not in {"prepared-dataset", "synthetic-vocabulary"}
        or value._verification_token is not _BINDING_VERIFICATION_TOKEN
    ):
        raise ValueError(f"{name} has no supported verified identity provenance")
    return value


def _require_float32_tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if value.layout != torch.strided:
        raise TypeError(f"{name} must be a dense strided tensor")
    return value


def _validate_tensor_device(*values: Tensor) -> None:
    first = values[0].device
    if first.type not in ("cpu", "cuda"):
        raise ValueError("inputs must use a supported CPU or CUDA device")
    if first.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if any(value.device != first for value in values[1:]):
        raise ValueError("points and geometry tensors must use the same device")


def _all_true(value: Tensor) -> bool:
    return bool(value.all().item())


def _validate_points_tensor(points: object, *, require_bounded: bool) -> Tensor:
    points = _require_float32_tensor(points, "points")
    if points.ndim != 2:
        raise ValueError("points must have exact shape [N, d]")
    if points.shape[1] < 1:
        raise ValueError("the point dimension d must be at least 1")
    _validate_tensor_device(points)
    if not _all_true(torch.isfinite(points)):
        raise ValueError("points must contain only finite values")
    if require_bounded and (
        not _all_true(points >= -1.0) or not _all_true(points <= 1.0)
    ):
        raise ValueError("bounded points must lie inside the closed domain [-1, 1]^d")
    return points


def _validate_corner_tensors(lower: object, upper: object) -> tuple[Tensor, Tensor]:
    lower = _require_float32_tensor(lower, "lower")
    upper = _require_float32_tensor(upper, "upper")
    if lower.ndim != 2 or upper.ndim != 2:
        raise ValueError("lower and upper must have exact shape [C, d]")
    if lower.shape != upper.shape:
        raise ValueError("lower and upper must have identical [C, d] shapes")
    if lower.shape[1] < 1:
        raise ValueError("the geometry dimension d must be at least 1")
    _validate_tensor_device(lower, upper)
    if not _all_true(torch.isfinite(lower)) or not _all_true(torch.isfinite(upper)):
        raise ValueError("box corners must contain only finite values")
    if not _all_true(lower >= -1.0) or not _all_true(upper <= 1.0):
        raise ValueError("box corners must lie inside the fixed domain [-1, 1]^d")
    if not _all_true(lower <= upper):
        raise ValueError(
            "lower must be less than or equal to upper in every coordinate"
        )
    return lower, upper


def validate_bounded_points(points: Tensor) -> Tensor:
    """Strictly validate `[N,d]` float32 points in the closed bounded domain.

    This separate adapter-facing validator accepts exact domain faces and
    returns the original tensor, preserving its identity and autograd history.
    Pure inference queries themselves may intentionally be outside the domain.
    """
    return _validate_points_tensor(points, require_bounded=True)


@dataclass(frozen=True, slots=True)
class BoxInferenceResult:
    """Identity-bearing typed outputs from one pure bounded BOX query.

    Tensor fields are retained without cloning or detaching; frozen dataclass
    fields do not make their underlying storage deeply immutable.
    """

    class_binding: ClassBinding
    contains: Tensor
    signed_coordinate_clearance: Tensor
    domain_contains: Tensor
    family: Literal["box"] = field(default="box", init=False)

    def __post_init__(self) -> None:
        _require_verified_binding(self.class_binding, "class_binding")
        for name, value in (
            ("contains", self.contains),
            ("signed_coordinate_clearance", self.signed_coordinate_clearance),
            ("domain_contains", self.domain_contains),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.layout != torch.strided:
                raise TypeError(f"{name} must be a dense strided tensor")
        if self.contains.dtype is not torch.bool or self.contains.ndim != 2:
            raise ValueError("contains must be a Boolean matrix [N, C]")
        if self.signed_coordinate_clearance.dtype is not torch.float32:
            raise TypeError("signed_coordinate_clearance must have dtype torch.float32")
        if self.signed_coordinate_clearance.ndim != 2:
            raise ValueError("signed_coordinate_clearance must be a matrix [N, C]")
        if self.contains.shape != self.signed_coordinate_clearance.shape:
            raise ValueError(
                "contains and signed-coordinate clearance shapes must match"
            )
        if self.contains.shape[1] != len(self.class_binding.trainable_class_ids):
            raise ValueError("result class columns must match the exact class binding")
        if (
            self.domain_contains.dtype is not torch.bool
            or self.domain_contains.ndim != 1
        ):
            raise ValueError("domain_contains must be a Boolean vector [N]")
        if self.domain_contains.shape[0] != self.contains.shape[0]:
            raise ValueError("domain_contains batch size must match class outputs")
        if (
            self.contains.device != self.signed_coordinate_clearance.device
            or self.contains.device != self.domain_contains.device
        ):
            raise ValueError("all result tensors must be on the same device")


@dataclass(frozen=True, slots=True)
class BoundBoxClassifier:
    """Stateless immutable class binding for pure queries of bound box rows."""

    class_binding: ClassBinding

    def __post_init__(self) -> None:
        _require_verified_binding(self.class_binding, "class_binding")

    def infer(self, points: Tensor, geometry: BoundBoxGeometry) -> BoxInferenceResult:
        """Preflight exact identity and inputs, then call unchanged BOX-03 kernels."""
        _require_verified_binding(self.class_binding, "classifier class_binding")
        if not isinstance(geometry, BoundBoxGeometry):
            raise TypeError("geometry must be a BoundBoxGeometry")
        if geometry.row_class_ids != self.class_binding.trainable_class_ids:
            raise ValueError(
                "geometry ordered row class IDs do not match classifier binding"
            )
        if geometry.class_manifest_sha256 != self.class_binding.class_manifest_sha256:
            raise ValueError(
                "geometry vocabulary hash does not match classifier binding"
            )

        points = _validate_points_tensor(points, require_bounded=False)
        lower, upper = _validate_corner_tensors(geometry.lower, geometry.upper)
        if lower.shape[0] != len(geometry.row_class_ids):
            raise ValueError("geometry row count no longer matches its row class IDs")
        if points.shape[1] != lower.shape[1]:
            raise ValueError("point and geometry dimensions must match")
        _validate_tensor_device(points, lower, upper)

        # Complete preflight happens before either evaluator is called. These
        # direct calls are the accepted, unchanged BOX-03 mathematical kernels.
        contains = bounded_multi_box_contains(points, lower, upper)
        clearance = multi_box_signed_clearance(points, lower, upper)
        domain_contains = ((points >= -1.0) & (points <= 1.0)).all(dim=-1)
        return BoxInferenceResult(
            class_binding=self.class_binding,
            contains=contains,
            signed_coordinate_clearance=clearance,
            domain_contains=domain_contains,
        )
