from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import struct
import json

import pytest
import torch

from town_model.dataset.image_data import load_prepared_dataset
from town_model.models.bounded_multi_box import (
    bounded_multi_box_contains,
    multi_box_signed_clearance,
)
from town_model.models.bounded_multi_box_fixture import (
    BOUNDED_MULTI_BOX_INSPECTION_FIXTURE,
)
from town_model.models.bound_box_inference import (
    BoundBoxClassifier,
    BoundBoxGeometry,
    ClassBinding,
    validate_bounded_points,
)

SYNTHETIC_VOCABULARY_BYTES = (
    b'{"classes":[{"class_id":12,"trainable":true},'
    b'{"class_id":27,"trainable":true}],"trainable_class_ids":[12,27]}\n'
)
SYNTHETIC_VOCABULARY_SHA256 = (
    "fd793ac6c2c061c851f6f4dcf15fc6b417e5c7b4d554b569f71245587d174b8b"
)


def _synthetic_binding() -> ClassBinding:
    return ClassBinding.from_synthetic_vocabulary_bytes(SYNTHETIC_VOCABULARY_BYTES)


def _geometry(
    binding: ClassBinding,
    *,
    row_class_ids: tuple[int, ...] = (12, 27),
    vocabulary_hash: str | None = None,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
) -> BoundBoxGeometry:
    if lower is None:
        lower = torch.tensor([[-1.0, -0.5], [-0.25, -1.0]], dtype=torch.float32)
    if upper is None:
        upper = torch.tensor([[0.5, 1.0], [1.0, 0.5]], dtype=torch.float32)
    return BoundBoxGeometry.from_ordered_rows(
        lower,
        upper,
        row_class_ids=row_class_ids,
        class_manifest_sha256=(
            binding.class_manifest_sha256
            if vocabulary_hash is None
            else vocabulary_hash
        ),
        expected_binding=binding,
    )


def _serialized(binding: ClassBinding) -> dict[str, object]:
    return binding.to_dict()


def test_ID01_ID02_prepared_dataset_binding_revalidates_authoritative_vocabulary(
    prepared_dataset,
) -> None:
    dataset = load_prepared_dataset(prepared_dataset, validate=True)
    vocabulary_bytes = (
        dataset.root / "vocabularies" / "ontology_classes.json"
    ).read_bytes()

    binding = ClassBinding.from_prepared_dataset(dataset)

    assert binding.trainable_class_ids == dataset.canonical_class_ids
    assert binding.class_manifest_sha256 == hashlib.sha256(vocabulary_bytes).hexdigest()
    assert binding.class_manifest_sha256 == dataset.class_manifest_sha256
    assert binding.schema_id == "town-model.class-binding.v1"
    assert binding._verification_source == "prepared-dataset"
    assert (
        binding.class_manifest_sha256
        != hashlib.sha256(repr(binding.trainable_class_ids).encode()).hexdigest()
    )


def test_ID02_prepared_dataset_reported_identity_cannot_override_revalidated_bytes(
    prepared_dataset,
) -> None:
    dataset = load_prepared_dataset(prepared_dataset, validate=True)

    with pytest.raises(ValueError, match="class IDs disagree"):
        ClassBinding.from_prepared_dataset(
            replace(
                dataset,
                canonical_class_ids=tuple(reversed(dataset.canonical_class_ids)),
            )
        )
    with pytest.raises(ValueError, match="hash disagrees"):
        ClassBinding.from_prepared_dataset(
            replace(dataset, class_manifest_sha256="0" * 64)
        )


def test_ID01_synthetic_binding_derives_order_and_hash_from_fixed_vocabulary_bytes() -> (
    None
):
    binding = _synthetic_binding()

    assert binding.trainable_class_ids == (12, 27)
    assert binding.class_manifest_sha256 == SYNTHETIC_VOCABULARY_SHA256
    assert (
        hashlib.sha256(SYNTHETIC_VOCABULARY_BYTES).hexdigest()
        == SYNTHETIC_VOCABULARY_SHA256
    )
    assert binding._verification_source == "synthetic-vocabulary"

    whitespace_variant = SYNTHETIC_VOCABULARY_BYTES.replace(
        b'],"trainable', b'], "trainable'
    )
    same_ids_different_bytes = ClassBinding.from_synthetic_vocabulary_bytes(
        whitespace_variant
    )
    assert same_ids_different_bytes.trainable_class_ids == binding.trainable_class_ids
    assert (
        same_ids_different_bytes.class_manifest_sha256 != binding.class_manifest_sha256
    )
    assert same_ids_different_bytes != binding


def test_ID01_synthetic_binding_allows_verified_empty_vocabulary_only_from_empty_bytes() -> (
    None
):
    empty_bytes = b'{"classes":[],"trainable_class_ids":[]}\n'
    binding = ClassBinding.from_synthetic_vocabulary_bytes(empty_bytes)

    assert binding.trainable_class_ids == ()
    assert binding.class_manifest_sha256 == hashlib.sha256(empty_bytes).hexdigest()


@pytest.mark.parametrize(
    ("raw", "error_type"),
    (
        (b"not-json", ValueError),
        (b'{"classes":[],"trainable_class_ids":[],"classes":[]}', ValueError),
        (
            b'{"classes":[{"class_id":true,"trainable":true}],"trainable_class_ids":[1]}',
            ValueError,
        ),
        (
            b'{"classes":[{"class_id":2,"trainable":1}],"trainable_class_ids":[2]}',
            ValueError,
        ),
        (
            b'{"classes":[{"class_id":2,"trainable":true},{"class_id":2,"trainable":false}],"trainable_class_ids":[2]}',
            ValueError,
        ),
        (
            b'{"classes":[{"class_id":2,"trainable":true}],"trainable_class_ids":[3]}',
            ValueError,
        ),
        (
            b'{"classes":[{"class_id":2,"trainable":true}],"trainable_class_ids":[true]}',
            TypeError,
        ),
        (bytearray(SYNTHETIC_VOCABULARY_BYTES), TypeError),
    ),
)
def test_ID02_synthetic_vocabulary_rejects_malformed_or_unverified_bytes(
    raw, error_type
) -> None:
    with pytest.raises(error_type):
        ClassBinding.from_synthetic_vocabulary_bytes(raw)


def test_ID03_deterministic_strict_json_round_trip_requires_expected_binding() -> None:
    binding = _synthetic_binding()
    expected_json = (
        '{"class_manifest_sha256":"'
        + SYNTHETIC_VOCABULARY_SHA256
        + '","schema":"town-model.class-binding.v1","trainable_class_ids":[12,27]}'
    )

    serialized = binding.to_json()
    restored = ClassBinding.from_json(serialized, expected_binding=_synthetic_binding())

    assert serialized == expected_json
    assert binding.to_json() == serialized
    assert restored == binding
    assert restored is not binding
    assert restored._verification_source == "synthetic-vocabulary"
    with pytest.raises(TypeError):
        ClassBinding.from_json(serialized)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="verified ClassBinding"):
        ClassBinding.from_json(serialized, expected_binding=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="verified ClassBinding"):
        ClassBinding.from_dict(binding.to_dict(), expected_binding=object())  # type: ignore[arg-type]


def test_ID03_serialized_mapping_is_a_fresh_json_compatible_value() -> None:
    binding = _synthetic_binding()
    serialized = binding.to_dict()
    serialized["trainable_class_ids"] = [27, 12]

    assert binding.trainable_class_ids == (12, 27)
    assert binding.to_dict()["trainable_class_ids"] == [12, 27]


@pytest.mark.parametrize(
    ("mutate", "error_type", "message"),
    (
        (
            lambda value: value.update(schema="town-model.class-binding.v2"),
            ValueError,
            "unsupported",
        ),
        (lambda value: value.pop("schema"), ValueError, "missing or extra"),
        (lambda value: value.update(extra="no"), ValueError, "missing or extra"),
        (
            lambda value: value.update(trainable_class_ids=(12, 27)),
            TypeError,
            "JSON array",
        ),
        (
            lambda value: value.update(trainable_class_ids=[12, True]),
            TypeError,
            "nonnegative integers",
        ),
        (
            lambda value: value.update(trainable_class_ids=[12, -1]),
            ValueError,
            "nonnegative",
        ),
        (
            lambda value: value.update(trainable_class_ids=[12, 12]),
            ValueError,
            "duplicate",
        ),
        (
            lambda value: value.update(class_manifest_sha256="A" * 64),
            ValueError,
            "lowercase",
        ),
        (
            lambda value: value.update(class_manifest_sha256="abc"),
            ValueError,
            "lowercase",
        ),
        (
            lambda value: value.update(trainable_class_ids=[27, 12]),
            ValueError,
            "ordered class IDs",
        ),
        (
            lambda value: value.update(class_manifest_sha256="0" * 64),
            ValueError,
            "vocabulary hash",
        ),
    ),
)
def test_ID03_ID04_deserialization_rejects_invalid_or_mismatched_identity(
    mutate, error_type, message
) -> None:
    expected = _synthetic_binding()
    before = expected.to_json()
    payload = _serialized(expected)
    mutate(payload)

    with pytest.raises(error_type, match=message):
        ClassBinding.from_dict(payload, expected_binding=expected)

    assert expected.to_json() == before


def test_ID03_deserialization_rejects_duplicate_json_fields() -> None:
    binding = _synthetic_binding()
    serialized = binding.to_json()
    duplicate = serialized[:-1] + ',"schema":"town-model.class-binding.v1"}'

    with pytest.raises(ValueError, match="duplicate fields"):
        ClassBinding.from_json(duplicate, expected_binding=binding)
    with pytest.raises(ValueError, match="JSON object"):
        ClassBinding.from_json("[]", expected_binding=binding)


def test_ID05_class_binding_construction_is_factory_only_and_immutable() -> None:
    binding = _synthetic_binding()

    with pytest.raises(TypeError, match="verified ClassBinding factory"):
        ClassBinding()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ClassBinding._create((12, 27), "0" * 64, "prepared-dataset")  # type: ignore[call-arg]
    with pytest.raises(FrozenInstanceError):
        binding.trainable_class_ids = (27, 12)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("ids", "error_type", "message"),
    (
        ((12, 12), ValueError, "duplicate"),
        ((12, -1), ValueError, "nonnegative"),
        ((12, True), TypeError, "bool is not accepted"),
    ),
)
def test_ID05_geometry_rejects_duplicate_invalid_or_boolean_row_ids(
    ids, error_type, message
) -> None:
    binding = _synthetic_binding()

    with pytest.raises(error_type, match=message):
        _geometry(binding, row_class_ids=ids)


@pytest.mark.parametrize(
    ("ids", "message"),
    (
        ((27, 12), "ordered binding"),
        ((12,), "ordered binding"),
        ((12, 27, 33), "ordered binding"),
    ),
)
def test_GEO01_GEO02_geometry_rejects_same_width_permutation_missing_and_extra_ids(
    ids, message
) -> None:
    binding = _synthetic_binding()

    with pytest.raises(ValueError, match=message):
        _geometry(binding, row_class_ids=ids)


def test_GEO01_GEO02_geometry_rejects_wrong_vocabulary_hash_before_record_creation() -> (
    None
):
    binding = _synthetic_binding()
    lower = torch.tensor([[-1.0, -0.5], [-0.25, -1.0]], dtype=torch.float32)
    upper = torch.tensor([[0.5, 1.0], [1.0, 0.5]], dtype=torch.float32)
    lower_before = lower.clone()
    upper_before = upper.clone()

    with pytest.raises(ValueError, match="hash does not match"):
        _geometry(binding, vocabulary_hash="0" * 64, lower=lower, upper=upper)

    assert torch.equal(lower, lower_before)
    assert torch.equal(upper, upper_before)


def test_GEO01_geometry_keeps_explicit_row_association_and_tensor_graph() -> None:
    binding = _synthetic_binding()
    lower = torch.tensor(
        [[-1.0, -0.5], [-0.25, -1.0]], dtype=torch.float32, requires_grad=True
    )
    upper = torch.tensor(
        [[0.5, 1.0], [1.0, 0.5]], dtype=torch.float32, requires_grad=True
    )

    geometry = _geometry(binding, lower=lower, upper=upper)

    assert geometry.row_class_ids == (12, 27)
    assert geometry.class_manifest_sha256 == binding.class_manifest_sha256
    assert geometry.lower is lower
    assert geometry.upper is upper
    assert geometry.lower.requires_grad and geometry.upper.requires_grad
    with pytest.raises(FrozenInstanceError):
        geometry.row_class_ids = (27, 12)  # type: ignore[misc]
    with pytest.raises(TypeError, match="from_ordered_rows"):
        BoundBoxGeometry()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("lower", "upper", "error_type", "message"),
    (
        (torch.zeros((2,)), torch.ones((2,)), ValueError, "exact shape"),
        (torch.zeros((2, 2)), torch.ones((1, 2)), ValueError, "identical"),
        (torch.zeros((2, 0)), torch.ones((2, 0)), ValueError, "dimension d"),
        (
            torch.zeros((2, 2), dtype=torch.float64),
            torch.ones((2, 2)),
            TypeError,
            "float32",
        ),
        (torch.zeros((2, 2)), torch.full((2, 2), float("nan")), ValueError, "finite"),
        (
            torch.tensor([[-1.1, 0.0], [-0.2, 0.0]]),
            torch.zeros((2, 2)),
            ValueError,
            "domain",
        ),
        (
            torch.zeros((2, 2)),
            torch.tensor([[0.0, 1.1], [0.0, 0.0]]),
            ValueError,
            "domain",
        ),
        (
            torch.tensor([[0.5, 0.0], [-0.2, 0.0]]),
            torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            ValueError,
            "less than or equal",
        ),
        (
            torch.sparse_coo_tensor(
                torch.tensor([[0], [0]]), torch.tensor([-0.5]), (2, 2)
            ),
            torch.zeros((2, 2)),
            TypeError,
            "dense strided",
        ),
    ),
)
def test_GEO04_geometry_construction_strictly_validates_physical_corner_tensors(
    lower, upper, error_type, message
) -> None:
    binding = _synthetic_binding()

    with pytest.raises(error_type, match=message):
        _geometry(binding, lower=lower, upper=upper)


def test_GEO01_geometry_row_count_must_match_associated_class_ids() -> None:
    binding = _synthetic_binding()
    lower = torch.tensor([[-1.0, -0.5]], dtype=torch.float32)
    upper = torch.tensor([[0.5, 1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="row count"):
        _geometry(binding, lower=lower, upper=upper)


def _fixture_tensors(device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
    fixture = BOUNDED_MULTI_BOX_INSPECTION_FIXTURE
    return (
        torch.tensor(fixture.points, dtype=torch.float32, device=device),
        torch.tensor(fixture.lower, dtype=torch.float32, device=device),
        torch.tensor(fixture.upper, dtype=torch.float32, device=device),
    )


def _device_for_test(device_name: str) -> torch.device:
    if device_name == "cuda":
        assert (
            torch.cuda.is_available()
        ), "BOX-04 CUDA acceptance requires a usable CUDA runtime"
        return torch.device("cuda:0")
    return torch.device("cpu")


def _bound_fixture(
    device: torch.device | str = "cpu", binding: ClassBinding | None = None
):
    binding = _synthetic_binding() if binding is None else binding
    points, lower, upper = _fixture_tensors(device)
    geometry = BoundBoxGeometry.from_ordered_rows(
        lower,
        upper,
        row_class_ids=binding.trainable_class_ids,
        class_manifest_sha256=binding.class_manifest_sha256,
        expected_binding=binding,
    )
    return binding, points, lower, upper, geometry


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_INF01_INF02_wrapper_returns_typed_identity_and_exact_BOX03_fixture(
    device_name: str,
) -> None:
    device = _device_for_test(device_name)
    binding, points, lower, upper, geometry = _bound_fixture(device)
    result = BoundBoxClassifier(binding).infer(points, geometry)
    expected = BOUNDED_MULTI_BOX_INSPECTION_FIXTURE

    assert result.family == "box"
    assert result.class_binding is binding
    assert result.contains.shape == (6, 2)
    assert result.signed_coordinate_clearance.shape == (6, 2)
    assert result.domain_contains.shape == (6,)
    assert result.contains.dtype is torch.bool
    assert result.signed_coordinate_clearance.dtype is torch.float32
    assert result.domain_contains.dtype is torch.bool
    assert result.contains.device == device
    assert result.signed_coordinate_clearance.device == device
    assert result.domain_contains.device == device
    assert torch.equal(
        result.contains,
        torch.tensor(expected.expected_membership, dtype=torch.bool, device=device),
    )
    assert torch.equal(
        result.signed_coordinate_clearance,
        torch.tensor(expected.expected_clearances, dtype=torch.float32, device=device),
    )
    assert result.domain_contains.tolist() == [True, True, True, True, True, False]
    assert torch.equal(
        result.contains, bounded_multi_box_contains(points, lower, upper)
    )
    assert torch.equal(
        result.signed_coordinate_clearance,
        multi_box_signed_clearance(points, lower, upper),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def test_INF03_INF04_domain_thing_is_separate_and_outside_domain_query_is_permitted() -> (
    None
):
    binding, points, lower, upper, geometry = _bound_fixture()
    result = BoundBoxClassifier(binding).infer(points, geometry)

    assert result.contains.shape == (len(points), len(binding.trainable_class_ids))
    assert result.domain_contains.shape == (len(points),)
    assert result.domain_contains.tolist() == [True, True, True, True, True, False]
    assert result.contains[-1].tolist() == [False, False]
    assert result.signed_coordinate_clearance[-1].tolist() == [-0.25, -1.0]
    assert torch.equal(
        result.signed_coordinate_clearance,
        multi_box_signed_clearance(points, lower, upper),
    )


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
@pytest.mark.parametrize("point_count,box_count", ((3, 2), (0, 2), (3, 0), (0, 0)))
def test_INF06_empty_axes_preserve_valid_identity_shapes_and_device(
    device_name: str, point_count: int, box_count: int
) -> None:
    device = _device_for_test(device_name)
    binding = (
        _synthetic_binding()
        if box_count
        else ClassBinding.from_synthetic_vocabulary_bytes(
            b'{"classes":[],"trainable_class_ids":[]}\n'
        )
    )
    points = torch.zeros((point_count, 2), dtype=torch.float32, device=device)
    if box_count:
        lower = torch.tensor(
            [[-1.0, -0.5], [-0.25, -1.0]], dtype=torch.float32, device=device
        )
        upper = torch.tensor(
            [[0.5, 1.0], [1.0, 0.5]], dtype=torch.float32, device=device
        )
    else:
        lower = torch.empty((0, 2), dtype=torch.float32, device=device)
        upper = torch.empty((0, 2), dtype=torch.float32, device=device)
    geometry = BoundBoxGeometry.from_ordered_rows(
        lower,
        upper,
        row_class_ids=binding.trainable_class_ids,
        class_manifest_sha256=binding.class_manifest_sha256,
        expected_binding=binding,
    )

    result = BoundBoxClassifier(binding).infer(points, geometry)

    assert result.contains.shape == (point_count, box_count)
    assert result.signed_coordinate_clearance.shape == (point_count, box_count)
    assert result.domain_contains.shape == (point_count,)
    assert result.contains.device == device
    assert result.signed_coordinate_clearance.device == device
    assert result.domain_contains.device == device
    assert torch.equal(
        result.contains, bounded_multi_box_contains(points, lower, upper)
    )
    assert torch.equal(
        result.signed_coordinate_clearance,
        multi_box_signed_clearance(points, lower, upper),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def test_INF06_nonempty_binding_cannot_be_bypassed_by_empty_geometry() -> None:
    binding = _synthetic_binding()
    lower = torch.empty((0, 2), dtype=torch.float32)
    upper = torch.empty((0, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="row class IDs"):
        BoundBoxGeometry.from_ordered_rows(
            lower,
            upper,
            row_class_ids=(),
            class_manifest_sha256=binding.class_manifest_sha256,
            expected_binding=binding,
        )


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_INF07_empty_axes_do_not_bypass_point_or_mutated_corner_validation(
    device_name: str, monkeypatch
) -> None:
    import town_model.models.bound_box_inference as inference_module

    device = _device_for_test(device_name)
    empty_binding = ClassBinding.from_synthetic_vocabulary_bytes(
        b'{"classes":[],"trainable_class_ids":[]}\n'
    )
    empty_lower = torch.empty((0, 1), dtype=torch.float32, device=device)
    empty_upper = torch.empty((0, 1), dtype=torch.float32, device=device)
    empty_geometry = BoundBoxGeometry.from_ordered_rows(
        empty_lower,
        empty_upper,
        row_class_ids=(),
        class_manifest_sha256=empty_binding.class_manifest_sha256,
        expected_binding=empty_binding,
    )
    binding = _synthetic_binding()
    points = torch.empty((0, 2), dtype=torch.float32, device=device)
    lower = torch.tensor(
        [[-1.0, -0.5], [-0.25, -1.0]], dtype=torch.float32, device=device
    )
    upper = torch.tensor([[0.5, 1.0], [1.0, 0.5]], dtype=torch.float32, device=device)
    geometry = _geometry(binding, lower=lower, upper=upper)
    calls: list[str] = []
    monkeypatch.setattr(
        inference_module,
        "bounded_multi_box_contains",
        lambda *args: calls.append("contains"),
    )
    monkeypatch.setattr(
        inference_module,
        "multi_box_signed_clearance",
        lambda *args: calls.append("clearance"),
    )

    with pytest.raises(ValueError, match="finite"):
        BoundBoxClassifier(empty_binding).infer(
            torch.tensor([[float("nan")]], dtype=torch.float32, device=device),
            empty_geometry,
        )
    with torch.no_grad():
        lower[0, 0] = -1.25
    with pytest.raises(ValueError, match="domain"):
        BoundBoxClassifier(binding).infer(points, geometry)
    assert calls == []
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _different_hash_binding() -> ClassBinding:
    alternate = SYNTHETIC_VOCABULARY_BYTES.replace(b'],"trainable', b'], "trainable')
    return ClassBinding.from_synthetic_vocabulary_bytes(alternate)


def _reversed_binding() -> ClassBinding:
    reversed_bytes = (
        b'{"classes":[{"class_id":27,"trainable":true},'
        b'{"class_id":12,"trainable":true}],"trainable_class_ids":[27,12]}\n'
    )
    return ClassBinding.from_synthetic_vocabulary_bytes(reversed_bytes)


@pytest.mark.parametrize("mismatch", ("row-order", "vocabulary-hash"))
def test_GEO02_INF07_identity_mismatch_rejects_before_either_kernel(
    mismatch: str, monkeypatch
) -> None:
    import town_model.models.bound_box_inference as inference_module

    expected = _synthetic_binding()
    other = (
        _reversed_binding() if mismatch == "row-order" else _different_hash_binding()
    )
    lower = torch.tensor([[-1.0, -0.5], [-0.25, -1.0]], dtype=torch.float32)
    upper = torch.tensor([[0.5, 1.0], [1.0, 0.5]], dtype=torch.float32)
    geometry = BoundBoxGeometry.from_ordered_rows(
        lower,
        upper,
        row_class_ids=other.trainable_class_ids,
        class_manifest_sha256=other.class_manifest_sha256,
        expected_binding=other,
    )
    calls: list[str] = []
    monkeypatch.setattr(
        inference_module,
        "bounded_multi_box_contains",
        lambda *args: calls.append("contains"),
    )
    monkeypatch.setattr(
        inference_module,
        "multi_box_signed_clearance",
        lambda *args: calls.append("clearance"),
    )

    with pytest.raises(ValueError, match="row class IDs|vocabulary hash"):
        BoundBoxClassifier(expected).infer(
            torch.zeros((1, 2), dtype=torch.float32), geometry
        )

    assert calls == []
    assert torch.equal(lower, torch.tensor([[-1.0, -0.5], [-0.25, -1.0]]))
    assert torch.equal(upper, torch.tensor([[0.5, 1.0], [1.0, 0.5]]))


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_GEO06_mutated_invalid_geometry_is_revalidated_before_scoring(
    device_name: str, monkeypatch
) -> None:
    import town_model.models.bound_box_inference as inference_module

    device = _device_for_test(device_name)
    binding, points, lower, upper, geometry = _bound_fixture(device)
    calls: list[str] = []
    monkeypatch.setattr(
        inference_module,
        "bounded_multi_box_contains",
        lambda *args: calls.append("contains"),
    )
    monkeypatch.setattr(
        inference_module,
        "multi_box_signed_clearance",
        lambda *args: calls.append("clearance"),
    )
    with torch.no_grad():
        lower[0, 0] = -1.25

    with pytest.raises(ValueError, match="domain"):
        BoundBoxClassifier(binding).infer(points, geometry)

    assert calls == []
    assert upper is geometry.upper


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_INF07_wrapper_preflight_checks_points_dimension_finiteness_and_device(
    device_name: str, monkeypatch
) -> None:
    import town_model.models.bound_box_inference as inference_module

    device = _device_for_test(device_name)
    binding, points, _, _, geometry = _bound_fixture(device)
    calls: list[str] = []
    monkeypatch.setattr(
        inference_module,
        "bounded_multi_box_contains",
        lambda *args: calls.append("contains"),
    )
    monkeypatch.setattr(
        inference_module,
        "multi_box_signed_clearance",
        lambda *args: calls.append("clearance"),
    )
    invalid_points = (
        (torch.zeros((1, 2, 1), dtype=torch.float32, device=device), ValueError),
        (torch.zeros((1, 3), dtype=torch.float32, device=device), ValueError),
        (
            torch.tensor([[float("nan"), 0.0]], dtype=torch.float32, device=device),
            ValueError,
        ),
        (torch.zeros((1, 2), dtype=torch.float64, device=device), TypeError),
        (
            torch.zeros(
                (1, 2),
                dtype=torch.float32,
                device="cpu" if device.type == "cuda" else "cuda",
            ),
            ValueError,
        ),
    )
    for bad_points, error_type in invalid_points:
        with pytest.raises(error_type):
            BoundBoxClassifier(binding).infer(bad_points, geometry)
    assert calls == []
    assert points.device == device


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_INF05_bounded_point_validator_accepts_exact_faces_empty_and_preserves_tensor(
    device_name: str,
) -> None:
    device = _device_for_test(device_name)
    points = torch.tensor(
        [[-1.0, 1.0], [0.0, 0.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    empty = torch.empty((0, 2), dtype=torch.float32, device=device, requires_grad=True)

    assert validate_bounded_points(points) is points
    assert validate_bounded_points(empty) is empty
    assert points.requires_grad and empty.requires_grad
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@pytest.mark.parametrize(
    "bad_points,error_type",
    (
        (torch.zeros((1, 1), dtype=torch.float64), TypeError),
        (torch.zeros((1, 1, 1), dtype=torch.float32), ValueError),
        (torch.empty((1, 0), dtype=torch.float32), ValueError),
        (torch.tensor([[1.0001]], dtype=torch.float32), ValueError),
        (torch.tensor([[-1.0001]], dtype=torch.float32), ValueError),
        (torch.tensor([[float("inf")]], dtype=torch.float32), ValueError),
        (
            torch.sparse_coo_tensor(
                torch.tensor([[0], [0]]), torch.tensor([0.0]), (1, 1)
            ),
            TypeError,
        ),
    ),
)
def test_INF05_bounded_point_validator_rejects_invalid_types_shapes_and_domain(
    bad_points: torch.Tensor, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        validate_bounded_points(bad_points)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_GEO05_INF08_result_keeps_original_autograd_paths_and_per_box_gradient_isolation(
    device_name: str,
) -> None:
    device = _device_for_test(device_name)
    binding = _synthetic_binding()
    points = torch.tensor(
        [[0.0, 0.0]], dtype=torch.float32, device=device, requires_grad=True
    )
    lower = torch.tensor(
        [[-1.0, -0.5], [-0.25, -1.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    upper = torch.tensor(
        [[0.5, 1.0], [1.0, 0.5]], dtype=torch.float32, device=device, requires_grad=True
    )
    geometry = _geometry(binding, lower=lower, upper=upper)

    result = BoundBoxClassifier(binding).infer(points, geometry)
    point_grad, lower_grad, upper_grad = torch.autograd.grad(
        result.signed_coordinate_clearance[0, 0], (points, lower, upper)
    )

    assert result.signed_coordinate_clearance.grad_fn is not None
    assert result.signed_coordinate_clearance.requires_grad
    assert result.contains.grad_fn is None
    assert result.domain_contains.grad_fn is None
    assert geometry.lower is lower and geometry.upper is upper
    assert point_grad.tolist() == [[-0.5, 0.5]]
    assert lower_grad[0].tolist() == [0.0, -0.5]
    assert upper_grad[0].tolist() == [0.5, 0.0]
    assert torch.equal(lower_grad[1], torch.zeros_like(lower_grad[1]))
    assert torch.equal(upper_grad[1], torch.zeros_like(upper_grad[1]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_GEO05_GEO06_tied_amin_gradients_match_direct_BOX03_per_device(
    device_name: str,
) -> None:
    device = _device_for_test(device_name)
    binding = _synthetic_binding()
    points = torch.zeros((1, 2), dtype=torch.float32, device=device, requires_grad=True)
    lower = torch.full(
        (2, 2), -1.0, dtype=torch.float32, device=device, requires_grad=True
    )
    upper = torch.full(
        (2, 2), 1.0, dtype=torch.float32, device=device, requires_grad=True
    )
    geometry = _geometry(binding, lower=lower, upper=upper)

    result = BoundBoxClassifier(binding).infer(points, geometry)
    bound_gradients = torch.autograd.grad(
        result.signed_coordinate_clearance[0, 0],
        (points, lower, upper),
        retain_graph=True,
    )
    direct = multi_box_signed_clearance(points, lower, upper)
    direct_gradients = torch.autograd.grad(direct[0, 0], (points, lower, upper))

    assert result.signed_coordinate_clearance[0, 0].item() == 1.0
    for bound_gradient, direct_gradient in zip(
        bound_gradients, direct_gradients, strict=True
    ):
        assert torch.equal(bound_gradient, direct_gradient)
    assert torch.equal(bound_gradients[1][1], torch.zeros_like(bound_gradients[1][1]))
    assert torch.equal(bound_gradients[2][1], torch.zeros_like(bound_gradients[2][1]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _ordered_float32_bits(value: float) -> int:
    bits = int.from_bytes(struct.pack(">f", value), byteorder="big")
    if bits & 0x80000000:
        return 0x80000000 - (bits & 0x7FFFFFFF)
    return bits | 0x80000000


def test_N03_cpu_cuda_wrapper_forward_values_follow_declared_ulp_and_domain_budget() -> (
    None
):
    binding = _synthetic_binding()
    fixture = BOUNDED_MULTI_BOX_INSPECTION_FIXTURE
    lower_cpu = torch.tensor(fixture.lower, dtype=torch.float32)
    upper_cpu = torch.tensor(fixture.upper, dtype=torch.float32)
    edge = torch.tensor(0.5, dtype=torch.float32)
    neighbors = (
        torch.nextafter(edge, torch.tensor(float("-inf"))),
        edge,
        torch.nextafter(edge, torch.tensor(float("inf"))),
    )
    maximum = torch.finfo(torch.float32).max
    smallest = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)).item()
    points_cpu = torch.tensor(
        [
            [float(neighbors[0]), 0.0],
            [float(neighbors[1]), 0.0],
            [float(neighbors[2]), 0.0],
            [-1.25, 0.0],
            [maximum, -maximum],
            [smallest, -smallest],
        ],
        dtype=torch.float32,
    )
    geometry_cpu = BoundBoxGeometry.from_ordered_rows(
        lower_cpu,
        upper_cpu,
        row_class_ids=binding.trainable_class_ids,
        class_manifest_sha256=binding.class_manifest_sha256,
        expected_binding=binding,
    )
    cpu_result = BoundBoxClassifier(binding).infer(points_cpu, geometry_cpu)

    device = _device_for_test("cuda")
    points_cuda = points_cpu.to(device)
    lower_cuda = lower_cpu.to(device)
    upper_cuda = upper_cpu.to(device)
    geometry_cuda = BoundBoxGeometry.from_ordered_rows(
        lower_cuda,
        upper_cuda,
        row_class_ids=binding.trainable_class_ids,
        class_manifest_sha256=binding.class_manifest_sha256,
        expected_binding=binding,
    )
    cuda_result = BoundBoxClassifier(binding).infer(points_cuda, geometry_cuda)
    torch.cuda.synchronize(device)

    cpu_clearance = cpu_result.signed_coordinate_clearance
    cuda_clearance = cuda_result.signed_coordinate_clearance.cpu()
    assert torch.isfinite(cpu_clearance).all()
    assert torch.isfinite(cuda_clearance).all()
    assert torch.equal(cpu_result.contains, cuda_result.contains.cpu())
    assert torch.equal(cpu_result.domain_contains, cuda_result.domain_contains.cpu())
    assert torch.equal(cpu_clearance == 0, cuda_clearance == 0)
    nonzero = cpu_clearance != 0
    assert torch.equal(
        torch.signbit(cpu_clearance[nonzero]), torch.signbit(cuda_clearance[nonzero])
    )
    ulp_distances = [
        abs(_ordered_float32_bits(left) - _ordered_float32_bits(right))
        for left, right in zip(
            cpu_clearance.reshape(-1).tolist(),
            cuda_clearance.reshape(-1).tolist(),
            strict=True,
        )
    ]
    assert max(ulp_distances, default=0) <= 1


def test_GEO05_cpu_cuda_tied_degenerate_gradients_follow_declared_absolute_budget() -> (
    None
):
    binding = _synthetic_binding()

    def evaluate(device: torch.device) -> tuple[torch.Tensor, ...]:
        points = torch.zeros(
            (1, 2), dtype=torch.float32, device=device, requires_grad=True
        )
        lower = torch.zeros(
            (2, 2), dtype=torch.float32, device=device, requires_grad=True
        )
        upper = torch.zeros(
            (2, 2), dtype=torch.float32, device=device, requires_grad=True
        )
        geometry = _geometry(binding, lower=lower, upper=upper)
        result = BoundBoxClassifier(binding).infer(points, geometry)
        gradients = torch.autograd.grad(
            result.signed_coordinate_clearance.sum(), (points, lower, upper)
        )
        assert torch.isfinite(result.signed_coordinate_clearance).all()
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        return tuple(gradient.detach().cpu() for gradient in gradients)

    cpu_gradients = evaluate(torch.device("cpu"))
    cuda_gradients = evaluate(_device_for_test("cuda"))
    torch.cuda.synchronize()
    maximum_absolute_difference = max(
        float((cpu - cuda).abs().max().item())
        for cpu, cuda in zip(cpu_gradients, cuda_gradients, strict=True)
    )
    assert maximum_absolute_difference <= 1e-6
