"""Synthetic class-identity fixture for the read-only BOX inference inspector.

IDs 12 and 27 are illustrative synthetic IDs; they are not claims about
production vocabulary positions. Physical point/box matrices are reused from
``bounded_multi_box_fixture`` instead of duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass

from town_model.models.bound_box_inference import ClassBinding

SYNTHETIC_CLASS_VOCABULARY_BYTES = (
    b'{"classes":[{"class_id":12,"trainable":true},'
    b'{"class_id":27,"trainable":true}],"trainable_class_ids":[12,27]}\n'
)
SYNTHETIC_CLASS_VOCABULARY_SHA256 = (
    "fd793ac6c2c061c851f6f4dcf15fc6b417e5c7b4d554b569f71245587d174b8b"
)
SYNTHETIC_CLASS_BINDING = ClassBinding.from_synthetic_vocabulary_bytes(
    SYNTHETIC_CLASS_VOCABULARY_BYTES
)


@dataclass(frozen=True, slots=True)
class BoundBoxInferenceInspectionFixture:
    """Synthetic row association and independent fixed DomainThing oracle."""

    binding: ClassBinding
    row_labels: tuple[str, ...]
    row_class_ids: tuple[int, ...]
    expected_domain_contains: tuple[bool, ...]


BOUND_BOX_INFERENCE_INSPECTION_FIXTURE = BoundBoxInferenceInspectionFixture(
    binding=SYNTHETIC_CLASS_BINDING,
    row_labels=("Box 0", "Box 1"),
    row_class_ids=(12, 27),
    expected_domain_contains=(True, True, True, True, True, False),
)
