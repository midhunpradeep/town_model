"""Independent literal fixture for bounded multi-box inspection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoundedMultiBoxInspectionFixture:
    box_labels: tuple[str, ...]
    lower: tuple[tuple[float, float], ...]
    upper: tuple[tuple[float, float], ...]
    labels: tuple[str, ...]
    points: tuple[tuple[float, float], ...]
    expected_membership: tuple[tuple[bool, ...], ...]
    expected_clearances: tuple[tuple[float, ...], ...]


BOUNDED_MULTI_BOX_INSPECTION_FIXTURE = BoundedMultiBoxInspectionFixture(
    box_labels=("Box 0", "Box 1"),
    lower=((-1.0, -0.5), (-0.25, -1.0)),
    upper=((0.5, 1.0), (1.0, 0.5)),
    labels=("A", "B", "C", "D", "E", "F"),
    points=(
        (0.0, 0.0),
        (-0.75, 0.0),
        (0.75, 0.0),
        (-0.75, -0.75),
        (0.5, 0.5),
        (-1.25, 0.0),
    ),
    expected_membership=(
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (True, True),
        (False, False),
    ),
    expected_clearances=(
        (0.50, 0.25),
        (0.25, -0.50),
        (-0.25, 0.25),
        (-0.25, -0.50),
        (0.0, 0.0),
        (-0.25, -1.0),
    ),
)
