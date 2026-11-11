"""The single literal point fixture shared by bounded-box tests and inspection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoundedBoxInspectionFixture:
    lower: tuple[float, float]
    upper: tuple[float, float]
    labels: tuple[str, ...]
    points: tuple[tuple[float, float], ...]
    expected_contains: tuple[bool, ...]
    expected_clearances: tuple[float, ...]


BOUNDED_BOX_INSPECTION_FIXTURE = BoundedBoxInspectionFixture(
    lower=(-1.0, -0.5),
    upper=(0.5, 1.0),
    labels=("A", "B", "C", "D", "E", "F"),
    points=(
        (0.0, 0.0),
        (-1.0, 0.0),
        (-1.0, 1.0),
        (0.5, 0.0),
        (0.75, 0.0),
        (-1.25, 0.0),
    ),
    expected_contains=(True, True, True, True, False, False),
    expected_clearances=(0.50, 0.00, 0.00, 0.00, -0.25, -0.25),
)
