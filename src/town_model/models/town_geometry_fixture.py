"""Immutable literal fixture for the TOWN-01 strict-interior inspector."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TownStrictInteriorFixture:
    lower: tuple[float, float]
    upper: tuple[float, float]
    labels: tuple[str, ...]
    points: tuple[tuple[float, float], ...]
    expected_contains: tuple[bool, ...]


TOWN_STRICT_INTERIOR_FIXTURE = TownStrictInteriorFixture(
    lower=(-1.0, -0.25),
    upper=(0.5, 0.25),
    labels=("A", "B", "C", "D", "E", "F", "G", "H", "I"),
    points=(
        (0.0, 0.0),
        (-1.0, 0.0),
        (0.5, 0.0),
        (0.0, -0.25),
        (0.0, 0.25),
        (-1.0, -0.25),
        (-0.5, 0.125),
        (0.75, 0.0),
        (-1.25, 0.0),
    ),
    expected_contains=(True, False, False, False, False, False, True, False, False),
)
