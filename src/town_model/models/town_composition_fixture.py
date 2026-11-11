"""Immutable hand-specified fixtures for Boolean house and town composition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TownCompositionFixture:
    name: str
    frame_lower: tuple[tuple[float, ...], ...]
    frame_upper: tuple[tuple[float, ...], ...]
    window_lower: tuple[tuple[tuple[float, ...], ...], ...]
    window_upper: tuple[tuple[tuple[float, ...], ...], ...]
    labels: tuple[str, ...]
    points: tuple[tuple[float, ...], ...]
    expected_contains: tuple[bool, ...]


TOWN_COMPOSITION_FIXTURES = (
    TownCompositionFixture(
        name="F1 single window",
        frame_lower=((-1.0, -1.0),),
        frame_upper=((1.0, 1.0),),
        window_lower=(((-0.5, -0.5),),),
        window_upper=(((0.5, 0.5),),),
        labels=("A", "B", "C", "D", "E"),
        points=((0.0, 0.0), (0.5, 0.0), (-0.5, -0.5), (1.0, 0.0), (1.25, 0.0)),
        expected_contains=(False, True, True, True, False),
    ),
    TownCompositionFixture(
        name="F2 overlapping windows",
        frame_lower=((-1.0, -1.0),),
        frame_upper=((1.0, 1.0),),
        window_lower=(((-0.5, -0.5), (0.25, -0.25)),),
        window_upper=(((0.5, 0.5), (0.75, 0.25)),),
        labels=("A", "B", "C", "D", "E", "F", "G", "H"),
        points=(
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 0.25),
            (-0.5, 0.0),
            (0.7, 0.0),
            (0.75, 0.0),
            (1.0, 0.0),
            (1.25, 0.0),
        ),
        expected_contains=(False, False, True, True, False, True, True, False),
    ),
    TownCompositionFixture(
        name="F3 protruding window",
        frame_lower=((-0.75, -0.75),),
        frame_upper=((0.75, 0.75),),
        window_lower=(((-0.5, -0.25),),),
        window_upper=(((1.0, 0.25),),),
        labels=("A", "B", "C", "D", "E", "F"),
        points=((0.0, 0.0), (0.7, 0.0), (-0.5, 0.0), (0.7, 0.4), (0.9, 0.0), (0.75, 0.25)),
        expected_contains=(False, False, True, True, False, True),
    ),
    TownCompositionFixture(
        name="F4 zero windows",
        frame_lower=((-0.5, -0.5),),
        frame_upper=((0.5, 0.5),),
        window_lower=((),),
        window_upper=((),),
        labels=("A", "B", "C", "D"),
        points=((0.0, 0.0), (0.5, 0.5), (0.5001, 0.0), (1.25, 0.0)),
        expected_contains=(True, True, False, False),
    ),
    TownCompositionFixture(
        name="F5 two-house town",
        frame_lower=((-1.0, -1.0), (0.0, -1.0)),
        frame_upper=((0.5, 1.0), (1.0, 1.0)),
        window_lower=(((-0.5, -0.5),), ((0.75, -0.25),)),
        window_upper=(((0.25, 0.5),), ((1.0, 0.25),)),
        labels=("A", "B", "C", "D", "E", "F"),
        points=((0.0, 0.0), (-0.25, 0.0), (0.9, 0.0), (1.0, 0.0), (-0.75, 0.0), (1.25, 0.0)),
        expected_contains=(True, False, False, True, True, False),
    ),
)


DEGENERATE_WINDOW_FIXTURES = (
    TownCompositionFixture(
        name="degenerate window on one axis",
        frame_lower=((-1.0, -1.0),),
        frame_upper=((1.0, 1.0),),
        window_lower=(((0.0, -0.75),),),
        window_upper=(((0.0, 0.75),),),
        labels=("A", "B", "C", "D"),
        points=((0.0, 0.0), (0.0, 0.75), (0.25, 0.0), (-1.0, 0.0)),
        expected_contains=(True, True, True, True),
    ),
    TownCompositionFixture(
        name="fully collapsed window",
        frame_lower=((-1.0, -1.0),),
        frame_upper=((1.0, 1.0),),
        window_lower=(((0.0, 0.0),),),
        window_upper=(((0.0, 0.0),),),
        labels=("A", "B", "C"),
        points=((0.0, 0.0), (0.25, 0.25), (1.0, 1.0)),
        expected_contains=(True, True, True),
    ),
)


DEGENERATE_FRAME_FIXTURE = TownCompositionFixture(
    name="lower-dimensional closed frame",
    frame_lower=((0.0, -0.5),),
    frame_upper=((0.0, 0.5),),
    window_lower=((),),
    window_upper=((),),
    labels=("A", "B", "C", "D", "E"),
    points=((0.0, 0.0), (0.0, 0.5), (0.0001, 0.0), (-0.0001, 0.0), (0.0, -0.5)),
    expected_contains=(True, True, False, False, True),
)
