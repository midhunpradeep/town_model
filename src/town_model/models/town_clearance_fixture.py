"""Immutable literal inputs for the read-only TOWN-03 clearance inspector."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TownClearanceFixture:
    name: str
    frame_lower: tuple[tuple[float, ...], ...]
    frame_upper: tuple[tuple[float, ...], ...]
    window_lower: tuple[tuple[tuple[float, ...], ...], ...]
    window_upper: tuple[tuple[tuple[float, ...], ...], ...]
    labels: tuple[str, ...]
    points: tuple[tuple[float, ...], ...]
    expected_clearance: tuple[float, ...]
    expected_membership: tuple[bool, ...]


TOWN_CLEARANCE_INSPECTION_FIXTURES = (
    TownClearanceFixture(
        name="C1 frame/window boundaries and hole interior",
        frame_lower=((-1.0, -1.0),),
        frame_upper=((1.0, 1.0),),
        window_lower=(((-0.5, -0.5),),),
        window_upper=(((0.5, 0.5),),),
        labels=("A", "B", "C", "D"),
        points=((-1.0, 0.0), (0.5, 0.0), (0.0, 0.0), (0.75, 0.75)),
        expected_clearance=(0.0, 0.0, -0.5, 0.25),
        expected_membership=(True, True, False, True),
    ),
    TownClearanceFixture(
        name="C2 one-axis-degenerate window",
        frame_lower=((-1.0, -1.0),),
        frame_upper=((1.0, 1.0),),
        window_lower=(((0.0, -0.75),),),
        window_upper=(((0.0, 0.75),),),
        labels=("A", "B", "C"),
        points=((0.0, 0.0), (0.0, 0.75), (0.25, 0.0)),
        expected_clearance=(0.0, 0.0, 0.25),
        expected_membership=(True, True, True),
    ),
    TownClearanceFixture(
        name="C3 collapsed closed frame",
        frame_lower=((0.0, 0.0),),
        frame_upper=((0.0, 0.0),),
        window_lower=((),),
        window_upper=((),),
        labels=("A", "B", "C"),
        points=((0.0, 0.0), (0.0, 0.25), (0.0, -0.25)),
        expected_clearance=(0.0, -0.25, -0.25),
        expected_membership=(True, False, False),
    ),
    TownClearanceFixture(
        name="C4 two-house town boundary and holes",
        frame_lower=((-1.0,), (0.0,)),
        frame_upper=((0.0,), (1.0,)),
        window_lower=(((-0.5,),), ((0.0,),)),
        window_upper=(((0.0,),), ((0.5,),)),
        labels=("A", "B", "C"),
        points=((0.0,), (0.25,), (-0.25,)),
        expected_clearance=(0.0, -0.25, -0.25),
        expected_membership=(True, False, False),
    ),
    TownClearanceFixture(
        name="C5 overlapping windows and retained boundaries",
        frame_lower=((-1.0,),),
        frame_upper=((1.0,),),
        window_lower=(((-0.5,), (0.0,)),),
        window_upper=(((0.25,), (0.75,)),),
        labels=("A", "B", "C", "D", "E", "F", "G"),
        points=((-0.5,), (-0.25,), (0.0,), (0.25,), (0.5,), (0.75,), (1.0,)),
        expected_clearance=(0.0, -0.25, -0.25, -0.25, -0.25, 0.0, 0.0),
        expected_membership=(True, False, False, False, False, True, True),
    ),
)
