"""Standalone, fixed-fixture, read-only TOWN-01 mathematical inspector."""

from __future__ import annotations

import pandas as pd
import torch

from town_model.models.town_geometry import (
    box_strict_interior,
    house_contains,
    house_signed_clearance,
    town_contains,
    town_signed_clearance,
)
from town_model.models.town_composition_fixture import (
    TOWN_COMPOSITION_FIXTURES,
    TownCompositionFixture,
)
from town_model.models.town_geometry_fixture import TOWN_STRICT_INTERIOR_FIXTURE
from town_model.models.town_clearance_fixture import (
    TOWN_CLEARANCE_INSPECTION_FIXTURES,
    TownClearanceFixture,
)


def _inspection_table() -> tuple[pd.DataFrame, tuple[tuple[str, bool, bool], ...]]:
    """Return literal expectations beside values supplied by the evaluator."""
    fixture = TOWN_STRICT_INTERIOR_FIXTURE
    points = torch.tensor(fixture.points, dtype=torch.float32, device="cpu")
    lower = torch.tensor(fixture.lower, dtype=torch.float32, device="cpu")
    upper = torch.tensor(fixture.upper, dtype=torch.float32, device="cpu")
    actual_values = box_strict_interior(points, lower, upper).tolist()

    rows = [
        {
            "Point": label,
            "Coordinates": coordinates,
            "Expected membership": expected,
            "Actual membership": bool(actual),
            "Agreement": expected == bool(actual),
        }
        for label, coordinates, expected, actual in zip(
            fixture.labels,
            fixture.points,
            fixture.expected_contains,
            actual_values,
            strict=True,
        )
    ]
    mismatches = tuple(
        (label, bool(expected), bool(actual))
        for label, expected, actual in zip(
            fixture.labels, fixture.expected_contains, actual_values, strict=True
        )
        if bool(expected) != bool(actual)
    )
    return pd.DataFrame(rows), mismatches


def _render_inspector(st) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fixture = TOWN_STRICT_INTERIOR_FIXTURE
    frame, mismatches = _inspection_table()

    figure, axis = plt.subplots(figsize=(6, 6))
    domain_patch = Rectangle(
        (-1.0, -1.0),
        2.0,
        2.0,
        fill=False,
        linestyle="--",
        edgecolor="black",
        linewidth=1.5,
    )
    box_patch = Rectangle(
        fixture.lower,
        fixture.upper[0] - fixture.lower[0],
        fixture.upper[1] - fixture.lower[1],
        fill=False,
        edgecolor="tab:orange",
        linewidth=2,
    )
    axis.add_patch(domain_patch)
    axis.add_patch(box_patch)
    axis.scatter(
        [point[0] for point in fixture.points],
        [point[1] for point in fixture.points],
        color="tab:blue",
        zorder=3,
    )
    label_offsets = {
        "A": (5, 5),
        "B": (5, 5),
        "C": (5, 5),
        "D": (5, -10),
        "E": (5, 5),
        "F": (5, -10),
        "G": (5, 5),
        "H": (5, 5),
        "I": (5, 5),
    }
    for label, (x, y) in zip(fixture.labels, fixture.points, strict=True):
        axis.annotate(label, (x, y), xytext=label_offsets[label], textcoords="offset points")
    axis.set_xlim(-1.4, 1.15)
    axis.set_ylim(-1.15, 1.15)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x₁")
    axis.set_ylabel("x₂")
    axis.legend(handles=[domain_patch, box_patch], labels=["Bounded domain Δ", "Physical box"])

    st.title("TOWN-01 · Strict-Box Interior")
    st.subheader("Mathematical Inspector")
    st.markdown(r"`I_B(x) = ∧ⱼ (lⱼ < xⱼ < rⱼ)` — ambient strict interior; every face is excluded.")
    st.caption(
        f"Domain Δ = [-1, 1]² · Fixed closed box B = "
        f"[{fixture.lower[0]}, {fixture.upper[0]}] × "
        f"[{fixture.lower[1]}, {fixture.upper[1]}]"
    )
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
    st.dataframe(frame, hide_index=True, width="stretch")

    if mismatches:
        details = "; ".join(
            f"{label}: expected {expected}, actual {actual}"
            for label, expected, actual in mismatches
        )
        st.error(
            f"Membership mismatches: {len(mismatches)} of {len(fixture.labels)} · {details}"
        )
    else:
        st.success(
            f"Membership mismatches: 0 of {len(fixture.labels)} · "
            f"fixture agreement: {len(fixture.labels)}/{len(fixture.labels)}"
        )

    _render_composition_inspector(st)
    _render_clearance_inspector(st)


def _composition_inspection_tables() -> tuple[
    tuple[
        TownCompositionFixture,
        pd.DataFrame,
        tuple[tuple[str, str, bool, bool], ...],
        tuple[bool, ...],
    ],
    ...,
]:
    """Pair each immutable literal expectation with public evaluator output."""
    inspections = []
    for fixture in TOWN_COMPOSITION_FIXTURES:
        points = torch.tensor(fixture.points, dtype=torch.float32, device="cpu")
        frame_lower = torch.tensor(fixture.frame_lower, dtype=torch.float32, device="cpu")
        frame_upper = torch.tensor(fixture.frame_upper, dtype=torch.float32, device="cpu")
        house_count = len(fixture.frame_lower)
        dimensions = len(fixture.frame_lower[0])
        window_count = len(fixture.window_lower[0])
        if window_count:
            window_lower = torch.tensor(fixture.window_lower, dtype=torch.float32, device="cpu")
            window_upper = torch.tensor(fixture.window_upper, dtype=torch.float32, device="cpu")
        else:
            window_lower = torch.empty((house_count, 0, dimensions), dtype=torch.float32)
            window_upper = torch.empty_like(window_lower)

        if house_count == 1:
            actual = house_contains(
                points, frame_lower[0], frame_upper[0], window_lower[0], window_upper[0]
            )
        else:
            actual = town_contains(
                points, frame_lower, frame_upper, window_lower, window_upper
            )
        actual_values = tuple(bool(value) for value in actual.tolist())
        rows = [
            {
                "Scenario": fixture.name,
                "Point": label,
                "Coordinates": coordinates,
                "Expected membership": bool(expected),
                "Actual membership": actual_value,
                "Agreement": bool(expected) == actual_value,
            }
            for label, coordinates, expected, actual_value in zip(
                fixture.labels,
                fixture.points,
                fixture.expected_contains,
                actual_values,
                strict=True,
            )
        ]
        mismatches = tuple(
            (fixture.name, label, bool(expected), actual_value)
            for label, expected, actual_value in zip(
                fixture.labels, fixture.expected_contains, actual_values, strict=True
            )
            if bool(expected) != actual_value
        )
        inspections.append((fixture, pd.DataFrame(rows), mismatches, actual_values))
    return tuple(inspections)


def _format_frame(lower: tuple[float, ...], upper: tuple[float, ...]) -> str:
    return " × ".join(
        f"[{lower[axis]}, {upper[axis]}]" for axis in range(len(lower))
    )


def _render_composition_inspector(st) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    st.header("TOWN-02 · Boolean House and Town Composition")
    st.markdown(
        r"`H = F \setminus \bigcup_w \operatorname{int}_{\mathbb{R}^d}(W_w)`; "
        r"`T = \bigcup_h H_h`. Closed frame faces are included; window boundaries "
        "are retained unless another strict window interior excludes the point."
    )

    frame_colors = ("tab:blue", "tab:purple", "tab:cyan")
    for fixture, table, mismatches, actual_values in _composition_inspection_tables():
        house_count = len(fixture.frame_lower)
        window_count = len(fixture.window_lower[0])
        frames = "; ".join(
            f"F{index + 1}={_format_frame(lower, upper)}"
            for index, (lower, upper) in enumerate(
                zip(fixture.frame_lower, fixture.frame_upper, strict=True)
            )
        )
        if window_count:
            windows = "; ".join(
                f"H{house + 1}: "
                + ", ".join(
                    _format_frame(lower, upper)
                    for lower, upper in zip(
                        fixture.window_lower[house], fixture.window_upper[house], strict=True
                    )
                )
                for house in range(house_count)
            )
        else:
            windows = "none (W=0)"

        st.subheader(fixture.name)
        st.caption(f"Closed frames: {frames} · Exclusion windows: {windows}")

        figure, axis = plt.subplots(figsize=(5, 4.5))
        axis.add_patch(
            Rectangle(
                (-1.0, -1.0),
                2.0,
                2.0,
                fill=False,
                linestyle="--",
                edgecolor="black",
                linewidth=1.2,
                zorder=0,
            )
        )
        for house in range(house_count):
            for window_lower, window_upper in zip(
                fixture.window_lower[house], fixture.window_upper[house], strict=True
            ):
                axis.add_patch(
                    Rectangle(
                        window_lower,
                        window_upper[0] - window_lower[0],
                        window_upper[1] - window_lower[1],
                        facecolor="tab:red",
                        alpha=0.16,
                        hatch="////",
                        edgecolor="tab:red",
                        linewidth=1.6,
                        zorder=1,
                    )
                )
        for house, (lower, upper) in enumerate(
            zip(fixture.frame_lower, fixture.frame_upper, strict=True)
        ):
            axis.add_patch(
                Rectangle(
                    lower,
                    upper[0] - lower[0],
                    upper[1] - lower[1],
                    fill=False,
                    edgecolor=frame_colors[house % len(frame_colors)],
                    linewidth=2,
                    zorder=2,
                )
            )
        axis.scatter(
            [point[0] for point in fixture.points],
            [point[1] for point in fixture.points],
            c=["tab:green" if actual else "tab:orange" for actual in actual_values],
            edgecolors="black",
            zorder=3,
        )
        for label, point in zip(fixture.labels, fixture.points, strict=True):
            axis.annotate(label, point, xytext=(5, 5), textcoords="offset points")
        axis.set_xlim(-1.35, 1.35)
        axis.set_ylim(-1.25, 1.25)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x₁")
        axis.set_ylabel("x₂")
        frame_handles = tuple(
            Patch(
                fill=False,
                edgecolor=frame_colors[house % len(frame_colors)],
                label=f"House {house + 1} closed frame; boundary included",
            )
            for house in range(house_count)
        )
        axis.legend(
            handles=(
                Patch(
                    fill=False,
                    linestyle="--",
                    edgecolor="black",
                    label="Bounded domain Δ = [-1, 1]²",
                ),
                *frame_handles,
                Patch(
                    facecolor="tab:red",
                    alpha=0.16,
                    hatch="////",
                    edgecolor="tab:red",
                    label="Strict window interior excluded; outline retained",
                ),
                Patch(facecolor="tab:green", label="Actual member"),
                Patch(facecolor="tab:orange", label="Actual nonmember"),
            ),
            loc="upper right",
            fontsize="small",
        )
        st.pyplot(figure, clear_figure=True)
        plt.close(figure)
        st.table(table)

        if mismatches:
            details = "; ".join(
                f"{scenario} · {label}: expected {expected}, actual {actual}"
                for scenario, label, expected, actual in mismatches
            )
            st.error(
                f"{fixture.name}: membership mismatches {len(mismatches)} of "
                f"{len(fixture.labels)} · {details}"
            )
        else:
            st.success(
                f"{fixture.name}: membership mismatches 0 of {len(fixture.labels)} · "
                f"fixture agreement {len(fixture.labels)}/{len(fixture.labels)}"
            )


def _clearance_inspection_tables() -> tuple[
    tuple[TownClearanceFixture, pd.DataFrame, tuple[tuple[str, str, bool, bool, float, float], ...]],
    ...,
]:
    """Show independently evaluated Boolean membership and signed clearance."""
    inspections = []
    for fixture in TOWN_CLEARANCE_INSPECTION_FIXTURES:
        points = torch.tensor(fixture.points, dtype=torch.float32, device="cpu")
        frame_lower = torch.tensor(fixture.frame_lower, dtype=torch.float32, device="cpu")
        frame_upper = torch.tensor(fixture.frame_upper, dtype=torch.float32, device="cpu")
        house_count = len(fixture.frame_lower)
        dimensions = len(fixture.frame_lower[0])
        window_count = len(fixture.window_lower[0])
        if window_count:
            window_lower = torch.tensor(fixture.window_lower, dtype=torch.float32, device="cpu")
            window_upper = torch.tensor(fixture.window_upper, dtype=torch.float32, device="cpu")
        else:
            window_lower = torch.empty((house_count, 0, dimensions), dtype=torch.float32)
            window_upper = torch.empty_like(window_lower)

        if house_count == 1:
            actual_scores = house_signed_clearance(
                points, frame_lower[0], frame_upper[0], window_lower[0], window_upper[0]
            )
            actual_membership = house_contains(
                points, frame_lower[0], frame_upper[0], window_lower[0], window_upper[0]
            )
        else:
            actual_scores = town_signed_clearance(
                points, frame_lower, frame_upper, window_lower, window_upper
            )
            actual_membership = town_contains(
                points, frame_lower, frame_upper, window_lower, window_upper
            )

        scores = tuple(float(value) for value in actual_scores.tolist())
        members = tuple(bool(value) for value in actual_membership.tolist())
        rows = []
        mismatches = []
        for label, coordinates, expected_score, expected_member, score, member in zip(
            fixture.labels,
            fixture.points,
            fixture.expected_clearance,
            fixture.expected_membership,
            scores,
            members,
            strict=True,
        ):
            strict_positive = score > 0.0
            rows.append(
                {
                    "Scenario": fixture.name,
                    "Point": label,
                    "Coordinates": coordinates,
                    "Expected membership": expected_member,
                    "Actual membership": member,
                    "Expected clearance": expected_score,
                    "Actual clearance": score,
                    "Clearance ≥ 0 (diagnostic)": score >= 0.0,
                    "Clearance > 0 (strict-positive diagnostic)": strict_positive,
                    "Strict-positive differs from membership": member != strict_positive,
                    "Boolean agreement": member == expected_member,
                    "Clearance agreement": score == expected_score,
                }
            )
            if member != expected_member or score != expected_score:
                mismatches.append(
                    (fixture.name, label, expected_member, member, expected_score, score)
                )
        inspections.append((fixture, pd.DataFrame(rows), tuple(mismatches)))
    return tuple(inspections)


def _render_clearance_inspector(st) -> None:
    st.header("TOWN-03 · Signed Clearance and Exact Membership")
    st.markdown(
        "Clearance and Boolean membership are evaluated independently. The approved "
        "equations make nonnegative clearance align with membership for valid geometry; "
        "the `clearance > 0` column is only a strict-positive boundary diagnostic and "
        "is never used to produce the Boolean output."
    )
    for fixture, table, mismatches in _clearance_inspection_tables():
        st.subheader(fixture.name)
        st.caption(
            f"H={len(fixture.frame_lower)} · W={len(fixture.window_lower[0])} "
            f"· d={len(fixture.frame_lower[0])} · fixed CPU fixture · read-only"
        )
        st.table(table)
        if mismatches:
            details = "; ".join(
                f"{scenario} · {label}: membership expected {expected_member}, "
                f"actual {actual_member}; clearance expected {expected_score}, "
                f"actual {actual_score}"
                for scenario, label, expected_member, actual_member, expected_score, actual_score
                in mismatches
            )
            st.error(
                f"{fixture.name}: Boolean/clearance mismatches {len(mismatches)} "
                f"of {len(fixture.labels)} · {details}"
            )
        else:
            zero_members = sum(
                score == 0.0 and member
                for score, member in zip(
                    table["Actual clearance"], table["Actual membership"], strict=True
                )
            )
            strict_positive_disagreements = int(
                table["Strict-positive differs from membership"].sum()
            )
            st.success(
                f"{fixture.name}: Boolean mismatches 0; clearance mismatches 0; "
                f"included zero-clearance points {zero_members}; strict-positive "
                f"boundary differences {strict_positive_disagreements}"
            )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="TOWN-01 Strict-Box Interior", layout="wide")
    _render_inspector(st)


if __name__ == "__main__":
    main()
