"""Standalone, fixed-fixture, read-only TOWN-01 mathematical inspector."""

from __future__ import annotations

import pandas as pd
import torch

from town_model.models.town_geometry import box_strict_interior
from town_model.models.town_geometry_fixture import TOWN_STRICT_INTERIOR_FIXTURE


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


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="TOWN-01 Strict-Box Interior", layout="wide")
    _render_inspector(st)


if __name__ == "__main__":
    main()
