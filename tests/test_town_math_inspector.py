from __future__ import annotations

from pathlib import Path

import torch
from streamlit.testing.v1 import AppTest

from town_model.dashboard import town_math_inspector
from town_model.models import town_geometry
from town_model.models.town_composition_fixture import TOWN_COMPOSITION_FIXTURES
from town_model.models.town_clearance_fixture import TOWN_CLEARANCE_INSPECTION_FIXTURES
from town_model.models.town_geometry_fixture import TOWN_STRICT_INTERIOR_FIXTURE


def test_inspector_table_uses_cpu_evaluator_and_independent_literal_fixture(monkeypatch) -> None:
    calls: list[tuple[torch.device, torch.device, torch.device]] = []

    def evaluator(
        points: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
    ) -> torch.Tensor:
        calls.append((points.device, lower.device, upper.device))
        return torch.tensor(
            [True, False, False, False, False, False, True, False, False],
            dtype=torch.bool,
            device=points.device,
        )

    monkeypatch.setattr(town_math_inspector, "box_strict_interior", evaluator)

    frame, mismatches = town_math_inspector._inspection_table()

    assert calls == [(torch.device("cpu"),) * 3]
    assert frame.columns.tolist() == [
        "Point", "Coordinates", "Expected membership", "Actual membership", "Agreement"
    ]
    assert frame["Point"].tolist() == list(TOWN_STRICT_INTERIOR_FIXTURE.labels)
    assert frame["Coordinates"].tolist() == list(TOWN_STRICT_INTERIOR_FIXTURE.points)
    assert frame["Expected membership"].tolist() == list(
        TOWN_STRICT_INTERIOR_FIXTURE.expected_contains
    )
    assert frame["Actual membership"].tolist() == list(
        TOWN_STRICT_INTERIOR_FIXTURE.expected_contains
    )
    assert frame["Agreement"].tolist() == [True] * 9
    assert mismatches == ()


def test_standalone_inspector_renders_fixed_fixture_and_is_read_only() -> None:
    app_path = Path(town_math_inspector.__file__)
    root = app_path.parents[3]
    artifacts = root / "artifacts"
    before_exists = artifacts.exists()
    before_files = {
        path.relative_to(artifacts): path.read_bytes()
        for path in artifacts.rglob("*")
        if path.is_file()
    } if before_exists else {}

    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app.exception
    assert any(item.value == "TOWN-01 · Strict-Box Interior" for item in app.title)
    assert any(item.value == "Mathematical Inspector" for item in app.subheader)
    assert any(
        "Fixed closed box B = [-1.0, 0.5] × [-0.25, 0.25]" in item.value
        for item in app.caption
    )
    assert len(app.dataframe) == 1
    table = app.dataframe[0].value
    assert table["Point"].tolist() == list(TOWN_STRICT_INTERIOR_FIXTURE.labels)
    assert table["Expected membership"].tolist() == list(
        TOWN_STRICT_INTERIOR_FIXTURE.expected_contains
    )
    assert table["Actual membership"].tolist() == list(
        TOWN_STRICT_INTERIOR_FIXTURE.expected_contains
    )
    assert table["Agreement"].tolist() == [True] * 9
    assert any("Membership mismatches: 0" in item.value for item in app.success)
    assert not app.error
    assert not app.button
    assert not app.slider
    assert not app.selectbox
    assert not app.number_input
    assert not app.text_input
    assert not app.checkbox

    after_exists = artifacts.exists()
    after_files = {
        path.relative_to(artifacts): path.read_bytes()
        for path in artifacts.rglob("*")
        if path.is_file()
    } if after_exists else {}
    assert after_exists == before_exists
    assert after_files == before_files


def test_composition_tables_use_cpu_evaluators_and_independent_literals(monkeypatch) -> None:
    house_calls: list[tuple[torch.device, ...]] = []
    town_calls: list[tuple[torch.device, ...]] = []
    house_evaluator = town_geometry.house_contains
    town_evaluator = town_geometry.town_contains

    def house_spy(
        points: torch.Tensor,
        frame_lower: torch.Tensor,
        frame_upper: torch.Tensor,
        window_lower: torch.Tensor,
        window_upper: torch.Tensor,
    ) -> torch.Tensor:
        house_calls.append(
            (points.device, frame_lower.device, frame_upper.device, window_lower.device, window_upper.device)
        )
        return house_evaluator(points, frame_lower, frame_upper, window_lower, window_upper)

    def town_spy(
        points: torch.Tensor,
        frame_lower: torch.Tensor,
        frame_upper: torch.Tensor,
        window_lower: torch.Tensor,
        window_upper: torch.Tensor,
    ) -> torch.Tensor:
        town_calls.append(
            (points.device, frame_lower.device, frame_upper.device, window_lower.device, window_upper.device)
        )
        return town_evaluator(points, frame_lower, frame_upper, window_lower, window_upper)

    monkeypatch.setattr(town_math_inspector, "house_contains", house_spy)
    monkeypatch.setattr(town_math_inspector, "town_contains", town_spy)

    inspections = town_math_inspector._composition_inspection_tables()

    assert len(house_calls) == 4 and all(devices == (torch.device("cpu"),) * 5 for devices in house_calls)
    assert town_calls == [(torch.device("cpu"),) * 5]
    assert [inspection[0].name for inspection in inspections] == [
        fixture.name for fixture in TOWN_COMPOSITION_FIXTURES
    ]
    for fixture, table, mismatches, actual in inspections:
        assert table["Scenario"].tolist() == [fixture.name] * len(fixture.labels)
        assert table["Point"].tolist() == list(fixture.labels)
        assert table["Expected membership"].tolist() == list(fixture.expected_contains)
        assert table["Actual membership"].tolist() == list(fixture.expected_contains)
        assert table["Agreement"].tolist() == [True] * len(fixture.labels)
        assert mismatches == ()
        assert actual == fixture.expected_contains


def test_composition_inspector_renders_five_scenarios_without_changing_town01_table() -> None:
    app_path = Path(town_math_inspector.__file__)
    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app.exception
    assert any(item.value == "TOWN-01 · Strict-Box Interior" for item in app.title)
    assert any(item.value == "TOWN-02 · Boolean House and Town Composition" for item in app.header)
    assert [item.value for item in app.subheader[1:6]] == [
        fixture.name for fixture in TOWN_COMPOSITION_FIXTURES
    ]
    assert len(app.dataframe) == 1
    assert len(app.table) == 10
    for fixture, element in zip(TOWN_COMPOSITION_FIXTURES, app.table[:5], strict=True):
        table = element.value
        assert table["Scenario"].tolist() == [fixture.name] * len(fixture.labels)
        assert table["Point"].tolist() == list(fixture.labels)
        assert [tuple(point) for point in table["Coordinates"].tolist()] == list(fixture.points)
        assert table["Expected membership"].tolist() == list(fixture.expected_contains)
        assert table["Actual membership"].tolist() == list(fixture.expected_contains)
        assert table["Agreement"].tolist() == [True] * len(fixture.labels)
    captions = [item.value for item in app.caption]
    assert any("Closed frames:" in value and "Exclusion windows:" in value for value in captions)
    assert any("none (W=0)" in value for value in captions)
    assert not app.error
    assert not app.button and not app.slider and not app.selectbox
    assert not app.number_input and not app.text_input and not app.checkbox


def test_composition_wrong_evaluator_identifies_each_scenario_and_point(monkeypatch) -> None:
    def deliberately_wrong(
        points: torch.Tensor,
        frame_lower: torch.Tensor,
        frame_upper: torch.Tensor,
        window_lower: torch.Tensor,
        window_upper: torch.Tensor,
    ) -> torch.Tensor:
        del frame_lower, frame_upper, window_lower, window_upper
        return torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)

    monkeypatch.setattr(town_geometry, "house_contains", deliberately_wrong)
    monkeypatch.setattr(town_geometry, "town_contains", deliberately_wrong)
    app_path = Path(town_math_inspector.__file__)
    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app.exception
    errors = [item.value for item in app.error]
    assert len(errors) == 10
    joined = "\n".join(errors)
    assert "F1 single window: membership mismatches 3 of 5" in joined
    assert "F1 single window · B: expected True, actual False" in joined
    assert "F2 overlapping windows · C: expected True, actual False" in joined
    assert "F3 protruding window · F: expected True, actual False" in joined
    assert "F4 zero windows · A: expected True, actual False" in joined
    assert "F5 two-house town · A: expected True, actual False" in joined
    assert "C1 frame/window boundaries and hole interior: Boolean/clearance mismatches 3 of 4" in joined
    assert "C2 one-axis-degenerate window: Boolean/clearance mismatches 3 of 3" in joined
    for fixture, element in zip(TOWN_COMPOSITION_FIXTURES, app.table[:5], strict=True):
        assert element.value["Actual membership"].tolist() == [False] * len(element.value)
        assert element.value["Agreement"].tolist() == [
            not expected for expected in fixture.expected_contains
        ]


def test_wrong_evaluator_injection_shows_point_specific_mismatches(monkeypatch) -> None:
    def deliberately_wrong(
        points: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
    ) -> torch.Tensor:
        del lower, upper
        return torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)

    monkeypatch.setattr(town_geometry, "box_strict_interior", deliberately_wrong)
    app_path = Path(town_math_inspector.__file__)
    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app.exception
    table = app.dataframe[0].value
    assert table["Actual membership"].tolist() == [False] * 9
    assert table["Agreement"].tolist() == [False, True, True, True, True, True, False, True, True]
    messages = [item.value for item in app.error]
    assert len(messages) == 1
    assert "A: expected True, actual False" in messages[0]
    assert "G: expected True, actual False" in messages[0]
    assert "Membership mismatches: 2 of 9" in messages[0]


def test_clearance_tables_use_cpu_evaluators_and_independent_literal_fixtures(monkeypatch) -> None:
    devices: list[tuple[str, tuple[torch.device, ...]]] = []
    house_score = town_geometry.house_signed_clearance
    town_score = town_geometry.town_signed_clearance
    house_member = town_geometry.house_contains
    town_member = town_geometry.town_contains

    def house_score_spy(points, frame_lower, frame_upper, window_lower, window_upper):
        devices.append(("house_score", tuple(t.device for t in (points, frame_lower, frame_upper, window_lower, window_upper))))
        return house_score(points, frame_lower, frame_upper, window_lower, window_upper)

    def town_score_spy(points, frame_lower, frame_upper, window_lower, window_upper):
        devices.append(("town_score", tuple(t.device for t in (points, frame_lower, frame_upper, window_lower, window_upper))))
        return town_score(points, frame_lower, frame_upper, window_lower, window_upper)

    def house_member_spy(points, frame_lower, frame_upper, window_lower, window_upper):
        devices.append(("house_member", tuple(t.device for t in (points, frame_lower, frame_upper, window_lower, window_upper))))
        return house_member(points, frame_lower, frame_upper, window_lower, window_upper)

    def town_member_spy(points, frame_lower, frame_upper, window_lower, window_upper):
        devices.append(("town_member", tuple(t.device for t in (points, frame_lower, frame_upper, window_lower, window_upper))))
        return town_member(points, frame_lower, frame_upper, window_lower, window_upper)

    monkeypatch.setattr(town_math_inspector, "house_signed_clearance", house_score_spy)
    monkeypatch.setattr(town_math_inspector, "town_signed_clearance", town_score_spy)
    monkeypatch.setattr(town_math_inspector, "house_contains", house_member_spy)
    monkeypatch.setattr(town_math_inspector, "town_contains", town_member_spy)

    inspections = town_math_inspector._clearance_inspection_tables()

    assert [entry[0].name for entry in inspections] == [
        fixture.name for fixture in TOWN_CLEARANCE_INSPECTION_FIXTURES
    ]
    assert [kind for kind, _ in devices].count("house_score") == 4
    assert [kind for kind, _ in devices].count("house_member") == 4
    assert [kind for kind, _ in devices].count("town_score") == 1
    assert [kind for kind, _ in devices].count("town_member") == 1
    assert all(device_tuple == (torch.device("cpu"),) * 5 for _, device_tuple in devices)

    for fixture, table, mismatches in inspections:
        assert table["Scenario"].tolist() == [fixture.name] * len(fixture.labels)
        assert table["Point"].tolist() == list(fixture.labels)
        assert table["Expected membership"].tolist() == list(fixture.expected_membership)
        assert table["Actual membership"].tolist() == list(fixture.expected_membership)
        assert table["Expected clearance"].tolist() == list(fixture.expected_clearance)
        assert table["Actual clearance"].tolist() == list(fixture.expected_clearance)
        assert table["Boolean agreement"].tolist() == [True] * len(fixture.labels)
        assert table["Clearance agreement"].tolist() == [True] * len(fixture.labels)
        assert mismatches == ()


def test_clearance_inspector_renders_boundaries_and_degeneracies_read_only() -> None:
    app_path = Path(town_math_inspector.__file__)
    root = app_path.parents[3]
    artifacts = root / "artifacts"
    before_exists = artifacts.exists()
    before_files = {
        path.relative_to(artifacts): path.read_bytes()
        for path in artifacts.rglob("*")
        if path.is_file()
    } if before_exists else {}

    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app.exception
    assert any(item.value == "TOWN-03 · Signed Clearance and Exact Membership" for item in app.header)
    assert [item.value for item in app.subheader[-5:]] == [
        fixture.name for fixture in TOWN_CLEARANCE_INSPECTION_FIXTURES
    ]
    assert len(app.table) == 10
    for fixture, element in zip(TOWN_CLEARANCE_INSPECTION_FIXTURES, app.table[5:], strict=True):
        table = element.value
        assert table["Point"].tolist() == list(fixture.labels)
        assert table["Expected membership"].tolist() == list(fixture.expected_membership)
        assert table["Actual membership"].tolist() == list(fixture.expected_membership)
        assert table["Expected clearance"].tolist() == list(fixture.expected_clearance)
        assert table["Actual clearance"].tolist() == list(fixture.expected_clearance)
        assert table["Clearance ≥ 0 (diagnostic)"].tolist() == [value >= 0 for value in fixture.expected_clearance]
        assert table["Strict-positive differs from membership"].tolist() == [
            member != (value > 0) for member, value in zip(fixture.expected_membership, fixture.expected_clearance, strict=True)
        ]
    assert any("clearance > 0` column is only a strict-positive boundary diagnostic" in item.value for item in app.markdown)
    assert any("included zero-clearance points" in item.value for item in app.success[-5:])
    assert not app.error
    assert not app.button and not app.slider and not app.selectbox
    assert not app.number_input and not app.text_input and not app.checkbox

    after_exists = artifacts.exists()
    after_files = {
        path.relative_to(artifacts): path.read_bytes()
        for path in artifacts.rglob("*")
        if path.is_file()
    } if after_exists else {}
    assert after_exists == before_exists
    assert after_files == before_files


def test_clearance_inspector_reports_literal_score_mismatches(monkeypatch) -> None:
    def wrong_house_score(points, frame_lower, frame_upper, window_lower, window_upper):
        del frame_lower, frame_upper, window_lower, window_upper
        return torch.full((points.shape[0],), 0.125, dtype=torch.float32, device=points.device)

    def wrong_town_score(points, frame_lower, frame_upper, window_lower, window_upper):
        del frame_lower, frame_upper, window_lower, window_upper
        return torch.full((points.shape[0],), 0.125, dtype=torch.float32, device=points.device)

    monkeypatch.setattr(town_geometry, "house_signed_clearance", wrong_house_score)
    monkeypatch.setattr(town_geometry, "town_signed_clearance", wrong_town_score)
    app_path = Path(town_math_inspector.__file__)
    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app.exception
    clearance_errors = [
        item.value for item in app.error
        if "Boolean/clearance mismatches" in item.value
    ]
    assert len(clearance_errors) == len(TOWN_CLEARANCE_INSPECTION_FIXTURES)
    joined = "\n".join(clearance_errors)
    assert "C1 frame/window boundaries and hole interior" in joined
    assert "A: membership expected True, actual True; clearance expected 0.0, actual 0.125" in joined
    assert "C3 collapsed closed frame" in joined
