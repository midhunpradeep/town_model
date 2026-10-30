from __future__ import annotations

from pathlib import Path

from town_model.application import load_explorer
from town_model.dashboard.app import target_state_rows


def test_dataset_explorer_exposes_real_target_masks_and_semantic_states():
    root = Path(__file__).parents[1] / "examples" / "pizza-48"
    explorer = load_explorer(root)
    assert explorer.summary.integrity_status == "current_pass"
    record = explorer.records[0]
    rows = target_state_rows(record.pizza_targets, record.pizza_known_mask, explorer.classes)
    assert len(rows) == len(explorer.classes)
    by_name = {row["class"]: row for row in rows}
    assert by_name["DomainThing"]["status"] == "fixed positive"
    assert by_name["DomainThing"]["target_state"] == "positive"
    assert by_name["Hot"]["status"] == "unsupported"
    assert by_name["Hot"]["target_state"] == "unknown"
    component = record.components[0]
    component_rows = target_state_rows(component.ontology_targets, component.ontology_known_mask, explorer.classes)
    assert any(row["target_state"] == "known negative" for row in component_rows)
    assert any(row["target_state"] == "positive" for row in component_rows)
