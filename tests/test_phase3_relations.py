from __future__ import annotations

import json

import numpy as np
import pytest

from types import SimpleNamespace

from town_model.dataset.ontology import OntologySnapshot
from town_model.inspection.relation_diagnostics import (
    box_region,
    diagnose_region_relation,
    evaluate_region_membership,
    save_relation_diagnostic,
    town_region,
)


def test_exact_box_relations_cover_equal_nested_partial_touch_and_disjoint_cases():
    equal_a = box_region(1, "A", [-0.5, -0.5], [0.5, 0.5])
    equal_b = box_region(2, "B", [-0.5, -0.5], [0.5, 0.5])
    assert diagnose_region_relation(equal_a, equal_b, "containment")["outcome"] is True
    assert diagnose_region_relation(equal_a, equal_b, "intersection")["outcome"] is True
    assert diagnose_region_relation(equal_a, equal_b, "disjointness")["outcome"] is False
    degenerate = box_region(14, "PointRegion", [0.0, 0.0], [0.0, 0.0])
    enclosing = box_region(15, "Enclosing", [-0.25, -0.25], [0.25, 0.25])
    assert diagnose_region_relation(degenerate, enclosing, "containment")["outcome"] is True
    assert diagnose_region_relation(degenerate, enclosing, "intersection")["verified_witness_points"] == [[0.0, 0.0]]

    inner = box_region(3, "Inner", [-0.25, -0.25], [0.25, 0.25])
    outer = box_region(4, "Outer", [-0.75, -0.75], [0.75, 0.75])
    assert diagnose_region_relation(inner, outer, "containment")["outcome"] is True
    nested_outcome = diagnose_region_relation(outer, inner, "containment")
    assert nested_outcome["outcome"] is False
    witness = np.asarray(nested_outcome["verified_counterexample_points"])
    assert evaluate_region_membership(outer, witness).all()
    assert not evaluate_region_membership(inner, witness).any()

    partial_a = box_region(5, "PartialA", [-0.5, -0.5], [0.3, 0.5])
    partial_b = box_region(6, "PartialB", [0.2, -0.2], [0.8, 0.2])
    assert diagnose_region_relation(partial_a, partial_b, "intersection")["outcome"] is True

    touching_a = box_region(7, "TouchA", [-1.0, -0.5], [0.0, 0.5])
    touching_b = box_region(8, "TouchB", [0.0, -0.5], [1.0, 0.5])
    touch = diagnose_region_relation(touching_a, touching_b, "intersection")
    assert touch["outcome"] is True
    assert touch["verified_witness_points"] == [[0.0, -0.5]]

    disjoint = box_region(9, "Disjoint", [0.1, -0.2], [0.8, 0.2])
    separation = diagnose_region_relation(touching_a, disjoint, "disjointness")
    assert separation["outcome"] is True
    assert separation["decision_scope"] == "exact"
    assert separation["separating_axis_certificate"]["gap"] > 0
    no_intersection = diagnose_region_relation(touching_a, disjoint, "intersection")
    assert no_intersection["outcome"] is False
    assert no_intersection["interpretation"] == "exact_proof_no_intersection"


def _overlapping_towns():
    # House 0 removes the closed central window. House 1 overlaps the frame and
    # re-includes the center; its own window is away from the tested point.
    return town_region(
        10, "TownA",
        frame_low=[[-1.0, -1.0], [-0.1, -0.1]],
        frame_high=[[1.0, 1.0], [0.1, 0.1]],
        window_low=[[[-0.25, -0.25]], [[0.05, 0.05]]],
        window_high=[[[0.25, 0.25]], [[0.08, 0.08]]],
    )


def test_town_membership_respects_overlapping_houses_closed_windows_and_reinclusion():
    town = _overlapping_towns()
    points = np.asarray([[0.0, 0.0], [0.25, 0.0], [0.5, 0.0]], dtype=np.float32)
    # Center is removed from house 0 but re-included by house 1. Window
    # boundary is excluded from house 0 and is outside house 1. The body is
    # otherwise occupied.
    assert evaluate_region_membership(town, points).tolist() == [True, False, True]


def test_town_sampling_finds_verified_counterexample_and_is_reproducible():
    source = town_region(
        11, "BroadTown",
        frame_low=[[-0.9, -0.9]], frame_high=[[0.9, 0.9]],
        window_low=[[[-0.1, -0.1]]], window_high=[[[0.1, 0.1]]],
    )
    target = town_region(
        12, "NarrowTown",
        frame_low=[[-0.25, -0.25]], frame_high=[[0.25, 0.25]],
        window_low=[[[-0.05, -0.05]]], window_high=[[[0.05, 0.05]]],
    )
    first = diagnose_region_relation(source, target, "containment", sample_count=1000, seed=17)
    second = diagnose_region_relation(source, target, "containment", sample_count=1000, seed=17)
    assert first == second
    assert first["decision_scope"] == "sampled_empirical"
    assert first["outcome"] is False
    assert first["sampling"]["candidate_point_count"] == 3000
    witnesses = np.asarray(first["verified_counterexample_points"], dtype=np.float32)
    assert witnesses.shape[0] > 0
    assert evaluate_region_membership(source, witnesses).all()
    assert not evaluate_region_membership(target, witnesses).any()


def test_town_no_sampled_containment_counterexample_is_inconclusive_not_proven():
    first_town = _overlapping_towns()
    same_town = town_region(
        13, "SameTown",
        frame_low=first_town.frame_low, frame_high=first_town.frame_high,
        window_low=first_town.window_low, window_high=first_town.window_high,
    )
    result = diagnose_region_relation(first_town, same_town, "containment", sample_count=512, seed=42)
    assert result["outcome"] is None
    assert result["interpretation"] == "inconclusive_no_sampled_counterexample"
    assert result["decision_scope"] == "sampled_empirical"
    assert result["verified_counterexample_points"] == []
    assert result["sampling"]["sample_count_total"] == 1536


def test_ontology_metadata_distinguishes_asserted_entailed_and_unknown_relations(tmp_path):
    ontology = OntologySnapshot.load(require_pinned_hash=True)
    assert ontology.geometric_relation_metadata("AnchovyTopping", "PizzaTopping", "containment")["status"] in ("asserted", "entailed")
    assert ontology.geometric_relation_metadata("Pizza", "PizzaTopping", "disjointness")["status"] == "asserted"
    unknown = ontology.geometric_relation_metadata("AnchovyTopping", "NonVegetarianPizza", "containment")
    assert unknown["status"] == "unknown"
    assert ontology.geometric_relation_metadata("AnchovyTopping", "NonVegetarianPizza", "intersection")["status"] == "unknown"
    from town_model.inspection.relation_diagnostics import validate_snapshot_ontology_mapping
    with pytest.raises(ValueError, match="snapshot class identity mismatch"):
        validate_snapshot_ontology_mapping(
            SimpleNamespace(header={"class_ids": [2], "class_names": ["NonVegetarianPizza"]}), ontology
        )

    a = box_region(1, "AnchovyTopping", [-0.5, -0.5], [0.5, 0.5])
    b = box_region(2, "PizzaTopping", [-0.7, -0.7], [0.7, 0.7])
    result = diagnose_region_relation(a, b, "containment", ontology=ontology)
    output = save_relation_diagnostic(result, tmp_path / "relation.json")
    reloaded = json.loads(output.read_text())
    assert reloaded["decision_scope"] == "exact"
    assert reloaded["ontology_relation"]["status"] in ("asserted", "entailed")
    with pytest.raises(FileExistsError, match="immutable"):
        save_relation_diagnostic(result, output)
