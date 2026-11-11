from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src" / "town_model"
ACCEPTED_HISTORY = "2de9c5f29b436f727720e87132d0b4be25a40768"
EXPECTED_NODE_DISPOSITIONS = {"deleted": 96, "replaced": 77, "split": 3}
EXPECTED_GROUP_DISPOSITIONS = {"deleted": 68, "replaced": 58, "split": 3}
FINAL_INTEGRITY_REGRESSION_NODES = {
    "tests/test_image_snapshot_integrity.py::test_reader_rejects_decreasing_uint64_offsets_with_other_metadata_valid",
    "tests/test_image_snapshot_integrity.py::test_reader_accepts_valid_uint64_offsets[increasing]",
    "tests/test_image_snapshot_integrity.py::test_reader_accepts_valid_uint64_offsets[empty-component-range]",
    "tests/test_image_snapshot_integrity.py::test_discovery_skips_decreasing_uint64_offsets_and_keeps_valid_snapshot",
    "tests/test_t3_architecture.py::test_retirement_manifest_rejects_coordinated_disposition_count_mutation",
}
T4_PRESENTATION_REGRESSION_NODES = {
    "tests/test_image_snapshot_integrity.py::test_v2_snapshot_requires_explicit_diagnostic_axis_schema",
}
BOX01_DASHBOARD_REGRESSION_NODES = {
    "tests/test_dashboard_app.py::test_mathematical_inspector_table_uses_cpu_evaluator_and_literal_fixture",
    "tests/test_dashboard_app.py::test_mathematical_inspector_navigation_is_read_only_and_minimal",
}
BOX01_MATHEMATICAL_TEST_NODE_COUNT = 47
BOX02_DASHBOARD_REGRESSION_NODES = {
    "tests/test_dashboard_app.py::test_mathematical_inspector_reports_clearance_mismatches_separately",
}
BOX02_MATHEMATICAL_TEST_NODE_COUNT = 24
TOWN01_INSPECTOR_REGRESSION_NODES = {
    "tests/test_town_math_inspector.py::test_inspector_table_uses_cpu_evaluator_and_independent_literal_fixture",
    "tests/test_town_math_inspector.py::test_standalone_inspector_renders_fixed_fixture_and_is_read_only",
    "tests/test_town_math_inspector.py::test_wrong_evaluator_injection_shows_point_specific_mismatches",
}
TOWN01_MATHEMATICAL_TEST_NODE_COUNT = 44
BOX04_DASHBOARD_REGRESSION_NODES = {
    "tests/test_dashboard_app.py::test_bound_box_inspector_uses_verified_synthetic_identity_and_direct_row_binding",
    "tests/test_dashboard_app.py::test_bound_box_inspector_reports_each_output_mismatch_with_identity",
}
BOX04_MATHEMATICAL_TEST_NODE_COUNT = 80
BOX03_DASHBOARD_REGRESSION_NODES = {
    "tests/test_dashboard_app.py::test_multi_box_inspector_matrices_use_cpu_evaluators_and_literal_fixture",
    "tests/test_dashboard_app.py::test_multi_box_inspector_reports_cell_specific_mismatches_separately",
}
BOX03_MATHEMATICAL_TEST_NODE_COUNT = 58


def _source_files() -> list[Path]:
    return sorted(SOURCE.rglob("*.py"))


def test_retired_modules_and_historical_executable_definitions_are_absent() -> None:
    removed_paths = (
        "models/classifiers.py",
        "models/geometry.py",
        "models/image_embeddings.py",
        "models/classifier_transport.py",
        "models/image_model_validation.py",
        "training/image_objectives.py",
        "experiments/synthetic.py",
        "dashboard/geometry.py",
    )
    for relative in removed_paths:
        assert not (SOURCE / relative).exists(), relative

    forbidden_definitions = {
        "NeuralBaseline", "BoxClassifier", "TownClassifier", "PointClassifier", "make_classifier",
        "BoxGeometry", "TownGeometry", "evaluate_town_corners", "SyntheticPointEncoder",
        "unreduced_penalties", "ImageOntologyModel", "GeometryEvaluation",
        "_penalties", "image_objective_terms", "_bounded_corners", "_set_bounded_corners",
    }
    forbidden_calls = {
        "binary_cross_entropy_with_logits", "softplus", "evaluate_town_corners",
        "physical_boxes", "physical_geometry", "load_state_dict",
    }
    definitions: set[str] = set()
    calls: set[str] = set()
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions.update(
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
    assert not forbidden_definitions & definitions
    assert not forbidden_calls & calls

    source_text = "\n".join(path.read_text(encoding="utf-8") for path in _source_files())
    assert "make_classifier(" not in source_text
    assert "torch.load(" not in source_text
    assert "signed_margin if" not in source_text


def test_supported_import_graph_does_not_load_or_reconstruct_retired_models() -> None:
    script = """
import importlib, sys
import town_model.models
import town_model.models.image_points
import town_model.experiments.image_config
import town_model.experiments.visual_features
import town_model.cli
import town_model.dashboard.app
import town_model.inspection.image_snapshots
import town_model.inspection.synthetic_metadata
import town_model.inspection.relation_diagnostics
removed = (
    'town_model.models.classifiers', 'town_model.models.geometry',
    'town_model.models.image_embeddings', 'town_model.models.classifier_transport',
    'town_model.training.image_objectives', 'town_model.experiments.synthetic',
    'town_model.dashboard.geometry',
)
assert not any(name in sys.modules for name in removed)
for name in removed:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(f'{name} unexpectedly exists')
assert not set(('NeuralBaseline', 'BoxClassifier', 'TownClassifier', 'PointClassifier', 'make_classifier')) & set(town_model.models.__all__)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, check=True)


def _disposition_counts(entries: list[dict[str, object]]) -> tuple[dict[str, int], dict[str, int]]:
    node_counts = {disposition: 0 for disposition in EXPECTED_NODE_DISPOSITIONS}
    group_counts = {disposition: 0 for disposition in EXPECTED_GROUP_DISPOSITIONS}
    for entry in entries:
        disposition = entry["disposition"]
        assert isinstance(disposition, str) and disposition in node_counts
        original_node_ids = entry["original_node_ids"]
        assert isinstance(original_node_ids, list)
        node_counts[disposition] += len(original_node_ids)
        group_counts[disposition] += 1
    return node_counts, group_counts


def _assert_manifest_disposition_counts(manifest: dict[str, object]) -> tuple[dict[str, int], dict[str, int]]:
    entries = manifest["entries"]
    assert isinstance(entries, list)
    node_counts, group_counts = _disposition_counts(entries)
    assert node_counts == EXPECTED_NODE_DISPOSITIONS
    assert group_counts == EXPECTED_GROUP_DISPOSITIONS
    assert manifest["disposition_node_counts"] == EXPECTED_NODE_DISPOSITIONS
    assert manifest["disposition_group_counts"] == EXPECTED_GROUP_DISPOSITIONS
    return node_counts, group_counts


def _collected_current_test_nodes() -> set[str]:
    """Collect node IDs only; this subprocess does not execute collected tests."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "--collect-only", "-q",
            "-p", "no:cacheprovider", "tests",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    }


def test_retirement_manifest_accounts_for_each_original_node_and_mixed_assertions() -> None:
    path = ROOT / "docs" / "reconstruction" / "T3_TEST_RETIREMENT_MANIFEST.md"
    text = path.read_text(encoding="utf-8")
    start_marker = "<!-- T3_RETIREMENT_DATA_START -->"
    end_marker = "<!-- T3_RETIREMENT_DATA_END -->"
    payload = text.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
    assert payload.startswith("```json") and payload.endswith("```")
    payload = payload.removeprefix("```json").removesuffix("```").strip()
    manifest = json.loads(payload)
    assert manifest["accepted_historical_revision"] == ACCEPTED_HISTORY
    assert manifest["starting_t2_revision"] == "c6aad87a45ac75835f02670f32fdeca5167f18f7"
    entries = manifest["entries"]
    assert entries
    current_nodes = _collected_current_test_nodes()
    assert current_nodes

    covered: set[str] = set()
    for entry in entries:
        for field in (
            "original_node_ids", "historical_behavior", "retirement_reason",
            "disposition", "independent_assertions_retained", "accepted_historical_revision",
        ):
            assert field in entry and entry[field] not in (None, "", []), (entry, field)
        assert "replacement_tests" in entry
        disposition = entry["disposition"]
        assert disposition in EXPECTED_NODE_DISPOSITIONS
        assert entry["accepted_historical_revision"] == ACCEPTED_HISTORY
        assert isinstance(entry["replacement_tests"], list)
        assert len(entry["replacement_tests"]) == len(set(entry["replacement_tests"])), entry
        if disposition in {"replaced", "split"}:
            assert entry["replacement_tests"], f"{disposition} entry has no replacement: {entry}"
        else:
            assert not entry["replacement_tests"], f"deleted entry claims replacement coverage: {entry}"
        for node_id in entry["original_node_ids"]:
            assert node_id not in covered, f"duplicate manifest node: {node_id}"
            assert node_id.startswith("tests/") and "::" in node_id
            covered.add(node_id)
        for node_id in entry["replacement_tests"]:
            assert isinstance(node_id, str) and node_id.startswith("tests/") and "::" in node_id, node_id
            assert node_id in current_nodes, f"replacement is not a collected pytest node: {node_id}"

    baseline_failed_ids = manifest["baseline_failed_node_ids"]
    baseline_nodes = set(baseline_failed_ids)
    assert len(baseline_nodes) == len(baseline_failed_ids)
    assert len(baseline_nodes) == manifest["baseline_full_suite"]["failed"]
    additional_accounted_ids = manifest["additional_accounted_node_ids"]
    additional_nodes = set(additional_accounted_ids)
    assert len(additional_nodes) == len(additional_accounted_ids)
    assert not baseline_nodes & additional_nodes
    assert covered == baseline_nodes | additional_nodes
    assert baseline_nodes <= covered
    node_dispositions, group_dispositions = _assert_manifest_disposition_counts(manifest)
    assert sum(node_dispositions.values()) == manifest["accounted_original_node_count"]
    assert len(covered) == manifest["accounted_original_node_count"]
    assert len(entries) == manifest["entry_count"]

    baseline_suite = manifest["baseline_full_suite"]
    assert baseline_suite == {"passed": 191, "failed": 132, "skipped": 0}
    summary = manifest["test_collection_change_summary"]
    assert summary["starting_t2_collected_nodes"] == sum(baseline_suite.values())
    assert isinstance(summary["submitted_t3_collected_nodes_before_remediation"], int)
    assert summary["submitted_t3_collected_nodes_before_remediation"] < len(current_nodes)
    checkpoint_node_count = summary["current_collected_nodes_at_remediation_checkpoint"]
    assert FINAL_INTEGRITY_REGRESSION_NODES <= current_nodes
    assert T4_PRESENTATION_REGRESSION_NODES <= current_nodes
    assert BOX01_DASHBOARD_REGRESSION_NODES <= current_nodes
    assert BOX02_DASHBOARD_REGRESSION_NODES <= current_nodes
    bounded_box_mathematical_nodes = {
        node for node in current_nodes if node.startswith("tests/test_bounded_box.py::")
    }
    box02_mathematical_nodes = {
        node for node in bounded_box_mathematical_nodes
        if "::test_box02_" in node
    }
    box01_mathematical_nodes = bounded_box_mathematical_nodes - box02_mathematical_nodes
    assert len(box01_mathematical_nodes) == BOX01_MATHEMATICAL_TEST_NODE_COUNT
    assert len(box02_mathematical_nodes) == BOX02_MATHEMATICAL_TEST_NODE_COUNT
    box01_nodes = box01_mathematical_nodes | BOX01_DASHBOARD_REGRESSION_NODES
    box02_nodes = box02_mathematical_nodes | BOX02_DASHBOARD_REGRESSION_NODES
    multi_box_mathematical_nodes = {
        node for node in current_nodes if node.startswith("tests/test_bounded_multi_box.py::")
    }
    assert len(multi_box_mathematical_nodes) == BOX03_MATHEMATICAL_TEST_NODE_COUNT
    assert BOX03_DASHBOARD_REGRESSION_NODES <= current_nodes
    box03_nodes = multi_box_mathematical_nodes | BOX03_DASHBOARD_REGRESSION_NODES
    town01_mathematical_nodes = {
        node for node in current_nodes if node.startswith("tests/test_town_geometry.py::")
    }
    assert len(town01_mathematical_nodes) == TOWN01_MATHEMATICAL_TEST_NODE_COUNT
    assert TOWN01_INSPECTOR_REGRESSION_NODES <= current_nodes
    town01_nodes = town01_mathematical_nodes | TOWN01_INSPECTOR_REGRESSION_NODES
    bound_box_mathematical_nodes = {
        node for node in current_nodes if node.startswith("tests/test_bound_box_inference.py::")
    }
    assert len(bound_box_mathematical_nodes) == BOX04_MATHEMATICAL_TEST_NODE_COUNT
    assert BOX04_DASHBOARD_REGRESSION_NODES <= current_nodes
    box04_nodes = bound_box_mathematical_nodes | BOX04_DASHBOARD_REGRESSION_NODES
    box_nodes = box01_nodes | box02_nodes | box03_nodes
    accounted_feature_nodes = box_nodes | town01_nodes | box04_nodes
    legacy_nodes = current_nodes - accounted_feature_nodes
    assert (
        checkpoint_node_count
        + len(FINAL_INTEGRITY_REGRESSION_NODES)
        + len(T4_PRESENTATION_REGRESSION_NODES)
        == len(legacy_nodes)
    )
    assert len(current_nodes) == (
        len(legacy_nodes)
        + len(box01_nodes)
        + len(box02_nodes)
        + len(box03_nodes)
        + len(town01_nodes)
        + len(box04_nodes)
    )
    assert summary["original_node_ids_accounted_in_entries"] == len(covered)
    assert (
        summary["unchanged_original_node_ids_retained"]
        + summary["same_node_id_retargeted_or_split"]
        + summary["new_node_ids_added"]
        == checkpoint_node_count
    )

    snapshot_entry = next(
        entry for entry in entries
        if "test_snapshot_diagnostic_axes_roundtrip_preserve_class_geometry_and_instance_activity"
        in " ".join(entry["original_node_ids"])
    )
    assert snapshot_entry["disposition"] == "replaced"
    assert snapshot_entry["replacement_tests"]
    assert any("test_reader_roundtrips_nonempty_diagnostic_axes" in node for node in snapshot_entry["replacement_tests"])

    relation_entry = next(
        entry for entry in entries
        if "test_ontology_metadata_distinguishes_asserted_entailed_and_unknown_relations" in " ".join(entry["original_node_ids"])
    )
    assert relation_entry["disposition"] == "split"
    assert "ontology" in relation_entry["independent_assertions_retained"].lower()
    assert relation_entry["replacement_tests"]

    d5a_entry = next(
        entry for entry in entries
        if "test_geometric_violation_does_not_change_ontology_entailment" in " ".join(entry["original_node_ids"])
    )
    assert d5a_entry["disposition"] == "split"
    assert "entailment" in d5a_entry["independent_assertions_retained"].lower()
    assert any("test_mushroom_vegetarian_ontology_entailment_is_independent_of_geometry" in node for node in d5a_entry["replacement_tests"])


def test_retirement_manifest_rejects_coordinated_disposition_count_mutation() -> None:
    path = ROOT / "docs" / "reconstruction" / "T3_TEST_RETIREMENT_MANIFEST.md"
    text = path.read_text(encoding="utf-8")
    start_marker = "<!-- T3_RETIREMENT_DATA_START -->"
    end_marker = "<!-- T3_RETIREMENT_DATA_END -->"
    payload = text.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
    payload = payload.removeprefix("```json").removesuffix("```").strip()
    original_manifest = json.loads(payload)
    mutated_manifest = dict(original_manifest)
    mutated_manifest["entries"] = [dict(entry) for entry in original_manifest["entries"]]
    entry = next(item for item in mutated_manifest["entries"] if item["disposition"] == "replaced")
    entry["disposition"] = "deleted"
    mutated_nodes, mutated_groups = _disposition_counts(mutated_manifest["entries"])
    mutated_manifest["disposition_node_counts"] = mutated_nodes
    mutated_manifest["disposition_group_counts"] = mutated_groups

    assert mutated_nodes == {"deleted": 97, "replaced": 76, "split": 3}
    assert mutated_groups == {"deleted": 69, "replaced": 57, "split": 3}
    with pytest.raises(AssertionError):
        _assert_manifest_disposition_counts(mutated_manifest)
