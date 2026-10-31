from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src" / "town_model"
ACCEPTED_HISTORY = "2de9c5f29b436f727720e87132d0b4be25a40768"


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
    covered: set[str] = set()
    for entry in entries:
        for field in (
            "original_node_ids", "historical_behavior", "retirement_reason",
            "disposition", "independent_assertions_retained", "accepted_historical_revision",
        ):
            assert field in entry and entry[field] not in (None, "", []), (entry, field)
        assert "replacement_tests" in entry
        assert entry["disposition"] in {"deleted", "split", "replaced"}
        assert entry["accepted_historical_revision"] == ACCEPTED_HISTORY
        assert isinstance(entry["replacement_tests"], list)
        for node_id in entry["original_node_ids"]:
            assert node_id not in covered, f"duplicate manifest node: {node_id}"
            assert node_id.startswith("tests/") and "::" in node_id
            covered.add(node_id)
    baseline_nodes = set(manifest["baseline_failed_node_ids"])
    assert len(baseline_nodes) == 132
    node_dispositions = {"deleted": 0, "split": 0, "replaced": 0}
    for entry in entries:
        node_dispositions[entry["disposition"]] += len(entry["original_node_ids"])
    assert node_dispositions == manifest["disposition_node_counts"]
    assert sum(node_dispositions.values()) == manifest["accounted_original_node_count"] == 176
    assert baseline_nodes <= covered
    assert manifest["baseline_full_suite"] == {"passed": 191, "failed": 132, "skipped": 0}
    assert manifest["test_collection_change_summary"] == {
        "starting_t2_collected_nodes": 323,
        "current_collected_nodes_at_manifest_checkpoint": 180,
        "unchanged_original_node_ids_retained": 147,
        "same_node_id_retargeted_or_split": 6,
        "new_node_ids_added": 27,
        "original_node_ids_accounted_in_entries": 176,
    }

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
