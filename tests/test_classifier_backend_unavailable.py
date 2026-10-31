from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from streamlit.testing.v1 import AppTest

from town_model.classifier_backend import ClassifierBackendUnavailableError
from town_model.contracts import FeatureDiagnostics, VisionFeatureBatch
from town_model.experiments.image_config import (
    DatasetSubsetConfig,
    ExecutionConfig,
    ImageExperimentConfig,
    ImageModelConfig,
    TrainingConfig,
    VisionConfig,
)
from town_model.models.image_points import ImagePointModel
from town_model.training.supervision import (
    ImageObjectiveTerms,
    aggregate_effective_batch,
    reduce_weighted_instances,
)

ROOT = Path(__file__).parents[1]


def _config() -> ImageExperimentConfig:
    return ImageExperimentConfig(
        dataset=DatasetSubsetConfig(dataset_path="examples/pizza-48", train_size=4, validation_size=2, selection_seed=31),
        vision=VisionConfig(pretrained_imagenet=False, visual_policy="frozen"),
        model=ImageModelConfig(family="baseline", pizza_mode="hybrid", latent_dimensions=5),
        training=TrainingConfig(seed=23, epochs=1, physical_batch_size=2, accumulation_steps=1, schedule="constant", warmup_steps=0),
        execution=ExecutionConfig(device="cpu", cpu_threads=1),
        purpose="smoke",
    )


def _expect_unavailable(call, operation: str | None = None) -> None:
    with pytest.raises(ClassifierBackendUnavailableError) as captured:
        call()
    message = str(captured.value)
    if operation is not None:
        assert operation in message
    assert "intentionally disconnected" in message
    assert "no reconstructed classifier backend is currently approved" in message
    assert "author's model reconstruction" in message
    assert "implemented and accepted" in message


def _tree_hash(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _features() -> VisionFeatureBatch:
    generator = torch.Generator().manual_seed(784)
    return VisionFeatureBatch(
        component_vectors=torch.randn((2, 3, 260), generator=generator),
        component_valid_mask=torch.tensor([[True, True, False], [True, False, False]]),
        global_spatial_features=torch.randn((2, 256, 7, 7), generator=generator),
        component_ids=(("a", "b", None), ("c", None, None)),
        diagnostics=FeatureDiagnostics(),
    )


def test_central_unavailable_error_is_explicit_and_actionable() -> None:
    _expect_unavailable(lambda: (_ for _ in ()).throw(ClassifierBackendUnavailableError("image classifier training")), "image classifier training")


def test_retired_model_modules_and_package_exports_are_absent() -> None:
    import town_model.models as models

    removed_symbols = (
        "NeuralBaseline", "BoxClassifier", "TownClassifier", "PointClassifier", "make_classifier",
        "BoxGeometry", "TownGeometry", "evaluate_town_corners", "SyntheticPointEncoder",
        "unreduced_penalties", "ImageOntologyModel", "ClassifierOutput", "GeometryEvaluation",
    )
    assert not set(removed_symbols) & set(models.__all__)
    for name in removed_symbols:
        assert not hasattr(models, name)
    for module in (
        "town_model.models.classifiers", "town_model.models.geometry",
        "town_model.models.image_embeddings", "town_model.models.classifier_transport",
        "town_model.models.image_model_validation", "town_model.training.image_objectives",
        "town_model.experiments.synthetic", "town_model.dashboard.geometry",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)


def test_supported_model_operations_fail_explicitly_without_side_effects(tmp_path: Path) -> None:
    from town_model.experiments.image_runs import (
        launch_image_experiment, request_stop, resolve_image_experiment,
        run_image_experiment, run_image_matrix,
    )
    from town_model.experiments.image_worker import run_worker
    from town_model.inspection.image_snapshots import save_image_snapshot
    from town_model.inspection.relation_diagnostics import diagnose_region_relation
    from town_model.training.image_trainer import (
        build_image_modules, evaluate_image_checkpoint, evaluate_image_model, train_image_model,
    )

    config = _config()
    output = tmp_path / "model-output"
    calls = (
        (lambda: build_image_modules(None, config), "module construction"),
        (lambda: train_image_model(None, None, None, config, output), "training"),
        (lambda: evaluate_image_model(None, None, None, config), "evaluation"),
        (lambda: evaluate_image_checkpoint(None, None, None, None, config, "missing", class_ids=()), "checkpoint evaluation"),
        (lambda: run_image_experiment(config, output), "training or resume"),
        (lambda: launch_image_experiment(config, output), "background worker"),
        (lambda: run_image_matrix(config, output), "matrix execution"),
        (lambda: request_stop(output), "historical image classifier worker"),
        (lambda: run_worker(), "background worker execution"),
        (lambda: save_image_snapshot(output / "snapshot"), "snapshot generation"),
        (lambda: diagnose_region_relation(None, None, "containment"), "relation diagnosis"),
    )
    for call, operation in calls:
        _expect_unavailable(call, operation)
    assert not output.exists()

    _expect_unavailable(
        lambda: resolve_image_experiment(config, repository_root=ROOT),
        "image classifier experiment preflight",
    )
    assert not output.exists()


def test_rejected_operations_do_not_mutate_existing_runs_or_create_cli_artifacts(tmp_path: Path, capsys) -> None:
    from town_model.cli import main
    from town_model.experiments.image_runs import request_stop, run_image_experiment

    run = tmp_path / "historical-run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"state": "running", "marker": "unchanged"}))
    (run / ".writer.lock").write_bytes(b"stable lock")
    (run / "stop.request").write_bytes(b"existing marker")
    before = _tree_hash(run)
    _expect_unavailable(lambda: run_image_experiment(_config(), run.parent, resume=True))
    _expect_unavailable(lambda: request_stop(run))
    assert _tree_hash(run) == before

    output = tmp_path / "cli-output"
    with pytest.raises(SystemExit) as error:
        main(["experiment", "fixed", "--output", str(output)])
    assert error.value.code == 2
    assert "synthetic classifier training" in capsys.readouterr().err
    assert not output.exists()


def test_classifier_dependent_cli_routes_reject_before_reads_or_publication(tmp_path: Path, capsys) -> None:
    from town_model.cli import main

    output = tmp_path / "result.json"
    missing = tmp_path / "missing.json"
    run = tmp_path / "missing-run"
    commands = (
        (["experiment", "image", "preflight", "--config", str(missing)], "image classifier experiment preflight"),
        (["experiment", "image", "run", "--config", str(missing), "--output", str(output)], "image-to-ontology classifier training"),
        (["experiment", "image", "relations", str(missing), "--source-class-id", "1", "--target-class-id", "2", "--relation", "containment", "--output", str(output)], "historical ontology-region relation diagnosis"),
        (["evaluate", str(run)], "synthetic classifier evaluation"),
        (["resume", str(run), "--additional-steps", "1"], "synthetic classifier training resume"),
        (["export-geometry", str(run), "--output", str(output)], "synthetic classifier geometry export"),
    )
    for argv, operation in commands:
        with pytest.raises(SystemExit) as error:
            main(argv)
        assert error.value.code == 2
        stderr = capsys.readouterr().err
        assert operation in stderr
        assert "intentionally disconnected" in stderr
        assert not output.exists()
        assert not missing.exists()
        assert not run.exists()


def test_image_and_synthetic_metadata_inspection_are_read_only(tmp_path: Path, capsys) -> None:
    from town_model.cli import main
    from town_model.experiments.image_runs import inspect_run
    from town_model.inspection.synthetic_metadata import inspect_saved_run

    image_run = tmp_path / "image-run"
    image_run.mkdir()
    (image_run / "manifest.json").write_text(json.dumps({"schema_version": "image-run-manifest-v1", "state": "completed", "identity_hash": "persisted"}))
    (image_run / "metrics.json").write_text('{"stored": 0.25}')
    image_before = _tree_hash(image_run)
    assert inspect_run(image_run)["identity_hash"] == "persisted"
    assert main(["experiment", "image", "inspect", str(image_run)]) == 0
    assert _tree_hash(image_run) == image_before
    assert json.loads(capsys.readouterr().out)["metrics"] == {"stored": 0.25}

    synthetic_run = tmp_path / "synthetic-run"
    synthetic_run.mkdir()
    for name, value in (("metrics.json", {"stored": 1}), ("resolved_config.json", {"family": "town"}), ("environment.json", {"python": "3.x"})):
        (synthetic_run / name).write_text(json.dumps(value))
    synthetic_before = _tree_hash(synthetic_run)
    result = inspect_saved_run(synthetic_run)
    assert result["metrics"] == {"stored": 1}
    assert _tree_hash(synthetic_run) == synthetic_before


def test_checksum_validated_snapshot_values_remain_readable_without_model_import(tmp_path: Path) -> None:
    from town_model.inspection.image_snapshots import load_image_snapshot

    snapshot_dir = tmp_path / "historical-snapshot"
    snapshot_dir.mkdir()
    arrays_path = snapshot_dir / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        sample_ids=np.asarray(["sample-1"], dtype=np.str_),
        component_offsets=np.asarray([0, 0], dtype=np.int64),
        component_count_total=np.asarray(0, dtype=np.int64),
        pizza_contains=np.asarray([[True]], dtype=np.bool_),
        pizza_signed_margin=np.asarray([[0.125]], dtype=np.float32),
    )
    digest = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    (snapshot_dir / "header.json").write_text(json.dumps({
        "schema_version": "image-research-snapshot-v2",
        "sample_ids": ["sample-1"],
        "diagnostic_axes_schema": {},
        "arrays_sha256": digest,
        "geometry": {},
    }))
    snapshot = load_image_snapshot(snapshot_dir)
    assert snapshot.arrays["pizza_contains"].tolist() == [[True]]
    assert snapshot.arrays["pizza_signed_margin"].tolist() == [[0.125]]


def test_snapshot_reader_rejects_unsupported_schema_without_model_reconstruction(tmp_path: Path) -> None:
    from town_model.inspection.image_snapshots import load_image_snapshot

    snapshot_dir = tmp_path / "legacy-snapshot"
    snapshot_dir.mkdir()
    arrays_path = snapshot_dir / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        sample_ids=np.asarray([], dtype=np.str_),
        component_offsets=np.asarray([0], dtype=np.int64),
        component_count_total=np.asarray(0, dtype=np.int64),
    )
    digest = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    (snapshot_dir / "header.json").write_text(json.dumps({
        "schema_version": "image-research-snapshot-v1",
        "sample_ids": [],
        "diagnostic_axes_schema": {},
        "arrays_sha256": digest,
        "geometry": {},
    }))
    with pytest.raises(ValueError, match="expected image-research-snapshot-v2 with explicit diagnostic axes"):
        load_image_snapshot(snapshot_dir)


def test_image_point_path_and_visual_only_feature_factory_remain_independent() -> None:
    points = ImagePointModel(5, pizza_mode="hybrid", seed=21).eval()
    features = _features()
    with torch.no_grad():
        result = points(features, ("sample-1", "sample-2"), diagnostics=True)
    assert result.sample_ids == ("sample-1", "sample-2")
    assert result.component_ids == features.component_ids
    assert result.component_valid_mask.shape == (2, 3)
    assert result.component_points.shape == (2, 3, 5)
    assert result.global_pizza_points.shape == (2, 5)
    assert result.composed_pizza_points.shape == (2, 5)
    assert result.final_pizza_points.shape == (2, 5)
    assert not hasattr(result, "contains")
    assert not hasattr(result, "logits")
    from town_model.experiments.visual_features import build_visual_feature_extractor
    assert callable(build_visual_feature_extractor)


def test_generic_weighted_reduction_and_effective_batch_aggregation_remain_independent() -> None:
    values = torch.tensor([[2.0, 10.0, 4.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    known = torch.tensor([[True, True, False]])
    reduced, eligible = reduce_weighted_instances(values, targets, known, positive_weights=torch.tensor([9.0, 1.0, 100.0]))
    assert reduced.tolist() == pytest.approx([2.8])
    assert eligible.tolist() == [True]
    reduced.sum().backward()
    assert torch.allclose(values.grad, torch.tensor([[0.9, 0.1, 0.0]]))

    terms = ImageObjectiveTerms(
        component_per_image=torch.tensor([1.0, 9.0]),
        component_eligible_images=torch.tensor([True, False]),
        pizza_per_image=torch.tensor([2.0, 6.0]),
        pizza_eligible_images=torch.tensor([True, True]),
        component_instance_count=1,
        pizza_instance_count=2,
    )
    aggregate = aggregate_effective_batch([terms])
    assert aggregate.component_loss.item() == 1.0
    assert aggregate.pizza_loss.item() == 4.0
    assert aggregate.total.item() == 5.0


def test_fresh_import_graph_and_model_pages_do_not_load_removed_implementations() -> None:
    script = """
import importlib, sys
import town_model.models
import town_model.models.image_points
import town_model.experiments.image_config
import town_model.cli
import town_model.dashboard.app
import town_model.dashboard.image_embeddings
import town_model.inspection.image_snapshots
import town_model.inspection.relation_diagnostics
assert not any(name in sys.modules for name in (
    'town_model.models.classifiers', 'town_model.models.geometry',
    'town_model.models.image_embeddings', 'town_model.experiments.synthetic',
    'town_model.training.image_objectives', 'town_model.dashboard.geometry',
))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, check=True)

    app_path = ROOT / "src" / "town_model" / "dashboard" / "app.py"
    for page in ("Geometry Laboratory", "Image Embeddings"):
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        app.sidebar.radio[0].set_value(page)
        app.run()
        assert not app.exception
        assert any("intentionally disconnected" in item.value for item in app.info)
        assert not app.button
