from __future__ import annotations

from dataclasses import replace
import hashlib
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
from town_model.experiments.image_runs import (
    inspect_run,
    launch_image_experiment,
    request_stop,
    resolve_image_experiment,
    run_image_experiment,
    run_image_matrix,
)
from town_model.inspection.image_snapshots import load_image_snapshot, save_image_snapshot
from town_model.models.image_points import ImagePointModel
from town_model.training.image_trainer import (
    _evaluate_image_checkpoint,
    _load_checkpoint,
    build_image_modules,
    build_visual_feature_extractor,
    evaluate_image_checkpoint,
    evaluate_image_model,
    train_image_model,
)

ROOT = Path(__file__).parents[1]


def _config() -> ImageExperimentConfig:
    return ImageExperimentConfig(
        dataset=DatasetSubsetConfig(
            dataset_path="examples/pizza-48",
            train_size=4,
            validation_size=2,
            selection_seed=31,
        ),
        vision=VisionConfig(pretrained_imagenet=False, visual_policy="frozen"),
        model=ImageModelConfig(family="baseline", pizza_mode="hybrid", latent_dimensions=5),
        training=TrainingConfig(
            seed=23,
            epochs=1,
            physical_batch_size=2,
            accumulation_steps=1,
            schedule="constant",
            warmup_steps=0,
        ),
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
        for path in sorted(root.rglob("*"))
        if path.is_file()
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
    _expect_unavailable(
        lambda: (_ for _ in ()).throw(ClassifierBackendUnavailableError("image classifier training")),
        "image classifier training",
    )


def test_classifier_construction_inference_geometry_and_losses_are_unavailable() -> None:
    from town_model.models.classifiers import (
        BoxClassifier,
        NeuralBaseline,
        TownClassifier,
        make_classifier,
        unreduced_penalties,
    )
    from town_model.models.geometry import BoxGeometry, TownGeometry, evaluate_town_corners
    from town_model.models.image_embeddings import ImageOntologyModel

    _expect_unavailable(lambda: NeuralBaseline(2, ("A",)))
    _expect_unavailable(lambda: BoxClassifier(2, ("A",)))
    _expect_unavailable(lambda: TownClassifier(2, ("A",)))
    _expect_unavailable(lambda: make_classifier("baseline", 2, ("A",)))
    _expect_unavailable(lambda: ImageOntologyModel(2, ("A",), family="baseline", pizza_mode="global"))
    _expect_unavailable(lambda: BoxGeometry(2, 1))
    _expect_unavailable(lambda: TownGeometry(2, 1))
    _expect_unavailable(lambda: evaluate_town_corners(None, None, None, None, None))
    _expect_unavailable(lambda: unreduced_penalties(None, None, mode="bce"))


def test_image_execution_and_checkpoint_entrypoints_fail_before_model_construction(monkeypatch) -> None:
    import town_model.models.classifiers as classifiers
    from town_model.vision.backbone import MobileNetV3LargeFPN

    def forbidden(*args, **kwargs):
        raise AssertionError("historical classifier constructor/factory was reached")

    for name in ("NeuralBaseline", "BoxClassifier", "TownClassifier", "make_classifier"):
        monkeypatch.setattr(classifiers, name, forbidden)
    monkeypatch.setattr(MobileNetV3LargeFPN, "__init__", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)

    config = _config()
    _expect_unavailable(lambda: build_image_modules(None, config), "module construction")
    _expect_unavailable(lambda: train_image_model(None, None, None, config, Path("unused")), "training")
    _expect_unavailable(
        lambda: evaluate_image_model(
            None, None, None, config, class_indices=(), device=torch.device("cpu")
        ),
        "evaluation",
    )
    _expect_unavailable(lambda: evaluate_image_checkpoint(None, None, None, None, config, "unused", class_ids=()))
    _expect_unavailable(lambda: _evaluate_image_checkpoint(None, None, None, None, config, "unused", class_ids=()))
    _expect_unavailable(lambda: _load_checkpoint(Path("unused"), {}, torch.device("cpu")))
    _expect_unavailable(lambda: run_image_experiment(config, Path("unused")))
    _expect_unavailable(lambda: launch_image_experiment(config, Path("unused")))
    _expect_unavailable(lambda: run_image_matrix(config, Path("unused")))
    _expect_unavailable(lambda: request_stop(Path("unused")))


def test_image_preflight_validates_without_constructing_or_claiming_a_backend(monkeypatch) -> None:
    from town_model.models.image_embeddings import ImageOntologyModel
    from town_model.vision.backbone import MobileNetV3LargeFPN

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight constructed a historical classifier or visual backbone")

    monkeypatch.setattr(ImageOntologyModel, "__init__", forbidden)
    monkeypatch.setattr(MobileNetV3LargeFPN, "__init__", forbidden)
    _expect_unavailable(
        lambda: resolve_image_experiment(_config(), repository_root=ROOT),
        "image classifier experiment preflight",
    )


def test_worker_snapshot_and_relation_execution_are_unavailable_before_side_effects(tmp_path) -> None:
    from town_model.experiments.image_worker import run_worker
    from town_model.inspection.relation_diagnostics import (
        diagnose_region_relation,
        evaluate_region_membership,
        region_from_snapshot,
    )

    _expect_unavailable(run_worker, "background worker")
    command = subprocess.run(
        [
            sys.executable,
            "-m",
            "town_model.experiments.image_worker",
            "--config",
            str(tmp_path / "missing-config.json"),
            "--output",
            str(tmp_path / "worker-output"),
            "--status",
            str(tmp_path / "worker-status.json"),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert command.returncode == 2
    assert "intentionally disconnected" in command.stderr
    assert not (tmp_path / "worker-output").exists()
    assert not (tmp_path / "worker-status.json").exists()
    _expect_unavailable(
        lambda: save_image_snapshot(
            tmp_path / "snapshot",
            dataset=None,
            train_subset=None,
            validation_subset=None,
            visual=None,
            model=None,
            config=None,
            class_ids=(),
            class_indices=(),
            epoch=0,
            optimizer_step=0,
            latest_checkpoint=tmp_path / "missing.pt",
            identity_hash="unused",
        ),
        "snapshot generation",
    )
    _expect_unavailable(lambda: region_from_snapshot(None, 0), "snapshot geometry")
    _expect_unavailable(lambda: evaluate_region_membership(None, []), "membership evaluation")
    _expect_unavailable(lambda: diagnose_region_relation(None, None, "containment"), "relation diagnosis")
    assert not (tmp_path / "snapshot").exists()


def test_synthetic_training_loading_evaluation_resume_and_geometry_are_unavailable(tmp_path) -> None:
    from town_model.experiments.synthetic import (
        SyntheticConfig,
        evaluate_saved_run,
        load_saved_run,
        resume_saved_run,
        run_experiment,
        run_fixed_experiments,
        saved_geometry,
    )

    output = tmp_path / "synthetic-runs"
    _expect_unavailable(lambda: run_experiment("fixed", "simple", output, config=SyntheticConfig()))
    _expect_unavailable(lambda: run_fixed_experiments(output))
    _expect_unavailable(lambda: load_saved_run(tmp_path / "old-run"))
    _expect_unavailable(lambda: evaluate_saved_run(tmp_path / "old-run"))
    _expect_unavailable(lambda: resume_saved_run(tmp_path / "old-run", 1))
    _expect_unavailable(lambda: saved_geometry(tmp_path / "old-run"))
    assert not output.exists()


def test_rejected_operations_do_not_create_runs_workers_checkpoints_or_exports(tmp_path) -> None:
    from town_model.experiments.synthetic import run_experiment, run_fixed_experiments

    config = _config()
    image_run_root = tmp_path / "image-run-root"
    worker_root = tmp_path / "worker-root"
    matrix_root = tmp_path / "matrix-root"
    synthetic_root = tmp_path / "synthetic-root"

    _expect_unavailable(lambda: run_image_experiment(config, image_run_root))
    _expect_unavailable(lambda: launch_image_experiment(config, worker_root))
    _expect_unavailable(lambda: run_image_matrix(config, matrix_root))
    cache_config = replace(
        config,
        execution=replace(
            config.execution,
            feature_cache_directory=str(tmp_path / "vision-cache"),
        ),
    )
    _expect_unavailable(lambda: run_image_experiment(cache_config, tmp_path / "cached-image-run"))
    _expect_unavailable(lambda: run_experiment("mapping", "simple", synthetic_root))
    _expect_unavailable(lambda: run_fixed_experiments(synthetic_root))

    for path in (image_run_root, worker_root, matrix_root, synthetic_root, tmp_path / "vision-cache", tmp_path / "cached-image-run"):
        assert not path.exists()
    assert not (worker_root / ".image-workers").exists()
    assert not (image_run_root / "checkpoints").exists()
    assert not (image_run_root / "manifest.json").exists()


def test_rejected_resume_and_stop_leave_existing_lifecycle_and_artifacts_unchanged(tmp_path) -> None:
    from town_model.experiments.synthetic import resume_saved_run

    image_run = tmp_path / "historical-image-run"
    image_run.mkdir()
    (image_run / "manifest.json").write_text(json.dumps({"state": "running", "marker": "unchanged"}))
    (image_run / ".writer.lock").write_bytes(b"do-not-touch-lock")
    (image_run / "stop.request").write_bytes(b"existing marker")
    (image_run / "checkpoints").mkdir()
    (image_run / "checkpoints" / "latest.pt").write_bytes(b"historical-checkpoint")
    before_image = _tree_hash(image_run)

    _expect_unavailable(lambda: run_image_experiment(_config(), image_run.parent, resume=True))
    _expect_unavailable(lambda: launch_image_experiment(_config(), image_run.parent, resume=True))
    _expect_unavailable(lambda: request_stop(image_run))
    assert _tree_hash(image_run) == before_image

    synthetic_run = tmp_path / "historical-synthetic-run"
    synthetic_run.mkdir()
    (synthetic_run / "checkpoint.pt").write_bytes(b"historical-synthetic-checkpoint")
    (synthetic_run / "metrics.json").write_text('{"marker":"unchanged"}')
    before_synthetic = _tree_hash(synthetic_run)
    _expect_unavailable(lambda: resume_saved_run(synthetic_run, 2))
    assert _tree_hash(synthetic_run) == before_synthetic


def test_read_only_run_inspection_and_checksums_remain_available(tmp_path) -> None:
    run = tmp_path / "old-run"
    run.mkdir()
    manifest_path = run / "manifest.json"
    manifest_path.write_text(json.dumps({"state": "running", "marker": "stable"}))
    lock_path = run / ".writer.lock"
    lock_path.write_bytes(b"stable lock bytes")
    worker_dir = tmp_path / ".image-workers"
    worker_dir.mkdir()
    worker_status = worker_dir / "old-run.json"
    worker_status.write_text('{"state":"launching","marker":"stable"}')
    before = {path: path.read_bytes() for path in (manifest_path, lock_path, worker_status)}

    assert inspect_run(run, refresh_interrupted=True) == {"state": "running", "marker": "stable"}
    assert {path: path.read_bytes() for path in before} == before

    snapshot_dir = tmp_path / "historical-snapshot"
    snapshot_dir.mkdir()
    arrays_path = snapshot_dir / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        sample_ids=np.asarray(["sample-1"], dtype=np.str_),
        component_offsets=np.asarray([0, 0], dtype=np.int64),
        component_count_total=np.asarray(0, dtype=np.int64),
    )
    arrays_digest = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    (snapshot_dir / "header.json").write_text(json.dumps({
        "schema_version": "image-research-snapshot-v2",
        "sample_ids": ["sample-1"],
        "diagnostic_axes_schema": {},
        "arrays_sha256": arrays_digest,
    }))
    snapshot = load_image_snapshot(snapshot_dir)
    assert snapshot.header["sample_ids"] == ["sample-1"]
    assert snapshot.arrays["component_offsets"].tolist() == [0, 0]


def test_image_point_path_and_visual_only_feature_extractor_remain_independent() -> None:
    points = ImagePointModel(5, pizza_mode="hybrid", seed=21).eval()
    with torch.no_grad():
        result = points(_features(), ("sample-1", "sample-2"), diagnostics=True)
    assert result.sample_ids == ("sample-1", "sample-2")
    assert result.component_ids == _features().component_ids
    assert result.component_valid_mask.shape == (2, 3)
    assert result.component_points.shape == (2, 3, 5)
    assert result.global_pizza_points.shape == (2, 5)
    assert result.composed_pizza_points.shape == (2, 5)
    assert result.final_pizza_points.shape == (2, 5)
    assert not hasattr(result, "contains")
    assert not hasattr(result, "logits")
    assert callable(build_visual_feature_extractor)


def test_fresh_process_imports_and_dashboard_model_pages_do_not_reach_classifiers() -> None:
    script = """
import sys
import town_model.cli
import town_model.dashboard.app
import town_model.dashboard.image_embeddings
import town_model.experiments.image_worker
import town_model.inspection.image_snapshots
import town_model.inspection.relation_diagnostics
import town_model.models.image_points
assert 'town_model.models.classifiers' not in sys.modules
assert 'town_model.models.geometry' not in sys.modules
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


@pytest.mark.parametrize(
    ("arguments", "expected_operation"),
    [
        (("experiment", "fixed", "--output", "{out}"), "synthetic classifier training"),
        (("experiment", "mapping", "--output", "{out}", "--scenario", "simple"), "synthetic classifier training"),
        (("experiment", "image", "preflight", "--config", "{missing}"), "image classifier experiment preflight"),
        (("experiment", "image", "run", "--config", "{missing}", "--output", "{out}"), "image-to-ontology classifier training"),
        (("experiment", "image", "matrix", "--config", "{missing}", "--output", "{out}"), "automatic image classifier matrix execution"),
        (("experiment", "image", "resume", "{run}"), "image-to-ontology classifier training resume"),
        (("experiment", "image", "evaluate", "{run}"), "historical image classifier checkpoint evaluation"),
        (("experiment", "image", "relations", "{missing}", "--source-class-id", "1", "--target-class-id", "2", "--relation", "containment", "--output", "{out}.json"), "historical ontology-region relation diagnosis"),
        (("experiment", "image", "stop", "{run}"), "control a historical image classifier worker"),
        (("evaluate", "{run}"), "synthetic classifier evaluation"),
        (("resume", "{run}", "--additional-steps", "1"), "synthetic classifier training resume"),
        (("export-geometry", "{run}", "--output", "{out}.json"), "synthetic classifier geometry export"),
    ],
)
def test_model_dependent_cli_commands_fail_nonzero_before_artifacts(
    arguments: tuple[str, ...], expected_operation: str, tmp_path, capsys
) -> None:
    from town_model.cli import main

    output = tmp_path / "output"
    missing = tmp_path / "missing.json"
    run = tmp_path / "historical-run"
    argv = [
        value.format(out=output, missing=missing, run=run)
        for value in arguments
    ]
    with pytest.raises(SystemExit) as captured:
        main(argv)
    assert captured.value.code == 2
    error = capsys.readouterr().err
    assert expected_operation in error
    assert "intentionally disconnected" in error
    assert "no reconstructed classifier backend is currently approved" in error
    assert not output.exists()
    assert not missing.exists()
    assert not run.exists()


def test_image_cli_metadata_inspection_remains_read_only(tmp_path, capsys) -> None:
    from town_model.cli import main

    run = tmp_path / "metadata-only-run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "schema_version": "image-run-manifest-v1",
        "run_id": run.name,
        "identity_hash": "stored-id",
        "state": "completed",
    }))
    (run / "metrics.json").write_text('{"historical_metric": 0.25}')
    before = _tree_hash(run)
    assert main(["experiment", "image", "inspect", str(run)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["manifest"]["identity_hash"] == "stored-id"
    assert output["metrics"] == {"historical_metric": 0.25}
    assert _tree_hash(run) == before
