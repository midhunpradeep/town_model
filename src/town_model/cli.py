"""Automation-friendly Phase 1 command line, backed by application services."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import Sequence

from .application import generate_dataset, inspect_dataset, validate_dataset_operation
from .classifier_backend import raise_classifier_backend_unavailable
from .config import DEFAULT_CONFIG_PATH, DEFAULT_PIZZA_TYPES, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="town",
        description="Prepare Pizzaïolo datasets; historical classifier execution is unavailable during classifier reconstruction.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Explicitly generate and prepare one new immutable dataset")
    generate.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    generate.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are never overwritten")
    generate.add_argument("--total-instances", type=int)
    generate.add_argument("--generation-seed", type=int)
    generate.add_argument("--split-seed", type=int)
    generate.add_argument("--pizza-types", nargs="+", choices=DEFAULT_PIZZA_TYPES)
    generate.add_argument("--type-counts", help='Optional JSON map, e.g. \'{"American": 3, "random": 3}\'')
    generate.add_argument("--train-size", type=float)
    generate.add_argument("--valid-size", type=float)
    generate.add_argument("--test-size", type=float)
    generate.add_argument("--source-repo", type=Path, help="Already pinned, clean upstream checkout (does not get modified)")

    validate = commands.add_parser("validate", help="Run dataset integrity and semantic validation")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--no-write-report", action="store_true", help="Do not update reports/validation.json")
    validate.add_argument("--json", action="store_true", help="Print machine-readable validation JSON")

    inspect = commands.add_parser("inspect", help="Print dataset, split, provenance, and class-support summary")
    inspect.add_argument("dataset", type=Path)
    inspect.add_argument("--json", action="store_true", help="Print machine-readable summary JSON")

    experiment = commands.add_parser("experiment", help="Historical classifier experiments (execution unavailable during classifier reconstruction)")
    experiment_commands = experiment.add_subparsers(dest="experiment_kind", required=True)
    fixed = experiment_commands.add_parser("fixed", help="Historical fixed-point classifier experiment (currently unavailable)")
    mapping = experiment_commands.add_parser("mapping", help="Historical synthetic point-mapping experiment (currently unavailable)")
    for command in (fixed, mapping):
        command.add_argument("--output", type=Path, required=True, help="Parent directory for immutable synthetic run groups")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--points", type=int, default=320)
        command.add_argument("--dimensions", type=int, default=2)
        command.add_argument("--steps", type=int, default=60)
        command.add_argument("--learning-rate", type=float, default=0.03)
        command.add_argument("--margin", type=float, default=0.1)
        command.add_argument("--window-target-volume", type=float, default=0.25)
        command.add_argument("--unknown-fraction", type=float, default=0.04)
        command.add_argument("--snapshot-interval", type=int, default=10)
    mapping.add_argument("--scenario", choices=("simple", "window"), default="window")

    image = experiment_commands.add_parser("image", help="Historical image classifier operations; only persisted run inspection remains available")
    image_commands = image.add_subparsers(dest="image_command", required=True)
    image_preflight = image_commands.add_parser("preflight", help="Validate an image classifier execution request (currently unavailable)")
    image_preflight.add_argument("--config", type=Path, required=True, help="JSON image experiment configuration")

    image_run = image_commands.add_parser("run", help="Train an image classifier (currently unavailable)")
    image_run.add_argument("--config", type=Path, required=True, help="JSON image experiment configuration")
    image_run.add_argument("--output", type=Path, required=True, help="Parent directory for immutable image runs")
    image_run.add_argument("--resume", action="store_true", help="Resume a stopped/interrupted/failed run with its exact config")

    image_matrix = image_commands.add_parser("matrix", help="Run an image classifier matrix (currently unavailable)")
    image_matrix.add_argument("--config", type=Path, required=True, help="JSON base configuration shared across all matrix cells")
    image_matrix.add_argument("--output", type=Path, required=True, help="Parent directory for immutable image runs and matrix manifest")
    image_matrix.add_argument("--families", nargs="+", choices=("baseline", "box", "town"), default=("baseline", "box", "town"))
    image_matrix.add_argument("--pizza-modes", nargs="+", choices=("global", "compositional", "hybrid"), default=("global", "compositional", "hybrid"))

    image_resume = image_commands.add_parser("resume", help="Resume historical image classifier training (currently unavailable)")
    image_resume.add_argument("run", type=Path)
    image_resume.add_argument("--config", type=Path, help="Optional JSON config override; identity must still match")
    image_evaluate = image_commands.add_parser("evaluate", help="Evaluate a historical image classifier checkpoint (currently unavailable)")
    image_evaluate.add_argument("run", type=Path)
    image_evaluate.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    image_evaluate.add_argument("--checkpoint", default="best", help="best, latest, or an explicit trusted checkpoint path")
    image_stop = image_commands.add_parser("stop", help="Control a historical image classifier worker (currently unavailable)")
    image_stop.add_argument("run", type=Path)
    image_inspect = image_commands.add_parser("inspect", help="Read JSON run lifecycle and metrics without loading checkpoint optimizer state")
    image_inspect.add_argument("run", type=Path)
    image_relations = image_commands.add_parser("relations", help="Interpret historical classifier geometry (currently unavailable)")
    image_relations.add_argument("snapshot", type=Path)
    image_relations.add_argument("--source-class-id", type=int, required=True)
    image_relations.add_argument("--target-class-id", type=int, required=True)
    image_relations.add_argument("--relation", choices=("containment", "intersection", "disjointness"), required=True)
    image_relations.add_argument("--sample-count", type=int, default=10000, help="Town draws per sampling stratum; box queries are exact")
    image_relations.add_argument("--seed", type=int, default=0)
    image_relations.add_argument("--output", type=Path, help="Optional immutable JSON result path")

    evaluate = commands.add_parser("evaluate", help="Evaluate a historical synthetic classifier checkpoint (currently unavailable)")
    evaluate.add_argument("run", type=Path)
    evaluate.add_argument("--split", choices=("train", "validation"), default="validation")
    evaluate.add_argument("--checkpoint", type=Path)

    resume = commands.add_parser("resume", help="Resume historical synthetic classifier training (currently unavailable)")
    resume.add_argument("run", type=Path)
    resume.add_argument("--additional-steps", type=int, required=True)

    inspect_run = commands.add_parser("inspect-run", help="Print resolved settings, metrics and environment for one saved model run")
    inspect_run.add_argument("run", type=Path)

    export = commands.add_parser("export-geometry", help="Export historical classifier geometry (currently unavailable)")
    export.add_argument("run", type=Path)
    export.add_argument("--snapshot", default="final", help="final, initial, or a saved step name such as step_00010")
    export.add_argument("--output", type=Path, help="Optional JSON output file (defaults to stdout)")

    dashboard = commands.add_parser("dashboard", help="Launch the local Streamlit research dashboard")
    dashboard.add_argument("--address", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.add_argument("--headless", action="store_true", help="Do not open a browser automatically")
    return parser


def _load_image_config(path: Path):
    from .experiments.image_config import image_config_from_mapping

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("image experiment config must be a JSON object")
    if "training_config" in payload:
        payload = payload["training_config"]
    if not isinstance(payload, dict):
        raise ValueError("persisted training_config must be a JSON object")
    return image_config_from_mapping(payload)


def _print_validation(report: object) -> None:
    assert hasattr(report, "to_dict")
    payload = report.to_dict()
    print(f"Validation: {'PASS' if payload['valid'] else 'FAIL'}")
    print(f"Dataset: {payload.get('dataset_id') or 'unknown'}")
    print(f"Errors: {payload['error_count']} | Warnings: {payload['warning_count']}")
    for issue in payload["errors"]:
        context = ", ".join(f"{key}={issue[key]}" for key in ("sample_id", "path", "field") if issue.get(key))
        print(f"ERROR [{issue['code']}] {issue['message']}" + (f" ({context})" if context else ""))
    shown_warnings = payload["warnings"][:20]
    for issue in shown_warnings:
        print(f"WARNING [{issue['code']}] {issue['message']}")
    if len(payload["warnings"]) > len(shown_warnings):
        print(f"... {len(payload['warnings']) - len(shown_warnings)} additional warnings; see reports/validation.json or use --json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            type_counts = None
            if args.type_counts is not None:
                type_counts = json.loads(args.type_counts)
                if not isinstance(type_counts, dict) or any(type(value) is not int for value in type_counts.values()):
                    raise ValueError("--type-counts must be a JSON object of integer counts")
            config = load_config(
                args.config,
                total_instances=args.total_instances,
                generation_seed=args.generation_seed,
                split_seed=args.split_seed,
                train_size=args.train_size,
                valid_size=args.valid_size,
                test_size=args.test_size,
                pizza_types=tuple(args.pizza_types) if args.pizza_types else None,
                type_counts=type_counts,
            )
            resolved = config.resolved_dict()
            print("Resolved generation settings:")
            print(json.dumps(resolved, indent=2, sort_keys=True))
            output = generate_dataset(
                config,
                args.output,
                source_repo=args.source_repo,
                progress=lambda fraction, message: print(f"[{fraction:5.1%}] {message}", flush=True),
            )
            print(f"Prepared dataset at {output}")
            return 0
        if args.command == "validate":
            report = validate_dataset_operation(args.dataset, write_report=not args.no_write_report)
            if args.json:
                print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
            else:
                _print_validation(report)
            return 0 if report.valid else 1
        if args.command == "inspect":
            summary = inspect_dataset(args.dataset)
            print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
            return 0 if summary.integrity_status == "current_pass" else 1
        if args.command == "experiment":
            repository_root = Path(__file__).resolve().parents[2]
            if args.experiment_kind in ("fixed", "mapping"):
                raise_classifier_backend_unavailable("synthetic classifier training")
            if args.experiment_kind == "image":
                unavailable_image_operations = {
                    "preflight": "image classifier experiment preflight",
                    "run": "image-to-ontology classifier training",
                    "matrix": "automatic image classifier matrix execution",
                    "resume": "image-to-ontology classifier training resume",
                    "evaluate": "historical image classifier checkpoint evaluation",
                    "relations": "historical ontology-region relation diagnosis",
                    "stop": "control a historical image classifier worker",
                }
                operation = unavailable_image_operations.get(args.image_command)
                if operation is not None:
                    raise_classifier_backend_unavailable(operation)
                from .dataset.image_data import deterministic_subset
                from .experiments.image_runs import (
                    inspect_run as inspect_image_run,
                    request_stop as request_image_stop,
                    resolve_image_experiment,
                    run_image_experiment,
                    run_image_matrix,
                )
                from .training.image_trainer import evaluate_image_checkpoint

                if args.image_command == "preflight":
                    image_config = _load_image_config(args.config)
                    resolved = resolve_image_experiment(image_config, repository_root=repository_root)
                    print(json.dumps({
                        "status": "preflight_passed",
                        "run_id": resolved.run_id,
                        "identity_hash": resolved.identity_hash,
                        "class_weight_vectors": resolved.class_weight_vectors(),
                        "training_protocol": "pre-detector exploratory GT-ROI; no run artifacts created",
                    }, indent=2, sort_keys=True))
                    return 0
                if args.image_command == "relations":
                    from town_model.dataset.ontology import OntologySnapshot
                    from town_model.inspection.image_snapshots import load_image_snapshot
                    from town_model.inspection.relation_diagnostics import diagnose_region_relation, region_from_snapshot, save_relation_diagnostic

                    snapshot = load_image_snapshot(args.snapshot)
                    ontology = OntologySnapshot.load(require_pinned_hash=True)
                    class_ids = tuple(snapshot.header["class_ids"])
                    if args.source_class_id not in class_ids or args.target_class_id not in class_ids:
                        raise ValueError("requested class IDs are not both present in the saved geometry snapshot")
                    source = region_from_snapshot(snapshot, class_ids.index(args.source_class_id), ontology=ontology)
                    target = region_from_snapshot(snapshot, class_ids.index(args.target_class_id), ontology=ontology)
                    result = diagnose_region_relation(
                        source, target, args.relation, sample_count=args.sample_count, seed=args.seed, ontology=ontology
                    )
                    if args.output:
                        save_relation_diagnostic(result, args.output)
                        print(json.dumps({"status": "saved", "path": str(args.output.resolve()), "result": result}, indent=2, sort_keys=True))
                    else:
                        print(json.dumps(result, indent=2, sort_keys=True))
                    return 0
                if args.image_command in ("run", "matrix"):
                    image_config = _load_image_config(args.config)
                    if args.image_command == "run":
                        run_path, result = run_image_experiment(
                            image_config, args.output, repository_root=repository_root, resume=args.resume
                        )
                        print(json.dumps({
                            "run_directory": str(run_path),
                            "run_id": run_path.name,
                            "status": result.status,
                            "optimizer_step": result.global_step,
                            "completed_epochs": result.completed_epochs,
                            "best_primary_monitor": result.best_primary_monitor,
                            "training_protocol": "pre-detector exploratory GT-ROI; not detector-pretrained Phase 3 acceptance evidence",
                        }, indent=2, sort_keys=True))
                    else:
                        matrix_path = run_image_matrix(
                            image_config,
                            args.output,
                            repository_root=repository_root,
                            families=args.families,
                            pizza_modes=args.pizza_modes,
                        )
                        print(json.dumps({"status": "completed", "matrix_manifest": str(matrix_path / "comparison.json")}, indent=2))
                    return 0
                if args.image_command == "resume":
                    if args.config:
                        image_config = _load_image_config(args.config)
                    else:
                        image_config = _load_image_config(args.run.expanduser().resolve() / "resolved-config.json")
                    run_path, result = run_image_experiment(
                        image_config,
                        args.run.expanduser().resolve().parent,
                        repository_root=repository_root,
                        resume=True,
                    )
                    if run_path != args.run.expanduser().resolve():
                        raise ValueError(f"resolved image run path does not match requested run: {run_path}")
                    print(json.dumps({"run_directory": str(run_path), "status": result.status, "optimizer_step": result.global_step}, indent=2))
                    return 0
                if args.image_command == "evaluate":
                    run_path = args.run.expanduser().resolve()
                    image_config = _load_image_config(run_path / "resolved-config.json")
                    resolved = resolve_image_experiment(image_config, repository_root=repository_root)
                    if resolved.run_id != run_path.name:
                        raise ValueError("saved image run identity does not match its resolved configuration")
                    if args.checkpoint == "best":
                        checkpoint = run_path / "checkpoints" / "best-validation.pt"
                    elif args.checkpoint == "latest":
                        checkpoint = run_path / "checkpoints" / "latest.pt"
                    else:
                        checkpoint = Path(args.checkpoint).expanduser()
                    if not checkpoint.is_absolute() and args.checkpoint not in ("best", "latest"):
                        checkpoint = run_path / checkpoint
                    if args.split == "train":
                        evaluation_subset = resolved.train_subset
                    elif args.split == "validation":
                        evaluation_subset = resolved.validation_subset
                    else:
                        evaluation_subset = deterministic_subset(
                            resolved.dataset, "test", size=None, seed=image_config.dataset.selection_seed + 2
                        )
                    result = evaluate_image_checkpoint(
                        resolved.dataset,
                        resolved.train_subset,
                        resolved.validation_subset,
                        evaluation_subset,
                        image_config,
                        checkpoint,
                        class_ids=resolved.class_ids,
                    )
                    print(json.dumps(result, indent=2, sort_keys=True))
                    return 0
                if args.image_command == "stop":
                    request_image_stop(args.run)
                    print(json.dumps({"run_directory": str(args.run.resolve()), "stop_requested": True}, indent=2))
                    return 0
                if args.image_command == "inspect":
                    manifest = inspect_image_run(args.run)
                    metrics_path = args.run.expanduser().resolve() / "metrics.json"
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
                    print(json.dumps({"manifest": manifest, "metrics": metrics}, indent=2, sort_keys=True))
                    return 0
            from .experiments.synthetic import SyntheticConfig, run_experiment, run_fixed_experiments

            config = SyntheticConfig(
                seed=args.seed,
                points=args.points,
                dimensions=args.dimensions,
                steps=args.steps,
                learning_rate=args.learning_rate,
                margin=args.margin,
                window_target_volume=args.window_target_volume,
                unknown_fraction=args.unknown_fraction,
                snapshot_interval=args.snapshot_interval,
            )
            if args.experiment_kind == "fixed":
                groups = run_fixed_experiments(args.output, config=config)
            else:
                groups = [run_experiment("mapping", args.scenario, args.output, config=config)]
            print(json.dumps({"status": "completed", "run_groups": [str(path) for path in groups]}, indent=2))
            return 0
        if args.command == "evaluate":
            raise_classifier_backend_unavailable("synthetic classifier evaluation")
            from .experiments.synthetic import evaluate_saved_run

            result = evaluate_saved_run(args.run, args.split, checkpoint=args.checkpoint)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "resume":
            raise_classifier_backend_unavailable("synthetic classifier training resume")
            from .experiments.synthetic import resume_saved_run

            metrics = resume_saved_run(args.run, args.additional_steps)
            print(json.dumps({"run": str(args.run.resolve()), "step": metrics["history"][-1]["step"], "resume_segments": metrics.get("resume_segments", [])}, indent=2))
            return 0
        if args.command == "inspect-run":
            from .experiments.synthetic import inspect_saved_run

            print(json.dumps(inspect_saved_run(args.run), indent=2, sort_keys=True))
            return 0
        if args.command == "export-geometry":
            raise_classifier_backend_unavailable("synthetic classifier geometry export")
            from .experiments.synthetic import saved_geometry

            snapshot = saved_geometry(args.run, args.snapshot)
            model = snapshot["model"]
            output = snapshot["output"]
            export_payload = {
                "run": str(args.run.resolve()),
                "family": model.family,
                "class_names": list(model.class_names),
                "snapshot": snapshot["name"],
                "step": snapshot["step"],
                "fixed_positive_classes": list(model.fixed_classes),
                "point_coordinates": snapshot["points"].detach().cpu().tolist(),
                "predictions": output.contains.detach().cpu().tolist(),
                "signed_margin": output.signed_margin.detach().cpu().tolist() if output.signed_margin is not None else None,
                "logits": output.logits.detach().cpu().tolist() if output.logits is not None else None,
            }
            if model.family == "box":
                low, high = model.geometry.physical_boxes()
                export_payload["geometry"] = {"box_low": low.detach().cpu().tolist(), "box_high": high.detach().cpu().tolist()}
            elif model.family == "town":
                low, high, window_low, window_high = model.geometry.physical_geometry()
                export_payload["geometry"] = {
                    "frame_low": low.detach().cpu().tolist(), "frame_high": high.detach().cpu().tolist(),
                    "window_low": window_low.detach().cpu().tolist(), "window_high": window_high.detach().cpu().tolist(),
                    "window_mode": model.geometry.window_mode,
                }
            else:
                export_payload["geometry"] = {
                    "weight": model.linear.weight.detach().cpu().tolist(),
                    "bias": model.linear.bias.detach().cpu().tolist(),
                    "decision_boundary": "actual logit == 0",
                }
            serialized = json.dumps(export_payload, indent=2, sort_keys=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized + "\n", encoding="utf-8")
                print(f"Exported actual saved geometry to {args.output}")
            else:
                print(serialized)
            return 0
        if args.command == "dashboard":
            app_path = files("town_model.dashboard").joinpath("app.py")
            command = [
                sys.executable, "-m", "streamlit", "run", str(app_path),
                "--server.address", args.address,
                "--server.port", str(args.port),
                "--server.headless", "true" if args.headless else "false",
            ]
            return subprocess.call(command)
        parser.error(f"Unknown command {args.command!r}")
    except (FileExistsError, FileNotFoundError, NotADirectoryError, RuntimeError, ValueError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
