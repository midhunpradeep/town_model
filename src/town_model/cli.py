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
    fixed = experiment_commands.add_parser("fixed", help="Unavailable historical fixed-point classifier operation")
    mapping = experiment_commands.add_parser("mapping", help="Unavailable historical synthetic point-mapping operation")
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
    image_preflight = image_commands.add_parser("preflight", help="Unavailable image classifier execution preflight")
    image_preflight.add_argument("--config", type=Path, required=True, help="JSON image experiment configuration")

    image_run = image_commands.add_parser("run", help="Unavailable historical image classifier training operation")
    image_run.add_argument("--config", type=Path, required=True, help="JSON image experiment configuration")
    image_run.add_argument("--output", type=Path, required=True, help="Parent directory for immutable image runs")
    image_run.add_argument("--resume", action="store_true", help="Resume a stopped/interrupted/failed run with its exact config")

    image_matrix = image_commands.add_parser("matrix", help="Unavailable historical image classifier matrix operation")
    image_matrix.add_argument("--config", type=Path, required=True, help="JSON base configuration shared across all matrix cells")
    image_matrix.add_argument("--output", type=Path, required=True, help="Parent directory for immutable image runs and matrix manifest")
    image_matrix.add_argument("--families", nargs="+", choices=("baseline", "box", "town"), default=("baseline", "box", "town"))
    image_matrix.add_argument("--pizza-modes", nargs="+", choices=("global", "compositional", "hybrid"), default=("global", "compositional", "hybrid"))

    image_resume = image_commands.add_parser("resume", help="Unavailable historical image classifier resume operation")
    image_resume.add_argument("run", type=Path)
    image_resume.add_argument("--config", type=Path, help="Optional JSON config override; identity must still match")
    image_evaluate = image_commands.add_parser("evaluate", help="Unavailable historical image classifier checkpoint evaluation")
    image_evaluate.add_argument("run", type=Path)
    image_evaluate.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    image_evaluate.add_argument("--checkpoint", default="best", help="best, latest, or an explicit trusted checkpoint path")
    image_stop = image_commands.add_parser("stop", help="Unavailable control for a historical image classifier worker")
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

    evaluate = commands.add_parser("evaluate", help="Unavailable historical synthetic classifier checkpoint evaluation")
    evaluate.add_argument("run", type=Path)
    evaluate.add_argument("--split", choices=("train", "validation"), default="validation")
    evaluate.add_argument("--checkpoint", type=Path)

    resume = commands.add_parser("resume", help="Unavailable historical synthetic classifier resume operation")
    resume.add_argument("run", type=Path)
    resume.add_argument("--additional-steps", type=int, required=True)

    inspect_run = commands.add_parser("inspect-run", help="Read stored JSON metadata for a historical synthetic run")
    inspect_run.add_argument("run", type=Path)

    export = commands.add_parser("export-geometry", help="Unavailable historical classifier geometry export")
    export.add_argument("run", type=Path)
    export.add_argument("--snapshot", default="final", help="final, initial, or a saved step name such as step_00010")
    export.add_argument("--output", type=Path, help="Optional JSON output file (defaults to stdout)")

    dashboard = commands.add_parser("dashboard", help="Launch the local Streamlit research dashboard")
    dashboard.add_argument("--address", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.add_argument("--headless", action="store_true", help="Do not open a browser automatically")
    return parser


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
            if args.experiment_kind in ("fixed", "mapping"):
                raise_classifier_backend_unavailable("synthetic classifier training")
            if args.experiment_kind == "image":
                if args.image_command == "inspect":
                    from .experiments.image_runs import inspect_run

                    manifest = inspect_run(args.run)
                    metrics_path = args.run.expanduser().resolve() / "metrics.json"
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
                    print(json.dumps({"manifest": manifest, "metrics": metrics}, indent=2, sort_keys=True))
                    return 0
                operations = {
                    "preflight": "image classifier experiment preflight",
                    "run": "image-to-ontology classifier training",
                    "matrix": "automatic image classifier matrix execution",
                    "resume": "image-to-ontology classifier training resume",
                    "evaluate": "historical image classifier checkpoint evaluation",
                    "relations": "historical ontology-region relation diagnosis",
                    "stop": "control a historical image classifier worker",
                }
                raise_classifier_backend_unavailable(operations[args.image_command])
        if args.command == "evaluate":
            raise_classifier_backend_unavailable("synthetic classifier evaluation")
        if args.command == "resume":
            raise_classifier_backend_unavailable("synthetic classifier training resume")
        if args.command == "inspect-run":
            from .inspection.synthetic_metadata import inspect_saved_run

            print(json.dumps(inspect_saved_run(args.run), indent=2, sort_keys=True))
            return 0
        if args.command == "export-geometry":
            raise_classifier_backend_unavailable("synthetic classifier geometry export")
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
