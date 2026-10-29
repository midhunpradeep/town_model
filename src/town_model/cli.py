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
from .config import DEFAULT_CONFIG_PATH, DEFAULT_PIZZA_TYPES, load_config
from .experiments.synthetic import (
    SyntheticConfig,
    evaluate_saved_run,
    inspect_saved_run,
    run_experiment,
    run_fixed_experiments,
    resume_saved_run,
    saved_geometry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="town",
        description="Prepare Pizzaïolo datasets and run controlled synthetic town-embedding experiments.",
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

    experiment = commands.add_parser("experiment", help="Run paired synthetic Phase 2 experiments (all three classifiers)")
    experiment_commands = experiment.add_subparsers(dest="experiment_kind", required=True)
    fixed = experiment_commands.add_parser("fixed", help="Fixed-point classifier-only experiment; runs simple and nonconvex fixtures")
    mapping = experiment_commands.add_parser("mapping", help="Jointly train matched synthetic point encoders and classifiers")
    for command in (fixed, mapping):
        command.add_argument("--output", type=Path, required=True, help="Parent directory for immutable synthetic run groups")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--points", type=int, default=320)
        command.add_argument("--dimensions", type=int, default=2)
        command.add_argument("--steps", type=int, default=60)
        command.add_argument("--learning-rate", type=float, default=0.03)
        command.add_argument("--margin", type=float, default=0.1)
        command.add_argument("--unknown-fraction", type=float, default=0.04)
        command.add_argument("--snapshot-interval", type=int, default=10)
    mapping.add_argument("--scenario", choices=("simple", "window"), default="window")

    evaluate = commands.add_parser("evaluate", help="Evaluate a saved Phase 2 checkpoint without training")
    evaluate.add_argument("run", type=Path)
    evaluate.add_argument("--split", choices=("train", "validation"), default="validation")
    evaluate.add_argument("--checkpoint", type=Path)

    resume = commands.add_parser("resume", help="Resume a saved Adam checkpoint with its immutable resolved configuration")
    resume.add_argument("run", type=Path)
    resume.add_argument("--additional-steps", type=int, required=True)

    inspect_run = commands.add_parser("inspect-run", help="Print resolved settings, metrics and environment for one saved model run")
    inspect_run.add_argument("run", type=Path)

    export = commands.add_parser("export-geometry", help="Export actual saved class geometry and predictions as JSON")
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
            config = SyntheticConfig(
                seed=args.seed,
                points=args.points,
                dimensions=args.dimensions,
                steps=args.steps,
                learning_rate=args.learning_rate,
                margin=args.margin,
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
            result = evaluate_saved_run(args.run, args.split, checkpoint=args.checkpoint)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "resume":
            metrics = resume_saved_run(args.run, args.additional_steps)
            print(json.dumps({"run": str(args.run.resolve()), "step": metrics["history"][-1]["step"], "resume_segments": metrics.get("resume_segments", [])}, indent=2))
            return 0
        if args.command == "inspect-run":
            print(json.dumps(inspect_saved_run(args.run), indent=2, sort_keys=True))
            return 0
        if args.command == "export-geometry":
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
