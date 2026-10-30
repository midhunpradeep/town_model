"""Detached command-line worker for dashboard-launched image experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import traceback
from typing import Callable, Sequence

from town_model.cli import main as cli_main
from town_model.experiments.image_runs import _atomic_json, _process_start_token


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="town-image-worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--reservation-fd", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def run_worker(argv: Sequence[str] | None = None, *, command_main: Callable[[Sequence[str]], int] = cli_main) -> int:
    """Run the same validated CLI command and persist its subprocess outcome."""
    args = _parser().parse_args(argv)
    try:
        current = json.loads(args.status.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current = {}
    current.update({
        "state": "running",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "process_start_token": _process_start_token(os.getpid()),
        "started_at": _now(),
    })
    _atomic_json(args.status, current)
    command = [
        "experiment", "image", "run",
        "--config", str(args.config),
        "--output", str(args.output),
    ]
    if args.resume:
        command.append("--resume")

    error_message = None
    try:
        code = command_main(command)
        code = int(code or 0)
    except SystemExit as error:
        code = int(error.code or 0) if isinstance(error.code, (int, type(None))) else 1
    except BaseException as error:  # status must explain worker failures, including unexpected exceptions
        code = 1
        error_message = f"{type(error).__name__}: {error}"
        traceback.print_exc(file=sys.stderr)
    finally:
        current.update({
            "state": "completed" if code == 0 else "failed",
            "exit_code": code,
            "exit_status": "observed wrapper/CLI exit code",
            "error": error_message,
            "finished_at": _now(),
        })
        _atomic_json(args.status, current)
        if args.reservation_fd is not None:
            try:
                os.close(args.reservation_fd)
            except OSError:
                pass
    return code


def main() -> int:
    return run_worker()


if __name__ == "__main__":
    raise SystemExit(main())
