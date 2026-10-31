"""Read-only JSON metadata inspection for historical synthetic runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def inspect_saved_run(run_path: Path | str) -> dict[str, Any]:
    """Read stored config/metrics/environment JSON without loading checkpoints."""
    run_dir = Path(run_path).expanduser().resolve()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    return {"path": str(run_dir), "config": config, "metrics": metrics, "environment": environment}
