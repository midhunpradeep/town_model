"""Explicit unavailable boundary for the retired model-dependent image worker."""

from __future__ import annotations

import sys
from typing import NoReturn, Sequence

from town_model.classifier_backend import (
    ClassifierBackendUnavailableError,
    raise_classifier_backend_unavailable,
)


def run_worker(argv: Sequence[str] | None = None, **kwargs) -> NoReturn:
    """Reject worker execution before parsing arguments or touching status files."""
    del argv, kwargs
    raise_classifier_backend_unavailable("image classifier background worker execution")


def main() -> int:
    try:
        run_worker()
    except ClassifierBackendUnavailableError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
