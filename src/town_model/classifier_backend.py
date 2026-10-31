"""Explicit failure for historical classifier-dependent operations."""

from __future__ import annotations

from typing import NoReturn


class ClassifierBackendUnavailableError(RuntimeError):
    """Raised when an operation requires a classifier backend that is not approved."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(
            f"{operation} is unavailable during classifier reconstruction. "
            "Historical classifier execution has been intentionally disconnected, "
            "and no reconstructed classifier backend is currently approved. "
            "This operation cannot proceed until the author's model reconstruction "
            "is implemented and accepted."
        )


def raise_classifier_backend_unavailable(operation: str) -> NoReturn:
    """Fail closed for an operation that would require historical classifier execution."""
    raise ClassifierBackendUnavailableError(operation)
