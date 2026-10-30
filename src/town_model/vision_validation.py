"""Pure MobileNet/FPN configuration checks shared by preflight and construction."""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version

import torch


def validate_mobilenet_runtime() -> None:
    """Check pinned adapter prerequisites without constructing Torchvision models."""
    torch_version = torch.__version__.split("+")[0]
    try:
        torchvision_version = version("torchvision").split("+")[0]
    except PackageNotFoundError as error:
        raise ValueError("Phase 3 requires the pinned torchvision 0.20.1 package") from error
    if torch_version != "2.5.1" or torchvision_version != "0.20.1":
        raise ValueError(
            "the Phase 3 visual adapter requires torch 2.5.1 and torchvision 0.20.1; "
            f"found torch {torch.__version__} and torchvision {torchvision_version}"
        )


def validate_mobilenet_block_indices(blocks: Iterable[object], *, maximum: int = 16) -> tuple[int, ...]:
    """Return canonical block indices, rejecting coercion and duplicate entries."""
    if isinstance(blocks, (str, bytes)):
        raise ValueError("trainable MobileNet blocks must be an iterable of integer feature indices")
    try:
        values = tuple(blocks)
    except TypeError as error:
        raise ValueError("trainable MobileNet blocks must be an iterable of integer feature indices") from error
    if any(type(index) is not int or index < 0 or index > maximum for index in values):
        raise ValueError(f"trainable MobileNet blocks must be integer feature indices in [0,{maximum}]")
    if len(set(values)) != len(values):
        raise ValueError("trainable MobileNet block indices must be unique")
    return tuple(sorted(values))
