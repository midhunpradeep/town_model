"""Validated generation and split settings shared by the CLI and dashboard."""

from __future__ import annotations

import json
import math
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_PIZZA_TYPES = (
    "American",
    "AmericanHot",
    "Cajun",
    "Capricciosa",
    "Fiorentina",
    "FourSeasons",
    "Giardiniera",
    "LaReine",
    "Mushroom",
    "Napoletana",
    "Parmense",
    "Siciliana",
    "SloppyGiuseppe",
    "Soho",
    "Veneziana",
    "random",
)
DEFAULT_GENERATION_SEED = 42
DEFAULT_SPLIT_SEED = 42
DEFAULT_SPLIT_RATIOS = (0.6, 0.2, 0.2)
DEFAULT_CONFIG_PATH = Path("dataset.toml")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Complete, validated inputs to deterministic image generation and splitting."""

    total_instances: int | None = None
    pizza_types: tuple[str, ...] = DEFAULT_PIZZA_TYPES
    type_counts: Mapping[str, int] | None = None
    generation_seed: int = DEFAULT_GENERATION_SEED
    split_seed: int = DEFAULT_SPLIT_SEED
    train_size: float = DEFAULT_SPLIT_RATIOS[0]
    valid_size: float = DEFAULT_SPLIT_RATIOS[1]
    test_size: float = DEFAULT_SPLIT_RATIOS[2]

    def validate(self, *, require_total: bool = False) -> None:
        total = self.total_instances
        if total is None:
            if require_total:
                raise ValueError("total_instances must be set before generation")
        elif isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise ValueError("total_instances must be a positive integer")

        if not self.pizza_types:
            raise ValueError("Select at least one Pizzaïolo generator type")
        if len(set(self.pizza_types)) != len(self.pizza_types):
            raise ValueError("pizza_types must not contain duplicates")
        unknown = set(self.pizza_types) - set(DEFAULT_PIZZA_TYPES)
        if unknown:
            raise ValueError(f"Unsupported Pizzaïolo generator types: {sorted(unknown)}")

        for name, seed, maximum in (
            ("generation_seed", self.generation_seed, 2**31 - 1),
            ("split_seed", self.split_seed, 2**32 - 1),
        ):
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= maximum:
                raise ValueError(f"{name} must be an integer from 0 through {maximum}")

        ratios = (self.train_size, self.valid_size, self.test_size)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in ratios
        ):
            raise ValueError("Split ratios must be finite numbers")
        if any(value <= 0 for value in ratios):
            raise ValueError("All split ratios must be positive")
        if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("train_size + valid_size + test_size must sum to 1")

        if self.type_counts is not None:
            unknown_counts = set(self.type_counts) - set(self.pizza_types)
            if unknown_counts:
                raise ValueError(
                    "Counts were supplied for unselected pizza types: "
                    f"{sorted(unknown_counts)}"
                )
            for name, count in self.type_counts.items():
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(f"Count for {name!r} must be a nonnegative integer")

    def allocate_type_counts(self) -> dict[str, int]:
        """Return the exact resolved per-type counts used by the pinned generator."""

        self.validate(require_total=True)
        assert self.total_instances is not None
        if self.type_counts is not None:
            counts = {name: self.type_counts.get(name, 0) for name in self.pizza_types}
            if sum(counts.values()) != self.total_instances:
                raise ValueError(
                    "pizza_type_counts must sum to total_instances "
                    f"({sum(counts.values())} != {self.total_instances})"
                )
        else:
            each, remainder = divmod(self.total_instances, len(self.pizza_types))
            counts = {name: each for name in self.pizza_types}
            remainder_order = list(self.pizza_types)
            random.Random(self.generation_seed).shuffle(remainder_order)
            for name in remainder_order[:remainder]:
                counts[name] += 1

        too_small = {name: count for name, count in counts.items() if 0 < count < 3}
        if too_small:
            raise ValueError(
                "Every generated type with a positive count needs at least three "
                "samples for the type-stratified train/validation/test split; "
                f"got {too_small}"
            )
        if not any(counts.values()):
            raise ValueError("At least one selected generator type needs a positive count")
        return counts

    def resolved_dict(self) -> dict[str, Any]:
        """Return JSON-ready configuration with allocation resolved and explicit."""

        counts = self.allocate_type_counts()
        return {
            "total_instances": self.total_instances,
            "pizza_types": list(self.pizza_types),
            "pizza_type_counts": counts,
            "generation_seed": self.generation_seed,
            "split_seed": self.split_seed,
            "split_ratios": {
                "train": self.train_size,
                "validation": self.valid_size,
                "test": self.test_size,
            },
        }


def load_config(
    path: Path,
    *,
    total_instances: int | None = None,
    generation_seed: int | None = None,
    split_seed: int | None = None,
    train_size: float | None = None,
    valid_size: float | None = None,
    test_size: float | None = None,
    pizza_types: tuple[str, ...] | None = None,
    type_counts: Mapping[str, int] | None = None,
) -> DatasetConfig:
    """Load the shared TOML config, then apply non-None caller overrides."""

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("rb") as file:
        raw = tomllib.load(file)

    allowed = {
        "total_instances",
        "pizza_types",
        "generation_seed",
        "split_seed",
        "train_size",
        "valid_size",
        "test_size",
        "pizza_type_counts",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown dataset config settings: {sorted(unknown)}")

    raw_types = raw.get("pizza_types", list(DEFAULT_PIZZA_TYPES))
    if not isinstance(raw_types, list) or not all(isinstance(value, str) for value in raw_types):
        raise ValueError("pizza_types must be an array of strings")
    raw_counts = raw.get("pizza_type_counts")
    if raw_counts is not None and not isinstance(raw_counts, dict):
        raise ValueError("[pizza_type_counts] must be a TOML table")

    config = DatasetConfig(
        total_instances=(total_instances if total_instances is not None else raw.get("total_instances")),
        pizza_types=(pizza_types if pizza_types is not None else tuple(raw_types)),
        type_counts=(type_counts if type_counts is not None else raw_counts),
        generation_seed=(generation_seed if generation_seed is not None else raw.get("generation_seed", DEFAULT_GENERATION_SEED)),
        split_seed=(split_seed if split_seed is not None else raw.get("split_seed", DEFAULT_SPLIT_SEED)),
        train_size=(train_size if train_size is not None else raw.get("train_size", DEFAULT_SPLIT_RATIOS[0])),
        valid_size=(valid_size if valid_size is not None else raw.get("valid_size", DEFAULT_SPLIT_RATIOS[1])),
        test_size=(test_size if test_size is not None else raw.get("test_size", DEFAULT_SPLIT_RATIOS[2])),
    )
    config.validate()
    return config


def config_from_overrides(
    *,
    total_instances: int,
    generation_seed: int = DEFAULT_GENERATION_SEED,
    split_seed: int = DEFAULT_SPLIT_SEED,
    pizza_types: tuple[str, ...] = DEFAULT_PIZZA_TYPES,
    train_size: float = DEFAULT_SPLIT_RATIOS[0],
    valid_size: float = DEFAULT_SPLIT_RATIOS[1],
    test_size: float = DEFAULT_SPLIT_RATIOS[2],
    type_counts: Mapping[str, int] | None = None,
) -> DatasetConfig:
    config = DatasetConfig(
        total_instances=total_instances,
        pizza_types=pizza_types,
        type_counts=type_counts,
        generation_seed=generation_seed,
        split_seed=split_seed,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
    )
    config.validate(require_total=True)
    config.allocate_type_counts()
    return config


def config_json(config: DatasetConfig) -> str:
    return json.dumps(config.resolved_dict(), indent=2, sort_keys=True)
