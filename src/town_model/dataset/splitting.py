"""Deterministic generator-type stratification adapted from fourth_attempt."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from ..config import DatasetConfig


def assign_splits(
    rows: list[dict[str, Any]],
    *,
    train_size: float = 0.6,
    valid_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, str]:
    """Return sample-id to split using the established two-stage type policy.

    Per type, the first seeded shuffle allocates training samples; a second,
    independent shuffle divides the remainder between validation and test.
    Rounding and RNG draws intentionally match the tested fourth_attempt policy.
    """

    ratios = (train_size, valid_size, test_size)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in ratios):
        raise ValueError("Split ratios must be finite numbers")
    if any(value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Split ratios must be positive and sum to 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("split seed must be an integer from 0 through 2**32 - 1")

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_ids: set[str] = set()
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in all_ids:
            raise ValueError(f"Duplicate sample ID before split: {sample_id}")
        all_ids.add(sample_id)
        by_type[str(row["generator_type"])].append(row)

    rng = np.random.RandomState(seed)
    assignment: dict[str, str] = {}
    validation_fraction = valid_size / (valid_size + test_size)
    for generator_type, members in by_type.items():
        count = len(members)
        if count < 3:
            raise ValueError(
                f"Generator type {generator_type!r} has {count} samples; at least "
                "three are required for train, validation, and test"
            )
        train_count = math.floor(train_size * count)
        remaining_count = count - train_count
        valid_count = math.floor(validation_fraction * remaining_count)
        test_count = remaining_count - valid_count
        if min(train_count, valid_count, test_count) <= 0:
            raise ValueError(
                f"Generator type {generator_type!r} cannot appear in every split "
                f"with {count} rows and ratios {ratios}"
            )

        train_state = int(rng.randint(0, 2**31 - 1))
        permutation = np.random.RandomState(train_state).permutation(count)
        train_indices = permutation[remaining_count:]
        remaining_indices = permutation[:remaining_count]
        for index in train_indices:
            assignment[str(members[int(index)]["sample_id"])] = "train"

        valid_state = int(rng.randint(0, 2**31 - 1))
        remaining_permutation = np.random.RandomState(valid_state).permutation(remaining_count)
        test_indices = remaining_indices[remaining_permutation[:test_count]]
        valid_indices = remaining_indices[remaining_permutation[test_count:]]
        for index in valid_indices:
            assignment[str(members[int(index)]["sample_id"])] = "validation"
        for index in test_indices:
            assignment[str(members[int(index)]["sample_id"])] = "test"

    if set(assignment) != all_ids:
        raise AssertionError("Split assignments do not partition the generated samples")
    split_sets = {
        name: {sample_id for sample_id, split in assignment.items() if split == name}
        for name in ("train", "validation", "test")
    }
    if any(split_sets[left] & split_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Generated split assignments overlap")
    return assignment


def split_config(config: DatasetConfig) -> dict[str, float | int]:
    config.validate(require_total=True)
    return {
        "seed": config.split_seed,
        "train_size": config.train_size,
        "valid_size": config.valid_size,
        "test_size": config.test_size,
    }
