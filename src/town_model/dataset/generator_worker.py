"""Subprocess entry point isolates upstream's process-wide working directory."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from ..config import DatasetConfig
from .generator import _load_upstream_class, seed_generation, working_directory


def _stable_candidate_key(candidate: Any) -> str:
    return str(getattr(candidate, "iri", None) or getattr(candidate, "name", candidate))


def _install_stable_random_candidates(generator_class: type) -> None:
    """Canonicalize unordered ontology candidates before seeded random choice.

    The pinned generator uses ``choice(PizzaBase.descendants())`` where
    Owlready returns a set, and ``choices`` over an ontology-backed collection.
    Set/proxy iteration can vary between processes even with identical seeds.
    This worker-local adapter sorts those candidates by ontology IRI while
    preserving the upstream selection and rendering algorithms.
    """

    module = sys.modules[generator_class.__module__]

    def stable_choice(candidates: Any) -> Any:
        return random.choice(sorted(candidates, key=_stable_candidate_key))

    def stable_choices(
        population: Any,
        weights: Any = None,
        *,
        cum_weights: Any = None,
        k: int = 1,
    ) -> list[Any]:
        if weights is not None or cum_weights is not None:
            raise ValueError("Pinned Pizzaïolo random choices do not use weighted populations")
        return random.choices(sorted(population, key=_stable_candidate_key), k=k)

    module.choice = stable_choice
    module.choices = stable_choices

    # The upstream overlap check iterates a set while also deriving list order
    # from it. Feed it a canonical sequence so retry decisions consume the
    # seeded random stream identically in separate worker processes.
    original_overlap_rate = generator_class._get_overlapping_rate

    def stable_overlap_rate(instance: Any, toppings: Any, boxes: Any) -> float:
        ordered_toppings = sorted(toppings, key=_stable_candidate_key)
        return original_overlap_rate(instance, ordered_toppings, boxes)

    generator_class._get_overlapping_rate = stable_overlap_rate


def generate(source: Path, destination: Path, payload: dict[str, Any]) -> None:
    config = DatasetConfig(
        total_instances=int(payload["total_instances"]),
        pizza_types=tuple(payload["pizza_types"]),
        type_counts=payload["pizza_type_counts"],
        generation_seed=int(payload["generation_seed"]),
        split_seed=int(payload["split_seed"]),
        train_size=float(payload["split_ratios"]["train"]),
        valid_size=float(payload["split_ratios"]["validation"]),
        test_size=float(payload["split_ratios"]["test"]),
    )
    counts = config.allocate_type_counts()
    seed_generation(config.generation_seed)
    Pizzaiolo = _load_upstream_class(source)
    _install_stable_random_candidates(Pizzaiolo)
    with working_directory(source):
        generator = Pizzaiolo()
        ontology = generator.getOntology()
        type_sequence: list[object] = []
        for name, count in counts.items():
            pizza_type = None if name == "random" else getattr(ontology, name)
            type_sequence.extend([pizza_type] * count)
        random.Random(config.generation_seed).shuffle(type_sequence)
        generator.cook(type_sequence, number_of_each=1, delivery_dir=str(destination))


def main() -> int:
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--config-json", required=True)
    arguments = parser.parse_args()
    payload = json.loads(arguments.config_json)
    generate(arguments.source.resolve(), arguments.destination.resolve(), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
