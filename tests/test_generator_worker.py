from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from town_model.dataset.generator_worker import _install_stable_random_candidates


def test_worker_canonicalizes_unordered_upstream_random_candidates(monkeypatch) -> None:
    module = ModuleType("fake_pizzaiolo_module")

    class Candidate:
        def __init__(self, iri: str) -> None:
            self.iri = iri
            self.name = iri.rsplit("#", 1)[-1]

    class Generator:
        def _get_overlapping_rate(self, toppings: Any, boxes: Any) -> list[str]:
            return [item.iri for item in toppings]

    Generator.__module__ = module.__name__
    module.choice = lambda candidates: None
    module.choices = lambda population, k=1: []
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr("town_model.dataset.generator_worker.random.choice", lambda population: population[0])
    monkeypatch.setattr(
        "town_model.dataset.generator_worker.random.choices",
        lambda population, k=1: list(population)[:k],
    )

    _install_stable_random_candidates(Generator)
    candidates = {Candidate("urn:test#z"), Candidate("urn:test#a"), Candidate("urn:test#m")}
    assert module.choice(candidates).iri == "urn:test#a"
    assert [item.iri for item in module.choices(candidates, k=3)] == [
        "urn:test#a", "urn:test#m", "urn:test#z",
    ]
    assert Generator()._get_overlapping_rate(candidates, {}) == [
        "urn:test#a", "urn:test#m", "urn:test#z",
    ]
