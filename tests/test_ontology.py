from __future__ import annotations

import json

import pytest

from town_model.dataset.constants import (
    DETECTOR_CATEGORIES,
    EXPECTED_OWL_CLASS_COUNT,
    EXPECTED_SUPERVISED_CLASS_COUNT,
    EXPECTED_TRAINABLE_CLASS_COUNT,
    FLAG_INDIVIDUALS,
    IMAGE_LABEL_CLASSES,
    UNSUPPORTED_CLASS_NAMES,
)
from town_model.dataset.ontology import (
    OntologySnapshot,
    build_class_vocabulary,
    build_detector_vocabulary,
)


def test_canonical_vocabulary_is_deterministic_and_keeps_class_inventory() -> None:
    ontology = OntologySnapshot.load(require_pinned_hash=True)
    first = build_class_vocabulary(ontology)
    second = build_class_vocabulary(ontology)
    assert first == second
    assert len(first["classes"]) == EXPECTED_OWL_CLASS_COUNT == 66
    assert len(first["trainable_class_ids"]) == EXPECTED_TRAINABLE_CLASS_COUNT == 60
    assert len(first["unsupported_class_ids"]) == 5
    assert len(IMAGE_LABEL_CLASSES) == 32
    assert UNSUPPORTED_CLASS_NAMES == {"ValuePartition", "Spiciness", "Hot", "Medium", "Mild"}
    ids = [item["class_id"] for item in first["classes"]]
    assert ids == list(range(66))
    assert [item["iri"] for item in first["classes"]] == sorted(item["iri"] for item in first["classes"])
    assert sum(item["trainable"] or item["fixed_positive"] for item in first["classes"]) == EXPECTED_SUPERVISED_CLASS_COUNT
    by_name = {item["name"]: item for item in first["classes"]}
    assert by_name["Pizza"]["pizza_supervision"] and not by_name["Pizza"]["component_supervision"]
    assert not by_name["Country"]["pizza_supervision"] and by_name["Country"]["component_supervision"]
    assert by_name["DomainThing"]["pizza_supervision"] and by_name["DomainThing"]["component_supervision"]


def test_concrete_component_targets_include_self_ancestors_and_equivalent_memberships() -> None:
    ontology = OntologySnapshot.load()
    assert ontology.component_class_targets("HamTopping") == {
        "HamTopping", "MeatTopping", "PizzaTopping", "Food", "DomainThing"
    }
    assert ontology.component_class_targets("DeepPanBase") == {
        "DeepPanBase", "PizzaBase", "Food", "DomainThing"
    }
    assert ontology.component_class_targets("JalapenoPepperTopping") == {
        "JalapenoPepperTopping", "PepperTopping", "VegetableTopping", "VegetarianTopping",
        "SpicyTopping", "PizzaTopping", "Food", "DomainThing",
    }
    for concrete in (
        "AnchovyTopping", "GarlicTopping", "GreenPepperTopping", "HamTopping",
        "JalapenoPepperTopping", "MushroomTopping", "OliveTopping", "OnionTopping",
        "ParmesanTopping", "PeperonataTopping", "PeperoniSausageTopping", "PrawnTopping",
        "RocketTopping", "SlicedTomatoTopping", "SpinachTopping", "SultanaTopping",
        "DeepPanBase", "ThinAndCrispyBase",
    ):
        assert concrete in ontology.component_class_targets(concrete)


def test_country_names_are_individuals_not_classes_or_detector_ontology_targets() -> None:
    ontology = OntologySnapshot.load()
    assert ontology.individuals == {"America", "England", "France", "Germany", "Italy"}
    assert not ontology.individuals.intersection(ontology.class_names)
    for flag in FLAG_INDIVIDUALS:
        assert ontology.component_class_targets(flag) == {"Country", "DomainThing"}
    detector = build_detector_vocabulary(ontology)
    flag_categories = [item for item in detector["categories"] if item["identity"]["kind"] == "owl_individual"]
    assert {item["name"] for item in flag_categories} == set(FLAG_INDIVIDUALS)
    assert all("Germany" != item["name"] for item in detector["categories"])
    with pytest.raises(ValueError, match="not rendered"):
        ontology.component_class_targets("Germany")
    with pytest.raises(ValueError, match="Unsupported detector category"):
        ontology.component_class_targets("HotTopping")


def test_detector_vocabulary_is_separate_from_class_vocabulary() -> None:
    ontology = OntologySnapshot.load()
    detector = build_detector_vocabulary(ontology)
    assert detector["background_category_id"] == 0
    assert len(detector["categories"]) == 22
    assert [item["category_id"] for item in detector["categories"]] == list(range(1, 23))
    assert set(item["name"] for item in detector["categories"]) == set(DETECTOR_CATEGORIES)
    assert all(item["name"] not in ontology.class_names for item in detector["categories"] if item["identity"]["kind"] == "owl_individual")
