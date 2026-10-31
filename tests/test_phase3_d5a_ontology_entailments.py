from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from town_model.dataset.constants import TOPPING_COLUMNS
from town_model.dataset.ontology import (
    ASSET_ONTOLOGY_PATH,
    OWL,
    RDF,
    OntologySnapshot,
    build_class_vocabulary,
)
from town_model.inspection.relation_diagnostics import box_region, diagnose_region_relation


@pytest.mark.parametrize(
    "source",
    [
        "MushroomTopping",
        "VegetableTopping",
        "CheeseTopping",
        "ParmesanTopping",
        "FruitTopping",
        "SultanaTopping",
    ],
)
def test_named_subclasses_of_vegetarian_union_are_entailed(source: str) -> None:
    ontology = OntologySnapshot.load(require_pinned_hash=True)

    metadata = ontology.geometric_relation_metadata(source, "VegetarianTopping", "containment")

    assert metadata["status"] == "entailed"
    assert "equivalent-class" in metadata["basis"]


def test_named_assertion_closure_and_unknown_direction_are_preserved() -> None:
    ontology = OntologySnapshot.load(require_pinned_hash=True)

    assert ontology.geometric_relation_metadata(
        "MushroomTopping", "VegetableTopping", "containment"
    )["status"] == "asserted"
    assert ontology.geometric_relation_metadata(
        "MushroomTopping", "PizzaTopping", "containment"
    )["status"] == "entailed"
    assert ontology.geometric_relation_metadata(
        "MushroomTopping", "MeatTopping", "containment"
    )["status"] == "unknown"
    assert ontology.geometric_relation_metadata(
        "VegetarianTopping", "MushroomTopping", "containment"
    )["status"] == "unknown"
    # Being mentioned as a disjunct does not establish that every vegetarian
    # topping is a mushroom (or even a vegetable topping).
    assert ontology.geometric_relation_metadata(
        "VegetarianTopping", "VegetableTopping", "containment"
    )["status"] == "unknown"


def test_supported_spiciness_equivalence_consequence_is_reported() -> None:
    ontology = OntologySnapshot.load(require_pinned_hash=True)

    assert ontology.geometric_relation_metadata(
        "JalapenoPepperTopping", "SpicyTopping", "containment"
    )["status"] == "entailed"
    assert ontology.geometric_relation_metadata(
        "MushroomTopping", "SpicyTopping", "containment"
    )["status"] == "unknown"


def test_equivalent_entailments_match_component_targets_without_changing_vocabulary() -> None:
    ontology = OntologySnapshot.load(require_pinned_hash=True)
    vocabulary = build_class_vocabulary(ontology)
    ids = {entry["name"]: entry["class_id"] for entry in vocabulary["classes"]}

    assert ids["MushroomTopping"] == 26
    assert ids["VegetarianTopping"] == 63
    assert len(vocabulary["classes"]) == 66
    assert len(vocabulary["trainable_class_ids"]) == 60
    assert ontology.component_class_targets("MushroomTopping") == {
        "MushroomTopping",
        "VegetableTopping",
        "VegetarianTopping",
        "PizzaTopping",
        "Food",
        "DomainThing",
    }

    for source in TOPPING_COLUMNS:
        for target in ("VegetarianTopping", "SpicyTopping"):
            is_component_target = target in ontology.component_class_targets(source)
            status = ontology.geometric_relation_metadata(source, target, "containment")["status"]
            assert (status != "unknown") == is_component_target


def _ontology_with_nested_vegetable_disjunct(tmp_path) -> OntologySnapshot:
    """Create an unsupported nested union operand that only mentions VegetableTopping."""
    tree = ET.parse(ASSET_ONTOLOGY_PATH)
    namespaces = {"owl": OWL, "rdf": RDF}
    vegetarian = next(
        element
        for element in tree.getroot().findall("owl:Class", namespaces)
        if element.get(f"{{{RDF}}}about") == "#VegetarianTopping"
    )
    equivalence = vegetarian.find("owl:equivalentClass", namespaces)
    assert equivalence is not None
    intersection = equivalence.find("owl:Class/owl:intersectionOf", namespaces)
    assert intersection is not None
    union_class = next(child for child in intersection if child.tag == f"{{{OWL}}}Class")
    union = union_class.find("owl:unionOf", namespaces)
    assert union is not None
    vegetable = next(
        child
        for child in union
        if child.get(f"{{{RDF}}}about") == "#VegetableTopping"
    )
    union.remove(vegetable)

    nested_intersection_class = ET.SubElement(union, f"{{{OWL}}}Class")
    nested_intersection = ET.SubElement(
        nested_intersection_class,
        f"{{{OWL}}}intersectionOf",
        {"{http://www.w3.org/1999/02/22-rdf-syntax-ns#}parseType": "Collection"},
    )
    ET.SubElement(nested_intersection, f"{{{RDF}}}Description", {f"{{{RDF}}}about": "#VegetableTopping"})
    ET.SubElement(nested_intersection, f"{{{RDF}}}Description", {f"{{{RDF}}}about": "#FruitTopping"})

    path = tmp_path / "nested-vegetarian-equivalence.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return OntologySnapshot.load(path)


def test_unsupported_nested_equivalent_expression_stays_unknown(tmp_path) -> None:
    ontology = _ontology_with_nested_vegetable_disjunct(tmp_path)

    assert ontology.geometric_relation_metadata(
        "MushroomTopping", "VegetarianTopping", "containment"
    )["status"] == "unknown"
    assert "VegetarianTopping" not in ontology.component_class_targets("MushroomTopping")


def test_existing_disjointness_metadata_is_unchanged() -> None:
    ontology = OntologySnapshot.load(require_pinned_hash=True)

    assert ontology.geometric_relation_metadata(
        "MushroomTopping", "OliveTopping", "disjointness"
    )["status"] == "asserted"
    assert ontology.geometric_relation_metadata(
        "MushroomTopping", "MeatTopping", "disjointness"
    )["status"] == "entailed"


def test_geometric_violation_does_not_change_ontology_entailment() -> None:
    ontology = OntologySnapshot.load(require_pinned_hash=True)
    mushroom_region = box_region(26, "MushroomTopping", [0.4, 0.4], [0.7, 0.7])
    vegetarian_region = box_region(63, "VegetarianTopping", [-0.8, -0.8], [0.3, 0.3])

    result = diagnose_region_relation(
        mushroom_region, vegetarian_region, "containment", ontology=ontology
    )

    assert result["outcome"] is False
    assert result["ontology_relation"]["status"] == "entailed"
