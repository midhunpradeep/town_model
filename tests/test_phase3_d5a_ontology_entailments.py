from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from town_model.dataset.constants import DETECTOR_CATEGORIES, TOPPING_COLUMNS
from town_model.dataset.ontology import (
    ASSET_ONTOLOGY_PATH,
    OWL,
    RDF,
    RDFS,
    OntologySnapshot,
    build_class_vocabulary,
)
from town_model.inspection.relation_diagnostics import box_region, diagnose_region_relation


XML = "http://www.w3.org/XML/1998/namespace"


def _ontology_with_mushroom_parent(
    tmp_path,
    reference: str | None,
    *,
    parent_base: str | None = None,
) -> OntologySnapshot:
    tree = ET.parse(ASSET_ONTOLOGY_PATH)
    mushroom = next(
        element
        for element in tree.getroot().findall("owl:Class", {"owl": OWL})
        if element.get(f"{{{RDF}}}about") == "#MushroomTopping"
    )
    parent = mushroom.find("rdfs:subClassOf", {"rdfs": RDFS})
    assert parent is not None
    if reference is None:
        mushroom.remove(parent)
    else:
        parent.set(f"{{{RDF}}}resource", reference)
    if parent_base is not None:
        parent.set(f"{{{XML}}}base", parent_base)
    path = tmp_path / "mushroom-parent-fixture.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return OntologySnapshot.load(path)


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


def test_legitimate_relative_named_parent_is_asserted() -> None:
    ontology = OntologySnapshot.load(require_pinned_hash=True)

    assert ontology.geometric_relation_metadata(
        "MushroomTopping", "VegetableTopping", "containment"
    )["status"] == "asserted"


def test_legitimate_absolute_named_parent_matching_declared_iri_is_asserted(tmp_path) -> None:
    pinned = OntologySnapshot.load(require_pinned_hash=True)
    ontology = _ontology_with_mushroom_parent(
        tmp_path, pinned.class_iris["VegetableTopping"]
    )

    assert ontology.geometric_relation_metadata(
        "MushroomTopping", "VegetableTopping", "containment"
    )["status"] == "asserted"
    assert ontology.component_class_targets("MushroomTopping") == pinned.component_class_targets(
        "MushroomTopping"
    )


@pytest.mark.parametrize(
    ("reference", "parent_base"),
    [
        ("https://foreign.example/ontology#VegetableTopping", None),
        ("#VegetableTopping", "https://foreign.example/ontology/"),
    ],
)
def test_foreign_named_parent_with_colliding_local_name_is_unsupported(
    tmp_path, reference: str, parent_base: str | None
) -> None:
    ontology = _ontology_with_mushroom_parent(
        tmp_path, reference, parent_base=parent_base
    )

    assert ontology.geometric_relation_metadata(
        "MushroomTopping", "VegetableTopping", "containment"
    )["status"] == "unknown"
    assert ontology.geometric_relation_metadata(
        "MushroomTopping", "VegetarianTopping", "containment"
    )["status"] == "unknown"
    targets = ontology.component_class_targets("MushroomTopping")
    assert "VegetableTopping" not in targets
    assert "VegetarianTopping" not in targets


@pytest.mark.parametrize("reference", [None, "#UnknownNamedSuperclass"])
def test_missing_or_unknown_named_superclass_is_not_inferred(tmp_path, reference) -> None:
    ontology = _ontology_with_mushroom_parent(tmp_path, reference)

    assert ontology.geometric_relation_metadata(
        "MushroomTopping", "VegetableTopping", "containment"
    )["status"] == "unknown"
    assert ontology.geometric_relation_metadata(
        "MushroomTopping", "VegetarianTopping", "containment"
    )["status"] == "unknown"


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


def test_pinned_class_ids_and_all_component_targets_remain_unchanged() -> None:
    ontology = OntologySnapshot.load(require_pinned_hash=True)
    vocabulary = build_class_vocabulary(ontology)
    ids = {entry["name"]: entry["class_id"] for entry in vocabulary["classes"]}

    assert ids["MushroomTopping"] == 26
    assert ids["VegetarianTopping"] == 63
    assert len(vocabulary["classes"]) == 66
    assert len(vocabulary["trainable_class_ids"]) == 60
    assert {
        category: ontology.component_class_targets(category)
        for category in DETECTOR_CATEGORIES
    } == {
        "America": frozenset({"Country", "DomainThing"}),
        "AnchovyTopping": frozenset({"AnchovyTopping", "DomainThing", "Food", "PizzaTopping", "SeafoodTopping"}),
        "DeepPanBase": frozenset({"DeepPanBase", "DomainThing", "Food", "PizzaBase"}),
        "England": frozenset({"Country", "DomainThing"}),
        "France": frozenset({"Country", "DomainThing"}),
        "GarlicTopping": frozenset({"DomainThing", "Food", "GarlicTopping", "PizzaTopping", "VegetableTopping", "VegetarianTopping"}),
        "GreenPepperTopping": frozenset({"DomainThing", "Food", "GreenPepperTopping", "PepperTopping", "PizzaTopping", "VegetableTopping", "VegetarianTopping"}),
        "HamTopping": frozenset({"DomainThing", "Food", "HamTopping", "MeatTopping", "PizzaTopping"}),
        "Italy": frozenset({"Country", "DomainThing"}),
        "JalapenoPepperTopping": frozenset({"DomainThing", "Food", "JalapenoPepperTopping", "PepperTopping", "PizzaTopping", "SpicyTopping", "VegetableTopping", "VegetarianTopping"}),
        "MushroomTopping": frozenset({"DomainThing", "Food", "MushroomTopping", "PizzaTopping", "VegetableTopping", "VegetarianTopping"}),
        "OliveTopping": frozenset({"DomainThing", "Food", "OliveTopping", "PizzaTopping", "VegetableTopping", "VegetarianTopping"}),
        "OnionTopping": frozenset({"DomainThing", "Food", "OnionTopping", "PizzaTopping", "VegetableTopping", "VegetarianTopping"}),
        "ParmesanTopping": frozenset({"CheeseTopping", "DomainThing", "Food", "ParmesanTopping", "PizzaTopping", "VegetarianTopping"}),
        "PeperonataTopping": frozenset({"DomainThing", "Food", "PeperonataTopping", "PepperTopping", "PizzaTopping", "SpicyTopping", "VegetableTopping", "VegetarianTopping"}),
        "PeperoniSausageTopping": frozenset({"DomainThing", "Food", "MeatTopping", "PeperoniSausageTopping", "PizzaTopping", "SpicyTopping"}),
        "PrawnTopping": frozenset({"DomainThing", "Food", "PizzaTopping", "PrawnTopping", "SeafoodTopping"}),
        "RocketTopping": frozenset({"DomainThing", "Food", "PizzaTopping", "RocketTopping", "VegetableTopping", "VegetarianTopping"}),
        "SlicedTomatoTopping": frozenset({"DomainThing", "Food", "PizzaTopping", "SlicedTomatoTopping", "VegetableTopping", "VegetarianTopping"}),
        "SpinachTopping": frozenset({"DomainThing", "Food", "PizzaTopping", "SpinachTopping", "VegetableTopping", "VegetarianTopping"}),
        "SultanaTopping": frozenset({"DomainThing", "Food", "FruitTopping", "PizzaTopping", "SultanaTopping", "VegetarianTopping"}),
        "ThinAndCrispyBase": frozenset({"DomainThing", "Food", "PizzaBase", "ThinAndCrispyBase"}),
    }


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
