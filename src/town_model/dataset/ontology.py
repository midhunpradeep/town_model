"""Pinned ontology inventory and narrow, audited target-expansion rules.

This parses only named classes, named individuals, named subclass edges, and the
specific equivalent-class fragments used by the Pizzaïolo component mapping. It
is not (and does not claim to be) a general-purpose OWL reasoner.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from .constants import (
    BASE_CATEGORIES,
    DETECTOR_CATEGORIES,
    EXPECTED_OWL_CLASS_COUNT,
    EXPECTED_SUPERVISED_CLASS_COUNT,
    EXPECTED_TRAINABLE_CLASS_COUNT,
    FIXED_POSITIVE_CLASS,
    FLAG_INDIVIDUALS,
    IMAGE_LABEL_CLASSES,
    ONTOLOGY_FILENAME,
    ONTOLOGY_SHA256,
    SCHEMA_VERSION,
    TOPPING_COLUMNS,
    UNSUPPORTED_CLASS_NAMES,
)

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
NS = {"rdf": RDF, "rdfs": RDFS, "owl": OWL}
ASSET_ONTOLOGY_PATH = Path(__file__).with_name("assets") / ONTOLOGY_FILENAME


def _local_name(iri: str | None) -> str | None:
    if not iri:
        return None
    return iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class OntologySnapshot:
    path: Path
    sha256: str
    base_iri: str
    class_iris: dict[str, str]
    direct_parents: dict[str, frozenset[str]]
    individual_types: dict[str, frozenset[str]]
    class_elements: dict[str, ET.Element]
    disjoint_pairs: frozenset[tuple[str, str]]

    @classmethod
    def load(cls, path: Path = ASSET_ONTOLOGY_PATH, *, require_pinned_hash: bool = False) -> "OntologySnapshot":
        path = path.resolve()
        digest = _hash_file(path)
        if require_pinned_hash and digest != ONTOLOGY_SHA256:
            raise ValueError(
                f"Ontology snapshot hash mismatch for {path}: {digest} != {ONTOLOGY_SHA256}"
            )
        tree = ET.parse(path)
        root = tree.getroot()
        base_iri = root.get("{http://www.w3.org/XML/1998/namespace}base", "")
        class_iris: dict[str, str] = {}
        class_elements: dict[str, ET.Element] = {}
        named_individual_types: dict[str, str] = {}

        for element in root:
            about = element.get(f"{{{RDF}}}about")
            name = _local_name(about)
            if element.tag == f"{{{OWL}}}Class" and name and about:
                class_iris[name] = f"{base_iri}{about}" if about.startswith("#") else about
                class_elements[name] = element
            named_individual = element.find("rdf:type[@rdf:resource='" + OWL + "NamedIndividual']", NS)
            if named_individual is not None and name:
                named_individual_types[name] = element.tag.rsplit("}", 1)[-1]
        individual_types = {
            name: frozenset({type_name})
            for name, type_name in named_individual_types.items()
            if type_name in class_iris
        }

        direct_parents: dict[str, frozenset[str]] = {}
        disjoint_pairs: set[tuple[str, str]] = set()
        for name, element in class_elements.items():
            parents = {
                parent
                for child in element.findall("rdfs:subClassOf", NS)
                if (parent := _local_name(child.get(f"{{{RDF}}}resource"))) in class_iris
            }
            direct_parents[name] = frozenset(parents)
            for child in element.findall("owl:disjointWith", NS):
                other = _local_name(child.get(f"{{{RDF}}}resource"))
                if other in class_iris:
                    disjoint_pairs.add(tuple(sorted((name, other))))
        for group in root.findall("owl:AllDisjointClasses/owl:members", NS):
            names = {
                name
                for member in group
                if (name := _local_name(member.get(f"{{{RDF}}}about"))) in class_iris
            }
            disjoint_pairs.update(tuple(sorted(pair)) for pair in combinations(names, 2))

        if len(class_iris) != EXPECTED_OWL_CLASS_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_OWL_CLASS_COUNT} named OWL classes in {path}; "
                f"found {len(class_iris)}"
            )
        if "Germany" in class_iris or "Germany" not in individual_types:
            raise ValueError("Ontology country-individual distinction changed: Germany")
        return cls(
            path=path,
            sha256=digest,
            base_iri=base_iri,
            class_iris=class_iris,
            direct_parents=direct_parents,
            individual_types={key: frozenset(value) for key, value in individual_types.items()},
            class_elements=class_elements,
            disjoint_pairs=frozenset(disjoint_pairs),
        )

    @property
    def class_names(self) -> frozenset[str]:
        return frozenset(self.class_iris)

    @property
    def individuals(self) -> frozenset[str]:
        return frozenset(self.individual_types)

    def ancestors(self, class_name: str, *, include_self: bool = True) -> frozenset[str]:
        """Return closure of named ``rdfs:subClassOf`` edges only."""
        if class_name not in self.class_iris:
            raise ValueError(f"{class_name!r} is not a named OWL class")
        found: set[str] = {class_name} if include_self else set()
        pending = list(self.direct_parents[class_name])
        while pending:
            parent = pending.pop()
            if parent in found:
                continue
            found.add(parent)
            pending.extend(self.direct_parents.get(parent, ()))
        return frozenset(found)

    def disjointness_status(self, first: str, second: str) -> str:
        """Return asserted/entailed/unknown using only named subclass/disjoint axioms."""
        if first not in self.class_iris or second not in self.class_iris:
            return "unknown"
        pair = tuple(sorted((first, second)))
        if pair in self.disjoint_pairs:
            return "asserted"
        first_ancestors = self.ancestors(first)
        second_ancestors = self.ancestors(second)
        if any(tuple(sorted((left, right))) in self.disjoint_pairs for left in first_ancestors for right in second_ancestors):
            return "entailed"
        return "unknown"

    def geometric_relation_metadata(self, source_class: str, target_class: str, relation: str) -> dict[str, str]:
        """Describe supported ontology metadata without treating absence as false."""
        if source_class not in self.class_iris or target_class not in self.class_iris:
            return {"status": "unknown", "basis": "one or both class IDs are absent from the pinned ontology"}
        if relation == "containment":
            if target_class in self.direct_parents.get(source_class, frozenset()):
                return {"status": "asserted", "basis": "direct named rdfs:subClassOf axiom"}
            if target_class in self.ancestors(source_class):
                return {"status": "entailed", "basis": "transitive closure of named rdfs:subClassOf axioms"}
            if target_class in self._supported_subclass_closure(source_class):
                return {
                    "status": "entailed",
                    "basis": "sound consequence of a recognized named equivalent-class fragment",
                }
            return {"status": "unknown", "basis": "no supported asserted/entailed named subclass relationship found"}
        if relation == "disjointness":
            status = self.disjointness_status(source_class, target_class)
            basis = {
                "asserted": "named owl:disjointWith or owl:AllDisjointClasses axiom",
                "entailed": "subclass closure under an asserted named disjointness axiom",
                "unknown": "no asserted/entailed named disjointness relationship found",
            }[status]
            return {"status": status, "basis": basis}
        if relation == "intersection":
            return {"status": "unknown", "basis": "the pinned ontology metadata does not assert named region-overlap existence"}
        raise ValueError(f"unsupported geometric relation {relation!r}")

    def _resolved_iri(self, reference: str | None) -> str | None:
        return urljoin(self.base_iri, reference) if reference else None

    def _named_class_reference(self, element: ET.Element) -> str | None:
        if element.tag != f"{{{RDF}}}Description" or len(element) != 0:
            return None
        reference = element.get(f"{{{RDF}}}about")
        name = _local_name(reference)
        if name not in self.class_iris or self._resolved_iri(reference) != self.class_iris[name]:
            return None
        return name

    def _named_intersection_union_equivalence(
        self, class_name: str
    ) -> tuple[frozenset[str], frozenset[str]] | None:
        """Parse only ``named intersection (named union)`` equivalent axioms."""
        element = self.class_elements[class_name]
        equivalences = element.findall("owl:equivalentClass", NS)
        if len(equivalences) != 1 or len(equivalences[0]) != 1:
            return None
        expression = equivalences[0][0]
        if expression.tag != f"{{{OWL}}}Class" or len(expression) != 1:
            return None
        intersection = expression[0]
        if (
            intersection.tag != f"{{{OWL}}}intersectionOf"
            or intersection.get(f"{{{RDF}}}parseType") != "Collection"
        ):
            return None

        required: list[str] = []
        union_expressions: list[ET.Element] = []
        for operand in intersection:
            name = self._named_class_reference(operand)
            if name is not None:
                required.append(name)
                continue
            if operand.tag != f"{{{OWL}}}Class" or len(operand) != 1:
                return None
            union = operand[0]
            if (
                union.tag != f"{{{OWL}}}unionOf"
                or union.get(f"{{{RDF}}}parseType") != "Collection"
            ):
                return None
            union_expressions.append(union)
        if not required or len(set(required)) != len(required) or len(union_expressions) != 1:
            return None

        members = [self._named_class_reference(member) for member in union_expressions[0]]
        if not members or any(member is None for member in members) or len(set(members)) != len(members):
            return None
        return frozenset(required), frozenset(members)

    def _vegetarian_equivalence_sources(self) -> frozenset[str]:
        """Return union operands only for the exact supported VegetarianTopping axiom."""
        definition = self._named_intersection_union_equivalence("VegetarianTopping")
        expected = {"CheeseTopping", "FruitTopping", "VegetableTopping"}
        if definition != (frozenset({"PizzaTopping"}), frozenset(expected)):
            return frozenset()
        return frozenset(expected)

    def _spicy_values(self, class_name: str) -> frozenset[str]:
        element = self.class_elements[class_name]
        values: set[str] = set()
        for restriction in element.findall("rdfs:subClassOf/owl:Restriction", NS):
            prop_element = restriction.find("owl:onProperty", NS)
            prop_reference = prop_element.get(f"{{{RDF}}}resource") if prop_element is not None else None
            if self._resolved_iri(prop_reference) != self._resolved_iri("#hasSpiciness"):
                continue
            for value in restriction.findall("owl:someValuesFrom", NS):
                value_reference = value.get(f"{{{RDF}}}resource")
                named = _local_name(value_reference)
                if named in self.class_iris and self._resolved_iri(value_reference) == self.class_iris[named]:
                    values.add(named)
        return frozenset(values)

    def _spicy_equivalence_sources(self) -> frozenset[str] | None:
        """Parse the supported ``PizzaTopping ∩ hasSpiciness some (Hot ∪ Medium)`` axiom."""
        element = self.class_elements["SpicyTopping"]
        equivalences = element.findall("owl:equivalentClass", NS)
        if len(equivalences) != 1 or len(equivalences[0]) != 1:
            return None
        expression = equivalences[0][0]
        if expression.tag != f"{{{OWL}}}Class" or len(expression) != 1:
            return None
        intersection = expression[0]
        if (
            intersection.tag != f"{{{OWL}}}intersectionOf"
            or intersection.get(f"{{{RDF}}}parseType") != "Collection"
            or len(intersection) != 2
        ):
            return None

        bases: list[str] = []
        restrictions: list[ET.Element] = []
        for operand in intersection:
            name = self._named_class_reference(operand)
            if name is not None:
                bases.append(name)
            elif operand.tag == f"{{{OWL}}}Restriction":
                restrictions.append(operand)
            else:
                return None
        if set(bases) != {"PizzaTopping"} or len(bases) != 1 or len(restrictions) != 1:
            return None

        restriction = restrictions[0]
        properties = restriction.findall("owl:onProperty", NS)
        fillers = restriction.findall("owl:someValuesFrom", NS)
        if (
            len(restriction) != 2
            or len(properties) != 1
            or self._resolved_iri(properties[0].get(f"{{{RDF}}}resource"))
            != self._resolved_iri("#hasSpiciness")
            or len(fillers) != 1
            or len(fillers[0]) != 1
            or fillers[0][0].tag != f"{{{OWL}}}Class"
            or len(fillers[0][0]) != 1
        ):
            return None
        union = fillers[0][0][0]
        if (
            union.tag != f"{{{OWL}}}unionOf"
            or union.get(f"{{{RDF}}}parseType") != "Collection"
        ):
            return None
        values = [self._named_class_reference(member) for member in union]
        if values != ["Medium", "Hot"] and values != ["Hot", "Medium"]:
            return None
        return frozenset(values)

    def _supported_subclass_closure(self, class_name: str) -> frozenset[str]:
        """Named closure plus consequences of the two audited equivalent fragments.

        The supported directions are (1) a subclass of PizzaTopping and any
        VegetarianTopping union member is a VegetarianTopping, and (2) a
        PizzaTopping subclass carrying a named ``hasSpiciness some Hot/Medium``
        restriction is a SpicyTopping. Union membership alone is never reversed.
        """
        derived = set(self.ancestors(class_name))
        vegetarian_sources = self._vegetarian_equivalence_sources()
        spicy_sources = self._spicy_equivalence_sources()
        while True:
            additional: set[str] = set()
            for name in derived:
                additional.update(self.direct_parents.get(name, ()))
            if vegetarian_sources:
                if "VegetarianTopping" in derived:
                    additional.add("PizzaTopping")
                if "PizzaTopping" in derived and derived.intersection(vegetarian_sources):
                    additional.add("VegetarianTopping")
            if spicy_sources:
                if "SpicyTopping" in derived:
                    additional.add("PizzaTopping")
                inherited_spiciness = set().union(*(self._spicy_values(name) for name in derived))
                if "PizzaTopping" in derived and inherited_spiciness.intersection(spicy_sources):
                    additional.add("SpicyTopping")
            additional.difference_update(derived)
            if not additional:
                return frozenset(derived)
            derived.update(additional)

    def component_class_targets(self, category_name: str) -> frozenset[str]:
        """Return verified class targets for a rendered detector category.

        Country flag identities are OWL individuals. The class targets of each
        flag are the named individual's declared class (`Country`) and its
        named superclass closure; the flag name is never promoted to a class.
        """

        if category_name in self.individual_types:
            if category_name not in FLAG_INDIVIDUALS:
                raise ValueError(f"Country flag {category_name!r} is not rendered by this generator")
            types = self.individual_types[category_name]
            targets: set[str] = set()
            for class_name in types:
                targets.update(self.ancestors(class_name))
            return frozenset(targets)
        if category_name not in (*TOPPING_COLUMNS, *BASE_CATEGORIES):
            raise ValueError(f"Unsupported detector category {category_name!r}")

        return self._supported_subclass_closure(category_name)

    def validate_component_memberships(self) -> list[str]:
        errors: list[str] = []
        if not set(IMAGE_LABEL_CLASSES).issubset(self.class_names):
            errors.append("Image label vocabulary contains undeclared OWL classes")
        if UNSUPPORTED_CLASS_NAMES.intersection(IMAGE_LABEL_CLASSES):
            errors.append("Unsupported value-partition classes entered image supervision")
        if FIXED_POSITIVE_CLASS not in self.class_names:
            errors.append("DomainThing is not declared as an OWL class")

        expected_supported = set(IMAGE_LABEL_CLASSES)
        for category in DETECTOR_CATEGORIES:
            try:
                expected_supported.update(self.component_class_targets(category))
            except ValueError as error:
                errors.append(str(error))
        if expected_supported.intersection(self.individuals):
            errors.append("An OWL individual was promoted into ontology class supervision")
        if not expected_supported.issubset(self.class_names):
            errors.append("Component or image targets contain undeclared ontology classes")
        if len(expected_supported) != EXPECTED_SUPERVISED_CLASS_COUNT:
            errors.append(
                "Expected 61 supported ontology classes (including fixed DomainThing); "
                f"found {len(expected_supported)}"
            )
        if len(expected_supported - {FIXED_POSITIVE_CLASS}) != EXPECTED_TRAINABLE_CLASS_COUNT:
            errors.append("Expected exactly 60 trainable ontology classes")
        if "Germany" in DETECTOR_CATEGORIES or "Germany" in self.class_iris:
            errors.append("Germany must remain an ontology individual, not a class/category")
        return errors


def build_class_vocabulary(ontology: OntologySnapshot | None = None) -> dict[str, Any]:
    ontology = ontology or OntologySnapshot.load(require_pinned_hash=True)
    errors = ontology.validate_component_memberships()
    if errors:
        raise ValueError("Invalid pinned ontology/vocabulary: " + "; ".join(errors))

    pizza_supervised: set[str] = set()
    for label in IMAGE_LABEL_CLASSES:
        pizza_supervised.update(ontology.ancestors(label))
    component_supervised: set[str] = set()
    for category in DETECTOR_CATEGORIES:
        component_supervised.update(ontology.component_class_targets(category))
    supervised = pizza_supervised | component_supervised

    classes = []
    for class_id, name in enumerate(sorted(ontology.class_iris, key=lambda value: ontology.class_iris[value])):
        fixed = name == FIXED_POSITIVE_CLASS
        trainable = name in supervised and not fixed
        unsupported = name not in supervised
        classes.append(
            {
                "class_id": class_id,
                "name": name,
                "iri": ontology.class_iris[name],
                "entity_kind": "owl_class",
                "trainable": trainable,
                "fixed_positive": fixed,
                "pizza_supervision": name in pizza_supervised,
                "component_supervision": name in component_supervised,
                "unsupported": unsupported,
                "supervision_status": "fixed_positive" if fixed else "trainable" if trainable else "unsupported",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "ontology_sha256": ontology.sha256,
        "target_encoding": "uint8-json-vectors-in-class_id-order",
        "classes": classes,
        "trainable_class_ids": [item["class_id"] for item in classes if item["trainable"]],
        "fixed_positive_class_ids": [item["class_id"] for item in classes if item["fixed_positive"]],
        "unsupported_class_ids": [item["class_id"] for item in classes if item["unsupported"]],
    }


def build_detector_vocabulary(ontology: OntologySnapshot | None = None) -> dict[str, Any]:
    ontology = ontology or OntologySnapshot.load(require_pinned_hash=True)
    classes_by_name = {
        item["name"]: item["class_id"] for item in build_class_vocabulary(ontology)["classes"]
    }
    categories = []
    for category_id, name in enumerate(DETECTOR_CATEGORIES, start=1):
        if name in ontology.individuals:
            if name not in FLAG_INDIVIDUALS:
                raise ValueError(f"Unsupported individual detector category {name!r}")
            targets = ontology.component_class_targets(name)
            identity = {"kind": "owl_individual", "iri": f"{ontology.base_iri}#{name}"}
        else:
            targets = ontology.component_class_targets(name)
            identity = {"kind": "owl_class", "iri": ontology.class_iris[name]}
        categories.append(
            {
                "category_id": category_id,
                "name": name,
                "identity": identity,
                "ontology_target_class_ids": sorted(classes_by_name[target] for target in targets),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "background_category_id": 0,
        "categories": categories,
    }


def targets_for_classes(names: set[str] | frozenset[str], vocabulary: dict[str, Any]) -> list[int]:
    class_ids = {entry["name"]: entry["class_id"] for entry in vocabulary["classes"]}
    unknown = names - set(class_ids)
    if unknown:
        raise ValueError(f"Targets reference unknown ontology classes: {sorted(unknown)}")
    return sorted(class_ids[name] for name in names)
