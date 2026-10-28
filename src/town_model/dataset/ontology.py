"""Pinned ontology inventory and narrow, audited target-expansion rules.

This parses only named classes, named individuals, named subclass edges, and the
specific equivalent-class fragments used by the Pizzaïolo component mapping. It
is not (and does not claim to be) a general-purpose OWL reasoner.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        for name, element in class_elements.items():
            parents = {
                parent
                for child in element.findall("rdfs:subClassOf", NS)
                if (parent := _local_name(child.get(f"{{{RDF}}}resource"))) in class_iris
            }
            direct_parents[name] = frozenset(parents)

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
        )

    @property
    def class_names(self) -> frozenset[str]:
        return frozenset(self.class_iris)

    @property
    def individuals(self) -> frozenset[str]:
        return frozenset(self.individual_types)

    def ancestors(self, class_name: str, *, include_self: bool = True) -> frozenset[str]:
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

    def _vegetarian_equivalence_sources(self) -> frozenset[str]:
        element = self.class_elements["VegetarianTopping"]
        equivalence = element.find("owl:equivalentClass", NS)
        if equivalence is None:
            raise ValueError("VegetarianTopping is missing its equivalent-class axiom")
        named = {
            name
            for child in equivalence.iter()
            if (name := _local_name(child.get(f"{{{RDF}}}about"))) in self.class_iris
        }
        required = {"CheeseTopping", "FruitTopping", "VegetableTopping"}
        if not required.issubset(named):
            raise ValueError("VegetarianTopping equivalent-class categories changed")
        return frozenset(required)

    def _spicy_values(self, class_name: str) -> frozenset[str]:
        element = self.class_elements[class_name]
        values: set[str] = set()
        for restriction in element.findall("rdfs:subClassOf/owl:Restriction", NS):
            prop_element = restriction.find("owl:onProperty", NS)
            prop = _local_name(prop_element.get(f"{{{RDF}}}resource")) if prop_element is not None else None
            if prop != "hasSpiciness":
                continue
            for value in restriction.findall("owl:someValuesFrom", NS):
                named = _local_name(value.get(f"{{{RDF}}}resource"))
                if named:
                    values.add(named)
        return frozenset(values)

    def _has_spicy_equivalence(self) -> None:
        element = self.class_elements["SpicyTopping"]
        equivalence = element.find("owl:equivalentClass", NS)
        if equivalence is None:
            raise ValueError("SpicyTopping is missing its equivalent-class axiom")
        values = {
            _local_name(node.get(f"{{{RDF}}}about"))
            for node in equivalence.iter()
            if node.tag == f"{{{RDF}}}Description"
        }
        properties = {
            _local_name(node.get(f"{{{RDF}}}resource"))
            for node in equivalence.iter()
            if node.tag == f"{{{OWL}}}onProperty"
        }
        if not {"Hot", "Medium"}.issubset(values) or "hasSpiciness" not in properties:
            raise ValueError("SpicyTopping equivalent-class axiom no longer uses Hot/Medium")

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

        targets = set(self.ancestors(category_name))
        if category_name in TOPPING_COLUMNS:
            vegetarian_sources = self._vegetarian_equivalence_sources()
            if targets.intersection(vegetarian_sources):
                targets.add("VegetarianTopping")
            self._has_spicy_equivalence()
            if self._spicy_values(category_name).intersection({"Hot", "Medium"}):
                targets.add("SpicyTopping")
        return frozenset(targets)

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
