from __future__ import annotations

from types import SimpleNamespace

import pytest

from town_model.dataset.ontology import OntologySnapshot
from town_model.inspection.relation_diagnostics import validate_snapshot_ontology_mapping


def test_ontology_metadata_distinguishes_asserted_entailed_and_unknown_relations() -> None:
    """Ontology facts remain independent of retired region calculations."""
    ontology = OntologySnapshot.load(require_pinned_hash=True)
    assert ontology.geometric_relation_metadata("AnchovyTopping", "PizzaTopping", "containment")["status"] in ("asserted", "entailed")
    assert ontology.geometric_relation_metadata("Pizza", "PizzaTopping", "disjointness")["status"] == "asserted"
    unknown = ontology.geometric_relation_metadata("AnchovyTopping", "NonVegetarianPizza", "containment")
    assert unknown["status"] == "unknown"
    assert ontology.geometric_relation_metadata("AnchovyTopping", "NonVegetarianPizza", "intersection")["status"] == "unknown"

    with pytest.raises(ValueError, match="snapshot class identity mismatch"):
        validate_snapshot_ontology_mapping(
            SimpleNamespace(header={"class_ids": [2], "class_names": ["NonVegetarianPizza"]}), ontology
        )
