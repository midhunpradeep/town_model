from __future__ import annotations

from town_model.dataset.constants import TOPPING_COLUMNS
from town_model.dataset.labels import infer_pizzaiolo_label


def row(*, toppings: dict[str, int] | None = None, base: str = "ThinAndCrispyBase", origin: str | None = None) -> dict:
    values = {name: 0 for name in TOPPING_COLUMNS}
    values.update(toppings or {})
    return {
        "ref": "sample",
        "hasTopping": [name for name, count in values.items() if count > 0],
        "hasBase": base,
        "hasCountryOfOrigin": origin,
        **values,
    }


def test_audited_american_recipe_predicate_is_preserved() -> None:
    result = infer_pizzaiolo_label(
        row(
            toppings={"PeperonataTopping": 1, "PeperoniSausageTopping": 5},
            base="DeepPanBase",
            origin="America",
        )
    )
    assert result == [
        "DomainThing", "Food", "Pizza", "American", "RealAmericanPizza",
        "SpicyPizza", "ThickPizza", "NamedPizza", "NonVegetarianPizza",
    ]


def test_complete_image_label_universe_remains_32_and_overlapping() -> None:
    observed = set(infer_pizzaiolo_label(row())) | {
        "GenerousPizza", "ProteinPizza", "American", "AmericanHot", "Cajun",
        "Capricciosa", "Fiorentina", "FourSeasons", "Giardiniera", "LaReine",
        "Mushroom", "Napoletana", "Parmense", "Siciliana", "SloppyGiuseppe",
        "Soho", "Veneziana", "RealAmericanPizza", "RealEnglishPizza",
        "RealFrenchPizza", "RealItalianPizza", "CheesyPizza", "SpicyPizza",
        "ThickPizza", "VegetarianPizza", "NonVegetarianPizza", "NamedPizza",
        "SpicyVegePizza", "VerySpicyPizza",
    }
    assert len(observed) == 32


def test_generous_protein_and_very_spicy_predicates_are_retained() -> None:
    generous = infer_pizzaiolo_label(row(toppings={"OliveTopping": 8}))
    assert "GenerousPizza" in generous

    protein = infer_pizzaiolo_label(row(toppings={"HamTopping": 1, "ParmesanTopping": 1}))
    assert "ProteinPizza" in protein
    assert "NonVegetarianPizza" in protein

    very_spicy = infer_pizzaiolo_label(row(toppings={"JalapenoPepperTopping": 4}))
    assert "SpicyPizza" in very_spicy
    assert "VerySpicyPizza" in very_spicy


def test_missing_origin_is_a_known_recipe_fact_not_an_origin_label() -> None:
    labels = infer_pizzaiolo_label(row(toppings={"OliveTopping": 1}, origin=None))
    assert not any(label.startswith("Real") for label in labels)
    assert not {"American", "LaReine", "Mushroom"}.intersection(labels)
