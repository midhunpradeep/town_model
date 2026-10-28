"""Audited Pizzaïolo image-label predicates adapted from fourth_attempt.

This preserves the original 32-class, overlapping source inference. The helpers
are intentionally generator-specific and are not used by a future model loader.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from typing import Any

from .constants import TOPPING_COLUMNS, TOPPING_SUBCLASSES

PIZZAIOLO_DEFAULT_TYPE = "unspecified"


def _topping_names(row: Mapping[str, Any]) -> set[str]:
    raw = row["hasTopping"]
    if isinstance(raw, str):
        parsed = ast.literal_eval(raw)
    else:
        parsed = raw
    if not isinstance(parsed, (list, tuple, set)) or not all(isinstance(v, str) for v in parsed):
        raise ValueError("hasTopping must be a serialized list of topping class names")
    return set(parsed)


def has_some_base(row: Mapping[str, Any], base: str) -> bool:
    return row.get("hasBase") == base


def has_only_base(row: Mapping[str, Any], base: str) -> bool:
    return row.get("hasBase") == base


def has_country_of_origin(row: Mapping[str, Any], country: str) -> bool:
    return row.get("hasCountryOfOrigin") == country


def has_some_topping(row: Mapping[str, Any], topping: str) -> bool:
    if topping in TOPPING_SUBCLASSES:
        return any(has_some_topping(row, subclass) for subclass in TOPPING_SUBCLASSES[topping])
    return int(row[topping]) > 0


def has_only_toppings(row: Mapping[str, Any], toppings: set[str]) -> bool:
    expanded: set[str] = set()
    for topping in toppings:
        expanded.update(TOPPING_SUBCLASSES.get(topping, (topping,)))
    return not (_topping_names(row) - expanded)


def has_at_least_n_of_topping(row: Mapping[str, Any], topping: str, n: int) -> bool:
    if topping in TOPPING_SUBCLASSES:
        return sum(int(row[subclass]) for subclass in TOPPING_SUBCLASSES[topping]) >= n
    return int(row[topping]) >= n


def infer_pizzaiolo_label(row: Mapping[str, Any]) -> list[str]:
    """Apply the established source rules without changing overlap semantics."""

    labels = ["DomainThing", "Food", "Pizza"]
    if (
        has_some_base(row, "DeepPanBase")
        and has_some_topping(row, "PeperonataTopping")
        and has_some_topping(row, "PeperoniSausageTopping")
        and has_only_base(row, "DeepPanBase")
        and has_only_toppings(row, {"PeperonataTopping", "PeperoniSausageTopping"})
        and has_country_of_origin(row, "America")
    ):
        labels.append("American")

    if (
        has_some_base(row, "DeepPanBase")
        and has_some_topping(row, "GreenPepperTopping")
        and has_some_topping(row, "JalapenoPepperTopping")
        and has_some_topping(row, "PeperoniSausageTopping")
        and has_only_base(row, "DeepPanBase")
        and has_only_toppings(row, {"GreenPepperTopping", "JalapenoPepperTopping", "PeperoniSausageTopping"})
        and has_country_of_origin(row, "America")
    ):
        labels.append("AmericanHot")

    if (
        has_some_base(row, "ThinAndCrispyBase")
        and has_some_topping(row, "OnionTopping")
        and has_some_topping(row, "PeperonataTopping")
        and has_some_topping(row, "PrawnTopping")
        and has_only_base(row, "ThinAndCrispyBase")
        and has_only_toppings(row, {"OnionTopping", "PeperonataTopping", "PrawnTopping"})
    ):
        labels.append("Cajun")

    if (
        has_some_base(row, "ThinAndCrispyBase")
        and all(has_some_topping(row, name) for name in ("AnchovyTopping", "HamTopping", "OliveTopping", "PeperonataTopping"))
        and has_only_base(row, "ThinAndCrispyBase")
        and has_only_toppings(row, {"AnchovyTopping", "HamTopping", "OliveTopping", "PeperonataTopping"})
    ):
        labels.append("Capricciosa")

    if has_some_topping(row, "CheeseTopping"):
        labels.append("CheesyPizza")

    if (
        has_some_base(row, "ThinAndCrispyBase")
        and all(has_some_topping(row, name) for name in ("GarlicTopping", "OliveTopping", "ParmesanTopping", "SlicedTomatoTopping", "SpinachTopping"))
        and has_only_base(row, "ThinAndCrispyBase")
        and has_only_toppings(row, {"GarlicTopping", "OliveTopping", "ParmesanTopping", "SlicedTomatoTopping", "SpinachTopping"})
        and has_country_of_origin(row, "Italy")
    ):
        labels.append("Fiorentina")

    if (
        has_some_base(row, "ThinAndCrispyBase")
        and all(has_some_topping(row, name) for name in ("AnchovyTopping", "MushroomTopping", "OliveTopping", "PeperoniSausageTopping"))
        and has_only_base(row, "ThinAndCrispyBase")
        and has_only_toppings(row, {"AnchovyTopping", "MushroomTopping", "OliveTopping", "PeperoniSausageTopping"})
    ):
        labels.append("FourSeasons")

    if has_at_least_n_of_topping(row, "OliveTopping", 8):
        labels.append("GenerousPizza")

    if (
        has_some_base(row, "ThinAndCrispyBase")
        and all(has_some_topping(row, name) for name in ("JalapenoPepperTopping", "MushroomTopping", "OliveTopping", "SlicedTomatoTopping"))
        and has_only_base(row, "ThinAndCrispyBase")
        and has_only_toppings(row, {"JalapenoPepperTopping", "MushroomTopping", "OliveTopping", "SlicedTomatoTopping"})
    ):
        labels.append("Giardiniera")

    if (
        has_some_base(row, "ThinAndCrispyBase")
        and all(has_some_topping(row, name) for name in ("HamTopping", "MushroomTopping", "OliveTopping"))
        and has_only_base(row, "ThinAndCrispyBase")
        and has_only_toppings(row, {"HamTopping", "MushroomTopping", "OliveTopping"})
        and has_country_of_origin(row, "France")
    ):
        labels.append("LaReine")

    if (
        has_some_base(row, "ThinAndCrispyBase")
        and has_some_topping(row, "MushroomTopping")
        and has_only_base(row, "ThinAndCrispyBase")
        and has_only_toppings(row, {"MushroomTopping"})
        and has_country_of_origin(row, "France")
    ):
        labels.append("Mushroom")

    if (
        has_some_base(row, "ThinAndCrispyBase")
        and has_some_topping(row, "AnchovyTopping")
        and has_some_topping(row, "OliveTopping")
        and has_only_base(row, "ThinAndCrispyBase")
        and has_only_toppings(row, {"AnchovyTopping", "OliveTopping"})
        and has_country_of_origin(row, "Italy")
    ):
        labels.append("Napoletana")

    if (
        has_some_base(row, "ThinAndCrispyBase")
        and has_some_topping(row, "HamTopping")
        and has_some_topping(row, "ParmesanTopping")
        and has_only_base(row, "ThinAndCrispyBase")
        and has_only_toppings(row, {"HamTopping", "ParmesanTopping"})
    ):
        labels.append("Parmense")

    if (has_some_topping(row, "MeatTopping") or has_some_topping(row, "SeafoodTopping")) and has_some_topping(row, "CheeseTopping"):
        labels.append("ProteinPizza")

    for country, label in (
        ("America", "RealAmericanPizza"),
        ("England", "RealEnglishPizza"),
        ("France", "RealFrenchPizza"),
        ("Italy", "RealItalianPizza"),
    ):
        if has_country_of_origin(row, country):
            labels.append(label)

    if (
        has_some_base(row, "ThinAndCrispyBase")
        and all(has_some_topping(row, name) for name in ("AnchovyTopping", "GarlicTopping", "HamTopping", "OliveTopping"))
        and has_only_base(row, "ThinAndCrispyBase")
        and has_only_toppings(row, {"AnchovyTopping", "GarlicTopping", "HamTopping", "OliveTopping"})
        and has_country_of_origin(row, "Italy")
    ):
        labels.append("Siciliana")

    if (
        has_some_base(row, "ThinAndCrispyBase")
        and has_some_topping(row, "GreenPepperTopping")
        and has_some_topping(row, "OnionTopping")
        and has_only_base(row, "ThinAndCrispyBase")
        and has_only_toppings(row, {"GreenPepperTopping", "OnionTopping"})
        and has_country_of_origin(row, "Italy")
    ):
        labels.append("SloppyGiuseppe")

    if (
        has_some_base(row, "DeepPanBase")
        and all(has_some_topping(row, name) for name in ("GarlicTopping", "OliveTopping", "ParmesanTopping", "RocketTopping"))
        and has_only_base(row, "DeepPanBase")
        and has_only_toppings(row, {"GarlicTopping", "OliveTopping", "ParmesanTopping", "RocketTopping"})
        and has_country_of_origin(row, "England")
    ):
        labels.append("Soho")

    if has_some_topping(row, "SpicyTopping"):
        labels.append("SpicyPizza")
    if has_some_base(row, "DeepPanBase"):
        labels.append("ThickPizza")
    if not has_some_topping(row, "MeatTopping") and not has_some_topping(row, "SeafoodTopping"):
        labels.append("VegetarianPizza")

    if (
        has_some_base(row, "ThinAndCrispyBase")
        and all(has_some_topping(row, name) for name in ("OliveTopping", "OnionTopping", "SultanaTopping"))
        and has_only_base(row, "ThinAndCrispyBase")
        and has_only_toppings(row, {"OliveTopping", "OnionTopping", "SultanaTopping"})
        and has_country_of_origin(row, "Italy")
    ):
        labels.append("Veneziana")

    named_pizzas = {
        "American", "AmericanHot", "Cajun", "Capricciosa", "Fiorentina",
        "FourSeasons", "Giardiniera", "LaReine", "Mushroom", "Napoletana",
        "Parmense", "Siciliana", "SloppyGiuseppe", "Soho", "Veneziana",
    }
    if named_pizzas.intersection(labels):
        labels.append("NamedPizza")
    if "VegetarianPizza" not in labels:
        labels.append("NonVegetarianPizza")
    if "SpicyPizza" in labels and "VegetarianPizza" in labels:
        labels.append("SpicyVegePizza")
    if "SpicyPizza" in labels and (
        has_at_least_n_of_topping(row, "HotTopping", 4)
        or has_at_least_n_of_topping(row, "SpicyTopping", 8)
    ):
        labels.append("VerySpicyPizza")
    return labels
