"""Stable Phase 1 schema identifiers and supported generator vocabulary."""

SCHEMA_VERSION = "1.0.0"
VALIDATOR_VERSION = "1.0.0"
GENERATOR_REPOSITORY = "https://github.com/SysReIC/pizzaiolo.git"
GENERATOR_COMMIT = "6c0bcb221a957c184012beaf103bebcde4b1f793"
GENERATOR_LICENSE = "CC BY-NC 4.0"
RAW_MASTER_FILENAME = "generator_master.csv"
RAW_MASTER_CANONICALIZATION = "hasTopping set values sorted lexicographically; source ordering is nondeterministic and semantically unordered"
ONTOLOGY_SHA256 = "e349737f26aebd5b41c228043cdd35991775ecbacbc68a8f543c76466fe97253"
ONTOLOGY_FILENAME = "pizzaiolo.xml"

TOPPING_COLUMNS = (
    "AnchovyTopping",
    "GarlicTopping",
    "GreenPepperTopping",
    "HamTopping",
    "JalapenoPepperTopping",
    "MushroomTopping",
    "OliveTopping",
    "OnionTopping",
    "ParmesanTopping",
    "PeperonataTopping",
    "PeperoniSausageTopping",
    "PrawnTopping",
    "RocketTopping",
    "SlicedTomatoTopping",
    "SpinachTopping",
    "SultanaTopping",
)
BASE_CATEGORIES = ("DeepPanBase", "ThinAndCrispyBase")
FLAG_INDIVIDUALS = ("America", "England", "France", "Italy")
UNSUPPORTED_FLAG_INDIVIDUALS = ("Germany",)
DETECTOR_CATEGORIES = tuple(sorted((*TOPPING_COLUMNS, *BASE_CATEGORIES, *FLAG_INDIVIDUALS)))

# Kept from the audited fourth_attempt label implementation. HotTopping is a
# recipe/counting helper only; it is not an OWL class or detector category.
TOPPING_SUBCLASSES: dict[str, tuple[str, ...]] = {
    "CheeseTopping": ("ParmesanTopping",),
    "FruitTopping": ("SultanaTopping",),
    "MeatTopping": ("HamTopping", "PeperoniSausageTopping"),
    "PepperTopping": ("GreenPepperTopping", "JalapenoPepperTopping", "PeperonataTopping"),
    "SeafoodTopping": ("AnchovyTopping", "PrawnTopping"),
    "SpicyTopping": ("PeperoniSausageTopping", "JalapenoPepperTopping", "PeperonataTopping"),
    "VegetableTopping": (
        "GarlicTopping", "GreenPepperTopping", "JalapenoPepperTopping",
        "MushroomTopping", "OliveTopping", "OnionTopping", "PeperonataTopping",
        "RocketTopping", "SlicedTomatoTopping", "SpinachTopping",
    ),
    "VegetarianTopping": (
        "GarlicTopping", "GreenPepperTopping", "JalapenoPepperTopping",
        "MushroomTopping", "OliveTopping", "OnionTopping", "ParmesanTopping",
        "PeperonataTopping", "RocketTopping", "SlicedTomatoTopping",
        "SpinachTopping", "SultanaTopping",
    ),
    "HotTopping": ("JalapenoPepperTopping",),
}

IMAGE_LABEL_CLASSES = frozenset(
    {
        "DomainThing", "Food", "Pizza", "American", "AmericanHot", "Cajun",
        "Capricciosa", "CheesyPizza", "Fiorentina", "FourSeasons", "GenerousPizza",
        "Giardiniera", "LaReine", "Mushroom", "NamedPizza", "Napoletana",
        "NonVegetarianPizza", "Parmense", "ProteinPizza", "RealAmericanPizza",
        "RealEnglishPizza", "RealFrenchPizza", "RealItalianPizza", "Siciliana",
        "SloppyGiuseppe", "Soho", "SpicyPizza", "SpicyVegePizza", "ThickPizza",
        "VegetarianPizza", "Veneziana", "VerySpicyPizza",
    }
)

UNSUPPORTED_CLASS_NAMES = frozenset({"ValuePartition", "Spiciness", "Hot", "Medium", "Mild"})
FIXED_POSITIVE_CLASS = "DomainThing"
EXPECTED_OWL_CLASS_COUNT = 66
EXPECTED_SUPERVISED_CLASS_COUNT = 61
EXPECTED_TRAINABLE_CLASS_COUNT = 60

IMAGE_DIR = "images"
LABEL_DIR = "labels"
ONTOLOGY_DIR = "ontology"
AUDIT_DIR = "audit"
REPORT_DIR = "reports"
VOCABULARY_DIR = "vocabularies"
SPLIT_DIR = "splits"
SAMPLES_FILENAME = "samples.jsonl"
MANIFEST_FILENAME = "manifest.json"
CLASS_VOCABULARY_FILENAME = "ontology_classes.json"
DETECTOR_VOCABULARY_FILENAME = "detector_categories.json"
VALIDATION_REPORT_FILENAME = "validation.json"

SPLIT_NAMES = ("train", "validation", "test")
