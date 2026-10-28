from __future__ import annotations

from pathlib import Path

import pytest

from town_model.config import DatasetConfig, load_config


def test_balanced_allocation_is_exact_and_seeded() -> None:
    config = DatasetConfig(total_instances=17, pizza_types=("American", "Cajun", "random"), generation_seed=42)
    assert config.allocate_type_counts() == config.allocate_type_counts()
    assert sum(config.allocate_type_counts().values()) == 17
    assert sorted(config.allocate_type_counts().values()) == [5, 6, 6]


def test_custom_counts_and_split_constraints_are_validated() -> None:
    config = DatasetConfig(total_instances=6, pizza_types=("American", "random"), type_counts={"American": 3, "random": 2})
    with pytest.raises(ValueError, match="must sum to total_instances"):
        config.allocate_type_counts()

    too_small = DatasetConfig(total_instances=5, pizza_types=("American", "random"), type_counts={"American": 2, "random": 3})
    with pytest.raises(ValueError, match="at least three"):
        too_small.allocate_type_counts()


def test_unknown_config_fields_and_invalid_ratios_fail(tmp_path: Path) -> None:
    path = tmp_path / "dataset.toml"
    path.write_text('total_instances = 6\npizza_types = ["American", "random"]\nunknown = true\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown dataset config settings"):
        load_config(path)

    config = DatasetConfig(total_instances=6, pizza_types=("American", "random"), train_size=0.6, valid_size=0.3, test_size=0.2)
    with pytest.raises(ValueError, match="sum to 1"):
        config.validate()


def test_toml_configuration_can_be_overridden_without_mutating_defaults(tmp_path: Path) -> None:
    path = tmp_path / "dataset.toml"
    path.write_text('total_instances = 6\npizza_types = ["American", "random"]\ngeneration_seed = 7\n', encoding="utf-8")
    config = load_config(path, generation_seed=91)
    assert config.generation_seed == 91
    assert config.pizza_types == ("American", "random")
    assert config.resolved_dict()["pizza_type_counts"] == {"American": 3, "random": 3}


def test_seeds_reject_bool_and_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="generation_seed"):
        DatasetConfig(total_instances=3, pizza_types=("random",), generation_seed=True).validate()
