from __future__ import annotations

import json
from pathlib import Path

import pytest

from town_model.dataset.image_data import (
    deterministic_subset,
    load_prepared_dataset,
    targets_in_trainable_order,
)


@pytest.fixture(scope="module")
def image_dataset():
    return load_prepared_dataset(Path(__file__).parents[1] / "examples" / "pizza-48")


def test_dataset_vocabulary_uses_authoritative_order_and_excludes_fixed_or_unsupported(image_dataset):
    vocabulary = json.loads((image_dataset.root / "vocabularies" / "ontology_classes.json").read_text())
    expected = tuple(item["class_id"] for item in vocabulary["classes"] if item["trainable"])
    assert image_dataset.canonical_class_ids == expected == tuple(vocabulary["trainable_class_ids"])
    assert 9 in image_dataset.fixed_positive_class_ids
    assert not set(image_dataset.canonical_class_ids) & set(image_dataset.fixed_positive_class_ids)
    assert not set(image_dataset.canonical_class_ids) & set(image_dataset.unsupported_class_ids)
    assert len(image_dataset.class_names) == len(image_dataset.canonical_class_ids)


def test_subsets_are_deterministic_and_never_reassign_original_partitions(image_dataset):
    first = deterministic_subset(image_dataset, "train", size=7, seed=19)
    again = deterministic_subset(image_dataset, "train", size=7, seed=19)
    other_seed = deterministic_subset(image_dataset, "train", size=7, seed=20)
    validation = deterministic_subset(image_dataset, "validation", size=7, seed=19)
    assert first.subset_id == again.subset_id
    assert [record.sample_id for record in first.records] == [record.sample_id for record in again.records]
    assert first.subset_id != other_seed.subset_id
    assert all(record.split == "train" for record in first.records)
    assert all(record.split == "validation" for record in validation.records)
    assert not {record.sample_id for record in first.records} & {record.sample_id for record in validation.records}
    assert first.to_manifest()["dataset_id"] == image_dataset.dataset_id
    assert first.to_manifest()["class_manifest_sha256"] == image_dataset.class_manifest_sha256
    assert len(first.class_support["class_ids"]) == len(image_dataset.class_names)


def test_target_projection_uses_manifest_class_ids_and_preserves_unknowns(image_dataset):
    record = image_dataset.records[0]
    pizza, pizza_known, components, component_known = targets_in_trainable_order(image_dataset, record)
    ids = image_dataset.canonical_class_ids
    assert pizza == tuple(record.pizza_targets[class_id] for class_id in ids)
    assert pizza_known == tuple(record.pizza_known_mask[class_id] for class_id in ids)
    assert components[0] == tuple(record.components[0].ontology_targets[class_id] for class_id in ids)
    assert component_known[0] == tuple(record.components[0].ontology_known_mask[class_id] for class_id in ids)
    assert len(pizza) == len(image_dataset.class_names)


def test_dataset_integrity_fails_before_records_are_exposed(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "pizza-48"
    import shutil

    copied = tmp_path / "copy"
    shutil.copytree(source, copied)
    samples = copied / "samples.jsonl"
    samples.write_text(samples.read_text().replace('"sample_id": "img_00000"', '"sample_id": "corrupt"', 1))
    with pytest.raises(ValueError, match="integrity validation failed"):
        load_prepared_dataset(copied)
