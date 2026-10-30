"""Validated image-record views and deterministic in-split research subsets.

This module remains framework independent: it owns semantic target selection,
class identities, records, and subset provenance, but does not load images or
import PyTorch/Torchvision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from .schema import DatasetRecord, load_records
from .validation import validate_dataset

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    root: Path
    dataset_id: str
    manifest_id: str
    schema_version: str
    records: tuple[DatasetRecord, ...]
    canonical_class_ids: tuple[int, ...]
    class_names: tuple[str, ...]
    class_iris: tuple[str, ...]
    fixed_positive_class_ids: tuple[int, ...]
    unsupported_class_ids: tuple[int, ...]
    class_manifest_sha256: str
    dataset_manifest_sha256: str

    def split_records(self, split: SplitName) -> tuple[DatasetRecord, ...]:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"unsupported dataset split {split!r}")
        return tuple(record for record in self.records if record.split == split)


@dataclass(frozen=True, slots=True)
class DatasetSubset:
    dataset: PreparedDataset
    split: SplitName
    records: tuple[DatasetRecord, ...]
    selection_seed: int
    requested_size: int | None
    selection_policy: str
    subset_id: str
    class_support: dict[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "image-subset-v1",
            "subset_id": self.subset_id,
            "dataset_id": self.dataset.dataset_id,
            "dataset_manifest_sha256": self.dataset.dataset_manifest_sha256,
            "class_manifest_sha256": self.dataset.class_manifest_sha256,
            "split": self.split,
            "selection_seed": self.selection_seed,
            "requested_size": self.requested_size,
            "selection_policy": self.selection_policy,
            "sample_ids": [record.sample_id for record in self.records],
            "class_support": self.class_support,
        }


def _read_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value, raw


def load_prepared_dataset(root: Path | str, *, validate: bool = True) -> PreparedDataset:
    """Load an integrity-checked dataset without importing image runtimes."""
    path = Path(root).expanduser().resolve()
    if validate:
        report = validate_dataset(path, write_report=False)
        if not report.valid:
            details = "; ".join(issue.message for issue in report.errors[:5])
            raise ValueError(f"dataset integrity validation failed at {path}: {details}")
    manifest, manifest_raw = _read_object(path / "manifest.json")
    vocabulary_path = path / "vocabularies" / "ontology_classes.json"
    vocabulary, vocabulary_raw = _read_object(vocabulary_path)
    classes = vocabulary.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("ontology class manifest must contain a nonempty classes list")
    if [item.get("class_id") for item in classes] != list(range(len(classes))):
        raise ValueError("ontology class IDs must be contiguous and in canonical manifest order")
    trainable = [item for item in classes if item.get("trainable") is True]
    if any(item.get("entity_kind") != "owl_class" for item in trainable):
        raise ValueError("only ontology classes may be trainable classifier outputs")
    if any(item.get("fixed_positive") for item in trainable):
        raise ValueError("fixed-positive classes must not be included in trainable outputs")
    canonical_ids = tuple(item["class_id"] for item in trainable)
    if canonical_ids != tuple(vocabulary.get("trainable_class_ids", ())):
        raise ValueError("trainable class IDs disagree with the authoritative manifest ordering")
    records = tuple(load_records(path / "samples.jsonl"))
    ids = [record.sample_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset records contain duplicate sample IDs")
    return PreparedDataset(
        root=path,
        dataset_id=str(manifest["dataset_id"]),
        manifest_id=str(manifest["manifest_id"]),
        schema_version=str(manifest["schema_version"]),
        records=records,
        canonical_class_ids=canonical_ids,
        class_names=tuple(item["name"] for item in trainable),
        class_iris=tuple(item["iri"] for item in trainable),
        fixed_positive_class_ids=tuple(vocabulary.get("fixed_positive_class_ids", ())),
        unsupported_class_ids=tuple(vocabulary.get("unsupported_class_ids", ())),
        class_manifest_sha256=hashlib.sha256(vocabulary_raw).hexdigest(),
        dataset_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
    )


def targets_in_trainable_order(
    dataset: PreparedDataset, record: DatasetRecord
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Project full manifest vectors to the authoritative ordered trainable IDs."""
    ids = dataset.canonical_class_ids
    pizza_targets = tuple(record.pizza_targets[class_id] for class_id in ids)
    pizza_known = tuple(record.pizza_known_mask[class_id] for class_id in ids)
    component_targets = tuple(
        tuple(component.ontology_targets[class_id] for class_id in ids)
        for component in record.components
    )
    component_known = tuple(
        tuple(component.ontology_known_mask[class_id] for class_id in ids)
        for component in record.components
    )
    return pizza_targets, pizza_known, component_targets, component_known


def _subset_support(dataset: PreparedDataset, records: tuple[DatasetRecord, ...]) -> dict[str, Any]:
    ids = dataset.canonical_class_ids
    levels: dict[str, dict[str, list[int]]] = {}
    for level in ("pizza", "component"):
        levels[level] = {
            "known_positive": [0] * len(ids),
            "known_negative": [0] * len(ids),
            "unknown": [0] * len(ids),
        }
    for record in records:
        for column, class_id in enumerate(ids):
            known = record.pizza_known_mask[class_id] == 1
            target = record.pizza_targets[class_id] == 1
            levels["pizza"]["known_positive" if known and target else "known_negative" if known else "unknown"][column] += 1
        for component in record.components:
            for column, class_id in enumerate(ids):
                known = component.ontology_known_mask[class_id] == 1
                target = component.ontology_targets[class_id] == 1
                levels["component"]["known_positive" if known and target else "known_negative" if known else "unknown"][column] += 1
    return {
        "class_ids": list(ids),
        "class_names": list(dataset.class_names),
        "pizza": levels["pizza"],
        "component": levels["component"],
    }


def deterministic_subset(
    dataset: PreparedDataset,
    split: SplitName,
    *,
    size: int | None,
    seed: int,
) -> DatasetSubset:
    """Select deterministically inside one original partition; never repartition."""
    if type(seed) is not int or seed < 0:
        raise ValueError("subset selection seed must be a nonnegative integer")
    source = dataset.split_records(split)
    if not source:
        raise ValueError(f"the original {split} partition is empty")
    if size is not None and (type(size) is not int or size < 1):
        raise ValueError("subset size must be a positive integer or None")
    limit = len(source) if size is None else min(size, len(source))
    # Round-robin independently hashed generator-type queues preserve the
    # source split and avoid a subset dominated by one recipe type.
    groups: dict[str, list[DatasetRecord]] = {}
    for record in source:
        groups.setdefault(record.generator_type, []).append(record)
    for generator_type, values in groups.items():
        values.sort(key=lambda record: hashlib.sha256(f"{seed}\0{record.sample_id}".encode()).digest())
    selected: list[DatasetRecord] = []
    types = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for generator_type in types:
            if groups[generator_type]:
                selected.append(groups[generator_type].pop(0))
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
    selected_records = tuple(selected)
    identity_payload = {
        "dataset_id": dataset.dataset_id,
        "manifest_sha256": dataset.dataset_manifest_sha256,
        "class_manifest_sha256": dataset.class_manifest_sha256,
        "split": split,
        "seed": seed,
        "requested_size": size,
        "sample_ids": [record.sample_id for record in selected_records],
    }
    subset_id = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    return DatasetSubset(
        dataset=dataset,
        split=split,
        records=selected_records,
        selection_seed=seed,
        requested_size=size,
        selection_policy="sha256(seed,sample_id) order, round-robin by original generator type; no split reassignment",
        subset_id=subset_id,
        class_support=_subset_support(dataset, selected_records),
    )
