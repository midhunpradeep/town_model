from __future__ import annotations

import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from town_model.dataset.image_data import deterministic_subset, load_prepared_dataset
from town_model.experiments.feature_cache import FrozenFeatureCache, _acquire_cache_lock, prepare_frozen_feature_cache
from town_model.experiments.image_config import (
    DatasetSubsetConfig,
    ExecutionConfig,
    ImageExperimentConfig,
    ImageModelConfig,
    TrainingConfig,
    VisionConfig,
)
from town_model.training.image_trainer import build_image_modules, train_image_model
from town_model.vision.batching import collate_image_records
from town_model.vision.preprocessing import ImageNetLetterbox


def _fixture_config(cache_directory=None):
    return ImageExperimentConfig(
        dataset=DatasetSubsetConfig(dataset_path="examples/pizza-48", train_size=4, validation_size=2, selection_seed=31),
        vision=VisionConfig(visual_policy="frozen"),
        model=ImageModelConfig(family="baseline", pizza_mode="hybrid", latent_dimensions=2),
        training=TrainingConfig(seed=47, epochs=1, physical_batch_size=2, accumulation_steps=1, schedule="constant", warmup_steps=0),
        execution=ExecutionConfig(device="cpu", cpu_threads=2, feature_cache_directory=cache_directory),
        purpose="smoke",
    )


def test_frozen_cache_has_strict_identity_and_only_deterministic_train_validation_panel(tmp_path):
    repository = Path(__file__).parents[1]
    dataset = load_prepared_dataset(repository / "examples" / "pizza-48")
    train = deterministic_subset(dataset, "train", size=4, seed=31)
    validation = deterministic_subset(dataset, "validation", size=2, seed=32)
    config = _fixture_config(str(tmp_path / "cache"))
    visual, _, _ = build_image_modules(dataset, config)
    cache = prepare_frozen_feature_cache(tmp_path / "cache", dataset, train, validation, visual, config, device=torch.device("cpu"))
    assert set(cache.identity["sample_splits"].values()) == {"train", "validation"}
    assert not any(split == "test" for split in cache.identity["sample_splits"].values())
    assert cache.identity["roi_source"] == "ground_truth"
    assert cache.identity["visual_state_sha256"]

    batch = collate_image_records(dataset, train.records[:2], transform=ImageNetLetterbox(config.vision.resolution))
    features = cache.get(batch, diagnostics=True)
    assert features.component_vectors.shape[0] == 2
    assert features.global_spatial_features.shape == (2, 256, 7, 7)
    assert features.diagnostics is not None
    for actual, expected in zip(features.component_ids, batch.component_ids, strict=True):
        assert actual[:len(expected)] == expected

    changed_config = replace(config, vision=replace(config.vision, resolution=256))
    with pytest.raises(ValueError, match="identity fields differ.*preprocessing"):
        train_image_model(
            dataset,
            train,
            validation,
            changed_config,
            tmp_path / "incompatible-cache-run",
            feature_cache=cache,
        )

    manifest = json.loads((cache.directory / "manifest.json").read_text())
    with pytest.raises(ValueError, match="incompatible producer identity"):
        FrozenFeatureCache.open(cache.directory, {**cache.identity, "roi_source": "predicted"})
    sample_entry = next(iter(manifest["samples"].values()))
    artifact = cache.directory / sample_entry["path"]
    artifact.write_bytes(b"corrupt-cache-entry")
    with pytest.raises(ValueError, match="missing or corrupt"):
        FrozenFeatureCache.open(cache.directory, cache.identity)


def test_feature_cache_lock_recovers_a_dead_local_writer_and_rejects_a_live_writer(tmp_path):
    lock = tmp_path / ".cache.lock"
    lock.write_text(json.dumps({"pid": 987654321, "hostname": socket.gethostname()}), encoding="utf-8")
    _acquire_cache_lock(lock)
    assert json.loads(lock.read_text(encoding="utf-8"))["pid"] != 987654321
    with pytest.raises(RuntimeError, match="already creating"):
        _acquire_cache_lock(lock)
    lock.unlink()


def test_frozen_feature_cache_does_not_change_training_trajectory(tmp_path):
    repository = Path(__file__).parents[1]
    dataset = load_prepared_dataset(repository / "examples" / "pizza-48")
    train = deterministic_subset(dataset, "train", size=4, seed=31)
    validation = deterministic_subset(dataset, "validation", size=2, seed=32)
    uncached_config = _fixture_config()
    cached_config = _fixture_config(str(tmp_path / "cache"))
    visual, _, _ = build_image_modules(dataset, cached_config)
    cache = prepare_frozen_feature_cache(tmp_path / "cache", dataset, train, validation, visual, cached_config, device=torch.device("cpu"))
    uncached = train_image_model(dataset, train, validation, uncached_config, tmp_path / "uncached")
    cached = train_image_model(dataset, train, validation, cached_config, tmp_path / "cached", feature_cache=cache)
    uncached_payload = torch.load(uncached.latest_checkpoint, map_location="cpu", weights_only=False)
    cached_payload = torch.load(cached.latest_checkpoint, map_location="cpu", weights_only=False)
    for key, value in uncached_payload["model_state"].items():
        torch.testing.assert_close(value, cached_payload["model_state"][key], rtol=5e-3, atol=3e-5)
    assert uncached_payload["optimizer_step"] == cached_payload["optimizer_step"]
