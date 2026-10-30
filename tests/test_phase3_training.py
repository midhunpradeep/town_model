from __future__ import annotations

from pathlib import Path

import pytest
import torch

from town_model.dataset.image_data import deterministic_subset, load_prepared_dataset
from town_model.experiments.image_config import (
    DatasetSubsetConfig,
    ExecutionConfig,
    ImageExperimentConfig,
    ImageModelConfig,
    TrainingConfig,
    VisionConfig,
)
from town_model.training.image_trainer import build_image_modules, evaluate_image_model, train_image_model


@pytest.fixture(scope="module")
def tiny_real_subsets():
    dataset = load_prepared_dataset(Path(__file__).parents[1] / "examples" / "pizza-48")
    train = deterministic_subset(dataset, "train", size=4, seed=31)
    validation = deterministic_subset(dataset, "validation", size=2, seed=32)
    return dataset, train, validation


def _tiny_config(family="baseline", mode="hybrid", **changes):
    values = dict(
        dataset=DatasetSubsetConfig(dataset_path="examples/pizza-48", train_size=4, validation_size=2, selection_seed=31),
        vision=VisionConfig(visual_policy="frozen"),
        model=ImageModelConfig(family=family, pizza_mode=mode, latent_dimensions=2),
        training=TrainingConfig(
            seed=23, epochs=1, physical_batch_size=2, accumulation_steps=1,
            schedule="constant", warmup_steps=0, exploratory_latest_only=True,
        ),
        execution=ExecutionConfig(device="cpu", cpu_threads=2),
        purpose="smoke",
    )
    values.update(changes)
    return ImageExperimentConfig(**values)


def test_one_real_image_optimizer_update_saves_resumable_training_state(tmp_path, tiny_real_subsets):
    dataset, train, validation = tiny_real_subsets
    result = train_image_model(dataset, train, validation, _tiny_config(), tmp_path / "tiny-run")
    assert result.status == "completed"
    assert result.global_step == 2
    assert result.latest_checkpoint.is_file()
    payload = torch.load(result.latest_checkpoint, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == "image-training-checkpoint-v1"
    assert payload["optimizer_step"] == 2
    assert payload["accumulation_window_open"] is False
    assert payload["identity"]["dataset_id"] == dataset.dataset_id
    assert payload["identity"]["train_subset_id"] == train.subset_id
    assert payload["history"]
    assert payload["history"][0]["gradient_activity"]["component_projection"]["gradient_l2"] > 0
    assert payload["history"][0]["gradient_activity"]["classifier_geometry"]["gradient_l2"] > 0
    assert payload["history"][-1]["validation"]["primary_monitor"] is None or isinstance(payload["history"][-1]["validation"]["primary_monitor"], float)


def test_mid_epoch_resume_matches_uninterrupted_training_trajectory(tmp_path, tiny_real_subsets):
    dataset, train, validation = tiny_real_subsets
    config = _tiny_config(training=TrainingConfig(
        seed=71, epochs=2, physical_batch_size=2, accumulation_steps=1,
        schedule="constant", warmup_steps=0, exploratory_latest_only=True,
    ))
    uninterrupted = train_image_model(dataset, train, validation, config, tmp_path / "uninterrupted")
    calls = 0

    def stop_after_first_update():
        nonlocal calls
        calls += 1
        return calls >= 2

    interrupted = train_image_model(
        dataset, train, validation, config, tmp_path / "resumed", stop_requested=stop_after_first_update
    )
    assert interrupted.status == "stopped"
    assert interrupted.global_step == 1
    resumed = train_image_model(
        dataset,
        train,
        validation,
        config,
        tmp_path / "resumed",
        resume_from=interrupted.latest_checkpoint,
    )
    assert resumed.global_step == uninterrupted.global_step == 4
    resumed_state = torch.load(resumed.latest_checkpoint, map_location="cpu", weights_only=False)
    full_state = torch.load(uninterrupted.latest_checkpoint, map_location="cpu", weights_only=False)
    for key in resumed_state["model_state"]:
        assert torch.equal(resumed_state["model_state"][key], full_state["model_state"][key]), key
    for key, values in resumed_state["optimizer_state"]["state"].items():
        for name, value in values.items():
            other = full_state["optimizer_state"]["state"][key][name]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    for key in resumed_state["visual_state"]:
        assert torch.equal(resumed_state["visual_state"][key], full_state["visual_state"][key]), key


@pytest.mark.parametrize("family", ["baseline", "box", "town"])
@pytest.mark.parametrize("mode", ["global", "compositional", "hybrid"])
def test_nine_real_image_training_smokes_update_save_reload_and_evaluate(tmp_path, tiny_real_subsets, family, mode):
    dataset, train, validation = tiny_real_subsets
    config = _tiny_config(family=family, mode=mode)
    result = train_image_model(dataset, train, validation, config, tmp_path / f"{family}-{mode}")
    assert result.status == "completed"
    assert result.global_step > 0
    payload = torch.load(result.latest_checkpoint, map_location="cpu", weights_only=False)
    visual, model, class_indices = build_image_modules(dataset, config)
    visual.load_state_dict(payload["visual_state"], strict=True)
    model.load_state_dict(payload["model_state"], strict=True)
    validation_after_reload = evaluate_image_model(
        visual, model, validation, config, class_indices=class_indices, device=torch.device("cpu")
    )
    validation_saved = payload["history"][-1]["validation"]
    assert validation_after_reload == validation_saved
    activity = payload["history"][0]["gradient_activity"]
    assert activity["classifier_geometry"]["gradient_l2"] > 0
    assert payload["identity"]["common_neural_state_hash"]
    assert payload["identity"]["visual_initial_state_hash"]


def test_inverse_frequency_weights_upweight_only_known_positive_observations(tiny_real_subsets):
    from town_model.training.image_trainer import _positive_weights

    dataset, train, _ = tiny_real_subsets
    class_ids = (0, 2)
    component = _positive_weights(
        train, dataset, class_ids, level="component", weighting="inverse_frequency", cap=20.0,
        beta=0.99, require_coverage=False,
    )
    pizza = _positive_weights(
        train, dataset, class_ids, level="pizza", weighting="inverse_frequency", cap=20.0,
        beta=0.99, require_coverage=False,
    )
    assert component[1] >= 1.0 and pizza[0] >= 1.0
    assert max(component + pizza) <= 20.0


def test_snapshot_capture_does_not_change_training_trajectory(tmp_path, tiny_real_subsets):
    from town_model.inspection.image_snapshots import save_image_snapshot

    dataset, train, validation = tiny_real_subsets
    config = _tiny_config(training=TrainingConfig(
        seed=99, epochs=2, physical_batch_size=2, accumulation_steps=1,
        schedule="constant", warmup_steps=0, exploratory_latest_only=True,
    ))
    reference = train_image_model(dataset, train, validation, config, tmp_path / "reference")

    def capture(epoch, step, visual, model, _is_final):
        save_image_snapshot(
            tmp_path / "captured" / "snapshots" / f"epoch-{epoch:05d}",
            dataset=dataset,
            train_subset=train,
            validation_subset=validation,
            visual=visual,
            model=model,
            config=config,
            class_ids=dataset.canonical_class_ids,
            class_indices=tuple(range(len(dataset.canonical_class_ids))),
            epoch=epoch,
            optimizer_step=step,
            latest_checkpoint=tmp_path / "captured" / "checkpoints" / "latest.pt",
            identity_hash=config.identity_hash(),
        )

    inspected = train_image_model(
        dataset, train, validation, config, tmp_path / "captured", snapshot_callback=capture
    )
    left = torch.load(reference.latest_checkpoint, map_location="cpu", weights_only=False)
    right = torch.load(inspected.latest_checkpoint, map_location="cpu", weights_only=False)
    for section in ("visual_state", "model_state"):
        for key in left[section]:
            assert torch.equal(left[section][key], right[section][key]), (section, key)
    for key, values in left["optimizer_state"]["state"].items():
        for name, value in values.items():
            assert torch.equal(value, right["optimizer_state"]["state"][key][name]) if isinstance(value, torch.Tensor) else value == right["optimizer_state"]["state"][key][name]


def test_fixed_visual_comparison_pairs_identical_initial_states_across_classifier_families(tiny_real_subsets):
    from town_model.training.image_trainer import _hash_state

    dataset, _, _ = tiny_real_subsets
    configs = [_tiny_config(family=family, mode="hybrid") for family in ("baseline", "box", "town")]
    states = [build_image_modules(dataset, config) for config in configs]
    visual_hashes = {_hash_state(visual.state_dict()) for visual, _, _ in states}
    common_hashes = {_hash_state(model.common_state_dict()) for _, model, _ in states}
    classifier_hashes = {_hash_state(model.classifier.state_dict()) for _, model, _ in states}
    assert len(visual_hashes) == len(common_hashes) == 1
    assert len(classifier_hashes) == 3
    assert [indices for _, _, indices in states] == [tuple(range(len(dataset.canonical_class_ids)))] * 3


def test_trainable_fpn_real_image_run_updates_fpn_and_reports_gradient_activity(tmp_path, tiny_real_subsets):
    dataset, train, validation = tiny_real_subsets
    config = _tiny_config(vision=VisionConfig(visual_policy="trainable_fpn"))
    result = train_image_model(dataset, train, validation, config, tmp_path / "trainable-fpn")
    payload = torch.load(result.latest_checkpoint, map_location="cpu", weights_only=False)
    assert result.global_step > 0
    activity = payload["history"][0]["gradient_activity"]
    assert activity["fpn"]["status"] == "active"
    assert activity["fpn"]["gradient_l2"] > 0
    assert activity["backbone_unfrozen"]["status"] == "frozen"
    from town_model.training.image_trainer import _hash_state

    assert _hash_state(payload["visual_state"]) != payload["identity"]["visual_initial_state_hash"]


def test_invalid_device_fails_before_checkpoint_directory_creation(tmp_path, tiny_real_subsets):
    dataset, train, validation = tiny_real_subsets
    config = _tiny_config(execution=ExecutionConfig(device="cuda", cpu_threads=2))
    with pytest.raises(ValueError, match="CUDA was requested"):
        train_image_model(dataset, train, validation, config, tmp_path / "no-run")
    assert not (tmp_path / "no-run").exists()
