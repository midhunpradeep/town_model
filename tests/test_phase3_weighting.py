from __future__ import annotations

import math

import pytest

from town_model.training.image_trainer import _positive_weights_from_counts


def _weights(counts, *, policy="inverse_sqrt", cap=100.0, beta=0.99, required=None):
    return _positive_weights_from_counts(
        counts,
        weighting=policy,
        cap=cap,
        beta=beta,
        require_coverage=False,
        required_mask=required,
    )


def test_inverse_sqrt_uses_known_positive_count_ratio_and_common_unit_scale():
    equal = _weights([24, 24])
    assert equal == (1.0, 1.0)

    actual = _weights([24, 1404])
    expected_ratio = math.sqrt(1404 / 24)
    assert actual == pytest.approx((expected_ratio, 1.0))
    assert actual[0] / actual[1] == pytest.approx(7.64852927)

    # The weighting resolver receives n+ only; changing negative imbalance
    # ratios cannot affect the vector because n- is not an input.
    assert _weights([24, 1404]) == _weights([24, 1404])


def test_effective_number_formula_is_hand_computable_for_two_betas():
    for beta in (0.9, 0.99):
        counts = [2, 10, 40]
        raw = [(1 - beta) / (1 - beta**count) for count in counts]
        scale = 1 / raw[-1]
        expected = tuple(min(100.0, value * scale) for value in raw)
        assert _weights(counts, policy="effective_number", beta=beta) == pytest.approx(expected)
    assert _weights([2, 10], policy="effective_number", beta=0.0) == (1.0, 1.0)


def test_weighting_semantic_version_rejects_legacy_checkpoint_identity(tmp_path):
    import hashlib
    import json
    import torch

    from town_model.experiments.image_config import ImageExperimentConfig, ImageModelConfig
    from town_model.training.image_trainer import ImageTrainingError, _load_checkpoint

    config = ImageExperimentConfig(
        model=ImageModelConfig(family="box"),
    ).validate()
    current_hash = config.identity_hash()
    legacy_identity_dict = config.identity_dict()
    legacy_objective = dict(legacy_identity_dict["resolved_objective"])
    legacy_objective.pop("positive_weighting_semantics")
    legacy_objective.pop("positive_weighting_formula")
    legacy_identity_dict["resolved_objective"] = legacy_objective
    legacy_hash = hashlib.sha256(json.dumps(legacy_identity_dict, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    assert legacy_hash != current_hash

    legacy_path = tmp_path / "legacy-weighting-checkpoint.pt"
    torch.save({
        "schema_version": "image-training-checkpoint-v1",
        "identity": {"config_hash": legacy_hash},
        "accumulation_window_open": False,
    }, legacy_path)
    with pytest.raises(ImageTrainingError, match="incompatible.*identity fields differ.*config_hash"):
        _load_checkpoint(legacy_path, {"config_hash": current_hash}, torch.device("cpu"))


def test_cap_zero_support_none_and_explicit_experimental_alternative():
    capped = _weights([1, 10000], cap=3.0)
    assert capped == (3.0, 1.0)

    exploratory_zero_support = _weights([0, 24], required=(False, True))
    assert exploratory_zero_support == (1.0, 1.0)
    with pytest.raises(ValueError, match="zero known-positive support"):
        _positive_weights_from_counts(
            [0, 24], weighting="inverse_sqrt", cap=10.0, beta=0.99,
            require_coverage=True, required_mask=(True, True),
        )

    assert _weights([1, 4], policy="experimental_inverse_frequency") == pytest.approx((4.0, 1.0))
    assert _weights([1, 4], policy="none") == (1.0, 1.0)
    with pytest.raises(ValueError, match="unsupported positive weighting policy"):
        _weights([1, 4], policy="inverse_frequency")
