from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import town_model.experiments.synthetic as synthetic
from town_model.experiments.synthetic import SyntheticConfig, run_experiment
from town_model.models.geometry import BoxGeometry, TownGeometry


_RESERVE = 16.0 * torch.finfo(torch.float32).eps
_BOX_LIMIT = 2.0 - 2.0 * _RESERVE
_TOWN_LIMIT = (2.0 - 5.0 * _RESERVE) / 3.0


def test_composed_config_rejects_infeasible_width_before_fixture_training_or_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SyntheticConfig(seed=2, points=30, steps=1, min_width=0.7)
    with pytest.raises(ValueError, match=r"min_width.*contained town"):
        config.validate()

    fixture_builder = Mock(wraps=synthetic.make_fixture)
    family_trainer = Mock(wraps=synthetic._run_one)
    monkeypatch.setattr(synthetic, "make_fixture", fixture_builder)
    monkeypatch.setattr(synthetic, "_run_one", family_trainer)

    output_root = tmp_path / "invalid-runs"
    with pytest.raises(ValueError, match=r"min_width.*contained town"):
        run_experiment("fixed", "simple", output_root, config=config)

    fixture_builder.assert_not_called()
    family_trainer.assert_not_called()
    assert not output_root.exists()
    assert not list(tmp_path.rglob("checkpoint.pt"))
    assert not list(tmp_path.rglob("metrics.json"))


def test_default_small_and_composed_town_boundary_widths_validate() -> None:
    assert SyntheticConfig().min_width == pytest.approx(1e-5)
    SyntheticConfig().validate()

    for width in (1e-8, _TOWN_LIMIT - 1e-6):
        config = SyntheticConfig(min_width=width)
        config.validate()
        BoxGeometry(2, 1, min_width=width)
        TownGeometry(2, 1, min_width=width)


def test_geometry_and_composed_config_agree_at_family_feasibility_limits() -> None:
    # Pure boxes have the wider domain-width limit.
    BoxGeometry(2, 1, min_width=_BOX_LIMIT - 1e-6)
    for width in (_BOX_LIMIT, _BOX_LIMIT + 1e-6):
        with pytest.raises(ValueError, match="pure box"):
            BoxGeometry(2, 1, min_width=width)

    # The synthetic harness always includes contained towns, so its limit is stricter.
    near_limit_config = SyntheticConfig(min_width=_TOWN_LIMIT - 1e-6)
    near_limit_config.validate()
    near_limit_town = TownGeometry(2, 1, min_width=_TOWN_LIMIT - 1e-6)
    frame_low, frame_high, window_low, window_high = near_limit_town.physical_geometry()
    requested = _TOWN_LIMIT - 1e-6
    assert torch.all(frame_high - frame_low >= requested)
    assert torch.all(window_high - window_low >= requested)
    assert torch.all(window_low - frame_low[:, :, None, :] >= requested)
    assert torch.all(frame_high[:, :, None, :] - window_high >= requested)

    for width in (_TOWN_LIMIT, _TOWN_LIMIT + 1e-6):
        with pytest.raises(ValueError, match="contained town"):
            TownGeometry(2, 1, min_width=width)
        with pytest.raises(ValueError, match="contained town"):
            SyntheticConfig(min_width=width).validate()


def test_invalid_min_width_values_are_rejected_by_config_and_constructors() -> None:
    invalid_values = (
        0,
        -1e-5,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "1e-5",
        None,
        1 + 2j,
    )
    for width in invalid_values:
        with pytest.raises(ValueError, match="min_width"):
            SyntheticConfig(min_width=width).validate()
        with pytest.raises(ValueError, match="min_width"):
            BoxGeometry(2, 1, min_width=width)
        with pytest.raises(ValueError, match="min_width"):
            TownGeometry(2, 1, min_width=width)

    # Standard real scalar types remain accepted rather than being rejected as
    # non-Python numeric objects.
    SyntheticConfig(min_width=np.float64(1e-5)).validate()
    BoxGeometry(2, 1, min_width=np.float64(1e-5))
    TownGeometry(2, 1, min_width=np.float64(1e-5))


def test_box_and_town_validate_their_actual_window_parameterization() -> None:
    BoxGeometry(2, 1, min_width=0.7)
    with pytest.raises(ValueError, match="contained town"):
        TownGeometry(2, 1, min_width=0.7, window_mode="contained")
    with pytest.raises(ValueError, match="contained"):
        TownGeometry(2, 1, min_width=1e-5, window_mode="independent")
    with pytest.raises(ValueError, match="contained"):
        SyntheticConfig(min_width=1e-5, window_mode="independent").validate()
