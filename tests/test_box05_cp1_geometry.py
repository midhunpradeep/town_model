from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from town_model.models.bounded_trainable_box_geometry import (
    BoundedBoxGeometry,
    analytic_endpoint_jacobian,
    decode_bounded_box_raw,
    hosting_frame_minimum_full_width,
    ordinary_minimum_full_width,
    validate_bounded_box_geometry,
)


def _device(device_name: str) -> torch.device:
    if device_name == "cuda":
        assert torch.cuda.is_available(), "CP1 CUDA characterization requires actual CUDA"
        return torch.device("cuda:0")
    return torch.device("cpu")


def _decode(
    raw_width: torch.Tensor,
    raw_center: torch.Tensor,
    width: float = 0.0001,
) -> BoundedBoxGeometry:
    return decode_bounded_box_raw(
        raw_width, raw_center, minimum_full_width=width
    )


def test_ordinary_and_hosting_width_policies_use_the_approved_reserve() -> None:
    assert ordinary_minimum_full_width(1e-4) == 1e-4
    assert hosting_frame_minimum_full_width(1e-4, 1e-4) == 1e-4 + 2.0 * 1e-4
    with pytest.raises(ValueError, match="positive"):
        ordinary_minimum_full_width(0.0)
    with pytest.raises(ValueError, match="less than"):
        ordinary_minimum_full_width(2.0)
    with pytest.raises(ValueError, match="less than"):
        hosting_frame_minimum_full_width(1.0, 0.5)
    with pytest.raises(TypeError, match="real scalar"):
        ordinary_minimum_full_width(True)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_CP1_forward_matches_literal_width_first_equations(device_name: str) -> None:
    device = _device(device_name)
    raw_width = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    raw_center = torch.tensor([[0.5, -0.25, 1.0]], dtype=torch.float32, device=device)
    minimum_width = hosting_frame_minimum_full_width(1e-4, 1e-4)

    result = decode_bounded_box_raw(
        raw_width, raw_center, minimum_full_width=minimum_width
    )
    expected_h = (
        minimum_width + (2.0 - minimum_width) * torch.sigmoid(raw_width)
    ) / 2.0
    expected_c = (1.0 - expected_h) * torch.tanh(raw_center)

    assert result.minimum_full_width == minimum_width
    assert torch.equal(result.half_widths, expected_h)
    assert torch.equal(result.centers, expected_c)
    assert torch.equal(result.lower, expected_c - expected_h)
    assert torch.equal(result.upper, expected_c + expected_h)
    assert result.lower.device == device
    validate_bounded_box_geometry(result)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def test_CP1_supports_empty_class_axis_without_a_zero_dimension_convention() -> None:
    raw_width = torch.empty((0, 3), dtype=torch.float32)
    raw_center = torch.empty((0, 3), dtype=torch.float32)
    result = _decode(raw_width, raw_center)
    validate_bounded_box_geometry(result)
    assert result.centers.shape == (0, 3)
    assert result.half_widths.shape == (0, 3)
    assert result.lower.shape == (0, 3)
    assert result.upper.shape == (0, 3)


def test_CP1_decoder_rejects_malformed_or_nonfinite_raw_inputs() -> None:
    valid = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(TypeError, match="torch.Tensor"):
        decode_bounded_box_raw([[0.0]], valid[:1, :1], minimum_full_width=1e-4)
    with pytest.raises(TypeError, match="float32"):
        decode_bounded_box_raw(valid.to(torch.float64), valid, minimum_full_width=1e-4)
    with pytest.raises(ValueError, match="identical"):
        decode_bounded_box_raw(valid, valid[:, :2], minimum_full_width=1e-4)
    with pytest.raises(ValueError, match="dimension"):
        decode_bounded_box_raw(
            torch.empty((2, 0), dtype=torch.float32),
            torch.empty((2, 0), dtype=torch.float32),
            minimum_full_width=1e-4,
        )
    with pytest.raises(ValueError, match="finite"):
        decode_bounded_box_raw(
            torch.tensor([[float("inf")]], dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.float32),
            minimum_full_width=1e-4,
        )
    with pytest.raises(ValueError, match="0 < w < 2"):
        decode_bounded_box_raw(valid, valid, minimum_full_width=2.0)


def test_CP1_decoder_does_not_clamp_extreme_finite_raw_values() -> None:
    raw_width = torch.tensor([[-40.0, 40.0]], dtype=torch.float32)
    raw_center = torch.tensor([[40.0, -40.0]], dtype=torch.float32)
    result = _decode(raw_width, raw_center)
    expected_h = (0.0001 + (2.0 - 0.0001) * torch.sigmoid(raw_width)) / 2.0
    expected_c = (1.0 - expected_h) * torch.tanh(raw_center)
    assert torch.equal(result.half_widths, expected_h)
    assert torch.equal(result.centers, expected_c)
    assert torch.equal(result.lower, expected_c - expected_h)
    assert torch.equal(result.upper, expected_c + expected_h)


def test_CP1_validator_rejects_float32_rounding_below_exact_configured_floor() -> None:
    raw_width = torch.tensor([[-40.0]], dtype=torch.float32)
    raw_center = torch.zeros_like(raw_width)
    result = _decode(raw_width, raw_center, 1e-4)
    represented_half = float(result.half_widths.item())
    assert represented_half < 1e-4 / 2.0
    with pytest.raises(ValueError, match="half_widths violate"):
        validate_bounded_box_geometry(result)


def test_CP1_validator_rejects_endpoint_subtraction_below_floor_even_when_halfwidth_passes() -> None:
    raw_width = torch.tensor([[-22.175695419311523]], dtype=torch.float32)
    raw_center = torch.tensor([[-4.0]], dtype=torch.float32)
    result = _decode(raw_width, raw_center, 0.01)
    assert result.half_widths.to(torch.float64).item() >= 0.01 / 2.0
    actual_endpoint_width = (
        result.upper.to(torch.float64) - result.lower.to(torch.float64)
    ).item()
    assert actual_endpoint_width < 0.01
    with pytest.raises(ValueError, match="stored endpoint widths"):
        validate_bounded_box_geometry(result)


def test_CP1_validator_checks_actual_corner_and_width_invariants_without_tolerance() -> None:
    raw_width = torch.zeros((1, 2), dtype=torch.float32)
    raw_center = torch.zeros_like(raw_width)
    valid = decode_bounded_box_raw(
        raw_width, raw_center, minimum_full_width=0.5
    )
    validate_bounded_box_geometry(valid)

    with pytest.raises(ValueError, match="inside"):
        validate_bounded_box_geometry(replace(valid, upper=torch.tensor([[0.5, 1.0001]])))
    with pytest.raises(ValueError, match="exactly equal"):
        validate_bounded_box_geometry(replace(valid, lower=valid.lower + 1e-6))
    with pytest.raises(ValueError, match="minimum full width"):
        too_narrow = BoundedBoxGeometry(
            centers=torch.zeros((1, 1), dtype=torch.float32),
            half_widths=torch.tensor([[0.249]], dtype=torch.float32),
            lower=torch.tensor([[-0.249]], dtype=torch.float32),
            upper=torch.tensor([[0.249]], dtype=torch.float32),
            minimum_full_width=0.5,
        )
        validate_bounded_box_geometry(too_narrow)


def test_CP1_validator_distinguishes_nonfinite_and_ordered_corner_failures() -> None:
    raw = torch.zeros((1, 1), dtype=torch.float32)
    valid = _decode(raw, raw)
    with pytest.raises(ValueError, match="finite"):
        validate_bounded_box_geometry(
            replace(valid, centers=torch.tensor([[float("nan")]], dtype=torch.float32))
        )
    with pytest.raises(ValueError, match="lower corners"):
        validate_bounded_box_geometry(
            replace(valid, upper=valid.lower - torch.ones_like(valid.lower))
        )


def test_CP1_jacobian_matches_independent_autograd_smoke() -> None:
    # Diagnostic-only agreement check; this is not an author-approved numeric budget.
    raw_width = torch.tensor(
        [[-4.0, -1.0, 0.0], [0.5, 1.0, 4.0]], dtype=torch.float32, requires_grad=True
    )
    raw_center = torch.tensor(
        [[-2.0, -0.5, 0.25], [1.5, 0.75, 3.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    jacobian = analytic_endpoint_jacobian(
        raw_width, raw_center, minimum_full_width=1e-4
    )
    result = _decode(raw_width, raw_center)
    lower_grads = torch.autograd.grad(
        result.lower.sum(), (raw_width, raw_center), retain_graph=True
    )
    upper_grads = torch.autograd.grad(result.upper.sum(), (raw_width, raw_center))

    autograd_jacobian = torch.stack(
        (
            torch.stack(lower_grads, dim=-1),
            torch.stack(upper_grads, dim=-1),
        ),
        dim=-2,
    )
    assert jacobian.shape == (2, 3, 2, 2)
    assert torch.isfinite(jacobian).all()
    assert torch.allclose(jacobian, autograd_jacobian, atol=2e-6, rtol=0.0)


def test_CP1_jacobian_zero_and_moderate_raw_values_have_expected_structure() -> None:
    raw_width = torch.zeros((1, 2), dtype=torch.float32)
    raw_center = torch.zeros_like(raw_width)
    jacobian = analytic_endpoint_jacobian(
        raw_width, raw_center, minimum_full_width=0.0001
    )
    assert jacobian.shape == (1, 2, 2, 2)
    assert torch.equal(jacobian[..., 0, 1], jacobian[..., 1, 1])
    assert torch.all(jacobian[..., 0, 0] < 0.0)
    assert torch.all(jacobian[..., 1, 0] > 0.0)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_CP1_autograd_jacobian_smoke_on_each_available_device(device_name: str) -> None:
    # Device-specific diagnostic smoke only; cross-device budgets remain pending.
    device = _device(device_name)
    raw_width = torch.tensor([[-3.0, -0.25], [0.75, 3.0]], device=device).requires_grad_()
    raw_center = torch.tensor([[0.5, -1.25], [2.0, -0.75]], device=device).requires_grad_()
    jacobian = analytic_endpoint_jacobian(
        raw_width, raw_center, minimum_full_width=3e-4
    )
    result = decode_bounded_box_raw(
        raw_width, raw_center, minimum_full_width=3e-4
    )
    lower_grad = torch.autograd.grad(
        result.lower.sum(), (raw_width, raw_center), retain_graph=True
    )
    upper_grad = torch.autograd.grad(result.upper.sum(), (raw_width, raw_center))
    observed = torch.stack(
        (
            torch.stack(lower_grad, dim=-1),
            torch.stack(upper_grad, dim=-1),
        ),
        dim=-2,
    )
    assert torch.isfinite(jacobian).all()
    assert torch.allclose(jacobian, observed, atol=2e-6, rtol=0.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def test_CP1_validator_requires_float32_matching_physical_shapes_and_device() -> None:
    raw = torch.zeros((2, 2), dtype=torch.float32)
    valid = _decode(raw, raw)
    with pytest.raises(TypeError, match="float32"):
        validate_bounded_box_geometry(
            replace(valid, upper=valid.upper.to(torch.float64))
        )
    with pytest.raises(ValueError, match=r"same \[C, d\] shape"):
        validate_bounded_box_geometry(replace(valid, lower=valid.lower[:1]))


def test_CP1_geometry_fields_preserve_raw_autograd_history() -> None:
    raw_width = torch.tensor([[0.5, -0.75]], dtype=torch.float32, requires_grad=True)
    raw_center = torch.tensor([[0.25, -0.5]], dtype=torch.float32, requires_grad=True)
    result = _decode(raw_width, raw_center)
    validate_bounded_box_geometry(result)
    (result.lower.sum() + result.upper.sum()).backward()
    assert raw_width.grad is not None
    assert raw_center.grad is not None
    assert torch.isfinite(raw_width.grad).all()
    assert torch.isfinite(raw_center.grad).all()
