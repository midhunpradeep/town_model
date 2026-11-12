#!/usr/bin/env python3
"""Run only the author-authorized CNN-00 CP2 probes P6, P7, P4 and P5.

This is isolated diagnostic code: no production edits, data loading, loss,
optimizer, parameter update, classifier, detector, or training path is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from town_model.contracts import (
    ImageBatch,
    ImageGeometry,
    ROIIdentity,
    VisionFeatureBatch,
)
from town_model.experiments.image_config import (
    ExecutionConfig,
    ImageExperimentConfig,
    TrainingConfig,
    VisionConfig,
)
from town_model.experiments.visual_features import build_visual_feature_extractor
from town_model.models.image_points import DeepSetsComposition, ImagePointModel
from town_model.vision.preprocessing import ImageNetLetterbox, normalized_roi_coordinates
from town_model.vision.roi import ExplicitROIProvider

SEED = 2026
DEVICE = torch.device("cpu")
DTYPE = torch.float32
GLOBAL_ATOL = 1e-6
GLOBAL_RTOL = 1e-5
COORD_ATOL = 1e-7
MASKED_ATOL = 0.0
P5_ATOL = 1e-7


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def tensor_summary(value: Tensor, *, near_bound: float = 0.99) -> dict[str, Any]:
    detached = value.detach().cpu()
    finite = bool(torch.isfinite(detached).all())
    if detached.numel() == 0:
        return {
            "shape": list(detached.shape), "dtype": str(detached.dtype), "finite": finite,
            "numel": 0, "exact_zero_count": 0, "nonzero_count": 0,
            "max_abs": 0.0, "l2_norm": 0.0, "min": None, "max": None,
            "near_abs_0_99_count": 0, "near_abs_0_99_fraction": None,
        }
    numeric = detached.to(torch.float64)
    count = int(torch.count_nonzero(detached).item())
    return {
        "shape": list(detached.shape), "dtype": str(detached.dtype), "finite": finite,
        "numel": detached.numel(), "exact_zero_count": detached.numel() - count,
        "nonzero_count": count, "max_abs": float(numeric.abs().max().item()),
        "l2_norm": float(torch.linalg.vector_norm(numeric).item()),
        "min": float(numeric.min().item()), "max": float(numeric.max().item()),
        "near_abs_0_99_count": int((detached.abs() >= near_bound).sum().item()),
        "near_abs_0_99_fraction": float((detached.abs() >= near_bound).float().mean().item()),
    }


def comparison(left: Tensor, right: Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    a, b = left.detach().cpu(), right.detach().cpu()
    delta = (a - b).to(torch.float64)
    return {
        "left_shape": list(a.shape), "right_shape": list(b.shape),
        "max_abs_error": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "l2_error": float(torch.linalg.vector_norm(delta).item()),
        "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        "atol": atol, "rtol": rtol,
    }


def fixed_image_batch() -> tuple[ImageBatch, np.ndarray]:
    """Two deterministic synthetic RGB images letterboxed at 64x64; no files/data."""
    y, x = np.mgrid[0:64, 0:64]
    rgb_images = []
    transformed = []
    geometries = []
    letterbox = ImageNetLetterbox(64)
    for image_index in range(2):
        rgb = np.stack(
            (
                (3 * x + 5 * y + 17 * image_index) % 256,
                (7 * x + 2 * y + 61 * image_index) % 256,
                (5 * x + 11 * y + 29 * image_index) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        rgb_images.append(rgb)
        normalized, geometry = letterbox(Image.fromarray(rgb, mode="RGB"))
        transformed.append(normalized)
        geometries.append(geometry)
    images = torch.stack(transformed).to(device=DEVICE, dtype=DTYPE)
    empty_boxes = tuple(torch.empty((0, 4), dtype=torch.float32) for _ in range(2))
    empty_targets = tuple(torch.empty((0, 1), dtype=torch.float32) for _ in range(2))
    empty_masks = tuple(torch.empty((0, 1), dtype=torch.bool) for _ in range(2))
    batch = ImageBatch(
        images=images,
        original_image_sizes=((64, 64), (64, 64)),
        processed_image_sizes=((64, 64), (64, 64)),
        image_geometries=tuple(geometries),
        sample_ids=("cnn00-cp2-synthetic-0", "cnn00-cp2-synthetic-1"),
        gt_component_boxes_xyxy=empty_boxes,
        component_ids=((), ()),
        component_detector_category_ids=((), ()),
        pizza_targets=torch.zeros((2, 1), dtype=torch.float32),
        pizza_known_masks=torch.zeros((2, 1), dtype=torch.bool),
        component_targets=empty_targets,
        component_known_masks=empty_masks,
        split_identity=("synthetic-fixture", "synthetic-fixture"),
    )
    batch.validate()
    return batch, np.stack(rgb_images)


def p6_features() -> tuple[VisionFeatureBatch, Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    component = (0.05 * torch.randn((2, 3, 260), generator=generator, dtype=DTYPE)).requires_grad_(True)
    global_spatial = (0.05 * torch.randn((2, 256, 7, 7), generator=generator, dtype=DTYPE)).requires_grad_(True)
    mask = torch.tensor(((True, True, False), (True, False, False)), dtype=torch.bool)
    ids = (("p6-s0-c0", "p6-s0-c1", None), ("p6-s1-c0", None, None))
    return VisionFeatureBatch(component, mask, global_spatial, ids), component, global_spatial


def p6_hooks(model: ImagePointModel) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    relu_results: dict[str, Any] = {}
    pre_tanh: dict[str, Tensor] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            def relu_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor, *, hook_name: str = name) -> None:
                value = output.detach()
                relu_results[hook_name] = {
                    "shape": list(value.shape), "positive_count": int((value > 0).sum().item()),
                    "zero_count": int((value == 0).sum().item()), "numel": value.numel(),
                    "positive_fraction": float((value > 0).float().mean().item()) if value.numel() else None,
                    "finite": bool(torch.isfinite(value).all()),
                    "max_abs": float(value.abs().max().item()) if value.numel() else 0.0,
                }
            handles.append(module.register_forward_hook(relu_hook))
        if name in {
            "component_projection.layers.6", "global_projection.output",
            "composition.rho.2", "fusion.output",
        }:
            def linear_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor, *, hook_name: str = name) -> None:
                pre_tanh[hook_name] = output.detach().clone()
            handles.append(module.register_forward_hook(linear_hook))
    return handles, relu_results, pre_tanh


def p6_parameter_group(name: str) -> str:
    return name.split(".", 1)[0]


def p6_run_case(
    mode: str, contraction: str, raw_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    model = ImagePointModel(dimensions=4, pizza_mode=mode, seed=SEED).to(DEVICE).eval()
    features0, component0, global0 = p6_features()
    # Each contraction uses fresh leaf tensors and a fresh forward graph.
    component = component0.detach().clone().requires_grad_(True)
    global_spatial = global0.detach().clone().requires_grad_(True)
    features = VisionFeatureBatch(
        component, features0.component_valid_mask, global_spatial, features0.component_ids,
    )
    handles, relu_results, pre_tanh = p6_hooks(model)
    try:
        result = model(features, ("p6-sample-0", "p6-sample-1"))
        if contraction == "component_points":
            coefficients = torch.tensor((0.31, -0.47, 0.73, -0.89), dtype=DTYPE)[None, None, :]
            row_scales = torch.tensor(((0.77, 1.19, 1.61), (1.37, 0.83, 1.11)), dtype=DTYPE)[:, :, None]
            coeff = coefficients * row_scales
            scalar = (result.component_points * coeff * features.component_valid_mask[:, :, None]).sum()
        else:
            coeff = torch.tensor(((0.23, -0.41, 0.67, -0.89), (-0.59, 0.73, -0.31, 0.97)), dtype=DTYPE)
            scalar = (result.final_pizza_points * coeff).sum()
        named_parameters = list(model.named_parameters())
        gradients = torch.autograd.grad(
            scalar, [parameter for _, parameter in named_parameters] + [component, global_spatial],
            allow_unused=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    param_grads = gradients[:len(named_parameters)]
    component_grad, global_grad = gradients[-2:]
    groups: dict[str, list[dict[str, Any]]] = {
        "component_projection": [], "global_projection": [], "composition": [], "fusion": [],
    }
    group_nonfinite = False
    for (name, _parameter), grad in zip(named_parameters, param_grads, strict=True):
        group = p6_parameter_group(name)
        record: dict[str, Any] = {"name": name, "is_none": grad is None}
        if grad is not None:
            summary = tensor_summary(grad)
            record.update(summary)
            record["sha256"] = tensor_sha256(grad)
            raw_arrays[f"p6__{mode}__{contraction}__parameter__{name}"] = grad.detach().cpu().numpy()
            group_nonfinite = group_nonfinite or not summary["finite"]
        groups[group].append(record)

    group_summaries: dict[str, Any] = {}
    for group, records in groups.items():
        present = [record for record in records if not record["is_none"]]
        group_summaries[group] = {
            "parameter_count": len(records),
            "none_count": len(records) - len(present),
            "present_tensor_count": len(present),
            "all_zero_parameter_tensor_count": sum(record["nonzero_count"] == 0 for record in present),
            "nonzero_element_count": sum(record["nonzero_count"] for record in present),
            "max_abs": max((record["max_abs"] for record in present), default=None),
            "combined_l2_norm": float(np.sqrt(sum(record["l2_norm"] ** 2 for record in present))) if present else None,
            "finite": all(record["finite"] for record in present),
            "per_parameter": records,
        }

    feature_grads: dict[str, Any] = {}
    for name, grad in (("component_features", component_grad), ("global_spatial_features", global_grad)):
        if grad is None:
            feature_grads[name] = {"is_none": True, "value": None}
            continue
        summary = tensor_summary(grad)
        key = f"p6__{mode}__{contraction}__input__{name}"
        raw_arrays[key] = grad.detach().cpu().numpy()
        feature_grads[name] = {"is_none": False, **summary, "sha256": tensor_sha256(grad)}

    masked_count_nonzero = None
    if component_grad is not None:
        invalid = ~features.component_valid_mask
        masked = component_grad[invalid]
        masked_count_nonzero = int(torch.count_nonzero(masked).item())

    outputs = {
        "component_points": result.component_points,
        "global_pizza_points": result.global_pizza_points,
        "composed_pizza_points": result.composed_pizza_points,
        "final_pizza_points": result.final_pizza_points,
    }
    output_summaries = {}
    for name, value in outputs.items():
        output_summaries[name] = tensor_summary(value)
        raw_arrays[f"p6__{mode}__{contraction}__output__{name}"] = value.detach().cpu().numpy()
    pre_tanh_summaries = {}
    for name, value in pre_tanh.items():
        pre_tanh_summaries[name] = tensor_summary(value)
        raw_arrays[f"p6__{mode}__{contraction}__pre_tanh__{name}"] = value.detach().cpu().numpy()

    # Save raw input values once per mode/condition to make the exact fixture auditable.
    raw_arrays[f"p6__{mode}__{contraction}__input__component_features"] = component.detach().cpu().numpy()
    raw_arrays[f"p6__{mode}__{contraction}__input__global_spatial_features"] = global_spatial.detach().cpu().numpy()
    raw_arrays[f"p6__{mode}__{contraction}__coefficients"] = coeff.detach().cpu().numpy()

    return {
        "mode": mode, "contraction": contraction, "scalar_contraction": float(scalar.detach().item()),
        "coefficient_values": coeff.detach().cpu().tolist(),
        "parameter_groups": group_summaries, "feature_gradients": feature_grads,
        "masked_component_input_gradient_nonzero_count": masked_count_nonzero,
        "outputs": output_summaries, "terminal_pre_tanh": pre_tanh_summaries,
        "relu_activity": relu_results,
        "any_nonfinite_gradient": group_nonfinite or any(
            not item.get("finite", True) for item in feature_grads.values() if not item["is_none"]
        ),
    }


def run_p6(raw_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    base, component, global_spatial = p6_features()
    raw_arrays["p6_fixture_component_vectors"] = component.detach().cpu().numpy()
    raw_arrays["p6_fixture_global_spatial_features"] = global_spatial.detach().cpu().numpy()
    raw_arrays["p6_fixture_valid_mask"] = base.component_valid_mask.cpu().numpy()
    outcomes = []
    expected = {
        "component_points": {
            "active": ["component_projection"],
            "inactive": ["global_projection", "composition", "fusion"],
            "component_input": "connected", "global_input": "none", "masked_zero": True,
        },
        "global_final": {
            "active": ["global_projection"],
            "inactive": ["component_projection", "composition", "fusion"],
            "component_input": "none", "global_input": "connected", "masked_zero": False,
        },
        "compositional_final": {
            "active": ["component_projection", "composition"],
            "inactive": ["global_projection", "fusion"],
            "component_input": "connected", "global_input": "none", "masked_zero": True,
        },
        "hybrid_final": {
            "active": ["component_projection", "global_projection", "composition", "fusion"],
            "inactive": [], "component_input": "connected", "global_input": "connected", "masked_zero": True,
        },
    }
    overall_pass = True
    local_activity: dict[str, str] = {}
    for mode in ("global", "compositional", "hybrid"):
        for contraction in ("component_points", "final_pizza_points"):
            key = "component_points" if contraction == "component_points" else f"{mode}_final"
            result = p6_run_case(mode, contraction, raw_arrays)
            expectation = expected[key]
            checks: dict[str, bool] = {}
            for group in expectation["active"]:
                summary = result["parameter_groups"][group]
                checks[f"active_{group}_connected"] = summary["none_count"] == 0
            for group in expectation["inactive"]:
                summary = result["parameter_groups"][group]
                checks[f"inactive_{group}_disconnected"] = summary["none_count"] == summary["parameter_count"]
            for feature_name, expected_state in (
                ("component_features", expectation["component_input"]),
                ("global_spatial_features", expectation["global_input"]),
            ):
                observed_none = result["feature_gradients"][feature_name]["is_none"]
                checks[f"{feature_name}_{expected_state}"] = (
                    observed_none if expected_state == "none" else not observed_none
                )
            if expectation["masked_zero"]:
                checks["masked_component_gradients_exact_zero"] = result["masked_component_input_gradient_nonzero_count"] == 0
            checks["all_gradients_finite"] = not result["any_nonfinite_gradient"]
            all_outputs_finite = all(value["finite"] for value in result["outputs"].values())
            checks["all_point_outputs_finite"] = all_outputs_finite
            outcome = "PASS" if all(checks.values()) else "FAIL"
            all_zero_active_groups = [
                group for group in expectation["active"]
                if result["parameter_groups"][group]["nonzero_element_count"] == 0
            ]
            if all_zero_active_groups and outcome == "PASS":
                # Connectivity still passes, but all-zero activity is not called useful
                # without explaining it via the recorded gate/saturation diagnostics.
                related = result["outputs"]["final_pizza_points"] if contraction == "final_pizza_points" else result["outputs"]["component_points"]
                near_bound = related["near_abs_0_99_fraction"] or 0.0
                dead_relus = [
                    name for name, stats in result["relu_activity"].items()
                    if stats["numel"] and stats["positive_count"] == 0
                ]
                local_activity[f"{mode}/{contraction}"] = (
                    "INCONCLUSIVE_ALL_ZERO_WITH_POTENTIAL_SATURATION_OR_DEAD_RELU"
                    if near_bound > 0 or dead_relus else "INCONCLUSIVE_ALL_ZERO_ACTIVITY_UNEXPLAINED"
                )
            result["expected_paths"] = expectation
            result["invariant_checks"] = checks
            result["invariant_status"] = outcome
            result["all_zero_active_groups"] = all_zero_active_groups
            outcomes.append(result)
            overall_pass = overall_pass and outcome == "PASS"
    return {
        "fixture": {
            "identity": "vision-feature-batch-v1-seed-2026",
            "component_shape": [2, 3, 260], "global_shape": [2, 256, 7, 7],
            "valid_mask": [[True, True, False], [True, False, False]],
            "component_ids": [["p6-s0-c0", "p6-s0-c1", None], ["p6-s1-c0", None, None]],
            "input_scale": 0.05, "dtype": "torch.float32", "device": "cpu", "seed": SEED,
        },
        "invariant_status": "PASS" if overall_pass else "FAIL",
        "cases": outcomes, "local_activity_caveats": local_activity,
        "interpretation_boundary": "Gradient connectivity is not evidence of useful optimization, learning, or scientific performance.",
    }


def module_owner(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else ""


def buffer_subtree_class(name: str, policy: str, trainable_blocks: tuple[int, ...]) -> str:
    if name.startswith("backbone.mobile_features."):
        try:
            index = int(name.split(".")[2])
        except (ValueError, IndexError):
            return "frozen"
        return "trainable" if index in trainable_blocks else "frozen"
    if name.startswith("backbone.fpn."):
        return "trainable" if policy != "frozen" else "frozen"
    return "other"


def run_p7(batch: ImageBatch, raw_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    policies = (
        ("frozen", ()),
        ("trainable_fpn", ()),
        ("trainable_fpn_last_blocks", (16,)),
    )
    policy_results = []
    shared_parameter_hashes: dict[str, str] | None = None
    shared_raw_saved = False
    all_pass = True
    for policy, blocks in policies:
        config = ImageExperimentConfig(
            vision=VisionConfig(
                pretrained_imagenet=False, resolution=64, visual_policy=policy,
                trainable_backbone_blocks=blocks,
            ),
            training=TrainingConfig(seed=SEED),
            execution=ExecutionConfig(device="cpu", cpu_threads=1, num_workers=0),
        ).validate()
        visual = build_visual_feature_extractor(config)
        visual.train()
        modules = dict(visual.named_modules())
        parameters = dict(visual.named_parameters())
        buffers = dict(visual.named_buffers())
        before_param_values = {name: value.detach().cpu().clone() for name, value in parameters.items()}
        before_param_hashes = {name: tensor_sha256(value) for name, value in parameters.items()}
        before_requires_grad = {name: bool(value.requires_grad) for name, value in parameters.items()}
        before_modes = {name: bool(module.training) for name, module in modules.items()}
        before_buffers = {name: value.detach().cpu().clone() for name, value in buffers.items()}
        before_buffer_hashes = {name: tensor_sha256(value) for name, value in buffers.items()}

        if shared_parameter_hashes is None or before_param_hashes != shared_parameter_hashes:
            for name, value in before_param_values.items():
                raw_arrays[f"p7__{policy}__parameter_before__{name}"] = value.numpy()
            shared_parameter_hashes = before_param_hashes
            shared_raw_saved = True
        for name, value in before_buffers.items():
            raw_arrays[f"p7__{policy}__buffer_before__{name}"] = value.numpy()

        bn_forward_calls: dict[str, int] = {}
        handles = []
        for name, module in modules.items():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                bn_forward_calls[name] = 0
                def bn_hook(_module: nn.Module, _inputs: tuple[Any, ...], _output: Tensor, *, hook_name: str = name) -> None:
                    bn_forward_calls[hook_name] += 1
                handles.append(module.register_forward_hook(bn_hook))
        try:
            with torch.inference_mode():
                features = visual(batch)
        finally:
            for handle in handles:
                handle.remove()

        after_parameters = dict(visual.named_parameters())
        after_buffers = dict(visual.named_buffers())
        after_param_hashes = {name: tensor_sha256(value) for name, value in after_parameters.items()}
        after_requires_grad = {name: bool(value.requires_grad) for name, value in after_parameters.items()}
        after_modes = {name: bool(module.training) for name, module in visual.named_modules()}
        after_buffer_values = {name: value.detach().cpu().clone() for name, value in after_buffers.items()}
        after_buffer_hashes = {name: tensor_sha256(value) for name, value in after_buffers.items()}
        for name, value in after_buffer_values.items():
            raw_arrays[f"p7__{policy}__buffer_after__{name}"] = value.numpy()

        parameter_comparisons = {}
        for name, value in after_parameters.items():
            before = before_param_values[name]
            after = value.detach().cpu()
            delta = (after - before).abs()
            equal = bool(torch.equal(before, after))
            parameter_comparisons[name] = {
                "shape": list(before.shape), "dtype": str(before.dtype),
                "before_sha256": before_param_hashes[name], "after_sha256": after_param_hashes[name],
                "torch_equal": equal, "max_abs_delta": float(delta.max().item()) if delta.numel() else 0.0,
                "changed_value_count": int(torch.count_nonzero(delta).item()),
                "requires_grad_before": before_requires_grad[name],
                "requires_grad_after": after_requires_grad[name],
            }

        buffer_comparisons = {}
        for name, after in after_buffer_values.items():
            before = before_buffers[name]
            delta = (after - before).abs()
            owner = module_owner(name)
            bn_module = modules.get(owner)
            subtree = buffer_subtree_class(name, policy, blocks)
            buffer_comparisons[name] = {
                "shape": list(before.shape), "dtype": str(before.dtype),
                "before_sha256": before_buffer_hashes[name], "after_sha256": after_buffer_hashes[name],
                "torch_equal": bool(torch.equal(before, after)),
                "changed_value_count": int(torch.count_nonzero(delta).item()),
                "max_abs_delta": float(delta.max().item()) if delta.numel() else 0.0,
                "subtree_class": subtree,
                "owner_module": owner,
                "is_batchnorm_buffer": isinstance(bn_module, nn.modules.batchnorm._BatchNorm),
                "batchnorm_training": bool(bn_module.training) if isinstance(bn_module, nn.modules.batchnorm._BatchNorm) else None,
                "batchnorm_track_running_stats": bool(bn_module.track_running_stats) if isinstance(bn_module, nn.modules.batchnorm._BatchNorm) else None,
                "forward_call_count": bn_forward_calls.get(owner),
            }

        bn_records = {}
        for name, module in modules.items():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                relevant = {}
                for suffix in ("running_mean", "running_var", "num_batches_tracked"):
                    buffer_name = f"{name}.{suffix}" if name else suffix
                    if buffer_name in buffer_comparisons:
                        relevant[suffix] = buffer_comparisons[buffer_name]
                bn_records[name] = {
                    "training_before": before_modes[name], "training_after": after_modes[name],
                    "track_running_stats": bool(module.track_running_stats),
                    "forward_call_count": bn_forward_calls[name],
                    "buffers": relevant,
                }

        expected_trainable = {}
        policy_block_indices = blocks if policy == "trainable_fpn_last_blocks" else ()
        for name in parameters:
            if name.startswith("backbone.fpn."):
                expected = policy != "frozen"
            elif name.startswith("backbone.mobile_features."):
                try:
                    index = int(name.split(".")[2])
                except (ValueError, IndexError):
                    index = -1
                expected = index in policy_block_indices
            else:
                expected = False
            expected_trainable[name] = expected

        module_policy_checks = {}
        backbone = visual.backbone
        for index, block in enumerate(backbone.mobile_features):
            name = f"backbone.mobile_features.{index}"
            expected_mode = index in policy_block_indices
            module_policy_checks[name] = bool(block.training) == expected_mode
        module_policy_checks["backbone.fpn"] = bool(backbone.fpn.training) == (policy != "frozen")
        mode_preserved = before_modes == after_modes
        requires_preserved = before_requires_grad == after_requires_grad
        params_unchanged = all(item["torch_equal"] for item in parameter_comparisons.values())
        requires_expected = before_requires_grad == expected_trainable
        frozen_buffers_unchanged = all(
            item["torch_equal"] for item in buffer_comparisons.values()
            if item["subtree_class"] == "frozen"
        )
        active_bn_counter_checks = {}
        for name, record in bn_records.items():
            if record["training_before"] and record["track_running_stats"] and record["forward_call_count"] > 0:
                counter_name = f"{name}.num_batches_tracked"
                comp = buffer_comparisons[counter_name]
                before = before_buffers[counter_name]
                after = after_buffer_values[counter_name]
                active_bn_counter_checks[name] = {
                    "expected_delta": 1, "observed_delta": int((after - before).item()),
                    "pass": int((after - before).item()) == 1,
                }
        counters_pass = all(item["pass"] for item in active_bn_counter_checks.values())
        policy_pass = all(module_policy_checks.values())
        checks = {
            "parameters_exactly_unchanged": params_unchanged,
            "requires_grad_unchanged": requires_preserved,
            "requires_grad_matches_policy": requires_expected,
            "module_modes_unchanged_by_forward": mode_preserved,
            "module_train_eval_policy": policy_pass,
            "frozen_subtree_buffers_exactly_unchanged": frozen_buffers_unchanged,
            "tracked_training_batchnorm_counters_increment_once": counters_pass,
            "component_features_shape": list(features.component_vectors.shape) == [2, 0, 260],
            "global_features_shape": list(features.global_spatial_features.shape) == [2, 256, 7, 7],
            "outputs_finite": bool(torch.isfinite(features.global_spatial_features).all()),
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        all_pass = all_pass and status == "PASS"
        policy_results.append({
            "policy": policy, "trainable_backbone_blocks": list(blocks),
            "parameter_count": len(parameters), "buffer_count": len(buffers),
            "parameter_values_before_after": parameter_comparisons,
            "requires_grad_before": before_requires_grad, "requires_grad_after": after_requires_grad,
            "expected_requires_grad": expected_trainable,
            "module_training_modes_before": before_modes, "module_training_modes_after": after_modes,
            "module_policy_checks": module_policy_checks,
            "buffer_values_before_after": buffer_comparisons,
            "batchnorm_modules": bn_records,
            "active_batchnorm_counter_checks": active_bn_counter_checks,
            "feature_summary": {
                "component_vectors": tensor_summary(features.component_vectors),
                "component_valid_mask": features.component_valid_mask.cpu().tolist(),
                "global_spatial_features": tensor_summary(features.global_spatial_features),
            },
            "invariant_checks": checks, "invariant_status": status,
        })
        del visual

    return {
        "fixture": {
            "identity": "two-image-rgb-formula-letterbox64-empty-rois-v1",
            "original_and_processed_size_hw": [[64, 64], [64, 64]],
            "images": ["cnn00-cp2-synthetic-0", "cnn00-cp2-synthetic-1"],
            "roi_count_per_image": [0, 0], "dtype": "torch.float32", "device": "cpu",
            "seed": SEED, "forward_count_per_policy": 1,
        },
        "factory": "build_visual_feature_extractor(ImageExperimentConfig.validate())",
        "pretrained_imagenet": False,
        "forward_mode": "visual.train() then one forward under torch.inference_mode(); no optimizer/update",
        "raw_initial_parameter_arrays_saved": shared_raw_saved,
        "invariant_status": "PASS" if all_pass else "FAIL",
        "policies": policy_results,
    }


def p4_provider(boxes_by_image: list[list[list[float]]], source: str) -> ExplicitROIProvider:
    boxes = tuple(torch.tensor(row, dtype=torch.float32).reshape(-1, 4) for row in boxes_by_image)
    identities = tuple(
        tuple(ROIIdentity(occurrence_id=f"{source}-img{image_index}-roi{roi_index}") for roi_index in range(len(row)))
        for image_index, row in enumerate(boxes_by_image)
    )
    return ExplicitROIProvider(boxes, identities, source=source)


def run_p4(batch: ImageBatch, rgb_images: np.ndarray, raw_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    config = ImageExperimentConfig(
        vision=VisionConfig(pretrained_imagenet=False, resolution=64, visual_policy="frozen"),
        training=TrainingConfig(seed=SEED),
        execution=ExecutionConfig(device="cpu", cpu_threads=1, num_workers=0),
    ).validate()
    visual = build_visual_feature_extractor(config).eval()
    providers = {
        "A": p4_provider([
            [[4, 4, 28, 28], [36, 32, 60, 60]],
            [[4, 4, 28, 28], [36, 32, 60, 60]],
        ], "provider-A"),
        "A_repeat": p4_provider([
            [[4, 4, 28, 28], [36, 32, 60, 60]],
            [[4, 4, 28, 28], [36, 32, 60, 60]],
        ], "provider-A-repeat"),
        "B": p4_provider([
            [[0, 0, 20, 24], [24, 8, 52, 40]],
            [[0, 0, 20, 24], [24, 8, 52, 40]],
        ], "provider-B"),
        "empty": p4_provider([[], []], "provider-empty"),
    }
    features_by_name = {}
    provider_boxes = {
        "A": [[[4, 4, 28, 28], [36, 32, 60, 60]], [[4, 4, 28, 28], [36, 32, 60, 60]]],
        "A_repeat": [[[4, 4, 28, 28], [36, 32, 60, 60]], [[4, 4, 28, 28], [36, 32, 60, 60]]],
        "B": [[[0, 0, 20, 24], [24, 8, 52, 40]], [[0, 0, 20, 24], [24, 8, 52, 40]]],
        "empty": [[], []],
    }
    with torch.inference_mode():
        for name, provider in providers.items():
            features_by_name[name] = visual(batch, provider, diagnostics=True)

    raw_arrays["p4_fixture_rgb_pixels"] = rgb_images
    raw_arrays["p4_fixture_normalized_images"] = batch.images.cpu().numpy()
    raw_arrays["p4_fixture_geometries"] = np.array([
        [*geometry.original_size_hw, *geometry.processed_size_hw, *geometry.resized_size_hw,
         *geometry.scale_xy, *geometry.pad_left_top]
        for geometry in batch.image_geometries
    ], dtype=np.float64)
    coordinate_checks = {}
    finite_checks = {}
    shape_checks = {}
    for name, features in features_by_name.items():
        diagnostic = features.diagnostics
        assert diagnostic is not None and diagnostic.roi_features is not None
        assert diagnostic.global_features is not None and diagnostic.normalized_roi_coordinates is not None
        expected_coordinates = []
        for boxes in provider_boxes[name]:
            tensor_boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            expected_coordinates.append(normalized_roi_coordinates(tensor_boxes, (64, 64)))
        max_rois = max((value.shape[0] for value in expected_coordinates), default=0)
        expected_padded = torch.zeros((2, max_rois, 4), dtype=torch.float32)
        for index, value in enumerate(expected_coordinates):
            if value.numel():
                expected_padded[index, :value.shape[0]] = value
        coord_cmp = comparison(
            diagnostic.normalized_roi_coordinates, expected_padded, atol=COORD_ATOL, rtol=0.0,
        )
        coordinate_checks[name] = coord_cmp
        finite_checks[name] = {
            "pooled_component_features_finite": bool(torch.isfinite(diagnostic.roi_features).all()),
            "global_spatial_features_finite": bool(torch.isfinite(features.global_spatial_features).all()),
            "component_vectors_finite": bool(torch.isfinite(features.component_vectors).all()),
        }
        shape_checks[name] = {
            "component_vectors": list(features.component_vectors.shape),
            "roi_features": list(diagnostic.roi_features.shape),
            "normalized_roi_coordinates": list(diagnostic.normalized_roi_coordinates.shape),
            "global_spatial_features": list(features.global_spatial_features.shape),
            "component_ids": [list(row) for row in features.component_ids],
            "roi_source": diagnostic.roi_source,
        }
        for attribute, tensor in (
            ("component_vectors", features.component_vectors),
            ("component_valid_mask", features.component_valid_mask),
            ("component_ids_not_array", None),
            ("global_spatial_features", features.global_spatial_features),
            ("pooled_roi_features", diagnostic.roi_features),
            ("normalized_roi_coordinates", diagnostic.normalized_roi_coordinates),
        ):
            if tensor is not None:
                raw_arrays[f"p4__{name}__{attribute}"] = tensor.detach().cpu().numpy()

    global_comparisons = {}
    for right in ("A_repeat", "B", "empty"):
        global_comparisons[f"A_vs_{right}"] = comparison(
            features_by_name["A"].global_spatial_features,
            features_by_name[right].global_spatial_features,
            atol=GLOBAL_ATOL, rtol=GLOBAL_RTOL,
        )
    roi_feature_comparisons = {
        "A_vs_B": comparison(
            features_by_name["A"].diagnostics.roi_features,
            features_by_name["B"].diagnostics.roi_features,
            atol=GLOBAL_ATOL, rtol=GLOBAL_RTOL,
        ),
        "A_vs_A_repeat": comparison(
            features_by_name["A"].diagnostics.roi_features,
            features_by_name["A_repeat"].diagnostics.roi_features,
            atol=GLOBAL_ATOL, rtol=GLOBAL_RTOL,
        ),
    }
    empty_shape_pass = (
        list(features_by_name["empty"].component_vectors.shape) == [2, 0, 260]
        and list(features_by_name["empty"].global_spatial_features.shape) == [2, 256, 7, 7]
    )
    all_coords_pass = all(value["allclose"] for value in coordinate_checks.values())
    all_finite = all(all(check.values()) for check in finite_checks.values())
    deterministic_control_pass = global_comparisons["A_vs_A_repeat"]["allclose"]
    if not deterministic_control_pass:
        global_invariance_status = "INCONCLUSIVE"
    else:
        global_invariance_status = "PASS" if all(
            global_comparisons[name]["allclose"] for name in ("A_vs_B", "A_vs_empty")
        ) else "FAIL"
    checks = {
        "coordinates_match_original_pixel_formula": all_coords_pass,
        "all_pooled_global_and_component_features_finite": all_finite,
        "empty_provider_shape_and_global_retained": empty_shape_pass,
        "repeat_A_global_determinism_control": deterministic_control_pass,
        "global_provider_invariance": global_invariance_status,
    }
    statuses = ["PASS" if value is True else value if isinstance(value, str) else "FAIL" for value in checks.values()]
    overall = "FAIL" if "FAIL" in statuses else "INCONCLUSIVE" if "INCONCLUSIVE" in statuses else "PASS"
    return {
        "fixture": {
            "identity": "two-image-rgb-formula-letterbox64-v1",
            "pixel_formula": [
                "R=(3*x+5*y+17*i) mod 256", "G=(7*x+2*y+61*i) mod 256",
                "B=(5*x+11*y+29*i) mod 256",
            ],
            "sample_ids": list(batch.sample_ids), "original_image_size_hw": [[64, 64], [64, 64]],
            "provider_boxes_original_xyxy": provider_boxes,
            "coordinate_formula": "(cx/W, cy/H, w/W, h/H)",
            "coordinate_tolerance": {"atol": COORD_ATOL, "rtol": 0.0},
            "global_tolerance": {"atol": GLOBAL_ATOL, "rtol": GLOBAL_RTOL},
            "global_pairwise_comparisons": global_comparisons,
        },
        "provider_observations": {
            name: {
                "shapes": shape_checks[name], "coordinate_comparison": coordinate_checks[name],
                "finite": finite_checks[name],
                "component_pooled_feature_summary": tensor_summary(features_by_name[name].diagnostics.roi_features),
                "component_coordinate_summary": tensor_summary(features_by_name[name].diagnostics.normalized_roi_coordinates),
                "global_feature_summary": tensor_summary(features_by_name[name].global_spatial_features),
            }
            for name in providers
        },
        "pooled_component_feature_comparisons": roi_feature_comparisons,
        "invariant_checks": checks,
        "invariant_statuses": {
            "coordinates": "PASS" if all_coords_pass else "FAIL",
            "feature_finiteness": "PASS" if all_finite else "FAIL",
            "empty_provider": "PASS" if empty_shape_pass else "FAIL",
            "global_repeat_control": "PASS" if deterministic_control_pass else "FAIL",
            "global_invariance": global_invariance_status,
        },
        "invariant_status": overall,
        "interpretation_boundary": "Component pooled feature differences characterize ROI sensitivity only; no semantic-quality claim is made.",
    }


def configure_p5_witness() -> DeepSetsComposition:
    model = DeepSetsComposition(dimensions=1).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.phi[0].weight[0, 0] = 1.0
        model.phi[2].weight[0, 0] = 1.0
        model.rho[0].weight[0, 0] = 1.0
        model.rho[2].weight[0, 0] = 1.0
    return model


def run_p5(raw_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    model = configure_p5_witness()
    mask = torch.tensor(((True, True, False, False),), dtype=torch.bool)
    first = torch.tensor([[[0.2], [0.4], [-0.7], [0.9]]], dtype=DTYPE, requires_grad=True)
    replacement = torch.tensor([[[0.2], [0.4], [0.8], [-0.6]]], dtype=DTYPE, requires_grad=True)
    first_output = model(first, mask)
    replacement_output = model(replacement, mask)
    first_grad = torch.autograd.grad(first_output.sum(), first)[0]
    replacement_grad = torch.autograd.grad(replacement_output.sum(), replacement)[0]
    mask_output_comparison = comparison(first_output, replacement_output, atol=P5_ATOL, rtol=0.0)
    first_masked_nonzero = int(torch.count_nonzero(first_grad[~mask]).item())
    replacement_masked_nonzero = int(torch.count_nonzero(replacement_grad[~mask]).item())
    valid_gradient_finite = bool(torch.isfinite(first_grad[mask]).all() and torch.isfinite(replacement_grad[mask]).all())
    valid_gradient_nonzero = bool(torch.count_nonzero(first_grad[mask]) > 0 and torch.count_nonzero(replacement_grad[mask]) > 0)

    one = torch.tensor([[[0.25]]], dtype=DTYPE)
    one_mask = torch.tensor(((True,),), dtype=torch.bool)
    duplicate = torch.tensor([[[0.25], [0.25]]], dtype=DTYPE)
    duplicate_mask = torch.tensor(((True, True),), dtype=torch.bool)
    one_output = model(one, one_mask)
    duplicate_output = model(duplicate, duplicate_mask)
    expected_one = torch.tanh(torch.tensor(0.25, dtype=DTYPE))
    expected_duplicate = torch.tanh(torch.tensor(0.5, dtype=DTYPE))
    one_expected_cmp = comparison(one_output.reshape(()), expected_one, atol=P5_ATOL, rtol=0.0)
    duplicate_expected_cmp = comparison(duplicate_output.reshape(()), expected_duplicate, atol=P5_ATOL, rtol=0.0)
    multiplicity_effect = comparison(one_output.reshape(()), duplicate_output.reshape(()), atol=P5_ATOL, rtol=0.0)
    checks = {
        "masked_padding_output_invariance": mask_output_comparison["allclose"],
        "masked_gradient_exact_zero_first": first_masked_nonzero == 0,
        "masked_gradient_exact_zero_replacement": replacement_masked_nonzero == 0,
        "valid_gradient_finite": valid_gradient_finite,
        "valid_gradient_nonzero": valid_gradient_nonzero,
        "one_occurrence_matches_tanh_0_25": one_expected_cmp["allclose"],
        "duplicate_occurrences_match_tanh_0_5": duplicate_expected_cmp["allclose"],
        "duplicate_final_rho_output_differs_for_witness": not multiplicity_effect["allclose"],
    }
    raw_arrays.update({
        "p5_phi_0_weight": model.phi[0].weight.detach().numpy(),
        "p5_phi_2_weight": model.phi[2].weight.detach().numpy(),
        "p5_rho_0_weight": model.rho[0].weight.detach().numpy(),
        "p5_rho_2_weight": model.rho[2].weight.detach().numpy(),
        "p5_mask": mask.numpy(), "p5_input_first": first.detach().numpy(),
        "p5_input_replacement": replacement.detach().numpy(),
        "p5_output_first": first_output.detach().numpy(), "p5_output_replacement": replacement_output.detach().numpy(),
        "p5_gradient_first": first_grad.detach().numpy(), "p5_gradient_replacement": replacement_grad.detach().numpy(),
        "p5_one_occurrence_input": one.numpy(), "p5_one_occurrence_output": one_output.detach().numpy(),
        "p5_duplicate_input": duplicate.numpy(), "p5_duplicate_output": duplicate_output.detach().numpy(),
    })
    return {
        "fixture": {
            "identity": "deepsets-d1-hand-set-nonsaturating-witness-v1",
            "map": "tanh(sum(max(x, 0)))", "dimensions": 1,
            "parameters_zeroed": True,
            "unit_weights": ["phi[0].weight[0,0]", "phi[2].weight[0,0]", "rho[0].weight[0,0]", "rho[2].weight[0,0]"],
            "masked_padding_inputs": {
                "first": first.detach().cpu().tolist(), "replacement": replacement.detach().cpu().tolist(),
                "valid_mask": mask.cpu().tolist(),
            },
            "duplicate_witness_inputs": {"one": one.tolist(), "two": duplicate.tolist()},
            "tolerance": {"atol": P5_ATOL, "rtol": 0.0},
        },
        "masked_padding": {
            "first_output": tensor_summary(first_output),
            "replacement_output": tensor_summary(replacement_output),
            "comparison": mask_output_comparison,
            "first_input_gradient": tensor_summary(first_grad),
            "replacement_input_gradient": tensor_summary(replacement_grad),
            "first_masked_gradient_nonzero_count": first_masked_nonzero,
            "replacement_masked_gradient_nonzero_count": replacement_masked_nonzero,
            "valid_gradients_finite": valid_gradient_finite,
            "valid_gradients_nonzero": valid_gradient_nonzero,
        },
        "multiplicity_witness": {
            "one_output": tensor_summary(one_output), "duplicate_output": tensor_summary(duplicate_output),
            "expected_one_tanh_0_25": float(expected_one.item()),
            "expected_duplicate_tanh_0_5": float(expected_duplicate.item()),
            "one_matches_expected": one_expected_cmp,
            "duplicate_matches_expected": duplicate_expected_cmp,
            "one_vs_duplicate": multiplicity_effect,
            "scope_note": "This hand-set witness shows that multiplicity can affect final rho output; it is not a universal guarantee about learned representations.",
        },
        "invariant_checks": checks,
        "invariant_status": "PASS" if all(checks.values()) else "FAIL",
    }


def environment_record() -> dict[str, Any]:
    try:
        cpu_model = next(
            (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
             if line.lower().startswith("model name")),
            platform.processor(),
        )
    except OSError:
        cpu_model = platform.processor()
    distributions = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--short", "--branch"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head, status = None, None
    commit = head or ""
    source_paths = (
        "src/town_model/contracts.py", "src/town_model/models/image_points.py",
        "src/town_model/vision/features.py", "src/town_model/vision/backbone.py",
        "src/town_model/vision/preprocessing.py", "src/town_model/vision/roi.py",
        "src/town_model/experiments/image_config.py", "src/town_model/experiments/visual_features.py",
    )
    blobs = {}
    for path in source_paths:
        if commit:
            try:
                blobs[path] = subprocess.check_output(["git", "rev-parse", f"{commit}:{path}"], text=True).strip()
            except subprocess.CalledProcessError:
                blobs[path] = None
    return {
        "repository": "midhunpradeep/town_model",
        "source_commit": head,
        "git_status_at_execution": status,
        "diagnostic_script": str(Path(__file__).resolve()),
        "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "relevant_source_blob_ids": blobs,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "os_uname": list(os.uname()) if hasattr(os, "uname") else None,
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "device": str(DEVICE),
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "numpy": np.__version__,
        "pillow": Image.__version__ if hasattr(Image, "__version__") else importlib.metadata.version("Pillow"),
        "cuda_build_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "environment_thread_variables": {
            name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            )
        },
        "all_installed_distributions": distributions,
        "seed": SEED,
        "dtype": "torch.float32",
        "pretrained_imagenet": False,
        "dataset_accessed": False,
        "pretrained_weights_downloaded": False,
        "commands": [
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python tools/cnn00_cp2/run_probes.py --output docs/reconstruction/evidence/cnn00_cp2",
            "python -m pip freeze --all (captured separately in dependencies.txt)",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("CP2 requires exactly one Torch intra-op and inter-op thread")
    if torch.device("cpu") != DEVICE or torch.get_default_dtype() != torch.float32:
        raise RuntimeError("CP2 runtime must use CPU float32")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    raw_arrays: dict[str, np.ndarray] = {}
    batch, rgb_pixels = fixed_image_batch()
    results = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": SEED, "device": str(DEVICE), "dtype": "torch.float32",
        "threads": {"torch_intra_op": torch.get_num_threads(), "torch_inter_op": torch.get_num_interop_threads()},
        "p6_gradient_connectivity": run_p6(raw_arrays),
        "p7_frozen_state": run_p7(batch, raw_arrays),
        "p4_roi_isolation": run_p4(batch, rgb_pixels, raw_arrays),
        "p5_deepsets_composition": run_p5(raw_arrays),
    }
    raw_path = output / "raw_observations.npz"
    np.savez_compressed(raw_path, **raw_arrays)
    results["raw_observations"] = {
        "path": str(raw_path), "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "array_count": len(raw_arrays), "array_keys": sorted(raw_arrays),
    }
    (output / "observations.json").write_text(
        json.dumps(jsonable(results), indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    environment = environment_record()
    (output / "environment.json").write_text(
        json.dumps(jsonable(environment), indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    statuses = {
        "P6": results["p6_gradient_connectivity"]["invariant_status"],
        "P7": results["p7_frozen_state"]["invariant_status"],
        "P4": results["p4_roi_isolation"]["invariant_status"],
        "P5": results["p5_deepsets_composition"]["invariant_status"],
    }
    print(json.dumps({"statuses": statuses, "environment": {
        "source_commit": environment["source_commit"], "python": environment["python"].split()[0],
        "torch": environment["torch"], "torchvision": environment["torchvision"],
        "numpy": environment["numpy"], "device": environment["device"],
        "torch_threads": [environment["torch_num_threads"], environment["torch_num_interop_threads"]],
    }, "raw_observations_sha256": results["raw_observations"]["sha256"]}, indent=2))
    return 0 if all(status == "PASS" for status in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
