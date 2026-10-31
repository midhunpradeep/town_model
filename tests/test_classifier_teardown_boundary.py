from __future__ import annotations

import os
import subprocess
import sys

import torch

from town_model.contracts import FeatureDiagnostics, VisionFeatureBatch
from town_model.models.image_points import ImagePointModel


def _features() -> VisionFeatureBatch:
    generator = torch.Generator().manual_seed(784)
    return VisionFeatureBatch(
        component_vectors=torch.randn((2, 3, 260), generator=generator),
        component_valid_mask=torch.tensor([[True, True, False], [True, False, False]]),
        global_spatial_features=torch.randn((2, 256, 7, 7), generator=generator),
        component_ids=(("a", "b", None), ("c", None, None)),
        diagnostics=FeatureDiagnostics(),
    )


def test_dataset_and_point_imports_do_not_load_or_expose_retired_implementations() -> None:
    script = """
import importlib, sys
from town_model.models.image_points import ImagePointModel
import town_model.models as models
import town_model.experiments.image_config
import town_model.cli
import town_model.dashboard.app
import town_model.inspection.image_snapshots
assert not any(name in sys.modules for name in (
    'town_model.models.classifiers', 'town_model.models.geometry',
    'town_model.models.image_embeddings', 'town_model.models.classifier_transport',
    'town_model.training.image_objectives', 'town_model.experiments.synthetic',
    'town_model.dashboard.geometry',
))
for name in ('NeuralBaseline', 'BoxClassifier', 'TownClassifier', 'make_classifier', 'BoxGeometry', 'TownGeometry'):
    assert not hasattr(models, name)
"""
    environment = os.environ.copy()
    source = str(__import__("pathlib").Path(__file__).parents[1] / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], cwd=__import__("pathlib").Path(__file__).parents[1], env=environment, check=True)


def test_image_point_model_preserves_points_and_identity_without_classifier_outputs() -> None:
    features = _features()
    sample_ids = ("sample-1", "sample-2")
    model = ImagePointModel(5, pizza_mode="hybrid", seed=21).eval()
    with torch.no_grad():
        result = model(features, sample_ids, diagnostics=True)

    assert result.sample_ids == sample_ids
    assert result.component_ids == features.component_ids
    assert result.component_valid_mask.equal(features.component_valid_mask)
    assert result.pizza_mode == "hybrid"
    assert result.component_points.shape == (2, 3, 5)
    assert result.global_pizza_points.shape == (2, 5)
    assert result.composed_pizza_points.shape == (2, 5)
    assert result.final_pizza_points.shape == (2, 5)
    assert not hasattr(result, "contains")
    assert not hasattr(result, "logits")
    assert not hasattr(result, "signed_margin")
