from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import torch

from town_model.contracts import FeatureDiagnostics, VisionFeatureBatch
from town_model.models.image_points import ImagePointModel
from town_model.models.image_embeddings import ImageOntologyModel


ROOT = Path(__file__).parents[1]


def _features() -> VisionFeatureBatch:
    generator = torch.Generator().manual_seed(784)
    return VisionFeatureBatch(
        component_vectors=torch.randn((2, 3, 260), generator=generator),
        component_valid_mask=torch.tensor([[True, True, False], [True, False, False]]),
        global_spatial_features=torch.randn((2, 256, 7, 7), generator=generator),
        component_ids=(("a", "b", None), ("c", None, None)),
        diagnostics=FeatureDiagnostics(),
    )


def test_dataset_and_point_imports_do_not_load_concrete_classifiers() -> None:
    script = """
import sys
from town_model.models.image_points import ImagePointModel
from town_model.models.classifier_transport import ClassifierOutput
import town_model.experiments.image_config
import town_model.cli
import town_model.dashboard.app
import town_model.inspection.image_snapshots
assert 'town_model.models.classifiers' not in sys.modules
assert 'town_model.models.geometry' not in sys.modules
assert 'town_model.experiments.synthetic' not in sys.modules
assert 'town_model.experiments.image_runs' not in sys.modules
assert 'town_model.training.image_trainer' not in sys.modules
"""
    environment = os.environ.copy()
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=environment, check=True)


def test_classifier_free_points_match_compatibility_adapter_exactly() -> None:
    features = _features()
    sample_ids = ("sample-1", "sample-2")
    point_model = ImagePointModel(5, pizza_mode="hybrid", seed=21).eval()
    compatibility_model = ImageOntologyModel(
        5,
        ("A", "B"),
        family="box",
        pizza_mode="hybrid",
        seed=21,
        classifier_seed=34,
    ).eval()

    assert tuple(point_model.state_dict()) == tuple(compatibility_model.common_state_dict())
    for name, value in point_model.state_dict().items():
        assert torch.equal(value, compatibility_model.common_state_dict()[name])

    with torch.no_grad():
        points = point_model(features, sample_ids, diagnostics=True)
        result = compatibility_model(features, sample_ids, diagnostics=True)

    assert points.sample_ids == result.sample_ids == sample_ids
    assert points.component_ids == result.component_ids == features.component_ids
    assert points.component_valid_mask.equal(result.component_valid_mask)
    assert points.pizza_mode == result.pizza_mode == "hybrid"
    for name in (
        "component_points",
        "global_pizza_points",
        "composed_pizza_points",
        "final_pizza_points",
    ):
        assert torch.equal(getattr(points, name), getattr(result, name))
    assert not hasattr(points, "contains")
    assert not hasattr(points, "logits")
    assert not hasattr(points, "signed_margin")
    assert result.component_classifier.contains.shape == (2, 3, 2)
    assert result.pizza_classifier.contains.shape == (2, 2)
