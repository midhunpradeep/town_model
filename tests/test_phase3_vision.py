from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from town_model.contracts import ROIIdentity
from town_model.dataset.image_data import deterministic_subset, load_prepared_dataset
from town_model.vision.backbone import MobileNetV3LargeFPN
from town_model.vision.batching import collate_image_records
from town_model.vision.features import SharedVisualFeatureExtractor
from town_model.vision.preprocessing import ImageNetLetterbox, inverse_boxes_xyxy, transform_boxes_xyxy
from town_model.vision.roi import ExplicitROIProvider


@pytest.fixture(scope="module")
def prepared():
    return load_prepared_dataset(Path(__file__).parents[1] / "examples" / "pizza-48")


@pytest.fixture(scope="module")
def extracted(prepared):
    torch.set_num_threads(2)
    selected = deterministic_subset(prepared, "train", size=2, seed=2)
    batch = collate_image_records(prepared, selected.records)
    extractor = SharedVisualFeatureExtractor(MobileNetV3LargeFPN(pretrained=True))
    extractor.backbone.set_trainable(trainable_fpn=False)
    extractor.eval()
    with torch.no_grad():
        features = extractor(batch, diagnostics=True)
    return batch, features, extractor


def test_preprocessing_letterboxes_non_square_rgb_once_and_roundtrips_boxes():
    image = Image.new("RGBA", (400, 200), (255, 128, 0, 50))
    transform = ImageNetLetterbox(224)
    tensor, geometry = transform(image)
    assert tensor.shape == (3, 224, 224)
    assert geometry.original_size_hw == (200, 400)
    assert geometry.resized_size_hw == (112, 224)
    assert geometry.pad_left_top == (0, 56)
    box = torch.tensor([[100.0, 40.0, 150.0, 80.0], [0.1, 0.2, 0.5, 0.7], [0.0, 0.0, 400.0, 200.0]])
    processed = transform_boxes_xyxy(box, geometry)
    assert torch.allclose(processed[0], torch.tensor([56.0, 78.4, 84.0, 100.8]), atol=1e-5)
    assert torch.allclose(inverse_boxes_xyxy(processed, geometry), box, atol=1e-5)
    assert transform_boxes_xyxy(torch.empty((0, 4)), geometry).shape == (0, 4)
    with pytest.raises(ValueError, match="positive area"):
        transform_boxes_xyxy(torch.tensor([[3.0, 2.0, 2.0, 8.0]]), geometry)
    with pytest.raises(TypeError, match="PIL image"):
        transform(tensor)  # double normalization/resizing is not an accepted input path


def test_mobilenet_fpn_has_expected_stages_channels_strides_and_roi_features(extracted):
    batch, features, extractor = extracted
    maps = extractor.backbone(batch.images)
    assert tuple(maps) == ("0", "1", "2", "3", "pool")
    assert [tuple(maps[name].shape[-2:]) for name in ("0", "1", "2", "3")] == [(56, 56), (28, 28), (14, 14), (7, 7)]
    assert all(value.shape[1] == 256 for value in maps.values())
    assert features.component_vectors.shape == (2, max(len(x) for x in batch.component_ids), 260)
    assert features.global_spatial_features.shape == (2, 256, 7, 7)
    assert features.component_valid_mask.sum().item() == sum(map(len, batch.component_ids))
    assert torch.isfinite(features.component_vectors).all()
    assert features.diagnostics is not None
    assert features.diagnostics.normalized_roi_coordinates is not None
    assert torch.all(features.diagnostics.normalized_roi_coordinates[features.component_valid_mask] >= 0)
    assert torch.all(features.diagnostics.normalized_roi_coordinates[features.component_valid_mask] <= 1)


def test_roi_provider_is_replaceable_and_global_features_do_not_depend_on_component_count(extracted):
    batch, _, extractor = extracted
    boxes = tuple(value[:1].clone() for value in batch.gt_component_boxes_xyxy)
    ids = tuple((ROIIdentity(f"predicted/{sample_id}", detector_category_id=1, confidence=0.8),) for sample_id in batch.sample_ids)
    provider = ExplicitROIProvider(boxes, ids, source="controlled_predicted_roi_test")
    with torch.no_grad():
        features = extractor(batch, provider, diagnostics=True)
    assert features.component_valid_mask.sum().item() == len(batch.sample_ids)
    assert features.component_ids == tuple((item[0].occurrence_id,) for item in ids)
    assert features.diagnostics is not None and features.diagnostics.roi_source == "controlled_predicted_roi_test"
    assert features.global_spatial_features.shape == (len(batch.sample_ids), 256, 7, 7)


def test_roi_align_accepts_small_off_center_and_boundary_adjacent_arbitrary_boxes(extracted):
    batch, _, extractor = extracted
    first = torch.tensor([[0.2, 0.3, 1.0, 1.1], [3.0, 4.0, 39.0, 47.0], [107.0, 121.0, 223.0, 224.0]], dtype=torch.float32)
    second = torch.empty((0, 4), dtype=torch.float32)
    boxes = (first, second)
    identities = ((ROIIdentity("small"), ROIIdentity("top_left"), ROIIdentity("bottom_right")), ())
    with torch.no_grad():
        features = extractor(batch, ExplicitROIProvider(boxes, identities, source="coordinate_test"), diagnostics=True)
    assert features.component_vectors.shape == (2, 3, 260)
    assert features.component_valid_mask.tolist() == [[True, True, True], [False, False, False]]
    assert torch.isfinite(features.component_vectors).all()
    assert features.diagnostics is not None
    assert features.diagnostics.normalized_roi_coordinates[0, 0].tolist() == pytest.approx([0.6 / 224, 0.7 / 224, 0.8 / 224, 0.8 / 224])


def test_empty_roi_set_preserves_global_branch_and_has_zero_component_width(extracted):
    batch, _, extractor = extracted
    count = len(batch.sample_ids)
    empty = tuple(torch.empty((0, 4), dtype=torch.float32) for _ in range(count))
    identities = tuple(() for _ in range(count))
    provider = ExplicitROIProvider(empty, identities)
    with torch.no_grad():
        features = extractor(batch, provider)
    assert features.component_vectors.shape == (count, 0, 260)
    assert features.component_valid_mask.shape == (count, 0)
    assert features.global_spatial_features.shape == (count, 256, 7, 7)


def test_frozen_visual_path_and_trainable_fpn_gradient_activity(extracted):
    batch, _, extractor = extracted
    assert all(not parameter.requires_grad for parameter in extractor.backbone.mobile_features.parameters())
    assert all(not parameter.requires_grad for parameter in extractor.backbone.fpn.parameters())
    trainable = SharedVisualFeatureExtractor(MobileNetV3LargeFPN(pretrained=True))
    trainable.backbone.set_trainable(trainable_fpn=True)
    trainable.train()
    visual = trainable(batch)
    loss = visual.component_vectors[visual.component_valid_mask].square().mean() + visual.global_spatial_features.square().mean()
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in trainable.backbone.fpn.parameters())
    assert all(parameter.grad is None for parameter in trainable.backbone.mobile_features.parameters())
