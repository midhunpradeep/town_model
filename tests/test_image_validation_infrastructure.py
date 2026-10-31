from __future__ import annotations

from pathlib import Path

import pytest

from town_model.dataset.image_data import deterministic_subset, load_prepared_dataset
from town_model.experiments.image_validation import validate_image_training_data


def test_image_subset_validation_preserves_canonical_order_and_support_checks() -> None:
    repository = Path(__file__).parents[1]
    dataset = load_prepared_dataset(repository / "examples" / "pizza-48")
    train = deterministic_subset(dataset, "train", size=4, seed=31)
    validation = deterministic_subset(dataset, "validation", size=2, seed=32)
    class_ids = tuple(dataset.canonical_class_ids)

    validate_image_training_data(
        dataset,
        train,
        validation,
        class_ids,
        require_training_coverage=False,
        require_primary_monitor=False,
    )
    with pytest.raises(ValueError, match="canonical trainable class order"):
        validate_image_training_data(
            dataset,
            train,
            validation,
            tuple(reversed(class_ids)),
            require_training_coverage=False,
            require_primary_monitor=False,
        )

    support_ids = tuple(train.class_support["class_ids"])
    zero_component_support = next(
        class_id
        for class_id, count in zip(support_ids, train.class_support["component"]["known_positive"], strict=True)
        if class_id in dataset.component_supervision_class_ids and count == 0
    )
    with pytest.raises(ValueError, match="lacks known-positive component support"):
        validate_image_training_data(
            dataset,
            train,
            validation,
            (zero_component_support,),
            require_training_coverage=True,
            require_primary_monitor=False,
        )
