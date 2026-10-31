"""Explicit unavailable boundaries for retired image classifier operations.

Visual feature construction lives in :mod:`town_model.vision.factory`; data
validation lives in :mod:`town_model.experiments.image_validation`. This module
contains no training, scoring, checkpoint reconstruction, or geometry logic.
"""

from __future__ import annotations

from typing import NoReturn

from town_model.classifier_backend import raise_classifier_backend_unavailable


def build_image_modules(*args, **kwargs) -> NoReturn:
    raise_classifier_backend_unavailable("historical image classifier module construction")


def train_image_model(*args, **kwargs) -> NoReturn:
    raise_classifier_backend_unavailable("image-to-ontology classifier training")


def evaluate_image_model(*args, **kwargs) -> NoReturn:
    raise_classifier_backend_unavailable("historical image classifier evaluation")


def evaluate_image_checkpoint(*args, **kwargs) -> NoReturn:
    raise_classifier_backend_unavailable("historical image classifier checkpoint evaluation")
