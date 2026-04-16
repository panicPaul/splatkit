"""Initialization helpers for declarative training pipelines."""

from .initialize import (
    InitializedModel,
    initialize_gaussian_model_from_dataset,
    initialize_gaussian_scene_from_point_cloud,
)

__all__ = [
    "InitializedModel",
    "initialize_gaussian_model_from_dataset",
    "initialize_gaussian_scene_from_point_cloud",
]
