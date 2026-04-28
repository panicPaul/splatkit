"""Built-in setup-pipeline ops wrapping scene normalization utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

from jaxtyping import Float
from numpy import ndarray

from marimo_3dv.ops.normalization import (
    apply_rotation_to_quaternions,
    apply_rotation_to_sh_coefficients,
    apply_scale_to_log_scales,
    apply_to_cameras,
    apply_to_points,
    compose_transforms,
    pca_transform_from_points,
    similarity_from_cameras,
)


@runtime_checkable
class HasCameraToWorld(Protocol):
    """Protocol for scene data that exposes camera-to-world matrices."""

    @property
    def camera_to_world(self) -> Float[ndarray, "N 4 4"]:
        """Return camera-to-world matrices."""
        ...


@runtime_checkable
class HasPoints(Protocol):
    """Protocol for scene data that exposes a point cloud."""

    @property
    def points(self) -> Float[ndarray, "N 3"]:
        """Return point positions."""
        ...


def camera_similarity_op(
    *,
    center_method: Literal["focus", "poses"] = "focus",
    strict_scaling: bool = False,
) -> Callable[
    [HasCameraToWorld], tuple[Float[ndarray, "4 4"], HasCameraToWorld]
]:
    """Return a setup op that computes a similarity transform from camera poses.

    The op returns ``(transform, scene)`` so downstream ops can apply it to
    points, quaternions, and SH coefficients without recomputing.

    Args:
        center_method: ``"focus"`` (default) or ``"poses"``.
        strict_scaling: If True, scale so the scene fits inside the unit sphere.

    Returns:
        A setup op callable.
    """

    def op(
        scene: HasCameraToWorld,
    ) -> tuple[Float[ndarray, "4 4"], HasCameraToWorld]:
        """Compute camera-derived similarity transform."""
        transform = similarity_from_cameras(
            scene.camera_to_world,
            center_method=center_method,
            strict_scaling=strict_scaling,
        )
        return transform, scene

    return op


def pca_alignment_op(
    points: Float[ndarray, "N 3"],
) -> Callable[
    [tuple[Float[ndarray, "4 4"], HasCameraToWorld]],
    tuple[Float[ndarray, "4 4"], HasCameraToWorld],
]:
    """Return a setup op that refines orientation via PCA on a point cloud.

    Chains the PCA transform onto the existing similarity transform so that
    principal axes of the point cloud align with the coordinate axes.

    Args:
        points: (N, 3) reference point cloud for PCA alignment.

    Returns:
        A setup op callable.
    """

    def op(
        state: tuple[Float[ndarray, "4 4"], HasCameraToWorld],
    ) -> tuple[Float[ndarray, "4 4"], HasCameraToWorld]:
        """Refine transform via PCA alignment."""
        transform, scene = state
        aligned_points = apply_to_points(transform, points)
        pca_transform = pca_transform_from_points(aligned_points)
        combined = compose_transforms(pca_transform, transform)
        return combined, scene

    return op


__all__ = [
    "HasCameraToWorld",
    "HasPoints",
    "apply_rotation_to_quaternions",
    "apply_rotation_to_sh_coefficients",
    "apply_scale_to_log_scales",
    "apply_to_cameras",
    "apply_to_points",
    "camera_similarity_op",
    "compose_transforms",
    "pca_alignment_op",
    "pca_transform_from_points",
    "similarity_from_cameras",
]
