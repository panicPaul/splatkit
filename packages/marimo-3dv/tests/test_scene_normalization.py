from __future__ import annotations

import numpy as np

from marimo_3dv.ops.normalization import (
    apply_to_cameras,
    apply_to_points,
    pca_transform_from_points,
    similarity_from_cameras,
)


def test_similarity_from_cameras_aligns_average_up_to_positive_z() -> None:
    camera_to_world = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
    camera_to_world[:, :3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    camera_to_world[0, :3, 3] = np.array([1.0, 0.0, 0.0])
    camera_to_world[1, :3, 3] = np.array([-1.0, 0.0, 0.0])

    transform = similarity_from_cameras(
        camera_to_world,
        center_method="poses",
    )
    transformed_cameras = apply_to_cameras(transform, camera_to_world)

    up_vectors = np.sum(
        transformed_cameras[:, :3, :3] * np.array([0.0, -1.0, 0.0]),
        axis=-1,
    )
    mean_up = up_vectors.mean(axis=0)
    mean_up /= np.linalg.norm(mean_up)

    np.testing.assert_allclose(mean_up, np.array([0.0, 0.0, 1.0]), atol=1e-6)


def test_pca_transform_from_points_uses_viewer_aligned_z_polarity() -> None:
    points = np.array(
        [
            [-2.0, -1.0, 0.0],
            [-2.0, 1.0, 0.2],
            [2.0, -1.0, 0.1],
            [2.0, 1.0, 0.3],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )

    transform = pca_transform_from_points(points)
    transformed_points = apply_to_points(transform, points)

    assert transformed_points[:, 2].max() <= abs(transformed_points[:, 2].min())
