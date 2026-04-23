from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from splatkit.core import (
    CameraState,
    GaussianScene2D,
    GaussianScene3D,
    SparseVoxelScene,
)


@pytest.fixture
def cpu_scene() -> GaussianScene3D:
    center_position = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.1],
            [-0.2, 0.1, 0.2],
        ],
        dtype=torch.float32,
    )
    log_scales = torch.full((3, 3), -1.0, dtype=torch.float32)
    quaternion_orientation = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    logit_opacity = torch.tensor([2.0, 1.5, 1.0], dtype=torch.float32)
    feature = torch.zeros((3, 16, 3), dtype=torch.float32)
    feature[:, 0, :] = torch.tensor(
        [
            [0.9, 0.2, 0.2],
            [0.2, 0.9, 0.2],
            [0.2, 0.2, 0.9],
        ],
        dtype=torch.float32,
    )
    return GaussianScene3D(
        center_position=center_position,
        log_scales=log_scales,
        quaternion_orientation=quaternion_orientation,
        logit_opacity=logit_opacity,
        feature=feature,
        sh_degree=0,
    )


@pytest.fixture
def cpu_scene_2d() -> GaussianScene2D:
    center_position = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.1],
            [-0.2, 0.1, 0.2],
        ],
        dtype=torch.float32,
    )
    log_scales = torch.full((3, 2), -1.0, dtype=torch.float32)
    quaternion_orientation = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    logit_opacity = torch.tensor([2.0, 1.5, 1.0], dtype=torch.float32)
    feature = torch.zeros((3, 16, 3), dtype=torch.float32)
    feature[:, 0, :] = torch.tensor(
        [
            [0.9, 0.2, 0.2],
            [0.2, 0.9, 0.2],
            [0.2, 0.2, 0.9],
        ],
        dtype=torch.float32,
    )
    return GaussianScene2D(
        center_position=center_position,
        log_scales=log_scales,
        quaternion_orientation=quaternion_orientation,
        logit_opacity=logit_opacity,
        feature=feature,
        sh_degree=0,
    )


@pytest.fixture
def cpu_sparse_voxel_scene() -> SparseVoxelScene:
    return SparseVoxelScene(
        backend_name="new_cuda",
        active_sh_degree=0,
        max_num_levels=4,
        scene_center=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        scene_extent=torch.tensor([2.0], dtype=torch.float32),
        inside_extent=torch.tensor([1.0], dtype=torch.float32),
        octpath=torch.tensor([[0], [4]], dtype=torch.int64),
        octlevel=torch.tensor([[1], [1]], dtype=torch.int8),
        geo_grid_pts=torch.full((12, 1), -10.0, dtype=torch.float32),
        sh0=torch.zeros((2, 3), dtype=torch.float32),
        shs=torch.zeros((2, 0, 3), dtype=torch.float32),
    )


@pytest.fixture
def cuda_sparse_voxel_scene(
    cpu_sparse_voxel_scene: SparseVoxelScene,
) -> SparseVoxelScene:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for SVRaster native tests.")
    return cpu_sparse_voxel_scene.to(torch.device("cuda"))


@pytest.fixture
def cpu_camera() -> CameraState:
    cam_to_world = torch.eye(4, dtype=torch.float32)[None]
    cam_to_world[:, 2, 3] = 3.0
    return CameraState(
        width=torch.tensor([32], dtype=torch.int64),
        height=torch.tensor([32], dtype=torch.int64),
        fov_degrees=torch.tensor([60.0], dtype=torch.float32),
        cam_to_world=cam_to_world,
        camera_convention="opencv",
    )


@pytest.fixture
def cuda_scene(cpu_scene: GaussianScene3D) -> GaussianScene3D:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for gsplat backend tests.")
    return cpu_scene.to(torch.device("cuda"))


@pytest.fixture
def cpu_visible_scene(cpu_scene: GaussianScene3D) -> GaussianScene3D:
    return replace(
        cpu_scene,
        center_position=cpu_scene.center_position
        + torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32),
    )


@pytest.fixture
def cuda_visible_scene(cpu_visible_scene: GaussianScene3D) -> GaussianScene3D:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for gsplat backend tests.")
    return cpu_visible_scene.to(torch.device("cuda"))


@pytest.fixture
def cuda_scene_2d(cpu_scene_2d: GaussianScene2D) -> GaussianScene2D:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for gsplat backend tests.")
    return cpu_scene_2d.to(torch.device("cuda"))


@pytest.fixture
def cuda_camera(cpu_camera: CameraState) -> CameraState:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for gsplat backend tests.")
    return cpu_camera.to(torch.device("cuda"))


def assert_finite_tensor(tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), "tensor contains non-finite values"
