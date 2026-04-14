from __future__ import annotations

import pytest
import torch
from splatkit.core import CameraState, GaussianScene


@pytest.fixture
def cpu_scene() -> GaussianScene:
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
    return GaussianScene(
        center_position=center_position,
        log_scales=log_scales,
        quaternion_orientation=quaternion_orientation,
        logit_opacity=logit_opacity,
        feature=feature,
        sh_degree=0,
    )


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
def cuda_scene(cpu_scene: GaussianScene) -> GaussianScene:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for gsplat backend tests.")
    return cpu_scene.to(torch.device("cuda"))


@pytest.fixture
def cuda_camera(cpu_camera: CameraState) -> CameraState:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for gsplat backend tests.")
    return cpu_camera.to(torch.device("cuda"))


def assert_finite_tensor(tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), "tensor contains non-finite values"
