import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation
from splatkit.core import (
    CameraState,
    GaussianScene,
    camera_params_to_intrinsics,
    intrinsics_to_camera_params,
)


def test_camera_intrinsics_roundtrip_shape() -> None:
    width = torch.tensor([32, 64], dtype=torch.int64)
    height = torch.tensor([24, 48], dtype=torch.int64)
    fov_degrees = torch.tensor([60.0, 45.0], dtype=torch.float32)

    intrinsics = camera_params_to_intrinsics(width, height, fov_degrees)
    params = intrinsics_to_camera_params(intrinsics)

    assert intrinsics.shape == (2, 3, 3)
    assert params.width.shape == width.shape
    assert params.height.shape == height.shape
    assert params.fov_degrees.shape == fov_degrees.shape
    assert torch.equal(params.width, width)
    assert torch.equal(params.height, height)


def test_camera_state_to_moves_all_tensors(cpu_camera: CameraState) -> None:
    moved = cpu_camera.to(torch.device("cpu"))
    assert moved.width.device.type == "cpu"
    assert moved.height.device.type == "cpu"
    assert moved.fov_degrees.device.type == "cpu"
    assert moved.cam_to_world.device.type == "cpu"


def test_gaussian_scene_to_moves_all_tensors(cpu_scene: GaussianScene) -> None:
    moved = cpu_scene.to(torch.device("cpu"))
    assert moved.center_position.device.type == "cpu"
    assert moved.log_scales.device.type == "cpu"
    assert moved.quaternion_orientation.device.type == "cpu"
    assert moved.logit_opacity.device.type == "cpu"
    assert moved.feature.device.type == "cpu"


def test_camera_state_construction_validates_tensor_shape() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        CameraState(
            width=torch.tensor([32], dtype=torch.int64),
            height=torch.tensor([32], dtype=torch.int64),
            fov_degrees=torch.tensor([60.0], dtype=torch.float32),
            cam_to_world=torch.eye(4, dtype=torch.float32),
        )


def test_gaussian_scene_construction_validates_feature_shape() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        GaussianScene(
            center_position=torch.zeros((3, 3), dtype=torch.float32),
            log_scales=torch.zeros((3, 3), dtype=torch.float32),
            quaternion_orientation=torch.zeros((3, 3), dtype=torch.float32),
            logit_opacity=torch.zeros((3,), dtype=torch.float32),
            feature=torch.zeros((3, 16), dtype=torch.float32),
            sh_degree=0,
        )
