import numpy as np
import torch
from splatkit.core import CameraState
from splatkit.viewer import (
    ViewerState,
    camera_from_viewer_payload,
    camera_to_viewer_payload,
    resolve_viewer_mode,
    select_viewer_camera,
)


def test_select_viewer_camera_preserves_one_camera_batch() -> None:
    camera = CameraState(
        width=torch.tensor([32, 64], dtype=torch.int64),
        height=torch.tensor([24, 48], dtype=torch.int64),
        fov_degrees=torch.tensor([60.0, 45.0], dtype=torch.float32),
        cam_to_world=torch.stack(
            [
                torch.eye(4, dtype=torch.float32),
                2.0 * torch.eye(4, dtype=torch.float32),
            ],
            dim=0,
        ),
    )
    selected = select_viewer_camera(camera, index=1)
    assert selected.width.shape == (1,)
    assert int(selected.width[0].item()) == 64
    assert selected.cam_to_world.shape == (1, 4, 4)


def test_viewer_camera_payload_roundtrip_uses_core_camera_state() -> None:
    camera = CameraState(
        width=torch.tensor([80], dtype=torch.int64),
        height=torch.tensor([60], dtype=torch.int64),
        fov_degrees=torch.tensor([50.0], dtype=torch.float32),
        cam_to_world=torch.eye(4, dtype=torch.float32)[None],
        camera_convention="opencv",
    )
    payload = camera_to_viewer_payload(camera)
    restored = camera_from_viewer_payload(payload)
    assert int(restored.width[0].item()) == 80
    assert int(restored.height[0].item()) == 60
    assert restored.camera_convention == "opencv"
    assert np.allclose(restored.cam_to_world[0].numpy(), np.eye(4))


def test_resolve_viewer_mode_respects_runtime_default() -> None:
    assert resolve_viewer_mode("auto", running_in_notebook=True)
    assert not resolve_viewer_mode("auto", running_in_notebook=False)
    assert resolve_viewer_mode("force_on", running_in_notebook=False)
    assert not resolve_viewer_mode("force_off", running_in_notebook=True)


def test_viewer_state_tracks_training_flags() -> None:
    camera = CameraState(
        width=torch.tensor([80], dtype=torch.int64),
        height=torch.tensor([60], dtype=torch.int64),
        fov_degrees=torch.tensor([50.0], dtype=torch.float32),
        cam_to_world=torch.eye(4, dtype=torch.float32)[None],
    )
    state = ViewerState(camera=camera)
    state.set_training_active(True).set_interaction_active(True)
    assert state.training_active
    assert state.interaction_active
