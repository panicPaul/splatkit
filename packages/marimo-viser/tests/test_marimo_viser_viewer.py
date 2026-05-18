"""Tests for the marimo-viser Ember-camera viewer contract."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ember_core.core.contracts import CameraState
from marimo_viser import (
    NoopViserViewer,
    ViserRenderConfig,
    ViserRenderState,
    ViserServerConfig,
    ViserViewerState,
    apply_viser_config,
)
from marimo_viser.viewer import (
    _camera_state_from_viser,
    _core_camera_to_viser_pose,
    _numpy_image,
    connection_info,
)


def test_viser_camera_conversion_returns_ember_camera_state() -> None:
    camera = SimpleNamespace(
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        position=np.array([1.0, 2.0, 3.0]),
        fov=math.radians(60.0),
    )

    state = _camera_state_from_viser(
        camera,
        width=800,
        height=600,
        camera_convention="opencv",
    )

    assert isinstance(state, CameraState)
    assert int(state.width[0].item()) == 800
    assert int(state.height[0].item()) == 600
    assert state.camera_convention == "opencv"
    assert torch.allclose(
        state.cam_to_world[0, :3, 3],
        torch.tensor([1.0, 2.0, 3.0]),
    )
    focal = 600.0 / (2.0 * math.tan(math.radians(60.0) / 2.0))
    assert state.intrinsics is not None
    assert float(state.intrinsics[0, 1, 1].item()) == pytest.approx(focal)


def test_core_camera_to_viser_pose_uses_ember_intrinsics() -> None:
    camera = CameraState(
        width=torch.tensor([800], dtype=torch.int64),
        height=torch.tensor([600], dtype=torch.int64),
        fov_degrees=torch.tensor([70.0], dtype=torch.float32),
        cam_to_world=torch.eye(4, dtype=torch.float32)[None],
    )

    wxyz, position, vertical_fov = _core_camera_to_viser_pose(camera)

    assert np.allclose(wxyz, np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(position, np.zeros(3))
    assert vertical_fov > 0.0


def test_connection_info_generates_stable_urls_and_ssh_command() -> None:
    config = ViserServerConfig(
        host="0.0.0.0",
        port=8080,
        ssh_host="gpu-box",
        ssh_user="paul",
        local_forward_port=18080,
    )

    info = connection_info(config, port=8080)

    assert info.url == "http://127.0.0.1:8080"
    assert info.iframe_url == "http://127.0.0.1:8080"
    assert (
        info.ssh_forward_command
        == "ssh -N -L 127.0.0.1:18080:127.0.0.1:8080 paul@gpu-box"
    )


def test_apply_viser_config_updates_render_state() -> None:
    render_state = ViserRenderState()
    config = ViserRenderConfig(
        viewer_res=640,
        render_width=1280,
        render_height=720,
        move_jpeg_quality=35,
        static_jpeg_quality=85,
    )

    updated = apply_viser_config(render_state, config)

    assert updated is render_state
    assert render_state.viewer_res == 640
    assert render_state.render_width == 1280
    assert render_state.static_jpeg_quality == 85


def test_noop_viewer_keeps_ember_camera_state() -> None:
    camera = CameraState(
        width=torch.tensor([32], dtype=torch.int64),
        height=torch.tensor([24], dtype=torch.int64),
        fov_degrees=torch.tensor([60.0], dtype=torch.float32),
        cam_to_world=torch.eye(4, dtype=torch.float32)[None],
    )
    viewer = NoopViserViewer(state=ViserViewerState(camera=camera))

    viewer.update(3)
    viewer.rerender(wait=True)

    assert viewer.state.camera is camera
    assert viewer.state.step == 3


def test_numpy_image_accepts_torch_chw_float() -> None:
    image = torch.ones((3, 2, 4), dtype=torch.float32) * 0.5

    converted = _numpy_image(image)

    assert converted.shape == (2, 4, 3)
    assert converted.dtype == np.uint8
    assert int(converted[0, 0, 0]) == 127
