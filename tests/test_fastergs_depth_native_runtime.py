from __future__ import annotations

import pytest
import torch
from ember_native_faster_gs.faster_gs_depth.runtime import render


def _extract_camera_params(camera_state) -> tuple[int, int, float, float, float, float]:
    intrinsics = camera_state.get_intrinsics()[0]
    return (
        int(camera_state.width[0].item()),
        int(camera_state.height[0].item()),
        float(intrinsics[0, 0].item()),
        float(intrinsics[1, 1].item()),
        float(intrinsics[0, 2].item()),
        float(intrinsics[1, 2].item()),
    )


@pytest.mark.cuda
def test_expected_depth_render_returns_depth_and_gradients(
    cuda_scene, cuda_camera
) -> None:
    width, height, focal_x, focal_y, center_x, center_y = _extract_camera_params(
        cuda_camera
    )
    cam_to_world = cuda_camera.cam_to_world[0]
    center_positions = cuda_scene.center_position.detach().clone().requires_grad_(True)
    log_scales = cuda_scene.log_scales.detach().clone().requires_grad_(True)
    rotations = cuda_scene.quaternion_orientation.detach().clone().requires_grad_(True)
    opacities = cuda_scene.logit_opacity[:, None].detach().clone().requires_grad_(True)
    sh0 = cuda_scene.feature[:, :1, :].detach().clone().requires_grad_(True)
    shrest = cuda_scene.feature[:, 1:, :].detach().clone().requires_grad_(True)

    result = render(
        center_positions,
        log_scales,
        rotations,
        opacities,
        sh0,
        shrest,
        torch.linalg.inv(cam_to_world),
        cam_to_world[:3, 3],
        near_plane=0.01,
        far_plane=1000.0,
        width=width,
        height=height,
        focal_x=focal_x,
        focal_y=focal_y,
        center_x=center_x,
        center_y=center_y,
        bg_color=torch.zeros(3, device=center_positions.device),
        proper_antialiasing=False,
        active_sh_bases=int(cuda_scene.feature.shape[1]),
    )
    result.depth.sum().backward()

    assert result.image.shape == (3, 32, 32)
    assert result.depth.shape == (32, 32)
    assert torch.isfinite(result.image).all()
    assert torch.isfinite(result.depth).all()
    for grad in (
        center_positions.grad,
        log_scales.grad,
        rotations.grad,
        opacities.grad,
        sh0.grad,
        shrest.grad,
    ):
        assert grad is not None
        assert torch.isfinite(grad).all()
