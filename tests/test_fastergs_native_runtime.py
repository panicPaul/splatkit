from __future__ import annotations

import pytest
import torch
from splatkit_native_backends.faster_gs.runtime import (
    blend,
    preprocess,
    render,
    sort,
)
from splatkit_native_backends.faster_gs.runtime.ops import (
    preprocess_fwd_op,
    render_fwd_op,
)
from torch._subclasses.fake_tensor import FakeTensorMode


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
def test_preprocess_sort_blend_return_expected_shapes(
    cuda_scene,
    cuda_camera,
) -> None:
    width, height, focal_x, focal_y, center_x, center_y = _extract_camera_params(
        cuda_camera
    )
    cam_to_world = cuda_camera.cam_to_world[0]
    preprocess_result = preprocess(
        cuda_scene.center_position,
        cuda_scene.log_scales,
        cuda_scene.quaternion_orientation,
        cuda_scene.logit_opacity[:, None],
        cuda_scene.feature[:, :1, :],
        cuda_scene.feature[:, 1:, :],
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
        proper_antialiasing=False,
        active_sh_bases=int(cuda_scene.feature.shape[1]),
    )
    sort_result = sort(
        preprocess_result.depth_keys,
        preprocess_result.primitive_indices,
        preprocess_result.num_touched_tiles,
        preprocess_result.screen_bounds,
        preprocess_result.projected_means,
        preprocess_result.conic_opacity,
        preprocess_result.visible_count,
        preprocess_result.instance_count,
        width=width,
        height=height,
    )
    blend_result = blend(
        sort_result.instance_primitive_indices,
        sort_result.tile_instance_ranges,
        sort_result.tile_bucket_offsets,
        sort_result.bucket_count,
        preprocess_result.projected_means,
        preprocess_result.conic_opacity,
        preprocess_result.colors_rgb,
        torch.zeros(3, device=cuda_scene.center_position.device),
        False,
        width=width,
        height=height,
    )

    assert preprocess_result.projected_means.shape == (3, 2)
    assert preprocess_result.screen_bounds.shape == (3, 4)
    assert sort_result.tile_instance_ranges.shape[1] == 2
    assert blend_result.image.shape == (3, 32, 32)
    assert torch.isfinite(blend_result.image).all()


@pytest.mark.cuda
def test_render_fwd_matches_explicit_stage_composition(
    cuda_scene,
    cuda_camera,
) -> None:
    width, height, focal_x, focal_y, center_x, center_y = _extract_camera_params(
        cuda_camera
    )
    cam_to_world = cuda_camera.cam_to_world[0]
    render_outputs = render_fwd_op(
        cuda_scene.center_position,
        cuda_scene.log_scales,
        cuda_scene.quaternion_orientation,
        cuda_scene.logit_opacity[:, None],
        cuda_scene.feature[:, :1, :],
        cuda_scene.feature[:, 1:, :],
        torch.linalg.inv(cam_to_world),
        cam_to_world[:3, 3],
        0.01,
        1000.0,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        torch.zeros(3, device=cuda_scene.center_position.device),
        False,
        int(cuda_scene.feature.shape[1]),
    )
    preprocess_result = preprocess(
        cuda_scene.center_position,
        cuda_scene.log_scales,
        cuda_scene.quaternion_orientation,
        cuda_scene.logit_opacity[:, None],
        cuda_scene.feature[:, :1, :],
        cuda_scene.feature[:, 1:, :],
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
        proper_antialiasing=False,
        active_sh_bases=int(cuda_scene.feature.shape[1]),
    )
    sort_result = sort(
        preprocess_result.depth_keys,
        preprocess_result.primitive_indices,
        preprocess_result.num_touched_tiles,
        preprocess_result.screen_bounds,
        preprocess_result.projected_means,
        preprocess_result.conic_opacity,
        preprocess_result.visible_count,
        preprocess_result.instance_count,
        width=width,
        height=height,
    )
    blend_result = blend(
        sort_result.instance_primitive_indices,
        sort_result.tile_instance_ranges,
        sort_result.tile_bucket_offsets,
        sort_result.bucket_count,
        preprocess_result.projected_means,
        preprocess_result.conic_opacity,
        preprocess_result.colors_rgb,
        torch.zeros(3, device=cuda_scene.center_position.device),
        False,
        width=width,
        height=height,
    )

    assert len(render_outputs) == 20
    torch.testing.assert_close(render_outputs[0], blend_result.image)


@pytest.mark.cuda
def test_render_backward_produces_finite_gradients(cuda_scene, cuda_camera) -> None:
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
    result.image.sum().backward()

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


def test_raw_ops_support_fake_tensor_mode(cpu_scene, cpu_camera) -> None:
    width, height, focal_x, focal_y, center_x, center_y = _extract_camera_params(
        cpu_camera
    )
    cam_to_world = cpu_camera.cam_to_world[0]

    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        center_positions = mode.from_tensor(cpu_scene.center_position)
        log_scales = mode.from_tensor(cpu_scene.log_scales)
        rotations = mode.from_tensor(cpu_scene.quaternion_orientation)
        opacities = mode.from_tensor(cpu_scene.logit_opacity[:, None])
        sh0 = mode.from_tensor(cpu_scene.feature[:, :1, :])
        shrest = mode.from_tensor(cpu_scene.feature[:, 1:, :])
        world_2_camera = mode.from_tensor(torch.linalg.inv(cam_to_world))
        camera_position = mode.from_tensor(cam_to_world[:3, 3])
        bg_color = mode.from_tensor(torch.zeros(3, dtype=center_positions.dtype))
        preprocess_result = preprocess_fwd_op(
            center_positions,
            log_scales,
            rotations,
            opacities,
            sh0,
            shrest,
            world_2_camera,
            camera_position,
            0.01,
            1000.0,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            False,
            int(cpu_scene.feature.shape[1]),
        )
        render_result = render_fwd_op(
            center_positions,
            log_scales,
            rotations,
            opacities,
            sh0,
            shrest,
            world_2_camera,
            camera_position,
            0.01,
            1000.0,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            bg_color,
            False,
            int(cpu_scene.feature.shape[1]),
        )

    assert preprocess_result[0].shape == (3, 2)
    assert render_result[0].shape == (3, 32, 32)
    assert render_result[1].shape == (3, 2)


@pytest.mark.cuda
def test_render_op_supports_torch_compile(cuda_scene, cuda_camera) -> None:
    width, height, focal_x, focal_y, center_x, center_y = _extract_camera_params(
        cuda_camera
    )
    cam_to_world = cuda_camera.cam_to_world[0]
    world_2_camera = torch.linalg.inv(cam_to_world)
    camera_position = cam_to_world[:3, 3]
    bg_color = torch.zeros(3, device=cuda_scene.center_position.device)
    active_sh_bases = int(cuda_scene.feature.shape[1])

    def compiled_render(center_positions: torch.Tensor) -> torch.Tensor:
        return render(
            center_positions,
            cuda_scene.log_scales,
            cuda_scene.quaternion_orientation,
            cuda_scene.logit_opacity[:, None],
            cuda_scene.feature[:, :1, :],
            cuda_scene.feature[:, 1:, :],
            world_2_camera,
            camera_position,
            near_plane=0.01,
            far_plane=1000.0,
            width=width,
            height=height,
            focal_x=focal_x,
            focal_y=focal_y,
            center_x=center_x,
            center_y=center_y,
            bg_color=bg_color,
            proper_antialiasing=False,
            active_sh_bases=active_sh_bases,
        ).image.sum()

    compiled_fn = torch.compile(compiled_render, backend="eager", fullgraph=True)
    result = compiled_fn(cuda_scene.center_position)

    assert result.ndim == 0
    assert torch.isfinite(result)
