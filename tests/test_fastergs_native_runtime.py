from __future__ import annotations

import pytest
import torch
from splatkit_native_backends.faster_gs_native.runtime import (
    blend,
    preprocess,
    render,
    sort,
)
from splatkit_native_backends.faster_gs_native.runtime.ops import (
    _RENDER_CONTEXTS,
    _RENDER_OUTPUT_TO_CONTEXT,
    preprocess_fwd_op,
    render_fwd_op,
)
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.mark.cuda
def test_preprocess_sort_blend_return_expected_shapes(
    cuda_scene,
    cuda_camera,
) -> None:
    intrinsics = cuda_camera.get_intrinsics()[0]
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
        width=int(cuda_camera.width[0].item()),
        height=int(cuda_camera.height[0].item()),
        focal_x=float(intrinsics[0, 0].item()),
        focal_y=float(intrinsics[1, 1].item()),
        center_x=float(intrinsics[0, 2].item()),
        center_y=float(intrinsics[1, 2].item()),
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
        width=int(cuda_camera.width[0].item()),
        height=int(cuda_camera.height[0].item()),
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
        width=int(cuda_camera.width[0].item()),
        height=int(cuda_camera.height[0].item()),
    )

    assert preprocess_result.projected_means.shape == (3, 2)
    assert preprocess_result.screen_bounds.shape == (3, 4)
    assert sort_result.tile_instance_ranges.shape[1] == 2
    assert blend_result.image.shape == (3, 32, 32)
    assert torch.isfinite(blend_result.image).all()


@pytest.mark.cuda
def test_render_backward_produces_finite_gradients(cuda_scene, cuda_camera) -> None:
    intrinsics = cuda_camera.get_intrinsics()[0]
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
        width=int(cuda_camera.width[0].item()),
        height=int(cuda_camera.height[0].item()),
        focal_x=float(intrinsics[0, 0].item()),
        focal_y=float(intrinsics[1, 1].item()),
        center_x=float(intrinsics[0, 2].item()),
        center_y=float(intrinsics[1, 2].item()),
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
    intrinsics = cpu_camera.get_intrinsics()[0]
    width = int(cpu_camera.width[0].item())
    height = int(cpu_camera.height[0].item())
    focal_x = float(intrinsics[0, 0].item())
    focal_y = float(intrinsics[1, 1].item())
    center_x = float(intrinsics[0, 2].item())
    center_y = float(intrinsics[1, 2].item())
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


@pytest.mark.cuda
def test_render_op_supports_torch_compile(cuda_scene, cuda_camera) -> None:
    intrinsics = cuda_camera.get_intrinsics()[0]
    width = int(cuda_camera.width[0].item())
    height = int(cuda_camera.height[0].item())
    focal_x = float(intrinsics[0, 0].item())
    focal_y = float(intrinsics[1, 1].item())
    center_x = float(intrinsics[0, 2].item())
    center_y = float(intrinsics[1, 2].item())
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


@pytest.mark.cuda
def test_render_no_grad_does_not_accumulate_contexts(
    cuda_scene,
    cuda_camera,
) -> None:
    _RENDER_CONTEXTS.clear()
    _RENDER_OUTPUT_TO_CONTEXT.clear()
    for _ in range(8):
        with torch.no_grad():
            result = render(
                cuda_scene.center_position,
                cuda_scene.log_scales,
                cuda_scene.quaternion_orientation,
                cuda_scene.logit_opacity[:, None],
                cuda_scene.feature[:, :1, :],
                cuda_scene.feature[:, 1:, :],
                torch.linalg.inv(cuda_camera.cam_to_world[0]),
                cuda_camera.cam_to_world[0, :3, 3],
                near_plane=0.01,
                far_plane=1000.0,
                width=int(cuda_camera.width[0].item()),
                height=int(cuda_camera.height[0].item()),
                focal_x=float(cuda_camera.get_intrinsics()[0, 0, 0].item()),
                focal_y=float(cuda_camera.get_intrinsics()[0, 1, 1].item()),
                center_x=float(cuda_camera.get_intrinsics()[0, 0, 2].item()),
                center_y=float(cuda_camera.get_intrinsics()[0, 1, 2].item()),
                bg_color=torch.zeros(3, device=cuda_scene.center_position.device),
                proper_antialiasing=False,
                active_sh_bases=int(cuda_scene.feature.shape[1]),
            )
        del result
        torch.cuda.synchronize()
    assert _RENDER_CONTEXTS == {}
    assert _RENDER_OUTPUT_TO_CONTEXT == {}
