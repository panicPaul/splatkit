from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
import torch
from splatkit_native_faster_gs.faster_gs.runtime.ops.blend import (
    blend_bwd_op as native_blend_bwd_op,
)
from splatkit_native_faster_gs.faster_gs.runtime.ops.blend import (
    blend_fwd_op as native_blend_fwd_op,
)
from splatkit_native_faster_gs_mojo.core.runtime import (
    blend,
    preprocess,
    render,
    sort,
)
from splatkit_native_faster_gs_mojo.core.runtime._mojo import load_custom_op_library
from splatkit_native_faster_gs_mojo.core.runtime.ops import (
    blend_bwd_op,
    blend_fwd_op,
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    repo_root = _repo_root()
    pythonpath_parts = [
        str(repo_root / "packages" / "splatkit-native-faster-gs-mojo" / "src"),
        str(repo_root / "packages" / "splatkit-native-faster-gs" / "src"),
        str(repo_root / "packages" / "splatkit-adapter-backends" / "src"),
        str(repo_root / "packages" / "splatkit" / "src"),
    ]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def _run_subprocess_check(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_repo_root(),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "subprocess check failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _prepare_blend_stage_inputs(
    scene,
    camera,
    *,
    proper_antialiasing: bool,
):
    width, height, focal_x, focal_y, center_x, center_y = _extract_camera_params(camera)
    cam_to_world = camera.cam_to_world[0]
    preprocess_result = preprocess(
        scene.center_position,
        scene.log_scales,
        scene.quaternion_orientation,
        scene.logit_opacity[:, None],
        scene.feature[:, :1, :],
        scene.feature[:, 1:, :],
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
        proper_antialiasing=proper_antialiasing,
        active_sh_bases=int(scene.feature.shape[1]),
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
    return preprocess_result, sort_result, width, height


@pytest.mark.cuda
def test_load_custom_op_library_exposes_blend_ops() -> None:
    backend = load_custom_op_library()

    assert hasattr(backend, "blend_fwd")
    assert hasattr(backend, "blend_bwd")


@pytest.mark.cuda
def test_preprocess_sort_blend_return_expected_shapes(
    cuda_scene,
    cuda_camera,
) -> None:
    preprocess_result, sort_result, width, height = _prepare_blend_stage_inputs(
        cuda_scene,
        cuda_camera,
        proper_antialiasing=False,
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
    assert sort_result.tile_instance_ranges.shape[1] == 2
    assert blend_result.image.shape == (3, 32, 32)
    assert torch.isfinite(blend_result.image).all()


@pytest.mark.cuda
def test_blend_forward_matches_native_stage(cuda_scene, cuda_camera) -> None:
    preprocess_result, sort_result, width, height = _prepare_blend_stage_inputs(
        cuda_scene,
        cuda_camera,
        proper_antialiasing=False,
    )

    mojo_outputs = blend_fwd_op(
        sort_result.instance_primitive_indices,
        sort_result.tile_instance_ranges,
        sort_result.tile_bucket_offsets,
        sort_result.bucket_count,
        preprocess_result.projected_means,
        preprocess_result.conic_opacity,
        preprocess_result.colors_rgb,
        torch.zeros(3, device=preprocess_result.projected_means.device),
        False,
        width,
        height,
    )
    native_outputs = native_blend_fwd_op(
        sort_result.instance_primitive_indices,
        sort_result.tile_instance_ranges,
        sort_result.tile_bucket_offsets,
        sort_result.bucket_count,
        preprocess_result.projected_means,
        preprocess_result.conic_opacity,
        preprocess_result.colors_rgb,
        torch.zeros(3, device=preprocess_result.projected_means.device),
        False,
        width,
        height,
    )

    for mojo_tensor, native_tensor in zip(mojo_outputs[:2], native_outputs[:2], strict=True):
        torch.testing.assert_close(mojo_tensor, native_tensor, rtol=1e-4, atol=2e-4)
    for mojo_tensor, native_tensor in zip(mojo_outputs[2:5], native_outputs[2:5], strict=True):
        torch.testing.assert_close(mojo_tensor, native_tensor)
    torch.testing.assert_close(mojo_outputs[5], native_outputs[5], rtol=1e-4, atol=2e-4)


@pytest.mark.cuda
def test_blend_backward_matches_native_stage(cuda_scene, cuda_camera) -> None:
    preprocess_result, sort_result, width, height = _prepare_blend_stage_inputs(
        cuda_scene,
        cuda_camera,
        proper_antialiasing=True,
    )
    bg_color = torch.zeros(3, device=preprocess_result.projected_means.device)
    forward_outputs = native_blend_fwd_op(
        sort_result.instance_primitive_indices,
        sort_result.tile_instance_ranges,
        sort_result.tile_bucket_offsets,
        sort_result.bucket_count,
        preprocess_result.projected_means,
        preprocess_result.conic_opacity,
        preprocess_result.colors_rgb,
        bg_color,
        True,
        width,
        height,
    )
    grad_image = torch.randn_like(forward_outputs[0])

    mojo_grads = blend_bwd_op(
        grad_image,
        *forward_outputs[:1],
        sort_result.instance_primitive_indices,
        sort_result.tile_instance_ranges,
        sort_result.tile_bucket_offsets,
        preprocess_result.projected_means,
        preprocess_result.conic_opacity,
        preprocess_result.colors_rgb,
        bg_color,
        *forward_outputs[1:],
        True,
        width,
        height,
    )
    native_grads = native_blend_bwd_op(
        grad_image,
        *forward_outputs[:1],
        sort_result.instance_primitive_indices,
        sort_result.tile_instance_ranges,
        sort_result.tile_bucket_offsets,
        preprocess_result.projected_means,
        preprocess_result.conic_opacity,
        preprocess_result.colors_rgb,
        bg_color,
        *forward_outputs[1:],
        True,
        width,
        height,
    )

    for mojo_grad, native_grad in zip(mojo_grads, native_grads, strict=True):
        torch.testing.assert_close(mojo_grad, native_grad, rtol=2e-4, atol=5e-4)


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


@pytest.mark.cuda
def test_blend_backward_large_scene_repeated_call_does_not_crash() -> None:
    script = dedent(
        """
        import torch
        from splatkit.io import load_gaussian_ply
        from splatkit.core import CameraState
        from splatkit_native_faster_gs_mojo.core.runtime import preprocess, sort
        from splatkit_native_faster_gs_mojo.core.runtime.ops import blend_bwd_op, blend_fwd_op

        device = torch.device("cuda")
        scene = load_gaussian_ply("point_cloud.ply").to(device)
        width = 640
        height = 480
        cam_to_world = torch.eye(4, dtype=torch.float32, device=device)[None]
        cam_to_world[:, 2, 3] = 3.0
        camera = CameraState(
            width=torch.tensor([width], dtype=torch.int64, device=device),
            height=torch.tensor([height], dtype=torch.int64, device=device),
            fov_degrees=torch.tensor([60.0], dtype=torch.float32, device=device),
            cam_to_world=cam_to_world,
            camera_convention="opencv",
        )
        intrinsics = camera.get_intrinsics()[0]
        cam_to_world0 = camera.cam_to_world[0]
        preprocess_result = preprocess(
            scene.center_position,
            scene.log_scales,
            scene.quaternion_orientation,
            scene.logit_opacity[:, None],
            scene.feature[:, :1, :],
            scene.feature[:, 1:, :],
            torch.linalg.inv(cam_to_world0),
            cam_to_world0[:3, 3],
            near_plane=0.01,
            far_plane=1000.0,
            width=width,
            height=height,
            focal_x=float(intrinsics[0, 0].item()),
            focal_y=float(intrinsics[1, 1].item()),
            center_x=float(intrinsics[0, 2].item()),
            center_y=float(intrinsics[1, 2].item()),
            proper_antialiasing=False,
            active_sh_bases=int(scene.feature.shape[1]),
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
        forward = blend_fwd_op(
            sort_result.instance_primitive_indices,
            sort_result.tile_instance_ranges,
            sort_result.tile_bucket_offsets,
            sort_result.bucket_count,
            preprocess_result.projected_means,
            preprocess_result.conic_opacity,
            preprocess_result.colors_rgb,
            torch.zeros(3, device=device),
            False,
            width,
            height,
        )
        grad_image = torch.randn_like(forward[0])
        for _ in range(3):
            grads = blend_bwd_op(
                grad_image,
                forward[0],
                sort_result.instance_primitive_indices,
                sort_result.tile_instance_ranges,
                sort_result.tile_bucket_offsets,
                preprocess_result.projected_means,
                preprocess_result.conic_opacity,
                preprocess_result.colors_rgb,
                torch.zeros(3, device=device),
                forward[1],
                forward[2],
                forward[3],
                forward[4],
                forward[5],
                False,
                width,
                height,
            )
            torch.cuda.synchronize()
            assert all(torch.isfinite(grad).all().item() for grad in grads)
        """
    )
    _run_subprocess_check(script)


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

    assert len(preprocess_result) == 10
    assert len(render_result) == 20
