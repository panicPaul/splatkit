from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
import torch
from ember_native_faster_gs.faster_gs_depth.runtime import (
    render as render_depth,
)
from ember_native_faster_gs.faster_gs.runtime import render


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
        str(repo_root / "packages" / "ember-native-faster-gs" / "src"),
        str(repo_root / "packages" / "ember-adapter-backends" / "src"),
        str(repo_root / "packages" / "ember-core" / "src"),
    ]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def _run_reference_check(script: str) -> None:
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
            "reference check failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _run_torch_backward(device: torch.device) -> None:
    x = torch.randn((32, 32), device=device, requires_grad=True)
    y = x.square().sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def _run_native_backward(cuda_scene, cuda_camera) -> None:
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


def _run_depth_backward(cuda_scene, cuda_camera) -> None:
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
    result = render_depth(
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
    (result.image.sum() + result.depth.sum()).backward()
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
def test_render_matches_reference_cuda_backend(cuda_scene, cuda_camera) -> None:
    del cuda_scene, cuda_camera
    _run_reference_check(
        dedent(
            """
            import torch
            from FasterGSCudaBackend import RasterizerSettings, diff_rasterize
            from ember_core.core import CameraState, GaussianScene3D
            from ember_native_faster_gs.faster_gs.runtime import render

            scene = GaussianScene3D(
                center_position=torch.tensor(
                    [[0.0, 0.0, 0.0], [0.2, 0.0, 0.1], [-0.2, 0.1, 0.2]],
                    dtype=torch.float32,
                    device="cuda",
                ),
                log_scales=torch.full((3, 3), -1.0, dtype=torch.float32, device="cuda"),
                quaternion_orientation=torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]] * 3,
                    dtype=torch.float32,
                    device="cuda",
                ),
                logit_opacity=torch.tensor([2.0, 1.5, 1.0], dtype=torch.float32, device="cuda"),
                feature=torch.zeros((3, 16, 3), dtype=torch.float32, device="cuda"),
                sh_degree=0,
            )
            scene.feature[:, 0, :] = torch.tensor(
                [[0.9, 0.2, 0.2], [0.2, 0.9, 0.2], [0.2, 0.2, 0.9]],
                dtype=torch.float32,
                device="cuda",
            )
            cam_to_world = torch.eye(4, dtype=torch.float32, device="cuda")[None]
            cam_to_world[:, 2, 3] = 3.0
            camera = CameraState(
                width=torch.tensor([32], dtype=torch.int64, device="cuda"),
                height=torch.tensor([32], dtype=torch.int64, device="cuda"),
                fov_degrees=torch.tensor([60.0], dtype=torch.float32, device="cuda"),
                cam_to_world=cam_to_world,
                camera_convention="opencv",
            )
            cam_to_world = camera.cam_to_world[0]
            intrinsics = camera.get_intrinsics()[0]
            width = int(camera.width[0].item())
            height = int(camera.height[0].item())
            focal_x = float(intrinsics[0, 0].item())
            focal_y = float(intrinsics[1, 1].item())
            center_x = float(intrinsics[0, 2].item())
            center_y = float(intrinsics[1, 2].item())

            native = render(
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
                bg_color=torch.zeros(3, device="cuda"),
                proper_antialiasing=False,
                active_sh_bases=int(scene.feature.shape[1]),
            ).image
            reference = diff_rasterize(
                means=scene.center_position,
                scales=scene.log_scales,
                rotations=scene.quaternion_orientation,
                opacities=scene.logit_opacity[:, None],
                sh_coefficients_0=scene.feature[:, :1, :].contiguous(),
                sh_coefficients_rest=scene.feature[:, 1:, :].contiguous(),
                densification_info=torch.empty(0, device="cuda"),
                rasterizer_settings=RasterizerSettings(
                    w2c=torch.linalg.inv(cam_to_world),
                    cam_position=cam_to_world[:3, 3].contiguous(),
                    bg_color=torch.zeros(3, device="cuda"),
                    active_sh_bases=int(scene.feature.shape[1]),
                    width=width,
                    height=height,
                    focal_x=focal_x,
                    focal_y=focal_y,
                    center_x=center_x,
                    center_y=center_y,
                    near_plane=0.01,
                    far_plane=1000.0,
                    proper_antialiasing=False,
                ),
            )
            torch.testing.assert_close(native, reference, rtol=1e-4, atol=2e-4)
            """
        )
    )


@pytest.mark.cuda
def test_render_gradients_match_reference_cuda_backend(
    cuda_scene,
    cuda_camera,
) -> None:
    del cuda_scene, cuda_camera
    _run_reference_check(
        dedent(
            """
            import torch
            from FasterGSCudaBackend import RasterizerSettings, diff_rasterize
            from ember_core.core import CameraState, GaussianScene3D
            from ember_native_faster_gs.faster_gs.runtime import render

            scene = GaussianScene3D(
                center_position=torch.tensor(
                    [[0.0, 0.0, 0.0], [0.2, 0.0, 0.1], [-0.2, 0.1, 0.2]],
                    dtype=torch.float32,
                    device="cuda",
                ),
                log_scales=torch.full((3, 3), -1.0, dtype=torch.float32, device="cuda"),
                quaternion_orientation=torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]] * 3,
                    dtype=torch.float32,
                    device="cuda",
                ),
                logit_opacity=torch.tensor([2.0, 1.5, 1.0], dtype=torch.float32, device="cuda"),
                feature=torch.zeros((3, 16, 3), dtype=torch.float32, device="cuda"),
                sh_degree=0,
            )
            scene.feature[:, 0, :] = torch.tensor(
                [[0.9, 0.2, 0.2], [0.2, 0.9, 0.2], [0.2, 0.2, 0.9]],
                dtype=torch.float32,
                device="cuda",
            )
            cam_to_world = torch.eye(4, dtype=torch.float32, device="cuda")[None]
            cam_to_world[:, 2, 3] = 3.0
            camera = CameraState(
                width=torch.tensor([32], dtype=torch.int64, device="cuda"),
                height=torch.tensor([32], dtype=torch.int64, device="cuda"),
                fov_degrees=torch.tensor([60.0], dtype=torch.float32, device="cuda"),
                cam_to_world=cam_to_world,
                camera_convention="opencv",
            )
            cam_to_world = camera.cam_to_world[0]
            intrinsics = camera.get_intrinsics()[0]
            width = int(camera.width[0].item())
            height = int(camera.height[0].item())
            focal_x = float(intrinsics[0, 0].item())
            focal_y = float(intrinsics[1, 1].item())
            center_x = float(intrinsics[0, 2].item())
            center_y = float(intrinsics[1, 2].item())
            weights = torch.linspace(
                0.5,
                1.5,
                steps=3 * height * width,
                device="cuda",
                dtype=torch.float32,
            ).view(3, height, width)

            native_inputs = [
                scene.center_position.detach().clone().requires_grad_(True),
                scene.log_scales.detach().clone().requires_grad_(True),
                scene.quaternion_orientation.detach().clone().requires_grad_(True),
                scene.logit_opacity[:, None].detach().clone().requires_grad_(True),
                scene.feature[:, :1, :].detach().clone().requires_grad_(True),
                scene.feature[:, 1:, :].detach().clone().requires_grad_(True),
            ]
            reference_inputs = [
                tensor.detach().clone().requires_grad_(True) for tensor in native_inputs
            ]

            native_image = render(
                native_inputs[0],
                native_inputs[1],
                native_inputs[2],
                native_inputs[3],
                native_inputs[4],
                native_inputs[5],
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
                bg_color=torch.zeros(3, device="cuda"),
                proper_antialiasing=False,
                active_sh_bases=int(scene.feature.shape[1]),
            ).image
            (native_image * weights).sum().backward()

            reference_image = diff_rasterize(
                means=reference_inputs[0],
                scales=reference_inputs[1],
                rotations=reference_inputs[2],
                opacities=reference_inputs[3],
                sh_coefficients_0=reference_inputs[4].contiguous(),
                sh_coefficients_rest=reference_inputs[5].contiguous(),
                densification_info=torch.empty(0, device="cuda"),
                rasterizer_settings=RasterizerSettings(
                    w2c=torch.linalg.inv(cam_to_world),
                    cam_position=cam_to_world[:3, 3].contiguous(),
                    bg_color=torch.zeros(3, device="cuda"),
                    active_sh_bases=int(scene.feature.shape[1]),
                    width=width,
                    height=height,
                    focal_x=focal_x,
                    focal_y=focal_y,
                    center_x=center_x,
                    center_y=center_y,
                    near_plane=0.01,
                    far_plane=1000.0,
                    proper_antialiasing=False,
                ),
            )
            (reference_image * weights).sum().backward()

            for native_grad, reference_grad in zip(
                [tensor.grad for tensor in native_inputs],
                [tensor.grad for tensor in reference_inputs],
                strict=True,
            ):
                assert native_grad is not None
                assert reference_grad is not None
                torch.testing.assert_close(
                    native_grad,
                    reference_grad,
                    rtol=1e-4,
                    atol=3e-4,
                )
            """
        )
    )


@pytest.mark.cuda
def test_native_backends_leave_cuda_context_healthy(
    cuda_scene,
    cuda_camera,
) -> None:
    _run_native_backward(cuda_scene, cuda_camera)
    _run_torch_backward(cuda_scene.center_position.device)
    _run_depth_backward(cuda_scene, cuda_camera)
    _run_torch_backward(cuda_scene.center_position.device)
