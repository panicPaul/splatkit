from __future__ import annotations

from pathlib import Path

import pytest
import torch
from splatkit_native_3dgrt.core.runtime import (
    TraceStateConfig,
    acquire_state_token,
    build_acc,
    destroy_acc,
    pack_particle_density,
    render,
    trace,
    update_acc,
)
from torch._subclasses.fake_tensor import FakeTensorMode

_REPO_ROOT = Path(__file__).resolve().parent.parent


class _FakeOptixTracer:
    def __init__(self) -> None:
        self.build_calls: list[tuple[bool, bool]] = []

    def build_bvh(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        densities: torch.Tensor,
        rebuild: bool,
        allow_update: bool,
    ) -> None:
        del positions, rotations, scales, densities
        self.build_calls.append((bool(rebuild), bool(allow_update)))

    def trace(
        self,
        frame_number: int,
        ray_to_world: torch.Tensor,
        ray_ori: torch.Tensor,
        ray_dir: torch.Tensor,
        particle_density: torch.Tensor,
        particle_radiance: torch.Tensor,
        render_opts: int,
        sph_degree: int,
        min_transmittance: float,
    ) -> tuple[torch.Tensor, ...]:
        del (
            frame_number,
            ray_to_world,
            render_opts,
            sph_degree,
            min_transmittance,
        )
        batch_size, height, width, _ = ray_ori.shape
        device = ray_ori.device
        dtype = ray_ori.dtype
        num_particles = int(particle_density.shape[0])
        radiance = torch.full((batch_size, height, width, 3), 0.25, device=device, dtype=dtype)
        density = torch.full((batch_size, height, width, 1), 0.5, device=device, dtype=dtype)
        hit = torch.zeros((batch_size, height, width, 2), device=device, dtype=dtype)
        hit[..., 0] = 3.0
        normals = ray_dir.clone()
        hitcounts = torch.full((batch_size, height, width, 1), 2.0, device=device, dtype=dtype)
        visibility = torch.ones((num_particles, 1), device=device, dtype=dtype)
        weights = torch.full((num_particles, 1), 0.75, device=device, dtype=dtype)
        sample_cache = torch.zeros((batch_size, height, width, 16), device=device, dtype=dtype)
        return (
            radiance,
            density,
            hit,
            normals,
            hitcounts,
            visibility,
            weights,
            sample_cache,
        )

    def trace_bwd(
        self,
        frame_number: int,
        ray_to_world: torch.Tensor,
        ray_ori: torch.Tensor,
        ray_dir: torch.Tensor,
        ray_radiance: torch.Tensor,
        ray_density: torch.Tensor,
        ray_hit: torch.Tensor,
        ray_sample_cache: torch.Tensor,
        ray_normals: torch.Tensor,
        particle_density: torch.Tensor,
        particle_radiance: torch.Tensor,
        ray_radiance_grad: torch.Tensor,
        ray_density_grad: torch.Tensor,
        ray_hit_grad: torch.Tensor,
        ray_normals_grad: torch.Tensor,
        render_opts: int,
        sph_degree: int,
        min_transmittance: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del (
            frame_number,
            ray_to_world,
            ray_ori,
            ray_dir,
            ray_radiance,
            ray_density,
            ray_hit,
            ray_sample_cache,
            ray_normals,
            ray_radiance_grad,
            ray_density_grad,
            ray_hit_grad,
            ray_normals_grad,
            render_opts,
            sph_degree,
            min_transmittance,
        )
        return (
            torch.full_like(particle_density, 0.1),
            torch.full_like(particle_radiance, 0.2),
            torch.zeros((particle_density.shape[0], 1), device=particle_density.device, dtype=particle_density.dtype),
        )


def _state_config() -> TraceStateConfig:
    return TraceStateConfig(
        particle_kernel_density_clamping=False,
        enable_normals=True,
        enable_hitcounts=True,
        max_consecutive_bvh_update=2,
    )


def _build_rays(camera_state) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cam_to_world = camera_state.cam_to_world
    intrinsics = camera_state.get_intrinsics()
    num_cams = int(cam_to_world.shape[0])
    height = int(camera_state.height[0].item())
    width = int(camera_state.width[0].item())
    x = torch.arange(width, device=cam_to_world.device, dtype=cam_to_world.dtype)
    y = torch.arange(height, device=cam_to_world.device, dtype=cam_to_world.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.view(1, height, width).expand(num_cams, -1, -1)
    yy = yy.view(1, height, width).expand(num_cams, -1, -1)
    fx = intrinsics[:, 0, 0].view(num_cams, 1, 1)
    fy = intrinsics[:, 1, 1].view(num_cams, 1, 1)
    cx = intrinsics[:, 0, 2].view(num_cams, 1, 1)
    cy = intrinsics[:, 1, 2].view(num_cams, 1, 1)
    rays_dir = torch.stack(
        (
            ((xx + 0.5) - cx) / fx,
            ((yy + 0.5) - cy) / fy,
            torch.ones((num_cams, height, width), device=cam_to_world.device, dtype=cam_to_world.dtype),
        ),
        dim=-1,
    )
    return (
        cam_to_world.contiguous(),
        torch.zeros_like(rays_dir),
        torch.nn.functional.normalize(rays_dir, dim=-1),
    )


@pytest.mark.cuda
def test_traced_runtime_build_update_render_and_backward(
    cuda_scene,
    cuda_camera,
    monkeypatch,
) -> None:
    fake_tracer = _FakeOptixTracer()
    monkeypatch.setattr(
        "splatkit_native_3dgrt.core.runtime.state._make_tracer_wrapper",
        lambda config: fake_tracer,
    )
    state_token = acquire_state_token(_state_config(), cuda_scene.center_position.device)
    particle_density = pack_particle_density(
        cuda_scene.center_position.detach().clone().requires_grad_(True),
        torch.sigmoid(cuda_scene.logit_opacity[:, None].detach().clone()).requires_grad_(True),
        torch.nn.functional.normalize(
            cuda_scene.quaternion_orientation.detach().clone(), dim=1
        ).requires_grad_(True),
        torch.exp(cuda_scene.log_scales.detach().clone()).requires_grad_(True),
    )
    particle_density.retain_grad()
    particle_radiance = cuda_scene.feature[:, :1, :].reshape(cuda_scene.feature.shape[0], -1)
    particle_radiance = particle_radiance.detach().clone().requires_grad_(True)
    ray_to_world, ray_ori, ray_dir = _build_rays(cuda_camera)

    build_acc(state_token, particle_density)
    update_acc(state_token, particle_density)

    trace_result = trace(
        state_token,
        ray_to_world,
        ray_ori,
        ray_dir,
        particle_density,
        particle_radiance,
        sph_degree=cuda_scene.sh_degree,
        min_transmittance=0.001,
    )
    render_result = render(
        state_token,
        ray_to_world,
        ray_ori,
        ray_dir,
        particle_density,
        particle_radiance,
        torch.ones(3, device=cuda_scene.center_position.device),
        sph_degree=cuda_scene.sh_degree,
        min_transmittance=0.001,
    )
    loss = (
        trace_result.radiance.sum()
        + render_result.render.sum()
        + render_result.depth.sum()
        + render_result.normals.sum()
    )
    loss.backward()
    destroy_acc(state_token)

    assert fake_tracer.build_calls == [(True, True), (False, True)]
    assert render_result.render.shape == (1, 32, 32, 3)
    assert render_result.alphas.shape == (1, 32, 32)
    assert render_result.depth.shape == (1, 32, 32)
    assert render_result.normals.shape == (1, 32, 32, 3)
    assert render_result.hitcounts.shape == (1, 32, 32)
    assert torch.isfinite(render_result.render).all()
    assert particle_density.grad is not None
    assert particle_radiance.grad is not None


def test_traced_render_supports_fake_tensor_mode(cpu_scene, cpu_camera) -> None:
    ray_to_world, ray_ori, ray_dir = _build_rays(cpu_camera)
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        particle_density = mode.from_tensor(
            pack_particle_density(
                cpu_scene.center_position,
                torch.sigmoid(cpu_scene.logit_opacity[:, None]),
                torch.nn.functional.normalize(cpu_scene.quaternion_orientation, dim=1),
                torch.exp(cpu_scene.log_scales),
            )
        )
        particle_radiance = mode.from_tensor(
            cpu_scene.feature[:, :1, :].reshape(cpu_scene.feature.shape[0], -1)
        )
        fake_ray_to_world = mode.from_tensor(ray_to_world)
        fake_ray_ori = mode.from_tensor(ray_ori)
        fake_ray_dir = mode.from_tensor(ray_dir)
        bg_color = mode.from_tensor(torch.zeros(3, dtype=cpu_scene.center_position.dtype))
        token = mode.from_tensor(torch.zeros((), dtype=torch.int64))
        result = render(
            token,
            fake_ray_to_world,
            fake_ray_ori,
            fake_ray_dir,
            particle_density,
            particle_radiance,
            bg_color,
            sph_degree=cpu_scene.sh_degree,
            min_transmittance=0.001,
        )
    assert result.render.shape == (1, 32, 32, 3)


def test_traced_native_code_no_longer_imports_upstream_runtime() -> None:
    traced_sources = [
        _REPO_ROOT
        / "packages"
        / "splatkit-native-3dgrt"
        / "src"
        / "splatkit_native_3dgrt"
        / "native_build"
        / "stoch3dgs.py",
        _REPO_ROOT
        / "packages"
        / "splatkit-native-3dgrt"
        / "src"
        / "splatkit_native_3dgrt"
        / "core"
        / "runtime"
        / "state.py",
        _REPO_ROOT
        / "packages"
        / "splatkit-native-3dgrt"
        / "src"
        / "splatkit_native_3dgrt"
        / "stoch3dgs"
        / "renderer.py",
    ]
    forbidden_terms = ("import threedgrut", "from threedgrut", "import threedgrt_tracer", "from threedgrt_tracer")
    for path in traced_sources:
        source = path.read_text()
        for term in forbidden_terms:
            assert term not in source, f"{path} still references upstream runtime import {term!r}."


def test_native_backends_pyproject_no_longer_declares_threedgrut() -> None:
    pyproject = (
        _REPO_ROOT
        / "packages"
        / "splatkit-native-3dgrt"
        / "pyproject.toml"
    ).read_text()
    assert "threedgrut" not in pyproject
