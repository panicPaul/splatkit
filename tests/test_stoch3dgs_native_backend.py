from __future__ import annotations

from typing import cast

import pytest
import torch
from splatkit.core import BACKEND_REGISTRY, render
from splatkit_native_3dgrt.stoch3dgs import renderer as stoch3dgs_native_renderer
from splatkit_native_3dgrt.stoch3dgs import (
    Stoch3DGSNativeRenderOptions,
    Stoch3DGSNativeRenderOutput,
    register,
    render_stoch3dgs_native,
)

register()


class _FakeOptixTracer:
    def __init__(self) -> None:
        self.build_calls = 0

    def build_bvh(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        densities: torch.Tensor,
        rebuild: bool,
        allow_update: bool,
    ) -> None:
        del positions, rotations, scales, densities, rebuild, allow_update
        self.build_calls += 1

    def trace(self, *_args, **_kwargs) -> tuple[torch.Tensor, ...]:
        ray_ori = _args[2]
        particle_density = _args[4]
        device = ray_ori.device
        dtype = ray_ori.dtype
        batch_size, height, width, _ = ray_ori.shape
        num_particles = int(particle_density.shape[0])
        radiance = torch.full((batch_size, height, width, 3), 0.25, device=device, dtype=dtype)
        density = torch.full((batch_size, height, width, 1), 0.5, device=device, dtype=dtype)
        hit = torch.zeros((batch_size, height, width, 2), device=device, dtype=dtype)
        hit[..., 0] = 3.0
        normals = torch.nn.functional.normalize(_args[3], dim=3)
        hitcounts = torch.full((batch_size, height, width, 1), 2.0, device=device, dtype=dtype)
        visibility = torch.ones((num_particles, 1), device=device, dtype=dtype)
        weights = torch.full((num_particles, 1), 0.75, device=device, dtype=dtype)
        sample_cache = torch.zeros((batch_size, height, width, 16), device=device, dtype=dtype)
        return (radiance, density, hit, normals, hitcounts, visibility, weights, sample_cache)

    def trace_bwd(self, *_args, **_kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        particle_density = _args[9]
        particle_radiance = _args[10]
        return (
            torch.full_like(particle_density, 0.1),
            torch.full_like(particle_radiance, 0.2),
            torch.zeros((particle_density.shape[0], 1), device=particle_density.device, dtype=particle_density.dtype),
        )


class _ContiguityCheckingOptixTracer(_FakeOptixTracer):
    def __init__(self) -> None:
        super().__init__()
        self.last_build_inputs_contiguous: tuple[bool, bool, bool, bool] | None = None

    def build_bvh(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        densities: torch.Tensor,
        rebuild: bool,
        allow_update: bool,
    ) -> None:
        self.last_build_inputs_contiguous = (
            positions.is_contiguous(),
            rotations.is_contiguous(),
            scales.is_contiguous(),
            densities.is_contiguous(),
        )
        super().build_bvh(
            positions,
            rotations,
            scales,
            densities,
            rebuild,
            allow_update,
        )


@pytest.mark.backend
@pytest.mark.cuda
def test_render_stoch3dgs_native_returns_full_surface(
    cuda_scene,
    cuda_camera,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "splatkit_native_3dgrt.core.runtime.state._make_tracer_wrapper",
        lambda config: _FakeOptixTracer(),
    )

    output = cast(
        Stoch3DGSNativeRenderOutput,
        render_stoch3dgs_native(
            cuda_scene,
            cuda_camera,
            options=Stoch3DGSNativeRenderOptions(
                background_color=torch.ones(3, device="cuda")
            ),
        ),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.alphas.shape == (1, 32, 32)
    assert output.depth.shape == (1, 32, 32)
    assert output.normals.shape == (1, 32, 32, 3)
    assert output.hitcounts.shape == (1, 32, 32)
    assert output.visibility.shape == (3, 1)
    assert output.weights.shape == (3, 1)
    assert torch.allclose(
        output.render[0, 0, 0],
        torch.tensor([0.75, 0.75, 0.75], device=output.render.device),
    )


@pytest.mark.backend
@pytest.mark.cuda
def test_generic_render_dispatches_to_stoch3dgs_native(
    cuda_scene,
    cuda_camera,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "splatkit_native_3dgrt.core.runtime.state._make_tracer_wrapper",
        lambda config: _FakeOptixTracer(),
    )

    output = cast(
        Stoch3DGSNativeRenderOutput,
        render(cuda_scene, cuda_camera, backend="3dgrt.stoch3dgs"),
    )

    assert BACKEND_REGISTRY["3dgrt.stoch3dgs"].name == "3dgrt.stoch3dgs"
    assert output.depth.shape == (1, 32, 32)


def test_render_stoch3dgs_native_rejects_cpu_scene(cpu_scene, cpu_camera) -> None:
    with pytest.raises(ValueError, match="scene tensors on CUDA"):
        render_stoch3dgs_native(cpu_scene, cpu_camera)


def test_render_stoch3dgs_native_rejects_2d_projections(
    cpu_scene,
    cpu_camera,
) -> None:
    with pytest.raises(ValueError, match="2D Gaussian projections"):
        render_stoch3dgs_native(
            cpu_scene,
            cpu_camera,
            return_2d_projections=True,
        )


@pytest.mark.backend
@pytest.mark.cuda
def test_render_stoch3dgs_native_builds_bvh_with_contiguous_fields(
    cuda_scene,
    cuda_camera,
    monkeypatch,
) -> None:
    stoch3dgs_native_renderer._STATE_TOKEN_CACHE.clear()
    tracer = _ContiguityCheckingOptixTracer()
    monkeypatch.setattr(
        "splatkit_native_3dgrt.core.runtime.state._make_tracer_wrapper",
        lambda config: tracer,
    )

    render_stoch3dgs_native(cuda_scene, cuda_camera)

    assert tracer.last_build_inputs_contiguous == (True, True, True, True)
