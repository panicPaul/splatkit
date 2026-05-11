from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch
from ember_core.core import BACKEND_REGISTRY, render
from ember_native_3dgrt.stoch3dgs import (
    Stoch3DGSNativeRenderOptions,
    Stoch3DGSNativeRenderOutput,
    register,
    render_stoch3dgs_native,
)
from ember_native_3dgrt.stoch3dgs import renderer as stoch3dgs_native_renderer

register()


def test_stoch3dgs_native_options_match_paper_defaults() -> None:
    options = Stoch3DGSNativeRenderOptions()

    assert options.max_consecutive_bvh_update == 15
    assert options.ray_principal_point_mode == "image_center"


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
        radiance = torch.full(
            (batch_size, height, width, 3), 0.25, device=device, dtype=dtype
        )
        density = torch.full(
            (batch_size, height, width, 1), 0.5, device=device, dtype=dtype
        )
        hit = torch.zeros(
            (batch_size, height, width, 2), device=device, dtype=dtype
        )
        hit[..., 0] = 3.0
        normals = torch.nn.functional.normalize(_args[3], dim=3)
        hitcounts = torch.full(
            (batch_size, height, width, 1), 2.0, device=device, dtype=dtype
        )
        visibility = torch.ones((num_particles, 1), device=device, dtype=dtype)
        weights = torch.full(
            (num_particles, 1), 0.75, device=device, dtype=dtype
        )
        sample_cache = torch.zeros(
            (batch_size, height, width, 16), device=device, dtype=dtype
        )
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
        self, *_args, **_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        particle_density = _args[9]
        particle_radiance = _args[10]
        return (
            torch.full_like(particle_density, 0.1),
            torch.full_like(particle_radiance, 0.2),
            torch.zeros(
                (particle_density.shape[0], 1),
                device=particle_density.device,
                dtype=particle_density.dtype,
            ),
        )


class _ContiguityCheckingOptixTracer(_FakeOptixTracer):
    def __init__(self) -> None:
        super().__init__()
        self.last_build_inputs_contiguous: (
            tuple[bool, bool, bool, bool] | None
        ) = None

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
        "ember_native_3dgrt.core.runtime.state._make_tracer_wrapper",
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
        "ember_native_3dgrt.core.runtime.state._make_tracer_wrapper",
        lambda config: _FakeOptixTracer(),
    )

    output = cast(
        Stoch3DGSNativeRenderOutput,
        render(cuda_scene, cuda_camera, backend="3dgrt.stoch3dgs"),
    )

    assert BACKEND_REGISTRY["3dgrt.stoch3dgs"].name == "3dgrt.stoch3dgs"
    assert output.depth.shape == (1, 32, 32)


def test_render_stoch3dgs_native_rejects_cpu_scene(
    cpu_scene, cpu_camera
) -> None:
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


def test_stoch3dgs_native_rays_default_to_upstream_image_center(
    cpu_camera,
) -> None:
    intrinsics = torch.tensor(
        [[[2.0, 0.0, 1.0], [0.0, 4.0, 0.25], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    camera = replace(
        cpu_camera,
        width=torch.tensor([4], dtype=torch.int64),
        height=torch.tensor([2], dtype=torch.int64),
        intrinsics=intrinsics,
    )

    _origins, image_center_dirs = stoch3dgs_native_renderer._build_batch(
        camera,
        principal_point_mode="image_center",
    )
    _origins, intrinsics_dirs = stoch3dgs_native_renderer._build_batch(
        camera,
        principal_point_mode="intrinsics",
    )

    image_center_ratio = (
        image_center_dirs[0, 0, 0, :2] / image_center_dirs[0, 0, 0, 2]
    )
    intrinsics_ratio = (
        intrinsics_dirs[0, 0, 0, :2] / intrinsics_dirs[0, 0, 0, 2]
    )
    torch.testing.assert_close(
        image_center_ratio,
        torch.tensor([-0.75, -0.125]),
    )
    torch.testing.assert_close(
        intrinsics_ratio,
        torch.tensor([-0.25, 0.0625]),
    )


def test_stoch3dgs_native_flattens_sh_like_upstream_features(cpu_scene) -> None:
    feature = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    scene = cpu_scene.with_fields(feature=feature, sh_degree=1)

    flattened = stoch3dgs_native_renderer._flatten_sh_features(scene)

    expected = torch.cat((feature[:, 0, :], feature[:, 1:, :].reshape(2, -1)), dim=1)
    torch.testing.assert_close(flattened, expected)
    assert flattened.is_contiguous()


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
        "ember_native_3dgrt.core.runtime.state._make_tracer_wrapper",
        lambda config: tracer,
    )

    render_stoch3dgs_native(cuda_scene, cuda_camera)

    assert tracer.last_build_inputs_contiguous == (True, True, True, True)
