from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation
from splatkit.core import BACKEND_REGISTRY, RenderOptions, render
from splatkit_backends.fastgs import (
    FastGSRenderOptions,
    FastGSRenderOutput,
    register,
    render_fastgs,
)

register()


@dataclass(frozen=True)
class _FakeSettings:
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    mult: float
    prefiltered: bool
    debug: bool
    get_flag: bool
    metric_map: torch.Tensor


class _FakeRasterizer:
    def __init__(self, raster_settings: _FakeSettings) -> None:
        self.raster_settings = raster_settings

    def __call__(
        self,
        *,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        dc: torch.Tensor,
        shs: torch.Tensor,
        colors_precomp: torch.Tensor | None,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov3D_precomp: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del colors_precomp, cov3D_precomp, opacities, scales, rotations
        assert dc.shape == (means3D.shape[0], 1, 3)
        assert shs.shape[0] == means3D.shape[0]
        assert means2D.requires_grad is True
        height = self.raster_settings.image_height
        width = self.raster_settings.image_width
        image = torch.full(
            (3, height, width),
            0.25,
            device=means3D.device,
            dtype=means3D.dtype,
        )
        radii = torch.tensor([3, 0, 2], device=means3D.device, dtype=torch.int32)
        counts = torch.arange(
            height * width,
            device=means3D.device,
            dtype=torch.int32,
        )
        return image, radii, counts


class _EmptyMetricMapRasterizer(_FakeRasterizer):
    def __call__(
        self,
        *,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        dc: torch.Tensor,
        shs: torch.Tensor,
        colors_precomp: torch.Tensor | None,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov3D_precomp: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, radii, _counts = super().__call__(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        empty_counts = torch.empty(
            0,
            device=means3D.device,
            dtype=torch.int32,
        )
        return image, radii, empty_counts


class _QuaternionCheckingRasterizer(_FakeRasterizer):
    def __call__(
        self,
        *,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        dc: torch.Tensor,
        shs: torch.Tensor,
        colors_precomp: torch.Tensor | None,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov3D_precomp: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rotation_norms = torch.linalg.vector_norm(rotations, dim=-1)
        assert torch.allclose(rotation_norms, torch.ones_like(rotation_norms))
        return super().__call__(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )


@pytest.mark.backend
@pytest.mark.cuda
def test_render_fastgs_returns_expected_shapes(
    cuda_scene, cuda_camera, monkeypatch
) -> None:
    monkeypatch.setattr(
        "splatkit_backends.fastgs.renderer._import_fastgs_runtime",
        lambda: (_FakeSettings, _FakeRasterizer),
    )

    output = cast(FastGSRenderOutput, render_fastgs(cuda_scene, cuda_camera))

    assert output.render.shape == (1, 32, 32, 3)
    assert output.viewspace_points.shape == (1, 3, 4)
    assert output.viewspace_points.requires_grad is True
    assert output.visibility_filter.shape == (1, 3)
    assert output.visibility_filter.dtype == torch.bool
    assert output.radii.shape == (1, 3)
    assert output.radii.dtype == torch.int32
    assert output.accum_metric_counts.shape == (1, 32, 32)
    assert output.accum_metric_counts.dtype == torch.int32
    assert torch.equal(
        output.visibility_filter[0],
        torch.tensor([True, False, True], device=output.render.device),
    )


@pytest.mark.backend
@pytest.mark.cuda
def test_render_fastgs_accepts_empty_metric_map(
    cuda_scene, cuda_camera, monkeypatch
) -> None:
    monkeypatch.setattr(
        "splatkit_backends.fastgs.renderer._import_fastgs_runtime",
        lambda: (_FakeSettings, _EmptyMetricMapRasterizer),
    )

    output = cast(FastGSRenderOutput, render_fastgs(cuda_scene, cuda_camera))

    assert output.accum_metric_counts.shape == (1, 32, 32)
    assert output.accum_metric_counts.dtype == torch.int32
    assert torch.count_nonzero(output.accum_metric_counts).item() == 0


@pytest.mark.backend
@pytest.mark.cuda
def test_render_fastgs_normalizes_quaternions(
    cuda_scene, cuda_camera, monkeypatch
) -> None:
    monkeypatch.setattr(
        "splatkit_backends.fastgs.renderer._import_fastgs_runtime",
        lambda: (_FakeSettings, _QuaternionCheckingRasterizer),
    )
    unnormalized_scene = type(cuda_scene)(
        center_position=cuda_scene.center_position,
        log_scales=cuda_scene.log_scales,
        quaternion_orientation=cuda_scene.quaternion_orientation * 3.0,
        logit_opacity=cuda_scene.logit_opacity,
        feature=cuda_scene.feature,
        sh_degree=cuda_scene.sh_degree,
    )

    output = cast(
        FastGSRenderOutput,
        render_fastgs(unnormalized_scene, cuda_camera),
    )

    assert output.render.shape == (1, 32, 32, 3)


def test_render_fastgs_rejects_cpu_scene(cpu_scene, cpu_camera) -> None:
    with pytest.raises(ValueError, match="scene tensors on CUDA"):
        render_fastgs(cpu_scene, cpu_camera)


def test_render_fastgs_rejects_non_sh_features(cpu_scene, cpu_camera) -> None:
    bad_scene = type(cpu_scene)(
        center_position=cpu_scene.center_position,
        log_scales=cpu_scene.log_scales,
        quaternion_orientation=cpu_scene.quaternion_orientation,
        logit_opacity=cpu_scene.logit_opacity,
        feature=torch.zeros((3, 3), dtype=torch.float32),
        sh_degree=0,
    )
    if torch.cuda.is_available():
        bad_scene = bad_scene.to(torch.device("cuda"))
        cpu_camera = cpu_camera.to(torch.device("cuda"))
    with pytest.raises(ValueError, match="spherical harmonics"):
        render_fastgs(bad_scene, cpu_camera)


def test_render_fastgs_rejects_non_opencv_camera(cpu_scene, cpu_camera) -> None:
    bad_camera = type(cpu_camera)(
        cam_to_world=cpu_camera.cam_to_world,
        fov_degrees=cpu_camera.fov_degrees,
        width=cpu_camera.width,
        height=cpu_camera.height,
        camera_convention="opengl",
    )
    if torch.cuda.is_available():
        cpu_scene = cpu_scene.to(torch.device("cuda"))
        bad_camera = bad_camera.to(torch.device("cuda"))
    with pytest.raises(ValueError, match="opencv convention"):
        render_fastgs(cpu_scene, bad_camera)


def test_render_fastgs_rejects_2d_projections(cpu_scene, cpu_camera) -> None:
    with pytest.raises(ValueError, match="2D Gaussian projections"):
        render_fastgs(
            cpu_scene,
            cpu_camera,
            return_2d_projections=True,
        )


@pytest.mark.backend
@pytest.mark.cuda
def test_render_fastgs_rejects_debug_path(cuda_scene, cuda_camera) -> None:
    with pytest.raises(ValueError, match="debug path is not supported"):
        render_fastgs(
            cuda_scene,
            cuda_camera,
            options=FastGSRenderOptions(debug=True),
        )


def test_render_fastgs_beartype_rejects_wrong_options(
    cpu_scene, cpu_camera
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        render_fastgs(
            cpu_scene,
            cpu_camera,
            options=RenderOptions(),  # type: ignore[arg-type]
        )


def test_registry_contains_fastgs() -> None:
    assert "fastgs" in BACKEND_REGISTRY
    assert isinstance(
        BACKEND_REGISTRY["fastgs"].default_options,
        FastGSRenderOptions,
    )


@pytest.mark.backend
@pytest.mark.cuda
def test_generic_render_fastgs_returns_backend_specific_signals(
    cuda_scene, cuda_camera, monkeypatch
) -> None:
    monkeypatch.setattr(
        "splatkit_backends.fastgs.renderer._import_fastgs_runtime",
        lambda: (_FakeSettings, _FakeRasterizer),
    )

    output = cast(
        FastGSRenderOutput,
        render(
            cuda_scene,
            cuda_camera,
            backend="fastgs",
        ),
    )
    assert output.viewspace_points.shape == (1, 3, 4)
    assert output.radii.shape == (1, 3)
