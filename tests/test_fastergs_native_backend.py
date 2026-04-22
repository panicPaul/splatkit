from __future__ import annotations

from typing import cast

import pytest
import torch
from splatkit.core import BACKEND_REGISTRY, render
from splatkit_adapter_backends.fastergs import (
    FasterGSRenderOutput,
    render_fastergs,
)
from splatkit_adapter_backends.fastergs import (
    register as register_fastergs,
)
from splatkit_native_backends.faster_gs import (
    FasterGSNativeRenderOutput,
    register,
    render_faster_gs_native,
)

register()
register_fastergs()


@pytest.mark.backend
@pytest.mark.cuda
def test_render_faster_gs_native_returns_expected_shapes(
    cuda_scene,
    cuda_camera,
) -> None:
    output = cast(
        FasterGSNativeRenderOutput,
        render_faster_gs_native(cuda_scene, cuda_camera),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.render.dtype == cuda_scene.center_position.dtype
    assert torch.isfinite(output.render).all()


@pytest.mark.backend
@pytest.mark.cuda
def test_generic_render_dispatches_to_faster_gs_native(
    cuda_scene,
    cuda_camera,
) -> None:
    output = cast(
        FasterGSNativeRenderOutput,
        render(cuda_scene, cuda_camera, backend="faster_gs"),
    )

    assert BACKEND_REGISTRY["faster_gs"].name == "faster_gs"
    assert output.render.shape == (1, 32, 32, 3)


@pytest.mark.backend
@pytest.mark.cuda
def test_native_backend_matches_fastergs_backend(
    cuda_scene,
    cuda_camera,
) -> None:
    native_output = cast(
        FasterGSNativeRenderOutput,
        render_faster_gs_native(cuda_scene, cuda_camera),
    )
    reference_output = cast(
        FasterGSRenderOutput,
        render_fastergs(cuda_scene, cuda_camera),
    )

    torch.testing.assert_close(
        native_output.render,
        reference_output.render,
        rtol=1e-4,
        atol=2e-4,
    )


def test_render_faster_gs_native_rejects_cpu_scene(cpu_scene, cpu_camera) -> None:
    with pytest.raises(ValueError, match="scene tensors on CUDA"):
        render_faster_gs_native(cpu_scene, cpu_camera)


def test_render_faster_gs_native_rejects_non_opencv_camera(
    cpu_scene,
    cpu_camera,
) -> None:
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
        render_faster_gs_native(cpu_scene, bad_camera)


def test_render_faster_gs_native_rejects_2d_projections(
    cpu_scene,
    cpu_camera,
) -> None:
    with pytest.raises(ValueError, match="2D Gaussian projections"):
        render_faster_gs_native(
            cpu_scene,
            cpu_camera,
            return_2d_projections=True,
        )
