from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch
from ember_adapter_backends.fastergs import (
    FasterGSDensificationRenderOutput,
    FasterGSRenderOptions,
    FasterGSRenderOutput,
    render_fastergs,
)
from ember_adapter_backends.fastergs import (
    register as register_fastergs,
)
from ember_core.core import BACKEND_REGISTRY, render
from ember_native_faster_gs.faster_gs import (
    FasterGSNativeDensificationRenderOutput,
    FasterGSNativeRenderOptions,
    FasterGSNativeRenderOutput,
    register,
    render_faster_gs_native,
)

register()
register_fastergs()


def _clone_scene_with_grad(scene):
    return replace(
        scene,
        center_position=scene.center_position.detach()
        .clone()
        .requires_grad_(True),
        log_scales=scene.log_scales.detach().clone().requires_grad_(True),
        quaternion_orientation=scene.quaternion_orientation.detach()
        .clone()
        .requires_grad_(True),
        logit_opacity=scene.logit_opacity.detach().clone().requires_grad_(True),
        feature=scene.feature.detach().clone().requires_grad_(True),
    )


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
        render(cuda_scene, cuda_camera, backend="faster_gs.core"),
    )

    assert BACKEND_REGISTRY["faster_gs.core"].name == "faster_gs.core"
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


@pytest.mark.backend
@pytest.mark.cuda
def test_native_densification_info_matches_fastergs_backend(
    cuda_visible_scene,
    cuda_camera,
) -> None:
    native_scene = _clone_scene_with_grad(cuda_visible_scene)
    reference_scene = _clone_scene_with_grad(cuda_visible_scene)

    native_output = cast(
        FasterGSNativeDensificationRenderOutput,
        render_faster_gs_native(
            native_scene,
            cuda_camera,
            options=FasterGSNativeRenderOptions(
                collect_densification_info=True
            ),
        ),
    )
    reference_output = cast(
        FasterGSDensificationRenderOutput,
        render_fastergs(
            reference_scene,
            cuda_camera,
            options=FasterGSRenderOptions(collect_densification_info=True),
        ),
    )

    native_output.render.sum().backward()
    reference_output.render.sum().backward()

    assert native_output.densification_info.shape == (2, 3)
    assert reference_output.densification_info.shape == (2, 3)
    assert torch.isfinite(native_output.densification_info).all()
    assert torch.isfinite(reference_output.densification_info).all()
    torch.testing.assert_close(
        native_output.densification_info[0],
        reference_output.densification_info[0],
        rtol=1e-4,
        atol=2e-4,
    )
    torch.testing.assert_close(
        native_output.densification_info[1],
        reference_output.densification_info[1],
        rtol=1e-2,
        atol=1e-2,
    )


def test_render_faster_gs_native_rejects_cpu_scene(
    cpu_scene, cpu_camera
) -> None:
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
