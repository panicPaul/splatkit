from __future__ import annotations

from typing import Literal

import pytest
import torch
from ember_core.core import (
    BACKEND_REGISTRY,
    CameraState,
    GaussianScene3D,
    render,
)
from ember_core.meshification import MESHIFIER_REGISTRY, meshify
from ember_native_faster_gs import register as register_faster_gs_backends
from ember_native_faster_gs.gaussian_wrapping import GaussianWrappingScene
from ember_native_faster_gs.gaussian_wrapping.renderer import (
    GaussianWrappingNativeRenderOptions,
    GaussianWrappingNativeRenderOutput,
)


def test_gaussian_wrapping_backend_registers() -> None:
    register_faster_gs_backends()

    assert "faster_gs.gaussian_wrapping" in BACKEND_REGISTRY
    assert "wrapping" in MESHIFIER_REGISTRY


@pytest.mark.parametrize("rasterizer_mode", ["ours", "radegs"])
def test_gaussian_wrapping_render_outputs(
    cpu_visible_scene: GaussianScene3D,
    cpu_camera: CameraState,
    rasterizer_mode: Literal["ours", "radegs"],
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Gaussian Wrapping native tests.")
    register_faster_gs_backends()
    cuda_scene = cpu_visible_scene.detached_copy(torch.device("cuda"))
    cuda_camera = cpu_camera.to(torch.device("cuda"))

    output = render(
        cuda_scene,
        cuda_camera,
        backend="faster_gs.gaussian_wrapping",
        return_alpha=True,
        return_depth=True,
        return_normals=True,
        options=GaussianWrappingNativeRenderOptions(
            rasterizer_mode=rasterizer_mode,
        ),
    )

    assert isinstance(output, GaussianWrappingNativeRenderOutput)
    assert output.render.shape == (1, 32, 32, 3)
    assert output.alphas.shape == (1, 32, 32)
    assert output.depth.shape == (1, 32, 32)
    assert output.normals.shape == (1, 32, 32, 3)
    assert output.median_depth.shape == output.depth.shape
    assert output.expected_depth.shape == output.depth.shape
    assert output.radii.shape == (1, 3)


@pytest.mark.parametrize("rasterizer_mode", ["ours", "radegs"])
def test_gaussian_wrapping_render_backward(
    cpu_visible_scene: GaussianScene3D,
    cpu_camera: CameraState,
    rasterizer_mode: Literal["ours", "radegs"],
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Gaussian Wrapping native tests.")
    register_faster_gs_backends()
    cuda_scene = cpu_visible_scene.detached_copy(torch.device("cuda"))
    cuda_camera = cpu_camera.to(torch.device("cuda"))
    for parameter in cuda_scene.parameters():
        parameter.requires_grad_(True)

    output = render(
        cuda_scene,
        cuda_camera,
        backend="faster_gs.gaussian_wrapping",
        return_alpha=True,
        return_depth=True,
        return_normals=True,
        options=GaussianWrappingNativeRenderOptions(
            rasterizer_mode=rasterizer_mode,
        ),
    )
    loss = (
        output.render.sum()
        + output.alphas.sum()
        + output.depth.sum()
        + output.normals.sum()
    )

    loss.backward()

    assert cuda_scene.center_position.grad is not None
    assert torch.isfinite(cuda_scene.center_position.grad).all()


@pytest.mark.parametrize("rasterizer_mode", ["ours", "radegs"])
def test_gaussian_wrapping_meshify(
    cpu_visible_scene: GaussianScene3D,
    cpu_camera: CameraState,
    rasterizer_mode: Literal["ours", "radegs"],
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Gaussian Wrapping native tests.")
    register_faster_gs_backends()
    cuda_scene = cpu_visible_scene.to(torch.device("cuda"))
    cuda_camera = cpu_camera.to(torch.device("cuda"))

    result = meshify(
        cuda_scene,
        cuda_camera,
        backend="faster_gs.gaussian_wrapping",
        meshifier="wrapping",
        backend_options=GaussianWrappingNativeRenderOptions(
            rasterizer_mode=rasterizer_mode,
        ),
    )

    assert result.mesh.vertices.shape == (8, 3)
    assert result.mesh.faces.shape == (12, 3)
    assert result.diagnostics["meshifier"] == "wrapping"


def test_gaussian_wrapping_rejects_cpu_scene(
    cpu_visible_scene: GaussianScene3D,
    cpu_camera: CameraState,
) -> None:
    register_faster_gs_backends()

    with pytest.raises(ValueError, match="requires scene tensors on CUDA"):
        render(
            cpu_visible_scene,
            cpu_camera,
            backend="faster_gs.gaussian_wrapping",
            return_alpha=True,
            options=GaussianWrappingNativeRenderOptions(),
        )


def test_gaussian_wrapping_scene_adds_wrapping_fields(
    cpu_scene: GaussianScene3D,
) -> None:
    scene = GaussianWrappingScene(
        center_position=cpu_scene.center_position.detach(),
        log_scales=cpu_scene.log_scales.detach(),
        quaternion_orientation=cpu_scene.quaternion_orientation.detach(),
        logit_opacity=cpu_scene.logit_opacity.detach(),
        feature=cpu_scene.feature.detach(),
        sh_degree=cpu_scene.sh_degree,
    )

    assert scene.normal_features.shape == (3, 3)
    assert scene.base_occupancy.shape == (3, 9)
    assert scene.occupancy_shift.shape == (3, 9)
    assert torch.allclose(scene.occupancy, torch.full((3, 9), 0.5))
