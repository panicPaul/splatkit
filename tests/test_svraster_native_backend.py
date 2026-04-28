from __future__ import annotations

from typing import cast

import pytest
import torch
from ember_core.core import BACKEND_REGISTRY, render
from ember_native_svraster.svraster import (
    SVRasterDepthRenderOutput,
    SVRasterRenderOutput,
    register,
    render_svraster,
)

register()


@pytest.mark.backend
@pytest.mark.cuda
def test_render_svraster_native_returns_expected_shapes(
    cuda_sparse_voxel_scene,
    cuda_camera,
) -> None:
    output = cast(
        SVRasterRenderOutput,
        render_svraster(cuda_sparse_voxel_scene, cuda_camera),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.render.dtype == cuda_sparse_voxel_scene.sh0.dtype
    assert torch.isfinite(output.render).all()


@pytest.mark.backend
@pytest.mark.cuda
def test_render_svraster_native_returns_depth_surface(
    cuda_sparse_voxel_scene,
    cuda_camera,
) -> None:
    output = cast(
        SVRasterDepthRenderOutput,
        render_svraster(
            cuda_sparse_voxel_scene,
            cuda_camera,
            return_depth=True,
        ),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.depth.shape == (1, 3, 32, 32)
    assert torch.isfinite(output.depth).all()


@pytest.mark.backend
@pytest.mark.cuda
def test_generic_render_dispatches_to_svraster_native(
    cuda_sparse_voxel_scene,
    cuda_camera,
) -> None:
    output = cast(
        SVRasterRenderOutput,
        render(cuda_sparse_voxel_scene, cuda_camera, backend="svraster.core"),
    )

    assert BACKEND_REGISTRY["svraster.core"].name == "svraster.core"
    assert output.render.shape == (1, 32, 32, 3)


def test_render_svraster_native_rejects_cpu_scene(
    cpu_sparse_voxel_scene,
    cpu_camera,
) -> None:
    with pytest.raises(ValueError, match="scene tensors on CUDA"):
        render_svraster(cpu_sparse_voxel_scene, cpu_camera)


def test_render_svraster_native_rejects_non_opencv_camera(
    cpu_sparse_voxel_scene,
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
        cpu_sparse_voxel_scene = cpu_sparse_voxel_scene.to(torch.device("cuda"))
        bad_camera = bad_camera.to(torch.device("cuda"))
    with pytest.raises(ValueError, match="opencv convention"):
        render_svraster(cpu_sparse_voxel_scene, bad_camera)
