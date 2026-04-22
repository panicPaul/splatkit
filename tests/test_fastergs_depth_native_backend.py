from __future__ import annotations

from typing import cast

import pytest
import torch
from splatkit.core import render
from splatkit_adapter_backends.gsplat import GsplatRenderOutput, render_gsplat
from splatkit_native_backends.faster_gs_depth import (
    FasterGSDepthNativeDepthRenderOutput,
    FasterGSDepthNativeRenderOutput,
    register,
    render_faster_gs_depth,
)
from splatkit_native_backends.faster_gs import (
    FasterGSNativeRenderOutput,
    render_faster_gs_native,
)
from splatkit_native_backends.faster_gs import (
    register as register_root,
)

register()
register_root()


@pytest.mark.backend
@pytest.mark.cuda
def test_depth_backend_returns_depth(cuda_scene, cuda_camera) -> None:
    output = cast(
        FasterGSDepthNativeDepthRenderOutput,
        render_faster_gs_depth(
            cuda_scene,
            cuda_camera,
            return_depth=True,
        ),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.depth.shape == (1, 32, 32)
    assert torch.isfinite(output.render).all()
    assert torch.isfinite(output.depth).all()


@pytest.mark.backend
@pytest.mark.cuda
def test_depth_backend_rgb_path_matches_root_backend(
    cuda_scene,
    cuda_camera,
) -> None:
    depth_backend_output = cast(
        FasterGSDepthNativeRenderOutput,
        render_faster_gs_depth(cuda_scene, cuda_camera),
    )
    root_output = cast(
        FasterGSNativeRenderOutput,
        render_faster_gs_native(cuda_scene, cuda_camera),
    )

    torch.testing.assert_close(
        depth_backend_output.render,
        root_output.render,
        rtol=1e-4,
        atol=2e-4,
    )


@pytest.mark.backend
@pytest.mark.cuda
def test_depth_backend_matches_gsplat_expected_depth(
    cuda_scene,
    cuda_camera,
) -> None:
    depth_backend_output = cast(
        FasterGSDepthNativeDepthRenderOutput,
        render_faster_gs_depth(
            cuda_scene,
            cuda_camera,
            return_depth=True,
        ),
    )
    gsplat_output = cast(
        GsplatRenderOutput,
        render_gsplat(
            cuda_scene,
            cuda_camera,
            return_depth=True,
        ),
    )

    torch.testing.assert_close(
        depth_backend_output.depth,
        gsplat_output.depth,
        rtol=1e-4,
        atol=2e-4,
    )


@pytest.mark.backend
@pytest.mark.cuda
def test_generic_render_dispatches_to_depth_backend(
    cuda_scene,
    cuda_camera,
) -> None:
    output = cast(
        FasterGSDepthNativeDepthRenderOutput,
        render(
            cuda_scene,
            cuda_camera,
            backend="faster_gs_depth",
            return_depth=True,
        ),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.depth.shape == (1, 32, 32)
