from __future__ import annotations

from typing import cast

import pytest
import torch
from ember_core.core import BACKEND_REGISTRY, render
from ember_native_faster_gs.faster_gs import (
    FasterGSNativeRenderOutput,
    render_faster_gs_native,
)
from ember_native_faster_gs.faster_gs import (
    register as register_faster_gs,
)
from ember_native_faster_gs_mojo.core import (
    FasterGSMojoRenderOutput,
    register,
    render_faster_gs_mojo,
)

register()
register_faster_gs()


@pytest.mark.backend
@pytest.mark.cuda
def test_render_faster_gs_mojo_returns_expected_shapes(
    cuda_scene,
    cuda_camera,
) -> None:
    output = cast(
        FasterGSMojoRenderOutput,
        render_faster_gs_mojo(cuda_scene, cuda_camera),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.render.dtype == cuda_scene.center_position.dtype
    assert torch.isfinite(output.render).all()


@pytest.mark.backend
@pytest.mark.cuda
def test_generic_render_dispatches_to_faster_gs_mojo(
    cuda_scene,
    cuda_camera,
) -> None:
    output = cast(
        FasterGSMojoRenderOutput,
        render(cuda_scene, cuda_camera, backend="faster_gs_mojo.core"),
    )

    assert BACKEND_REGISTRY["faster_gs_mojo.core"].name == "faster_gs_mojo.core"
    assert output.render.shape == (1, 32, 32, 3)


@pytest.mark.backend
@pytest.mark.cuda
def test_faster_gs_mojo_matches_current_faster_gs_backend(
    cuda_scene,
    cuda_camera,
) -> None:
    mojo_output = cast(
        FasterGSMojoRenderOutput,
        render_faster_gs_mojo(cuda_scene, cuda_camera),
    )
    reference_output = cast(
        FasterGSNativeRenderOutput,
        render_faster_gs_native(cuda_scene, cuda_camera),
    )

    torch.testing.assert_close(
        mojo_output.render,
        reference_output.render,
        rtol=1e-4,
        atol=2e-4,
    )
