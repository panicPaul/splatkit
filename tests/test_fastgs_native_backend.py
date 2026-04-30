from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch
from ember_adapter_backends.fastgs import (
    FastGSRenderOptions,
    FastGSRenderOutput,
    render_fastgs,
)
from ember_adapter_backends.fastgs import (
    register as register_adapter_fastgs,
)
from ember_core.core import BACKEND_REGISTRY, render, resolve_backend_trait
from ember_core.densification import GaussianMetricAttribution
from ember_native_faster_gs.faster_gs import (
    register as register_faster_gs,
)
from ember_native_faster_gs.fastgs import (
    FastGSNativeRenderOptions,
    FastGSNativeRenderOutput,
    register,
    render_fastgs_native,
)

register()
register_faster_gs()
register_adapter_fastgs()


@pytest.mark.backend
@pytest.mark.cuda
def test_render_fastgs_native_returns_expected_shapes(
    cuda_scene,
    cuda_camera,
) -> None:
    output = cast(
        FastGSNativeRenderOutput,
        render_fastgs_native(cuda_scene, cuda_camera),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.render.dtype == cuda_scene.center_position.dtype
    assert torch.isfinite(output.render).all()


@pytest.mark.backend
@pytest.mark.cuda
def test_generic_render_dispatches_to_fastgs_native(
    cuda_scene,
    cuda_camera,
) -> None:
    output = cast(
        FastGSNativeRenderOutput,
        render(cuda_scene, cuda_camera, backend="faster_gs.fastgs"),
    )

    assert BACKEND_REGISTRY["faster_gs.fastgs"].name == "faster_gs.fastgs"
    assert BACKEND_REGISTRY["faster_gs.fastgs"].supported_outputs == frozenset()
    assert output.render.shape == (1, 32, 32, 3)


@pytest.mark.backend
@pytest.mark.cuda
def test_fastgs_native_backend_accepts_compact_box_scale(
    cuda_scene,
    cuda_camera,
) -> None:
    output = cast(
        FastGSNativeRenderOutput,
        render_fastgs_native(
            cuda_scene,
            cuda_camera,
            options=FastGSNativeRenderOptions(compact_box_scale=0.7),
        ),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert torch.isfinite(output.render).all()


@pytest.mark.backend
@pytest.mark.cuda
def test_fastgs_native_backend_matches_existing_fastgs_adapter(
    cuda_scene,
    cuda_camera,
) -> None:
    pytest.importorskip(
        "diff_gaussian_rasterization_fastgs",
        exc_type=ImportError,
    )
    fastgs_output = cast(
        FastGSNativeRenderOutput,
        render_fastgs_native(
            cuda_scene,
            cuda_camera,
            options=FastGSNativeRenderOptions(proper_antialiasing=False),
        ),
    )
    adapter_output = cast(
        FastGSRenderOutput,
        render_fastgs(
            cuda_scene,
            cuda_camera,
            options=FastGSRenderOptions(mult=0.5),
        ),
    )

    torch.testing.assert_close(
        fastgs_output.render,
        adapter_output.render,
        rtol=1e-4,
        atol=2e-4,
    )


@pytest.mark.backend
@pytest.mark.cuda
def test_fastgs_native_metric_attribution_matches_existing_fastgs_adapter(
    cuda_scene,
    cuda_camera,
) -> None:
    pytest.importorskip(
        "diff_gaussian_rasterization_fastgs",
        exc_type=ImportError,
    )
    fastgs_provider = resolve_backend_trait(
        "faster_gs.fastgs",
        GaussianMetricAttribution,
    )
    adapter_provider = resolve_backend_trait(
        "adapter.fastgs",
        GaussianMetricAttribution,
    )
    metric_map = torch.ones(
        (32, 32),
        device=cuda_scene.center_position.device,
        dtype=torch.int32,
    )

    fastgs_counts = fastgs_provider.attribute_metric_map(
        cuda_scene,
        cuda_camera,
        metric_map,
        options=FastGSNativeRenderOptions(proper_antialiasing=False),
    )
    adapter_counts = adapter_provider.attribute_metric_map(
        cuda_scene,
        cuda_camera,
        metric_map,
        options=FastGSRenderOptions(mult=0.5),
    )

    torch.testing.assert_close(fastgs_counts, adapter_counts)


@pytest.mark.backend
@pytest.mark.cuda
def test_fastgs_native_metric_attribution_rejects_invalid_inputs(
    cuda_scene,
    cuda_camera,
) -> None:
    provider = resolve_backend_trait(
        "faster_gs.fastgs",
        GaussianMetricAttribution,
    )
    batched_camera = replace(
        cuda_camera,
        width=cuda_camera.width.repeat(2),
        height=cuda_camera.height.repeat(2),
        fov_degrees=cuda_camera.fov_degrees.repeat(2),
        cam_to_world=cuda_camera.cam_to_world.repeat(2, 1, 1),
        intrinsics=(
            None
            if cuda_camera.intrinsics is None
            else cuda_camera.intrinsics.repeat(2, 1, 1)
        ),
    )

    with pytest.raises(ValueError, match="single probe camera"):
        provider.attribute_metric_map(
            cuda_scene,
            batched_camera,
            torch.ones(
                (32, 32),
                device=cuda_scene.center_position.device,
                dtype=torch.int32,
            ),
            options=FastGSNativeRenderOptions(),
        )

    with pytest.raises(ValueError, match="2D metric map"):
        provider.attribute_metric_map(
            cuda_scene,
            cuda_camera,
            torch.ones(
                (1, 32, 32),
                device=cuda_scene.center_position.device,
                dtype=torch.int32,
            ),
            options=FastGSNativeRenderOptions(),
        )


def test_faster_gs_core_does_not_expose_fastgs_metric_attribution() -> None:
    with pytest.raises(ValueError, match="does not provide trait"):
        resolve_backend_trait("faster_gs.core", GaussianMetricAttribution)


def test_render_fastgs_native_rejects_gaussian_impact_score(
    cpu_scene,
    cpu_camera,
) -> None:
    with pytest.raises(ValueError, match="Gaussian impact scores"):
        render_fastgs_native(
            cpu_scene,
            cpu_camera,
            return_gaussian_impact_score=True,
        )


def test_render_fastgs_native_rejects_2d_projections(
    cpu_scene,
    cpu_camera,
) -> None:
    with pytest.raises(ValueError, match="2D Gaussian projections"):
        render_fastgs_native(
            cpu_scene,
            cpu_camera,
            return_2d_projections=True,
        )
