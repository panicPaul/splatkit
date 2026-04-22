from typing import cast

import pytest
import torch
from splatkit.core import (
    RenderWith2DProjections,
    RenderWithDepth2DProjections,
    RenderWithDepthProjectiveIntersectionTransforms,
    render,
)
from splatkit_adapter_backends.gsplat import (
    GsplatAlphaRenderOutput,
    GsplatRenderOptions,
    GsplatRenderOutput,
    render_gsplat,
    render_gsplat_2dgs,
)

pytestmark = [pytest.mark.backend, pytest.mark.cuda, pytest.mark.integration]


def assert_finite_tensor(tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), "tensor contains non-finite values"


def test_render_gsplat_rgb_returns_expected_shapes(
    cuda_scene, cuda_camera
) -> None:
    output = cast(
        GsplatAlphaRenderOutput, render_gsplat(cuda_scene, cuda_camera)
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.alphas.shape == (1, 32, 32)
    assert_finite_tensor(output.render)
    assert_finite_tensor(output.alphas)
    assert torch.all(output.alphas >= 0.0)
    assert torch.all(output.alphas <= 1.0)


def test_render_gsplat_depth_returns_expected_shapes(
    cuda_scene, cuda_camera
) -> None:
    output = cast(
        GsplatRenderOutput,
        render_gsplat(cuda_scene, cuda_camera, return_depth=True),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.alphas.shape == (1, 32, 32)
    assert output.depth.shape == (1, 32, 32)
    assert_finite_tensor(output.depth)


def test_generic_render_returns_2d_projections(cuda_scene, cuda_camera) -> None:
    output: RenderWith2DProjections = render(
        cuda_scene,
        cuda_camera,
        backend="adapter.gsplat",
        return_2d_projections=True,
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.projected_means.shape == (1, 3, 2)
    assert output.projected_conics.shape == (1, 3, 3)
    assert_finite_tensor(output.projected_means)
    assert_finite_tensor(output.projected_conics)


def test_generic_render_returns_depth_and_2d_projections(
    cuda_scene, cuda_camera
) -> None:
    output: RenderWithDepth2DProjections = render(
        cuda_scene,
        cuda_camera,
        backend="adapter.gsplat",
        return_depth=True,
        return_2d_projections=True,
    )

    assert output.depth.shape == (1, 32, 32)
    assert output.projected_means.shape == (1, 3, 2)
    assert output.projected_conics.shape == (1, 3, 3)


def test_gsplat_packed_default_is_false() -> None:
    assert GsplatRenderOptions().packed is False


def test_gsplat_packed_true_renders_without_2d_projections(
    cuda_scene, cuda_camera
) -> None:
    output = cast(
        GsplatAlphaRenderOutput,
        render_gsplat(
            cuda_scene,
            cuda_camera,
            options=GsplatRenderOptions(packed=True),
        ),
    )
    assert output.render.shape == (1, 32, 32, 3)
    assert output.alphas.shape == (1, 32, 32)


def test_gsplat_packed_true_rejects_2d_projections(
    cuda_scene, cuda_camera
) -> None:
    with pytest.raises(ValueError, match="packed=False"):
        render(
            cuda_scene,
            cuda_camera,
            backend="adapter.gsplat",
            return_2d_projections=True,
            options=GsplatRenderOptions(packed=True),
        )


def test_render_gsplat_2dgs_rgb_returns_expected_shapes(
    cuda_scene_2d, cuda_camera
) -> None:
    output = cast(
        GsplatAlphaRenderOutput,
        render_gsplat_2dgs(cuda_scene_2d, cuda_camera),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.alphas.shape == (1, 32, 32)
    assert_finite_tensor(output.render)
    assert_finite_tensor(output.alphas)


def test_render_gsplat_2dgs_depth_returns_expected_shapes(
    cuda_scene_2d, cuda_camera
) -> None:
    output = cast(
        GsplatRenderOutput,
        render_gsplat_2dgs(
            cuda_scene_2d,
            cuda_camera,
            return_depth=True,
        ),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.alphas.shape == (1, 32, 32)
    assert output.depth.shape == (1, 32, 32)
    assert_finite_tensor(output.depth)


def test_generic_render_gsplat_2dgs_rejects_2d_projections(
    cuda_scene_2d, cuda_camera
) -> None:
    with pytest.raises(ValueError, match="does not support requested outputs"):
        render(
            cuda_scene_2d,
            cuda_camera,
            backend="adapter.gsplat_2dgs",
            return_2d_projections=True,
        )


def test_generic_render_gsplat_2dgs_returns_intersection_transforms(
    cuda_scene_2d, cuda_camera
) -> None:
    output: RenderWithDepthProjectiveIntersectionTransforms = render(
        cuda_scene_2d,
        cuda_camera,
        backend="adapter.gsplat_2dgs",
        return_depth=True,
        return_projective_intersection_transforms=True,
    )

    assert output.depth.shape == (1, 32, 32)
    assert output.projected_means.shape == (1, 3, 2)
    assert output.projective_intersection_transforms.shape == (1, 3, 3, 3)
    assert_finite_tensor(output.projected_means)
    assert_finite_tensor(output.projective_intersection_transforms)
