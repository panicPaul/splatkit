from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch
from splatkit.core import BACKEND_REGISTRY, render
from splatkit_native_backends.faster_gs_depth import (
    FasterGSDepthNativeDepthRenderOutput,
    register as register_depth,
    render_faster_gs_depth,
)
from splatkit_native_backends.faster_gs import (
    FasterGSNativeRenderOutput,
    register as register_root,
    render_faster_gs_native,
)
from splatkit_native_backends.gaussian_pop import (
    GaussianPopNativeDepthGaussianImpactScoreRenderOutput,
    GaussianPopNativeGaussianImpactScoreRenderOutput,
    register,
    render_gaussian_pop,
)

register()
register_root()
register_depth()


def _remove_primitive(scene, primitive_index: int):
    keep_mask = torch.ones(
        scene.center_position.shape[0],
        dtype=torch.bool,
        device=scene.center_position.device,
    )
    keep_mask[primitive_index] = False
    return replace(
        scene,
        center_position=scene.center_position[keep_mask],
        log_scales=scene.log_scales[keep_mask],
        quaternion_orientation=scene.quaternion_orientation[keep_mask],
        logit_opacity=scene.logit_opacity[keep_mask],
        feature=scene.feature[keep_mask],
    )


@pytest.mark.backend
@pytest.mark.cuda
def test_gaussian_pop_backend_returns_score_without_perturbing_rgb(
    cuda_visible_scene,
    cuda_camera,
) -> None:
    output = cast(
        GaussianPopNativeGaussianImpactScoreRenderOutput,
        render_gaussian_pop(
            cuda_visible_scene,
            cuda_camera,
            return_gaussian_impact_score=True,
        ),
    )
    root_output = cast(
        FasterGSNativeRenderOutput,
        render_faster_gs_native(cuda_visible_scene, cuda_camera),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.gaussian_impact_score.shape == (
        1,
        cuda_visible_scene.center_position.shape[0],
    )
    assert torch.isfinite(output.gaussian_impact_score).all()
    assert output.render.abs().sum() > 0
    torch.testing.assert_close(output.render, root_output.render, rtol=1e-4, atol=2e-4)


@pytest.mark.backend
@pytest.mark.cuda
def test_gaussian_pop_score_matches_naive_leave_one_out_render_error(
    cuda_visible_scene,
    cuda_camera,
) -> None:
    output = cast(
        GaussianPopNativeGaussianImpactScoreRenderOutput,
        render_gaussian_pop(
            cuda_visible_scene,
            cuda_camera,
            return_gaussian_impact_score=True,
        ),
    )
    full_render = output.render[0]
    expected_scores = []
    assert full_render.abs().sum() > 0
    for primitive_index in range(int(cuda_visible_scene.center_position.shape[0])):
        removed_scene = _remove_primitive(cuda_visible_scene, primitive_index)
        removed_output = cast(
            FasterGSNativeRenderOutput,
            render_faster_gs_native(removed_scene, cuda_camera),
        )
        expected_scores.append(
            ((full_render - removed_output.render[0]) ** 2).sum()
        )
    expected = torch.stack(expected_scores, dim=0)
    torch.testing.assert_close(
        output.gaussian_impact_score[0],
        expected,
        rtol=1e-4,
        atol=2e-4,
    )


@pytest.mark.backend
@pytest.mark.cuda
def test_gaussian_pop_depth_and_score_match_depth_backend(
    cuda_visible_scene,
    cuda_camera,
) -> None:
    score_output = cast(
        GaussianPopNativeGaussianImpactScoreRenderOutput,
        render_gaussian_pop(
            cuda_visible_scene,
            cuda_camera,
            return_gaussian_impact_score=True,
        ),
    )
    depth_output = cast(
        GaussianPopNativeDepthGaussianImpactScoreRenderOutput,
        render_gaussian_pop(
            cuda_visible_scene,
            cuda_camera,
            return_depth=True,
            return_gaussian_impact_score=True,
        ),
    )
    expected_depth = cast(
        FasterGSDepthNativeDepthRenderOutput,
        render_faster_gs_depth(
            cuda_visible_scene,
            cuda_camera,
            return_depth=True,
        ),
    )

    torch.testing.assert_close(depth_output.render, expected_depth.render, rtol=1e-4, atol=2e-4)
    torch.testing.assert_close(depth_output.depth, expected_depth.depth, rtol=1e-4, atol=2e-4)
    torch.testing.assert_close(
        depth_output.gaussian_impact_score,
        score_output.gaussian_impact_score,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.backend
@pytest.mark.cuda
def test_generic_render_dispatches_to_gaussian_pop_backend(
    cuda_visible_scene,
    cuda_camera,
) -> None:
    output = cast(
        GaussianPopNativeDepthGaussianImpactScoreRenderOutput,
        render(
            cuda_visible_scene,
            cuda_camera,
            backend="gaussian_pop",
            return_depth=True,
            return_gaussian_impact_score=True,
        ),
    )

    assert BACKEND_REGISTRY["gaussian_pop"].name == "gaussian_pop"
    assert output.render.shape == (1, 32, 32, 3)
    assert output.depth.shape == (1, 32, 32)
    assert output.gaussian_impact_score.shape == (
        1,
        cuda_visible_scene.center_position.shape[0],
    )
