"""Tests for declarative GS render-view nodes and effects."""

from dataclasses import dataclass

import numpy as np
import torch

from marimo_3dv.ops.gs import (
    CompiledGsRenderView,
    FilterOpacityConfig,
    FilterSizeConfig,
    MaxShDegreeConfig,
    ShowDistributionConfig,
    compile_gs_render_view,
    filter_opacity_op,
    filter_size_op,
    gs_render_view,
    max_sh_degree_op,
    show_distribution_op,
)
from marimo_3dv.pipeline.context import ViewerContext
from marimo_3dv.pipeline.gui import RenderResult
from marimo_3dv.viewer.widget import ViewerState


@dataclass
class _FakeSplats:
    center_positions: torch.Tensor
    log_half_extents: torch.Tensor
    quaternion_orientation: torch.Tensor
    spherical_harmonics: torch.Tensor
    opacity_logits: torch.Tensor
    sh_degree: int


def _make_splats(
    n: int = 10,
    sh_degree: int = 3,
    opacity_logit: float = 0.0,
    log_scale: float = 0.0,
) -> _FakeSplats:
    num_bases = (sh_degree + 1) ** 2
    return _FakeSplats(
        center_positions=torch.zeros(n, 3),
        log_half_extents=torch.full((n, 3), log_scale),
        quaternion_orientation=torch.zeros(n, 4),
        spherical_harmonics=torch.zeros(n, num_bases, 3),
        opacity_logits=torch.full((n, 1), opacity_logit),
        sh_degree=sh_degree,
    )


def _context() -> ViewerContext:
    return ViewerContext(viewer_state=ViewerState(), last_click=None)


def test_max_sh_degree_is_symbolic_until_compile() -> None:
    splats = _make_splats(sh_degree=3)
    view = gs_render_view(splats)
    result = max_sh_degree_op().apply(
        view,
        MaxShDegreeConfig(max_sh_degree=1),
        _context(),
    )

    assert result is not view
    assert result.max_sh_degree == 1
    assert splats.spherical_harmonics.shape[1] == 16

    compiled = compile_gs_render_view(result)
    assert compiled.sh_degree == 1
    assert compiled.spherical_harmonics.shape[1] == 4


def test_filter_opacity_builds_symbolic_mask() -> None:
    low = _make_splats(n=5, opacity_logit=-5.0)
    high = _make_splats(n=5, opacity_logit=2.0)

    @dataclass
    class Mixed:
        center_positions: torch.Tensor
        log_half_extents: torch.Tensor
        quaternion_orientation: torch.Tensor
        spherical_harmonics: torch.Tensor
        opacity_logits: torch.Tensor
        sh_degree: int = 3

    mixed = Mixed(
        center_positions=torch.zeros(10, 3),
        log_half_extents=torch.cat(
            [low.log_half_extents, high.log_half_extents]
        ),
        quaternion_orientation=torch.zeros(10, 4),
        spherical_harmonics=torch.cat(
            [low.spherical_harmonics, high.spherical_harmonics]
        ),
        opacity_logits=torch.cat([low.opacity_logits, high.opacity_logits]),
    )

    view = filter_opacity_op(default_threshold=0.1).apply(
        gs_render_view(mixed),
        FilterOpacityConfig(opacity_threshold=0.1),
        _context(),
    )
    assert view.keep_mask is not None
    compiled = compile_gs_render_view(view)
    assert compiled.opacity_logits.shape[0] == 5


def test_filter_size_composes_with_existing_mask() -> None:
    splats = _make_splats(n=8, opacity_logit=2.0, log_scale=0.5)
    opacity_view = filter_opacity_op(default_threshold=0.1).apply(
        gs_render_view(splats),
        FilterOpacityConfig(opacity_threshold=0.1),
        _context(),
    )
    filtered_view = filter_size_op(default_max_log_extent=0.1).apply(
        opacity_view,
        FilterSizeConfig(max_log_extent=0.1),
        _context(),
    )

    assert filtered_view.keep_mask is not None
    compiled = compile_gs_render_view(filtered_view)
    assert compiled.log_half_extents.shape[0] == 0
    assert splats.log_half_extents.shape[0] == 8


def test_compile_gs_render_view_preserves_source_tensors_without_filters() -> (
    None
):
    splats = _make_splats(n=4)
    compiled = compile_gs_render_view(gs_render_view(splats))
    assert isinstance(compiled, CompiledGsRenderView)
    assert (
        compiled.opacity_logits.data_ptr() == splats.opacity_logits.data_ptr()
    )
    assert (
        compiled.spherical_harmonics.data_ptr()
        == splats.spherical_harmonics.data_ptr()
    )


def test_show_distribution_no_op_when_disabled() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = RenderResult(
        image=image,
        metadata={"projected_means": torch.zeros(5, 2)},
    )
    out = show_distribution_op().apply(
        result,
        ShowDistributionConfig(show_distribution=False),
        _context(),
        None,
    )
    assert out is result


def test_show_distribution_no_op_without_metadata() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = RenderResult(image=image, metadata={})
    out = show_distribution_op().apply(
        result,
        ShowDistributionConfig(show_distribution=True),
        _context(),
        None,
    )
    assert out is result


def test_show_distribution_blends_image() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    means = torch.tensor([[4.0, 4.0], [8.0, 8.0], [12.0, 12.0]])
    result = RenderResult(image=image, metadata={"projected_means": means})
    out = show_distribution_op().apply(
        result,
        ShowDistributionConfig(show_distribution=True, distribution_alpha=0.5),
        _context(),
        None,
    )
    assert not np.array_equal(out.image, image)
