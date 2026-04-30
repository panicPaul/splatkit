"""Public staged runtime API for the native FastGS backend."""

from __future__ import annotations

from torch import Tensor

from ember_native_faster_gs.faster_gs.runtime.packing import (
    make_render_result,
    parse_preprocess_outputs,
    parse_sort_outputs,
)
from ember_native_faster_gs.faster_gs.runtime.types import (
    PreprocessResult,
    RenderResult,
    SortResult,
)
from ember_native_faster_gs.fastgs.runtime.ops import (
    preprocess_fwd_op,
    render_op,
    sort_fwd_op,
)


def preprocess(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    *,
    near_plane: float,
    far_plane: float,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    proper_antialiasing: bool,
    active_sh_bases: int,
    compact_box_scale: float,
) -> PreprocessResult:
    """Run the native FastGS preprocess stage."""
    return parse_preprocess_outputs(
        preprocess_fwd_op(
            center_positions,
            log_scales,
            unnormalized_rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            world_2_camera,
            camera_position,
            near_plane,
            far_plane,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            proper_antialiasing,
            active_sh_bases,
            compact_box_scale,
        )
    )


def sort(
    depth_keys: Tensor,
    primitive_indices: Tensor,
    num_touched_tiles: Tensor,
    screen_bounds: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    visible_count: Tensor,
    instance_count: Tensor,
    *,
    width: int,
    height: int,
    compact_box_scale: float,
) -> SortResult:
    """Run the native FastGS sort stage."""
    return parse_sort_outputs(
        sort_fwd_op(
            depth_keys,
            primitive_indices,
            num_touched_tiles,
            screen_bounds,
            projected_means,
            conic_opacity,
            visible_count,
            instance_count,
            width,
            height,
            compact_box_scale,
        )
    )


def render(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    *,
    near_plane: float,
    far_plane: float,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    bg_color: Tensor,
    proper_antialiasing: bool,
    active_sh_bases: int,
    compact_box_scale: float,
    densification_info: Tensor | None = None,
) -> RenderResult:
    """Run the full native FastGS render stage."""
    resolved_densification_info = (
        densification_info
        if densification_info is not None
        else center_positions.new_empty((0,))
    )
    return make_render_result(
        render_op(
            center_positions,
            log_scales,
            unnormalized_rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            world_2_camera,
            camera_position,
            near_plane,
            far_plane,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            bg_color,
            proper_antialiasing,
            active_sh_bases,
            compact_box_scale,
            resolved_densification_info,
        )
    )


__all__ = [
    "PreprocessResult",
    "RenderResult",
    "SortResult",
    "preprocess",
    "render",
    "sort",
]
