"""Public staged runtime API for the FasterGS native backend."""

from __future__ import annotations

from torch import Tensor

from ember_native_faster_gs.faster_gs.runtime.ops import (
    blend_metric_counts_fwd_op,
    blend_op,
    preprocess_op,
    render_op,
    sort_op,
)
from ember_native_faster_gs.faster_gs.runtime.packing import (
    make_render_result,
    parse_blend_outputs,
    parse_preprocess_outputs,
    parse_sort_outputs,
)
from ember_native_faster_gs.faster_gs.runtime.types import (
    BlendResult,
    PreprocessResult,
    RenderResult,
    SortResult,
)


def preprocess(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_to_camera_matrix: Tensor,
    camera_position: Tensor,
    *,
    near_plane: float,
    far_plane: float,
    image_width: int,
    image_height: int,
    focal_length_x: float,
    focal_length_y: float,
    principal_point_x: float,
    principal_point_y: float,
    mip_splatting_screen_filter: bool,
    active_sh_bases: int,
) -> PreprocessResult:
    """Run the native preprocess stage."""
    return parse_preprocess_outputs(
        preprocess_op(
            center_positions,
            log_scales,
            unnormalized_rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            world_to_camera_matrix,
            camera_position,
            near_plane,
            far_plane,
            image_width,
            image_height,
            focal_length_x,
            focal_length_y,
            principal_point_x,
            principal_point_y,
            mip_splatting_screen_filter,
            active_sh_bases,
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
    image_width: int,
    image_height: int,
) -> SortResult:
    """Run the native sort stage."""
    return parse_sort_outputs(
        sort_op(
            depth_keys,
            primitive_indices,
            num_touched_tiles,
            screen_bounds,
            projected_means,
            conic_opacity,
            visible_count,
            instance_count,
            image_width,
            image_height,
        )
    )


def blend(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    bg_color: Tensor,
    mip_splatting_screen_filter: bool,
    *,
    image_width: int,
    image_height: int,
) -> BlendResult:
    """Run the native blend stage."""
    return parse_blend_outputs(
        blend_op(
            instance_primitive_indices,
            tile_instance_ranges,
            tile_bucket_offsets,
            bucket_count,
            projected_means,
            conic_opacity,
            colors_rgb,
            bg_color,
            mip_splatting_screen_filter,
            image_width,
            image_height,
        )
    )


def blend_metric_counts(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    bg_color: Tensor,
    metric_map: Tensor,
    mip_splatting_screen_filter: bool,
    *,
    image_width: int,
    image_height: int,
) -> Tensor:
    """Attribute a binary/int metric map to native blend contributors."""
    return blend_metric_counts_fwd_op(
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bucket_count,
        projected_means,
        conic_opacity,
        colors_rgb,
        bg_color,
        metric_map,
        mip_splatting_screen_filter,
        image_width,
        image_height,
    )


def render(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_to_camera_matrix: Tensor,
    camera_position: Tensor,
    *,
    near_plane: float,
    far_plane: float,
    image_width: int,
    image_height: int,
    focal_length_x: float,
    focal_length_y: float,
    principal_point_x: float,
    principal_point_y: float,
    bg_color: Tensor,
    mip_splatting_screen_filter: bool,
    active_sh_bases: int,
    densification_info: Tensor | None = None,
) -> RenderResult:
    """Run the full native render stage."""
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
            world_to_camera_matrix,
            camera_position,
            near_plane,
            far_plane,
            image_width,
            image_height,
            focal_length_x,
            focal_length_y,
            principal_point_x,
            principal_point_y,
            bg_color,
            mip_splatting_screen_filter,
            active_sh_bases,
            resolved_densification_info,
        )
    )


__all__ = [
    "BlendResult",
    "PreprocessResult",
    "RenderResult",
    "SortResult",
    "blend",
    "blend_metric_counts",
    "preprocess",
    "render",
    "sort",
]
