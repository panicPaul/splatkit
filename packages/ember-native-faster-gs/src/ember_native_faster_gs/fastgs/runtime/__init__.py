"""Public staged runtime API for the native FastGS backend."""

from __future__ import annotations

from torch import Tensor

from ember_native_faster_gs.faster_gs.runtime.packing import (
    make_render_result,
    parse_blend_outputs,
    parse_preprocess_outputs,
    parse_render_outputs,
    parse_sort_outputs,
)
from ember_native_faster_gs.faster_gs.runtime.types import (
    BlendResult,
    PreprocessResult,
    RenderResult,
    SortResult,
)
from ember_native_faster_gs.fastgs.runtime.ops import (
    blend_metric_counts_fwd_op,
    blend_op,
    preprocess_fwd_op,
    render_op,
    sort_fwd_op,
    update_densification_radii_fwd,
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
    mip_splatting_screen_filter: bool,
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
            mip_splatting_screen_filter,
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
    """Run the native FastGS blend stage."""
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
    """Attribute a binary/int metric map to native FastGS contributors."""
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
    mip_splatting_screen_filter: bool,
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
    raw_outputs = render_op(
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
        mip_splatting_screen_filter,
        active_sh_bases,
        compact_box_scale,
        resolved_densification_info,
    )
    if (
        densification_info is not None
        and densification_info.ndim == 2
        and densification_info.shape[0] >= 4
    ):
        preprocess_result = parse_render_outputs(raw_outputs).preprocess
        update_densification_radii_fwd(
            preprocess_result.num_touched_tiles,
            preprocess_result.conic_opacity,
            densification_info,
        )
    return make_render_result(raw_outputs)


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
