"""GaussianPOP runtime render composition."""

from __future__ import annotations

from torch import Tensor

from splatkit_native_backends.faster_gs.reuse import (
    preprocess_op,
    sort_op,
)
from splatkit_native_backends.faster_gs.runtime.packing import (
    parse_preprocess_outputs,
    parse_sort_outputs,
)
from splatkit_native_backends.gaussian_pop.runtime.blend import (
    blend,
)
from splatkit_native_backends.gaussian_pop.runtime.types import (
    RenderResult,
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
    return_depth: bool,
    return_gaussian_impact_score: bool,
) -> RenderResult:
    """Run the GaussianPOP render path for one camera."""
    preprocess_result = parse_preprocess_outputs(
        preprocess_op(
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
        )
    )
    sort_result = parse_sort_outputs(
        sort_op(
            preprocess_result.depth_keys,
            preprocess_result.primitive_indices,
            preprocess_result.num_touched_tiles,
            preprocess_result.screen_bounds,
            preprocess_result.projected_means,
            preprocess_result.conic_opacity,
            preprocess_result.visible_count,
            preprocess_result.instance_count,
            width,
            height,
        )
    )
    blend_result = blend(
        sort_result.instance_primitive_indices,
        sort_result.tile_instance_ranges,
        sort_result.tile_bucket_offsets,
        sort_result.bucket_count,
        preprocess_result.projected_means,
        preprocess_result.conic_opacity,
        preprocess_result.colors_rgb,
        preprocess_result.primitive_depth,
        bg_color,
        proper_antialiasing,
        width=width,
        height=height,
        return_depth=return_depth,
        return_gaussian_impact_score=return_gaussian_impact_score,
    )
    return RenderResult(
        image=blend_result.image,
        depth=blend_result.depth,
        gaussian_impact_score=blend_result.gaussian_impact_score,
    )
