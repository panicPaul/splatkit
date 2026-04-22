"""Public runtime API for the FasterGS depth native backend."""

from __future__ import annotations

from torch import Tensor

from splatkit_native_faster_gs.faster_gs_depth.runtime.blend import (
    blend_op,
)
from splatkit_native_faster_gs.faster_gs_depth.runtime.packing import (
    make_render_result,
    parse_blend_outputs,
)
from splatkit_native_faster_gs.faster_gs_depth.runtime.render import (
    render_op,
)
from splatkit_native_faster_gs.faster_gs_depth.runtime.types import (
    BlendResult,
    RenderResult,
)


def blend(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    primitive_depth: Tensor,
    bg_color: Tensor,
    proper_antialiasing: bool,
    *,
    width: int,
    height: int,
) -> BlendResult:
    """Run the depth-aware blend stage."""
    return parse_blend_outputs(
        blend_op(
            instance_primitive_indices,
            tile_instance_ranges,
            tile_bucket_offsets,
            bucket_count,
            projected_means,
            conic_opacity,
            colors_rgb,
            primitive_depth,
            bg_color,
            proper_antialiasing,
            width,
            height,
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
) -> RenderResult:
    """Run the full depth-aware render stage."""
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
        )
    )


__all__ = [
    "BlendResult",
    "RenderResult",
    "blend",
    "render",
]
