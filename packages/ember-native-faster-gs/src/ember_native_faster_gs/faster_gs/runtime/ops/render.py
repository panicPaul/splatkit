"""Combined render custom ops for the FasterGS native runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ember_native_faster_gs.faster_gs.reuse.factories import (
    register_render_family,
)
from ember_native_faster_gs.faster_gs.runtime.ops._common import (
    requires_grad,
)
from ember_native_faster_gs.faster_gs.runtime.ops.blend import (
    _blend_bwd_fake,
    _blend_fwd_fake,
    blend_bwd_op,
    blend_fwd_op,
)
from ember_native_faster_gs.faster_gs.runtime.ops.preprocess import (
    _preprocess_bwd_fake,
    _preprocess_fwd_fake,
    preprocess_bwd_op,
    preprocess_fwd_op,
)
from ember_native_faster_gs.faster_gs.runtime.ops.sort import (
    _sort_fwd_fake,
    sort_fwd_op,
)
from ember_native_faster_gs.faster_gs.runtime.packing import (
    pack_render_outputs,
    parse_render_outputs,
)

RenderOpOutput = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]


@torch.library.custom_op("faster_gs::render_fwd", mutates_args=())
def render_fwd_op(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
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
) -> RenderOpOutput:
    """Low-level composed render forward op."""
    preprocess_outputs = preprocess_fwd_op(
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
    sort_outputs = sort_fwd_op(
        preprocess_outputs[4],
        preprocess_outputs[5],
        preprocess_outputs[6],
        preprocess_outputs[7],
        preprocess_outputs[0],
        preprocess_outputs[1],
        preprocess_outputs[8],
        preprocess_outputs[9],
        width,
        height,
    )
    blend_outputs = blend_fwd_op(
        sort_outputs[0],
        sort_outputs[1],
        sort_outputs[2],
        sort_outputs[3],
        preprocess_outputs[0],
        preprocess_outputs[1],
        preprocess_outputs[2],
        bg_color,
        proper_antialiasing,
        width,
        height,
    )
    return pack_render_outputs(
        preprocess_outputs=preprocess_outputs,
        sort_outputs=sort_outputs,
        blend_outputs=blend_outputs,
    )


@render_fwd_op.register_fake
def _render_fwd_fake(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
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
) -> RenderOpOutput:
    preprocess_outputs = _preprocess_fwd_fake(
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
    sort_outputs = _sort_fwd_fake(
        preprocess_outputs[4],
        preprocess_outputs[5],
        preprocess_outputs[6],
        preprocess_outputs[7],
        preprocess_outputs[0],
        preprocess_outputs[1],
        preprocess_outputs[8],
        preprocess_outputs[9],
        width,
        height,
    )
    blend_outputs = _blend_fwd_fake(
        sort_outputs[0],
        sort_outputs[1],
        sort_outputs[2],
        sort_outputs[3],
        preprocess_outputs[0],
        preprocess_outputs[1],
        preprocess_outputs[2],
        bg_color,
        proper_antialiasing,
        width,
        height,
    )
    return pack_render_outputs(
        preprocess_outputs=preprocess_outputs,
        sort_outputs=sort_outputs,
        blend_outputs=blend_outputs,
    )


@torch.library.custom_op("faster_gs::render_bwd", mutates_args=())
def render_bwd_op(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    num_touched_tiles: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    image: Tensor,
    bg_color: Tensor,
    tile_final_transmittances: Tensor,
    tile_max_n_processed: Tensor,
    tile_n_processed: Tensor,
    bucket_tile_index: Tensor,
    bucket_color_transmittance: Tensor,
    grad_image: Tensor,
    proper_antialiasing: bool,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    active_sh_bases: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Low-level composed render backward op."""
    grad_projected_means, grad_conic_opacity, grad_colors_rgb = blend_bwd_op(
        grad_image,
        image,
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        projected_means,
        conic_opacity,
        colors_rgb,
        bg_color,
        tile_final_transmittances,
        tile_max_n_processed,
        tile_n_processed,
        bucket_tile_index,
        bucket_color_transmittance,
        proper_antialiasing,
        width,
        height,
    )
    return preprocess_bwd_op(
        center_positions,
        log_scales,
        unnormalized_rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        world_2_camera,
        camera_position,
        num_touched_tiles,
        grad_projected_means,
        grad_conic_opacity,
        grad_colors_rgb,
        torch.zeros_like(projected_means[:, 0]),
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        proper_antialiasing,
        active_sh_bases,
    )


@render_bwd_op.register_fake
def _render_bwd_fake(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    num_touched_tiles: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    image: Tensor,
    bg_color: Tensor,
    tile_final_transmittances: Tensor,
    tile_max_n_processed: Tensor,
    tile_n_processed: Tensor,
    bucket_tile_index: Tensor,
    bucket_color_transmittance: Tensor,
    grad_image: Tensor,
    proper_antialiasing: bool,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    active_sh_bases: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del (
        image,
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bg_color,
        tile_final_transmittances,
        tile_max_n_processed,
        tile_n_processed,
        bucket_tile_index,
        bucket_color_transmittance,
    )
    grad_projected_means, grad_conic_opacity, grad_colors_rgb = _blend_bwd_fake(
        grad_image,
        center_positions,
        center_positions,
        center_positions,
        center_positions,
        projected_means,
        conic_opacity,
        colors_rgb,
        center_positions,
        center_positions,
        center_positions,
        center_positions,
        center_positions,
        center_positions,
        proper_antialiasing,
        width,
        height,
    )
    return _preprocess_bwd_fake(
        center_positions,
        log_scales,
        unnormalized_rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        world_2_camera,
        camera_position,
        num_touched_tiles,
        grad_projected_means,
        grad_conic_opacity,
        grad_colors_rgb,
        torch.zeros_like(projected_means[:, 0]),
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        proper_antialiasing,
        active_sh_bases,
    )


def _render_impl(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
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
) -> RenderOpOutput:
    """Autograd-enabled full render op."""
    return render_fwd_op(
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


def _render_fake(*args: Any) -> RenderOpOutput:
    """Fake implementation for the autograd render op."""
    return _render_fwd_fake(*args)


def _render_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, ...],
) -> None:
    if not requires_grad(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
    ):
        ctx.has_context = False
        return
    render_result = parse_render_outputs(output)
    ctx.has_context = True
    ctx.save_for_backward(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        render_result.preprocess.num_touched_tiles,
        render_result.preprocess.projected_means,
        render_result.preprocess.conic_opacity,
        render_result.preprocess.colors_rgb,
        render_result.sort.instance_primitive_indices,
        render_result.sort.tile_instance_ranges,
        render_result.sort.tile_bucket_offsets,
        render_result.image,
        inputs[16],
        render_result.blend.tile_final_transmittances,
        render_result.blend.tile_max_n_processed,
        render_result.blend.tile_n_processed,
        render_result.blend.bucket_tile_index,
        render_result.blend.bucket_color_transmittance,
    )
    ctx.width = inputs[10]
    ctx.height = inputs[11]
    ctx.focal_x = inputs[12]
    ctx.focal_y = inputs[13]
    ctx.center_x = inputs[14]
    ctx.center_y = inputs[15]
    ctx.proper_antialiasing = inputs[17]
    ctx.active_sh_bases = inputs[18]


def _render_backward(
    ctx: Any,
    grad_image: Tensor,
    *grad_aux: Tensor,
) -> tuple[Tensor | None, ...]:
    del grad_aux
    if not ctx.has_context:
        return (None,) * 19
    grads = render_bwd_op(
        *ctx.saved_tensors,
        grad_image,
        ctx.proper_antialiasing,
        ctx.width,
        ctx.height,
        ctx.focal_x,
        ctx.focal_y,
        ctx.center_x,
        ctx.center_y,
        ctx.active_sh_bases,
    )
    return (*grads, *(None,) * 13)


render_op = register_render_family(
    op_name="faster_gs::render",
    forward_impl=_render_impl,
    fake_impl=_render_fake,
    setup_context=_render_setup_context,
    backward_impl=_render_backward,
)
