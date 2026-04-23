"""Preprocess-stage custom ops for the FasterGS Mojo runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from splatkit_native_faster_gs.faster_gs.runtime.ops.preprocess import (
    _preprocess_bwd_fake,
    _preprocess_fwd_fake,
    preprocess_bwd_op as faster_preprocess_bwd_op,
)
from splatkit_native_faster_gs.faster_gs.runtime.ops.preprocess import (
    preprocess_fwd_op as faster_preprocess_fwd_op,
)
from splatkit_native_faster_gs.faster_gs.runtime.packing import (
    parse_preprocess_outputs,
)


@torch.library.custom_op("faster_gs_mojo::preprocess_fwd", mutates_args=())
def preprocess_fwd_op(
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
    proper_antialiasing: bool,
    active_sh_bases: int,
) -> tuple[
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
]:
    """Delegate preprocess forward to the native FasterGS root backend."""
    return faster_preprocess_fwd_op(
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


@preprocess_fwd_op.register_fake
def _preprocess_fwd_fake_local(
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
    proper_antialiasing: bool,
    active_sh_bases: int,
) -> tuple[
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
]:
    return _preprocess_fwd_fake(
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


@torch.library.custom_op("faster_gs_mojo::preprocess_bwd", mutates_args=())
def preprocess_bwd_op(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    num_touched_tiles: Tensor,
    grad_projected_means: Tensor,
    grad_conic_opacity: Tensor,
    grad_colors_rgb: Tensor,
    grad_primitive_depth: Tensor,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    proper_antialiasing: bool,
    active_sh_bases: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Delegate preprocess backward to the native FasterGS root backend."""
    return faster_preprocess_bwd_op(
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
        grad_primitive_depth,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        proper_antialiasing,
        active_sh_bases,
    )


@preprocess_bwd_op.register_fake
def _preprocess_bwd_fake_local(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    num_touched_tiles: Tensor,
    grad_projected_means: Tensor,
    grad_conic_opacity: Tensor,
    grad_colors_rgb: Tensor,
    grad_primitive_depth: Tensor,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    proper_antialiasing: bool,
    active_sh_bases: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        grad_primitive_depth,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        proper_antialiasing,
        active_sh_bases,
    )


@torch.library.custom_op("faster_gs_mojo::preprocess", mutates_args=())
def preprocess_op(
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
    proper_antialiasing: bool,
    active_sh_bases: int,
) -> tuple[
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
]:
    """Autograd-enabled preprocess op."""
    return preprocess_fwd_op(
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


@preprocess_op.register_fake
def _preprocess_fake(
    *args: Any,
) -> tuple[
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
]:
    return _preprocess_fwd_fake(*args)


def _preprocess_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, ...],
) -> None:
    preprocess_result = parse_preprocess_outputs(output)
    ctx.save_for_backward(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        preprocess_result.num_touched_tiles,
    )
    ctx.width = inputs[10]
    ctx.height = inputs[11]
    ctx.focal_x = inputs[12]
    ctx.focal_y = inputs[13]
    ctx.center_x = inputs[14]
    ctx.center_y = inputs[15]
    ctx.proper_antialiasing = inputs[16]
    ctx.active_sh_bases = inputs[17]


def _preprocess_backward(
    ctx: Any,
    grad_projected_means: Tensor,
    grad_conic_opacity: Tensor,
    grad_colors_rgb: Tensor,
    grad_primitive_depth: Tensor,
    grad_depth_keys: Tensor,
    grad_primitive_indices: Tensor,
    grad_num_touched_tiles: Tensor,
    grad_screen_bounds: Tensor,
    grad_visible_count: Tensor,
    grad_instance_count: Tensor,
) -> tuple[Tensor | None, ...]:
    del (
        grad_depth_keys,
        grad_primitive_indices,
        grad_num_touched_tiles,
        grad_screen_bounds,
        grad_visible_count,
        grad_instance_count,
    )
    grads = preprocess_bwd_op(
        *ctx.saved_tensors,
        grad_projected_means,
        grad_conic_opacity,
        grad_colors_rgb,
        grad_primitive_depth,
        ctx.width,
        ctx.height,
        ctx.focal_x,
        ctx.focal_y,
        ctx.center_x,
        ctx.center_y,
        ctx.proper_antialiasing,
        ctx.active_sh_bases,
    )
    return (*grads, *(None,) * 12)


preprocess_op.register_autograd(
    _preprocess_backward,
    setup_context=_preprocess_setup_context,
)
