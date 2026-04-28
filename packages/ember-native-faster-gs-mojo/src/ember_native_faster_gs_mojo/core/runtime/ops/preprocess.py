"""Preprocess-stage custom ops for the FasterGS Mojo runtime."""

from __future__ import annotations

import torch
from ember_native_faster_gs.faster_gs.runtime.ops.preprocess import (
    _preprocess_bwd_fake as _preprocess_bwd_fake,
)
from ember_native_faster_gs.faster_gs.runtime.ops.preprocess import (
    _preprocess_fwd_fake,
)
from ember_native_faster_gs.faster_gs.runtime.ops.preprocess import (
    preprocess_bwd_op as faster_preprocess_bwd_op,
)
from torch import Tensor

from ember_native_faster_gs_mojo.core.runtime.ops._common import (
    mojo_backend,
    normalize_active_sh_bases,
)


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
    """Run the MAX/Mojo preprocess forward stage."""
    if center_positions.device.type != "cuda":
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

    primitive_count = center_positions.shape[0]
    center_positions = center_positions.contiguous()
    log_scales = log_scales.contiguous()
    unnormalized_rotations = unnormalized_rotations.contiguous()
    opacities = opacities.contiguous()
    sh_coefficients_0 = sh_coefficients_0.contiguous()
    sh_coefficients_rest = sh_coefficients_rest.contiguous()
    world_2_camera = world_2_camera.contiguous()
    camera_position = camera_position.contiguous()
    device = center_positions.device
    float_dtype = center_positions.dtype
    normalized_active_sh_bases = normalize_active_sh_bases(active_sh_bases)
    sh_coefficients_rest_for_op = sh_coefficients_rest
    if normalized_active_sh_bases == 1 and sh_coefficients_rest.shape[1] == 0:
        # MAX graph/runtime handling of zero-sized dynamic dimensions is still
        # brittle in some custom-op paths. The SH=1 specialization never reads
        # higher-order coefficients, so a zero-filled placeholder is safe here.
        sh_coefficients_rest_for_op = torch.zeros(
            (primitive_count, 1, 3),
            device=device,
            dtype=float_dtype,
        )
    outputs = (
        torch.empty((primitive_count, 2), device=device, dtype=float_dtype),
        torch.empty((primitive_count, 4), device=device, dtype=float_dtype),
        torch.empty((primitive_count, 3), device=device, dtype=float_dtype),
        torch.empty((primitive_count,), device=device, dtype=float_dtype),
        torch.empty((primitive_count,), device=device, dtype=torch.int32),
        torch.empty((primitive_count,), device=device, dtype=torch.int32),
        torch.empty((primitive_count,), device=device, dtype=torch.int32),
        torch.empty((primitive_count, 4), device=device, dtype=torch.uint16),
        torch.empty((1,), device=device, dtype=torch.int32),
        torch.empty((1,), device=device, dtype=torch.int32),
    )
    getattr(
        mojo_backend(),
        f"preprocess_fwd_pa{int(proper_antialiasing)}_sh"
        f"{normalized_active_sh_bases}",
    )(
        *outputs,
        center_positions,
        log_scales,
        unnormalized_rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest_for_op,
        world_2_camera,
        camera_position,
        torch.tensor([near_plane], device=device, dtype=float_dtype),
        torch.tensor([far_plane], device=device, dtype=float_dtype),
        torch.tensor([width], device=device, dtype=torch.int32),
        torch.tensor([height], device=device, dtype=torch.int32),
        torch.tensor([focal_x], device=device, dtype=float_dtype),
        torch.tensor([focal_y], device=device, dtype=float_dtype),
        torch.tensor([center_x], device=device, dtype=float_dtype),
        torch.tensor([center_y], device=device, dtype=float_dtype),
    )
    return outputs


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
    """Delegate preprocess backward to the FasterGS core reference."""
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
) -> tuple[Tensor, ...]:
    """Alias the staged preprocess forward op."""
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
