"""Gaussian Wrapping staged CUDA custom ops."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ember_native_faster_gs.gaussian_wrapping.runtime._extension import (
    load_ours_extension,
    load_radegs_extension,
)


def _requires_grad(*tensors: Tensor) -> bool:
    return any(tensor.requires_grad for tensor in tensors)


def _zero_like_if_none(grad: Tensor | None, reference: Tensor) -> Tensor:
    if grad is None:
        return torch.zeros_like(reference)
    return grad


OursRenderForwardOutput = tuple[
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
OursRenderBackwardOutput = tuple[
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
RadegsRenderForwardOutput = tuple[
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
RadegsRenderBackwardOutput = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]


def ours_backend() -> Any:
    """Return the loaded Gaussian Wrapping ``ours`` CUDA extension."""
    return load_ours_extension()


def radegs_backend() -> Any:
    """Return the loaded Gaussian Wrapping RaDe-GS CUDA extension."""
    return load_radegs_extension()


@torch.library.custom_op("gaussian_wrapping::ours_render_fwd", mutates_args=())
def ours_render_fwd_op(
    background: Tensor,
    means3D: Tensor,
    colors: Tensor,
    opacity: Tensor,
    scales: Tensor,
    rotations: Tensor,
    cov3D_precomp: Tensor,
    sh: Tensor,
    sg_axis: Tensor,
    sg_sharpness: Tensor,
    sg_color: Tensor,
    sh_degree: int,
    sg_degree: int,
    scale_modifier: float,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    image_height: int,
    image_width: int,
    campos: Tensor,
    prefiltered: bool,
    require_depth: bool,
    debug: bool,
) -> OursRenderForwardOutput:
    """Run the upstream Gaussian Wrapping ``ours`` render forward CUDA stage."""
    return ours_backend().render_fwd(
        background,
        means3D,
        colors,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
        sh,
        sg_axis,
        sg_sharpness,
        sg_color,
        sh_degree,
        sg_degree,
        scale_modifier,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        image_height,
        image_width,
        campos,
        prefiltered,
        require_depth,
        debug,
    )


@ours_render_fwd_op.register_fake
def _ours_render_fwd_fake(
    background: Tensor,
    means3D: Tensor,
    colors: Tensor,
    opacity: Tensor,
    scales: Tensor,
    rotations: Tensor,
    cov3D_precomp: Tensor,
    sh: Tensor,
    sg_axis: Tensor,
    sg_sharpness: Tensor,
    sg_color: Tensor,
    sh_degree: int,
    sg_degree: int,
    scale_modifier: float,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    image_height: int,
    image_width: int,
    campos: Tensor,
    prefiltered: bool,
    require_depth: bool,
    debug: bool,
) -> OursRenderForwardOutput:
    del (
        background,
        colors,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
        sh,
        sg_axis,
        sg_sharpness,
        sg_color,
        sh_degree,
        sg_degree,
        scale_modifier,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        campos,
        prefiltered,
        require_depth,
        debug,
    )
    num_splats = int(means3D.shape[0])
    device = means3D.device
    dtype = means3D.dtype
    return (
        torch.empty((1,), device=device, dtype=torch.int32),
        torch.empty((3, image_height, image_width), device=device, dtype=dtype),
        torch.empty((1, image_height, image_width), device=device, dtype=dtype),
        torch.empty((3, image_height, image_width), device=device, dtype=dtype),
        torch.empty((1, image_height, image_width), device=device, dtype=dtype),
        torch.empty((1, image_height, image_width), device=device, dtype=dtype),
        torch.empty((1, image_height, image_width), device=device, dtype=dtype),
        torch.empty((1, image_height, image_width), device=device, dtype=dtype),
        torch.empty((num_splats,), device=device, dtype=torch.int32),
        torch.empty((0,), device=device, dtype=torch.uint8),
        torch.empty((0,), device=device, dtype=torch.uint8),
        torch.empty((0,), device=device, dtype=torch.uint8),
        torch.empty((0,), device=device, dtype=torch.uint8),
    )


@torch.library.custom_op("gaussian_wrapping::ours_render_bwd", mutates_args=())
def ours_render_bwd_op(
    background: Tensor,
    means3D: Tensor,
    colors: Tensor,
    opacity: Tensor,
    scales: Tensor,
    rotations: Tensor,
    cov3D_precomp: Tensor,
    sh: Tensor,
    sg_axis: Tensor,
    sg_sharpness: Tensor,
    sg_color: Tensor,
    sh_degree: int,
    sg_degree: int,
    scale_modifier: float,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    grad_color: Tensor,
    grad_median_depth: Tensor,
    grad_color_square: Tensor,
    grad_depth: Tensor,
    grad_depth_square: Tensor,
    grad_alpha: Tensor,
    grad_normal: Tensor,
    alpha: Tensor,
    normal: Tensor,
    median_depth: Tensor,
    campos: Tensor,
    radii: Tensor,
    geom_buffer: Tensor,
    rendered_count: Tensor,
    binning_buffer: Tensor,
    image_buffer: Tensor,
    tile_buffer: Tensor,
    require_depth: bool,
    debug: bool,
) -> OursRenderBackwardOutput:
    """Run the upstream Gaussian Wrapping ``ours`` render backward CUDA stage."""
    return ours_backend().render_bwd(
        background,
        means3D,
        colors,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
        sh,
        sg_axis,
        sg_sharpness,
        sg_color,
        sh_degree,
        sg_degree,
        scale_modifier,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        grad_color,
        grad_median_depth,
        grad_color_square,
        grad_depth,
        grad_depth_square,
        grad_alpha,
        grad_normal,
        alpha,
        normal,
        median_depth,
        campos,
        radii,
        geom_buffer,
        rendered_count,
        binning_buffer,
        image_buffer,
        tile_buffer,
        require_depth,
        debug,
    )


@ours_render_bwd_op.register_fake
def _ours_render_bwd_fake(
    background: Tensor,
    means3D: Tensor,
    colors: Tensor,
    opacity: Tensor,
    scales: Tensor,
    rotations: Tensor,
    cov3D_precomp: Tensor,
    sh: Tensor,
    sg_axis: Tensor,
    sg_sharpness: Tensor,
    sg_color: Tensor,
    sh_degree: int,
    sg_degree: int,
    scale_modifier: float,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    grad_color: Tensor,
    grad_median_depth: Tensor,
    grad_color_square: Tensor,
    grad_depth: Tensor,
    grad_depth_square: Tensor,
    grad_alpha: Tensor,
    grad_normal: Tensor,
    alpha: Tensor,
    normal: Tensor,
    median_depth: Tensor,
    campos: Tensor,
    radii: Tensor,
    geom_buffer: Tensor,
    rendered_count: Tensor,
    binning_buffer: Tensor,
    image_buffer: Tensor,
    tile_buffer: Tensor,
    require_depth: bool,
    debug: bool,
) -> OursRenderBackwardOutput:
    del (
        background,
        opacity,
        scale_modifier,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        grad_color,
        grad_median_depth,
        grad_color_square,
        grad_depth,
        grad_depth_square,
        grad_alpha,
        grad_normal,
        alpha,
        normal,
        median_depth,
        campos,
        radii,
        geom_buffer,
        rendered_count,
        binning_buffer,
        image_buffer,
        tile_buffer,
        require_depth,
        debug,
    )
    return (
        torch.empty(
            (means3D.shape[0], 3), device=means3D.device, dtype=means3D.dtype
        ),
        torch.empty_like(colors),
        torch.empty_like(opacity),
        torch.empty_like(means3D),
        torch.empty_like(cov3D_precomp),
        torch.empty_like(sh),
        torch.empty_like(sg_axis),
        torch.empty_like(sg_sharpness),
        torch.empty_like(sg_color),
        torch.empty_like(scales),
        torch.empty_like(rotations),
    )


def _ours_render_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: OursRenderForwardOutput,
) -> None:
    if not _requires_grad(*inputs[:11]):
        ctx.has_context = False
        return
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
        inputs[8],
        inputs[9],
        inputs[10],
        inputs[14],
        inputs[15],
        inputs[21],
        output[2],
        output[3],
        output[4],
        output[8],
        output[9],
        output[0],
        output[10],
        output[11],
        output[12],
    )
    ctx.sh_degree = inputs[11]
    ctx.sg_degree = inputs[12]
    ctx.scale_modifier = inputs[13]
    ctx.tan_fovx = inputs[16]
    ctx.tan_fovy = inputs[17]
    ctx.kernel_size = inputs[18]
    ctx.require_depth = inputs[23]
    ctx.debug = inputs[24]


def _ours_render_backward(
    ctx: Any,
    grad_rendered_count: Tensor,
    grad_color: Tensor,
    grad_alpha: Tensor,
    grad_normal: Tensor,
    grad_median_depth: Tensor,
    grad_color_square: Tensor | None,
    grad_depth: Tensor | None,
    grad_depth_square: Tensor | None,
    grad_radii: Tensor,
    *grad_buffers: Tensor,
) -> tuple[Tensor | None, ...]:
    del grad_rendered_count, grad_radii, grad_buffers
    if not ctx.has_context:
        return (None,) * 25
    (
        background,
        means3D,
        colors,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
        sh,
        sg_axis,
        sg_sharpness,
        sg_color,
        viewmatrix,
        projmatrix,
        campos,
        alpha,
        normal,
        median_depth,
        radii,
        geom_buffer,
        rendered_count,
        binning_buffer,
        image_buffer,
        tile_buffer,
    ) = ctx.saved_tensors
    grad_color_square = _zero_like_if_none(grad_color_square, alpha)
    grad_depth = _zero_like_if_none(grad_depth, alpha)
    grad_depth_square = _zero_like_if_none(grad_depth_square, alpha)
    (
        _grad_means2D,
        grad_colors,
        grad_opacity,
        grad_means3D,
        grad_cov3D_precomp,
        grad_sh,
        grad_sg_axis,
        grad_sg_sharpness,
        grad_sg_color,
        grad_scales,
        grad_rotations,
    ) = ours_render_bwd_op(
        background,
        means3D,
        colors,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
        sh,
        sg_axis,
        sg_sharpness,
        sg_color,
        ctx.sh_degree,
        ctx.sg_degree,
        ctx.scale_modifier,
        viewmatrix,
        projmatrix,
        ctx.tan_fovx,
        ctx.tan_fovy,
        ctx.kernel_size,
        grad_color,
        grad_median_depth,
        grad_color_square,
        grad_depth,
        grad_depth_square,
        grad_alpha,
        grad_normal,
        alpha,
        normal,
        median_depth,
        campos,
        radii,
        geom_buffer,
        rendered_count,
        binning_buffer,
        image_buffer,
        tile_buffer,
        ctx.require_depth,
        ctx.debug,
    )
    return (
        None,
        grad_means3D,
        grad_colors,
        grad_opacity,
        grad_scales,
        grad_rotations,
        grad_cov3D_precomp,
        grad_sh,
        grad_sg_axis,
        grad_sg_sharpness,
        grad_sg_color,
        *(None,) * 14,
    )


@torch.library.custom_op("gaussian_wrapping::ours_render", mutates_args=())
def ours_render_op(
    background: Tensor,
    means3D: Tensor,
    colors: Tensor,
    opacity: Tensor,
    scales: Tensor,
    rotations: Tensor,
    cov3D_precomp: Tensor,
    sh: Tensor,
    sg_axis: Tensor,
    sg_sharpness: Tensor,
    sg_color: Tensor,
    sh_degree: int,
    sg_degree: int,
    scale_modifier: float,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    image_height: int,
    image_width: int,
    campos: Tensor,
    prefiltered: bool,
    require_depth: bool,
    debug: bool,
) -> OursRenderForwardOutput:
    """Autograd-enabled Gaussian Wrapping ``ours`` render stage."""
    return ours_render_fwd_op(
        background,
        means3D,
        colors,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
        sh,
        sg_axis,
        sg_sharpness,
        sg_color,
        sh_degree,
        sg_degree,
        scale_modifier,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        image_height,
        image_width,
        campos,
        prefiltered,
        require_depth,
        debug,
    )


ours_render_op.register_fake(_ours_render_fwd_fake)
ours_render_op.register_autograd(
    _ours_render_backward,
    setup_context=_ours_render_setup_context,
)


@torch.library.custom_op(
    "gaussian_wrapping::ours_integrate_points_fwd",
    mutates_args=(),
)
def ours_integrate_points_fwd_op(
    points3D: Tensor,
    means3D: Tensor,
    opacity: Tensor,
    scales: Tensor,
    rotations: Tensor,
    scale_modifier: float,
    cov3D_precomp: Tensor,
    view2gaussian_precomp: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    image_height: int,
    image_width: int,
    campos: Tensor,
    prefiltered: bool,
    debug: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the upstream ``ours`` point-integration CUDA stage."""
    return ours_backend().integrate_points_fwd(
        points3D,
        means3D,
        opacity,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        view2gaussian_precomp,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        image_height,
        image_width,
        campos,
        prefiltered,
        debug,
    )


@ours_integrate_points_fwd_op.register_fake
def _ours_integrate_points_fwd_fake(
    points3D: Tensor,
    means3D: Tensor,
    opacity: Tensor,
    scales: Tensor,
    rotations: Tensor,
    scale_modifier: float,
    cov3D_precomp: Tensor,
    view2gaussian_precomp: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    image_height: int,
    image_width: int,
    campos: Tensor,
    prefiltered: bool,
    debug: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    del (
        opacity,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        view2gaussian_precomp,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        image_height,
        image_width,
        campos,
        prefiltered,
        debug,
    )
    return (
        torch.empty((1,), device=means3D.device, dtype=torch.int32),
        torch.empty(
            (points3D.shape[0],), device=means3D.device, dtype=means3D.dtype
        ),
        torch.empty(
            (points3D.shape[0],), device=means3D.device, dtype=torch.bool
        ),
    )


@torch.library.custom_op("gaussian_wrapping::ours_sdf_fwd", mutates_args=())
def ours_sdf_fwd_op(
    points3D: Tensor,
    means3D: Tensor,
    opacity: Tensor,
    scales: Tensor,
    rotations: Tensor,
    scale_modifier: float,
    cov3D_precomp: Tensor,
    view2gaussian_precomp: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    image_height: int,
    image_width: int,
    campos: Tensor,
    prefiltered: bool,
    debug: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run the upstream ``ours`` single-view SDF CUDA stage."""
    return ours_backend().sdf_fwd(
        points3D,
        means3D,
        opacity,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        view2gaussian_precomp,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        image_height,
        image_width,
        campos,
        prefiltered,
        debug,
    )


@ours_sdf_fwd_op.register_fake
def _ours_sdf_fwd_fake(
    points3D: Tensor,
    means3D: Tensor,
    opacity: Tensor,
    scales: Tensor,
    rotations: Tensor,
    scale_modifier: float,
    cov3D_precomp: Tensor,
    view2gaussian_precomp: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    image_height: int,
    image_width: int,
    campos: Tensor,
    prefiltered: bool,
    debug: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del (
        opacity,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        view2gaussian_precomp,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        image_height,
        image_width,
        campos,
        prefiltered,
        debug,
    )
    return (
        torch.empty((1,), device=means3D.device, dtype=torch.int32),
        torch.empty(
            (points3D.shape[0],), device=means3D.device, dtype=means3D.dtype
        ),
        torch.empty(
            (points3D.shape[0],), device=means3D.device, dtype=means3D.dtype
        ),
        torch.empty(
            (points3D.shape[0],), device=means3D.device, dtype=torch.bool
        ),
    )


@torch.library.custom_op(
    "gaussian_wrapping::radegs_render_fwd", mutates_args=()
)
def radegs_render_fwd_op(
    background: Tensor,
    means3D: Tensor,
    colors: Tensor,
    opacity: Tensor,
    scales: Tensor,
    rotations: Tensor,
    scale_modifier: float,
    cov3D_precomp: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    image_height: int,
    image_width: int,
    sh: Tensor,
    degree: int,
    campos: Tensor,
    prefiltered: bool,
    require_coord: bool,
    require_depth: bool,
    debug: bool,
) -> RadegsRenderForwardOutput:
    """Run the upstream Gaussian Wrapping RaDe-GS render forward CUDA stage."""
    return radegs_backend().render_fwd(
        background,
        means3D,
        colors,
        opacity,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        image_height,
        image_width,
        sh,
        degree,
        campos,
        prefiltered,
        require_coord,
        require_depth,
        debug,
    )


@radegs_render_fwd_op.register_fake
def _radegs_render_fwd_fake(
    background: Tensor,
    means3D: Tensor,
    colors: Tensor,
    opacity: Tensor,
    scales: Tensor,
    rotations: Tensor,
    scale_modifier: float,
    cov3D_precomp: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    image_height: int,
    image_width: int,
    sh: Tensor,
    degree: int,
    campos: Tensor,
    prefiltered: bool,
    require_coord: bool,
    require_depth: bool,
    debug: bool,
) -> RadegsRenderForwardOutput:
    del (
        background,
        colors,
        opacity,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        sh,
        degree,
        campos,
        prefiltered,
        require_coord,
        require_depth,
        debug,
    )
    num_splats = int(means3D.shape[0])
    device = means3D.device
    dtype = means3D.dtype
    return (
        torch.empty((1,), device=device, dtype=torch.int32),
        torch.empty((3, image_height, image_width), device=device, dtype=dtype),
        torch.empty((3, image_height, image_width), device=device, dtype=dtype),
        torch.empty((3, image_height, image_width), device=device, dtype=dtype),
        torch.empty((1, image_height, image_width), device=device, dtype=dtype),
        torch.empty((3, image_height, image_width), device=device, dtype=dtype),
        torch.empty((1, image_height, image_width), device=device, dtype=dtype),
        torch.empty((1, image_height, image_width), device=device, dtype=dtype),
        torch.empty((1, image_height, image_width), device=device, dtype=dtype),
        torch.empty((1, image_height, image_width), device=device, dtype=dtype),
        torch.empty((1, image_height, image_width), device=device, dtype=dtype),
        torch.empty((num_splats,), device=device, dtype=torch.int32),
        torch.empty((0,), device=device, dtype=torch.uint8),
        torch.empty((0,), device=device, dtype=torch.uint8),
        torch.empty((0,), device=device, dtype=torch.uint8),
    )


@torch.library.custom_op(
    "gaussian_wrapping::radegs_render_bwd", mutates_args=()
)
def radegs_render_bwd_op(
    background: Tensor,
    means3D: Tensor,
    radii: Tensor,
    colors: Tensor,
    scales: Tensor,
    rotations: Tensor,
    scale_modifier: float,
    cov3D_precomp: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    grad_color: Tensor,
    grad_coord: Tensor,
    grad_median_coord: Tensor,
    grad_depth: Tensor,
    grad_median_depth: Tensor,
    grad_color_square: Tensor,
    grad_depth_sum: Tensor,
    grad_depth_square: Tensor,
    grad_alpha: Tensor,
    grad_normal: Tensor,
    normal: Tensor,
    sh: Tensor,
    degree: int,
    campos: Tensor,
    geom_buffer: Tensor,
    rendered_count: Tensor,
    binning_buffer: Tensor,
    image_buffer: Tensor,
    alpha: Tensor,
    require_coord: bool,
    require_depth: bool,
    debug: bool,
) -> RadegsRenderBackwardOutput:
    """Run the upstream Gaussian Wrapping RaDe-GS render backward CUDA stage."""
    return radegs_backend().render_bwd(
        background,
        means3D,
        radii,
        colors,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        grad_color,
        grad_coord,
        grad_median_coord,
        grad_depth,
        grad_median_depth,
        grad_color_square,
        grad_depth_sum,
        grad_depth_square,
        grad_alpha,
        grad_normal,
        normal,
        sh,
        degree,
        campos,
        geom_buffer,
        rendered_count,
        binning_buffer,
        image_buffer,
        alpha,
        require_coord,
        require_depth,
        debug,
    )


@radegs_render_bwd_op.register_fake
def _radegs_render_bwd_fake(
    background: Tensor,
    means3D: Tensor,
    radii: Tensor,
    colors: Tensor,
    scales: Tensor,
    rotations: Tensor,
    scale_modifier: float,
    cov3D_precomp: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    grad_color: Tensor,
    grad_coord: Tensor,
    grad_median_coord: Tensor,
    grad_depth: Tensor,
    grad_median_depth: Tensor,
    grad_color_square: Tensor,
    grad_depth_sum: Tensor,
    grad_depth_square: Tensor,
    grad_alpha: Tensor,
    grad_normal: Tensor,
    normal: Tensor,
    sh: Tensor,
    degree: int,
    campos: Tensor,
    geom_buffer: Tensor,
    rendered_count: Tensor,
    binning_buffer: Tensor,
    image_buffer: Tensor,
    alpha: Tensor,
    require_coord: bool,
    require_depth: bool,
    debug: bool,
) -> RadegsRenderBackwardOutput:
    del (
        background,
        radii,
        colors,
        scale_modifier,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        grad_color,
        grad_coord,
        grad_median_coord,
        grad_depth,
        grad_median_depth,
        grad_color_square,
        grad_depth_sum,
        grad_depth_square,
        grad_alpha,
        grad_normal,
        normal,
        degree,
        campos,
        geom_buffer,
        rendered_count,
        binning_buffer,
        image_buffer,
        alpha,
        require_coord,
        require_depth,
        debug,
    )
    return (
        torch.empty(
            (means3D.shape[0], 3), device=means3D.device, dtype=means3D.dtype
        ),
        torch.empty(
            (means3D.shape[0], 3), device=means3D.device, dtype=means3D.dtype
        ),
        torch.empty(
            (means3D.shape[0], 1), device=means3D.device, dtype=means3D.dtype
        ),
        torch.empty_like(means3D),
        torch.empty_like(cov3D_precomp),
        torch.empty_like(sh),
        torch.empty_like(scales),
        torch.empty_like(rotations),
    )


def _radegs_render_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: RadegsRenderForwardOutput,
) -> None:
    if not _requires_grad(
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[7],
        inputs[15],
    ):
        ctx.has_context = False
        return
    ctx.has_context = True
    ctx.save_for_backward(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[4],
        inputs[5],
        inputs[7],
        inputs[8],
        inputs[9],
        inputs[15],
        inputs[17],
        output[11],
        output[5],
        output[12],
        output[0],
        output[13],
        output[14],
        output[4],
    )
    ctx.scale_modifier = inputs[6]
    ctx.tan_fovx = inputs[10]
    ctx.tan_fovy = inputs[11]
    ctx.kernel_size = inputs[12]
    ctx.degree = inputs[16]
    ctx.require_coord = inputs[19]
    ctx.require_depth = inputs[20]
    ctx.debug = inputs[21]


def _radegs_render_backward(
    ctx: Any,
    grad_rendered_count: Tensor,
    grad_color: Tensor,
    grad_coord: Tensor,
    grad_median_coord: Tensor,
    grad_alpha: Tensor,
    grad_normal: Tensor,
    grad_depth: Tensor,
    grad_median_depth: Tensor,
    grad_color_square: Tensor | None,
    grad_depth_sum: Tensor | None,
    grad_depth_square: Tensor | None,
    grad_radii: Tensor,
    *grad_buffers: Tensor,
) -> tuple[Tensor | None, ...]:
    del grad_rendered_count, grad_radii, grad_buffers
    if not ctx.has_context:
        return (None,) * 22
    (
        background,
        means3D,
        colors,
        scales,
        rotations,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        sh,
        campos,
        radii,
        normal,
        geom_buffer,
        rendered_count,
        binning_buffer,
        image_buffer,
        alpha,
    ) = ctx.saved_tensors
    grad_color_square = _zero_like_if_none(grad_color_square, alpha)
    grad_depth_sum = _zero_like_if_none(grad_depth_sum, alpha)
    grad_depth_square = _zero_like_if_none(grad_depth_square, alpha)
    (
        _grad_means2D,
        grad_colors,
        grad_opacity,
        grad_means3D,
        grad_cov3D_precomp,
        grad_sh,
        grad_scales,
        grad_rotations,
    ) = radegs_render_bwd_op(
        background,
        means3D,
        radii,
        colors,
        scales,
        rotations,
        ctx.scale_modifier,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        ctx.tan_fovx,
        ctx.tan_fovy,
        ctx.kernel_size,
        grad_color,
        grad_coord,
        grad_median_coord,
        grad_depth,
        grad_median_depth,
        grad_color_square,
        grad_depth_sum,
        grad_depth_square,
        grad_alpha,
        grad_normal,
        normal,
        sh,
        ctx.degree,
        campos,
        geom_buffer,
        rendered_count,
        binning_buffer,
        image_buffer,
        alpha,
        ctx.require_coord,
        ctx.require_depth,
        ctx.debug,
    )
    return (
        None,
        grad_means3D,
        grad_colors,
        grad_opacity,
        grad_scales,
        grad_rotations,
        None,
        grad_cov3D_precomp,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        grad_sh,
        *(None,) * 6,
    )


@torch.library.custom_op("gaussian_wrapping::radegs_render", mutates_args=())
def radegs_render_op(
    background: Tensor,
    means3D: Tensor,
    colors: Tensor,
    opacity: Tensor,
    scales: Tensor,
    rotations: Tensor,
    scale_modifier: float,
    cov3D_precomp: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    image_height: int,
    image_width: int,
    sh: Tensor,
    degree: int,
    campos: Tensor,
    prefiltered: bool,
    require_coord: bool,
    require_depth: bool,
    debug: bool,
) -> RadegsRenderForwardOutput:
    """Autograd-enabled Gaussian Wrapping RaDe-GS render stage."""
    return radegs_render_fwd_op(
        background,
        means3D,
        colors,
        opacity,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        image_height,
        image_width,
        sh,
        degree,
        campos,
        prefiltered,
        require_coord,
        require_depth,
        debug,
    )


radegs_render_op.register_fake(_radegs_render_fwd_fake)
radegs_render_op.register_autograd(
    _radegs_render_backward,
    setup_context=_radegs_render_setup_context,
)


@torch.library.custom_op(
    "gaussian_wrapping::radegs_integrate_points_fwd",
    mutates_args=(),
)
def radegs_integrate_points_fwd_op(
    background: Tensor,
    points3D: Tensor,
    means3D: Tensor,
    colors: Tensor,
    opacity: Tensor,
    scales: Tensor,
    rotations: Tensor,
    scale_modifier: float,
    cov3D_precomp: Tensor,
    view2gaussian_precomp: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    subpixel_offset: Tensor,
    image_height: int,
    image_width: int,
    sh: Tensor,
    degree: int,
    campos: Tensor,
    prefiltered: bool,
    debug: bool,
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
    """Run the upstream RaDe-GS point-integration CUDA stage."""
    return radegs_backend().integrate_points_fwd(
        background,
        points3D,
        means3D,
        colors,
        opacity,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        view2gaussian_precomp,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        subpixel_offset,
        image_height,
        image_width,
        sh,
        degree,
        campos,
        prefiltered,
        debug,
    )


@radegs_integrate_points_fwd_op.register_fake
def _radegs_integrate_points_fwd_fake(
    background: Tensor,
    points3D: Tensor,
    means3D: Tensor,
    colors: Tensor,
    opacity: Tensor,
    scales: Tensor,
    rotations: Tensor,
    scale_modifier: float,
    cov3D_precomp: Tensor,
    view2gaussian_precomp: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    tan_fovx: float,
    tan_fovy: float,
    kernel_size: float,
    subpixel_offset: Tensor,
    image_height: int,
    image_width: int,
    sh: Tensor,
    degree: int,
    campos: Tensor,
    prefiltered: bool,
    debug: bool,
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
    del (
        background,
        colors,
        opacity,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        view2gaussian_precomp,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        subpixel_offset,
        sh,
        degree,
        campos,
        prefiltered,
        debug,
    )
    num_splats = int(means3D.shape[0])
    device = means3D.device
    dtype = means3D.dtype
    return (
        torch.empty((1,), device=device, dtype=torch.int32),
        torch.empty((9, image_height, image_width), device=device, dtype=dtype),
        torch.empty((points3D.shape[0],), device=device, dtype=dtype),
        torch.empty((points3D.shape[0], 3), device=device, dtype=dtype),
        torch.empty((points3D.shape[0], 2), device=device, dtype=dtype),
        torch.empty((points3D.shape[0],), device=device, dtype=dtype),
        torch.empty((num_splats,), device=device, dtype=torch.int32),
        torch.empty((0,), device=device, dtype=torch.uint8),
        torch.empty((0,), device=device, dtype=torch.uint8),
        torch.empty((0,), device=device, dtype=torch.uint8),
    )
