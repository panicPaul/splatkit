"""Rasterization custom ops for the PowerFoam Warp runtime."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

import torch
from torch import Tensor

from ember_native_powerfoam.powerfoam.native.warp.camera import TorchCamera
from ember_native_powerfoam.powerfoam.native.warp.rasterize import Rasterizer

_RasterizeOutput = tuple[
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
    Tensor,
    Tensor,
    Tensor,
]


class _CapturedContext:
    def __init__(self) -> None:
        self.saved_tensors: tuple[Tensor | None, ...] = ()

    def save_for_backward(self, *tensors: Tensor | None) -> None:
        self.saved_tensors = tensors


def _empty_tensor_like_device(reference: Tensor) -> Tensor:
    return torch.empty((0,), dtype=reference.dtype, device=reference.device)


def _detach_saved_tensor(tensor: Tensor | None) -> Tensor | None:
    return None if tensor is None else tensor.detach()


def _contiguous_grad_or_zeros(
    grad: Tensor | None,
    reference: Tensor,
) -> Tensor:
    """Return a Warp-compatible gradient tensor."""
    return torch.zeros_like(reference) if grad is None else grad.contiguous()


def _optional_contiguous_grad(grad: Tensor | None) -> Tensor | None:
    """Return a contiguous optional gradient for native Warp launches."""
    return None if grad is None else grad.contiguous()


def _rasterizer_args(
    *,
    render_objective: str,
    num_texel_sites: int,
    sv_dof: int,
    disable_coop_prim_load: bool,
    disable_coop_adj_load: bool,
    is_pinhole: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        render_objective=render_objective,
        num_texel_sites=int(num_texel_sites),
        sv_dof=int(sv_dof),
        disable_coop_prim_load=bool(disable_coop_prim_load),
        disable_coop_adj_load=bool(disable_coop_adj_load),
        is_pinhole=bool(is_pinhole),
    )


def _camera(
    *,
    eye: Tensor,
    right: Tensor,
    up: Tensor,
    ray_maps: Tensor,
    width: int,
    height: int,
) -> TorchCamera:
    return TorchCamera(
        eye=eye,
        right=right,
        up=up,
        width=int(width),
        height=int(height),
        ray_maps=ray_maps,
    )


def _rasterizer(
    *,
    points: Tensor,
    render_objective: str,
    num_texel_sites: int,
    sv_dof: int,
    disable_coop_prim_load: bool,
    disable_coop_adj_load: bool,
    is_pinhole: bool,
    attr_dtype: str,
) -> Rasterizer:
    return Rasterizer(
        _rasterizer_args(
            render_objective=render_objective,
            num_texel_sites=num_texel_sites,
            sv_dof=sv_dof,
            disable_coop_prim_load=disable_coop_prim_load,
            disable_coop_adj_load=disable_coop_adj_load,
            is_pinhole=is_pinhole,
        ),
        points.device,
        attr_dtype,
    )


@torch.library.custom_op("powerfoam::rasterize_fwd", mutates_args=())
def rasterize_fwd_op(
    camera_eye: Tensor,
    camera_right: Tensor,
    camera_up: Tensor,
    camera_ray_maps: Tensor,
    camera_width: int,
    camera_height: int,
    depth_quantiles: Tensor,
    has_depth_quantiles: bool,
    points: Tensor,
    radii: Tensor,
    density: Tensor,
    normals: Tensor,
    texel_sites: Tensor,
    texel_rgb: Tensor,
    texel_height: Tensor,
    adjacency: Tensor,
    adjacency_offsets: Tensor,
    ray_gt: Tensor,
    has_ray_gt: bool,
    return_point_err: bool,
    render_objective: str,
    num_texel_sites: int,
    sv_dof: int,
    disable_coop_prim_load: bool,
    disable_coop_adj_load: bool,
    is_pinhole: bool,
    attr_dtype: str,
) -> _RasterizeOutput:
    """Rasterize PowerFoam primitives with vendored Warp kernels."""
    rasterizer = _rasterizer(
        points=points,
        render_objective=render_objective,
        num_texel_sites=num_texel_sites,
        sv_dof=sv_dof,
        disable_coop_prim_load=disable_coop_prim_load,
        disable_coop_adj_load=disable_coop_adj_load,
        is_pinhole=is_pinhole,
        attr_dtype=attr_dtype,
    )
    camera = _camera(
        eye=camera_eye,
        right=camera_right,
        up=camera_up,
        ray_maps=camera_ray_maps,
        width=camera_width,
        height=camera_height,
    )
    capture = _CapturedContext()
    (
        color,
        opacity,
        normal_distance,
        normal,
        quantile_depths,
        err,
        contrib,
        point_err,
        prim_visible_mask,
    ) = rasterizer._forward(
        capture,
        camera,
        depth_quantiles if has_depth_quantiles else None,
        points,
        radii,
        density,
        normals,
        texel_sites,
        texel_rgb,
        texel_height,
        adjacency,
        adjacency_offsets,
        ray_gt if has_ray_gt else None,
        return_point_err,
    )
    (
        saved_depth_quantiles,
        all_spheres,
        all_nsigmas,
        all_texel_sites,
        all_texel_rgbh,
        saved_adjacency,
        saved_adjacency_offsets,
        adjacency_diff,
        tile_prim_indices,
        offsets,
        tile_early_stop_counter,
        saved_ray_gt,
        saved_color,
        log_t,
    ) = capture.saved_tensors
    empty = _empty_tensor_like_device(points)
    return (
        color,
        opacity,
        normal_distance,
        normal,
        quantile_depths if quantile_depths is not None else empty.clone(),
        err if err is not None else empty.clone(),
        contrib,
        point_err if point_err is not None else empty.clone(),
        prim_visible_mask,
        (
            saved_depth_quantiles.clone()
            if saved_depth_quantiles is not None
            else empty.clone()
        ),
        all_spheres,
        all_nsigmas,
        all_texel_sites.clone(),
        all_texel_rgbh,
        saved_adjacency.clone(),
        saved_adjacency_offsets.clone(),
        adjacency_diff,
        tile_prim_indices,
        offsets,
        tile_early_stop_counter,
        saved_ray_gt.clone() if saved_ray_gt is not None else empty.clone(),
        saved_color.clone(),
        log_t,
    )


@rasterize_fwd_op.register_fake
def _rasterize_fwd_fake(
    camera_eye: Tensor,
    camera_right: Tensor,
    camera_up: Tensor,
    camera_ray_maps: Tensor,
    camera_width: int,
    camera_height: int,
    depth_quantiles: Tensor,
    has_depth_quantiles: bool,
    points: Tensor,
    radii: Tensor,
    density: Tensor,
    normals: Tensor,
    texel_sites: Tensor,
    texel_rgb: Tensor,
    texel_height: Tensor,
    adjacency: Tensor,
    adjacency_offsets: Tensor,
    ray_gt: Tensor,
    has_ray_gt: bool,
    return_point_err: bool,
    render_objective: str,
    num_texel_sites: int,
    sv_dof: int,
    disable_coop_prim_load: bool,
    disable_coop_adj_load: bool,
    is_pinhole: bool,
    attr_dtype: str,
) -> _RasterizeOutput:
    del (
        camera_eye,
        camera_right,
        camera_up,
        density,
        normals,
        texel_rgb,
        texel_height,
        has_ray_gt,
        render_objective,
        num_texel_sites,
        sv_dof,
        disable_coop_prim_load,
        disable_coop_adj_load,
        is_pinhole,
        attr_dtype,
    )
    height = int(camera_height)
    width = int(camera_width)
    empty = _empty_tensor_like_device(points)
    maybe_depth = (
        torch.empty_like(depth_quantiles)
        if has_depth_quantiles
        else empty
    )
    maybe_err = (
        torch.empty((height, width), dtype=points.dtype, device=points.device)
        if return_point_err
        else empty
    )
    return (
        torch.empty((height, width, 3), dtype=points.dtype, device=points.device),
        torch.empty((height, width), dtype=points.dtype, device=points.device),
        torch.empty((height, width), dtype=points.dtype, device=points.device),
        torch.empty((height, width, 3), dtype=points.dtype, device=points.device),
        maybe_depth,
        maybe_err,
        torch.empty((points.shape[0],), dtype=points.dtype, device=points.device),
        maybe_err,
        torch.empty((points.shape[0],), dtype=torch.bool, device=points.device),
        maybe_depth,
        torch.empty((points.shape[0], 4), dtype=points.dtype, device=points.device),
        torch.empty((points.shape[0], 4), dtype=points.dtype, device=points.device),
        torch.empty_like(texel_sites),
        torch.empty((*texel_sites.shape[:-1], 4), dtype=points.dtype, device=points.device),
        adjacency,
        adjacency_offsets,
        torch.empty((adjacency.shape[0], 4), dtype=torch.float16, device=points.device),
        torch.empty((0,), dtype=torch.int32, device=points.device),
        torch.empty((0,), dtype=torch.long, device=points.device),
        torch.empty((0,), dtype=torch.int32, device=points.device),
        ray_gt if ray_gt.numel() else empty,
        torch.empty((height, width, 3), dtype=points.dtype, device=points.device),
        torch.empty((height, width), dtype=points.dtype, device=points.device),
    )


def _rasterize_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: _RasterizeOutput,
) -> None:
    ctx.save_for_backward(*output[9:])
    ctx.has_depth_quantiles = inputs[7]
    ctx.has_ray_gt = inputs[18]
    ctx.return_point_err = inputs[19]
    ctx.render_objective = inputs[20]
    ctx.num_texel_sites = inputs[21]
    ctx.sv_dof = inputs[22]
    ctx.disable_coop_prim_load = inputs[23]
    ctx.disable_coop_adj_load = inputs[24]
    ctx.is_pinhole = inputs[25]
    ctx.attr_dtype = inputs[26]
    ctx.camera = _camera(
        eye=inputs[0],
        right=inputs[1],
        up=inputs[2],
        ray_maps=inputs[3],
        width=inputs[4],
        height=inputs[5],
    )


def _rasterize_backward(
    ctx: Any,
    grad_color: Tensor | None,
    grad_opacity: Tensor | None,
    grad_normal_distance: Tensor | None,
    grad_normal: Tensor | None,
    grad_quantile_depths: Tensor | None,
    grad_err: Tensor | None,
    grad_contrib: Tensor | None,
    grad_point_err: Tensor | None,
    grad_prim_visible_mask: Tensor | None,
    *grad_aux: Tensor | None,
) -> tuple[Tensor | None, ...]:
    del grad_aux
    (
        saved_depth_quantiles,
        all_spheres,
        all_nsigmas,
        all_texel_sites,
        all_texel_rgbh,
        saved_adjacency,
        saved_adjacency_offsets,
        adjacency_diff,
        tile_prim_indices,
        offsets,
        tile_early_stop_counter,
        saved_ray_gt,
        saved_color,
        log_t,
    ) = ctx.saved_tensors
    capture = _CapturedContext()
    capture.camera = ctx.camera
    # The vendored Warp launches consume the forward buffers as raw CUDA arrays.
    # Autograd bookkeeping stays in this wrapper, so the replayed native stage
    # receives detached intermediates that match upstream tensor values.
    capture.saved_tensors = (
        _detach_saved_tensor(saved_depth_quantiles)
        if ctx.has_depth_quantiles
        else None,
        all_spheres.detach(),
        all_nsigmas.detach(),
        all_texel_sites.detach(),
        all_texel_rgbh.detach(),
        saved_adjacency.detach(),
        saved_adjacency_offsets.detach(),
        adjacency_diff.detach(),
        tile_prim_indices.detach(),
        offsets.detach(),
        tile_early_stop_counter.detach(),
        _detach_saved_tensor(saved_ray_gt) if ctx.has_ray_gt else None,
        saved_color.detach(),
        log_t.detach(),
    )
    rasterizer = _rasterizer(
        points=all_spheres,
        render_objective=ctx.render_objective,
        num_texel_sites=ctx.num_texel_sites,
        sv_dof=ctx.sv_dof,
        disable_coop_prim_load=ctx.disable_coop_prim_load,
        disable_coop_adj_load=ctx.disable_coop_adj_load,
        is_pinhole=ctx.is_pinhole,
        attr_dtype=ctx.attr_dtype,
    )
    capture.rasterizer = rasterizer
    grad_quantile_depths_arg = (
        _optional_contiguous_grad(grad_quantile_depths)
        if ctx.has_depth_quantiles
        else None
    )
    grad_err_arg = _optional_contiguous_grad(grad_err) if ctx.has_ray_gt else None
    grad_contrib_arg = (
        torch.zeros_like(all_spheres[:, 0])
        if grad_contrib is None
        else grad_contrib.contiguous()
    )
    (
        _grad_self,
        _grad_camera,
        _grad_depth_quantiles,
        grad_points,
        grad_radii,
        grad_density,
        grad_normals,
        grad_texel_sites,
        grad_texel_rgb,
        grad_texel_height,
        _grad_adjacency,
        _grad_adjacency_offsets,
        _grad_ray_gt,
        _grad_return_point_err,
    ) = rasterizer._backward(
        capture,
        _contiguous_grad_or_zeros(grad_color, saved_color),
        _contiguous_grad_or_zeros(grad_opacity, log_t),
        _contiguous_grad_or_zeros(grad_normal_distance, log_t),
        _contiguous_grad_or_zeros(grad_normal, saved_color),
        grad_quantile_depths_arg,
        grad_err_arg,
        grad_contrib_arg,
        _optional_contiguous_grad(grad_point_err),
        _optional_contiguous_grad(grad_prim_visible_mask),
    )
    return (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        grad_points,
        grad_radii,
        grad_density,
        grad_normals,
        grad_texel_sites,
        grad_texel_rgb,
        grad_texel_height,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


rasterize_fwd_op.register_autograd(
    _rasterize_backward,
    setup_context=_rasterize_setup_context,
)


def rasterize_powerfoam(
    camera: TorchCamera,
    depth_quantiles: Tensor | None,
    points: Tensor,
    radii: Tensor,
    density: Tensor,
    normals: Tensor,
    texel_sites: Tensor,
    texel_rgb: Tensor,
    texel_height: Tensor,
    adjacency: Tensor,
    adjacency_offsets: Tensor,
    ray_gt: Tensor | None,
    return_point_err: bool,
    *,
    render_objective: Literal["volume", "surface"],
    num_texel_sites: int,
    sv_dof: int,
    disable_coop_prim_load: bool,
    disable_coop_adj_load: bool,
    is_pinhole: bool,
    attr_dtype: Literal["float", "half"],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor, Tensor | None, Tensor]:
    """Return PowerFoam raster outputs from the custom op stage."""
    empty = _empty_tensor_like_device(points)
    outputs = rasterize_fwd_op(
        camera.eye,
        camera.right,
        camera.up,
        camera.ray_maps if camera.ray_maps is not None else empty,
        camera.width,
        camera.height,
        depth_quantiles if depth_quantiles is not None else empty,
        depth_quantiles is not None,
        points,
        radii,
        density,
        normals,
        texel_sites,
        texel_rgb,
        texel_height,
        adjacency,
        adjacency_offsets,
        ray_gt if ray_gt is not None else empty,
        ray_gt is not None,
        return_point_err,
        render_objective,
        num_texel_sites,
        sv_dof,
        disable_coop_prim_load,
        disable_coop_adj_load,
        is_pinhole,
        attr_dtype,
    )
    return (
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4] if depth_quantiles is not None else None,
        outputs[5] if ray_gt is not None else None,
        outputs[6],
        outputs[7] if return_point_err else None,
        outputs[8],
    )


__all__ = [
    "rasterize_fwd_op",
    "rasterize_powerfoam",
]
