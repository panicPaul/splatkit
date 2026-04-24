"""Autograd helpers for SVRaster trilinear gather kernels."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from splatkit_native_svraster.core.runtime._extension import load_extension


class _GatherGeoParams(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        vox_key: Tensor,
        care_idx: Tensor,
        grid_pts: Tensor,
    ) -> Tensor:
        geo_params = load_extension().gather_triinterp_geo_params(
            vox_key,
            care_idx,
            grid_pts,
        )
        ctx.num_grid_pts = int(grid_pts.numel())
        ctx.save_for_backward(vox_key, care_idx)
        return geo_params

    @staticmethod
    def backward(
        ctx: Any,
        grad_geo_params: Tensor,
    ) -> tuple[None, None, Tensor]:
        vox_key, care_idx = ctx.saved_tensors
        grad_grid_pts = load_extension().gather_triinterp_geo_params_bw(
            vox_key,
            care_idx,
            ctx.num_grid_pts,
            grad_geo_params,
        )
        return None, None, grad_grid_pts


class _GatherFeatParams(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        vox_key: Tensor,
        care_idx: Tensor,
        grid_pts: Tensor,
    ) -> Tensor:
        feat_params = load_extension().gather_triinterp_feat_params(
            vox_key,
            care_idx,
            grid_pts,
        )
        ctx.num_grid_pts = int(grid_pts.shape[0])
        ctx.save_for_backward(vox_key, care_idx)
        return feat_params

    @staticmethod
    def backward(
        ctx: Any,
        grad_feat_params: Tensor,
    ) -> tuple[None, None, Tensor]:
        vox_key, care_idx = ctx.saved_tensors
        grad_grid_pts = load_extension().gather_triinterp_feat_params_bw(
            vox_key,
            care_idx,
            ctx.num_grid_pts,
            grad_feat_params,
        )
        return None, None, grad_grid_pts


def gather_triinterp_geo_params(
    vox_key: Tensor,
    care_idx: Tensor,
    grid_pts: Tensor,
) -> Tensor:
    """Gather trilinear geometry corner values for the requested voxels."""
    return _GatherGeoParams.apply(vox_key, care_idx, grid_pts)


def gather_triinterp_feat_params(
    vox_key: Tensor,
    care_idx: Tensor,
    grid_pts: Tensor,
) -> Tensor:
    """Gather trilinear feature corner values for the requested voxels."""
    return _GatherFeatParams.apply(vox_key, care_idx, grid_pts)
