"""Spherical-harmonic evaluation custom ops for the SVRaster runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from splatkit_native_svraster.core.runtime.ops._common import backend


@torch.library.custom_op("svraster::sh_eval", mutates_args=())
def sh_eval_op(
    active_sh_degree: int,
    indices: Tensor,
    vox_centers: Tensor,
    cam_pos: Tensor,
    sh0: Tensor,
    shs: Tensor,
) -> Tensor:
    """Evaluate SVRaster spherical harmonics into RGB colors."""
    return backend().sh_compute(
        active_sh_degree,
        indices,
        vox_centers,
        cam_pos,
        sh0,
        shs,
    )


@sh_eval_op.register_fake
def _sh_eval_fake(
    active_sh_degree: int,
    indices: Tensor,
    vox_centers: Tensor,
    cam_pos: Tensor,
    sh0: Tensor,
    shs: Tensor,
) -> Tensor:
    del active_sh_degree, indices, cam_pos, shs
    return torch.empty(
        (int(vox_centers.shape[0]), 3),
        device=sh0.device,
        dtype=sh0.dtype,
    )


def _sh_eval_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: Tensor,
) -> None:
    ctx.active_sh_degree = inputs[0]
    ctx.max_coefficients = 1 + int(inputs[5].shape[1])
    ctx.save_for_backward(inputs[1], inputs[2], inputs[3], output)


def _sh_eval_backward(
    ctx: Any,
    grad_rgbs: Tensor,
) -> tuple[Tensor | None, ...]:
    indices, vox_centers, cam_pos, rgbs = ctx.saved_tensors
    grad_sh0, grad_shs = backend().sh_compute_bw(
        ctx.active_sh_degree,
        ctx.max_coefficients,
        indices,
        vox_centers,
        cam_pos,
        rgbs,
        grad_rgbs,
    )
    return (None, None, None, None, grad_sh0, grad_shs)


sh_eval_op.register_autograd(
    _sh_eval_backward,
    setup_context=_sh_eval_setup_context,
)
