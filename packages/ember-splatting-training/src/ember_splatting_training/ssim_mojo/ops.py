"""Torch custom-op registration for Mojo SSIM."""

from __future__ import annotations

from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ember_splatting_training.ssim_mojo.library import run_inplace_custom_op


def validate_nchw(
    prediction: Tensor,
    target: Tensor,
) -> None:
    """Validate the NCHW float32 CUDA image contract for SSIM Mojo."""
    if prediction.shape != target.shape:
        raise ValueError(
            "SSIM expects prediction and target to share NCHW shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    if prediction.ndim != 4:
        raise ValueError(
            f"SSIM expects NCHW rank-4 tensors, got rank {prediction.ndim}."
        )
    if prediction.dtype != torch.float32 or target.dtype != torch.float32:
        raise TypeError("SSIM Mojo currently supports float32 tensors only.")
    if prediction.device.type != "cuda":
        raise ValueError("SSIM Mojo currently requires CUDA tensors.")


@torch.library.custom_op("ssim_mojo::fwd", mutates_args=())
def ssim_mojo_fwd_op(
    prediction: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Low-level Mojo SSIM forward op returning map and derivative caches."""
    validate_nchw(prediction, target)
    ssim_map = torch.empty_like(prediction)
    dm_dmu1 = torch.empty_like(prediction)
    dm_dsigma1_sq = torch.empty_like(prediction)
    dm_dsigma12 = torch.empty_like(prediction)
    run_inplace_custom_op(
        "ssim_fwd_inplace",
        [ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12],
        [
            prediction.detach().contiguous(),
            target.detach().contiguous(),
        ],
    )
    return ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12


@ssim_mojo_fwd_op.register_fake
def ssim_mojo_fwd_fake(
    prediction: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fake implementation for the raw SSIM forward op."""
    del target
    return (
        torch.empty_like(prediction),
        torch.empty_like(prediction),
        torch.empty_like(prediction),
        torch.empty_like(prediction),
    )


@torch.library.custom_op("ssim_mojo::bwd", mutates_args=())
def ssim_mojo_bwd_op(
    prediction: Tensor,
    target: Tensor,
    grad_map: Tensor,
    dm_dmu1: Tensor,
    dm_dsigma1_sq: Tensor,
    dm_dsigma12: Tensor,
) -> Tensor:
    """Low-level Mojo SSIM backward op returning dL/dprediction."""
    validate_nchw(prediction, target)
    grad_prediction = torch.empty_like(prediction)
    run_inplace_custom_op(
        "ssim_bwd_inplace",
        [grad_prediction],
        [
            prediction.detach().contiguous(),
            target.detach().contiguous(),
            grad_map.detach().contiguous(),
            dm_dmu1.detach().contiguous(),
            dm_dsigma1_sq.detach().contiguous(),
            dm_dsigma12.detach().contiguous(),
        ],
    )
    return grad_prediction


@ssim_mojo_bwd_op.register_fake
def ssim_mojo_bwd_fake(
    prediction: Tensor,
    target: Tensor,
    grad_map: Tensor,
    dm_dmu1: Tensor,
    dm_dsigma1_sq: Tensor,
    dm_dsigma12: Tensor,
) -> Tensor:
    """Fake implementation for the raw SSIM backward op."""
    del target, grad_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12
    return torch.empty_like(prediction)


@torch.library.custom_op("ssim_mojo::bwd_mean", mutates_args=())
def ssim_mojo_bwd_mean_op(
    prediction: Tensor,
    target: Tensor,
    grad_scale: Tensor,
    dm_dmu1: Tensor,
    dm_dsigma1_sq: Tensor,
    dm_dsigma12: Tensor,
) -> Tensor:
    """Low-level Mojo SSIM backward op for a mean-reduced SSIM map."""
    validate_nchw(prediction, target)
    grad_prediction = torch.empty_like(prediction)
    run_inplace_custom_op(
        "ssim_bwd_mean_inplace",
        [grad_prediction],
        [
            prediction.detach().contiguous(),
            target.detach().contiguous(),
            grad_scale.detach().contiguous(),
            dm_dmu1.detach().contiguous(),
            dm_dsigma1_sq.detach().contiguous(),
            dm_dsigma12.detach().contiguous(),
        ],
    )
    return grad_prediction


@ssim_mojo_bwd_mean_op.register_fake
def ssim_mojo_bwd_mean_fake(
    prediction: Tensor,
    target: Tensor,
    grad_scale: Tensor,
    dm_dmu1: Tensor,
    dm_dsigma1_sq: Tensor,
    dm_dsigma12: Tensor,
) -> Tensor:
    """Fake implementation for the mean-reduced SSIM backward op."""
    del target, grad_scale, dm_dmu1, dm_dsigma1_sq, dm_dsigma12
    return torch.empty_like(prediction)


def ssim_mojo_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    """Save tensors needed by PyTorch autograd for SSIM Mojo."""
    _, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = output
    ctx.save_for_backward(
        inputs[0],
        inputs[1],
        dm_dmu1,
        dm_dsigma1_sq,
        dm_dsigma12,
    )


def ssim_mojo_backward(
    ctx: Any,
    grad_map: Tensor,
    grad_dm_dmu1: Tensor,
    grad_dm_dsigma1_sq: Tensor,
    grad_dm_dsigma12: Tensor,
) -> tuple[Tensor | None, ...]:
    """Run the registered backward op for PyTorch autograd."""
    del grad_dm_dmu1, grad_dm_dsigma1_sq, grad_dm_dsigma12
    prediction, target, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
    grad_prediction = ssim_mojo_bwd_op(
        prediction,
        target,
        grad_map,
        dm_dmu1,
        dm_dsigma1_sq,
        dm_dsigma12,
    )
    return grad_prediction, None


ssim_mojo_fwd_op.register_autograd(
    ssim_mojo_backward,
    setup_context=ssim_mojo_setup_context,
)


@torch.library.custom_op("ssim_mojo::mean_fwd", mutates_args=())
def ssim_mojo_mean_fwd_op(
    prediction: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Forward op for mean SSIM plus backward derivative caches."""
    ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ssim_mojo_fwd_op(
        prediction,
        target,
    )
    return ssim_map.mean(), dm_dmu1, dm_dsigma1_sq, dm_dsigma12


@ssim_mojo_mean_fwd_op.register_fake
def ssim_mojo_mean_fwd_fake(
    prediction: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fake implementation for the mean SSIM forward op."""
    del target
    return (
        prediction.new_empty(()),
        torch.empty_like(prediction),
        torch.empty_like(prediction),
        torch.empty_like(prediction),
    )


def ssim_mojo_mean_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    """Save tensors needed by PyTorch autograd for mean SSIM Mojo."""
    _, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = output
    ctx.numel = inputs[0].numel()
    ctx.save_for_backward(
        inputs[0],
        inputs[1],
        dm_dmu1,
        dm_dsigma1_sq,
        dm_dsigma12,
    )


def ssim_mojo_mean_backward(
    ctx: Any,
    grad_score: Tensor,
    grad_dm_dmu1: Tensor,
    grad_dm_dsigma1_sq: Tensor,
    grad_dm_dsigma12: Tensor,
) -> tuple[Tensor | None, ...]:
    """Run mean-specialized SSIM backward for PyTorch autograd."""
    del grad_dm_dmu1, grad_dm_dsigma1_sq, grad_dm_dsigma12
    prediction, target, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
    grad_scale = grad_score.reshape(1).to(torch.float32) / float(ctx.numel)
    grad_prediction = ssim_mojo_bwd_mean_op(
        prediction,
        target,
        grad_scale,
        dm_dmu1,
        dm_dsigma1_sq,
        dm_dsigma12,
    )
    return grad_prediction, None


ssim_mojo_mean_fwd_op.register_autograd(
    ssim_mojo_mean_backward,
    setup_context=ssim_mojo_mean_setup_context,
)


def ssim_mojo(
    prediction: Float[Tensor, " batch channels height width"],
    target: Float[Tensor, " batch channels height width"],
    *,
    padding: Literal["same", "valid"] = "same",
) -> Tensor:
    """Compute mean SSIM for NCHW images through the Mojo custom op."""
    validate_nchw(prediction, target)
    if padding == "same":
        return ssim_mojo_mean_fwd_op(
            prediction.contiguous(),
            target.contiguous(),
        )[0]
    if padding == "valid":
        ssim_map = ssim_mojo_fwd_op(
            prediction.contiguous(),
            target.contiguous(),
        )[0]
        return ssim_map[..., 5:-5, 5:-5].mean()
    raise ValueError(f"Unsupported SSIM padding mode: {padding!r}.")


__all__ = [
    "ssim_mojo",
    "ssim_mojo_bwd_mean_op",
    "ssim_mojo_bwd_op",
    "ssim_mojo_fwd_op",
    "ssim_mojo_mean_fwd_op",
]
