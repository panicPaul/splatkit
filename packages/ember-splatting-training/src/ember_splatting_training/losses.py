"""Reusable splatting reconstruction losses."""

from __future__ import annotations

from typing import Any

import torch
from ember_core.core.contracts import GaussianScene3D
from ember_core.training import LossResult, TrainState
from fused_ssim import fused_ssim
from jaxtyping import Float
from torch import Tensor


def nhwc_to_nchw(
    image: Float[Tensor, " batch height width channels"],
) -> Float[Tensor, " batch channels height width"]:
    """Convert a contiguous NHWC image batch to NCHW for SSIM backends."""
    return image.permute(0, 3, 1, 2).contiguous()


def ssim_score(
    prediction: Float[Tensor, " batch height width channels"],
    target: Float[Tensor, " batch height width channels"],
    *,
    padding: str = "same",
    train: bool = True,
    backend: str = "cuda",
) -> Tensor:
    """Compute fused SSIM over NHWC image batches."""
    if prediction.shape != target.shape:
        raise ValueError(
            "SSIM expects prediction and target to share NHWC shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    if prediction.ndim != 4:
        raise ValueError(
            f"SSIM expects NHWC rank-4 tensors, got rank {prediction.ndim}."
        )
    prediction_nchw = nhwc_to_nchw(prediction)
    target_nchw = nhwc_to_nchw(target)
    if backend == "cuda":
        return fused_ssim(
            prediction_nchw,
            target_nchw,
            padding=padding,
            train=train,
        )
    if backend in {"ssim_mojo", "mojo"}:
        if not train:
            with torch.no_grad():
                from ember_splatting_training.ssim_mojo import ssim_mojo

                return ssim_mojo(prediction_nchw, target_nchw, padding=padding)
        from ember_splatting_training.ssim_mojo import ssim_mojo

        return ssim_mojo(prediction_nchw, target_nchw, padding=padding)
    raise ValueError(f"Unsupported SSIM backend: {backend!r}.")


def dssim_loss(
    prediction: Float[Tensor, " batch height width channels"],
    target: Float[Tensor, " batch height width channels"],
    *,
    padding: str = "same",
    backend: str = "cuda",
) -> Tensor:
    """Compute DSSIM from fused SSIM."""
    return (
        1.0 - ssim_score(prediction, target, padding=padding, backend=backend)
    ) / 2.0


def rgb_l1_dssim_loss(
    state: TrainState,
    batch: Any,
    render_output: Any,
    *,
    weights: dict[str, float],
    lambda_l1: float = 0.8,
    lambda_dssim: float = 0.2,
    lambda_opacity_regularization: float = 0.0,
    lambda_scale_regularization: float = 0.0,
    ssim_backend: str = "cuda",
) -> LossResult:
    """Gaussian RGB reconstruction loss with optional simple regularizers."""
    del weights
    prediction = render_output.render
    target = batch.images
    if prediction.shape != target.shape:
        raise ValueError(
            "RGB reconstruction loss expects render and target images to "
            f"share NHWC shape, got {tuple(prediction.shape)} and "
            f"{tuple(target.shape)}."
        )
    l1_loss = (prediction - target).abs().mean()
    dssim = dssim_loss(prediction, target, backend=ssim_backend)
    scene = state.model.scene
    if not isinstance(scene, GaussianScene3D):
        raise TypeError("rgb_l1_dssim_loss expects a GaussianScene3D model.")
    opacity_regularization = torch.sigmoid(scene.logit_opacity).mean()
    scale_regularization = torch.exp(scene.log_scales).mean()
    loss = (
        lambda_l1 * l1_loss
        + lambda_dssim * dssim
        + lambda_opacity_regularization * opacity_regularization
        + lambda_scale_regularization * scale_regularization
    )
    return LossResult(
        loss=loss,
        metrics={
            "l1": float(l1_loss.detach().item()),
            "dssim": float(dssim.detach().item()),
            "opacity_regularization": float(
                opacity_regularization.detach().item()
            ),
            "scale_regularization": float(scale_regularization.detach().item()),
        },
    )


__all__ = [
    "dssim_loss",
    "nhwc_to_nchw",
    "rgb_l1_dssim_loss",
    "ssim_score",
]
