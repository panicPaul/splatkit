"""Public runtime API for FasterGS native training utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from ember_native_faster_gs.faster_gs.training.runtime._extension import (
    load_extension,
)
from ember_native_faster_gs.faster_gs.training.runtime.ops import (
    morton_codes_op,
)


def relocation_adjustment(
    old_opacities: Tensor,
    old_scales: Tensor,
    n_samples_per_primitive: Tensor,
) -> tuple[Tensor, Tensor]:
    """Adjust sampled Gaussian opacity/scale according to MCMC counts."""
    return load_extension().relocation_adjustment(
        old_opacities,
        old_scales,
        n_samples_per_primitive,
    )


def add_noise(
    raw_scales: Tensor,
    raw_rotations: Tensor,
    raw_opacities: Tensor,
    means: Tensor,
    current_lr: float,
) -> None:
    """Inject FasterGS MCMC noise into Gaussian means."""
    random_samples = torch.randn_like(means)
    load_extension().add_noise(
        raw_scales,
        raw_rotations,
        raw_opacities,
        random_samples,
        means,
        current_lr,
    )


def morton_codes(
    positions: Tensor,
    *,
    scene_min: Tensor | None = None,
    scene_extent: float | None = None,
) -> Tensor:
    """Return Morton codes for CUDA 3D Gaussian center positions."""
    if positions.device.type != "cuda":
        raise ValueError("morton_codes requires CUDA positions.")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "morton_codes expects positions with shape (num_points, 3)."
        )
    positions = positions.contiguous()
    if scene_min is None:
        scene_min = positions.amin(dim=0)
    else:
        scene_min = scene_min.to(device=positions.device, dtype=positions.dtype)
    if scene_extent is None:
        scene_max = positions.amax(dim=0)
        scene_extent = float((scene_max - scene_min).amax().clamp_min(1e-12))
    return morton_codes_op(
        positions,
        scene_min.contiguous(),
        float(scene_extent),
    )


def morton_order(positions: Tensor) -> Tensor:
    """Return indices that sort 3D positions by Morton code."""
    return torch.argsort(morton_codes(positions), stable=True)


def update_3d_filter(
    positions: Tensor,
    w2c: Tensor,
    filter_3d: Tensor,
    visibility_mask: Tensor,
    *,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    near_plane: float,
    clipping_tolerance: float,
    distance2filter: float,
) -> None:
    """Update Mip-Splatting 3D filter buffers in-place for one camera."""
    load_extension().update_3d_filter(
        positions.contiguous(),
        w2c.contiguous(),
        filter_3d,
        visibility_mask,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        clipping_tolerance,
        distance2filter,
    )


class FusedAdam(torch.optim.Adam):
    """FasterGS fused Adam optimizer."""

    def __init__(self, params: object, lr: float, eps: float) -> None:
        super().__init__(params=params, lr=lr, eps=eps)

    @torch.no_grad()
    def step(self, closure: object = None) -> None:
        """Run one fused Adam update."""
        del closure
        backend = load_extension()
        for group in self.param_groups:
            if len(group["params"]) != 1:
                raise ValueError("FusedAdam expects one tensor per group.")
            parameter = group["params"][0]
            if parameter.grad is None or parameter.numel() == 0:
                continue

            state = self.state[parameter]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)

            state["step"] += 1
            backend.adam_step(
                parameter.grad,
                parameter,
                state["exp_avg"],
                state["exp_avg_sq"],
                state["step"],
                group["lr"],
                *group["betas"],
                group["eps"],
            )


__all__ = [
    "FusedAdam",
    "add_noise",
    "morton_codes",
    "morton_order",
    "relocation_adjustment",
    "update_3d_filter",
]
