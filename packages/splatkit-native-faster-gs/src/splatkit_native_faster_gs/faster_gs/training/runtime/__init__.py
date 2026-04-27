"""Public runtime API for FasterGS native training utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from splatkit_native_faster_gs.faster_gs.training.runtime._extension import (
    load_extension,
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
    "relocation_adjustment",
]
