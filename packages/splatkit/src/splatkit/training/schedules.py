"""Torch-native learning-rate schedule helpers for declarative training."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def exponential_decay_to(
    optimizer: Optimizer,
    *,
    final_lr: float,
    max_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Build a step-wise exponential decay schedule to a target final LR."""
    if final_lr <= 0.0:
        raise ValueError("final_lr must be > 0.")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")
    initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if any(lr <= 0.0 for lr in initial_lrs):
        raise ValueError("All optimizer learning rates must be > 0.")

    def lr_lambda(step: int) -> float:
        bounded_step = min(max(step, 0), max_steps)
        if bounded_step == 0:
            return 1.0
        ratio = bounded_step / max_steps
        return math.exp(math.log(final_lr / initial_lrs[0]) * ratio)

    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


__all__ = ["exponential_decay_to"]
