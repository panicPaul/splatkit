"""Acceleration-state lifecycle ops for traced native backends."""

from __future__ import annotations

import torch
from torch import Tensor

from splatkit_native_backends.traced_native_core.runtime.state import (
    build_or_update_acc,
    destroy_state_token,
)


@torch.library.custom_op("traced_native_core::build_acc", mutates_args=())
def build_acc_op(
    state_token: Tensor,
    particle_density: Tensor,
) -> tuple[Tensor]:
    """Build the acceleration structure for a traced state."""
    build_or_update_acc(state_token, particle_density, force_rebuild=True)
    return (state_token.clone(),)


@build_acc_op.register_fake
def _build_acc_fake(
    state_token: Tensor,
    particle_density: Tensor,
) -> tuple[Tensor]:
    del particle_density
    return (state_token.clone(),)


@torch.library.custom_op("traced_native_core::update_acc", mutates_args=())
def update_acc_op(
    state_token: Tensor,
    particle_density: Tensor,
) -> tuple[Tensor]:
    """Update the acceleration structure for a traced state."""
    build_or_update_acc(state_token, particle_density, force_rebuild=False)
    return (state_token.clone(),)


@update_acc_op.register_fake
def _update_acc_fake(
    state_token: Tensor,
    particle_density: Tensor,
) -> tuple[Tensor]:
    del particle_density
    return (state_token.clone(),)


@torch.library.custom_op("traced_native_core::destroy_acc", mutates_args=())
def destroy_acc_op(state_token: Tensor) -> tuple[Tensor]:
    """Release the traced acceleration state."""
    destroy_state_token(state_token)
    return (state_token.clone(),)


@destroy_acc_op.register_fake
def _destroy_acc_fake(state_token: Tensor) -> tuple[Tensor]:
    return (state_token.clone(),)


__all__ = [
    "build_acc_op",
    "destroy_acc_op",
    "update_acc_op",
]
