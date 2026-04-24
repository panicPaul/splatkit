"""Glue helpers for composing staged FasterGS-style custom ops."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import torch
from torch import Tensor

OpOutputs = TypeVar("OpOutputs", bound=tuple[Tensor, ...])


def register_blend_family(
    *,
    op_name: str,
    forward_impl: Callable[..., OpOutputs],
    fake_impl: Callable[..., OpOutputs],
    setup_context: Callable[[Any, tuple[Any, ...], OpOutputs], None],
    backward_impl: Callable[..., tuple[Tensor | None, ...]],
) -> Any:
    """Register an autograd-carrying blend op from raw forward/backward pieces."""
    blend_op = torch.library.custom_op(op_name, mutates_args=())(forward_impl)
    blend_op.register_fake(fake_impl)
    blend_op.register_autograd(
        backward_impl,
        setup_context=setup_context,
    )
    return blend_op


def register_render_family(
    *,
    op_name: str,
    forward_impl: Callable[..., OpOutputs],
    fake_impl: Callable[..., OpOutputs],
    setup_context: Callable[[Any, tuple[Any, ...], OpOutputs], None],
    backward_impl: Callable[..., tuple[Tensor | None, ...]],
) -> Any:
    """Register an autograd-carrying render op from raw stage functions."""
    render_op = torch.library.custom_op(op_name, mutates_args=())(forward_impl)
    render_op.register_fake(fake_impl)
    render_op.register_autograd(
        backward_impl,
        setup_context=setup_context,
    )
    return render_op


__all__ = [
    "register_blend_family",
    "register_render_family",
]
