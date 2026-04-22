"""Registered custom ops for the traced native core."""

from splatkit_native_backends.traced_native_core.runtime.ops.lifecycle import (
    build_acc_op,
    destroy_acc_op,
    update_acc_op,
)
from splatkit_native_backends.traced_native_core.runtime.ops.render import (
    render_bwd_op,
    render_fwd_op,
    render_op,
)
from splatkit_native_backends.traced_native_core.runtime.ops.trace import (
    trace_bwd_op,
    trace_fwd_op,
    trace_op,
)

__all__ = [
    "build_acc_op",
    "destroy_acc_op",
    "render_bwd_op",
    "render_fwd_op",
    "render_op",
    "trace_bwd_op",
    "trace_fwd_op",
    "trace_op",
    "update_acc_op",
]
