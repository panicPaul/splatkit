"""MAX custom-op library loading for SSIM kernels."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from concurrent import futures
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


def custom_op_library_path() -> Path:
    """Return the source package path for the SSIM Mojo custom ops."""
    ops_root = Path(__file__).resolve().parent / "operations"
    if not ops_root.exists():
        raise RuntimeError(f"Missing SSIM Mojo operations package at {ops_root}.")
    return ops_root


@lru_cache(maxsize=1)
def custom_op_kernel_library() -> Any:
    """Return the loaded MAX kernel library for SSIM custom ops."""
    try:
        from max.experimental.torch.torch import KernelLibrary
    except Exception as exc:
        raise RuntimeError(
            "Failed to import MAX KernelLibrary support required for "
            f"`ember_splatting_training.ssim_mojo` ({exc!r})."
        ) from exc

    kernel_library = KernelLibrary()
    kernel_library.load_paths([custom_op_library_path()])
    return kernel_library


@lru_cache(maxsize=1)
def load_custom_op_library() -> Any:
    """Load the MAX custom-op library for the SSIM Mojo kernels."""
    try:
        from max.experimental.torch import CustomOpLibrary
    except Exception as exc:
        raise RuntimeError(
            "Failed to import MAX/Mojo custom-op support required for "
            f"`ember_splatting_training.ssim_mojo` ({exc!r})."
        ) from exc

    ops_root = custom_op_library_path()
    try:
        return CustomOpLibrary(ops_root)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load SSIM Mojo custom-op package from {ops_root} "
            f"({exc!r})."
        ) from exc


@lru_cache(maxsize=1)
def inference_session() -> Any:
    """Return the MAX inference session used by the SSIM custom-op graphs."""
    try:
        from max.experimental.torch.torch import (
            Accelerator,
            InferenceSession,
            accelerator_count,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to import MAX inference support required for "
            f"`ember_splatting_training.ssim_mojo` ({exc!r})."
        ) from exc

    devices = [Accelerator(index) for index in range(accelerator_count())]
    return InferenceSession(devices=devices)


def buffer_from_torch(tensor: torch.Tensor) -> Any:
    """Convert a Torch tensor to a MAX buffer using the current CUDA stream."""
    try:
        from max.experimental.torch.torch import Buffer, max_device
    except Exception as exc:
        raise RuntimeError(
            "Failed to import MAX DLPack conversion support required for "
            f"`ember_splatting_training.ssim_mojo` ({exc!r})."
        ) from exc

    if tensor.device.type == "cuda":
        stream = torch.cuda.current_stream(tensor.device).cuda_stream
        data = tensor.__dlpack__()
        try:
            return Buffer._from_dlpack(data, max_device(tensor.device), stream)
        except Exception:
            return Buffer.from_dlpack(tensor)
    return Buffer.from_dlpack(tensor)


inplace_model_cache: dict[str, futures.Future[Any]] = {}
inplace_model_cache_lock = threading.Lock()


def inplace_model_cache_key(
    op_name: str,
    output_tensors: Sequence[torch.Tensor],
    input_tensors: Sequence[torch.Tensor],
) -> str:
    """Return a stable compile-cache key for one in-place custom-op graph."""
    return ",".join(
        [
            op_name,
            *(
                f"out:{tensor.dtype}:{tuple(tensor.shape)}:{tensor.device}"
                for tensor in output_tensors
            ),
            *(
                f"in:{tensor.dtype}:{tuple(tensor.shape)}:{tensor.device}"
                for tensor in input_tensors
            ),
        ]
    )


def compile_inplace_custom_model(
    op_name: str,
    output_tensors: Sequence[torch.Tensor],
    input_tensors: Sequence[torch.Tensor],
) -> Any:
    """Compile a MAX graph that calls a Mojo op with mutable output buffers."""
    try:
        from max.experimental.torch.torch import (
            MLIRThreadPoolExecutor,
            call_with_default_mlir_context,
            max_tensor_type,
        )
        from max.graph import Graph, ops
    except Exception as exc:
        raise RuntimeError(
            "Failed to import MAX graph support required for "
            f"`ember_splatting_training.ssim_mojo` ({exc!r})."
        ) from exc

    cache_key = inplace_model_cache_key(op_name, output_tensors, input_tensors)
    with inplace_model_cache_lock:
        model_future = inplace_model_cache.get(cache_key)
        if model_future is None:

            def build_model() -> Any:
                output_types = [
                    max_tensor_type(tensor) for tensor in output_tensors
                ]
                input_types = [
                    max_tensor_type(tensor) for tensor in input_tensors
                ]
                graph_input_types = [
                    *(output_type.as_buffer() for output_type in output_types),
                    *input_types,
                ]
                with Graph(
                    f"{op_name}_graph",
                    input_types=graph_input_types,
                    kernel_library=custom_op_kernel_library(),
                ) as graph:
                    output_buffers = [
                        graph_input.buffer
                        for graph_input in graph.inputs[: len(output_types)]
                    ]
                    input_values = [
                        graph_input.tensor
                        for graph_input in graph.inputs[len(output_types) :]
                    ]
                    ops.inplace_custom(
                        op_name,
                        device=output_types[0].device,
                        values=[*output_buffers, *input_values],
                    )
                    graph.output()
                return inference_session().load(graph)

            executor = MLIRThreadPoolExecutor()
            model_future = executor.submit(
                lambda: call_with_default_mlir_context(build_model)
            )
            inplace_model_cache[cache_key] = model_future
    return model_future.result()


def run_inplace_custom_op(
    op_name: str,
    output_tensors: Sequence[torch.Tensor],
    input_tensors: Sequence[torch.Tensor],
) -> None:
    """Run a Mojo custom op through an in-place MAX graph."""
    model = compile_inplace_custom_model(op_name, output_tensors, input_tensors)
    converted_tensors = [
        *(buffer_from_torch(tensor) for tensor in output_tensors),
        *(buffer_from_torch(tensor) for tensor in input_tensors),
    ]
    model.execute(*converted_tensors)
