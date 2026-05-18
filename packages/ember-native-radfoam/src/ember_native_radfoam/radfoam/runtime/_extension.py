"""Private RADFOAM extension loader."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from ember_core.native.torch_extensions import load_torch_extension


@lru_cache(maxsize=1)
def load_extension() -> Any:
    """Compile and load Ember's vendored RADFOAM tracing extension."""
    native_root = (
        Path(__file__).resolve().parent.parent / "native"
    )
    source_root = native_root / "src"
    torch_bindings_root = native_root / "torch_bindings"
    return load_torch_extension(
        name="ember_radfoam_native_ext",
        sources=[
            str(native_root / "bindings.cpp"),
            str(torch_bindings_root / "pipeline_bindings.cpp"),
            str(torch_bindings_root / "triangulation_bindings.cpp"),
            str(source_root / "aabb_tree" / "aabb_tree.cu"),
            str(source_root / "delaunay" / "delaunay.cu"),
            str(source_root / "delaunay" / "sample_initial_tets.cu"),
            str(source_root / "delaunay" / "growth_iteration.cu"),
            str(source_root / "delaunay" / "delete_violations.cu"),
            str(source_root / "delaunay" / "triangulation_ops.cu"),
            str(source_root / "tracing" / "pipeline.cu"),
        ],
        extra_include_paths=[
            str(source_root),
            str(torch_bindings_root),
            str(native_root / "external"),
            str(native_root / "external" / "include"),
        ],
        extra_cflags=[
            "-O3",
            "-std=c++17",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
        ],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-std=c++17",
            "--extended-lambda",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-Xcompiler=-fvisibility=hidden",
            "-Xcompiler=-fvisibility-inlines-hidden",
        ],
        extra_ldflags=["-lcuda"],
        with_cuda=True,
        verbose=False,
    )
