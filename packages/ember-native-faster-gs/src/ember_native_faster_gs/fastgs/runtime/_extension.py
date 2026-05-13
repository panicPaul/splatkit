"""Private JIT extension loader for the native FastGS runtime."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from ember_core.native.torch_extensions import load_torch_extension


@lru_cache(maxsize=1)
def load_extension() -> Any:
    """Compile and load the FastGS-specific rasterization extension."""
    extension_name = "ember_fastgs_native_ext"
    fastgs_root = Path(__file__).resolve().parent.parent / "native"
    core_native_root = (
        Path(__file__).resolve().parents[2] / "faster_gs" / "native"
    )
    return load_torch_extension(
        name=extension_name,
        sources=[
            str(fastgs_root / "bindings.cpp"),
            str(fastgs_root / "stages.cu"),
        ],
        extra_include_paths=[
            str(fastgs_root),
            str(core_native_root / "utils"),
            str(core_native_root / "rasterization" / "include"),
            str(core_native_root / "rasterization" / "src"),
        ],
        extra_cflags=[
            "-O3",
            "-std=c++17",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
        ],
        extra_cuda_cflags=[
            "-O3",
            "-use_fast_math",
            "-std=c++17",
            "-Xcompiler=-fvisibility=hidden",
            "-Xcompiler=-fvisibility-inlines-hidden",
        ],
        with_cuda=True,
        verbose=False,
    )
