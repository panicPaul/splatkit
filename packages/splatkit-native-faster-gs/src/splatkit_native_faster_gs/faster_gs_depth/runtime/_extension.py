"""Private JIT extension loader for the FasterGS depth runtime."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_extension() -> Any:
    """Compile and load the depth-only blend extension."""
    package_root = Path(__file__).resolve().parent.parent
    native_root = package_root / "native"
    core_native_root = package_root.parent / "faster_gs" / "native"
    return load(
        name="splatkit_faster_gs_depth_ext",
        sources=[
            str(native_root / "bindings.cpp"),
            str(native_root / "depth_blend.cu"),
        ],
        extra_include_paths=[
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
