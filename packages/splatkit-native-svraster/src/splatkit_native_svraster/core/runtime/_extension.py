"""Private JIT extension loader for the SVRaster native runtime."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from torch.utils.cpp_extension import load


def _cuda_include_paths() -> list[str]:
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    target_include = cuda_home / "targets" / "x86_64-linux" / "include"
    cccl_include = target_include / "cccl"
    include_paths: list[str] = []
    for path in (target_include, cccl_include):
        if path.exists():
            include_paths.append(str(path))
    return include_paths


@lru_cache(maxsize=1)
def load_extension() -> Any:
    """Compile and load the vendored SVRaster extension."""
    native_root = Path(__file__).resolve().parent.parent / "native"
    source_root = native_root / "src"
    return load(
        name="splatkit_svraster_native_ext",
        sources=[
            str(native_root / "bindings.cpp"),
            str(source_root / "raster_state.cu"),
            str(source_root / "preprocess.cu"),
            str(source_root / "forward.cu"),
            str(source_root / "backward.cu"),
            str(source_root / "geo_params_gather.cu"),
            str(source_root / "sh_compute.cu"),
            str(source_root / "utils.cu"),
        ],
        extra_include_paths=[
            str(native_root),
            str(source_root),
            *_cuda_include_paths(),
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
