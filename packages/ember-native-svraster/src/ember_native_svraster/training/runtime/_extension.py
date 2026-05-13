"""Private JIT extension loader for SVRaster native training utilities."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from ember_core.native.torch_extensions import load_torch_extension


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
    """Compile and load the vendored SVRaster training extension."""
    extension_name = "ember_svraster_training_native_ext"
    training_root = Path(__file__).resolve().parent.parent / "native"
    source_root = training_root / "src"
    return load_torch_extension(
        name=extension_name,
        sources=[
            str(training_root / "bindings.cpp"),
            str(source_root / "adam_step.cu"),
            str(source_root / "tv_compute.cu"),
        ],
        extra_include_paths=[
            str(training_root),
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
