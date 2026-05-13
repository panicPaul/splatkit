"""Private JIT extension loader for FasterGS native training utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from ember_core.native.torch_extensions import load_torch_extension


@lru_cache(maxsize=1)
def load_extension() -> Any:
    """Compile and load the vendored FasterGS training extension."""
    extension_name = "ember_faster_gs_training_native_ext"
    training_root = Path(__file__).resolve().parent.parent / "native"
    faster_gs_root = training_root.parent.parent
    return load_torch_extension(
        name=extension_name,
        sources=[
            str(training_root / "bindings.cpp"),
            str(training_root / "adam" / "src" / "adam.cu"),
            str(
                training_root / "densification" / "src" / "densification_api.cu"
            ),
            str(training_root / "densification" / "src" / "mcmc.cu"),
            str(
                training_root
                / "mip_splatting_3d_filter"
                / "src"
                / "mip_splatting_3d_filter.cu"
            ),
            str(training_root / "morton" / "src" / "morton.cu"),
        ],
        extra_include_paths=[
            str(faster_gs_root / "native" / "utils"),
            str(training_root / "adam" / "include"),
            str(training_root / "adam" / "src"),
            str(training_root / "densification" / "include"),
            str(training_root / "densification" / "src"),
            str(training_root / "mip_splatting_3d_filter" / "include"),
            str(training_root / "mip_splatting_3d_filter" / "src"),
            str(training_root / "morton" / "include"),
            str(training_root / "morton" / "src"),
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
