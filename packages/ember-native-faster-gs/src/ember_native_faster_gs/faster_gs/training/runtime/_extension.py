"""Private JIT extension loader for FasterGS native training utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def load_extension() -> Any:
    """Compile and load the vendored FasterGS training extension."""
    from torch.utils.cpp_extension import load

    training_root = Path(__file__).resolve().parent.parent / "native"
    faster_gs_root = training_root.parent.parent
    return load(
        name="ember_faster_gs_training_native_ext",
        sources=[
            str(training_root / "bindings.cpp"),
            str(training_root / "adam" / "src" / "adam.cu"),
            str(
                training_root / "densification" / "src" / "densification_api.cu"
            ),
            str(training_root / "densification" / "src" / "mcmc.cu"),
        ],
        extra_include_paths=[
            str(faster_gs_root / "native" / "utils"),
            str(training_root / "adam" / "include"),
            str(training_root / "adam" / "src"),
            str(training_root / "densification" / "include"),
            str(training_root / "densification" / "src"),
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
