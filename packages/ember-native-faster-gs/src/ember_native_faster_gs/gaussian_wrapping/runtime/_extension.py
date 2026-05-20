"""Private JIT extension loaders for Gaussian Wrapping CUDA stages."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from ember_core.native.torch_extensions import load_torch_extension


def _load_wrapping_extension(name: str, native_root: Path) -> Any:
    upstream_root = native_root / "upstream"
    return load_torch_extension(
        name=name,
        sources=[
            str(native_root / "bindings.cpp"),
            str(upstream_root / "rasterize_points.cu"),
            str(upstream_root / "cuda_rasterizer" / "rasterizer_impl.cu"),
            *[
                str(path)
                for path in sorted(
                    (upstream_root / "cuda_rasterizer").glob("*_forward.cu")
                )
            ],
            *[
                str(path)
                for path in sorted(
                    (upstream_root / "cuda_rasterizer").glob("*_backward.cu")
                )
            ],
            *[
                str(path)
                for path in sorted(
                    (upstream_root / "cuda_rasterizer").glob("forward.cu")
                )
            ],
            *[
                str(path)
                for path in sorted(
                    (upstream_root / "cuda_rasterizer").glob("backward.cu")
                )
            ],
        ],
        extra_include_paths=[
            str(native_root),
            str(upstream_root),
            str(upstream_root / "cuda_rasterizer"),
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
            "--extended-lambda",
            "-Xcompiler=-fvisibility=hidden",
            "-Xcompiler=-fvisibility-inlines-hidden",
        ],
        with_cuda=True,
        verbose=False,
    )


@lru_cache(maxsize=1)
def load_ours_extension() -> Any:
    """Compile and load the Gaussian Wrapping ``ours`` CUDA extension."""
    native_root = Path(__file__).resolve().parent.parent / "native" / "ours"
    return _load_wrapping_extension(
        "ember_gaussian_wrapping_ours_native_ext",
        native_root,
    )


@lru_cache(maxsize=1)
def load_radegs_extension() -> Any:
    """Compile and load the Gaussian Wrapping RaDe-GS CUDA extension."""
    native_root = Path(__file__).resolve().parent.parent / "native" / "radegs"
    return _load_wrapping_extension(
        "ember_gaussian_wrapping_radegs_native_ext",
        native_root,
    )
