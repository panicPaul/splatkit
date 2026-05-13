"""Vendored NHT rasterizer native build helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from ember_core.native.torch_extensions import load_torch_extension
from torch.utils.cpp_extension import CUDA_HOME


@dataclass(frozen=True)
class NHTRasterizerBuildConfig:
    """Specialization inputs for the vendored NHT CUDA rasterizer."""

    fast_math: bool = True
    num_channels: tuple[int, ...] = (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        12,
        16,
        17,
        20,
        24,
        28,
        32,
        33,
        36,
        40,
        44,
        48,
        49,
        64,
        65,
        80,
        96,
        128,
        129,
        256,
        257,
        512,
        513,
    )


@dataclass(frozen=True)
class NHTRasterizerVendoredRuntime:
    """Loaded vendored rasterizer module plus the staged source root."""

    module: Any
    source_root: str


def get_cuda_home() -> str:
    """Return the CUDA toolkit path used for native compilation."""
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build native NHT rasterization."
        )
    return CUDA_HOME


def _native_root() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "threedgut"
        / "core"
        / "native"
        / "nht_rasterizer"
    )


def _runtime_key(config: NHTRasterizerBuildConfig) -> str:
    payload = json.dumps(config.__dict__, separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _copy_vendored_sources(stage_root: Path) -> None:
    source_root = _native_root()
    for relative_path in ("bindings.cpp", "stages.h", "upstream"):
        source_path = source_root / relative_path
        destination_path = stage_root / relative_path
        if source_path.is_dir():
            if destination_path.exists():
                shutil.rmtree(destination_path)
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)


def _stage_runtime_sources(config: NHTRasterizerBuildConfig) -> Path:
    stage_root = (
        _native_root()
        / "build"
        / "generated"
        / _runtime_key(config)
        / "nht_rasterizer_vendored"
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    _copy_vendored_sources(stage_root)
    return stage_root


def _source_files(stage_root: Path) -> list[str]:
    upstream_source_root = stage_root / "upstream" / "csrc"
    return [
        str(upstream_source_root / "Intersect.cpp"),
        str(upstream_source_root / "IntersectTile.cu"),
        str(upstream_source_root / "Projection.cpp"),
        str(upstream_source_root / "ProjectionEWA3DGSFused.cu"),
        str(upstream_source_root / "ProjectionEWA3DGSPacked.cu"),
        str(upstream_source_root / "ProjectionEWASimple.cu"),
        str(upstream_source_root / "ProjectionUT3DGSFused.cu"),
        str(upstream_source_root / "QuatScaleToCovar.cpp"),
        str(upstream_source_root / "QuatScaleToCovarCUDA.cu"),
        str(upstream_source_root / "Rasterization.cpp"),
        str(upstream_source_root / "RasterizationNHT.cpp"),
        str(upstream_source_root / "RasterizeToIndices3DGS.cu"),
        str(upstream_source_root / "RasterizeToPixels3DGSFwd.cu"),
        str(upstream_source_root / "RasterizeToPixels3DGSBwd.cu"),
        str(upstream_source_root / "RasterizeToPixelsFromWorld3DGSFwd.cu"),
        str(upstream_source_root / "RasterizeToPixelsFromWorld3DGSBwd.cu"),
        str(upstream_source_root / "RasterizeToPixelsFromWorldNHT3DGSFwd.cu"),
        str(upstream_source_root / "RasterizeToPixelsFromWorldNHT3DGSBwd.cu"),
        str(stage_root / "bindings.cpp"),
    ]


@cache
def load_nht_rasterizer_runtime(
    config: NHTRasterizerBuildConfig = NHTRasterizerBuildConfig(),
) -> NHTRasterizerVendoredRuntime:
    """Compile or load the staged vendored NHT rasterizer extension."""
    stage_root = _stage_runtime_sources(config)
    module_name = f"ember_nht_rasterizer_native_{_runtime_key(config)}"
    build_directory = stage_root / "torch_extensions"
    build_directory.mkdir(parents=True, exist_ok=True)
    channel_macro = ",".join(str(channel) for channel in config.num_channels)
    cuda_channel_macro = '"' + channel_macro.replace(",", "\\,") + '"'

    common_defines = [
        "-DNDEBUG",
        "-DGSPLAT_BUILD_2DGS=0",
        "-DGSPLAT_BUILD_3DGS=1",
        "-DGSPLAT_BUILD_3DGUT=1",
        "-DGSPLAT_BUILD_ADAM=0",
        "-DGSPLAT_BUILD_RELOC=0",
    ]
    cxx_defines = [*common_defines, f"-DGSPLAT_NUM_CHANNELS={channel_macro}"]
    cuda_defines = [
        *common_defines,
        f"-DGSPLAT_NUM_CHANNELS={cuda_channel_macro}",
    ]
    common_compile_flags = [
        "-O3",
        "-std=c++20",
        *cxx_defines,
        "-fvisibility=hidden",
        "-fvisibility-inlines-hidden",
    ]
    cuda_compile_flags = [
        "-O3",
        "-std=c++20",
        *cuda_defines,
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fvisibility=hidden",
        "-Xcompiler=-fvisibility-inlines-hidden",
    ]
    if config.fast_math:
        cuda_compile_flags.append("-use_fast_math")

    module = load_torch_extension(
        name=module_name,
        sources=_source_files(stage_root),
        extra_include_paths=[
            str(stage_root / "upstream" / "include"),
            str(stage_root / "upstream" / "csrc"),
            str(stage_root),
        ],
        extra_cflags=common_compile_flags,
        extra_cuda_cflags=cuda_compile_flags,
        build_directory=str(build_directory),
        with_cuda=True,
        verbose=False,
    )
    return NHTRasterizerVendoredRuntime(
        module=module,
        source_root=str(stage_root),
    )


__all__ = [
    "NHTRasterizerBuildConfig",
    "NHTRasterizerVendoredRuntime",
    "get_cuda_home",
    "load_nht_rasterizer_runtime",
]
