"""Vendored Stoch3DGS native build helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from torch.utils.cpp_extension import CUDA_HOME, load


@dataclass(frozen=True)
class Stoch3DGSPluginConfig:
    """Specialization inputs for the vendored Stoch3DGS OptiX tracer."""

    pipeline_type: str = "fullStochastic"
    backward_pipeline_type: str = "fullStochasticBwd"
    primitive_type: str = "instances"
    particle_kernel_degree: int = 4
    particle_kernel_min_response: float = 0.0113
    particle_kernel_density_clamping: bool = True
    particle_kernel_min_alpha: float = 1.0 / 255.0
    particle_kernel_max_alpha: float = 0.99
    particle_radiance_sph_degree: int = 0
    enable_normals: bool = True
    enable_hitcounts: bool = True


@dataclass(frozen=True)
class Stoch3DGSVendoredRuntime:
    """Loaded vendored tracer module plus the staged source root it uses."""

    tracer_class: type[Any]
    source_root: str


def get_cuda_home() -> str:
    """Return the CUDA toolkit path used for native compilation."""
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build the vendored Stoch3DGS backend."
        )
    return CUDA_HOME


def _native_root() -> Path:
    return (
        Path(__file__).resolve().parent.parent / "core" / "native" / "stoch3dgs"
    )


def _runtime_key(config: Stoch3DGSPluginConfig) -> str:
    payload = json.dumps(config.__dict__, separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _resolve_slangc() -> str | None:
    env_path = os.environ.get("SLANGC")
    if env_path:
        slangc = Path(env_path)
        if not slangc.is_file():
            raise RuntimeError(
                f"SLANGC points to a missing executable: {slangc}."
            )
        return str(slangc)

    path_slangc = shutil.which("slangc")
    if path_slangc is not None:
        return path_slangc

    return None


def _resolve_optix_include_dir() -> Path:
    vendored_include = _native_root() / "dependencies" / "optix-dev" / "include"
    if (vendored_include / "optix.h").is_file():
        return vendored_include

    env_include = os.environ.get("OPTIX_INCLUDE_DIR")
    if env_include:
        include_dir = Path(env_include)
        if not (include_dir / "optix.h").is_file():
            raise RuntimeError(
                f"OPTIX_INCLUDE_DIR must contain optix.h; missing at {include_dir / 'optix.h'}."
            )
        return include_dir

    env_home = os.environ.get("OPTIX_HOME")
    if env_home:
        include_dir = Path(env_home) / "include"
        if not (include_dir / "optix.h").is_file():
            raise RuntimeError(
                f"OPTIX_HOME must contain include/optix.h; missing at {include_dir / 'optix.h'}."
            )
        return include_dir

    for candidate in (
        Path("/usr/local/NVIDIA-OptiX-SDK/include"),
        Path("/opt/optix/include"),
        Path("/usr/include"),
    ):
        if (candidate / "optix.h").is_file():
            return candidate

    raise RuntimeError(
        "Could not find the OptiX SDK headers. "
        "Expected the vendored optix-dev/include bundle or set OPTIX_INCLUDE_DIR / OPTIX_HOME."
    )


def _copy_vendored_sources(stage_root: Path) -> None:
    source_root = _native_root()
    for relative in ("bindings.cpp", "include", "src"):
        src = source_root / relative
        dst = stage_root / relative
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _prepare_optix_symlink(stage_root: Path, optix_include_dir: Path) -> None:
    include_link = stage_root / "dependencies" / "optix-dev" / "include"
    include_link.parent.mkdir(parents=True, exist_ok=True)
    if include_link.is_symlink() or include_link.exists():
        if (
            include_link.is_symlink()
            and include_link.resolve() == optix_include_dir.resolve()
        ):
            return
        if include_link.is_symlink() or include_link.is_file():
            include_link.unlink()
        else:
            shutil.rmtree(include_link)
    include_link.symlink_to(optix_include_dir, target_is_directory=True)


def _run_slang_codegen(
    stage_root: Path,
    config: Stoch3DGSPluginConfig,
    *,
    slangc: str | None,
) -> None:
    include_root = stage_root / "include"
    slang_root = include_root / "3dgrt" / "kernels" / "slang"
    output_header = slang_root / "gaussianParticles.cuh"
    if slangc is None:
        if output_header.is_file():
            return
        raise RuntimeError(
            "slangc is required to regenerate gaussianParticles.cuh for the vendored "
            "Stoch3DGS backend, and no pre-generated header was found."
        )
    subprocess.run(
        [
            slangc,
            "-target",
            "cuda",
            "-I",
            str(include_root),
            "-line-directive-mode",
            "none",
            "-matrix-layout-row-major",
            "-O2",
            f"-DPARTICLE_RADIANCE_NUM_COEFFS={(config.particle_radiance_sph_degree + 1) ** 2}",
            f"-DGAUSSIAN_PARTICLE_KERNEL_DEGREE={config.particle_kernel_degree}",
            f"-DGAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY={config.particle_kernel_min_response}",
            f"-DGAUSSIAN_PARTICLE_MIN_ALPHA={config.particle_kernel_min_alpha}",
            f"-DGAUSSIAN_PARTICLE_MAX_ALPHA={config.particle_kernel_max_alpha}",
            f"-DGAUSSIAN_PARTICLE_ENABLE_NORMAL={int(config.enable_normals)}",
            f"-DGAUSSIAN_PARTICLE_SURFEL={int(config.primitive_type == 'trisurfel')}",
            str(slang_root / "models" / "gaussianParticles.slang"),
            str(slang_root / "models" / "shRadiativeParticles.slang"),
            "-o",
            str(output_header),
        ],
        check=True,
    )


def _stage_runtime_sources(config: Stoch3DGSPluginConfig) -> Path:
    stage_root = (
        _native_root()
        / "build"
        / "generated"
        / _runtime_key(config)
        / "stoch3dgs_vendored"
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    _copy_vendored_sources(stage_root)
    _prepare_optix_symlink(stage_root, _resolve_optix_include_dir())
    _run_slang_codegen(stage_root, config, slangc=_resolve_slangc())
    return stage_root


@cache
def load_stoch3dgs_optix_tracer_runtime(
    config: Stoch3DGSPluginConfig,
) -> Stoch3DGSVendoredRuntime:
    """Compile or load the vendored OptiX tracer extension and source tree."""
    stage_root = _stage_runtime_sources(config)
    cuda_home = get_cuda_home()
    optix_include_dir = _resolve_optix_include_dir()
    module_name = f"ember_stoch3dgs_native_{_runtime_key(config)}"
    build_directory = stage_root / "torch_extensions"
    build_directory.mkdir(parents=True, exist_ok=True)
    module = load(
        name=module_name,
        sources=[
            str(stage_root / "src" / "optixTracer.cpp"),
            str(stage_root / "src" / "particlePrimitives.cu"),
            str(stage_root / "bindings.cpp"),
        ],
        extra_cflags=[
            "-DNVDR_TORCH",
            "-O3",
            "-std=c++17",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
        ],
        extra_cuda_cflags=[
            "-DNVDR_TORCH",
            "-O3",
            "-std=c++17",
            "--extended-lambda",
            "--expt-relaxed-constexpr",
            "-Xcompiler=-fno-strict-aliasing",
            "-Xcompiler=-fvisibility=hidden",
            "-Xcompiler=-fvisibility-inlines-hidden",
        ],
        extra_ldflags=[
            f"-L{Path(cuda_home) / 'lib' / 'stubs'}",
            f"-L{Path(cuda_home) / 'targets' / 'x86_64-linux' / 'lib'}",
            f"-L{Path(cuda_home) / 'targets' / 'x86_64-linux' / 'lib' / 'stubs'}",
            "-lcuda",
            "-lnvrtc",
        ],
        extra_include_paths=[
            str(Path(cuda_home) / "targets" / "x86_64-linux" / "include"),
            str(stage_root / "include"),
            str(optix_include_dir),
        ],
        build_directory=str(build_directory),
        with_cuda=True,
        verbose=False,
    )
    return Stoch3DGSVendoredRuntime(
        tracer_class=module.VendoredOptixTracer,
        source_root=str(stage_root),
    )


__all__ = [
    "Stoch3DGSPluginConfig",
    "Stoch3DGSVendoredRuntime",
    "get_cuda_home",
    "load_stoch3dgs_optix_tracer_runtime",
]
