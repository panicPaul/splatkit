"""CLI entrypoint for Gaussian render benchmarks."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from splatkit.benchmarks import RenderBenchmarkResult, benchmark_backend_render
from splatkit.core import BACKEND_REGISTRY, CameraState, GaussianScene3D
from splatkit.data import load_colmap_scene_record, resolve_colmap_scene_path
from splatkit.initialization import initialize_gaussian_scene_from_scene_record

_OPTIONAL_BACKEND_MODULES = (
    "splatkit_adapter_backends.fastgs",
    "splatkit_adapter_backends.fastergs",
    "splatkit_adapter_backends.gsplat",
    "splatkit_adapter_backends.inria",
    "splatkit_adapter_backends.stoch3dgs",
    "splatkit_native_faster_gs.faster_gs",
    "splatkit_native_faster_gs.faster_gs_depth",
    "splatkit_native_faster_gs.gaussian_pop",
    "splatkit_native_faster_gs_mojo.core",
    "splatkit_native_3dgrt.stoch3dgs",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Gaussian render speed on a COLMAP sample scene."
    )
    parser.add_argument("--colmap-root", type=Path, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--compare-to", default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--sh-degree", type=int, default=0)
    parser.add_argument("--initial-scale", type=float, default=0.01)
    parser.add_argument("--initial-opacity", type=float, default=0.1)
    return parser


def _register_optional_backends() -> None:
    for module_name in _OPTIONAL_BACKEND_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        register = getattr(module, "register", None)
        if callable(register):
            register()


def _available_gaussian_backends() -> list[str]:
    return sorted(
        backend_name
        for backend_name, backend in BACKEND_REGISTRY.items()
        if any(
            issubclass(GaussianScene3D, scene_type)
            for scene_type in backend.accepted_scene_types
        )
    )


def _pick_backend(requested_backend: str | None) -> str:
    available = _available_gaussian_backends()
    if requested_backend is not None:
        if requested_backend not in available:
            raise ValueError(
                f"Backend {requested_backend!r} is not available. "
                f"Choices: {available!r}."
            )
        return requested_backend
    if "adapter.gsplat" in available:
        return "adapter.gsplat"
    if not available:
        raise ValueError(
            "No Gaussian render backends are registered. Install and import "
            "at least one backend package before running this benchmark."
        )
    return available[0]


def _build_comparison_payload(
    primary: RenderBenchmarkResult,
    comparison: RenderBenchmarkResult | None = None,
) -> dict[str, Any]:
    """Format one or two render benchmark results for JSON output."""
    if comparison is None:
        return asdict(primary)
    faster_backend = (
        primary.backend
        if primary.mean_ms_per_frame <= comparison.mean_ms_per_frame
        else comparison.backend
    )
    return {
        "primary": asdict(primary),
        "comparison": asdict(comparison),
        "delta_ms_per_frame": float(
            comparison.mean_ms_per_frame - primary.mean_ms_per_frame
        ),
        "ratio_vs_primary": (
            float(comparison.mean_ms_per_frame / primary.mean_ms_per_frame)
            if primary.mean_ms_per_frame > 0.0
            else None
        ),
        "faster_backend": faster_backend,
    }


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _select_first_camera(camera: CameraState) -> CameraState:
    return CameraState(
        width=camera.width[:1],
        height=camera.height[:1],
        fov_degrees=camera.fov_degrees[:1],
        cam_to_world=camera.cam_to_world[:1],
        intrinsics=(
            None if camera.intrinsics is None else camera.intrinsics[:1]
        ),
        camera_convention=camera.camera_convention,
    )


def main() -> None:
    """Run the render benchmark CLI and print JSON metrics."""
    args = _build_parser().parse_args()
    _register_optional_backends()
    backend = _pick_backend(args.backend)
    compare_to = (
        None
        if args.compare_to is None or args.compare_to == backend
        else _pick_backend(args.compare_to)
    )
    scene_record = load_colmap_scene_record(
        resolve_colmap_scene_path(args.colmap_root)
    )
    device = _resolve_device(args.device)
    scene = initialize_gaussian_scene_from_scene_record(
        scene_record,
        sh_degree=args.sh_degree,
        initial_scale=args.initial_scale,
        initial_opacity=args.initial_opacity,
    ).to(device)
    camera = _select_first_camera(
        scene_record.resolve_camera_sensor().camera
    ).to(device)
    result = benchmark_backend_render(
        scene,
        camera,
        backend=backend,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
    )
    comparison = (
        None
        if compare_to is None
        else benchmark_backend_render(
            scene,
            camera,
            backend=compare_to,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
        )
    )
    print(
        json.dumps(
            _build_comparison_payload(result, comparison),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
