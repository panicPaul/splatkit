"""CLI entrypoint for Gaussian render benchmarks."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ember_core.benchmarks import (
    RenderAutogradBenchmarkResult,
    RenderBenchmarkResult,
    benchmark_backend_render,
    benchmark_backend_render_autograd,
)
from ember_core.core import BACKEND_REGISTRY, CameraState, GaussianScene3D
from ember_core.data import load_colmap_scene_record, resolve_colmap_scene_path
from ember_core.initialization import initialize_gaussian_scene_from_scene_record

_OPTIONAL_BACKEND_MODULES = (
    "ember_adapter_backends.fastgs",
    "ember_adapter_backends.fastergs",
    "ember_adapter_backends.gsplat",
    "ember_adapter_backends.inria",
    "ember_adapter_backends.stoch3dgs",
    "ember_native_faster_gs.faster_gs",
    "ember_native_faster_gs.faster_gs_depth",
    "ember_native_faster_gs.gaussian_pop",
    "ember_native_faster_gs_mojo.core",
    "ember_native_3dgrt.stoch3dgs",
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
    parser.add_argument(
        "--include-backward",
        action="store_true",
        help="Measure render forward and backward timings separately.",
    )
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


def _build_autograd_comparison_payload(
    primary: RenderAutogradBenchmarkResult,
    comparison: RenderAutogradBenchmarkResult | None = None,
) -> dict[str, Any]:
    """Format one or two autograd render benchmark results for JSON output."""
    if comparison is None:
        return asdict(primary)
    faster_forward_backend = (
        primary.backend
        if primary.mean_forward_ms <= comparison.mean_forward_ms
        else comparison.backend
    )
    faster_backward_backend = (
        primary.backend
        if primary.mean_backward_ms <= comparison.mean_backward_ms
        else comparison.backend
    )
    faster_total_backend = (
        primary.backend
        if primary.mean_total_ms <= comparison.mean_total_ms
        else comparison.backend
    )
    return {
        "primary": asdict(primary),
        "comparison": asdict(comparison),
        "delta_mean_forward_ms": float(
            comparison.mean_forward_ms - primary.mean_forward_ms
        ),
        "delta_mean_backward_ms": float(
            comparison.mean_backward_ms - primary.mean_backward_ms
        ),
        "delta_mean_total_ms": float(
            comparison.mean_total_ms - primary.mean_total_ms
        ),
        "ratio_forward_vs_primary": (
            float(comparison.mean_forward_ms / primary.mean_forward_ms)
            if primary.mean_forward_ms > 0.0
            else None
        ),
        "ratio_backward_vs_primary": (
            float(comparison.mean_backward_ms / primary.mean_backward_ms)
            if primary.mean_backward_ms > 0.0
            else None
        ),
        "ratio_total_vs_primary": (
            float(comparison.mean_total_ms / primary.mean_total_ms)
            if primary.mean_total_ms > 0.0
            else None
        ),
        "faster_forward_backend": faster_forward_backend,
        "faster_backward_backend": faster_backward_backend,
        "faster_total_backend": faster_total_backend,
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
    if args.include_backward:
        autograd_result = benchmark_backend_render_autograd(
            scene,
            camera,
            backend=backend,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
        )
        autograd_comparison = (
            None
            if compare_to is None
            else benchmark_backend_render_autograd(
                scene,
                camera,
                backend=compare_to,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
            )
        )
        payload = _build_autograd_comparison_payload(
            autograd_result,
            autograd_comparison,
        )
    else:
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
        payload = _build_comparison_payload(result, comparison)
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
