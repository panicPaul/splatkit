"""CLI entrypoint for Gaussian render benchmarks."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from splatkit.benchmarks import benchmark_backend_render
from splatkit.core import BACKEND_REGISTRY, CameraState, GaussianScene3D
from splatkit.data import load_colmap_dataset, resolve_colmap_scene_path
from splatkit.initialization import initialize_gaussian_scene_from_point_cloud

_OPTIONAL_BACKEND_MODULES = (
    "splatkit_adapter_backends.fastgs",
    "splatkit_adapter_backends.fastergs",
    "splatkit_adapter_backends.gsplat",
    "splatkit_adapter_backends.inria",
    "splatkit_adapter_backends.stoch3dgs",
    "splatkit_native_faster_gs.faster_gs",
    "splatkit_native_faster_gs.faster_gs_depth",
    "splatkit_native_faster_gs.gaussian_pop",
    "splatkit_native_3dgrt.stoch3dgs",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Gaussian render speed on a COLMAP sample scene."
    )
    parser.add_argument("--colmap-root", type=Path, default=None)
    parser.add_argument("--backend", default=None)
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
    scene_dataset = load_colmap_dataset(
        resolve_colmap_scene_path(args.colmap_root)
    )
    device = _resolve_device(args.device)
    scene = initialize_gaussian_scene_from_point_cloud(
        scene_dataset,
        sh_degree=args.sh_degree,
        initial_scale=args.initial_scale,
        initial_opacity=args.initial_opacity,
    ).to(device)
    camera = _select_first_camera(
        scene_dataset.resolve_camera_sensor().camera
    ).to(device)
    result = benchmark_backend_render(
        scene,
        camera,
        backend=backend,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
