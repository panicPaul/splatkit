"""CLI entrypoint for benchmarking a Gaussian PLY render path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ember_core.benchmarks import (
    benchmark_backend_render,
    benchmark_backend_render_autograd,
)
from ember_core.core import CameraState
from ember_core.io import load_gaussian_ply

from .render import (
    _build_autograd_comparison_payload,
    _build_comparison_payload,
    _pick_backend,
    _register_optional_backends,
    _resolve_device,
)


def _default_repo_point_cloud_path() -> Path:
    """Return the repository-local default Gaussian PLY benchmark asset."""
    return Path(__file__).resolve().parents[5] / "point_cloud.ply"


def _build_default_camera(
    *,
    device: torch.device,
    width: int,
    height: int,
    fov_degrees: float,
    camera_z: float,
) -> CameraState:
    """Build a simple fixed camera for Gaussian PLY benchmark renders."""
    cam_to_world = torch.eye(4, dtype=torch.float32, device=device)[None]
    cam_to_world[:, 2, 3] = camera_z
    return CameraState(
        width=torch.tensor([width], dtype=torch.int64, device=device),
        height=torch.tensor([height], dtype=torch.int64, device=device),
        fov_degrees=torch.tensor(
            [fov_degrees],
            dtype=torch.float32,
            device=device,
        ),
        cam_to_world=cam_to_world,
        camera_convention="opencv",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Gaussian render speed on the repository-local "
            "`point_cloud.ply` asset."
        )
    )
    parser.add_argument(
        "--ply-path",
        type=Path,
        default=_default_repo_point_cloud_path(),
    )
    parser.add_argument("--backend", default="faster_gs_mojo.core")
    parser.add_argument("--compare-to", default="faster_gs.core")
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
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov-degrees", type=float, default=60.0)
    parser.add_argument("--camera-z", type=float, default=3.0)
    return parser


def main() -> None:
    """Run the Gaussian PLY render benchmark CLI and print JSON metrics."""
    args = _build_parser().parse_args()
    _register_optional_backends()

    primary_backend = _pick_backend(args.backend)
    compare_to = (
        None
        if args.compare_to is None or args.compare_to == primary_backend
        else _pick_backend(args.compare_to)
    )

    ply_path = args.ply_path.expanduser()
    if not ply_path.exists():
        raise FileNotFoundError(
            f"Gaussian PLY benchmark asset was not found at {ply_path}."
        )

    device = _resolve_device(args.device)
    scene = load_gaussian_ply(ply_path).to(device)
    camera = _build_default_camera(
        device=device,
        width=args.width,
        height=args.height,
        fov_degrees=args.fov_degrees,
        camera_z=args.camera_z,
    )

    if args.include_backward:
        primary_autograd = benchmark_backend_render_autograd(
            scene,
            camera,
            backend=primary_backend,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
        )
        comparison_autograd = (
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
            primary_autograd,
            comparison_autograd,
        )
    else:
        primary = benchmark_backend_render(
            scene,
            camera,
            backend=primary_backend,
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
        payload = _build_comparison_payload(primary, comparison)
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
