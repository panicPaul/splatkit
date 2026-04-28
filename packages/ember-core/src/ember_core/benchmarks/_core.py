"""Shared benchmark helpers for dataset and render timing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ember_core.core import CameraState, GaussianScene3D, render


@dataclass(frozen=True)
class DataloaderBenchmarkResult:
    """Aggregate dataloader benchmark metrics."""

    initialization_ms: float
    warmup_steps: int
    measured_steps: int
    warmup_ms_per_batch: float
    mean_ms_per_batch: float
    p50_ms_per_batch: float
    p90_ms_per_batch: float
    iters_per_sec: float


@dataclass(frozen=True)
class RenderBenchmarkResult:
    """Aggregate render benchmark metrics."""

    backend: str
    device: str
    image_size: tuple[int, int]
    num_points: int
    warmup_steps: int
    measured_steps: int
    first_frame_ms: float
    mean_ms_per_frame: float
    p50_ms_per_frame: float
    p90_ms_per_frame: float
    fps: float


@dataclass(frozen=True)
class RenderAutogradBenchmarkResult:
    """Aggregate forward and backward render benchmark metrics."""

    backend: str
    device: str
    image_size: tuple[int, int]
    num_points: int
    warmup_steps: int
    measured_steps: int
    first_forward_ms: float
    first_backward_ms: float
    mean_forward_ms: float
    p50_forward_ms: float
    p90_forward_ms: float
    mean_backward_ms: float
    p50_backward_ms: float
    p90_backward_ms: float
    mean_total_ms: float
    forward_fps: float
    backward_fps: float
    total_fps: float


def _duration_stats(samples_ms: list[float]) -> tuple[float, float, float]:
    if not samples_ms:
        return 0.0, 0.0, 0.0
    durations = np.asarray(samples_ms, dtype=np.float64)
    return (
        float(durations.mean()),
        float(np.quantile(durations, 0.5)),
        float(np.quantile(durations, 0.9)),
    )


def benchmark_dataloader(
    dataloader: DataLoader[Any],
    *,
    warmup_steps: int = 10,
    measured_steps: int = 100,
) -> DataloaderBenchmarkResult:
    """Benchmark dataloader initialization and steady-state iteration."""
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0.")
    if measured_steps <= 0:
        raise ValueError("measured_steps must be >= 1.")

    initialization_start = perf_counter()
    iterator = iter(dataloader)
    try:
        next(iterator)
    except StopIteration:
        initialization_ms = (perf_counter() - initialization_start) * 1000.0
        return DataloaderBenchmarkResult(
            initialization_ms=float(initialization_ms),
            warmup_steps=warmup_steps,
            measured_steps=0,
            warmup_ms_per_batch=0.0,
            mean_ms_per_batch=0.0,
            p50_ms_per_batch=0.0,
            p90_ms_per_batch=0.0,
            iters_per_sec=0.0,
        )
    initialization_ms = (perf_counter() - initialization_start) * 1000.0

    warmup_start = perf_counter()
    for _ in range(warmup_steps):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            next(iterator)
    warmup_ms = (perf_counter() - warmup_start) * 1000.0

    measured_samples_ms: list[float] = []
    for _ in range(measured_steps):
        sample_start = perf_counter()
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            next(iterator)
        measured_samples_ms.append((perf_counter() - sample_start) * 1000.0)

    mean_ms, p50_ms, p90_ms = _duration_stats(measured_samples_ms)
    return DataloaderBenchmarkResult(
        initialization_ms=float(initialization_ms),
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        warmup_ms_per_batch=(
            float(warmup_ms / warmup_steps) if warmup_steps > 0 else 0.0
        ),
        mean_ms_per_batch=mean_ms,
        p50_ms_per_batch=p50_ms,
        p90_ms_per_batch=p90_ms,
        iters_per_sec=float(1000.0 / mean_ms) if mean_ms > 0.0 else 0.0,
    )


def _synchronize_if_cuda(camera: CameraState) -> None:
    if camera.width.device.type == "cuda":
        torch.cuda.synchronize(camera.width.device)


def _clone_scene_for_grad(scene: GaussianScene3D) -> GaussianScene3D:
    """Clone benchmark inputs so repeated backward passes are independent."""
    return replace(
        scene,
        center_position=scene.center_position.detach()
        .clone()
        .requires_grad_(True),
        log_scales=scene.log_scales.detach().clone().requires_grad_(True),
        quaternion_orientation=scene.quaternion_orientation.detach()
        .clone()
        .requires_grad_(True),
        logit_opacity=scene.logit_opacity.detach().clone().requires_grad_(True),
        feature=scene.feature.detach().clone().requires_grad_(True),
    )


def benchmark_backend_render(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    backend: str,
    warmup_steps: int = 10,
    measured_steps: int = 100,
    options: Any | None = None,
) -> RenderBenchmarkResult:
    """Benchmark one backend's render path for a fixed scene and camera."""
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0.")
    if measured_steps <= 0:
        raise ValueError("measured_steps must be >= 1.")

    def timed_render() -> float:
        _synchronize_if_cuda(camera)
        start = perf_counter()
        render(scene, camera, backend=backend, options=options)
        _synchronize_if_cuda(camera)
        return (perf_counter() - start) * 1000.0

    first_frame_ms = timed_render()
    for _ in range(warmup_steps):
        timed_render()
    measured_samples_ms = [timed_render() for _ in range(measured_steps)]
    mean_ms, p50_ms, p90_ms = _duration_stats(measured_samples_ms)
    return RenderBenchmarkResult(
        backend=backend,
        device=str(camera.width.device),
        image_size=(
            int(camera.width[0].item()),
            int(camera.height[0].item()),
        ),
        num_points=int(scene.center_position.shape[0]),
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        first_frame_ms=float(first_frame_ms),
        mean_ms_per_frame=mean_ms,
        p50_ms_per_frame=p50_ms,
        p90_ms_per_frame=p90_ms,
        fps=float(1000.0 / mean_ms) if mean_ms > 0.0 else 0.0,
    )


def benchmark_backend_render_autograd(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    backend: str,
    warmup_steps: int = 10,
    measured_steps: int = 100,
    options: Any | None = None,
) -> RenderAutogradBenchmarkResult:
    """Benchmark one backend's render forward and backward paths."""
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0.")
    if measured_steps <= 0:
        raise ValueError("measured_steps must be >= 1.")

    def timed_forward_backward() -> tuple[float, float]:
        benchmark_scene = _clone_scene_for_grad(scene)

        _synchronize_if_cuda(camera)
        forward_start = perf_counter()
        output = render(
            benchmark_scene,
            camera,
            backend=backend,
            options=options,
        ).render
        _synchronize_if_cuda(camera)
        forward_ms = (perf_counter() - forward_start) * 1000.0

        loss = output.sum()
        _synchronize_if_cuda(camera)
        backward_start = perf_counter()
        loss.backward()
        _synchronize_if_cuda(camera)
        backward_ms = (perf_counter() - backward_start) * 1000.0
        return forward_ms, backward_ms

    first_forward_ms, first_backward_ms = timed_forward_backward()
    for _ in range(warmup_steps):
        timed_forward_backward()
    measured_samples_ms = [
        timed_forward_backward() for _ in range(measured_steps)
    ]
    forward_samples_ms = [
        forward_ms for forward_ms, _backward_ms in measured_samples_ms
    ]
    backward_samples_ms = [
        backward_ms for _forward_ms, backward_ms in measured_samples_ms
    ]
    total_samples_ms = [
        forward_ms + backward_ms
        for forward_ms, backward_ms in measured_samples_ms
    ]
    mean_forward_ms, p50_forward_ms, p90_forward_ms = _duration_stats(
        forward_samples_ms
    )
    mean_backward_ms, p50_backward_ms, p90_backward_ms = _duration_stats(
        backward_samples_ms
    )
    mean_total_ms, _p50_total_ms, _p90_total_ms = _duration_stats(
        total_samples_ms
    )
    return RenderAutogradBenchmarkResult(
        backend=backend,
        device=str(camera.width.device),
        image_size=(
            int(camera.width[0].item()),
            int(camera.height[0].item()),
        ),
        num_points=int(scene.center_position.shape[0]),
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        first_forward_ms=float(first_forward_ms),
        first_backward_ms=float(first_backward_ms),
        mean_forward_ms=mean_forward_ms,
        p50_forward_ms=p50_forward_ms,
        p90_forward_ms=p90_forward_ms,
        mean_backward_ms=mean_backward_ms,
        p50_backward_ms=p50_backward_ms,
        p90_backward_ms=p90_backward_ms,
        mean_total_ms=mean_total_ms,
        forward_fps=(
            float(1000.0 / mean_forward_ms) if mean_forward_ms > 0.0 else 0.0
        ),
        backward_fps=(
            float(1000.0 / mean_backward_ms) if mean_backward_ms > 0.0 else 0.0
        ),
        total_fps=float(1000.0 / mean_total_ms) if mean_total_ms > 0.0 else 0.0,
    )


__all__ = [
    "DataloaderBenchmarkResult",
    "RenderAutogradBenchmarkResult",
    "RenderBenchmarkResult",
    "benchmark_backend_render",
    "benchmark_backend_render_autograd",
    "benchmark_dataloader",
]
