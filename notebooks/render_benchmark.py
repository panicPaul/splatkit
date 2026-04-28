"""Benchmark Gaussian render speed on the bundled COLMAP smoke scene."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import importlib
    import json
    import os
    from dataclasses import asdict
    from pathlib import Path

    import ember_core as sk
    import marimo as mo
    import torch
    from ember_core.benchmarks import benchmark_backend_render
    from ember_core.core import BACKEND_REGISTRY, GaussianScene3D
    from ember_core.initialization import (
        initialize_gaussian_scene_from_scene_record,
    )

    OPTIONAL_BACKEND_MODULES = (
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

    def register_optional_backends() -> None:
        """Register any Gaussian backends available in the environment."""
        for module_name in OPTIONAL_BACKEND_MODULES:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            register = getattr(module, "register", None)
            if callable(register):
                register()

    def available_gaussian_backends() -> list[str]:
        """Return registered backends that accept GaussianScene3D inputs."""
        register_optional_backends()
        return sorted(
            backend_name
            for backend_name, backend in BACKEND_REGISTRY.items()
            if any(
                issubclass(GaussianScene3D, scene_type)
                for scene_type in backend.accepted_scene_types
            )
        )

    def select_first_camera(camera: sk.CameraState) -> sk.CameraState:
        """Extract a single-camera batch for render benchmarking."""
        return sk.CameraState(
            width=camera.width[:1],
            height=camera.height[:1],
            fov_degrees=camera.fov_degrees[:1],
            cam_to_world=camera.cam_to_world[:1],
            intrinsics=(
                None if camera.intrinsics is None else camera.intrinsics[:1]
            ),
            camera_convention=camera.camera_convention,
        )


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Render Benchmark

    Benchmark a Gaussian backend on the bundled COLMAP smoke scene or an
    explicit local COLMAP root.
    """)
    return


@app.cell
def _():
    scene_root = mo.ui.text(
        value=os.environ.get(
            "EMBER_COLMAP_ROOT",
            str(sk.get_sample_scene_path()),
        ),
        label="COLMAP Root",
        full_width=True,
    )
    return (scene_root,)


@app.cell
def _():
    backend_options = available_gaussian_backends()
    backend = mo.ui.dropdown(
        options=backend_options,
        value=(
            "adapter.gsplat"
            if "adapter.gsplat" in backend_options
            else (backend_options[0] if backend_options else None)
        ),
        label="Backend",
        full_width=True,
    )
    compare_backend = mo.ui.dropdown(
        options=["(none)", *backend_options],
        value=(
            "faster_gs.core"
            if "faster_gs.core" in backend_options
            else "(none)"
        ),
        label="Compare To",
        full_width=True,
    )
    return backend, backend_options, compare_backend


@app.cell
def _():
    device = mo.ui.dropdown(
        options=["auto", "cpu", "cuda"],
        value="auto",
        label="Device",
        full_width=True,
    )
    warmup_steps = mo.ui.number(start=0, stop=1_000, value=10, label="Warmup")
    measured_steps = mo.ui.number(
        start=1,
        stop=10_000,
        value=100,
        label="Measured",
    )
    return device, measured_steps, warmup_steps


@app.cell
def _():
    run_button = mo.ui.button(
        value=0,
        label="Run Render Benchmark",
        on_click=lambda value: (0 if value is None else int(value)) + 1,
    )
    return (run_button,)


@app.cell
def _(
    backend,
    compare_backend,
    device,
    measured_steps,
    run_button,
    scene_root,
    warmup_steps,
):
    controls = mo.vstack(
        [
            scene_root,
            mo.hstack([backend, compare_backend, device], widths="equal"),
            mo.hstack([warmup_steps, measured_steps], widths="equal"),
            run_button,
        ],
        gap=0.75,
    )
    controls
    return


@app.cell
def _(
    backend,
    backend_options,
    compare_backend,
    device,
    measured_steps,
    run_button,
    scene_root,
    warmup_steps,
):
    if not backend_options:
        benchmark_result = None
        comparison_result = None
        _ = mo.callout(
            "No Gaussian backends are registered in this environment.",
            kind="warn",
        )
    elif (run_button.value or 0) <= 0:
        benchmark_result = None
        comparison_result = None
        _ = mo.callout(
            "Press 'Run Render Benchmark' to benchmark the selected backend.",
            kind="info",
        )
    else:
        root = Path(scene_root.value).expanduser()
        if not root.exists():
            benchmark_result = None
            comparison_result = None
            _ = mo.callout(f"COLMAP root `{root}` does not exist.", kind="warn")
        else:
            resolved_device = (
                torch.device("cuda")
                if device.value == "auto" and torch.cuda.is_available()
                else torch.device(device.value)
                if device.value != "auto"
                else torch.device("cpu")
            )
            if resolved_device.type == "cuda" and not torch.cuda.is_available():
                benchmark_result = None
                comparison_result = None
                _ = mo.callout("CUDA is not available.", kind="warn")
            else:
                scene_record = sk.load_colmap_scene_record(root)
                scene = initialize_gaussian_scene_from_scene_record(
                    scene_record
                ).to(resolved_device)
                camera = select_first_camera(
                    scene_record.resolve_camera_sensor().camera
                ).to(resolved_device)
                benchmark_result = benchmark_backend_render(
                    scene,
                    camera,
                    backend=backend.value,
                    warmup_steps=int(warmup_steps.value),
                    measured_steps=int(measured_steps.value),
                )
                comparison_result = None
                if (
                    compare_backend.value not in ("(none)", backend.value)
                    and compare_backend.value in backend_options
                ):
                    comparison_result = benchmark_backend_render(
                        scene,
                        camera,
                        backend=compare_backend.value,
                        warmup_steps=int(warmup_steps.value),
                        measured_steps=int(measured_steps.value),
                    )
                if comparison_result is None:
                    _ = mo.callout(
                        (
                            f"{benchmark_result.backend}: "
                            f"{benchmark_result.mean_ms_per_frame:.2f} ms/frame, "
                            f"{benchmark_result.fps:.2f} FPS"
                        ),
                        kind="success",
                    )
                else:
                    ratio = (
                        comparison_result.mean_ms_per_frame
                        / benchmark_result.mean_ms_per_frame
                    )
                    faster_backend = (
                        benchmark_result.backend
                        if benchmark_result.mean_ms_per_frame
                        <= comparison_result.mean_ms_per_frame
                        else comparison_result.backend
                    )
                    _ = mo.callout(
                        (
                            f"{benchmark_result.backend}: "
                            f"{benchmark_result.mean_ms_per_frame:.2f} ms/frame\n"
                            f"{comparison_result.backend}: "
                            f"{comparison_result.mean_ms_per_frame:.2f} ms/frame\n"
                            f"ratio vs primary: {ratio:.2f}x\n"
                            f"faster backend: {faster_backend}"
                        ),
                        kind="success",
                    )
    return benchmark_result, comparison_result


@app.cell
def _(benchmark_result, comparison_result):
    payload = None
    if benchmark_result is not None:
        payload = asdict(benchmark_result)
        if comparison_result is not None:
            payload = {
                "primary": asdict(benchmark_result),
                "comparison": asdict(comparison_result),
                "delta_ms_per_frame": (
                    comparison_result.mean_ms_per_frame
                    - benchmark_result.mean_ms_per_frame
                ),
                "ratio_vs_primary": (
                    comparison_result.mean_ms_per_frame
                    / benchmark_result.mean_ms_per_frame
                ),
                "faster_backend": (
                    benchmark_result.backend
                    if benchmark_result.mean_ms_per_frame
                    <= comparison_result.mean_ms_per_frame
                    else comparison_result.backend
                ),
            }
    None if payload is None else mo.md(
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )
    return


if __name__ == "__main__":
    app.run()
