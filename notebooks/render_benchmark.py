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

    import marimo as mo
    import splatkit as sk
    import torch
    from splatkit.benchmarks import benchmark_backend_render
    from splatkit.core import BACKEND_REGISTRY, GaussianScene3D
    from splatkit.initialization import initialize_gaussian_scene_from_scene_record

    OPTIONAL_BACKEND_MODULES = (
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
            "SPLATKIT_COLMAP_ROOT",
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
    return backend, backend_options


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
def _(backend, device, measured_steps, run_button, scene_root, warmup_steps):
    controls = mo.vstack(
        [
            scene_root,
            mo.hstack([backend, device], widths="equal"),
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
    device,
    measured_steps,
    run_button,
    scene_root,
    warmup_steps,
):
    if not backend_options:
        benchmark_result = None
        _ = mo.callout(
            "No Gaussian backends are registered in this environment.",
            kind="warn",
        )
    elif (run_button.value or 0) <= 0:
        benchmark_result = None
        _ = mo.callout(
            "Press 'Run Render Benchmark' to benchmark the selected backend.",
            kind="info",
        )
    else:
        root = Path(scene_root.value).expanduser()
        if not root.exists():
            benchmark_result = None
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
                _ = mo.callout("CUDA is not available.", kind="warn")
            else:
                scene_record = sk.load_colmap_scene_record(root)
                scene = initialize_gaussian_scene_from_scene_record(
                    scene_record
                ).to(
                    resolved_device
                )
                camera = select_first_camera(
                    scene_record.resolve_camera_sensor().camera
                ).to(
                    resolved_device
                )
                benchmark_result = benchmark_backend_render(
                    scene,
                    camera,
                    backend=backend.value,
                    warmup_steps=int(warmup_steps.value),
                    measured_steps=int(measured_steps.value),
                )
                _ = mo.callout(
                    (
                        f"{benchmark_result.backend}: "
                        f"{benchmark_result.mean_ms_per_frame:.2f} ms/frame, "
                        f"{benchmark_result.fps:.2f} FPS"
                    ),
                    kind="success",
                )
    return (benchmark_result,)


@app.cell
def _(benchmark_result):
    None if benchmark_result is None else mo.md(
        f"```json\n{json.dumps(asdict(benchmark_result), indent=2)}\n```"
    )
    return


if __name__ == "__main__":
    app.run()
