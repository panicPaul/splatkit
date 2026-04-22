"""Minimal SV Raster viewer notebook."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import ctypes
    import signal
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import splatkit as sk
    import splatkit_native_svraster.svraster as sk_svraster
    import torch
    from marimo_3dv import (
        CameraState,
        RenderResult,
        Viewer,
        ViewerState,
        cleanup_before_splat_reload,
        form_gui,
    )
    from pydantic import BaseModel

    sk_svraster.register()

    def install_linux_parent_death_signal() -> None:
        """Ensure the viewer subprocess exits when the parent process dies."""
        if sys.platform != "linux":
            return

        libc = ctypes.CDLL(None, use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0)

    install_linux_parent_death_signal()

    def raise_cuda_context_error(stage: str, error: BaseException) -> None:
        """Raise a consistent error for asynchronous CUDA context failures."""
        raise RuntimeError(
            "SV Raster viewer hit a broken CUDA context while "
            f"{stage}. This usually means an earlier CUDA kernel failed "
            "asynchronously. Restart the marimo kernel before retrying. "
            "For debugging, re-run with CUDA_LAUNCH_BLOCKING=1."
        ) from error

    def ensure_cuda_context_healthy(stage: str) -> None:
        """Synchronize CUDA to surface any pending asynchronous failures."""
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception as error:
            raise_cuda_context_error(stage, error)

    def synchronize_after_render(scene: sk.SparseVoxelScene) -> None:
        """Synchronize SV Raster renders so CUDA failures surface promptly."""
        if scene.octpath.device.type != "cuda":
            return
        try:
            torch.cuda.synchronize(device=scene.octpath.device)
        except Exception as error:
            raise_cuda_context_error("rendering the SV Raster frame", error)


@app.cell
def _():
    class LoadConfig(BaseModel):
        run_path: Path = Path("tmp/quarter_res_full_train/garden")
        iteration: int = -1

    load_form = form_gui(
        LoadConfig,
        value=LoadConfig(),
        label="SV Raster Scene",
        live_update=False,
    )
    return (load_form,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    # SV Raster Viewer
    """)
    return


@app.cell
def _(load_form):
    load_form
    return


@app.cell
def _():
    viewer_state = ViewerState(camera_convention="opencv")
    return (viewer_state,)


@app.cell
def _(load_form, viewer_state):
    iteration = (
        None if load_form.value.iteration < 0 else load_form.value.iteration
    )
    cleanup_before_splat_reload(
        viewer_state,
        close_existing_viewer=True,
        empty_cuda_cache=True,
    )
    scene = sk.load_svraster_checkpoint(
        load_form.value.run_path,
        iteration=iteration,
    )
    if torch.cuda.is_available():
        ensure_cuda_context_healthy("moving the sparse-voxel scene to CUDA")
        scene = scene.to(torch.device("cuda"))
    return (scene,)


@app.function
@torch.no_grad()
def rasterize_scene(
    camera: CameraState,
    scene: sk.SparseVoxelScene | None,
) -> RenderResult:
    """Render a sparse-voxel scene through splatkit."""
    if scene is None:
        return RenderResult(
            image=np.full((camera.height, camera.width, 3), 245, dtype=np.uint8)
        )

    backend_camera = sk.CameraState(
        width=torch.tensor([camera.width], dtype=torch.int64),
        height=torch.tensor([camera.height], dtype=torch.int64),
        fov_degrees=torch.tensor([camera.fov_degrees], dtype=torch.float32),
        cam_to_world=torch.from_numpy(
            camera.with_convention("opencv").cam_to_world
        ).to(dtype=torch.float32)[None],
        camera_convention="opencv",
    )
    render_output = sk.render(
        scene,
        backend_camera.to(scene.octpath.device),
        backend="svraster.core",
    )
    synchronize_after_render(scene)
    image = render_output.render[0].clamp(0.0, 1.0).cpu().numpy()
    return RenderResult(image=(image * 255).astype(np.uint8))


@app.cell
def _(scene, viewer_state):
    def render_frame(camera_state):
        return rasterize_scene(camera_state, scene).image

    viewer = Viewer(render_frame, state=viewer_state)
    return (viewer,)


@app.cell
def _(viewer):
    viewer
    return


if __name__ == "__main__":
    app.run()
