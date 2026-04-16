"""Side-by-side scene comparison notebook with linked viewer navigation."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

with app.setup:
    import ctypes
    import signal
    import sys
    from pathlib import Path
    from typing import Literal

    import marimo as mo
    import numpy as np
    import splatkit as sk
    import splatkit_backends.fastergs as sk_fastergs
    import splatkit_backends.gsplat as sk_gsplat
    import splatkit_backends.svraster as sk_svraster
    import torch
    from marimo_3dv import (
        CameraState,
        RenderResult,
        Viewer,
        ViewerState,
        cleanup_before_splat_reload,
        form_gui,
        link_viewer_states,
    )
    from pydantic import BaseModel
    from splatkit_backends.inria import register as register_inria

    sk_fastergs.register()
    sk_gsplat.register()
    sk_svraster.register()
    register_inria()

    SceneType = Literal["3dgs_ply", "svraster_checkpoint"]

    SCENE_TYPE_OPTIONS = {
        "3DGS PLY": "3dgs_ply",
        "SV Raster Checkpoint": "svraster_checkpoint",
    }

    class GaussianLoadConfig(BaseModel):
        """Parameters for loading a 3DGS PLY scene."""

        path: Path = Path("tmp/example_scene.ply")

    class SVRasterLoadConfig(BaseModel):
        """Parameters for loading an SV Raster checkpoint."""

        run_path: Path = Path("tmp/quarter_res_full_train/garden")
        iteration: int = -1

    class LinkConfig(BaseModel):
        """Viewer-state linking options for the comparison UI."""

        enabled: bool = True
        link_camera: bool = True
        link_show_axes: bool = False
        link_show_horizon: bool = False
        link_show_origin: bool = False
        link_show_stats: bool = False
        aspect_ratio: float = 1.1

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
            "Comparison viewer hit a broken CUDA context while "
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

    def synchronize_after_render(scene: sk.Scene) -> None:
        """Synchronize SV Raster renders so CUDA failures surface promptly."""
        if not isinstance(scene, sk.SparseVoxelScene):
            return
        if scene.octpath.device.type != "cuda":
            return
        try:
            torch.cuda.synchronize(device=scene.octpath.device)
        except Exception as error:
            raise_cuda_context_error("rendering the SV Raster frame", error)

    def scene_class_for_type(scene_type: SceneType) -> type[sk.Scene]:
        """Return the scene class associated with a UI scene-type value."""
        if scene_type == "3dgs_ply":
            return sk.GaussianScene3D
        return sk.SparseVoxelScene

    def available_backends(scene_type: SceneType) -> list[str]:
        """List registered backends compatible with the selected scene type."""
        scene_class = scene_class_for_type(scene_type)
        return sorted(
            backend_name
            for backend_name, registered_backend in sk.BACKEND_REGISTRY.items()
            if any(
                issubclass(scene_class, accepted_scene_type)
                for accepted_scene_type in registered_backend.accepted_scene_types
            )
        )

    def load_scene_artifact(
        scene_type: SceneType,
        load_value: GaussianLoadConfig | SVRasterLoadConfig,
    ) -> sk.Scene | None:
        """Load the selected scene artifact and move it to CUDA when available."""
        if scene_type == "3dgs_ply":
            assert isinstance(load_value, GaussianLoadConfig)
            if not load_value.path.exists():
                return None
            scene = sk.load_gaussian_ply(load_value.path)
        else:
            assert isinstance(load_value, SVRasterLoadConfig)
            if not load_value.run_path.exists():
                return None
            iteration = (
                None if load_value.iteration < 0 else load_value.iteration
            )
            scene = sk.load_svraster_checkpoint(
                load_value.run_path,
                iteration=iteration,
            )

        if torch.cuda.is_available():
            ensure_cuda_context_healthy("moving the scene to CUDA")
            scene = scene.to(torch.device("cuda"))
        return scene

    def camera_device_for_scene(scene: sk.Scene) -> torch.device:
        """Return the device that should own camera tensors for the scene."""
        if isinstance(scene, sk.GaussianScene3D):
            return scene.center_position.device
        if isinstance(scene, sk.SparseVoxelScene):
            return scene.octpath.device
        raise TypeError(f"Unsupported scene type {type(scene).__name__}.")

    active_link = {"handle": None}


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Compare Viewers
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Interactive
    """)
    return


@app.cell
def _(viewers):
    viewers
    return


@app.cell
def _(controls):
    controls
    return


@app.cell
def _():
    left_scene_type = mo.ui.dropdown(
        SCENE_TYPE_OPTIONS,
        value="3DGS PLY",
        label="Left scene type",
        full_width=True,
    )
    right_scene_type = mo.ui.dropdown(
        SCENE_TYPE_OPTIONS,
        value="SV Raster Checkpoint",
        label="Right scene type",
        full_width=True,
    )
    return left_scene_type, right_scene_type


@app.cell
def _():
    left_gaussian_form = form_gui(
        GaussianLoadConfig,
        value=GaussianLoadConfig(),
        label="Left 3DGS Scene",
        live_update=False,
    )
    right_gaussian_form = form_gui(
        GaussianLoadConfig,
        value=GaussianLoadConfig(),
        label="Right 3DGS Scene",
        live_update=False,
    )
    left_svraster_form = form_gui(
        SVRasterLoadConfig,
        value=SVRasterLoadConfig(),
        label="Left SV Raster Scene",
        live_update=False,
    )
    right_svraster_form = form_gui(
        SVRasterLoadConfig,
        value=SVRasterLoadConfig(),
        label="Right SV Raster Scene",
        live_update=False,
    )
    link_form = form_gui(
        LinkConfig,
        value=LinkConfig(),
        label="Linked Navigation",
        live_update=True,
    )
    return (
        left_gaussian_form,
        left_svraster_form,
        link_form,
        right_gaussian_form,
        right_svraster_form,
    )


@app.cell
def _(left_scene_type):
    left_backend_options = available_backends(left_scene_type.value)
    left_backend = mo.ui.dropdown(
        left_backend_options,
        value=left_backend_options[0] if left_backend_options else None,
        label="Left backend",
        full_width=True,
    )
    return (left_backend,)


@app.cell
def _(right_scene_type):
    right_backend_options = available_backends(right_scene_type.value)
    right_backend = mo.ui.dropdown(
        right_backend_options,
        value=right_backend_options[0] if right_backend_options else None,
        label="Right backend",
        full_width=True,
    )
    return (right_backend,)


@app.cell
def _(left_backend, left_gaussian_form, left_scene_type, left_svraster_form):
    left_selected_form = (
        left_gaussian_form
        if left_scene_type.value == "3dgs_ply"
        else left_svraster_form
    )
    left_controls = mo.vstack(
        [left_scene_type, left_backend, left_selected_form],
        gap=0.75,
    )
    return (left_controls,)


@app.cell
def _(
    right_backend,
    right_gaussian_form,
    right_scene_type,
    right_svraster_form,
):
    right_selected_form = (
        right_gaussian_form
        if right_scene_type.value == "3dgs_ply"
        else right_svraster_form
    )
    right_controls = mo.vstack(
        [right_scene_type, right_backend, right_selected_form],
        gap=0.75,
    )
    return (right_controls,)


@app.cell
def _():
    left_viewer_state = ViewerState(camera_convention="opencv")
    right_viewer_state = ViewerState(camera_convention="opencv")
    return left_viewer_state, right_viewer_state


@app.cell
def _(link_form):
    viewer_aspect_ratio = link_form.value.aspect_ratio
    return (viewer_aspect_ratio,)


@app.cell
def _(
    left_gaussian_form,
    left_scene_type,
    left_svraster_form,
    left_viewer_state,
):
    cleanup_before_splat_reload(
        left_viewer_state,
        close_existing_viewer=True,
        empty_cuda_cache=True,
    )
    left_load_value = (
        left_gaussian_form.value
        if left_scene_type.value == "3dgs_ply"
        else left_svraster_form.value
    )
    left_scene = load_scene_artifact(
        left_scene_type.value,
        left_load_value,
    )
    return (left_scene,)


@app.cell
def _(
    right_gaussian_form,
    right_scene_type,
    right_svraster_form,
    right_viewer_state,
):
    cleanup_before_splat_reload(
        right_viewer_state,
        close_existing_viewer=True,
        empty_cuda_cache=True,
    )
    right_load_value = (
        right_gaussian_form.value
        if right_scene_type.value == "3dgs_ply"
        else right_svraster_form.value
    )
    right_scene = load_scene_artifact(
        right_scene_type.value,
        right_load_value,
    )
    return (right_scene,)


@app.function
@torch.no_grad()
def rasterize_scene(
    camera: CameraState,
    scene: sk.Scene | None,
    *,
    backend: str,
) -> RenderResult:
    """Render one scene through the selected splatkit backend."""
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
        backend_camera.to(camera_device_for_scene(scene)),
        backend=backend,
    )
    synchronize_after_render(scene)
    image = render_output.render[0].clamp(0.0, 1.0).cpu().numpy()
    return RenderResult(image=(image * 255).astype(np.uint8))


@app.cell
def _(left_backend, left_scene, left_viewer_state, viewer_aspect_ratio):
    left_viewer_state.aspect_ratio = viewer_aspect_ratio

    def render_left(camera_state):
        return rasterize_scene(
            camera_state,
            left_scene,
            backend=left_backend.value,
        ).image

    left_viewer = Viewer(render_left, state=left_viewer_state)
    return (left_viewer,)


@app.cell
def _(right_backend, right_scene, right_viewer_state, viewer_aspect_ratio):
    right_viewer_state.aspect_ratio = viewer_aspect_ratio

    def render_right(camera_state):
        return rasterize_scene(
            camera_state,
            right_scene,
            backend=right_backend.value,
        ).image

    right_viewer = Viewer(render_right, state=right_viewer_state)
    return (right_viewer,)


@app.cell
def _(left_viewer_state, link_form, right_viewer_state):
    if active_link["handle"] is not None:
        active_link["handle"].close()
        active_link["handle"] = None

    selected_fields: list[str] = []
    if link_form.value.link_camera:
        selected_fields.append("camera_state")
    if link_form.value.link_show_axes:
        selected_fields.append("show_axes")
    if link_form.value.link_show_horizon:
        selected_fields.append("show_horizon")
    if link_form.value.link_show_origin:
        selected_fields.append("show_origin")
    if link_form.value.link_show_stats:
        selected_fields.append("show_stats")

    if link_form.value.enabled and selected_fields:
        active_link["handle"] = link_viewer_states(
            left_viewer_state,
            right_viewer_state,
            fields=tuple(selected_fields),
            bidirectional=True,
        )
    return


@app.cell
def _(left_controls, link_form, right_controls):
    controls = mo.vstack(
        [
            link_form,
            mo.hstack(
                [left_controls, right_controls],
                widths="equal",
                align="start",
                gap=1.0,
            ),
        ],
        gap=1.0,
    )

    return (controls,)


@app.cell
def _(left_viewer, right_viewer):
    viewers = mo.hstack(
        [left_viewer, right_viewer],
        widths="equal",
        align="start",
        gap=1.0,
    )
    return (viewers,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
