"""Minimal SV Raster viewer notebook."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="columns")

with app.setup:
    import importlib
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
    )
    from marimo_config_gui import form_gui
    from pydantic import BaseModel

    _adapter_spec = importlib.util.find_spec("new_svraster_cuda")
    if _adapter_spec is None:
        raw_svraster_renderer = None
    else:
        raw_svraster_renderer = importlib.import_module(
            "new_svraster_cuda.renderer"
        )

    sk_svraster.register()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # SV Raster Viewer
    """)
    return


@app.cell
def _(viewer):
    viewer
    return


@app.cell
def _(load_form):
    load_form
    return


@app.cell
def _():
    options = {"Native (`svraster.core`)": "svraster.core"}
    if raw_svraster_renderer is not None:
        options["Adapter (`new_svraster_cuda`)"] = "adapter.svraster"
    backend_selector = mo.ui.dropdown(
        options=options,
        value="Native (`svraster.core`)",
        label="Render Backend",
        full_width=True,
    )
    return (backend_selector,)


@app.cell
def _(backend_selector):
    adapter_status = None
    if raw_svraster_renderer is None:
        adapter_status = mo.md(
            """
            Adapter mode is unavailable.
            Install the optional `new-svraster-cuda` package to enable
            `adapter.svraster` comparison in this notebook.
            """
        )
        adapter_status
    backend_selector
    return


@app.cell(column=1)
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Scene Controls
    """)
    return


@app.class_definition
class LoadConfig(BaseModel):
    """Notebook controls for loading an SVRaster run directory."""

    run_path: Path = Path("tmp/quarter_res_full_train/garden")
    iteration: int = -1


@app.cell
def _():
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
    ## Viewer State
    """)
    return


@app.cell
def _():
    viewer_state = ViewerState(camera_convention="opencv")
    return (viewer_state,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Viewer
    """)
    return


@app.function
def blank_frame(camera: CameraState) -> RenderResult:
    """Return a neutral placeholder frame for empty scenes."""
    return RenderResult(
        image=np.full(
            (camera.height, camera.width, 3),
            245,
            dtype=np.uint8,
        )
    )


@app.function
def build_backend_camera(camera: CameraState) -> sk.CameraState:
    """Convert a viewer camera into the splatkit camera contract."""
    return sk.CameraState(
        width=torch.tensor([camera.width], dtype=torch.int64),
        height=torch.tensor([camera.height], dtype=torch.int64),
        fov_degrees=torch.tensor([camera.fov_degrees], dtype=torch.float32),
        cam_to_world=torch.from_numpy(
            camera.with_convention("opencv").cam_to_world
        ).to(dtype=torch.float32)[None],
        camera_convention="opencv",
    )


@app.function
def build_adapter_raster_settings(
    scene: sk.SparseVoxelScene,
    camera: sk.CameraState,
) -> object:
    """Build low-level SVRaster settings for the adapter path."""
    if raw_svraster_renderer is None:
        raise RuntimeError(
            "adapter.svraster requires the optional new_svraster_cuda package."
        )
    intrinsics = camera.get_intrinsics()[0]
    width = int(camera.width[0].item())
    height = int(camera.height[0].item())
    fx = float(intrinsics[0, 0].item())
    fy = float(intrinsics[1, 1].item())
    cam_to_world = camera.cam_to_world[0].to(
        device=scene.octpath.device,
        dtype=torch.float32,
    )
    return raw_svraster_renderer.RasterSettings(
        color_mode="sh",
        n_samp_per_vox=1,
        image_width=width,
        image_height=height,
        tanfovx=(width * 0.5) / fx,
        tanfovy=(height * 0.5) / fy,
        cx=float(intrinsics[0, 2].item()),
        cy=float(intrinsics[1, 2].item()),
        w2c_matrix=torch.linalg.inv(cam_to_world),
        c2w_matrix=cam_to_world,
        bg_color=0.0,
        near=0.02,
        need_depth=False,
    )


@app.function
@torch.no_grad()
def render_adapter_scene(
    scene: sk.SparseVoxelScene,
    camera: sk.CameraState,
) -> torch.Tensor:
    """Render through the low-level SVRaster adapter path."""
    if raw_svraster_renderer is None:
        raise RuntimeError(
            "adapter.svraster requires the optional new_svraster_cuda package."
        )
    geos = scene.voxel_geometries

    def vox_fn(
        _idx: torch.Tensor,
        cam_pos: torch.Tensor,
        _color_mode: str,
    ) -> dict[str, torch.Tensor]:
        rgbs = raw_svraster_renderer.SH_eval.apply(
            scene.active_sh_degree,
            None,
            scene.vox_center,
            cam_pos,
            None,
            scene.sh0,
            scene.shs,
        )
        subdiv_p = torch.ones(
            (scene.num_voxels, 1),
            dtype=scene.sh0.dtype,
            device=scene.sh0.device,
        )
        return {"geos": geos, "rgbs": rgbs, "subdiv_p": subdiv_p}

    color, _depth, _normal, _transmittance, _max_w = (
        raw_svraster_renderer.rasterize_voxels(
            build_adapter_raster_settings(scene, camera),
            scene.octpath.reshape(-1),
            scene.vox_center,
            scene.vox_size.reshape(-1),
            vox_fn,
        )
    )
    return color.permute(1, 2, 0).contiguous().clamp(0.0, 1.0)


@app.function
@torch.no_grad()
def rasterize_scene(
    camera: CameraState,
    scene: sk.SparseVoxelScene | None,
    *,
    backend: str,
) -> RenderResult:
    """Render a sparse-voxel scene through the selected backend."""
    if scene is None:
        return blank_frame(camera)

    backend_camera = build_backend_camera(camera).to(scene.octpath.device)
    if backend == "adapter.svraster":
        image = render_adapter_scene(scene, backend_camera).cpu().numpy()
    else:
        render_output = sk.render(
            scene,
            backend_camera,
            backend="svraster.core",
        )
        image = render_output.render[0].clamp(0.0, 1.0).cpu().numpy()
    return RenderResult(image=(image * 255).astype(np.uint8))


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
        scene = scene.to(torch.device("cuda"))
    return (scene,)


@app.cell
def _(backend_selector, scene, viewer_state):
    def render_frame(camera_state):
        return rasterize_scene(
            camera_state,
            scene,
            backend=backend_selector.value,
        ).image

    viewer = Viewer(render_frame, state=viewer_state)
    return (viewer,)


if __name__ == "__main__":
    app.run()
