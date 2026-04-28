# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.0",
#     "marimo-3dv",
#     "numpy==2.4.4",
#     "gsplat==1.5.3",
#     "torch==2.11.0",
# ]
#
# [tool.uv.sources]
# marimo-3dv = { path = "..", editable = true }
# ///

"""Minimal Gaussian splat viewer notebook."""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    from gsplat import rasterization

    from marimo_3dv import (
        CameraState,
        RenderResult,
        SplatScene,
        Viewer,
        ViewerState,
        apply_viewer_pipeline_config,
        gs_backend_bundle,
        load_splat_scene_from_config,
        pick_splat_load_config,
        splat_load_form,
        viewer_pipeline_controls_gui,
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Splat Viewer
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Interactive Result
    """)
    return


@app.cell
def _(viewer_controls_gui):
    viewer_controls_gui
    return


@app.cell
def _(viewer):
    viewer
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Scene
    """)
    return


@app.cell
def _():
    viewer_state = ViewerState(camera_convention="opencv")
    return (viewer_state,)


@app.cell
def _():
    load_form = splat_load_form()
    load_form
    return (load_form,)


@app.cell
def _(load_form, viewer_state):
    if mo.running_in_notebook():
        scene = load_splat_scene_from_config(load_form.value, viewer_state)
    else:
        load_config = pick_splat_load_config()
        scene = load_splat_scene_from_config(load_config, viewer_state)
    return (scene,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Rendering
    """)
    return


@app.function
@torch.no_grad()
def rasterize_scene(
    camera: CameraState, scene: SplatScene | None
) -> RenderResult:
    """Render a splat scene using gsplat."""
    if scene is None:
        return RenderResult(
            image=np.full((camera.height, camera.width, 3), 245, dtype=np.uint8)
        )

    gsplat_camera = camera.with_convention("opencv")
    device = scene.center_positions.device
    w2c = np.linalg.inv(gsplat_camera.cam_to_world)
    viewmats = torch.from_numpy(w2c).to(device=device, dtype=torch.float32)[
        None
    ]

    half_fov_rad = np.radians(gsplat_camera.fov_degrees / 2)
    focal = (gsplat_camera.height / 2) / np.tan(half_fov_rad)
    intrinsics = torch.tensor(
        [
            [focal, 0.0, gsplat_camera.width / 2],
            [0.0, focal, gsplat_camera.height / 2],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )[None]

    render_colors, _render_alphas, metadata = rasterization(
        means=scene.center_positions,
        quats=scene.quaternion_orientation,
        scales=torch.exp(scene.log_half_extents),
        opacities=torch.sigmoid(scene.opacity_logits.squeeze(-1)),
        colors=scene.spherical_harmonics,
        viewmats=viewmats,
        Ks=intrinsics,
        width=gsplat_camera.width,
        height=gsplat_camera.height,
        sh_degree=scene.sh_degree,
    )
    image = render_colors[0].clamp(0.0, 1.0).cpu().numpy()
    image_uint8 = (image * 255).astype(np.uint8)

    metadata_out: dict = {}
    if "means2d" in metadata:
        metadata_out["projected_means"] = metadata["means2d"][0]

    return RenderResult(image=image_uint8, metadata=metadata_out)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Viewer
    """)
    return


@app.cell
def _():
    backend_bundle = gs_backend_bundle()
    viewer_pipeline = backend_bundle.pipeline()
    return backend_bundle, viewer_pipeline


@app.cell
def _(scene, viewer_pipeline, viewer_state):
    pipeline_result = viewer_pipeline.build(scene, viewer_state)
    return (pipeline_result,)


@app.cell
def _(backend_bundle, pipeline_result, viewer_state):
    viewer_controls = viewer_pipeline_controls_gui(
        viewer_state,
        pipeline_result,
        viewer_default_config=backend_bundle.viewer_controls(viewer_state),
    )
    viewer_controls_gui = viewer_controls.gui
    viewer_controls_default_config = viewer_controls.default_config
    return (viewer_controls_gui,)


@app.cell
def _(pipeline_result, viewer_controls_gui, viewer_state):
    cache = {"config_json": None, "render_fn": None}

    def render_frame(camera_state):
        combined_config = viewer_controls_gui.value
        config_json = combined_config.model_dump_json()
        if cache["render_fn"] is None or config_json != cache["config_json"]:
            pipeline_config = apply_viewer_pipeline_config(
                viewer_state,
                combined_config,
            )
            cache["render_fn"] = pipeline_result.bind(
                pipeline_config,
                backend_fn=rasterize_scene,
            )
            cache["config_json"] = config_json
        return cache["render_fn"](camera_state).image

    viewer = Viewer(
        render_frame,
        state=viewer_state,
        controls=viewer_controls_gui,
    )
    return (viewer,)


if __name__ == "__main__":
    app.run()
