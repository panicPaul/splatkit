"""Minimal Gaussian splat viewer notebook."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import ctypes
    import signal
    import sys
    from typing import Literal

    import marimo as mo
    import numpy as np
    import splatkit as sk
    import splatkit_backends.fastergs as sk_fastergs
    import splatkit_backends.gsplat as sk_gsplat
    import splatkit_backends.stoch3dgs as sk_stoch
    import torch
    from marimo_3dv import (
        CameraState,
        RenderResult,
        SplatScene,
        Viewer,
        ViewerState,
        apply_viewer_pipeline_config,
        cleanup_before_splat_reload,
        form_gui,
        gs_backend_bundle,
        pick_splat_load_config,
        splat_load_form,
        viewer_pipeline_controls_gui,
    )
    from pydantic import BaseModel
    from splatkit_backends.inria import register as register_inria

    sk_fastergs.register()
    sk_gsplat.register()
    sk_stoch.register()
    register_inria()

    def _install_linux_parent_death_signal() -> None:
        if sys.platform != "linux":
            return

        libc = ctypes.CDLL(None, use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0)

    _install_linux_parent_death_signal()


@app.cell
def _():
    class BackendConfig(BaseModel):
        backend: Literal["gsplat", "inria", "fastergs", "stoch3dgs"] = "gsplat"

    backend_form = form_gui(
        BackendConfig,
        value=BackendConfig(),
        label="Backend",
        live_update=True,
    )
    return (backend_form,)


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
def _(backend_form, viewer_controls_gui):
    mo.vstack([backend_form, viewer_controls_gui])
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


@app.function
def load_gaussian_scene(path):
    """Load a Gaussian scene and move it to CUDA when available."""
    backend_scene = sk.load_gaussian_ply(path)
    if torch.cuda.is_available():
        backend_scene = backend_scene.to(torch.device("cuda"))
    if backend_scene.feature.ndim != 3:
        raise ValueError(
            "Gaussian viewer expects SH coefficients with shape "
            "(num_splats, num_bases, 3)."
        )
    return SplatScene(
        center_positions=backend_scene.center_position,
        log_half_extents=backend_scene.log_scales,
        quaternion_orientation=backend_scene.quaternion_orientation,
        spherical_harmonics=backend_scene.feature,
        opacity_logits=backend_scene.logit_opacity[:, None],
        sh_degree=backend_scene.sh_degree,
    )


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
        cleanup_before_splat_reload(
            viewer_state,
            close_existing_viewer=True,
            empty_cuda_cache=True,
        )
        scene = load_gaussian_scene(load_form.value.ply_path)
    else:
        load_config = pick_splat_load_config()
        scene = (
            None
            if load_config is None
            else load_gaussian_scene(load_config.ply_path)
        )
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
    camera: CameraState,
    scene: SplatScene | None,
    *,
    backend: Literal["gsplat", "inria", "fastergs", "stoch3dgs"] = "gsplat",
) -> RenderResult:
    """Render a splat scene through splatkit."""
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
    backend_scene = sk.GaussianScene3D(
        center_position=scene.center_positions,
        log_scales=scene.log_half_extents,
        quaternion_orientation=scene.quaternion_orientation,
        logit_opacity=scene.opacity_logits.squeeze(-1),
        feature=scene.spherical_harmonics,
        sh_degree=scene.sh_degree,
    )
    render_kwargs = {
        "backend": backend,
        "return_2d_projections": backend == "gsplat",
    }
    render_output = sk.render(
        backend_scene,
        backend_camera.to(scene.center_positions.device),
        **render_kwargs,
    )
    image = render_output.render[0].clamp(0.0, 1.0).cpu().numpy()
    image_uint8 = (image * 255).astype(np.uint8)

    metadata = {}
    if backend == "gsplat":
        metadata = {
            "projected_means": render_output.projected_means[0],
            "projected_conics": render_output.projected_conics[0],
        }
    return RenderResult(image=image_uint8, metadata=metadata)


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
def _(backend_form, pipeline_result, viewer_controls_gui, viewer_state):
    cache = {"config_json": None, "render_fn": None}

    def render_frame(camera_state):
        combined_config = viewer_controls_gui.value
        backend = backend_form.value.backend
        config_json = f"{backend}:{combined_config.model_dump_json()}"
        if cache["render_fn"] is None or config_json != cache["config_json"]:
            pipeline_config = apply_viewer_pipeline_config(
                viewer_state,
                combined_config,
            )
            cache["render_fn"] = pipeline_result.bind(
                pipeline_config,
                backend_fn=lambda camera, compiled_view: rasterize_scene(
                    camera,
                    compiled_view,
                    backend=backend,
                ),
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
