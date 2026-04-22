"""Gaussian POP viewer notebook with score distribution chart."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import altair as alt
    import marimo as mo
    import numpy as np
    import splatkit as sk
    import splatkit_native_backends.gaussian_pop_native as skn_gaussian_pop
    import threading
    import torch
    from marimo_3dv import (
        CameraState,
        RenderResult,
        SplatLoadConfig,
        SplatScene,
        Viewer,
        ViewerState,
        apply_viewer_pipeline_config,
        cleanup_before_splat_reload,
        gs_backend_bundle,
        pick_splat_load_config,
        viewer_pipeline_controls_gui,
    )
    from marimo_config_gui import (
        config_error,
        config_form,
        config_value,
        create_config_state,
    )

    skn_gaussian_pop.register()


@app.cell
def _():
    view_mode_options = ["image", "depth"]
    colormap_options = [
        "viridis",
        "turbo",
        "magma",
        "inferno",
        "cividis",
        "gray",
    ]
    return colormap_options, view_mode_options


@app.cell
def _(colormap_options):
    colormap = mo.ui.dropdown(
        colormap_options,
        value="viridis",
        label="Colormap",
        full_width=True,
    )
    return (colormap,)


@app.cell
def _():
    normalization_percent = mo.ui.slider(
        start=50,
        stop=100,
        step=1,
        value=90,
        label="Quantile range (%)",
    )
    return (normalization_percent,)


@app.cell
def _():
    normalization_bias = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.5,
        label="Quantile bias",
    )
    return (normalization_bias,)


@app.cell
def _():
    invert_colormap = mo.ui.checkbox(
        value=False,
        label="Invert colormap",
    )
    return (invert_colormap,)


@app.cell
def _(view_mode_options):
    view_mode = mo.ui.dropdown(
        view_mode_options,
        value=view_mode_options[0],
        label="View mode",
        full_width=True,
    )
    return (view_mode,)


@app.cell
def _():
    score_bucket_count = mo.ui.slider(
        start=5,
        stop=200,
        step=1,
        value=50,
        label="Score buckets",
    )
    return (score_bucket_count,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Gaussian POP Viewer
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Interactive Result
    """)
    return


@app.cell
def _(
    colormap,
    invert_colormap,
    normalization_bias,
    normalization_percent,
    score_bucket_count,
    view_mode,
    viewer_controls_gui,
):
    selectors = mo.hstack(
        [view_mode, colormap, score_bucket_count],
        widths="equal",
        align="start",
        gap=1.0,
    )
    normalization_controls = mo.hstack(
        [normalization_percent, normalization_bias, invert_colormap],
        widths="equal",
        align="start",
        gap=1.0,
    )
    mo.vstack(
        [selectors, normalization_controls, viewer_controls_gui],
        gap=0.75,
    )
    return


@app.cell
def _(viewer):
    viewer
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Score Distribution
    """)
    return


@app.cell
def _():
    score_refresh = mo.ui.refresh(
        options=[0.25, 0.5, 1.0, 2.0],
        default_interval=0.5,
        label="Live refresh (s)",
    )
    return (score_refresh,)


@app.cell
def _(score_histogram, score_refresh):
    mo.vstack([score_refresh, score_histogram], gap=0.75)
    return


@app.cell
def _(load_error, load_form):
    mo.vstack([load_error, load_form])
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


@app.function
def apply_colormap(
    values,
    *,
    colormap: str = "viridis",
    invert: bool = False,
):
    """Map normalized values in [0, 1] to an RGB uint8 image."""
    from matplotlib import colormaps

    clipped = np.clip(values, 0.0, 1.0)
    cmap = colormaps[colormap]
    if invert:
        cmap = cmap.reversed()
    colored = cmap(clipped)[..., :3]
    return (colored * 255).astype(np.uint8)


@app.function
def normalize_scalar_field(
    field: torch.Tensor,
    *,
    quantile_percent: float = 90.0,
    quantile_bias: float = 0.5,
    invert: bool = False,
    require_positive: bool = False,
):
    """Normalize a scalar field to [0, 1] using a biased quantile window."""
    values = field.detach().cpu().numpy()
    valid = np.isfinite(values)
    if require_positive:
        valid &= values > 0.0
    if not np.any(valid):
        return np.zeros_like(values, dtype=np.float32)

    window_fraction = float(np.clip(quantile_percent, 1.0, 100.0)) / 100.0
    bias = float(np.clip(quantile_bias, 0.0, 1.0))
    remaining_fraction = 1.0 - window_fraction
    lower_quantile = remaining_fraction * bias
    upper_quantile = lower_quantile + window_fraction
    lower = float(np.quantile(values[valid], lower_quantile))
    upper = float(np.quantile(values[valid], upper_quantile))

    normalized = np.zeros_like(values, dtype=np.float32)
    if upper - lower < 1e-6:
        normalized[valid] = 0.5
    else:
        normalized[valid] = (
            (values[valid] - lower) / (upper - lower)
        ).astype(np.float32)
    normalized = np.clip(normalized, 0.0, 1.0)
    if invert:
        normalized[valid] = 1.0 - normalized[valid]
    return normalized


@app.function
def depth_to_image(
    depth: torch.Tensor,
    *,
    colormap: str = "viridis",
    quantile_percent: float = 90.0,
    quantile_bias: float = 0.5,
    invert_colormap: bool = False,
):
    """Convert a depth map to a colormapped RGB uint8 image."""
    return apply_colormap(
        normalize_scalar_field(
            depth,
            quantile_percent=quantile_percent,
            quantile_bias=quantile_bias,
            invert=True,
            require_positive=True,
        ),
        colormap=colormap,
        invert=invert_colormap,
    )


@app.function
def score_histogram_chart(score_values, *, num_buckets: int = 50):
    """Build an Altair histogram for the current Gaussian impact scores."""
    if score_values is None:
        return mo.callout(
            "Move the viewer to render the current POP scores.",
            kind="info",
        )

    flattened = np.asarray(score_values, dtype=np.float32).reshape(-1)
    finite_scores = flattened[np.isfinite(flattened)]
    if finite_scores.size == 0:
        return mo.callout(
            "The current render did not produce any finite Gaussian scores.",
            kind="warn",
        )

    counts, bin_edges = np.histogram(
        finite_scores,
        bins=max(1, int(num_buckets)),
    )
    rows = [
        {
            "score_start": float(bin_edges[index]),
            "score_end": float(bin_edges[index + 1]),
            "count": int(counts[index]),
        }
        for index in range(len(counts))
    ]
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(color="#0f766e")
        .encode(
            x=alt.X("score_start:Q", title="Score"),
            x2="score_end:Q",
            y=alt.Y("count:Q", title="Number of gaussians"),
            tooltip=[
                alt.Tooltip("score_start:Q", title="Score from"),
                alt.Tooltip("score_end:Q", title="Score to"),
                alt.Tooltip("count:Q", title="Gaussians"),
            ],
        )
        .properties(height=320)
    )
    return mo.vstack(
        [
            mo.md(
                f"Showing `{finite_scores.size:,}` Gaussian POP scores "
                "from the latest viewer render."
            ),
            chart,
        ],
        gap=0.5,
    )


@app.cell
def _():
    viewer_state = ViewerState(camera_convention="opencv")
    return (viewer_state,)


@app.cell
def _():
    (
        load_form_gui_state,
        load_json_gui_state,
        load_bindings,
    ) = create_config_state(
        SplatLoadConfig,
        value=SplatLoadConfig(),
    )
    return load_bindings, load_form_gui_state, load_json_gui_state


@app.cell
def _(load_bindings, load_form_gui_state):
    load_form = config_form(
        load_bindings,
        form_gui_state=load_form_gui_state,
        label="Scene",
    )
    return (load_form,)


@app.cell
def _(load_bindings, load_form_gui_state, load_json_gui_state):
    load_error = config_error(
        load_bindings,
        form_gui_state=load_form_gui_state,
        json_gui_state=load_json_gui_state,
    )
    return (load_error,)


@app.cell
def _(load_bindings, load_form_gui_state, load_json_gui_state):
    load_config = config_value(
        load_bindings,
        form_gui_state=load_form_gui_state,
        json_gui_state=load_json_gui_state,
    )
    return (load_config,)


@app.cell
def _(load_config, viewer_state):
    if mo.running_in_notebook():
        cleanup_before_splat_reload(
            viewer_state,
            close_existing_viewer=True,
            empty_cuda_cache=True,
        )
        scene = (
            None
            if load_config is None
            else load_gaussian_scene(load_config.ply_path)
        )
    else:
        picked_load_config = pick_splat_load_config()
        scene = (
            None
            if picked_load_config is None
            else load_gaussian_scene(picked_load_config.ply_path)
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
    view_mode: str = "image",
    colormap: str = "viridis",
    quantile_percent: float = 90.0,
    quantile_bias: float = 0.5,
    invert_colormap: bool = False,
):
    """Render a splat scene through the Gaussian POP backend."""
    if scene is None:
        return (
            RenderResult(
                image=np.full(
                    (camera.height, camera.width, 3),
                    245,
                    dtype=np.uint8,
                )
            ),
            None,
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
    render_output = sk.render(
        backend_scene,
        backend_camera.to(scene.center_positions.device),
        backend="gaussian_pop_native",
        return_depth=view_mode == "depth",
        return_gaussian_impact_score=True,
    )
    score_values = (
        render_output.gaussian_impact_score[0]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )
    if view_mode == "depth":
        image_uint8 = depth_to_image(
            render_output.depth[0],
            colormap=colormap,
            quantile_percent=quantile_percent,
            quantile_bias=quantile_bias,
            invert_colormap=invert_colormap,
        )
    else:
        image = render_output.render[0].clamp(0.0, 1.0).cpu().numpy()
        image_uint8 = (image * 255).astype(np.uint8)

    return RenderResult(image=image_uint8), score_values


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
    return (viewer_controls_gui,)


@app.cell
def _():
    score_store = {"values": None}
    score_store_lock = threading.Lock()
    return score_store, score_store_lock


@app.cell
def _(
    colormap,
    invert_colormap,
    normalization_bias,
    normalization_percent,
    pipeline_result,
    score_store,
    score_store_lock,
    view_mode,
    viewer_controls_gui,
    viewer_state,
):
    cache = {"config_json": None, "render_fn": None}

    def render_frame(camera_state):
        combined_config = viewer_controls_gui.value
        selected_colormap = colormap.value
        selected_invert_colormap = invert_colormap.value
        selected_quantile_bias = normalization_bias.value
        selected_quantile_percent = normalization_percent.value
        selected_view_mode = view_mode.value
        config_json = (
            f"{selected_view_mode}:{selected_colormap}:"
            f"{selected_quantile_percent}:{selected_quantile_bias}:"
            f"{selected_invert_colormap}:"
            f"{combined_config.model_dump_json()}"
        )
        if cache["render_fn"] is None or config_json != cache["config_json"]:
            pipeline_config = apply_viewer_pipeline_config(
                viewer_state,
                combined_config,
            )

            def backend_fn(camera, compiled_view):
                render_result, score_values = rasterize_scene(
                    camera,
                    compiled_view,
                    view_mode=selected_view_mode,
                    colormap=selected_colormap,
                    quantile_percent=selected_quantile_percent,
                    quantile_bias=selected_quantile_bias,
                    invert_colormap=selected_invert_colormap,
                )
                with score_store_lock:
                    score_store["values"] = score_values
                return render_result

            cache["render_fn"] = pipeline_result.bind(
                pipeline_config,
                backend_fn=backend_fn,
            )
            cache["config_json"] = config_json
        return cache["render_fn"](camera_state).image

    viewer = Viewer(
        render_frame,
        state=viewer_state,
        controls=viewer_controls_gui,
    )
    return (viewer,)


@app.cell
def _(score_bucket_count, score_refresh, score_store, score_store_lock):
    _ = score_refresh.value
    with score_store_lock:
        score_values = score_store["values"]
    score_histogram = score_histogram_chart(
        score_values,
        num_buckets=score_bucket_count.value,
    )
    return (score_histogram,)


if __name__ == "__main__":
    app.run()
