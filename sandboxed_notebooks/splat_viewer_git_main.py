# /// script
# dependencies = [
#     "marimo",
#     "marimo-3dv[desktop] @ git+https://github.com/panicPaul/marimo-3dv@main",
#     "marimo-config-gui @ git+https://github.com/panicPaul/splatkit@main#subdirectory=packages/marimo-config-gui",
#     "torch @ https://download.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp314-cp314-manylinux_2_28_x86_64.whl",
#     "splatkit[cu130] @ https://github.com/panicPaul/splatkit/archive/refs/heads/main.zip#subdirectory=packages/splatkit",
#     "splatkit-native-3dgrt[cu130] @ https://github.com/panicPaul/splatkit/archive/refs/heads/main.zip#subdirectory=packages/splatkit-native-3dgrt",
#     "splatkit-native-faster-gs[cu130] @ https://github.com/panicPaul/splatkit/archive/refs/heads/main.zip#subdirectory=packages/splatkit-native-faster-gs",
#     "splatkit-native-faster-gs-mojo[cu130] @ https://github.com/panicPaul/splatkit/archive/refs/heads/main.zip#subdirectory=packages/splatkit-native-faster-gs-mojo",
#     "splatkit-native-svraster[cu130] @ https://github.com/panicPaul/splatkit/archive/refs/heads/main.zip#subdirectory=packages/splatkit-native-svraster",
# ]
# requires-python = ">=3.14"
#
# [tool.uv]
# prerelease = "allow"
#
# [tool.uv.sources]
# max = { index = "modular-nightly" }
# mojo = { index = "modular-nightly" }
#
# [[tool.uv.index]]
# name = "modular-nightly"
# url = "https://whl.modular.com/nightly/simple/"
# ///

"""Minimal Gaussian splat viewer notebook installed from Git sources."""

import marimo

__generated_with = "0.23.2"
app = marimo.App(
    width="columns",
    layout_file="layouts/splat_viewer_git_main.slides.json",
)

with app.setup:
    from dataclasses import replace

    import marimo as mo
    import numpy as np
    import splatkit as sk
    import splatkit_native_3dgrt.stoch3dgs as skn_stoch
    import splatkit_native_faster_gs.faster_gs as skn_fastergs
    import splatkit_native_faster_gs.faster_gs_depth as skn_fastergs_depth
    import splatkit_native_faster_gs.gaussian_pop as skn_gaussian_pop
    import splatkit_native_faster_gs_mojo.core as skn_fastergs_mojo
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

    skn_fastergs_depth.register()
    skn_fastergs.register()
    skn_fastergs_mojo.register()
    skn_gaussian_pop.register()
    skn_stoch.register()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Splat Viewer
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Interactive Result
    """)
    return


@app.cell
def _(load_form):
    load_form
    return


@app.cell(hide_code=True)
def _(
    antialiasing,
    backend,
    colormap,
    invert_colormap,
    normalization_bias,
    normalization_percent,
    view_mode,
    viewer_controls_gui,
):
    selectors = mo.hstack(
        [backend, view_mode, colormap],
        widths="equal",
        align="start",
        gap=1.0,
    )
    normalization_controls = mo.hstack(
        [
            normalization_percent,
            normalization_bias,
            invert_colormap,
            antialiasing,
        ],
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


@app.cell(column=1)
def _():
    view_mode_options = ["image", "alpha", "depth"]
    view_mode_required_outputs = {
        "image": frozenset(),
        "alpha": frozenset({"alpha"}),
        "depth": frozenset({"depth"}),
    }
    colormap_options = [
        "viridis",
        "turbo",
        "magma",
        "inferno",
        "cividis",
        "gray",
    ]
    return colormap_options, view_mode_options, view_mode_required_outputs


@app.function
def available_gaussian_backends(
    required_outputs: frozenset[str] = frozenset(),
) -> list[str]:
    """List registered backends that accept GaussianScene3D."""
    return sorted(
        backend_name
        for backend_name, registered_backend in sk.BACKEND_REGISTRY.items()
        if any(
            issubclass(sk.GaussianScene3D, accepted_scene_type)
            for accepted_scene_type in registered_backend.accepted_scene_types
        )
        and required_outputs.issubset(registered_backend.supported_outputs)
    )


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
def _(load_bindings, load_form_gui_state, load_json_gui_state):
    load_config = config_value(
        load_bindings,
        form_gui_state=load_form_gui_state,
        json_gui_state=load_json_gui_state,
    )
    return (load_config,)


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
        picked_load_config = load_config or pick_splat_load_config()
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
def apply_colormap(
    values: np.ndarray,
    *,
    colormap: str = "viridis",
    invert: bool = False,
) -> np.ndarray:
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
) -> np.ndarray:
    """Normalize a scalar field to [0, 1] using a biased quantile window."""
    values = field.cpu().numpy()
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
        normalized[valid] = ((values[valid] - lower) / (upper - lower)).astype(
            np.float32
        )
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
) -> np.ndarray:
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
def alpha_to_image(
    alpha: torch.Tensor,
    *,
    colormap: str = "viridis",
    quantile_percent: float = 90.0,
    quantile_bias: float = 0.5,
    invert_colormap: bool = False,
) -> np.ndarray:
    """Convert a single-channel alpha map to a colormapped RGB uint8 image."""
    return apply_colormap(
        normalize_scalar_field(
            alpha,
            quantile_percent=quantile_percent,
            quantile_bias=quantile_bias,
        ),
        colormap=colormap,
        invert=invert_colormap,
    )


@app.function
def render_options_for_backend(
    backend: str,
    *,
    antialiasing: bool,
) -> sk.RenderOptions | None:
    """Build backend-specific viewer render options."""
    if backend == "faster_gs.core":
        return replace(
            skn_fastergs.FasterGSNativeRenderOptions(),
            proper_antialiasing=antialiasing,
        )
    if backend == "faster_gs.depth":
        return replace(
            skn_fastergs_depth.FasterGSDepthNativeRenderOptions(),
            proper_antialiasing=antialiasing,
        )
    if backend == "faster_gs.gaussian_pop":
        return replace(
            skn_gaussian_pop.GaussianPopNativeRenderOptions(),
            proper_antialiasing=antialiasing,
        )
    if backend == "faster_gs_mojo.core":
        return replace(
            skn_fastergs_mojo.FasterGSMojoRenderOptions(),
            proper_antialiasing=antialiasing,
        )
    return None


@app.function
@torch.no_grad()
def rasterize_scene(
    camera: CameraState,
    scene: SplatScene | None,
    *,
    backend: str = "faster_gs.core",
    antialiasing: bool = True,
    view_mode: str = "image",
    colormap: str = "viridis",
    quantile_percent: float = 90.0,
    quantile_bias: float = 0.5,
    invert_colormap: bool = False,
) -> RenderResult:
    """Render a splat scene through splatkit."""
    if scene is None:
        return RenderResult(
            image=np.full(
                (camera.height, camera.width, 3),
                245,
                dtype=np.uint8,
            )
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
        "return_alpha": view_mode == "alpha",
        "return_depth": view_mode == "depth",
    }
    render_options = render_options_for_backend(
        backend,
        antialiasing=antialiasing,
    )
    if render_options is not None:
        render_kwargs["options"] = render_options
    render_output = sk.render(
        backend_scene,
        backend_camera.to(scene.center_positions.device),
        **render_kwargs,
    )
    if view_mode == "alpha":
        image_uint8 = alpha_to_image(
            render_output.alphas[0],
            colormap=colormap,
            quantile_percent=quantile_percent,
            quantile_bias=quantile_bias,
            invert_colormap=invert_colormap,
        )
    elif view_mode == "depth":
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

    return RenderResult(image=image_uint8)


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
def _(
    antialiasing,
    backend,
    colormap,
    invert_colormap,
    normalization_bias,
    normalization_percent,
    pipeline_result,
    view_mode,
    viewer_controls_gui,
    viewer_state,
):
    cache = {"config_json": None, "render_fn": None}

    def render_frame(camera_state):
        selected_backend = backend.value
        if selected_backend is None:
            return np.full(
                (camera_state.height, camera_state.width, 3),
                245,
                dtype=np.uint8,
            )
        combined_config = viewer_controls_gui.value
        selected_colormap = colormap.value
        selected_invert_colormap = invert_colormap.value
        selected_antialiasing = antialiasing.value
        selected_quantile_bias = normalization_bias.value
        selected_quantile_percent = normalization_percent.value
        selected_view_mode = view_mode.value
        config_json = (
            f"{selected_backend}:{selected_view_mode}:"
            f"{selected_colormap}:{selected_quantile_percent}:"
            f"{selected_quantile_bias}:{selected_invert_colormap}:"
            f"{selected_antialiasing}:"
            f"{combined_config.model_dump_json()}"
        )
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
                    backend=selected_backend,
                    antialiasing=selected_antialiasing,
                    view_mode=selected_view_mode,
                    colormap=selected_colormap,
                    quantile_percent=selected_quantile_percent,
                    quantile_bias=selected_quantile_bias,
                    invert_colormap=selected_invert_colormap,
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


@app.cell(column=2, hide_code=True)
def _():
    mo.md("""
    ## Scene Controls
    """)
    return


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
def _():
    antialiasing = mo.ui.checkbox(
        value=True,
        label="Anti-aliasing",
    )
    return (antialiasing,)


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
def _(view_mode, view_mode_required_outputs):
    backend_options = available_gaussian_backends(
        view_mode_required_outputs[view_mode.value]
    )
    backend = mo.ui.dropdown(
        backend_options,
        value=backend_options[0] if backend_options else None,
        label="Backend",
        full_width=True,
    )
    return (backend,)


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
def _(load_error):
    load_error
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Viewer Controls
    """)
    return


@app.cell
def _(backend_bundle, pipeline_result, viewer_state):
    viewer_controls = viewer_pipeline_controls_gui(
        viewer_state,
        pipeline_result,
        viewer_default_config=backend_bundle.viewer_controls(viewer_state),
    )
    viewer_controls_gui = viewer_controls.gui
    return (viewer_controls_gui,)


if __name__ == "__main__":
    app.run()
