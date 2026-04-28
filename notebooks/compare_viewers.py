"""Side-by-side scene comparison notebook with linked viewer navigation."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

with app.setup:
    import ctypes
    import signal
    import sys
    from enum import Enum, StrEnum
    from pathlib import Path

    import ember_adapter_backends.fastergs as sk_fastergs
    import ember_adapter_backends.fastgs as sk_fastgs
    import ember_adapter_backends.gsplat as sk_gsplat
    import ember_adapter_backends.inria as sk_inria
    import ember_adapter_backends.stoch3dgs as sk_stoch
    import ember_core as sk
    import ember_native_svraster.svraster as sk_svraster
    import marimo as mo
    import numpy as np
    import torch
    from marimo_3dv import (
        CameraState,
        RenderResult,
        Viewer,
        ViewerState,
        cleanup_before_splat_reload,
        link_viewer_states,
    )
    from marimo_config_gui import (
        config_commit_button,
        config_committed_value,
        config_error,
        config_form,
        config_value,
        create_committed_config_state,
        create_config_state,
    )
    from pydantic import BaseModel, Field

    sk_fastgs.register()
    sk_fastergs.register()
    sk_gsplat.register()
    sk_inria.register()
    sk_stoch.register()
    sk_svraster.register()

    active_link = {"handle": None}

    BLANK_COLOR = 245

    class SceneTypeChoice(StrEnum):
        """Supported scene families for comparison loading."""

        GAUSSIAN = "3dgs_ply"
        SVRASTER = "svraster_checkpoint"

    class GaussianLoadConfig(BaseModel):
        """Parameters for loading a 3DGS PLY scene."""

        path: Path = Path("tmp/example_scene.ply")

    class SVRasterLoadConfig(BaseModel):
        """Parameters for loading an SV Raster checkpoint."""

        run_path: Path = Path("tmp/quarter_res_full_train/garden")
        iteration: int = -1

    class SceneLoaderConfig(BaseModel):
        """Loader settings for one side of the comparison view."""

        scene_type: SceneTypeChoice = Field(
            default=SceneTypeChoice.GAUSSIAN,
            description="Scene family used for loading and backend selection.",
        )
        gaussian: GaussianLoadConfig = Field(default_factory=GaussianLoadConfig)
        svraster: SVRasterLoadConfig = Field(default_factory=SVRasterLoadConfig)

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

    def scene_class_for_type(
        scene_type: SceneTypeChoice,
    ) -> type[sk.GaussianScene3D] | type[sk.SparseVoxelScene]:
        """Return the scene class associated with a loader selection."""
        if scene_type is SceneTypeChoice.GAUSSIAN:
            return sk.GaussianScene3D
        return sk.SparseVoxelScene

    def available_backends(scene_type: SceneTypeChoice) -> list[str]:
        """List registered backends compatible with a scene type."""
        scene_class = scene_class_for_type(scene_type)
        return sorted(
            backend_name
            for backend_name, registered_backend in sk.BACKEND_REGISTRY.items()
            if any(
                issubclass(scene_class, accepted_scene_type)
                for accepted_scene_type in registered_backend.accepted_scene_types
            )
        )

    def selected_loader_value(
        config: SceneLoaderConfig,
    ) -> GaussianLoadConfig | SVRasterLoadConfig:
        """Return the active nested loader config."""
        if config.scene_type is SceneTypeChoice.GAUSSIAN:
            return config.gaussian
        return config.svraster

    def load_scene_artifact(config: SceneLoaderConfig) -> sk.Scene | None:
        """Load the selected scene artifact and move it to CUDA when available."""
        if config.scene_type is SceneTypeChoice.GAUSSIAN:
            load_value = config.gaussian
            if not load_value.path.exists():
                return None
            scene = sk.load_gaussian_ply(load_value.path)
        else:
            load_value = config.svraster
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
            scene = scene.to(torch.device("cuda"))
        return scene

    def camera_device_for_scene(scene: sk.Scene) -> torch.device:
        """Return the device that should own camera tensors for the scene."""
        if isinstance(scene, sk.GaussianScene3D):
            return scene.center_position.device
        if isinstance(scene, sk.SparseVoxelScene):
            return scene.octpath.device
        raise TypeError(f"Unsupported scene type {type(scene).__name__}.")

    def blank_frame(camera: CameraState) -> RenderResult:
        """Return an empty fallback frame."""
        return RenderResult(
            image=np.full(
                (camera.height, camera.width, 3), BLANK_COLOR, dtype=np.uint8
            )
        )

    install_linux_parent_death_signal()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Compare Viewers
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
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
    (
        left_loader_form_gui_state,
        left_loader_json_gui_state,
        left_loader_bindings,
    ) = create_config_state(
        SceneLoaderConfig,
        value=SceneLoaderConfig(),
    )
    (
        left_loader_committed_state,
        set_left_loader_committed_state,
    ) = create_committed_config_state(
        SceneLoaderConfig,
        value=SceneLoaderConfig(),
    )
    return (
        left_loader_bindings,
        left_loader_committed_state,
        left_loader_form_gui_state,
        left_loader_json_gui_state,
        set_left_loader_committed_state,
    )


@app.cell
def _():
    (
        right_loader_form_gui_state,
        right_loader_json_gui_state,
        right_loader_bindings,
    ) = create_config_state(
        SceneLoaderConfig,
        value=SceneLoaderConfig(scene_type=SceneTypeChoice.SVRASTER),
    )
    (
        right_loader_committed_state,
        set_right_loader_committed_state,
    ) = create_committed_config_state(
        SceneLoaderConfig,
        value=SceneLoaderConfig(scene_type=SceneTypeChoice.SVRASTER),
    )
    return (
        right_loader_bindings,
        right_loader_committed_state,
        right_loader_form_gui_state,
        right_loader_json_gui_state,
        set_right_loader_committed_state,
    )


@app.cell
def _():
    (
        link_form_gui_state,
        link_json_gui_state,
        link_bindings,
    ) = create_config_state(
        LinkConfig,
        value=LinkConfig(),
    )
    return link_bindings, link_form_gui_state, link_json_gui_state


@app.cell
def _(
    left_loader_bindings, left_loader_form_gui_state, left_loader_json_gui_state
):
    left_loader_form = config_form(
        left_loader_bindings,
        form_gui_state=left_loader_form_gui_state,
        label="Left scene",
        nested_models_multiple_open=False,
    )
    left_loader_error = config_error(
        left_loader_bindings,
        form_gui_state=left_loader_form_gui_state,
        json_gui_state=left_loader_json_gui_state,
    )
    left_loader_draft = config_value(
        left_loader_bindings,
        form_gui_state=left_loader_form_gui_state,
        json_gui_state=left_loader_json_gui_state,
    )
    return left_loader_draft, left_loader_error, left_loader_form


@app.cell
def _(
    left_loader_bindings,
    left_loader_committed_state,
    left_loader_form_gui_state,
    left_loader_json_gui_state,
    set_left_loader_committed_state,
):
    left_load_button = config_commit_button(
        left_loader_bindings,
        form_gui_state=left_loader_form_gui_state,
        json_gui_state=left_loader_json_gui_state,
        committed_state=left_loader_committed_state,
        set_committed_state=set_left_loader_committed_state,
        label="Load left scene",
    )
    return (left_load_button,)


@app.cell
def _(left_loader_bindings, left_loader_committed_state):
    left_loader_config = config_committed_value(
        left_loader_bindings,
        committed_state=left_loader_committed_state,
    )
    return (left_loader_config,)


@app.cell
def _(
    right_loader_bindings,
    right_loader_form_gui_state,
    right_loader_json_gui_state,
):
    right_loader_form = config_form(
        right_loader_bindings,
        form_gui_state=right_loader_form_gui_state,
        label="Right scene",
        nested_models_multiple_open=False,
    )
    right_loader_error = config_error(
        right_loader_bindings,
        form_gui_state=right_loader_form_gui_state,
        json_gui_state=right_loader_json_gui_state,
    )
    right_loader_draft = config_value(
        right_loader_bindings,
        form_gui_state=right_loader_form_gui_state,
        json_gui_state=right_loader_json_gui_state,
    )
    return right_loader_draft, right_loader_error, right_loader_form


@app.cell
def _(
    right_loader_bindings,
    right_loader_committed_state,
    right_loader_form_gui_state,
    right_loader_json_gui_state,
    set_right_loader_committed_state,
):
    right_load_button = config_commit_button(
        right_loader_bindings,
        form_gui_state=right_loader_form_gui_state,
        json_gui_state=right_loader_json_gui_state,
        committed_state=right_loader_committed_state,
        set_committed_state=set_right_loader_committed_state,
        label="Load right scene",
    )
    return (right_load_button,)


@app.cell
def _(right_loader_bindings, right_loader_committed_state):
    right_loader_config = config_committed_value(
        right_loader_bindings,
        committed_state=right_loader_committed_state,
    )
    return (right_loader_config,)


@app.cell
def _(link_bindings, link_form_gui_state, link_json_gui_state):
    link_form = config_form(
        link_bindings,
        form_gui_state=link_form_gui_state,
        label="Linked navigation",
    )
    link_error = config_error(
        link_bindings,
        form_gui_state=link_form_gui_state,
        json_gui_state=link_json_gui_state,
    )
    link_config = config_value(
        link_bindings,
        form_gui_state=link_form_gui_state,
        json_gui_state=link_json_gui_state,
    )
    return link_config, link_error, link_form


@app.cell
def _(left_loader_draft):
    left_backend_options = available_backends(
        left_loader_draft.scene_type
        if left_loader_draft is not None
        else SceneTypeChoice.GAUSSIAN
    )
    left_backend = mo.ui.dropdown(
        left_backend_options,
        value=left_backend_options[0] if left_backend_options else None,
        label="Left backend",
        full_width=True,
    )
    return (left_backend,)


@app.cell
def _(right_loader_draft):
    right_backend_options = available_backends(
        right_loader_draft.scene_type
        if right_loader_draft is not None
        else SceneTypeChoice.SVRASTER
    )
    right_backend = mo.ui.dropdown(
        right_backend_options,
        value=right_backend_options[0] if right_backend_options else None,
        label="Right backend",
        full_width=True,
    )
    return (right_backend,)


@app.cell
def _():
    left_viewer_state = ViewerState(camera_convention="opencv")
    right_viewer_state = ViewerState(camera_convention="opencv")
    return left_viewer_state, right_viewer_state


@app.cell
def _(left_loader_config, left_viewer_state):
    cleanup_before_splat_reload(
        left_viewer_state,
        close_existing_viewer=True,
        empty_cuda_cache=True,
    )
    left_scene = (
        None
        if left_loader_config is None
        else load_scene_artifact(left_loader_config)
    )
    return (left_scene,)


@app.cell
def _(right_loader_config, right_viewer_state):
    cleanup_before_splat_reload(
        right_viewer_state,
        close_existing_viewer=True,
        empty_cuda_cache=True,
    )
    right_scene = (
        None
        if right_loader_config is None
        else load_scene_artifact(right_loader_config)
    )
    return (right_scene,)


@app.function
@torch.no_grad()
def rasterize_scene(
    camera: CameraState,
    scene: sk.Scene | None,
    *,
    backend: str | None,
) -> RenderResult:
    """Render one scene through the selected ember-core backend."""
    if scene is None or not backend:
        return blank_frame(camera)

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
    image = render_output.render[0].clamp(0.0, 1.0).cpu().numpy()
    return RenderResult(image=(image * 255).astype(np.uint8))


@app.cell
def _(left_backend, left_scene, left_viewer_state, link_config):
    left_viewer_state.aspect_ratio = (
        link_config.aspect_ratio
        if link_config is not None
        else LinkConfig().aspect_ratio
    )

    def render_left(camera_state):
        return rasterize_scene(
            camera_state,
            left_scene,
            backend=left_backend.value,
        ).image

    left_viewer = Viewer(render_left, state=left_viewer_state)
    return (left_viewer,)


@app.cell
def _(link_config, right_backend, right_scene, right_viewer_state):
    right_viewer_state.aspect_ratio = (
        link_config.aspect_ratio
        if link_config is not None
        else LinkConfig().aspect_ratio
    )

    def render_right(camera_state):
        return rasterize_scene(
            camera_state,
            right_scene,
            backend=right_backend.value,
        ).image

    right_viewer = Viewer(render_right, state=right_viewer_state)
    return (right_viewer,)


@app.cell
def _(left_viewer_state, link_config, right_viewer_state):
    if active_link["handle"] is not None:
        active_link["handle"].close()
        active_link["handle"] = None

    resolved_link_config = link_config or LinkConfig()

    selected_fields: list[str] = []
    if resolved_link_config.link_camera:
        selected_fields.append("camera_state")
    if resolved_link_config.link_show_axes:
        selected_fields.append("show_axes")
    if resolved_link_config.link_show_horizon:
        selected_fields.append("show_horizon")
    if resolved_link_config.link_show_origin:
        selected_fields.append("show_origin")
    if resolved_link_config.link_show_stats:
        selected_fields.append("show_stats")

    if resolved_link_config.enabled and selected_fields:
        active_link["handle"] = link_viewer_states(
            left_viewer_state,
            right_viewer_state,
            fields=tuple(selected_fields),
            bidirectional=True,
        )
    return


@app.cell
def _(
    left_backend,
    left_load_button,
    left_loader_error,
    left_loader_form,
    link_error,
    link_form,
    right_backend,
    right_load_button,
    right_loader_error,
    right_loader_form,
):
    left_controls = mo.vstack(
        [
            left_loader_error,
            left_backend,
            mo.hstack([left_loader_form, left_load_button], align="end"),
        ],
        gap=0.75,
    )
    right_controls = mo.vstack(
        [
            right_loader_error,
            right_backend,
            mo.hstack([right_loader_form, right_load_button], align="end"),
        ],
        gap=0.75,
    )
    controls = mo.vstack(
        [
            link_error,
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


if __name__ == "__main__":
    app.run()
