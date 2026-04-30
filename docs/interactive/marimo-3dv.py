"""Interactive documentation for marimo-3dv."""

# ruff: noqa: ANN001, ANN202, B018

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="wide")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import torch

    mo.md(
        r"""
        # marimo-3dv

        This notebook introduces the native marimo viewer, typed camera state,
        reusable viewer controls, linked viewers, click state, and error
        handling. The examples use a synthetic directional renderer so the
        notebook stays lightweight.
        """
    )
    return mo, torch


@app.cell
def _():
    from typing import Literal

    from marimo_3dv import (
        CameraState,
        Viewer,
        ViewerState,
        apply_viewer_config,
        link_viewer_states,
        viewer_controls_gui,
    )

    return (
        CameraState,
        Literal,
        Viewer,
        ViewerState,
        apply_viewer_config,
        link_viewer_states,
        viewer_controls_gui,
    )


@app.cell
def _(CameraState, Literal, torch):
    def render_direction_field(
        camera_state: CameraState,
        *,
        mode: Literal["direction", "depth-cue"] = "direction",
    ) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        width = max(1, int(camera_state.width))
        height = max(1, int(camera_state.height))
        xs = torch.linspace(-1.0, 1.0, width, device=device)
        ys = torch.linspace(-1.0, 1.0, height, device=device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        dirs = torch.stack(
            (
                grid_x,
                -grid_y,
                torch.ones_like(grid_x),
            ),
            dim=-1,
        )
        rotation = torch.as_tensor(
            camera_state.cam_to_world[:3, :3],
            device=device,
            dtype=torch.float32,
        )
        world_dirs = torch.einsum("ij,hwj->hwi", rotation, dirs)
        world_dirs = torch.nn.functional.normalize(world_dirs, dim=-1)
        base = (world_dirs + 1.0) * 0.5
        if mode == "depth-cue":
            depth = world_dirs[..., 2:3].abs()
            base = torch.cat((base[..., :2], depth), dim=-1)
        return (base.clamp(0, 1) * 255).to(torch.uint8)

    return (render_direction_field,)


@app.cell
def _(ViewerState):
    primary_state = ViewerState(camera_convention="opencv")
    secondary_state = ViewerState(camera_convention="opencv")
    return primary_state, secondary_state


@app.cell
def _(primary_state, viewer_controls_gui):
    controls = viewer_controls_gui(primary_state, label="Viewer controls")
    controls.gui
    return (controls,)


@app.cell
def _(apply_viewer_config, controls, primary_state):
    apply_viewer_config(primary_state, controls.value)
    return


@app.cell(hide_code=True)
def _(Viewer, primary_state, render_direction_field):
    viewer = Viewer(render_direction_field, state=primary_state)
    viewer
    return (viewer,)


@app.cell(hide_code=True)
def _(mo, viewer):
    camera = viewer.get_camera_state()
    mo.md(
        f"""
        ---

        ## Camera State

        The Python callback receives the live camera state. Current widget
        render size: `{camera.width} x {camera.height}`. Convention:
        `{camera.camera_convention}`.
        """
    )
    return (camera,)


@app.cell
def _(link_viewer_states, primary_state, secondary_state):
    viewer_link = link_viewer_states(
        primary_state,
        secondary_state,
        fields=("camera_state", "show_axes", "show_stats"),
    )
    return (viewer_link,)


@app.cell(hide_code=True)
def _(Viewer, render_direction_field, secondary_state):
    linked_viewer = Viewer(
        lambda camera_state: render_direction_field(
            camera_state,
            mode="depth-cue",
        ),
        state=secondary_state,
    )
    linked_viewer
    return (linked_viewer,)


@app.cell(hide_code=True)
def _(mo, viewer):
    click = viewer.get_last_click()
    mo.md(
        f"""
        ---

        ## Click State

        `viewer.get_last_click()` returns the most recent primary-button click.
        Current value:

        ```python
        {click!r}
        ```
        """
    )
    return


@app.cell
def _(CameraState, torch):
    def broken_render_fn(camera_state: CameraState) -> torch.Tensor:
        del camera_state
        raise RuntimeError(
            "This deliberate error demonstrates viewer fallback."
        )

    return (broken_render_fn,)


@app.cell(hide_code=True)
def _(Viewer, ViewerState, broken_render_fn, mo):
    error_demo_enabled = mo.ui.checkbox(label="Show deliberate render error")
    error_viewer = Viewer(broken_render_fn, state=ViewerState())
    mo.vstack(
        [error_demo_enabled, error_viewer if error_demo_enabled.value else ""]
    )
    return error_demo_enabled, error_viewer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Splat Ops

        The viewer is only one layer. Import `marimo_3dv.ops` when a notebook
        needs splat loading, filtering, normalization, setup transforms, or
        click-driven overlays:

        ```python
        from marimo_3dv.ops import (
            SplatLoadConfig,
            filter_opacity_op,
            load_splat_scene_from_config,
            pca_alignment_op,
            paint_ray_op,
        )
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Reference

        Common imports:

        ```python
        from marimo_3dv import (
            Viewer,
            ViewerState,
            CameraState,
            viewer_controls_gui,
            apply_viewer_config,
            link_viewer_states,
        )
        ```

        Use `Viewer(...)` for notebooks that should also run as scripts.
        Use `marimo_viewer(...)` directly when a live browser widget is
        required.
        """
    )
    return


if __name__ == "__main__":
    app.run()
