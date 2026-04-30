"""Interactive Ember viewer tutorial."""

# ruff: noqa: ANN001, ANN202, B018

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="wide")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import torch
    from marimo_3dv import Viewer, ViewerState, viewer_controls_gui

    mo.md(
        r"""
        # Ember Viewer Workflow

        Ember keeps viewer logic optional. The core package owns small bridge
        helpers for camera payloads and viewer preparation; `marimo-3dv` owns
        the live notebook viewer.
        """
    )
    return Viewer, ViewerState, mo, torch, viewer_controls_gui


@app.cell
def _(ViewerState):
    viewer_state = ViewerState(camera_convention="opencv")
    return (viewer_state,)


@app.cell
def _(viewer_controls_gui, viewer_state):
    controls = viewer_controls_gui(viewer_state, label="Viewer controls")
    controls.gui
    return (controls,)


@app.cell
def _(torch):
    def render_demo(camera_state):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        width = max(1, int(camera_state.width))
        height = max(1, int(camera_state.height))
        xs = torch.linspace(-1.0, 1.0, width, device=device)
        ys = torch.linspace(-1.0, 1.0, height, device=device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        radius = torch.sqrt(grid_x.square() + grid_y.square()).clamp(0, 1)
        image = torch.stack((1.0 - radius, grid_x.abs(), grid_y.abs()), dim=-1)
        return (image * 255).to(torch.uint8)

    return (render_demo,)


@app.cell(hide_code=True)
def _(Viewer, render_demo, viewer_state):
    viewer = Viewer(render_demo, state=viewer_state)
    viewer
    return (viewer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Ember Bridge Pattern

        A real Ember viewer notebook usually follows this shape:

        ```python
        scene = sk.load_scene("scene.ply")
        viewer_state = ViewerState()
        viewer = Viewer(render_fn, state=viewer_state)
        ```

        The render function converts the `marimo-3dv` camera state into the
        `ember-core` camera contract, then calls `sk.render(...)` with a
        registered backend.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Viewer Responsibilities

        - `marimo-3dv`: browser widget, controls, linked state, clicks.
        - `ember-core.viewer`: camera conversion, mode resolution, prep/cache
          helpers, stats helpers.
        - Backend package: actual rendering.

        Keep expensive scene loading behind an explicit button in real
        notebooks so edits to viewer controls do not reload large assets.
        """
    )
    return


if __name__ == "__main__":
    app.run()
