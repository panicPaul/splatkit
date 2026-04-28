"""Example marimo notebook for the viser widget."""

import marimo

__generated_with = "0.22.4"
app = marimo.App(width="columns")

with app.setup:
    import nerfview
    import numpy as np
    from jaxtyping import UInt8

    from marimo_viser.viser_widget import viser_marimo


@app.function
def render_fn(
    camera_state: nerfview.CameraState,
    render_tab_state: nerfview.RenderTabState,
) -> UInt8[np.ndarray, "H W 3"]:
    """Render a dummy RGB image for the current viewer camera."""
    # Get camera parameters.
    width = render_tab_state.viewer_width
    height = render_tab_state.viewer_height
    c2w = camera_state.c2w
    K = camera_state.get_K([width, height])

    # Render a dummy image as a function of camera direction.
    camera_dirs = np.einsum(
        "ij,hwj->hwi",
        np.linalg.inv(K),
        np.pad(
            np.stack(
                np.meshgrid(np.arange(width), np.arange(height), indexing="xy"),
                -1,
            )
            + 0.5,
            ((0, 0), (0, 0), (0, 1)),
            constant_values=1.0,
        ),
    )
    dirs = np.einsum("ij,hwj->hwi", c2w[:3, :3], camera_dirs)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    img = ((dirs + 1.0) / 2.0 * 255.0).astype(np.uint8)
    # raise ValueError("test")
    return img


@app.cell
def _(server):
    server.gui.add_slider(
        "scale", min=0.0, max=5.0, step=0.1, initial_value=0.0
    )
    return


@app.cell
def _():
    server_2, viewer_2, widget_2 = viser_marimo(render_fn=render_fn)
    return (widget_2,)


@app.cell
def _(widget, widget_2):
    widget_2.value["camera_state_json"] = widget.value["camera_state_json"]
    return


@app.cell
def _(widget_2):
    widget_2.value["camera_state_json"]
    return


@app.cell(column=1)
def _():
    server, viewer, widget = viser_marimo(render_fn=render_fn)
    widget
    return server, widget


@app.cell
def _(widget):
    widget.value["camera_state_json"]
    return


@app.cell
def _(widget_2):
    widget_2
    return


if __name__ == "__main__":
    app.run()
