# marimo-3dv

`marimo-3dv` provides 3D viewer widgets and Gaussian splat utilities for
[`marimo`](https://marimo.io) notebooks.

It gives you:

- a marimo-reactive widget by default
- a native image-based viewer for custom renderers
- Gaussian splat loading and viewer pipeline helpers
- explicit save/restore camera state controls
- safer `render_fn` error handling in notebooks

For an interactive version, run from the repository root:

```bash
marimo run docs/interactive/marimo-3dv.py
```

## Installation

Targets Python 3.11+.

```bash
uv pip install marimo-3dv
```

Desktop Qt controls are optional:

```bash
uv pip install "marimo-3dv[desktop]"
```

## Main Concepts

The package has three layers:

- `marimo_3dv.viewer`: the native image-streaming viewer, camera state,
  linked viewer state, and reusable viewer controls.
- `marimo_3dv.ops`: reusable Gaussian splat loading, filtering,
  normalization, setup, and overlay operations.
- `marimo_3dv.pipeline`: small composition primitives for building notebook
  setup and GUI flows around those ops.

Use the viewer layer when you already have a render function. Use the ops layer
when the notebook needs to load, normalize, filter, or annotate splat data
before rendering it. Use pipeline helpers when a notebook has enough setup
state that keeping the loader, controls, and rendered result bundled together
is clearer than passing loose values between cells.

## Native Viewer

The simplest viewer takes a Python callback that maps a typed camera state to
an image tensor. The callback receives the live camera pose, field of view, and
measured widget size. It returns an `H x W x 3` or `H x W x 4` image-like value,
usually a `torch.Tensor`, NumPy array, or PIL-compatible image.

```python
import torch
from typing import Literal

from marimo_3dv import CameraState, ViewerState, marimo_viewer


def render_fn(
    camera_state: CameraState,
    noise: float = 0.0,
    noise_kind: Literal["white", "blue"] = "white",
) -> torch.Tensor:
    device = torch.device("cuda")
    width = camera_state.width
    height = camera_state.height
    cam_to_world = torch.as_tensor(
        camera_state.cam_to_world,
        device=device,
        dtype=torch.float32,
    )
    focal_length = 0.5 * height / torch.tan(
        torch.deg2rad(torch.tensor(camera_state.fov_degrees, device=device))
        / 2.0
    )
    intrinsics = torch.eye(3, device=device, dtype=torch.float32)
    intrinsics[0, 0] = focal_length
    intrinsics[1, 1] = focal_length
    intrinsics[0, 2] = width / 2.0
    intrinsics[1, 2] = height / 2.0

    pixel_x, pixel_y = torch.meshgrid(
        torch.arange(width, device=device, dtype=torch.float32),
        torch.arange(height, device=device, dtype=torch.float32),
        indexing="xy",
    )
    pixel_centers = torch.stack((pixel_x, pixel_y), dim=-1) + 0.5
    homogeneous_pixels = torch.nn.functional.pad(pixel_centers, (0, 1), value=1.0)
    camera_dirs = torch.einsum(
        "ij,hwj->hwi",
        torch.linalg.inv(intrinsics),
        homogeneous_pixels,
    )
    world_dirs = torch.einsum("ij,hwj->hwi", cam_to_world[:3, :3], camera_dirs)
    world_dirs = world_dirs / torch.linalg.norm(world_dirs, dim=-1, keepdim=True)
    base_image = (world_dirs + 1.0) / 2.0
    quantized_dirs = torch.round((world_dirs + 1.0) * 1024.0)
    direction_hashes = torch.stack(
        (
            quantized_dirs[..., 0] * 127.1
            + quantized_dirs[..., 1] * 311.7
            + quantized_dirs[..., 2] * 74.7,
            quantized_dirs[..., 0] * 269.5
            + quantized_dirs[..., 1] * 183.3
            + quantized_dirs[..., 2] * 246.1,
            quantized_dirs[..., 0] * 113.5
            + quantized_dirs[..., 1] * 271.9
            + quantized_dirs[..., 2] * 124.6,
        ),
        dim=-1,
    )
    white_noise = (
        torch.frac(torch.sin(direction_hashes) * 43758.5453) * 2.0 - 1.0
    )
    if noise_kind == "blue":
        white_noise_chw = white_noise.permute(2, 0, 1).unsqueeze(0)
        local_mean = torch.nn.functional.avg_pool2d(
            white_noise_chw,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        directional_noise = (
            white_noise_chw - local_mean
        ).squeeze(0).permute(1, 2, 0)
        directional_noise = directional_noise / directional_noise.abs().amax(
            dim=(0, 1),
            keepdim=True,
        ).clamp_min(1e-4)
    else:
        directional_noise = white_noise
    return ((base_image + directional_noise * noise).clip(0, 1) * 255.0).to(torch.uint8)


viewer_state = ViewerState()
viewer = marimo_viewer(
    render_fn,
    state=viewer_state,
)
viewer
```

The callback receives a typed `CameraState` with:

- `fov_degrees`
- `width`
- `height`
- `cam_to_world`
- `camera_convention`, currently `Literal["opencv", "opengl", "blender", "colmap"]`

`width` and `height` are measured from the rendered marimo widget size, so you
do not pass them into `marimo_viewer()`.

If you want the view to survive reruns while `render_fn` changes, reuse the
same `ViewerState` object and pass it with `state=...`.

The higher-level `Viewer(...)` helper returns a live marimo viewer in notebook
mode and a `NoopViewer` placeholder in script mode. That makes notebooks easier
to execute as normal Python scripts or smoke tests:

```python
from marimo_3dv import Viewer, ViewerState

viewer_state = ViewerState()
viewer = Viewer(render_fn, state=viewer_state)
```

## Camera Conventions

The native viewer currently defaults to OpenCV convention, exposed as
`camera_convention="opencv"`. The widget converts between `opencv`, `opengl`,
`blender`, and `colmap` conventions at the viewer boundary so the Python
callback sees a `cam_to_world` matrix consistent with the declared convention.

Set the convention on the `ViewerState` when your renderer expects another
camera basis:

```python
viewer_state = ViewerState(camera_convention="opengl")
viewer = marimo_viewer(render_fn, state=viewer_state)
```

Use `get_camera_state()` and `set_camera_state(...)` for typed state transfer:

```python
state = viewer.get_camera_state()
viewer.set_camera_state(state)
```

The widget also exposes the last primary-button click:

```python
click = viewer.get_last_click()
```

Dragging, panning, and keyboard movement do not register as clicks.

Controls:

- left drag to orbit
- right drag to pan
- wheel to zoom
- `WASD` to move
- `Q` / `E` to move down / up

## Viewer State And Controls

`ViewerState` owns the state that should survive cell reruns:

- camera pose and field of view
- overlay toggles for axes, horizon, origin, and stats
- render-quality settings
- keyboard and pointer tuning
- viewer-frame origin and rotation

The reusable controls are Pydantic models, so they work naturally with
`marimo-config-gui`:

```python
from marimo_3dv import (
    ViewerState,
    apply_viewer_config,
    viewer_controls_gui,
)

viewer_state = ViewerState()
controls = viewer_controls_gui(viewer_state, label="Viewer controls")
viewer_config = controls.value
apply_viewer_config(viewer_state, viewer_config)
```

The default control tree is `ViewerControlsConfig`, with nested camera,
overlay, render, navigation, interaction, and transform sections. For small
notebooks, using `viewer_controls_gui(...)` directly is usually enough. For
larger notebooks, keep the controls in a side column and apply the current
value before constructing the viewer.

## Linked Viewers

Multiple viewers can share selected state. This is useful for comparing
backends, render modes, filters, or scene normalizations from the same camera:

```python
from marimo_3dv import ViewerState, link_viewer_states

left_state = ViewerState()
right_state = ViewerState()

link = link_viewer_states(
    left_state,
    right_state,
    fields=("camera_state", "show_axes", "show_stats"),
)
```

The returned `ViewerStateLink` can be closed when the synchronization should
stop:

```python
link.close()
```

## Gaussian Splat Ops

The `marimo_3dv.ops` namespace contains reusable setup pieces for splat
notebooks:

- `SplatLoadConfig`, `splat_load_form(...)`, and `load_splat_scene(...)` for
  loading splat assets.
- `filter_opacity_op(...)`, `filter_size_op(...)`, and `max_sh_degree_op(...)`
  for common data reduction controls.
- `pca_alignment_op(...)` and `camera_similarity_op(...)` for coordinate
  normalization.
- `paint_ray_op(...)` for click-driven ray overlays.
- low-level transform helpers such as `compose_transforms(...)`,
  `pca_transform_from_points(...)`, and `similarity_from_cameras(...)`.

These ops are intended to be notebook building blocks rather than one hidden
viewer framework. Keep the expensive scene-loading cell behind an explicit
marimo run button when reloading would compile kernels or move large tensors.

## Pydantic GUI Integration

You can also generate marimo controls from a small Pydantic model:

```python
from pydantic import BaseModel, Field

from marimo_config_gui import config_gui_panel, create_config_state


class RenderSettings(BaseModel):
    enabled: bool = True
    steps: int = Field(default=32, ge=1, le=128)
    opacity: float = Field(default=0.5, ge=0.0, le=1.0)
    title: str = "viewer"
```

The generated UI can be backed by shared marimo state:

```python
form_state, json_state, bindings = create_config_state(RenderSettings)
form = config_gui_panel(bindings, form_gui_state=form_state)
form
```

Then in a downstream cell:

```python
submitted = form.value
```

`submitted` is a typed `RenderSettings` instance when the current form payload
is valid.

## Saved Camera State

The widget includes:

- `Save Camera State`
- `Restore Saved Camera State`

The saved state is exposed through `widget.value["camera_state_json"]`, so
viewer-to-viewer sync can be as simple as:

```python
widget_b.value["camera_state_json"] = widget_a.value["camera_state_json"]
```

For convenience, the same state is also available through typed helpers:

```python
state = widget.get_camera_state()
widget.set_camera_state(state)
```

The typed representation is `CameraState` with:

- `fov_degrees`
- `width`
- `height`
- `cam_to_world`
- `camera_convention`

## Render Errors

If `render_fn` raises, the kernel stays alive.

- the traceback is printed server-side
- the viewer shows an error image
- the browser-side viewer shows a copyable traceback panel

See the notebooks in this repository for examples.

## Script Mode

`Viewer(...)` returns `NoopViewer` outside a live marimo runtime. The placeholder
preserves `ViewerState`, exposes `get_camera_state()`, `set_camera_state(...)`,
`get_last_click()`, `close()`, and `rerender(...)`, and raises only for
browser-only operations such as `anywidget()` and `get_snapshot()`.

This is the recommended entrypoint for notebooks that should also run in CI or
as command-line scripts:

```python
viewer = Viewer(render_fn, state=viewer_state)
```

Use `marimo_viewer(...)` directly when a live notebook widget is required.

## Public Surface

The root package re-exports the common viewer API:

- `CameraState`
- `ViewerState`
- `Viewer`
- `MarimoViewer`
- `NoopViewer`
- `ViewerControlsConfig` and nested control models
- `viewer_controls_gui(...)`
- `viewer_controls_config(...)`
- `apply_viewer_config(...)`
- `link_viewer_states(...)`

Import splat and setup ops from `marimo_3dv.ops` when you need them.

## License

This package is distributed under the Apache License 2.0.
