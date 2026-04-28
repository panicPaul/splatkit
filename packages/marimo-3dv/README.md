# marimo-3dv

`marimo-3dv` provides 3D viewer widgets and Gaussian splat utilities for
[`marimo`](https://marimo.io) notebooks.

It gives you:

- a marimo-reactive widget by default
- a native image-based viewer for custom renderers
- Gaussian splat loading and viewer pipeline helpers
- explicit save/restore camera state controls
- safer `render_fn` error handling in notebooks

## Installation

Targets Python 3.11+.

```bash
uv pip install marimo-3dv
```

Desktop Qt controls are optional:

```bash
uv pip install "marimo-3dv[desktop]"
```

## Usage

Native viewer mode:

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

The native viewer callback receives a typed `CameraState` with:

- `fov_degrees`
- `width`
- `height`
- `cam_to_world`
- `camera_convention`, currently `Literal["opencv", "opengl", "blender", "colmap"]`

`width` and `height` are measured from the rendered marimo widget size, so you
do not pass them into `marimo_viewer()`.

If you want the view to survive reruns while `render_fn` changes, reuse the
same `ViewerState` object and pass it with `state=...`.

The native viewer currently defaults to OpenCV convention, exposed as
`camera_convention="opencv"`. The widget converts between `opencv`, `opengl`,
`blender`, and `colmap` conventions at the viewer boundary so the Python
callback sees a `cam_to_world` matrix consistent with the declared convention.

The widget also exposes the last primary-button click through
`viewer.get_last_click()`. Dragging or panning does not register as a click.

Controls:

- left drag to orbit
- right drag to pan
- wheel to zoom
- `WASD` to move
- `Q` / `E` to move down / up

## Pydantic GUI

You can also generate marimo controls from a small Pydantic model:

```python
from pydantic import BaseModel, Field

from marimo_config_gui import config_gui


class RenderSettings(BaseModel):
    enabled: bool = True
    steps: int = Field(default=32, ge=1, le=128)
    opacity: float = Field(default=0.5, ge=0.0, le=1.0)
    title: str = "viewer"
```

The generated UI is submit-gated and includes both structured controls and a
JSON editor tab:

```python
form = config_gui(RenderSettings, mode="form")
form
```

Then in a downstream cell:

```python
submitted = form.value
```

`submitted` is either `None` before the first valid submit or a typed
`RenderSettings` instance afterwards.

## Camera State

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
