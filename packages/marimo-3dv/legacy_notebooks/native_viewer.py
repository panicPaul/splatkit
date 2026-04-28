"""Example marimo notebook for the native viewer widget."""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="columns")

with app.setup:
    from functools import partial
    from typing import Literal

    import marimo as mo
    import torch
    from rich import print

    from marimo_viser import CameraState, NativeViewerState, native_viewer


@app.cell
def _():
    viewer_state = NativeViewerState()
    return (viewer_state,)


@app.cell
def _():
    viewer_render_fn = partial(render_fn, noise=0.30, noise_kind="blue")
    return (viewer_render_fn,)


@app.cell
def _(viewer_render_fn, viewer_state):
    viewer = native_viewer(
        viewer_render_fn,
        interactive_quality=50,
        interactive_scale=0.8,
        state=viewer_state,
        # settled_quality='png'
    )
    viewer
    return (viewer,)


@app.cell
def _(viewer):
    viewer.get_debug_info()
    return


@app.cell
def _(viewer):
    debug_info = viewer.get_debug_info()
    print(
        f"unaccounted leaf (sample): "
        f"{debug_info['unaccounted_leaf_latency_sample_ms']}"
    )
    print(
        f"unaccounted leaf (avg): {debug_info['unaccounted_leaf_latency_ms']}"
    )

    print(
        f"unaccounted coarse (sample): "
        f"{debug_info['unaccounted_coarse_latency_sample_ms']}"
    )
    print(
        f"unaccounted coarse (avg): "
        f"{debug_info['unaccounted_coarse_latency_ms']}"
    )
    return


@app.cell
def _(viewer):
    mo.md(f"""
    Current camera JSON:

    ```json
    {viewer.get_camera_state().to_json()}
    ```
    """)
    return


@app.function
def render_fn(
    camera_state: CameraState,
    noise: float = 0.0,
    noise_kind: Literal["white", "blue"] = "white",
) -> torch.Tensor:
    """Render a ray-direction visualization with deterministic directional noise."""
    device = torch.device("cuda")
    width = camera_state.width
    height = camera_state.height
    cam_to_world = torch.as_tensor(
        camera_state.cam_to_world,
        device=device,
        dtype=torch.float32,
    )
    focal_length = (
        0.5
        * height
        / torch.tan(
            torch.deg2rad(torch.tensor(camera_state.fov_degrees, device=device))
            / 2.0
        )
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
    homogeneous_pixels = torch.nn.functional.pad(
        pixel_centers,
        (0, 1),
        value=1.0,
    )
    camera_dirs = torch.einsum(
        "ij,hwj->hwi",
        torch.linalg.inv(intrinsics),
        homogeneous_pixels,
    )
    world_dirs = torch.einsum("ij,hwj->hwi", cam_to_world[:3, :3], camera_dirs)
    world_dirs = world_dirs / torch.linalg.norm(
        world_dirs, dim=-1, keepdim=True
    )
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
            (white_noise_chw - local_mean).squeeze(0).permute(1, 2, 0)
        )
        directional_noise = directional_noise / directional_noise.abs().amax(
            dim=(0, 1),
            keepdim=True,
        ).clamp_min(1e-4)
    else:
        directional_noise = white_noise
    return ((base_image + directional_noise * noise).clip(0, 1) * 255.0).to(
        torch.uint8
    )


@app.cell
def _(viewer):
    print(viewer.get_camera_state())
    return


if __name__ == "__main__":
    app.run()
