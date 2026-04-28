"""Spherical harmonics demo notebook."""

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="columns")

with app.setup:
    from dataclasses import replace

    import marimo as mo
    import nerfview
    import numpy as np
    import torch
    import torch.nn.functional as F
    from jaxtyping import Float, UInt8
    from py_jaxtyping import PyArray
    from pydantic import BaseModel, Field
    from torch import Tensor

    from marimo_viser import form_gui, viser_marimo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Spherical Harmonics for 3DGS

    This notebook visualizes spherical harmonics as a 3DGS appearance model:
    spherical harmonics modulate RGB, while geometry stays fixed.

    `Point mode` shows the color of a single Gaussian under the current view
    direction. `Axis preview mode` paints the full directional color field on
    a sphere so the whole angular function is visible at once.
    """)
    return


@app.function
def spherical_harmonics(
    dirs: Float[Tensor, "*batch 3"],
    coefficients: Float[Tensor, "num_bases 3"],
    degrees_to_use: int,
) -> Float[Tensor, "*batch 3"]:
    """Evaluate real spherical harmonics at unit directions."""
    assert 0 <= degrees_to_use <= 4, (
        f"Only degrees 0-4 supported, got {degrees_to_use}"
    )
    num_bases = (degrees_to_use + 1) ** 2
    assert num_bases <= coefficients.shape[-2], (
        f"Need at least {num_bases} SH bases, got {coefficients.shape[-2]}"
    )

    dirs = F.normalize(dirs, p=2, dim=-1)
    x, y, z = dirs.unbind(-1)

    basis_values = torch.zeros(
        (*dirs.shape[:-1], coefficients.shape[-2]),
        dtype=dirs.dtype,
        device=dirs.device,
    )

    basis_values[..., 0] = 0.2820947917738781
    if degrees_to_use == 0:
        return (basis_values[..., None] * coefficients).sum(dim=-2)

    basis_values[..., 1] = 0.48860251190292 * y
    basis_values[..., 2] = -0.48860251190292 * z
    basis_values[..., 3] = -0.48860251190292 * x
    if degrees_to_use == 1:
        return (basis_values[..., None] * coefficients).sum(dim=-2)

    z2 = z * z
    cos_2_azimuth = x * x - y * y
    sin_2_azimuth = 2.0 * x * y

    basis_values[..., 4] = 0.5462742152960395 * sin_2_azimuth
    basis_values[..., 5] = -1.092548430592079 * z * y
    basis_values[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    basis_values[..., 7] = -1.092548430592079 * z * x
    basis_values[..., 8] = 0.5462742152960395 * cos_2_azimuth
    if degrees_to_use == 2:
        return (basis_values[..., None] * coefficients).sum(dim=-2)

    cos_3_azimuth = x * cos_2_azimuth - y * sin_2_azimuth
    sin_3_azimuth = x * sin_2_azimuth + y * cos_2_azimuth

    basis_values[..., 9] = -0.5900435899266435 * sin_3_azimuth
    basis_values[..., 10] = 1.445305721320277 * z * sin_2_azimuth
    basis_values[..., 11] = (-2.285228997322329 * z2 + 0.4570457994644658) * y
    basis_values[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    basis_values[..., 13] = (-2.285228997322329 * z2 + 0.4570457994644658) * x
    basis_values[..., 14] = 1.445305721320277 * z * cos_2_azimuth
    basis_values[..., 15] = -0.5900435899266435 * cos_3_azimuth
    if degrees_to_use == 3:
        return (basis_values[..., None] * coefficients).sum(dim=-2)

    cos_4_azimuth = x * cos_3_azimuth - y * sin_3_azimuth
    sin_4_azimuth = x * sin_3_azimuth + y * cos_3_azimuth

    basis_values[..., 16] = 0.6258357354491763 * sin_4_azimuth
    basis_values[..., 17] = -1.770130769779931 * z * sin_3_azimuth
    basis_values[..., 18] = (
        3.31161143515146 * z2 - 0.47308734787878
    ) * sin_2_azimuth
    basis_values[..., 19] = (
        z * (-4.683325804901025 * z2 + 2.007139630671868) * y
    )
    basis_values[..., 20] = 1.984313483298443 * z2 * (
        1.865881662950577 * z2 - 1.119528997770346
    ) - 1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
    basis_values[..., 21] = (
        z * (-4.683325804901025 * z2 + 2.007139630671868) * x
    )
    basis_values[..., 22] = (
        3.31161143515146 * z2 - 0.47308734787878
    ) * cos_2_azimuth
    basis_values[..., 23] = -1.770130769779931 * z * cos_3_azimuth
    basis_values[..., 24] = 0.6258357354491763 * cos_4_azimuth

    return (basis_values[..., None] * coefficients).sum(dim=-2)


@app.function
def coefficients_to_tensor(config: object) -> Tensor:
    """Flatten nested per-degree SH blocks into the standard 25x3 layout."""
    blocks = [
        torch.as_tensor(config.sh_0.coefficients, dtype=torch.float32),
        torch.as_tensor(config.sh_1.coefficients, dtype=torch.float32),
        torch.as_tensor(config.sh_2.coefficients, dtype=torch.float32),
        torch.as_tensor(config.sh_3.coefficients, dtype=torch.float32),
        torch.as_tensor(config.sh_4.coefficients, dtype=torch.float32),
    ]
    return torch.cat(blocks, dim=0)


@app.function
def sh_to_rgb(sh_values: Tensor) -> UInt8[Tensor, "*batch 3"]:
    """Convert raw degree-0 SH coefficients into displayable uint8 RGB values."""
    rgb = torch.clamp((sh_values * 0.2820947917738781) + 0.5, 0.0, 1.0)
    return (rgb * 255.0).to(torch.uint8)


@app.function
def evaluated_sh_to_rgb(sh_values: Tensor) -> UInt8[Tensor, "*batch 3"]:
    """Convert evaluated SH values into displayable uint8 RGB values."""
    rgb = torch.clamp(sh_values + 0.5, 0.0, 1.0)
    return (rgb * 255.0).to(torch.uint8)


@app.function
def rgb_to_sh(rgb: Tensor) -> Float[Tensor, "*batch 3"]:
    """Convert range [0,1] RGB to sh_0 coeffs."""
    return (rgb - 0.5) / 0.2820947917738781


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ## Visualization

    `Point mode` colors the whole sphere with the SH-evaluated RGB for the
    current view direction. `Axis preview mode` uses sphere directions as the
    query directions, so the full angular color field becomes visible.
    """)
    return


@app.cell
def _(Config):
    gui_widget = form_gui(Config, value=Config(), live_update=True)
    gui_widget
    return (gui_widget,)


@app.cell
def _(widget):
    widget
    return


@app.cell
def _(gui_widget):
    config = gui_widget.value  # if gui_widget.value is not None else Config()
    return (config,)


@app.cell
def _(config):
    config.sh_0.coefficients
    return


@app.cell
def _(config):
    config.sh_1.coefficients
    return


@app.cell
def _(config):
    config.sh_2.coefficients
    return


@app.cell
def _(config):
    config.sh_3.coefficients
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    ## Rendering Code
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Math
    """)
    return


@app.function
def orbit_position(
    position: Float[Tensor, "3"],
    orbit_radius: float,
) -> Float[Tensor, "3"]:
    """Project a torch camera offset back onto the fixed orbit sphere."""
    norm = torch.linalg.norm(position)
    fallback = torch.tensor([1.0, 1.0, 0.6], device=position.device)
    direction = torch.where(norm < 1e-6, fallback, position)
    return F.normalize(direction, dim=0) * orbit_radius


@app.function
def intersect_sphere(
    ray_origin: Float[Tensor, "3"],
    ray_dirs: Float[Tensor, "height width 3"],
    center: Float[Tensor, "3"],
    radius: float,
) -> tuple[Float[Tensor, "height width"], torch.Tensor]:
    """Intersect a ray bundle with a sphere and return depth and hit mask."""
    origin_offset = ray_origin - center
    offset_dot_dir = torch.einsum("i,hwi->hw", origin_offset, ray_dirs)
    offset_norm_sq = torch.dot(origin_offset, origin_offset)
    discriminant = offset_dot_dir.square() - (offset_norm_sq - radius**2)
    valid = discriminant >= 0.0
    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0.0))
    t_near = -offset_dot_dir - sqrt_discriminant
    t_far = -offset_dot_dir + sqrt_discriminant
    t = torch.where(t_near > 0.0, t_near, t_far)
    hit_mask = valid & (t > 0.0)
    t = torch.where(hit_mask, t, torch.full_like(t, torch.inf))
    return t, hit_mask


@app.function
def image_grid(
    width: int,
    height: int,
    device: torch.device,
) -> Float[Tensor, "height width 2"]:
    """Create a pixel-center image grid on the target device."""
    xs = torch.arange(width, device=device, dtype=torch.float32)
    ys = torch.arange(height, device=device, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack((grid_x, grid_y), dim=-1) + 0.5


@app.function
def camera_rays(
    c2w: Float[Tensor, "4 4"],
    K: Float[Tensor, "3 3"],
    width: int,
    height: int,
) -> Float[Tensor, "height width 3"]:
    """Construct normalized world-space ray directions for each pixel."""
    grid = image_grid(width, height, c2w.device)
    pixel_h = torch.cat(
        (
            grid,
            torch.ones((height, width, 1), device=c2w.device),
        ),
        dim=-1,
    )
    camera_dirs = torch.einsum("ij,hwj->hwi", torch.linalg.inv(K), pixel_h)
    ray_dirs = torch.einsum("ij,hwj->hwi", c2w[:3, :3], camera_dirs)
    return F.normalize(ray_dirs, dim=-1)


@app.function
def render_spheres(
    c2w: Float[Tensor, "4 4"],
    K: Float[Tensor, "3 3"],
    coefficients: Float[Tensor, "25 3"],
    degrees_to_use: int,
    orbit_radius: float,
    sphere_radius: float,
    width: int,
    height: int,
) -> UInt8[Tensor, "height width 3"]:
    """Render split-screen spheres with a canonical view direction."""
    left_width = width // 2
    right_width = width - left_width
    sphere_center = torch.zeros(3, dtype=torch.float32, device=c2w.device)
    ray_origin = orbit_position(c2w[:3, 3], orbit_radius)
    image = torch.zeros(
        (height, width, 3),
        dtype=torch.uint8,
        device=c2w.device,
    )

    def split_intrinsics(
        split_width: int,
        split_K: Float[Tensor, "3 3"],
    ) -> Float[Tensor, "3 3"]:
        """Center the principal point for a split-screen sub-view."""
        subview_K = split_K.clone()
        subview_K[0, 2] = split_width * 0.5
        return subview_K

    left_ray_dirs = camera_rays(
        c2w,
        split_intrinsics(left_width, K),
        left_width,
        height,
    )
    left_t, left_hit = intersect_sphere(
        ray_origin,
        left_ray_dirs,
        sphere_center,
        sphere_radius,
    )
    point_rgb = evaluated_sh_to_rgb(
        spherical_harmonics(
            ray_origin[None, :],
            coefficients,
            degrees_to_use,
        )
    )[0]
    image[:, :left_width, :] = torch.where(
        left_hit[..., None],
        point_rgb,
        image[:, :left_width, :],
    )

    right_ray_dirs = camera_rays(
        c2w,
        split_intrinsics(right_width, K),
        right_width,
        height,
    )
    right_t, right_hit = intersect_sphere(
        ray_origin,
        right_ray_dirs,
        sphere_center,
        sphere_radius,
    )
    axis_t = torch.where(right_hit, right_t, torch.zeros_like(right_t))
    axis_points = ray_origin[None, None, :] + axis_t[..., None] * right_ray_dirs
    axis_rgb = evaluated_sh_to_rgb(
        spherical_harmonics(
            axis_points[right_hit],
            coefficients,
            degrees_to_use,
        )
    )
    image[:, left_width:, :][right_hit] = axis_rgb
    return image


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Viewer
    """)
    return


@app.cell
def _(state):
    def render_fn(
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
        coefficients: Tensor,
        degrees_to_use: int,
    ) -> np.ndarray:
        """Marshal viewer inputs to torch, run the compiled path, and convert back."""
        width = render_tab_state.viewer_width
        height = render_tab_state.viewer_height
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c2w = torch.as_tensor(
            camera_state.c2w, dtype=torch.float32, device=device
        )
        K = torch.as_tensor(
            camera_state.get_K([width, height]),
            dtype=torch.float32,
            device=device,
        )
        # compiled_render_spheres = torch.compile(render_spheres)
        image = render_spheres(
            c2w,
            K,
            coefficients.to(device=device, dtype=torch.float32),
            degrees_to_use,
            float(state["orbit_radius"]),
            float(state["sphere_radius"]),
            width,
            height,
        )
        return image.detach().cpu().numpy()

    return (render_fn,)


@app.cell
def _():
    state = {
        "coefficients": torch.zeros((25, 3), dtype=torch.float32),
        "degrees_to_use": 3,
        "orbit_radius": 6.0,
        "sphere_radius": 0.9,
    }
    return (state,)


@app.cell
def _(state):
    def orbit_camera_position(position: np.ndarray) -> np.ndarray:
        """Project a numpy camera offset back onto the fixed orbit sphere."""
        direction = np.asarray(position, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([1.0, 1.0, 0.6], dtype=np.float32)
            norm = float(np.linalg.norm(direction))
        return direction / max(norm, 1e-6) * state["orbit_radius"]

    return (orbit_camera_position,)


@app.cell
def _(orbit_camera_position, server, state, viewer, widget):
    status_handle = server.gui.add_markdown(
        "**Background:** black  \n"
        "**Left sphere:** `point`  \n"
        "**Right sphere:** `axis_preview`  \n"
        f"**Active degree:** `{state['degrees_to_use']}`"
    )

    @server.on_client_connect
    def _(client: object) -> None:
        """Initialize and constrain each connected client camera."""
        syncing_camera = False
        client.camera.look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        client.camera.position = np.array(
            [0.0, 0.0, state["orbit_radius"]],
            dtype=np.float32,
        )

        def _snap_camera() -> None:
            """Reset translation to the canonical orbit and convert radius changes into FOV."""
            nonlocal syncing_camera
            if syncing_camera:
                return
            look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            relative_position = np.asarray(
                client.camera.position, dtype=np.float32
            ) - np.asarray(client.camera.look_at, dtype=np.float32)
            distance = float(np.linalg.norm(relative_position))
            safe_distance = max(distance, 1e-6)
            zoomed_fov = 2.0 * np.arctan(
                np.tan(float(client.camera.fov) * 0.5)
                * safe_distance
                / float(state["orbit_radius"])
            )
            zoomed_fov = float(
                np.clip(zoomed_fov, np.deg2rad(5.0), np.deg2rad(175.0))
            )
            client_id = getattr(
                client, "client_id", getattr(client, "id", None)
            )
            camera_state = widget.get_camera_state(client_id=client_id)
            syncing_camera = True
            try:
                widget.set_camera_state(
                    replace(
                        camera_state,
                        position=orbit_camera_position(
                            relative_position
                        ).astype(np.float64),
                        look_at=look_at.astype(np.float64),
                        fov=zoomed_fov,
                    ),
                    client_id=client_id,
                    update_reset_view=True,
                    sync_gui=True,
                )
            finally:
                syncing_camera = False

        _snap_camera()

        @client.camera.on_update
        def _(_camera: object) -> None:
            """Cancel translation changes and rerender after orbit updates."""
            _snap_camera()
            viewer.rerender(None)

        viewer.rerender(None)

    return


@app.cell
def _(config, state, viewer):
    if config is not None:
        state["coefficients"] = coefficients_to_tensor(config)
        state["degrees_to_use"] = config.degrees_to_use
        viewer.rerender(None)
    return


@app.cell
def _(render_fn, state):
    def bound_render_fn(
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ) -> np.ndarray:
        """Bind the current notebook state into the viewer render callback."""
        return render_fn(
            camera_state,
            render_tab_state,
            state["coefficients"],
            state["degrees_to_use"],
        )

    server, viewer, widget = viser_marimo(
        render_fn=bound_render_fn,
        height=680,
    )
    return server, viewer, widget


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    ## Configuration

    Coefficients are grouped by degree using the same real SH ordering used
    in 3DGS. `degrees_to_use` truncates evaluation without hiding the
    higher-order tabs.

    Degree-0 coefficients use the 3DGS-style mapping `clip(sh * C0 + 0.5, 0, 1)`.
    Evaluated SH values use `clip(value + 0.5, 0, 1)`.
    """)
    return


@app.cell
def _():
    class Config(BaseModel):
        """Configuration for the SH sphere demo."""

        sh_0: SH0 = Field(default_factory=SH0)
        sh_1: SH1 = Field(default_factory=SH1)
        sh_2: SH2 = Field(default_factory=SH2)
        sh_3: SH3 = Field(default_factory=SH3)
        sh_4: SH4 = Field(default_factory=SH4)
        degrees_to_use: int = Field(
            3,
            ge=0,
            le=4,
            description="Maximum SH degree used during evaluation.",
        )

    return (Config,)


@app.function
def default_coefficients(rows: int) -> np.ndarray:
    """Create zero SH coefficients for a neutral gray sphere."""
    return np.ones((rows, 3), dtype=np.float32) * -0.0


@app.class_definition
class SH0(BaseModel):
    """Degree-0 SH coefficients."""

    coefficients: PyArray[Float, float, "1 3"] = Field(
        default_factory=lambda: default_coefficients(1),
        description=(
            "Degree-0 RGB coefficient block. Zero gives neutral gray. "
            "Because this is an SH coefficient, a saturated color usually "
            "also needs negative values in the other channels."
        ),
        json_schema_extra={
            "matrix_min": -1.8,
            "matrix_max": 1.8,
            "matrix_step": 0.025,
        },
    )


@app.class_definition
class SH1(BaseModel):
    """Degree-1 SH coefficients."""

    coefficients: PyArray[Float, float, "3 3"] = Field(
        default_factory=lambda: default_coefficients(3),
        description="Degree-1 RGB coefficient block.",
        json_schema_extra={
            "matrix_min": -2.0,
            "matrix_max": 2.0,
            "matrix_step": 0.025,
        },
    )


@app.class_definition
class SH2(BaseModel):
    """Degree-2 SH coefficients."""

    coefficients: PyArray[Float, float, "5 3"] = Field(
        default_factory=lambda: default_coefficients(5),
        description="Degree-2 RGB coefficient block.",
        json_schema_extra={
            "matrix_min": -2.0,
            "matrix_max": 2.0,
            "matrix_step": 0.025,
        },
    )


@app.class_definition
class SH3(BaseModel):
    """Degree-3 SH coefficients."""

    coefficients: PyArray[Float, float, "7 3"] = Field(
        default_factory=lambda: default_coefficients(7),
        description="Degree-3 RGB coefficient block.",
        json_schema_extra={
            "matrix_min": -2.0,
            "matrix_max": 2.0,
            "matrix_step": 0.025,
        },
    )


@app.class_definition
class SH4(BaseModel):
    """Degree-4 SH coefficients."""

    coefficients: PyArray[Float, float, "9 3"] = Field(
        default_factory=lambda: default_coefficients(9),
        description="Degree-4 RGB coefficient block.",
        json_schema_extra={
            "matrix_min": -2.0,
            "matrix_max": 2.0,
            "matrix_step": 0.025,
        },
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
