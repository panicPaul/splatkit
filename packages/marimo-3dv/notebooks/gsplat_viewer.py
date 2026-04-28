# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.0",
#     "numpy==2.4.4",
#     "marimo-3dv",
#     "gsplat==1.5.3",
#     "jaxtyping==0.3.9",
#     "torch==2.11.0",
#     "plyfile",
#     "pydantic",
# ]
#
# [tool.uv.sources]
# marimo-3dv = { path = "..", editable = true }
# ///

"""Interactive 3D Gaussian splatting viewer using gsplat and marimo-3dv."""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")

with app.setup:
    import gc
    from dataclasses import dataclass
    from math import isqrt
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import torch
    from gsplat import rasterization
    from jaxtyping import Float
    from plyfile import PlyData
    from pydantic import BaseModel, Field
    from torch import Tensor

    from marimo_config_gui import config_gui
    from marimo_3dv import (
        CameraState,
        RenderResult,
        Viewer,
        ViewerState,
        apply_viewer_pipeline_config,
        gs_backend_bundle,
        viewer_pipeline_controls_gui,
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    # 3DGS Viewer
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Viewer
    """)
    return


@app.cell
def _(load_form):
    load_form
    return


@app.cell
def _():
    viewer_state = ViewerState(
        camera_convention="opencv",
        interactive_quality=50,
        interactive_max_side=1980,
        internal_render_max_side=3840,
    )
    return (viewer_state,)


@app.cell
def _(viewer_state):
    (
        viewer_state.set_show_origin(False)
        .set_show_stats(True)
        .set_show_horizon(False)
        .set_show_axes(True)
    )
    return


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
    viewer_controls_default_config = viewer_controls.default_config
    viewer_controls_gui
    return viewer_controls_default_config, viewer_controls_gui


@app.cell
def _(viewer):
    viewer
    return


@app.cell
def _(
    pipeline_result,
    viewer_controls_gui,
    viewer_state,
):
    cache = {"config_json": None, "render_fn": None}

    def render_frame(camera_state):
        combined_config = viewer_controls_gui.value
        config_json = combined_config.model_dump_json()
        if cache["render_fn"] is None or config_json != cache["config_json"]:
            pipeline_config = apply_viewer_pipeline_config(
                viewer_state,
                combined_config,
            )
            cache["render_fn"] = pipeline_result.bind(
                pipeline_config,
                backend_fn=rasterize_scene,
            )
            cache["config_json"] = config_json
        return cache["render_fn"](camera_state).image

    viewer = Viewer(
        render_frame,
        state=viewer_state,
        controls=viewer_controls_gui,
    )
    return (viewer,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Scene Definition
    """)
    return


@app.class_definition
@dataclass
class SplatScene:
    """Minimal 3DGS scene loaded from a PLY file."""

    center_positions: Float[Tensor, "num_splats 3"]
    log_half_extents: Float[Tensor, "num_splats 3"]
    quaternion_orientation: Float[Tensor, "num_splats 4"]
    spherical_harmonics: Float[Tensor, "num_splats num_bases 3"]
    opacity_logits: Float[Tensor, "num_splats 1"]
    sh_degree: int


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Rasterization
    """)
    return


@app.function
@torch.no_grad()
def rasterize_scene(
    camera: CameraState, scene: SplatScene | None
) -> RenderResult:
    """Render a SplatScene from the given camera using gsplat rasterization."""
    if scene is None:
        return RenderResult(
            image=np.full((camera.height, camera.width, 3), 245, dtype=np.uint8)
        )

    gsplat_camera = camera.with_convention("opencv")
    device = scene.center_positions.device
    w2c = np.linalg.inv(gsplat_camera.cam_to_world)
    viewmats = torch.from_numpy(w2c).to(device=device, dtype=torch.float32)[
        None
    ]

    half_fov_rad = np.radians(gsplat_camera.fov_degrees / 2)
    focal = (gsplat_camera.height / 2) / np.tan(half_fov_rad)
    K = torch.tensor(
        [
            [focal, 0.0, gsplat_camera.width / 2],
            [0.0, focal, gsplat_camera.height / 2],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )[None]

    render_colors, _render_alphas, meta = rasterization(
        means=scene.center_positions,
        quats=scene.quaternion_orientation,
        scales=torch.exp(scene.log_half_extents),
        opacities=torch.sigmoid(scene.opacity_logits.squeeze(-1)),
        colors=scene.spherical_harmonics,
        viewmats=viewmats,
        Ks=K,
        width=gsplat_camera.width,
        height=gsplat_camera.height,
        sh_degree=scene.sh_degree,
    )
    image = render_colors[0].clamp(0.0, 1.0).cpu().numpy()
    image_uint8 = (image * 255).astype(np.uint8)

    metadata: dict = {}
    if "means2d" in meta:
        metadata["projected_means"] = meta["means2d"][0]

    return RenderResult(image=image_uint8, metadata=metadata)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## GUI Definition
    """)
    return


@app.cell
def _():
    class GpuLoadOptions(BaseModel):
        """Notebook-only controls for scene replacement behavior."""

        close_existing_viewer: bool = Field(
            default=True,
            description="Close the active viewer before loading a new scene.",
        )
        empty_cuda_cache: bool = Field(
            default=True,
            description="Release unused CUDA allocator cache before reload.",
        )

    class LoadConfig(BaseModel):
        """Configuration for loading a PLY file."""

        ply_path: Path = Field(
            default=Path.cwd() / "point_cloud.ply",
            description="Path to a 3DGS-style `.ply` file.",
        )
        gpu: GpuLoadOptions = Field(
            default_factory=GpuLoadOptions,
            description="How to clean up GPU resources before replacing a scene.",
        )

    load_form = config_gui(
        LoadConfig, value=LoadConfig(), submit_label="Load File"
    )
    return (load_form,)


@app.function
def format_num_bytes(num_bytes: int) -> str:
    """Format a byte count in GiB with two decimals."""
    gib = num_bytes / (1024**3)
    return f"{gib:.2f} GiB"


@app.function
def cuda_memory_snapshot() -> dict[str, int | bool]:
    """Return a lightweight snapshot of current CUDA allocator state."""
    if not torch.cuda.is_available():
        return {
            "available": False,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
        }

    return {
        "available": True,
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


@app.function
def format_cuda_memory_report(
    before_cleanup: dict[str, int | bool],
    after_cleanup: dict[str, int | bool],
    after_load: dict[str, int | bool],
) -> str:
    """Render a markdown report for the latest scene-replacement cycle."""
    if not bool(after_load["available"]):
        return "CUDA is unavailable."

    lines = [
        "| Phase | Allocated | Reserved | Peak Allocated | Peak Reserved |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, snapshot in (
        ("Before Cleanup", before_cleanup),
        ("After Cleanup", after_cleanup),
        ("After Load", after_load),
    ):
        lines.append(
            "| "
            f"{label} | "
            f"{format_num_bytes(int(snapshot['allocated_bytes']))} | "
            f"{format_num_bytes(int(snapshot['reserved_bytes']))} | "
            f"{format_num_bytes(int(snapshot['max_allocated_bytes']))} | "
            f"{format_num_bytes(int(snapshot['max_reserved_bytes']))} |"
        )
    return "\n".join(lines)


@app.function
def cleanup_before_scene_reload(
    viewer_state: ViewerState,
    *,
    close_existing_viewer: bool,
    empty_cuda_cache: bool,
) -> tuple[dict[str, int | bool], dict[str, int | bool]]:
    """Tear down viewer-owned resources before replacing the scene."""
    before_cleanup = cuda_memory_snapshot()

    if close_existing_viewer:
        active_ref = viewer_state._active_marimo_viewer_ref
        active_viewer = None if active_ref is None else active_ref()
        if active_viewer is not None:
            active_viewer.close()

    gc.collect()

    if torch.cuda.is_available() and empty_cuda_cache:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    after_cleanup = cuda_memory_snapshot()
    return before_cleanup, after_cleanup


@app.cell
def _(load_form, viewer_state):
    if mo.running_in_notebook():
        load_config = load_form.value
        before_cleanup, after_cleanup = cleanup_before_scene_reload(
            viewer_state,
            close_existing_viewer=(
                load_config.gpu.close_existing_viewer
                if load_config is not None
                else True
            ),
            empty_cuda_cache=(
                load_config.gpu.empty_cuda_cache
                if load_config is not None
                else True
            ),
        )
        scene = (
            load_splat_scene(load_config.ply_path)
            if load_config is not None and load_config.ply_path.exists()
            else None
        )
        after_load = cuda_memory_snapshot()
        load_report = format_cuda_memory_report(
            before_cleanup,
            after_cleanup,
            after_load,
        )
    else:
        import subprocess

        result = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--title=Open PLY file",
                "--file-filter=*.ply",
            ],
            capture_output=True,
            text=True,
        )
        ply_path = Path(result.stdout.strip())
        scene = load_splat_scene(ply_path) if ply_path.exists() else None
        snapshot = cuda_memory_snapshot()
        load_report = format_cuda_memory_report(
            snapshot,
            snapshot,
            snapshot,
        )
    return load_report, scene


@app.cell
def _(load_report):
    mo.md(f"""
    ## CUDA Memory

    {load_report}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## PLY Loading
    """)
    return


@app.function
def infer_sh_degree(num_bases: int) -> int:
    """Infer the SH degree from the number of basis functions."""
    degree = isqrt(num_bases) - 1
    if (degree + 1) ** 2 != num_bases:
        raise ValueError(f"Invalid SH basis count: {num_bases}")
    if not 0 <= degree <= 4:
        raise ValueError(f"Only SH degrees 0-4 are supported, got {degree}")
    return degree


@app.function
def get_gsplat_device() -> torch.device:
    """Return the device used for gsplat rasterization."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "gsplat viewer requires CUDA, but CUDA is unavailable."
        )
    return torch.device("cuda")


@app.function
def load_splat_scene(path: Path) -> SplatScene:
    """Load a 3DGS-style `.ply` file into a SplatScene."""
    device = get_gsplat_device()
    ply_data = PlyData.read(path)
    vertices = ply_data["vertex"]
    property_names = list(vertices.data.dtype.names)

    centers = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]],
        axis=1,
    ).astype(np.float32)

    dc_coefficients = np.stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]],
        axis=1,
    ).astype(np.float32)

    rest_feature_names = sorted(
        [name for name in property_names if name.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    num_rest_coefficients = len(rest_feature_names)
    if num_rest_coefficients % 3 != 0:
        raise ValueError(
            "Expected the number of `f_rest_*` attributes to be divisible by 3."
        )

    num_bases = 1 + num_rest_coefficients // 3
    sh_degree = infer_sh_degree(num_bases)
    sh_coefficients = np.zeros(
        (centers.shape[0], num_bases, 3), dtype=np.float32
    )
    sh_coefficients[:, 0, :] = dc_coefficients
    if rest_feature_names:
        rest_coefficients = np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in rest_feature_names
            ],
            axis=1,
        )
        rest_coefficients = rest_coefficients.reshape(
            centers.shape[0], 3, num_bases - 1
        )
        sh_coefficients[:, 1:num_bases, :] = np.transpose(
            rest_coefficients, (0, 2, 1)
        )

    scale_feature_names = sorted(
        [name for name in property_names if name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    rotation_feature_names = sorted(
        [name for name in property_names if name.startswith("rot")],
        key=lambda name: int(name.split("_")[-1]),
    )
    log_scales = (
        np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in scale_feature_names
            ],
            axis=1,
        )
        if scale_feature_names
        else np.full((centers.shape[0], 3), np.log(0.01), dtype=np.float32)
    )
    rotations = (
        np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in rotation_feature_names
            ],
            axis=1,
        )
        if rotation_feature_names
        else np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (centers.shape[0], 1),
        )
    )
    opacity_logits = np.asarray(vertices["opacity"], dtype=np.float32)[:, None]

    return SplatScene(
        center_positions=torch.from_numpy(centers).to(device=device),
        log_half_extents=torch.from_numpy(log_scales).to(device=device),
        quaternion_orientation=torch.from_numpy(rotations).to(device=device),
        spherical_harmonics=torch.from_numpy(sh_coefficients).to(device=device),
        opacity_logits=torch.from_numpy(opacity_logits).to(device=device),
        sh_degree=sh_degree,
    )


if __name__ == "__main__":
    app.run()
