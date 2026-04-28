"""Core SVRaster rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from beartype import beartype
from ember_core.core.capabilities import HasDepth
from ember_core.core.contracts import (
    CameraState,
    RenderOptions,
    RenderOutput,
    SparseVoxelScene,
)
from ember_core.core.sparse_voxel import SUPPORTED_SVRASTER_BACKENDS
from torch import Tensor

from ember_native_svraster.core.runtime import render as render_runtime


@dataclass(frozen=True)
class SVRasterCoreRenderOutput(RenderOutput):
    """Base SVRaster render output."""


@dataclass(frozen=True)
class SVRasterCoreDepthRenderOutput(SVRasterCoreRenderOutput, HasDepth):
    """SVRaster render output with depth."""

    depth: Tensor


@dataclass(frozen=True)
class SVRasterCoreRenderOptions(RenderOptions):
    """SVRaster-specific render configuration."""

    near_plane: float = 0.02
    color_mode: Literal["sh"] = "sh"


def _background_scalar(options: SVRasterCoreRenderOptions) -> float:
    return float(options.background_color.mean().item())


def _build_raster_settings(
    camera: CameraState,
    camera_index: int,
) -> dict[str, float | Tensor | int]:
    intrinsics = camera.get_intrinsics()[camera_index]
    width = int(camera.width[camera_index].item())
    height = int(camera.height[camera_index].item())
    fx = float(intrinsics[0, 0].item())
    fy = float(intrinsics[1, 1].item())
    cam_to_world = camera.cam_to_world[camera_index].to(
        dtype=torch.float32,
    )
    return {
        "image_width": width,
        "image_height": height,
        "tanfovx": (width * 0.5) / fx,
        "tanfovy": (height * 0.5) / fy,
        "cx": float(intrinsics[0, 2].item()),
        "cy": float(intrinsics[1, 2].item()),
        "w2c_matrix": torch.linalg.inv(cam_to_world),
        "c2w_matrix": cam_to_world,
    }


def _validate_inputs(scene: SparseVoxelScene, camera: CameraState) -> None:
    if scene.backend_name not in SUPPORTED_SVRASTER_BACKENDS:
        raise ValueError(
            f"Unsupported SparseVoxelScene backend {scene.backend_name!r}. "
            f"Supported backends: {sorted(SUPPORTED_SVRASTER_BACKENDS)}."
        )
    if scene.octpath.device.type != "cuda":
        raise ValueError("svraster.core requires scene tensors on CUDA.")
    if camera.cam_to_world.device.type != "cuda":
        raise ValueError("svraster.core requires camera tensors on CUDA.")
    if camera.camera_convention != "opencv":
        raise ValueError(
            "svraster.core currently expects cameras in opencv convention; "
            f"got {camera.camera_convention!r}."
        )


def _render_single_camera(
    scene: SparseVoxelScene,
    camera: CameraState,
    camera_index: int,
    options: SVRasterCoreRenderOptions,
    *,
    return_depth: bool,
) -> tuple[Tensor, Tensor]:
    geos = scene.voxel_geometries
    raster_settings = _build_raster_settings(
        camera,
        camera_index,
    )
    render_result = render_runtime(
        active_sh_degree=scene.active_sh_degree,
        image_width=int(raster_settings["image_width"]),
        image_height=int(raster_settings["image_height"]),
        tanfovx=float(raster_settings["tanfovx"]),
        tanfovy=float(raster_settings["tanfovy"]),
        cx=float(raster_settings["cx"]),
        cy=float(raster_settings["cy"]),
        w2c_matrix=raster_settings["w2c_matrix"].to(device=scene.octpath.device),
        c2w_matrix=raster_settings["c2w_matrix"].to(device=scene.octpath.device),
        near=options.near_plane,
        bg_color=_background_scalar(options),
        octree_paths=scene.octpath.reshape(-1),
        vox_centers=scene.vox_center,
        vox_lengths=scene.vox_size.reshape(-1),
        geos=geos,
        sh0=scene.sh0,
        shs=scene.shs,
        need_depth=return_depth,
    )
    rgb = render_result.color.permute(1, 2, 0).contiguous().clamp(0.0, 1.0)
    depth = render_result.depth
    if depth.ndim == 3 and depth.shape[0] == 1:
        return rgb, depth.squeeze(0)
    return rgb, depth


@beartype
def render_svraster_core(
    scene: SparseVoxelScene,
    camera: CameraState,
    *,
    return_alpha: bool = False,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: SVRasterCoreRenderOptions | None = None,
) -> SVRasterCoreRenderOutput | SVRasterCoreDepthRenderOutput:
    """Render a sparse-voxel scene with the SVRaster core runtime."""
    if return_alpha:
        raise ValueError("svraster.core does not expose alpha output.")
    if return_gaussian_impact_score:
        raise ValueError(
            "svraster.core does not expose Gaussian impact scores."
        )
    if return_normals:
        raise ValueError("svraster.core does not expose normals.")
    if return_2d_projections:
        raise ValueError("svraster.core does not expose 2D projections.")
    if return_projective_intersection_transforms:
        raise ValueError(
            "svraster.core does not expose projective intersection "
            "transforms."
        )

    _validate_inputs(scene, camera)
    resolved_options = options or SVRasterCoreRenderOptions()
    renders: list[Tensor] = []
    depths: list[Tensor] = []
    for camera_index in range(camera.cam_to_world.shape[0]):
        render, depth = _render_single_camera(
            scene,
            camera,
            camera_index,
            resolved_options,
            return_depth=return_depth,
        )
        renders.append(render)
        depths.append(depth)

    stacked_render = torch.stack(renders, dim=0)
    if not return_depth:
        return SVRasterCoreRenderOutput(render=stacked_render)
    return SVRasterCoreDepthRenderOutput(
        render=stacked_render,
        depth=torch.stack(depths, dim=0),
    )
