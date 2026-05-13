"""Core SVRaster rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as torch_functional
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
from ember_native_svraster.core.runtime import utils as runtime_utils


@dataclass(frozen=True)
class SVRasterCoreRenderOutput(RenderOutput):
    """Base SVRaster render output."""


@dataclass(frozen=True)
class SVRasterCoreDepthRenderOutput(SVRasterCoreRenderOutput, HasDepth):
    """SVRaster render output with depth."""

    depth: Tensor


@dataclass(frozen=True)
class SVRasterCoreTrainingRenderOutput(SVRasterCoreDepthRenderOutput):
    """SVRaster render output with native training statistics."""

    normal: Tensor | None = None
    transmittance: Tensor | None = None
    raw_transmittance: Tensor | None = None
    max_weight: Tensor | None = None


@dataclass(frozen=True)
class SVRasterCoreRenderOptions(RenderOptions):
    """SVRaster-specific render configuration."""

    near_plane: float = 0.02
    color_mode: Literal["sh", "dontcare"] = "sh"
    samples_per_voxel: int = 1
    supersampling: float = 1.0
    return_transmittance: bool = False
    track_max_weight: bool = False
    random_background: bool = False
    white_background: bool = False
    black_background: bool = False
    color_concentration_weight: float = 0.0
    ascending_weight: float = 0.0
    distortion_weight: float = 0.0
    ground_truth_color: Tensor | None = None
    debug: bool = False


def _background_scalar(options: SVRasterCoreRenderOptions) -> float:
    if options.white_background:
        return 1.0
    return 0.0


def _build_raster_settings(
    camera: CameraState,
    camera_index: int,
    supersampling: float,
) -> dict[str, float | Tensor | int]:
    intrinsics = camera.get_intrinsics()[camera_index]
    base_width = float(camera.width[camera_index].item())
    base_height = float(camera.height[camera_index].item())
    width = round(base_width * supersampling)
    height = round(base_height * supersampling)
    fx = float(intrinsics[0, 0].item())
    fy = float(intrinsics[1, 1].item())
    camera_to_world = camera.cam_to_world[camera_index].to(
        dtype=torch.float32,
    )
    return {
        "image_width": width,
        "image_height": height,
        "tanfovx": (base_width * 0.5) / fx,
        "tanfovy": (base_height * 0.5) / fy,
        "cx": float(intrinsics[0, 2].item()) * supersampling,
        "cy": float(intrinsics[1, 2].item()) * supersampling,
        "world_to_camera": torch.linalg.inv(camera_to_world),
        "camera_to_world": camera_to_world,
    }


def _validate_inputs(scene: SparseVoxelScene, camera: CameraState) -> None:
    if scene.backend_name not in SUPPORTED_SVRASTER_BACKENDS:
        raise ValueError(
            f"Unsupported SparseVoxelScene backend {scene.backend_name!r}. "
            f"Supported backends: {sorted(SUPPORTED_SVRASTER_BACKENDS)}."
        )
    if scene.octpath.device.type != "cuda":
        raise ValueError("svraster.core requires scene tensors on CUDA.")
    native_max_num_levels = runtime_utils.max_num_levels()
    if scene.max_num_levels != native_max_num_levels:
        raise ValueError(
            "svraster.core requires SparseVoxelScene.max_num_levels to match "
            f"the native backend MAX_NUM_LEVELS={native_max_num_levels}; "
            f"got {scene.max_num_levels}."
        )
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
    return_normal: bool,
) -> tuple[
    Tensor,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor | None,
    Tensor | None,
]:
    voxel_geometries = scene.voxel_geometries
    raster_settings = _build_raster_settings(
        camera,
        camera_index,
        options.supersampling,
    )
    ground_truth_color = options.ground_truth_color
    if ground_truth_color is not None and ground_truth_color.numel() > 0:
        if ground_truth_color.ndim == 4:
            ground_truth_color = ground_truth_color[camera_index]
        if ground_truth_color.shape[-1] == 3:
            ground_truth_color = ground_truth_color.permute(2, 0, 1).contiguous()
        if options.supersampling != 1.0:
            target_size = (
                int(raster_settings["image_height"]),
                int(raster_settings["image_width"]),
            )
            ground_truth_color = torch_functional.interpolate(
                ground_truth_color.unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
    render_result = render_runtime(
        active_sh_degree=scene.active_sh_degree,
        image_width=int(raster_settings["image_width"]),
        image_height=int(raster_settings["image_height"]),
        tanfovx=float(raster_settings["tanfovx"]),
        tanfovy=float(raster_settings["tanfovy"]),
        cx=float(raster_settings["cx"]),
        cy=float(raster_settings["cy"]),
        world_to_camera=raster_settings["world_to_camera"].to(
            device=scene.octpath.device
        ),
        camera_to_world=raster_settings["camera_to_world"].to(
            device=scene.octpath.device
        ),
        near=options.near_plane,
        background_color=_background_scalar(options),
        octree_paths=scene.octpath.reshape(-1),
        voxel_centers=scene.vox_center,
        voxel_lengths=scene.vox_size.reshape(-1),
        voxel_geometries=voxel_geometries,
        sh0=scene.sh0,
        shs=scene.shs,
        return_depth=return_depth or options.distortion_weight > 0.0,
        return_normal=return_normal,
        track_max_weight=options.track_max_weight,
        samples_per_voxel=options.samples_per_voxel,
        subdivision_priority=scene.resolved_subdivision_priority,
        color_concentration_weight=options.color_concentration_weight,
        ascending_weight=options.ascending_weight,
        distortion_weight=options.distortion_weight,
        ground_truth_color=ground_truth_color,
        debug=options.debug,
        color_mode=options.color_mode,
    )
    color = render_result.color
    transmittance = render_result.transmittance
    raw_transmittance = transmittance
    if options.random_background:
        background = torch.rand(
            (3, 1, 1),
            dtype=color.dtype,
            device=color.device,
        )
        color = color + transmittance * background
    elif not options.white_background and not options.black_background:
        background = color.mean(dim=(1, 2), keepdim=True)
        color = color + transmittance * background
    if options.supersampling != 1.0:
        target_size = (
            int(camera.height[camera_index].item()),
            int(camera.width[camera_index].item()),
        )
        color = torch_functional.interpolate(
            color.unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        if render_result.depth.numel() > 0:
            depth = torch_functional.interpolate(
                render_result.depth.unsqueeze(0),
                size=target_size,
                mode="nearest",
            ).squeeze(0)
        else:
            depth = render_result.depth
        if render_result.normal.numel() > 0:
            normal = torch_functional.interpolate(
                render_result.normal.unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            normal = None
        transmittance = torch_functional.interpolate(
            transmittance.unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    else:
        depth = render_result.depth
        normal = render_result.normal if render_result.normal.numel() > 0 else None
    rgb = color.permute(1, 2, 0).contiguous().clamp(0.0, 1.0)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth.squeeze(0)
    return (
        rgb,
        depth,
        normal.permute(1, 2, 0).contiguous() if normal is not None else None,
        transmittance.squeeze(0),
        raw_transmittance.squeeze(0),
        render_result.max_weight if render_result.max_weight.numel() > 0 else None,
    )


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
    if return_2d_projections:
        raise ValueError("svraster.core does not expose 2D projections.")
    if return_projective_intersection_transforms:
        raise ValueError(
            "svraster.core does not expose projective intersection transforms."
        )

    _validate_inputs(scene, camera)
    resolved_options = options or SVRasterCoreRenderOptions()
    renders: list[Tensor] = []
    depths: list[Tensor] = []
    normals: list[Tensor] = []
    transmittances: list[Tensor] = []
    raw_transmittances: list[Tensor] = []
    max_weights: list[Tensor] = []
    for camera_index in range(camera.cam_to_world.shape[0]):
        render, depth, normal, transmittance, raw_transmittance, max_weight = (
            _render_single_camera(
                scene,
                camera,
                camera_index,
                resolved_options,
                return_depth=return_depth,
                return_normal=return_normals,
            )
        )
        renders.append(render)
        depths.append(depth)
        if normal is not None:
            normals.append(normal)
        if resolved_options.return_transmittance:
            transmittances.append(transmittance)
            raw_transmittances.append(raw_transmittance)
        if max_weight is not None:
            max_weights.append(max_weight)

    stacked_render = torch.stack(renders, dim=0)
    needs_training_output = (
        return_normals
        or resolved_options.return_transmittance
        or resolved_options.track_max_weight
    )
    if not return_depth and not needs_training_output:
        return SVRasterCoreRenderOutput(render=stacked_render)
    stacked_depth = torch.stack(depths, dim=0)
    if not needs_training_output:
        return SVRasterCoreDepthRenderOutput(
            render=stacked_render,
            depth=stacked_depth,
        )
    return SVRasterCoreTrainingRenderOutput(
        render=stacked_render,
        depth=stacked_depth,
        normal=torch.stack(normals, dim=0) if normals else None,
        transmittance=(
            torch.stack(transmittances, dim=0) if transmittances else None
        ),
        raw_transmittance=(
            torch.stack(raw_transmittances, dim=0)
            if raw_transmittances
            else None
        ),
        max_weight=torch.stack(max_weights, dim=0) if max_weights else None,
    )
