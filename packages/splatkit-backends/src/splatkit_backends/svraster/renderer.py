"""Backend-only SV Raster adapter for splatkit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from beartype import beartype
from new_svraster_cuda import renderer as svraster_renderer
from splatkit.core.capabilities import HasDepth
from splatkit.core.contracts import (
    CameraState,
    RenderOptions,
    RenderOutput,
    SparseVoxelScene,
)
from splatkit.core.registry import register_backend
from splatkit.core.sparse_voxel import SUPPORTED_SVRASTER_BACKENDS
from torch import Tensor

_SUPPORTED_OUTPUTS = frozenset({"depth"})


@dataclass(frozen=True)
class SVRasterRenderOutput(RenderOutput):
    """Base SV Raster render output."""


@dataclass(frozen=True)
class SVRasterDepthRenderOutput(SVRasterRenderOutput, HasDepth):
    """SV Raster render output with depth."""

    depth: Tensor


@dataclass(frozen=True)
class SVRasterRenderOptions(RenderOptions):
    """SV Raster-specific render configuration."""

    near_plane: float = 0.02
    color_mode: Literal["sh"] = "sh"


def _background_scalar(options: SVRasterRenderOptions) -> float:
    return float(options.background_color.mean().item())


def _build_raster_settings(
    scene: SparseVoxelScene,
    camera: CameraState,
    camera_index: int,
    options: SVRasterRenderOptions,
    *,
    need_depth: bool,
) -> svraster_renderer.RasterSettings:
    intrinsics = camera.get_intrinsics()[camera_index]
    width = int(camera.width[camera_index].item())
    height = int(camera.height[camera_index].item())
    fx = float(intrinsics[0, 0].item())
    fy = float(intrinsics[1, 1].item())
    cam_to_world = camera.cam_to_world[camera_index].to(
        device=scene.octpath.device,
        dtype=torch.float32,
    )
    return svraster_renderer.RasterSettings(
        color_mode=options.color_mode,
        n_samp_per_vox=1,
        image_width=width,
        image_height=height,
        tanfovx=(width * 0.5) / fx,
        tanfovy=(height * 0.5) / fy,
        cx=float(intrinsics[0, 2].item()),
        cy=float(intrinsics[1, 2].item()),
        w2c_matrix=torch.linalg.inv(cam_to_world),
        c2w_matrix=cam_to_world,
        bg_color=_background_scalar(options),
        near=options.near_plane,
        need_depth=need_depth,
    )


def _validate_inputs(scene: SparseVoxelScene, camera: CameraState) -> None:
    if scene.backend_name not in SUPPORTED_SVRASTER_BACKENDS:
        raise ValueError(
            f"Unsupported SparseVoxelScene backend {scene.backend_name!r}. "
            f"Supported backends: {sorted(SUPPORTED_SVRASTER_BACKENDS)}."
        )
    if scene.octpath.device.type != "cuda":
        raise ValueError("SV Raster requires scene tensors on CUDA.")
    if camera.cam_to_world.device.type != "cuda":
        raise ValueError("SV Raster requires camera tensors on CUDA.")
    if camera.camera_convention != "opencv":
        raise ValueError(
            "SV Raster currently expects cameras in opencv convention; got "
            f"{camera.camera_convention!r}."
        )


def _render_single_camera(
    scene: SparseVoxelScene,
    camera: CameraState,
    camera_index: int,
    options: SVRasterRenderOptions,
    *,
    return_depth: bool,
) -> tuple[Tensor, Tensor]:
    geos = scene.voxel_geometries

    def vox_fn(
        _idx: Tensor, cam_pos: Tensor, _color_mode: str
    ) -> dict[str, Tensor]:
        rgbs = svraster_renderer.SH_eval.apply(
            scene.active_sh_degree,
            None,
            scene.vox_center,
            cam_pos,
            None,
            scene.sh0,
            scene.shs,
        )
        subdiv_p = torch.ones(
            (scene.num_voxels, 1),
            dtype=scene.sh0.dtype,
            device=scene.sh0.device,
        )
        return {"geos": geos, "rgbs": rgbs, "subdiv_p": subdiv_p}

    color, depth, _normal, _transmittance, _max_w = (
        svraster_renderer.rasterize_voxels(
            _build_raster_settings(
                scene,
                camera,
                camera_index,
                options,
                need_depth=return_depth,
            ),
            scene.octpath.reshape(-1),
            scene.vox_center,
            scene.vox_size.reshape(-1),
            vox_fn,
        )
    )
    rgb = color.permute(1, 2, 0).contiguous().clamp(0.0, 1.0)
    if depth.ndim == 3 and depth.shape[0] == 1:
        return rgb, depth.squeeze(0)
    return rgb, depth


@beartype
def render_svraster(
    scene: SparseVoxelScene,
    camera: CameraState,
    *,
    return_alpha: bool = False,
    return_depth: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: SVRasterRenderOptions | None = None,
) -> SVRasterRenderOutput | SVRasterDepthRenderOutput:
    """Render a sparse-voxel scene with the backend-only SV Raster adapter."""
    if return_alpha:
        raise ValueError("The SV Raster backend does not expose alpha output.")
    if return_normals:
        raise ValueError("The SV Raster backend does not expose normals.")
    if return_2d_projections:
        raise ValueError(
            "The SV Raster backend does not expose 2D projections."
        )
    if return_projective_intersection_transforms:
        raise ValueError(
            "The SV Raster backend does not expose projective intersection "
            "transforms."
        )

    _validate_inputs(scene, camera)
    resolved_options = options or SVRasterRenderOptions()
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
        return SVRasterRenderOutput(render=stacked_render)
    return SVRasterDepthRenderOutput(
        render=stacked_render,
        depth=torch.stack(depths, dim=0),
    )


def register() -> None:
    """Register the SV Raster backend in the global splatkit registry."""
    register_backend(
        name="svraster",
        default_options=SVRasterRenderOptions(),
        accepted_scene_types=(SparseVoxelScene,),
        supported_outputs=_SUPPORTED_OUTPUTS,
    )(render_svraster)
