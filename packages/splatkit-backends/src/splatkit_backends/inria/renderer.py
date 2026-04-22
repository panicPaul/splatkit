"""Thin splatkit adapter for the Inria rasterizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from splatkit.core.capabilities import HasDepth
from splatkit.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
)
from splatkit.core.registry import register_backend
from torch import Tensor

_SUPPORTED_OUTPUTS = frozenset({"depth"})


@dataclass(frozen=True)
class InriaRenderOutput(RenderOutput):
    """Base Inria render output."""


@dataclass(frozen=True)
class InriaDepthRenderOutput(InriaRenderOutput, HasDepth):
    """Inria render output with depth."""

    depth: Tensor


@dataclass(frozen=True)
class InriaRenderOptions(RenderOptions):
    """Inria-specific render configuration."""

    scale_modifier: float = 1.0
    prefiltered: bool = False
    debug: bool = False
    antialiasing: bool = False
    near_plane: float = 0.01
    far_plane: float = 1000.0


def _normalized_quaternions(scene: GaussianScene3D) -> Tensor:
    """Return unit quaternions for rasterizers that expect normalized input."""
    quaternion_norms = torch.linalg.vector_norm(
        scene.quaternion_orientation,
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(scene.quaternion_orientation.dtype).eps)
    return scene.quaternion_orientation / quaternion_norms


def _camera_to_world_to_view(cam_to_world: Tensor) -> Tensor:
    return torch.linalg.inv(cam_to_world).transpose(0, 1).contiguous()


def _projection_matrix(
    tanfovx: float,
    tanfovy: float,
    znear: float,
    zfar: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    matrix = torch.zeros((4, 4), device=device, dtype=dtype)
    matrix[0, 0] = 1.0 / tanfovx
    matrix[1, 1] = 1.0 / tanfovy
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    matrix[3, 2] = 1.0
    return matrix.transpose(0, 1).contiguous()


def _build_raster_settings(
    scene: GaussianScene3D,
    camera: CameraState,
    camera_index: int,
    options: InriaRenderOptions,
) -> GaussianRasterizationSettings:
    intrinsics = camera.get_intrinsics()[camera_index]
    width = int(camera.width[camera_index].item())
    height = int(camera.height[camera_index].item())
    fx = float(intrinsics[0, 0].item())
    fy = float(intrinsics[1, 1].item())
    tanfovx = (width * 0.5) / fx
    tanfovy = (height * 0.5) / fy
    cam_to_world = camera.cam_to_world[camera_index]
    viewmatrix = _camera_to_world_to_view(cam_to_world)
    projmatrix = _projection_matrix(
        tanfovx,
        tanfovy,
        options.near_plane,
        options.far_plane,
        device=scene.center_position.device,
        dtype=scene.center_position.dtype,
    )
    full_proj_transform = viewmatrix @ projmatrix
    return GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=options.background_color.to(
            device=scene.center_position.device,
            dtype=scene.center_position.dtype,
        ),
        scale_modifier=options.scale_modifier,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_transform,
        sh_degree=scene.sh_degree,
        campos=cam_to_world[:3, 3].contiguous(),
        prefiltered=options.prefiltered,
        debug=options.debug,
        antialiasing=options.antialiasing,
    )


def _validate_inputs(scene: GaussianScene3D, camera: CameraState) -> None:
    if scene.center_position.device.type != "cuda":
        raise ValueError("The Inria rasterizer requires scene tensors on CUDA.")
    if camera.cam_to_world.device.type != "cuda":
        raise ValueError(
            "The Inria rasterizer requires camera tensors on CUDA."
        )
    if camera.camera_convention != "opencv":
        raise ValueError(
            "The Inria rasterizer wrapper expects OpenCV cameras; got "
            f"{camera.camera_convention!r}."
        )
    if scene.log_scales.shape[-1] != 3:
        raise ValueError(
            "The Inria rasterizer requires 3D Gaussian scales with shape "
            f"(num_splats, 3); got {tuple(scene.log_scales.shape)}."
        )
    if camera.width.shape[0] != camera.cam_to_world.shape[0]:
        raise ValueError("Camera batch fields must share the same batch size.")
    if camera.height.shape[0] != camera.cam_to_world.shape[0]:
        raise ValueError("Camera batch fields must share the same batch size.")
    if camera.fov_degrees.shape[0] != camera.cam_to_world.shape[0]:
        raise ValueError("Camera batch fields must share the same batch size.")
    if not (0.0 < camera.fov_degrees).all():
        raise ValueError("Camera field of view values must be positive.")


def _empty_like_scene(scene: GaussianScene3D, columns: int) -> Tensor:
    return torch.empty(
        (0, columns),
        dtype=scene.center_position.dtype,
        device=scene.center_position.device,
    )


def _scene_feature_inputs(scene: GaussianScene3D) -> tuple[Tensor, Tensor]:
    if scene.feature.ndim == 2:
        return _empty_like_scene(scene, 0), scene.feature.contiguous()
    if scene.feature.ndim == 3:
        colors = _empty_like_scene(scene, 0)
        return scene.feature.contiguous(), colors
    raise ValueError(
        "Expected scene.feature to have shape (num_splats, feature_dim) or "
        "(num_splats, sh_coeffs, 3); got "
        f"{tuple(scene.feature.shape)}."
    )


def _render_single_camera(
    scene: GaussianScene3D,
    camera: CameraState,
    camera_index: int,
    options: InriaRenderOptions,
) -> tuple[Tensor, Tensor]:
    rasterizer = GaussianRasterizer(
        raster_settings=_build_raster_settings(
            scene=scene,
            camera=camera,
            camera_index=camera_index,
            options=options,
        )
    )
    shs, colors_precomp = _scene_feature_inputs(scene)
    means2d = torch.zeros(
        (scene.center_position.shape[0], 3),
        dtype=scene.center_position.dtype,
        device=scene.center_position.device,
    )
    render, _radii, invdepth = rasterizer(
        means3D=scene.center_position,
        means2D=means2d,
        opacities=torch.sigmoid(scene.logit_opacity),
        shs=shs if shs.numel() else None,
        colors_precomp=colors_precomp if colors_precomp.numel() else None,
        scales=torch.exp(scene.log_scales),
        rotations=_normalized_quaternions(scene),
        cov3D_precomp=None,
    )
    rgb = render.permute(1, 2, 0).contiguous()
    inverse_depth = invdepth.squeeze(0)
    depth = torch.where(
        inverse_depth > 0,
        inverse_depth.reciprocal(),
        torch.zeros_like(inverse_depth),
    )
    return rgb, depth


@overload
def render_inria(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: Literal[False] = False,
    return_depth: Literal[False] = False,
    return_normals: Literal[False] = False,
    return_2d_projections: Literal[False] = False,
    return_projective_intersection_transforms: Literal[False] = False,
    options: InriaRenderOptions | None = None,
) -> InriaRenderOutput: ...


@overload
def render_inria(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: Literal[False] = False,
    return_depth: Literal[True] = True,
    return_normals: Literal[False] = False,
    return_2d_projections: Literal[False] = False,
    return_projective_intersection_transforms: Literal[False] = False,
    options: InriaRenderOptions | None = None,
) -> InriaDepthRenderOutput: ...


def render_inria(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = False,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: InriaRenderOptions | None = None,
) -> InriaRenderOutput | InriaDepthRenderOutput:
    """Render a scene with the Inria differential Gaussian rasterizer."""
    if return_alpha:
        raise ValueError("The Inria backend does not expose alpha output.")
    if return_gaussian_impact_score:
        raise ValueError(
            "The Inria backend does not expose Gaussian impact scores."
        )
    if return_normals:
        raise ValueError("The Inria backend does not expose normals.")
    if return_2d_projections:
        raise ValueError(
            "The Inria backend does not expose 2D Gaussian projections."
        )
    if return_projective_intersection_transforms:
        raise ValueError(
            "The Inria backend does not expose projective intersection "
            "transforms."
        )

    _validate_inputs(scene, camera)
    options = options or InriaRenderOptions()

    renders: list[Tensor] = []
    depths: list[Tensor] = []
    for camera_index in range(camera.cam_to_world.shape[0]):
        render, depth = _render_single_camera(
            scene=scene,
            camera=camera,
            camera_index=camera_index,
            options=options,
        )
        renders.append(render)
        depths.append(depth)

    stacked_renders = torch.stack(renders, dim=0)
    if not return_depth:
        return InriaRenderOutput(render=stacked_renders)

    stacked_depths = torch.stack(depths, dim=0)
    return InriaDepthRenderOutput(render=stacked_renders, depth=stacked_depths)


def register() -> None:
    """Register the Inria backend in the global splatkit registry."""
    register_backend(
        name="inria",
        default_options=InriaRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=_SUPPORTED_OUTPUTS,
    )(render_inria)


__all__ = [
    "InriaDepthRenderOutput",
    "InriaRenderOptions",
    "InriaRenderOutput",
    "register",
    "render_inria",
]
