"""Ember-native FasterGS depth backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import torch
from beartype import beartype
from ember_core.core.capabilities import HasDepth
from ember_core.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
)
from ember_core.core.registry import register_backend
from jaxtyping import Float
from torch import Tensor

from ember_native_faster_gs.faster_gs.renderer import (
    FasterGSNativeRenderOptions,
    _split_sh_coefficients,
    _validate_inputs,
    render_faster_gs_native,
)
from ember_native_faster_gs.faster_gs_depth.runtime import (
    render as render_runtime,
)

_SUPPORTED_OUTPUTS = frozenset({"depth"})
DepthTensor = Float[Tensor, " num_cams height width"]


@beartype
@dataclass(frozen=True)
class FasterGSDepthNativeRenderOutput(RenderOutput):
    """Base render output for the FasterGS depth backend."""


@beartype
@dataclass(frozen=True)
class FasterGSDepthNativeDepthRenderOutput(
    FasterGSDepthNativeRenderOutput,
    HasDepth,
):
    """Depth-capable render output for the FasterGS depth backend."""

    depth: DepthTensor


@beartype
@dataclass(frozen=True)
class FasterGSDepthNativeRenderOptions(RenderOptions):
    """Render configuration for the FasterGS depth backend."""

    near_plane: float = 0.01
    far_plane: float = 1000.0
    proper_antialiasing: bool = True


@overload
def render_faster_gs_depth(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: Literal[False] = False,
    return_depth: Literal[False] = False,
    return_normals: Literal[False] = False,
    return_2d_projections: Literal[False] = False,
    return_projective_intersection_transforms: Literal[False] = False,
    options: FasterGSDepthNativeRenderOptions | None = None,
) -> FasterGSDepthNativeRenderOutput: ...


@overload
def render_faster_gs_depth(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: Literal[False] = False,
    return_depth: Literal[True] = True,
    return_normals: Literal[False] = False,
    return_2d_projections: Literal[False] = False,
    return_projective_intersection_transforms: Literal[False] = False,
    options: FasterGSDepthNativeRenderOptions | None = None,
) -> FasterGSDepthNativeDepthRenderOutput: ...


@beartype
def render_faster_gs_depth(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = False,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: FasterGSDepthNativeRenderOptions | None = None,
) -> FasterGSDepthNativeRenderOutput | FasterGSDepthNativeDepthRenderOutput:
    """Render a scene with the FasterGS depth proof backend."""
    if return_alpha:
        raise ValueError(
            "The faster_gs_depth backend does not expose alpha output."
        )
    if return_gaussian_impact_score:
        raise ValueError(
            "The faster_gs_depth backend does not expose Gaussian "
            "impact scores."
        )
    if return_normals:
        raise ValueError("The faster_gs_depth backend does not expose normals.")
    if return_2d_projections:
        raise ValueError(
            "The faster_gs_depth backend does not expose 2D projections."
        )
    if return_projective_intersection_transforms:
        raise ValueError(
            "The faster_gs_depth backend does not expose projective "
            "intersection transforms."
        )

    _validate_inputs(scene, camera)
    options = options or FasterGSDepthNativeRenderOptions()
    if not return_depth:
        rgb_only = render_faster_gs_native(
            scene,
            camera,
            options=FasterGSNativeRenderOptions(
                background_color=options.background_color,
                near_plane=options.near_plane,
                far_plane=options.far_plane,
                proper_antialiasing=options.proper_antialiasing,
            ),
        )
        return FasterGSDepthNativeRenderOutput(render=rgb_only.render)

    sh_coefficients_0, sh_coefficients_rest = _split_sh_coefficients(scene)
    intrinsics = camera.get_intrinsics()
    background_color = options.background_color.to(
        device=scene.center_position.device,
        dtype=scene.center_position.dtype,
    )
    renders: list[Tensor] = []
    depths: list[Tensor] = []
    for camera_index in range(camera.cam_to_world.shape[0]):
        cam_to_world = camera.cam_to_world[camera_index]
        world_2_camera = torch.linalg.inv(cam_to_world)
        camera_intrinsics = intrinsics[camera_index]
        render_result = render_runtime(
            scene.center_position.contiguous(),
            scene.log_scales.contiguous(),
            scene.quaternion_orientation.contiguous(),
            scene.logit_opacity[:, None].contiguous(),
            sh_coefficients_0.contiguous(),
            sh_coefficients_rest.contiguous(),
            world_2_camera.contiguous(),
            cam_to_world[:3, 3].contiguous(),
            near_plane=options.near_plane,
            far_plane=options.far_plane,
            width=int(camera.width[camera_index].item()),
            height=int(camera.height[camera_index].item()),
            focal_x=float(camera_intrinsics[0, 0].item()),
            focal_y=float(camera_intrinsics[1, 1].item()),
            center_x=float(camera_intrinsics[0, 2].item()),
            center_y=float(camera_intrinsics[1, 2].item()),
            bg_color=background_color,
            proper_antialiasing=options.proper_antialiasing,
            active_sh_bases=int(scene.feature.shape[1]),
        )
        renders.append(render_result.image.permute(1, 2, 0).contiguous())
        depths.append(render_result.depth.contiguous())
    return FasterGSDepthNativeDepthRenderOutput(
        render=torch.stack(renders, dim=0).clamp(0.0, 1.0),
        depth=torch.stack(depths, dim=0),
    )


def register() -> None:
    """Register the FasterGS depth backend."""
    register_backend(
        name="faster_gs.depth",
        default_options=FasterGSDepthNativeRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=_SUPPORTED_OUTPUTS,
    )(render_faster_gs_depth)
