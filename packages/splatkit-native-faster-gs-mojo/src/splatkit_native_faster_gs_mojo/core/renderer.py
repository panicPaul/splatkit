"""Splatkit-native FasterGS Mojo backend adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from beartype import beartype
from jaxtyping import Float
from splatkit.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
)
from splatkit.core.registry import register_backend
from torch import Tensor

from splatkit_native_faster_gs.faster_gs.renderer import _split_sh_coefficients
from splatkit_native_faster_gs.faster_gs.renderer import _validate_inputs
from splatkit_native_faster_gs_mojo.core.runtime import render as render_runtime

_SUPPORTED_OUTPUTS = frozenset()


@beartype
@dataclass(frozen=True)
class FasterGSMojoRenderOutput(RenderOutput):
    """Mojo-backed FasterGS render output."""


@beartype
@dataclass(frozen=True)
class FasterGSMojoRenderOptions(RenderOptions):
    """Mojo-backed FasterGS render configuration."""

    near_plane: float = 0.01
    far_plane: float = 1000.0
    proper_antialiasing: bool = True


@beartype
def render_faster_gs_mojo(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = False,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: FasterGSMojoRenderOptions | None = None,
) -> FasterGSMojoRenderOutput:
    """Render a scene with the FasterGS Mojo runtime."""
    if return_alpha:
        raise ValueError("The faster_gs_mojo backend does not expose alpha output.")
    if return_depth:
        raise ValueError("The faster_gs_mojo backend does not expose depth output.")
    if return_gaussian_impact_score:
        raise ValueError(
            "The faster_gs_mojo backend does not expose Gaussian impact scores."
        )
    if return_normals:
        raise ValueError("The faster_gs_mojo backend does not expose normals.")
    if return_2d_projections:
        raise ValueError(
            "The faster_gs_mojo backend does not expose 2D Gaussian projections."
        )
    if return_projective_intersection_transforms:
        raise ValueError(
            "The faster_gs_mojo backend does not expose projective intersection "
            "transforms."
        )

    _validate_inputs(scene, camera)
    options = options or FasterGSMojoRenderOptions()
    sh_coefficients_0, sh_coefficients_rest = _split_sh_coefficients(scene)
    intrinsics = camera.get_intrinsics()
    background_color = options.background_color.to(
        device=scene.center_position.device,
        dtype=scene.center_position.dtype,
    )
    renders: list[Float[Tensor, " height width 3"]] = []

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

    return FasterGSMojoRenderOutput(
        render=torch.stack(renders, dim=0).clamp(0.0, 1.0)
    )


def register() -> None:
    """Register the FasterGS Mojo backend."""
    register_backend(
        name="faster_gs_mojo.core",
        default_options=FasterGSMojoRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=_SUPPORTED_OUTPUTS,
    )(render_faster_gs_mojo)
