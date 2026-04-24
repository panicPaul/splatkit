"""Splatkit-native FasterGS backend adapter."""

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

from splatkit_native_faster_gs.faster_gs.runtime import (
    render as render_runtime,
)

_SUPPORTED_OUTPUTS = frozenset()


@beartype
@dataclass(frozen=True)
class FasterGSNativeRenderOutput(RenderOutput):
    """Native FasterGS render output."""


@beartype
@dataclass(frozen=True)
class FasterGSNativeRenderOptions(RenderOptions):
    """Native FasterGS render configuration."""

    near_plane: float = 0.01
    far_plane: float = 1000.0
    proper_antialiasing: bool = True


@beartype
def _split_sh_coefficients(
    scene: GaussianScene3D,
) -> tuple[
    Float[Tensor, " num_splats 1 3"],
    Float[Tensor, " num_splats sh_coeffs_minus_one 3"],
]:
    if scene.feature.ndim != 3:
        raise ValueError(
            "faster_gs expects spherical harmonics with shape "
            f"(num_splats, sh_coeffs, 3); got {tuple(scene.feature.shape)}."
        )
    if scene.feature.shape[1] < 1:
        raise ValueError("faster_gs requires at least one SH basis.")
    return scene.feature[:, :1, :], scene.feature[:, 1:, :]


@beartype
def _validate_inputs(scene: GaussianScene3D, camera: CameraState) -> None:
    if scene.center_position.device.type != "cuda":
        raise ValueError("faster_gs requires scene tensors on CUDA.")
    if camera.cam_to_world.device.type != "cuda":
        raise ValueError("faster_gs requires camera tensors on CUDA.")
    if camera.camera_convention != "opencv":
        raise ValueError(
            "faster_gs currently expects cameras in opencv convention; "
            f"got {camera.camera_convention!r}."
        )
    if scene.log_scales.shape[-1] != 3:
        raise ValueError(
            "faster_gs only supports 3D Gaussian scales with shape "
            f"(num_splats, 3); got {tuple(scene.log_scales.shape)}."
        )


@beartype
def render_faster_gs_native(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = False,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: FasterGSNativeRenderOptions | None = None,
) -> FasterGSNativeRenderOutput:
    """Render a scene with the native FasterGS runtime."""
    if return_alpha:
        raise ValueError("The faster_gs backend does not expose alpha output.")
    if return_depth:
        raise ValueError("The faster_gs backend does not expose depth output.")
    if return_gaussian_impact_score:
        raise ValueError(
            "The faster_gs backend does not expose Gaussian impact scores."
        )
    if return_normals:
        raise ValueError("The faster_gs backend does not expose normals.")
    if return_2d_projections:
        raise ValueError(
            "The faster_gs backend does not expose 2D Gaussian projections."
        )
    if return_projective_intersection_transforms:
        raise ValueError(
            "The faster_gs backend does not expose projective intersection "
            "transforms."
        )

    _validate_inputs(scene, camera)
    options = options or FasterGSNativeRenderOptions()
    sh_coefficients_0, sh_coefficients_rest = _split_sh_coefficients(scene)
    intrinsics = camera.get_intrinsics()
    background_color = options.background_color.to(
        device=scene.center_position.device,
        dtype=scene.center_position.dtype,
    )
    renders: list[Tensor] = []

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

    return FasterGSNativeRenderOutput(
        render=torch.stack(renders, dim=0).clamp(0.0, 1.0)
    )


def register() -> None:
    """Register the native FasterGS backend."""
    register_backend(
        name="faster_gs.core",
        default_options=FasterGSNativeRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=_SUPPORTED_OUTPUTS,
    )(render_faster_gs_native)
