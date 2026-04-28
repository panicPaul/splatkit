"""Ember-native FasterGS Mojo backend adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from beartype import beartype
from ember_core.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
)
from ember_core.core.registry import register_backend
from ember_native_faster_gs.faster_gs.renderer import (
    _split_sh_coefficients,
    _validate_inputs,
)
from jaxtyping import Float
from torch import Tensor

from ember_native_faster_gs_mojo.core.runtime import render as render_runtime
from ember_native_faster_gs_mojo.core.runtime.ops.blend import (
    blend_image_only,
)
from ember_native_faster_gs_mojo.core.runtime.ops.preprocess import (
    preprocess_fwd_op,
)
from ember_native_faster_gs_mojo.core.runtime.ops.render import (
    _render_image_capacity,
)
from ember_native_faster_gs_mojo.core.runtime.ops.sort import sort_fwd_op

_SUPPORTED_OUTPUTS = frozenset()


def _scene_needs_grad(scene: GaussianScene3D) -> bool:
    return torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (
            scene.center_position,
            scene.log_scales,
            scene.quaternion_orientation,
            scene.logit_opacity,
            scene.feature,
        )
    )


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
        raise ValueError(
            "The faster_gs_mojo backend does not expose alpha output."
        )
    if return_depth:
        raise ValueError(
            "The faster_gs_mojo backend does not expose depth output."
        )
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
    needs_grad = _scene_needs_grad(scene)
    renders: list[Float[Tensor, " height width 3"]] = []

    for camera_index in range(camera.cam_to_world.shape[0]):
        cam_to_world = camera.cam_to_world[camera_index]
        world_2_camera = torch.linalg.inv(cam_to_world)
        camera_intrinsics = intrinsics[camera_index]
        width = int(camera.width[camera_index].item())
        height = int(camera.height[camera_index].item())
        focal_x = float(camera_intrinsics[0, 0].item())
        focal_y = float(camera_intrinsics[1, 1].item())
        center_x = float(camera_intrinsics[0, 2].item())
        center_y = float(camera_intrinsics[1, 2].item())
        active_sh_bases = int(scene.feature.shape[1])
        center_positions = scene.center_position.contiguous()
        log_scales = scene.log_scales.contiguous()
        unnormalized_rotations = scene.quaternion_orientation.contiguous()
        opacities = scene.logit_opacity[:, None].contiguous()
        sh_coefficients_0_contiguous = sh_coefficients_0.contiguous()
        sh_coefficients_rest_contiguous = sh_coefficients_rest.contiguous()
        world_2_camera_contiguous = world_2_camera.contiguous()
        camera_position = cam_to_world[:3, 3].contiguous()

        if needs_grad:
            render_result = render_runtime(
                center_positions,
                log_scales,
                unnormalized_rotations,
                opacities,
                sh_coefficients_0_contiguous,
                sh_coefficients_rest_contiguous,
                world_2_camera_contiguous,
                camera_position,
                near_plane=options.near_plane,
                far_plane=options.far_plane,
                width=width,
                height=height,
                focal_x=focal_x,
                focal_y=focal_y,
                center_x=center_x,
                center_y=center_y,
                bg_color=background_color,
                proper_antialiasing=options.proper_antialiasing,
                active_sh_bases=active_sh_bases,
            )
            image = render_result.image
        else:
            width_capacity, height_capacity, tile_capacity = (
                _render_image_capacity(
                    device=scene.center_position.device,
                    width=width,
                    height=height,
                )
            )
            preprocess_outputs = preprocess_fwd_op(
                center_positions,
                log_scales,
                unnormalized_rotations,
                opacities,
                sh_coefficients_0_contiguous,
                sh_coefficients_rest_contiguous,
                world_2_camera_contiguous,
                camera_position,
                options.near_plane,
                options.far_plane,
                width,
                height,
                focal_x,
                focal_y,
                center_x,
                center_y,
                options.proper_antialiasing,
                active_sh_bases,
            )
            sort_outputs = sort_fwd_op(
                preprocess_outputs[4],
                preprocess_outputs[5],
                preprocess_outputs[6],
                preprocess_outputs[7],
                preprocess_outputs[0],
                preprocess_outputs[1],
                preprocess_outputs[8],
                preprocess_outputs[9],
                width_capacity,
                height_capacity,
                tile_count_minimum=max(4096, tile_capacity),
                capacity_headroom_numerator=3,
                capacity_headroom_denominator=2,
                return_capacity=True,
            )
            image = blend_image_only(
                sort_outputs[0],
                sort_outputs[1],
                sort_outputs[2],
                sort_outputs[3],
                preprocess_outputs[0],
                preprocess_outputs[1],
                preprocess_outputs[2],
                background_color,
                width=width,
                height=height,
                stable_extent=True,
            )
        renders.append(image.permute(1, 2, 0).contiguous())

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
