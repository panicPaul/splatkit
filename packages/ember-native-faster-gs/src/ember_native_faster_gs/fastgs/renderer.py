"""Ember-native FastGS backend adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from beartype import beartype
from ember_core.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOutput,
)
from ember_core.core.registry import output_set, register_backend
from ember_core.densification.contracts import GaussianMetricAttribution
from jaxtyping import Float, Int
from torch import Tensor

from ember_native_faster_gs.faster_gs.renderer import (
    FasterGSNativeRenderOptions,
    _split_sh_coefficients,
    _validate_inputs,
)
from ember_native_faster_gs.fastgs.runtime import (
    blend_metric_counts,
    preprocess,
    sort,
)
from ember_native_faster_gs.fastgs.runtime import (
    render as render_runtime,
)

_SUPPORTED_OUTPUTS = output_set()


@beartype
@dataclass(frozen=True)
class FastGSNativeRenderOutput(RenderOutput):
    """Native FastGS render output."""


@beartype
@dataclass(frozen=True)
class FastGSNativeDensificationRenderOutput(FastGSNativeRenderOutput):
    """Native FastGS output with densification accumulators."""

    densification_info: Float[Tensor, " 4 num_splats"]


@beartype
@dataclass(frozen=True)
class FastGSNativeRenderOptions(FasterGSNativeRenderOptions):
    """Native FastGS render configuration."""

    mip_splatting_screen_filter: bool = False
    compact_box_scale: float = 0.5


@beartype
@dataclass(frozen=True)
class FastGSNativeGaussianMetricAttribution(GaussianMetricAttribution):
    """FastGS metric-map attribution trait provider."""

    def attribute_metric_map(
        self,
        scene: GaussianScene3D,
        camera: CameraState,
        metric_map: Int[Tensor, " *metric_dims"],
        *,
        options: FastGSNativeRenderOptions | None = None,
    ) -> Float[Tensor, " num_splats"]:
        """Attribute a single-camera FastGS metric map to Gaussians."""
        if camera.cam_to_world.shape[0] != 1:
            raise ValueError(
                "FastGS metric attribution expects a single probe camera."
            )
        if metric_map.ndim != 2:
            raise ValueError(
                "FastGS metric attribution expects a 2D metric map with "
                f"shape (height, width); got {tuple(metric_map.shape)}."
            )

        _validate_inputs(scene, camera)
        resolved_options = options or FastGSNativeRenderOptions()
        active_sh_bases = (
            int(scene.feature.shape[1])
            if resolved_options.active_sh_bases is None
            else resolved_options.active_sh_bases
        )
        sh_coefficients_0, sh_coefficients_rest = _split_sh_coefficients(scene)
        intrinsics = camera.get_intrinsics()[0]
        width = int(camera.width[0].item())
        height = int(camera.height[0].item())
        cam_to_world = camera.cam_to_world[0]
        world_2_camera = torch.linalg.inv(cam_to_world)
        background_color = resolved_options.background_color.to(
            device=scene.center_position.device,
            dtype=scene.center_position.dtype,
        )
        metric_map_flat = metric_map.reshape(-1).to(
            device=scene.center_position.device,
            dtype=torch.int32,
        )

        preprocess_result = preprocess(
            scene.center_position.contiguous(),
            scene.log_scales.contiguous(),
            scene.quaternion_orientation.contiguous(),
            scene.logit_opacity[:, None].contiguous(),
            sh_coefficients_0.contiguous(),
            sh_coefficients_rest.contiguous(),
            world_2_camera.contiguous(),
            cam_to_world[:3, 3].contiguous(),
            near_plane=resolved_options.near_plane,
            far_plane=resolved_options.far_plane,
            width=width,
            height=height,
            focal_x=float(intrinsics[0, 0].item()),
            focal_y=float(intrinsics[1, 1].item()),
            center_x=float(intrinsics[0, 2].item()),
            center_y=float(intrinsics[1, 2].item()),
            mip_splatting_screen_filter=(
                resolved_options.mip_splatting_screen_filter
            ),
            active_sh_bases=active_sh_bases,
            compact_box_scale=(resolved_options.compact_box_scale),
        )
        sort_result = sort(
            preprocess_result.depth_keys,
            preprocess_result.primitive_indices,
            preprocess_result.num_touched_tiles,
            preprocess_result.screen_bounds,
            preprocess_result.projected_means,
            preprocess_result.conic_opacity,
            preprocess_result.visible_count,
            preprocess_result.instance_count,
            width=width,
            height=height,
            compact_box_scale=(resolved_options.compact_box_scale),
        )
        metric_counts = blend_metric_counts(
            sort_result.instance_primitive_indices,
            sort_result.tile_instance_ranges,
            sort_result.tile_bucket_offsets,
            sort_result.bucket_count,
            preprocess_result.projected_means,
            preprocess_result.conic_opacity,
            preprocess_result.colors_rgb,
            background_color,
            metric_map_flat,
            resolved_options.mip_splatting_screen_filter,
            image_width=width,
            image_height=height,
        )
        return metric_counts.to(dtype=scene.center_position.dtype)


def _reject_unsupported_outputs(
    *,
    return_alpha: bool,
    return_depth: bool,
    return_gaussian_impact_score: bool,
    return_normals: bool,
    return_2d_projections: bool,
    return_projective_intersection_transforms: bool,
) -> None:
    if return_alpha:
        raise ValueError("The fastgs backend does not expose alpha output.")
    if return_depth:
        raise ValueError("The fastgs backend does not expose depth output.")
    if return_gaussian_impact_score:
        raise ValueError(
            "The fastgs backend does not expose Gaussian impact scores."
        )
    if return_normals:
        raise ValueError("The fastgs backend does not expose normals.")
    if return_2d_projections:
        raise ValueError(
            "The fastgs backend does not expose 2D Gaussian projections."
        )
    if return_projective_intersection_transforms:
        raise ValueError(
            "The fastgs backend does not expose projective intersection "
            "transforms."
        )


@beartype
def render_fastgs_native(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = False,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: FastGSNativeRenderOptions | None = None,
) -> FastGSNativeRenderOutput:
    """Render a scene with the native FastGS experiment backend."""
    _reject_unsupported_outputs(
        return_alpha=return_alpha,
        return_depth=return_depth,
        return_gaussian_impact_score=return_gaussian_impact_score,
        return_normals=return_normals,
        return_2d_projections=return_2d_projections,
        return_projective_intersection_transforms=(
            return_projective_intersection_transforms
        ),
    )
    resolved_options = options or FastGSNativeRenderOptions()
    _validate_inputs(scene, camera)
    active_sh_bases = (
        int(scene.feature.shape[1])
        if resolved_options.active_sh_bases is None
        else resolved_options.active_sh_bases
    )
    sh_coefficients_0, sh_coefficients_rest = _split_sh_coefficients(scene)
    intrinsics = camera.get_intrinsics()
    background_color = resolved_options.background_color.to(
        device=scene.center_position.device,
        dtype=scene.center_position.dtype,
    )
    renders: list[Tensor] = []
    densification_info = (
        torch.zeros(
            (4, scene.center_position.shape[0]),
            dtype=torch.float32,
            device=scene.center_position.device,
        )
        if resolved_options.collect_densification_info
        else torch.empty(0, device=scene.center_position.device)
    )

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
            near_plane=resolved_options.near_plane,
            far_plane=resolved_options.far_plane,
            width=int(camera.width[camera_index].item()),
            height=int(camera.height[camera_index].item()),
            focal_x=float(camera_intrinsics[0, 0].item()),
            focal_y=float(camera_intrinsics[1, 1].item()),
            center_x=float(camera_intrinsics[0, 2].item()),
            center_y=float(camera_intrinsics[1, 2].item()),
            bg_color=background_color,
            mip_splatting_screen_filter=(
                resolved_options.mip_splatting_screen_filter
            ),
            active_sh_bases=active_sh_bases,
            compact_box_scale=(resolved_options.compact_box_scale),
            densification_info=densification_info,
        )
        renders.append(render_result.image.permute(1, 2, 0).contiguous())

    rendered = torch.stack(renders, dim=0)
    if resolved_options.clamp_output:
        rendered = rendered.clamp(0.0, 1.0)
    if resolved_options.collect_densification_info:
        return FastGSNativeDensificationRenderOutput(
            render=rendered,
            densification_info=densification_info,
        )
    return FastGSNativeRenderOutput(render=rendered)


def register() -> None:
    """Register the native FastGS backend."""
    register_backend(
        name="faster_gs.fastgs",
        default_options=FastGSNativeRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=_SUPPORTED_OUTPUTS,
        trait_providers=(FastGSNativeGaussianMetricAttribution(),),
    )(render_fastgs_native)


__all__ = [
    "FastGSNativeDensificationRenderOutput",
    "FastGSNativeGaussianMetricAttribution",
    "FastGSNativeRenderOptions",
    "FastGSNativeRenderOutput",
    "register",
    "render_fastgs_native",
]
