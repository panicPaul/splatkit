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
from ember_core.core.registry import register_backend
from ember_core.densification.contracts import GaussianMetricAttribution
from jaxtyping import Float, Int
from torch import Tensor

from ember_native_faster_gs.faster_gs.renderer import (
    FasterGSNativeDensificationRenderOutput,
    FasterGSNativeRenderOptions,
    FasterGSNativeRenderOutput,
    _split_sh_coefficients,
    _validate_inputs,
    render_faster_gs_native,
)
from ember_native_faster_gs.faster_gs.runtime import (
    blend_metric_counts,
    preprocess,
    sort,
)

_SUPPORTED_OUTPUTS = frozenset()


@beartype
@dataclass(frozen=True)
class FastGSNativeRenderOutput(RenderOutput):
    """Native FastGS render output."""


@beartype
@dataclass(frozen=True)
class FastGSNativeDensificationRenderOutput(FastGSNativeRenderOutput):
    """Native FastGS output with densification accumulators."""

    densification_info: Float[Tensor, " 2 num_splats"]


@beartype
@dataclass(frozen=True)
class FastGSNativeRenderOptions(FasterGSNativeRenderOptions):
    """Native FastGS render configuration."""


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
            proper_antialiasing=resolved_options.proper_antialiasing,
            active_sh_bases=int(scene.feature.shape[1]),
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
            resolved_options.proper_antialiasing,
            width=width,
            height=height,
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
    native_output = render_faster_gs_native(
        scene,
        camera,
        options=resolved_options,
    )
    if isinstance(native_output, FasterGSNativeDensificationRenderOutput):
        return FastGSNativeDensificationRenderOutput(
            render=native_output.render,
            densification_info=native_output.densification_info,
        )
    if not isinstance(native_output, FasterGSNativeRenderOutput):
        raise TypeError(
            "faster_gs.core returned an unexpected render output type: "
            f"{type(native_output).__name__}."
        )
    return FastGSNativeRenderOutput(render=native_output.render)


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
