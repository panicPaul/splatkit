"""Stoch3DGS render backend with FastGS densification traits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal  # noqa: F401

import torch
import torch.nn.functional as torch_f
from beartype import beartype
from ember_core.core.contracts import CameraState, GaussianScene3D
from ember_core.core.registry import output_set, register_backend
from ember_core.densification.contracts import (
    DensificationContext,
    GaussianFastGSDensificationSignals,
    GaussianFastGSSignalProvider,
    GaussianMetricAttribution,
)
from jaxtyping import Float, Int
from torch import Tensor

from ember_native_3dgrt.core.runtime import (
    build_acc,
    pack_particle_density,
    update_acc,
)
from ember_native_3dgrt.core.runtime import (
    trace_metric_weights as trace_metric_weights_runtime,
)
from ember_native_3dgrt.stoch3dgs.renderer import (
    Stoch3DGSNativeRenderOptions,
    Stoch3DGSNativeRenderOutput,
    _build_batch,
    _flatten_sh_features,
    _get_state_token,
    _state_config,
    _validate_inputs,
    render_stoch3dgs_native,
)

_SUPPORTED_OUTPUTS = output_set("alpha", "depth", "normals")


@beartype
@dataclass(frozen=True)
class StochFastGSNativeRenderOptions(Stoch3DGSNativeRenderOptions):
    """Native Stoch3DGS render config with FastGS training collection flags."""

    collect_densification_info: bool = False


def _camera_position(camera: CameraState) -> Tensor:
    return camera.cam_to_world[0, :3, 3]


def _estimate_screen_radii(
    scene: GaussianScene3D,
    camera: CameraState,
) -> Float[Tensor, " num_splats"]:
    """Estimate conservative screen radii from depth, focal length, and scale."""
    world_to_camera = torch.linalg.inv(camera.cam_to_world[0])
    ones = torch.ones(
        (int(scene.center_position.shape[0]), 1),
        dtype=scene.center_position.dtype,
        device=scene.center_position.device,
    )
    centers_h = torch.cat((scene.center_position.detach(), ones), dim=1)
    camera_space = centers_h @ world_to_camera.to(
        dtype=scene.center_position.dtype
    ).T
    depth = camera_space[:, 2].abs().clamp_min(1e-6)
    intrinsics = camera.get_intrinsics()[0].to(
        device=scene.center_position.device,
        dtype=scene.center_position.dtype,
    )
    focal = torch.maximum(intrinsics[0, 0], intrinsics[1, 1])
    max_scale = torch.exp(scene.log_scales.detach()).max(dim=-1).values
    radii = max_scale * focal / depth
    return torch.nan_to_num(radii, nan=0.0, posinf=0.0, neginf=0.0)


def _visible_weights(output: object, scene: GaussianScene3D) -> Tensor:
    if not hasattr(output, "weights"):
        return torch.zeros(
            (int(scene.center_position.shape[0]),),
            dtype=scene.center_position.dtype,
            device=scene.center_position.device,
        )
    weights = output.weights.detach().reshape(-1)
    return weights.to(
        device=scene.center_position.device,
        dtype=scene.center_position.dtype,
    )


def stoch_fast_gs_metric_weights(
    scene: GaussianScene3D,
    camera: CameraState,
    metric_map: Tensor,
    *,
    options: StochFastGSNativeRenderOptions | None = None,
) -> Float[Tensor, " num_splats"]:
    """Accumulate Stoch3DGS particle weights only on metric-map pixels."""
    _validate_inputs(scene, camera)
    resolved_options = options or StochFastGSNativeRenderOptions()
    state_config = _state_config(scene, resolved_options)
    state_token, created = _get_state_token(
        state_config,
        scene.center_position.device,
    )
    particle_density = pack_particle_density(
        scene.center_position.contiguous(),
        torch.sigmoid(scene.logit_opacity[:, None]).contiguous(),
        torch_f.normalize(scene.quaternion_orientation, dim=1).contiguous(),
        torch.exp(scene.log_scales).contiguous(),
    )
    particle_radiance = _flatten_sh_features(scene)
    ray_ori, ray_dir = _build_batch(
        camera,
        principal_point_mode=resolved_options.ray_principal_point_mode,
    )
    if created:
        state_token = build_acc(state_token, particle_density)
    else:
        state_token = update_acc(state_token, particle_density)
    weights = trace_metric_weights_runtime(
        state_token,
        camera.cam_to_world.contiguous(),
        ray_ori.contiguous(),
        ray_dir.contiguous(),
        particle_density,
        particle_radiance,
        metric_map.to(
            device=scene.center_position.device,
            dtype=torch.int32,
        ).contiguous(),
        render_opts=0,
        sph_degree=scene.sh_degree,
        min_transmittance=resolved_options.min_transmittance,
    )
    return weights.reshape(-1).to(dtype=scene.center_position.dtype)


@beartype
@dataclass(frozen=True)
class StochFastGSSignalProvider(GaussianFastGSSignalProvider):
    """Extract FastGS-style density-control signals from Stoch3DGS outputs."""

    def prepare_fastgs_signals(self, context: DensificationContext) -> None:
        """Stoch3DGS gradients are available through the scene tensors."""
        del context

    def collect_fastgs_signals(
        self,
        context: DensificationContext,
    ) -> GaussianFastGSDensificationSignals | None:
        """Collect visibility, position-gradient, and screen-radius signals."""
        scene = context.state.model.scene
        if not isinstance(scene, GaussianScene3D):
            return None
        position_grad = scene.center_position.grad
        if position_grad is None:
            return None
        camera = context.batch.camera
        weights = _visible_weights(context.render_output, scene)
        visibility = weights > 0
        if hasattr(context.render_output, "visibility"):
            visibility |= context.render_output.visibility.detach().reshape(-1).to(
                device=scene.center_position.device
            ) > 0
        visible_count = visibility.to(dtype=scene.center_position.dtype)
        camera_distance = (
            scene.center_position.detach() - _camera_position(camera)
        ).norm(dim=-1)
        grad_norm = position_grad.detach().norm(dim=-1) * camera_distance
        grad_sum = grad_norm * visible_count
        return GaussianFastGSDensificationSignals(
            visible_count=visible_count,
            clone_grad_sum=grad_sum,
            split_grad_sum=grad_sum,
            max_screen_radii=_estimate_screen_radii(scene, camera)
            * visible_count,
        )


@beartype
@dataclass(frozen=True)
class StochFastGSMetricAttribution(GaussianMetricAttribution):
    """Attribute probe metric maps through Stoch3DGS per-Gaussian weights."""

    def attribute_metric_map(
        self,
        scene: GaussianScene3D,
        camera: CameraState,
        metric_map: Int[Tensor, " *metric_dims"],
        *,
        options: StochFastGSNativeRenderOptions | None = None,
    ) -> Float[Tensor, " num_splats"]:
        """Attribute a single-camera metric map to Gaussians."""
        if camera.cam_to_world.shape[0] != 1:
            raise ValueError(
                "Stoch-Fast-GS metric attribution expects a single probe camera."
            )
        if metric_map.ndim != 2:
            raise ValueError(
                "Stoch-Fast-GS metric attribution expects a 2D metric map with "
                f"shape (height, width); got {tuple(metric_map.shape)}."
            )
        with torch.no_grad():
            return stoch_fast_gs_metric_weights(
                scene,
                camera,
                metric_map,
                options=options,
            )


@beartype
def render_stoch_fast_gs_native(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = True,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: StochFastGSNativeRenderOptions | None = None,
) -> Stoch3DGSNativeRenderOutput:
    """Render through Stoch3DGS while accepting FastGS densification options."""
    return render_stoch3dgs_native(
        scene,
        camera,
        return_alpha=return_alpha,
        return_depth=return_depth,
        return_gaussian_impact_score=return_gaussian_impact_score,
        return_normals=return_normals,
        return_2d_projections=return_2d_projections,
        return_projective_intersection_transforms=(
            return_projective_intersection_transforms
        ),
        options=options or StochFastGSNativeRenderOptions(),
    )


def register() -> None:
    """Register the native Stoch-Fast-GS backend."""
    register_backend(
        name="3dgrt.stoch_fast_gs",
        default_options=StochFastGSNativeRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=_SUPPORTED_OUTPUTS,
        trait_providers=(
            StochFastGSSignalProvider(),
            StochFastGSMetricAttribution(),
        ),
    )(render_stoch_fast_gs_native)


__all__ = [
    "StochFastGSMetricAttribution",
    "StochFastGSNativeRenderOptions",
    "StochFastGSSignalProvider",
    "register",
    "render_stoch_fast_gs_native",
    "stoch_fast_gs_metric_weights",
]
