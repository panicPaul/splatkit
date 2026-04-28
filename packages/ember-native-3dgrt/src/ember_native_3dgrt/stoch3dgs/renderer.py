"""Ember-native Stoch3DGS backend adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as torch_f
from beartype import beartype
from ember_core.core.capabilities import HasAlpha, HasDepth, HasNormals
from ember_core.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
)
from ember_core.core.registry import register_backend
from jaxtyping import Float
from torch import Tensor

from ember_native_3dgrt.core.runtime import (
    TraceStateConfig,
    acquire_state_token,
    build_acc,
    pack_particle_density,
    update_acc,
)
from ember_native_3dgrt.core.runtime import (
    render as render_runtime,
)

_SUPPORTED_OUTPUTS = frozenset({"alpha", "depth", "normals"})
_STATE_TOKEN_CACHE: dict[tuple[object, ...], Tensor] = {}


@beartype
@dataclass(frozen=True)
class Stoch3DGSNativeRenderOutput(RenderOutput, HasAlpha, HasDepth, HasNormals):
    """Native Stoch3DGS render output."""

    alphas: Float[Tensor, "num_cams height width"]
    depth: Float[Tensor, "num_cams height width"]
    normals: Float[Tensor, "num_cams height width 3"]
    hitcounts: Float[Tensor, "num_cams height width"]
    visibility: Float[Tensor, " num_splats 1"]
    weights: Float[Tensor, " num_splats 1"]


@beartype
@dataclass(frozen=True)
class Stoch3DGSNativeRenderOptions(RenderOptions):
    """Native Stoch3DGS render configuration."""

    particle_kernel_degree: int = 4
    particle_kernel_density_clamping: bool = True
    particle_kernel_min_response: float = 0.0113
    particle_kernel_min_alpha: float = 1.0 / 255.0
    particle_kernel_max_alpha: float = 0.99
    primitive_type: str = "instances"
    min_transmittance: float = 0.001
    enable_normals: bool = False
    enable_hitcounts: bool = True
    max_consecutive_bvh_update: int = 1


def _flatten_sh_features(scene: GaussianScene3D) -> Tensor:
    """Flatten SH coefficients into the layout expected by the vendored tracer."""
    feature = scene.feature
    if feature.ndim != 3:
        raise ValueError(
            "3dgrt.stoch3dgs expects spherical harmonics with shape "
            f"(num_splats, sh_coeffs, 3); got {tuple(feature.shape)}."
        )
    dc = feature[:, 0, :]
    if feature.shape[1] == 1:
        return dc
    rest = feature[:, 1:, :].reshape(feature.shape[0], -1)
    return torch.cat((dc, rest), dim=1).contiguous()


def _validate_inputs(scene: GaussianScene3D, camera: CameraState) -> None:
    if scene.center_position.device.type != "cuda":
        raise ValueError("3dgrt.stoch3dgs requires scene tensors on CUDA.")
    if camera.cam_to_world.device.type != "cuda":
        raise ValueError("3dgrt.stoch3dgs requires camera tensors on CUDA.")
    if camera.camera_convention != "opencv":
        raise ValueError(
            "3dgrt.stoch3dgs currently expects cameras in opencv convention; "
            f"got {camera.camera_convention!r}."
        )
    if scene.log_scales.shape[-1] != 3:
        raise ValueError(
            "3dgrt.stoch3dgs only supports 3D Gaussian scales with shape "
            f"(num_splats, 3); got {tuple(scene.log_scales.shape)}."
        )
    if scene.feature.ndim != 3:
        raise ValueError(
            "3dgrt.stoch3dgs expects spherical harmonics with shape "
            f"(num_splats, sh_coeffs, 3); got {tuple(scene.feature.shape)}."
        )
    if not torch.equal(camera.width, camera.width[:1].expand_as(camera.width)):
        raise ValueError(
            "3dgrt.stoch3dgs requires a uniform image width across cameras."
        )
    if not torch.equal(
        camera.height, camera.height[:1].expand_as(camera.height)
    ):
        raise ValueError(
            "3dgrt.stoch3dgs requires a uniform image height across cameras."
        )


def _build_batch(camera: CameraState) -> tuple[Tensor, Tensor]:
    """Construct batched ray origins and directions from the camera intrinsics."""
    intrinsics = camera.get_intrinsics()
    num_cams = int(camera.cam_to_world.shape[0])
    height = int(camera.height[0].item())
    width = int(camera.width[0].item())
    device = camera.cam_to_world.device
    dtype = camera.cam_to_world.dtype

    x = torch.arange(width, device=device, dtype=dtype)
    y = torch.arange(height, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.view(1, height, width).expand(num_cams, -1, -1)
    yy = yy.view(1, height, width).expand(num_cams, -1, -1)

    fx = intrinsics[:, 0, 0].view(num_cams, 1, 1)
    fy = intrinsics[:, 1, 1].view(num_cams, 1, 1)
    cx = intrinsics[:, 0, 2].view(num_cams, 1, 1)
    cy = intrinsics[:, 1, 2].view(num_cams, 1, 1)

    dirs = torch.stack(
        (
            ((xx + 0.5) - cx) / fx,
            ((yy + 0.5) - cy) / fy,
            torch.ones((num_cams, height, width), device=device, dtype=dtype),
        ),
        dim=-1,
    )
    return torch.zeros_like(dirs), torch_f.normalize(dirs, dim=-1)


def _state_config(
    scene: GaussianScene3D,
    options: Stoch3DGSNativeRenderOptions,
) -> TraceStateConfig:
    """Project backend options onto the traced root state config."""
    return TraceStateConfig(
        pipeline_type="fullStochastic",
        backward_pipeline_type="fullStochasticBwd",
        primitive_type=options.primitive_type,
        particle_kernel_degree=options.particle_kernel_degree,
        particle_kernel_density_clamping=options.particle_kernel_density_clamping,
        particle_kernel_min_response=options.particle_kernel_min_response,
        particle_kernel_min_alpha=options.particle_kernel_min_alpha,
        particle_kernel_max_alpha=options.particle_kernel_max_alpha,
        particle_radiance_sph_degree=scene.sh_degree,
        min_transmittance=options.min_transmittance,
        enable_normals=options.enable_normals,
        enable_hitcounts=options.enable_hitcounts,
        max_consecutive_bvh_update=options.max_consecutive_bvh_update,
    )


def _state_cache_key(
    config: TraceStateConfig,
    device: torch.device,
) -> tuple[object, ...]:
    return (
        device.index,
        config.pipeline_type,
        config.backward_pipeline_type,
        config.primitive_type,
        config.particle_kernel_degree,
        config.particle_kernel_density_clamping,
        config.particle_kernel_min_response,
        config.particle_kernel_min_alpha,
        config.particle_kernel_max_alpha,
        config.particle_radiance_sph_degree,
        config.enable_normals,
        config.enable_hitcounts,
        config.max_consecutive_bvh_update,
    )


def _get_state_token(
    config: TraceStateConfig,
    device: torch.device,
) -> tuple[Tensor, bool]:
    """Get or create a cached traced state token for the requested config."""
    key = _state_cache_key(config, device)
    token = _STATE_TOKEN_CACHE.get(key)
    if token is None:
        token = acquire_state_token(config, device)
        _STATE_TOKEN_CACHE[key] = token
        return token, True
    return token, False


@beartype
def render_stoch3dgs_native(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = True,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: Stoch3DGSNativeRenderOptions | None = None,
) -> Stoch3DGSNativeRenderOutput:
    """Render a scene with the native traced Stoch3DGS runtime."""
    del return_alpha, return_depth
    if return_gaussian_impact_score:
        raise ValueError(
            "3dgrt.stoch3dgs does not expose Gaussian impact scores."
        )
    if return_2d_projections:
        raise ValueError(
            "3dgrt.stoch3dgs does not expose 2D Gaussian projections."
        )
    if return_projective_intersection_transforms:
        raise ValueError(
            "3dgrt.stoch3dgs does not expose projective intersection transforms."
        )

    _validate_inputs(scene, camera)
    options = options or Stoch3DGSNativeRenderOptions()
    if return_normals:
        options = Stoch3DGSNativeRenderOptions(
            background_color=options.background_color,
            particle_kernel_degree=options.particle_kernel_degree,
            particle_kernel_density_clamping=options.particle_kernel_density_clamping,
            particle_kernel_min_response=options.particle_kernel_min_response,
            particle_kernel_min_alpha=options.particle_kernel_min_alpha,
            particle_kernel_max_alpha=options.particle_kernel_max_alpha,
            primitive_type=options.primitive_type,
            min_transmittance=options.min_transmittance,
            enable_normals=True,
            enable_hitcounts=options.enable_hitcounts,
            max_consecutive_bvh_update=options.max_consecutive_bvh_update,
        )
    state_config = _state_config(scene, options)
    state_token, created = _get_state_token(
        state_config, scene.center_position.device
    )

    particle_density = pack_particle_density(
        scene.center_position.contiguous(),
        torch.sigmoid(scene.logit_opacity[:, None]).contiguous(),
        torch_f.normalize(scene.quaternion_orientation, dim=1).contiguous(),
        torch.exp(scene.log_scales).contiguous(),
    )
    particle_radiance = _flatten_sh_features(scene)
    ray_ori, ray_dir = _build_batch(camera)
    ray_to_world = camera.cam_to_world.contiguous()
    if created:
        state_token = build_acc(state_token, particle_density)
    else:
        state_token = update_acc(state_token, particle_density)
    background_color = options.background_color.to(
        device=scene.center_position.device,
        dtype=scene.center_position.dtype,
    )
    result = render_runtime(
        state_token,
        ray_to_world,
        ray_ori.contiguous(),
        ray_dir.contiguous(),
        particle_density,
        particle_radiance,
        background_color,
        render_opts=0,
        sph_degree=scene.sh_degree,
        min_transmittance=options.min_transmittance,
    )
    return Stoch3DGSNativeRenderOutput(
        render=result.render,
        alphas=result.alphas,
        depth=result.depth,
        normals=result.normals,
        hitcounts=result.hitcounts,
        visibility=result.visibility,
        weights=result.weights,
    )


def register() -> None:
    """Register the native traced Stoch3DGS backend."""
    register_backend(
        name="3dgrt.stoch3dgs",
        default_options=Stoch3DGSNativeRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=_SUPPORTED_OUTPUTS,
    )(render_stoch3dgs_native)
