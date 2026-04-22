"""Thin splatkit adapter for the stochastic 3DGRT renderer."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, cast, overload

import torch
import torch.nn.functional as torch_f
from beartype import beartype
from jaxtyping import Float
from splatkit.core.capabilities import HasAlpha, HasDepth
from splatkit.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
)
from splatkit.core.registry import register_backend
from torch import Tensor

_SUPPORTED_OUTPUTS = frozenset({"alpha", "depth"})
_TRACER_CACHE: dict[tuple[Any, ...], Any] = {}


@beartype
@dataclass(frozen=True)
class Stoch3DGSAlphaRenderOutput(RenderOutput, HasAlpha):
    """Stoch3DGS render output with alpha."""

    alphas: Float[Tensor, "num_cams height width"]


@beartype
@dataclass(frozen=True)
class Stoch3DGSRenderOutput(Stoch3DGSAlphaRenderOutput, HasDepth):
    """Stoch3DGS render output with alpha and depth."""

    depth: Float[Tensor, "num_cams height width"]


@beartype
@dataclass(frozen=True)
class Stoch3DGSRenderOptions(RenderOptions):
    """Stochastic 3DGRT-specific render configuration."""

    particle_kernel_degree: int = 4
    particle_kernel_density_clamping: bool = True
    particle_kernel_min_response: float = 0.0113
    particle_kernel_min_alpha: float = 1.0 / 255.0
    particle_kernel_max_alpha: float = 0.99
    primitive_type: Literal["instances"] = "instances"
    min_transmittance: float = 0.001
    enable_kernel_timings: bool = False


class _Background:
    def __init__(self, color: Tensor) -> None:
        self._color = color

    def __call__(
        self,
        _ray_to_world: Tensor,
        _rays_d: Tensor,
        rgb: Tensor,
        opacity: Tensor,
        _train: bool,
    ) -> tuple[Tensor, Tensor]:
        color = self._color.to(device=rgb.device, dtype=rgb.dtype)
        rgb = rgb + color.view(1, 1, 1, 3) * (1.0 - opacity)
        return rgb, opacity


class _SceneAdapter:
    def __init__(
        self,
        scene: GaussianScene3D,
        background_color: Tensor,
    ) -> None:
        self.positions = scene.center_position
        self.rotation = scene.quaternion_orientation
        self.scale = scene.log_scales
        self.density = scene.logit_opacity[:, None]
        self._features = _flatten_sh_features(scene)
        self.n_active_features = scene.sh_degree
        self.background = _Background(background_color)

    @property
    def num_gaussians(self) -> int:
        return int(self.positions.shape[0])

    @staticmethod
    def rotation_activation(rotation: Tensor) -> Tensor:
        return torch_f.normalize(rotation, dim=1)

    @staticmethod
    def scale_activation(scale: Tensor) -> Tensor:
        return torch.exp(scale)

    @staticmethod
    def density_activation(density: Tensor) -> Tensor:
        return torch.sigmoid(density)

    def get_rotation(self) -> Tensor:
        return self.rotation_activation(self.rotation)

    def get_scale(self) -> Tensor:
        return self.scale_activation(self.scale)

    def get_density(self) -> Tensor:
        return self.density_activation(self.density)

    def get_features(self) -> Tensor:
        return self._features


def _import_stoch_runtime() -> tuple[type[Any], type[Any]]:
    try:
        from threedgrt_tracer import Tracer
    except ImportError as exc:
        raise ImportError(
            "The Stoch3DGS backend requires the threedgrut dependency stack. "
            'Install it with `pip install "splatkit-adapter-backends[stoch3dgs]"`.'
        ) from exc
    return Tracer, SimpleNamespace


def _flatten_sh_features(scene: GaussianScene3D) -> Tensor:
    feature = scene.feature
    if feature.ndim != 3:
        raise ValueError(
            "Stoch3DGS expects spherical harmonics with shape "
            f"(num_splats, sh_coeffs, 3); got {tuple(feature.shape)}."
        )
    dc = feature[:, 0, :]
    if feature.shape[1] == 1:
        return dc
    # Upstream Stoch3DGS expects higher-order SH packed coefficient-major with
    # interleaved RGB triplets: [r1, g1, b1, r2, g2, b2, ...].
    rest = feature[:, 1:, :].reshape(feature.shape[0], -1)
    return torch.cat((dc, rest), dim=1).contiguous()


def _validate_inputs(scene: GaussianScene3D, camera: CameraState) -> None:
    if scene.center_position.device.type != "cuda":
        raise ValueError("Stoch3DGS requires scene tensors on CUDA.")
    if camera.cam_to_world.device.type != "cuda":
        raise ValueError("Stoch3DGS requires camera tensors on CUDA.")
    if camera.camera_convention != "opencv":
        raise ValueError(
            "Stoch3DGS currently expects cameras in opencv convention; got "
            f"{camera.camera_convention!r}."
        )
    if scene.log_scales.shape[-1] != 3:
        raise ValueError(
            "Stoch3DGS only supports 3D Gaussian scales with shape "
            f"(num_splats, 3); got {tuple(scene.log_scales.shape)}."
        )
    if scene.feature.ndim != 3:
        raise ValueError(
            "Stoch3DGS expects spherical harmonics with shape "
            f"(num_splats, sh_coeffs, 3); got {tuple(scene.feature.shape)}."
        )
    if not torch.equal(camera.width, camera.width[:1].expand_as(camera.width)):
        raise ValueError(
            "Stoch3DGS requires a uniform image width across cameras."
        )
    if not torch.equal(
        camera.height, camera.height[:1].expand_as(camera.height)
    ):
        raise ValueError(
            "Stoch3DGS requires a uniform image height across cameras."
        )


def _build_batch(camera: CameraState, batch_type: type[Any]) -> Any:
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
    rays_dir = torch_f.normalize(dirs, dim=-1)
    rays_ori = torch.zeros_like(rays_dir)
    return batch_type(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        T_to_world=camera.cam_to_world.contiguous(),
    )


def _tracer_cache_key(
    scene: GaussianScene3D,
    options: Stoch3DGSRenderOptions,
) -> tuple[Any, ...]:
    return (
        scene.center_position.device.index,
        scene.sh_degree,
        options.particle_kernel_degree,
        options.particle_kernel_density_clamping,
        options.particle_kernel_min_response,
        options.particle_kernel_min_alpha,
        options.particle_kernel_max_alpha,
        options.primitive_type,
        options.min_transmittance,
        options.enable_kernel_timings,
    )


def _build_config(
    scene: GaussianScene3D,
    options: Stoch3DGSRenderOptions,
) -> SimpleNamespace:
    render = SimpleNamespace(
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
        enable_normals=False,
        enable_hitcounts=True,
        enable_kernel_timings=options.enable_kernel_timings,
        max_consecutive_bvh_update=1,
    )
    return SimpleNamespace(render=render)


def _get_tracer(
    scene: GaussianScene3D,
    options: Stoch3DGSRenderOptions,
) -> Any:
    tracer_key = _tracer_cache_key(scene, options)
    tracer = _TRACER_CACHE.get(tracer_key)
    if tracer is None:
        tracer_type, _batch_type = _import_stoch_runtime()
        tracer = tracer_type(_build_config(scene, options))
        _TRACER_CACHE[tracer_key] = tracer
    return tracer


@overload
def render_stoch3dgs(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: Literal[False] = False,
    return_depth: Literal[False] = False,
    return_normals: Literal[False] = False,
    return_2d_projections: Literal[False] = False,
    return_projective_intersection_transforms: Literal[False] = False,
    options: Stoch3DGSRenderOptions | None = None,
) -> Stoch3DGSAlphaRenderOutput: ...


@overload
def render_stoch3dgs(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: Literal[True] = True,
    return_depth: Literal[False] = False,
    return_normals: Literal[False] = False,
    return_2d_projections: Literal[False] = False,
    return_projective_intersection_transforms: Literal[False] = False,
    options: Stoch3DGSRenderOptions | None = None,
) -> Stoch3DGSAlphaRenderOutput: ...


@overload
def render_stoch3dgs(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = True,
    return_depth: Literal[True] = True,
    return_normals: Literal[False] = False,
    return_2d_projections: Literal[False] = False,
    return_projective_intersection_transforms: Literal[False] = False,
    options: Stoch3DGSRenderOptions | None = None,
) -> Stoch3DGSRenderOutput: ...


@beartype
def render_stoch3dgs(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = True,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: Stoch3DGSRenderOptions | None = None,
) -> Stoch3DGSAlphaRenderOutput | Stoch3DGSRenderOutput:
    """Render a scene with the stochastic 3DGRT backend."""
    del return_alpha
    if return_gaussian_impact_score:
        raise ValueError("Stoch3DGS does not expose Gaussian impact scores.")
    if return_normals:
        raise ValueError("Stoch3DGS does not expose normals.")
    if return_2d_projections:
        raise ValueError("Stoch3DGS does not expose 2D Gaussian projections.")
    if return_projective_intersection_transforms:
        raise ValueError(
            "Stoch3DGS does not expose projective intersection transforms."
        )

    _validate_inputs(scene, camera)
    resolved_options = options or Stoch3DGSRenderOptions()
    tracer = _get_tracer(scene, resolved_options)
    _tracer_type, batch_type = _import_stoch_runtime()
    batch = _build_batch(camera, batch_type)
    adapted_scene = _SceneAdapter(scene, resolved_options.background_color)

    tracer.build_acc(adapted_scene, rebuild=True)
    outputs = tracer.render(adapted_scene, batch, train=False)
    render = cast(Tensor, outputs["pred_rgb"]).contiguous()
    alphas = cast(Tensor, outputs["pred_opacity"]).squeeze(-1).contiguous()
    if not return_depth:
        return Stoch3DGSAlphaRenderOutput(render=render, alphas=alphas)
    depth = cast(Tensor, outputs["pred_dist"]).squeeze(-1).contiguous()
    return Stoch3DGSRenderOutput(render=render, alphas=alphas, depth=depth)


def register() -> None:
    """Register the Stoch3DGS backend in the global splatkit registry."""
    register_backend(
        name="stoch3dgs",
        default_options=Stoch3DGSRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=_SUPPORTED_OUTPUTS,
    )(render_stoch3dgs)
