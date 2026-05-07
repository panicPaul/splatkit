"""NHT reference adapter backed by the pinned gsplat fork."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from beartype import beartype
from ember_core.core.capabilities import HasAlpha, HasDepth
from ember_core.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
)
from ember_core.core.registry import output_set, register_backend
from jaxtyping import Float
from torch import Tensor


@beartype
@dataclass(frozen=True)
class NHTAdapterRenderOutput(RenderOutput, HasAlpha, HasDepth):
    """NHT adapter render output."""

    features: Float[Tensor, " num_cams height width feature_dim_plus_ray"]
    alphas: Float[Tensor, " num_cams height width"]
    depth: Float[Tensor, " num_cams height width"]
    metadata: dict[str, Any]


@beartype
@dataclass(frozen=True)
class NHTAdapterRenderOptions(RenderOptions):
    """NHT reference rasterization options."""

    tile_size: int = 16
    eps2d: float = 0.3
    near_plane: float = 0.01
    far_plane: float = 1.0e10
    radius_clip: float = 0.0
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole"
    center_ray_mode: bool = False
    ray_dir_scale: float = 3.0
    packed: bool = False
    sparse_grad: bool = False


def _import_nht_rasterization() -> Any:
    try:
        from gsplat.rendering import rasterization
    except Exception as exc:  # pragma: no cover - dependency dependent
        raise RuntimeError(
            "adapter.nht requires the NHT-capable gsplat fork from "
            "third_party/neural-harmonic-textures/gsplat."
        ) from exc
    if "nht" not in rasterization.__code__.co_varnames:
        raise RuntimeError(
            "adapter.nht found a gsplat package without NHT rasterization "
            "support. Put third_party/neural-harmonic-textures/gsplat on "
            "PYTHONPATH before site-packages, or install that fork in the "
            "active environment."
        )
    return rasterization


def _validate_inputs(scene: GaussianScene3D, camera: CameraState) -> None:
    if scene.center_position.device.type != "cuda":
        raise ValueError("adapter.nht requires scene tensors on CUDA.")
    if camera.cam_to_world.device.type != "cuda":
        raise ValueError("adapter.nht requires camera tensors on CUDA.")
    if camera.camera_convention != "opencv":
        raise ValueError(
            "adapter.nht expects cameras in opencv convention; got "
            f"{camera.camera_convention!r}."
        )
    if scene.log_scales.shape[-1] != 3:
        raise ValueError("adapter.nht requires 3D Gaussian scales.")
    if scene.feature.ndim != 2:
        raise ValueError(
            "adapter.nht expects flat NHT features with shape "
            f"(num_splats, feature_dim), got {tuple(scene.feature.shape)}."
        )
    if scene.feature.shape[1] % 4 != 0:
        raise ValueError("adapter.nht requires feature_dim divisible by four.")
    if not torch.equal(camera.width, camera.width[:1].expand_as(camera.width)):
        raise ValueError("adapter.nht requires uniform width across cameras.")
    if not torch.equal(
        camera.height,
        camera.height[:1].expand_as(camera.height),
    ):
        raise ValueError("adapter.nht requires uniform height across cameras.")


@beartype
def render_nht_adapter(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = True,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: NHTAdapterRenderOptions | None = None,
) -> NHTAdapterRenderOutput:
    """Render NHT features with the reference gsplat implementation."""
    del return_alpha
    if return_gaussian_impact_score:
        raise ValueError("adapter.nht does not expose Gaussian impact scores.")
    if return_normals:
        raise ValueError("adapter.nht does not expose normals.")
    if return_2d_projections:
        raise ValueError("adapter.nht does not expose 2D projections.")
    if return_projective_intersection_transforms:
        raise ValueError("adapter.nht does not expose projective transforms.")
    _validate_inputs(scene, camera)
    options = options or NHTAdapterRenderOptions()
    rasterization = _import_nht_rasterization()
    render_mode: Literal["RGB", "RGB+ED"] = "RGB+ED" if return_depth else "RGB"
    render_colors, render_alphas, metadata = rasterization(
        means=scene.center_position.contiguous(),
        quats=torch.nn.functional.normalize(
            scene.quaternion_orientation, dim=-1
        ).contiguous(),
        scales=torch.exp(scene.log_scales).contiguous(),
        opacities=torch.sigmoid(scene.logit_opacity).contiguous(),
        colors=scene.feature.contiguous(),
        viewmats=torch.linalg.inv(camera.cam_to_world).contiguous(),
        Ks=camera.get_intrinsics().contiguous(),
        width=int(camera.width[0].item()),
        height=int(camera.height[0].item()),
        tile_size=options.tile_size,
        packed=options.packed,
        sparse_grad=options.sparse_grad,
        rasterize_mode=options.rasterize_mode,
        render_mode=render_mode,
        sh_degree=None,
        near_plane=options.near_plane,
        far_plane=options.far_plane,
        radius_clip=options.radius_clip,
        eps2d=options.eps2d,
        camera_model=options.camera_model,
        with_ut=True,
        with_eval3d=True,
        nht=True,
        center_ray_mode=options.center_ray_mode,
        ray_dir_scale=options.ray_dir_scale,
    )
    if return_depth:
        features = render_colors[..., :-1]
        depth = render_colors[..., -1]
    else:
        features = render_colors
        depth = torch.empty(
            render_alphas.shape[:-1],
            dtype=render_alphas.dtype,
            device=render_alphas.device,
        )
    return NHTAdapterRenderOutput(
        render=features[..., :3] if features.shape[-1] >= 3 else features,
        features=features,
        alphas=render_alphas.squeeze(-1),
        depth=depth,
        metadata=metadata,
    )


def register() -> None:
    """Register the NHT reference adapter backend."""
    register_backend(
        name="adapter.nht",
        default_options=NHTAdapterRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=output_set("alpha", "depth"),
    )(render_nht_adapter)


__all__ = [
    "NHTAdapterRenderOptions",
    "NHTAdapterRenderOutput",
    "register",
    "render_nht_adapter",
]
