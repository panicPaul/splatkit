"""Native NHT feature renderer for the 3DGUT-style backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from beartype import beartype
from ember_core.core.capabilities import HasAlpha, HasDepth
from ember_core.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
)
from jaxtyping import Float
from torch import Tensor

from ember_native_nht.threedgut.runtime import rasterization_nht


@beartype
@dataclass(frozen=True)
class NHT3DGUTRenderOutput(RenderOutput, HasAlpha, HasDepth):
    """NHT feature render output consumed by the deferred shader."""

    features: Float[Tensor, " num_cams height width feature_dim_plus_ray"]
    alphas: Float[Tensor, " num_cams height width"]
    depth: Float[Tensor, " num_cams height width"]
    visibility: Float[Tensor, " num_splats 1"]
    weights: Float[Tensor, " num_splats 1"]


@beartype
@dataclass(frozen=True)
class NHT3DGUTRenderOptions(RenderOptions):
    """Native NHT render options matching upstream defaults."""

    tile_size: int = 16
    eps2d: float = 0.3
    near_plane: float = 0.01
    far_plane: float = 1.0e10
    radius_clip: float = 0.0
    mip_splatting_screen_filter: bool = False
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole"
    ray_dir_scale: float = 3.0
    center_ray_mode: bool = False


def tetrahedron_vertices(
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Float[Tensor, " 4 3"]:
    """Return upstream NHT canonical tetrahedron vertices."""
    sqrt2 = 2.0**0.5
    sqrt6 = 6.0**0.5
    return torch.tensor(
        [
            [sqrt6, -sqrt2, -1.0],
            [-sqrt6, -sqrt2, -1.0],
            [0.0, 2.0 * sqrt2, -1.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=dtype,
        device=device,
    )


def barycentric_weights(
    canonical_positions: Float[Tensor, " ... 3"],
) -> Float[Tensor, " ... 4"]:
    """Interpolate canonical positions over the upstream tetrahedron."""
    vertices = tetrahedron_vertices(
        dtype=canonical_positions.dtype,
        device=canonical_positions.device,
    )
    transform = torch.cat(
        (
            vertices.transpose(0, 1),
            torch.ones((1, 4), dtype=vertices.dtype, device=vertices.device),
        ),
        dim=0,
    )
    rhs = torch.cat(
        (
            canonical_positions,
            torch.ones_like(canonical_positions[..., :1]),
        ),
        dim=-1,
    )
    return torch.linalg.solve(transform, rhs.unsqueeze(-1)).squeeze(-1)


def harmonic_encode(
    features: Float[Tensor, " ... channels"],
) -> Float[Tensor, " ... encoded_channels"]:
    """Encode interpolated NHT features with the upstream one-frequency basis."""
    return torch.cat((torch.sin(features), torch.cos(features)), dim=-1)


def _validate_inputs(scene: GaussianScene3D, camera: CameraState) -> None:
    if scene.center_position.device.type != "cuda":
        raise ValueError("nht.3dgut requires scene tensors on CUDA.")
    if camera.cam_to_world.device.type != "cuda":
        raise ValueError("nht.3dgut requires camera tensors on CUDA.")
    if camera.camera_convention != "opencv":
        raise ValueError(
            "nht.3dgut expects cameras in opencv convention; got "
            f"{camera.camera_convention!r}."
        )
    if scene.log_scales.shape[-1] != 3:
        raise ValueError("nht.3dgut requires 3D Gaussian scales.")
    if scene.feature.ndim != 2:
        raise ValueError(
            "nht.3dgut expects flat vertex features with shape "
            f"(num_splats, feature_dim), got {tuple(scene.feature.shape)}."
        )
    if scene.feature.shape[1] % 4 != 0:
        raise ValueError(
            "nht.3dgut requires feature_dim divisible by four so features can "
            "be split across tetrahedron vertices."
        )
    if not torch.equal(camera.width, camera.width[:1].expand_as(camera.width)):
        raise ValueError("nht.3dgut requires uniform width across cameras.")
    if not torch.equal(
        camera.height, camera.height[:1].expand_as(camera.height)
    ):
        raise ValueError("nht.3dgut requires uniform height across cameras.")


def _render_native_features(
    scene: GaussianScene3D,
    camera: CameraState,
    options: NHT3DGUTRenderOptions,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    render_colors, render_alphas, metadata = rasterization_nht(
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
        mip_splatting_screen_filter=options.mip_splatting_screen_filter,
        render_mode="RGB+ED",
        near_plane=options.near_plane,
        far_plane=options.far_plane,
        radius_clip=options.radius_clip,
        eps2d=options.eps2d,
        camera_model=options.camera_model,
        center_ray_mode=options.center_ray_mode,
        ray_dir_scale=options.ray_dir_scale,
    )
    features = render_colors[..., :-1]
    depth = render_colors[..., -1]
    num_splats = int(scene.center_position.shape[0])
    device = scene.center_position.device
    dtype = scene.center_position.dtype
    tiles_per_gauss = metadata.get("tiles_per_gauss")
    visibility = torch.zeros((num_splats, 1), dtype=dtype, device=device)
    weights = torch.zeros_like(visibility)
    if isinstance(tiles_per_gauss, Tensor):
        per_splat = tiles_per_gauss.reshape(-1, num_splats).sum(dim=0)
        visibility[:, 0] = (per_splat > 0).to(dtype)
        weights[:, 0] = per_splat.to(dtype)
    return features, render_alphas.squeeze(-1), depth, visibility, weights


@beartype
def render_nht_3dgut(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = True,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: NHT3DGUTRenderOptions | None = None,
) -> NHT3DGUTRenderOutput:
    """Render upstream-style NHT features for deferred shading."""
    del return_alpha, return_depth
    if return_gaussian_impact_score:
        raise ValueError("nht.3dgut does not expose Gaussian impact scores.")
    if return_normals:
        raise ValueError("nht.3dgut does not expose normals.")
    if return_2d_projections:
        raise ValueError("nht.3dgut does not expose 2D projections.")
    if return_projective_intersection_transforms:
        raise ValueError("nht.3dgut does not expose projective transforms.")
    _validate_inputs(scene, camera)
    resolved_options = options or NHT3DGUTRenderOptions()
    features, alphas, depth, visibility, weights = _render_native_features(
        scene,
        camera,
        resolved_options,
    )
    render = features[..., :3] if features.shape[-1] >= 3 else features
    return NHT3DGUTRenderOutput(
        render=render,
        features=features,
        alphas=alphas,
        depth=depth,
        visibility=visibility,
        weights=weights,
    )


__all__ = [
    "NHT3DGUTRenderOptions",
    "NHT3DGUTRenderOutput",
    "barycentric_weights",
    "harmonic_encode",
    "render_nht_3dgut",
    "tetrahedron_vertices",
]
