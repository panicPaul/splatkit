"""FasterGS backend contract surface."""

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
from FasterGSCudaBackend import RasterizerSettings, diff_rasterize
from jaxtyping import Float
from torch import Tensor

_SUPPORTED_OUTPUTS = frozenset()


@beartype
@dataclass(frozen=True)
class FasterGSRenderOutput(RenderOutput):
    """FasterGS render output."""


@beartype
@dataclass(frozen=True)
class FasterGSRenderOptions(RenderOptions):
    """FasterGS-specific render configuration."""

    near_plane: float = 0.01
    far_plane: float = 1000.0
    proper_antialiasing: bool = False


@beartype
def _split_sh_coefficients(
    scene: GaussianScene3D,
) -> tuple[
    Float[Tensor, "num_splats 1 3"],
    Float[Tensor, "num_splats sh_coeffs_minus_one 3"],
]:
    if scene.feature.ndim != 3:
        raise ValueError(
            "FasterGS expects spherical harmonics with shape "
            f"(num_splats, sh_coeffs, 3); got {tuple(scene.feature.shape)}."
        )
    if scene.feature.shape[1] < 1:
        raise ValueError("FasterGS requires at least one SH basis.")
    return scene.feature[:, :1, :], scene.feature[:, 1:, :]


@beartype
def _build_rasterizer_settings(
    scene: GaussianScene3D,
    camera: CameraState,
    camera_index: int,
    options: FasterGSRenderOptions,
) -> RasterizerSettings:
    intrinsics = camera.get_intrinsics()[camera_index]
    cam_to_world = camera.cam_to_world[camera_index]
    width = int(camera.width[camera_index].item())
    height = int(camera.height[camera_index].item())
    return RasterizerSettings(
        w2c=torch.linalg.inv(cam_to_world),
        cam_position=cam_to_world[:3, 3].contiguous(),
        bg_color=options.background_color.to(
            device=scene.center_position.device,
            dtype=scene.center_position.dtype,
        ),
        active_sh_bases=int(scene.feature.shape[1]),
        width=width,
        height=height,
        focal_x=float(intrinsics[0, 0].item()),
        focal_y=float(intrinsics[1, 1].item()),
        center_x=float(intrinsics[0, 2].item()),
        center_y=float(intrinsics[1, 2].item()),
        near_plane=options.near_plane,
        far_plane=options.far_plane,
        proper_antialiasing=options.proper_antialiasing,
    )


@beartype
def render_fastergs(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = False,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: FasterGSRenderOptions | None = None,
) -> FasterGSRenderOutput:
    """Render a scene with FasterGS."""
    if return_alpha:
        raise ValueError("The FasterGS backend does not expose alpha output.")
    if return_depth:
        raise ValueError("The FasterGS backend does not expose depth output.")
    if return_gaussian_impact_score:
        raise ValueError(
            "The FasterGS backend does not expose Gaussian impact scores."
        )
    if return_normals:
        raise ValueError("The FasterGS backend does not expose normals.")
    if return_2d_projections:
        raise ValueError(
            "The FasterGS backend does not expose 2D Gaussian projections."
        )
    if return_projective_intersection_transforms:
        raise ValueError(
            "The FasterGS backend does not expose projective intersection "
            "transforms."
        )
    if scene.log_scales.shape[-1] != 3:
        raise ValueError(
            "FasterGS only supports 3D Gaussian scales with shape "
            f"(num_splats, 3); got {tuple(scene.log_scales.shape)}."
        )
    if camera.camera_convention != "opencv":
        raise ValueError(
            "FasterGS currently expects cameras in opencv convention; got "
            f"{camera.camera_convention!r}."
        )

    options = options or FasterGSRenderOptions()
    sh_coefficients_0, sh_coefficients_rest = _split_sh_coefficients(scene)
    renders: list[Float[Tensor, "height width 3"]] = []

    for camera_index in range(camera.cam_to_world.shape[0]):
        image = diff_rasterize(
            means=scene.center_position,
            scales=scene.log_scales,
            rotations=scene.quaternion_orientation,
            opacities=scene.logit_opacity[:, None],
            sh_coefficients_0=sh_coefficients_0.contiguous(),
            sh_coefficients_rest=sh_coefficients_rest.contiguous(),
            densification_info=torch.empty(
                0, device=scene.center_position.device
            ),
            rasterizer_settings=_build_rasterizer_settings(
                scene,
                camera,
                camera_index,
                options,
            ),
        )
        renders.append(image.permute(1, 2, 0).contiguous().clamp(0.0, 1.0))

    return FasterGSRenderOutput(render=torch.stack(renders, dim=0))


def register() -> None:
    """Register the FasterGS backend in the global ember-core registry."""
    register_backend(
        name="adapter.fastergs",
        default_options=FasterGSRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=_SUPPORTED_OUTPUTS,
    )(render_fastergs)
