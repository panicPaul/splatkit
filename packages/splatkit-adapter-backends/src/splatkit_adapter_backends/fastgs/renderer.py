"""FastGS backend contract surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int
from splatkit.densification.contracts import GaussianMetricAttribution
from splatkit.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
)
from splatkit.core.registry import register_backend
from torch import Tensor

_SUPPORTED_OUTPUTS = frozenset()


@beartype
@dataclass(frozen=True)
class FastGSRenderOutput(RenderOutput):
    """FastGS render output with backend-specific refinement signals."""

    viewspace_points: Float[Tensor, " num_cams num_splats 4"]
    visibility_filter: Bool[Tensor, " num_cams num_splats"]
    radii: Int[Tensor, " num_cams num_splats"]


@beartype
@dataclass(frozen=True)
class FastGSRenderOptions(RenderOptions):
    """FastGS-specific render configuration."""

    mult: float = 0.5
    scale_modifier: float = 1.0
    debug: bool = False


@beartype
def _normalized_quaternions(
    scene: GaussianScene3D,
) -> Float[Tensor, " num_splats 4"]:
    """Return unit quaternions for rasterizers that expect normalized input."""
    quaternion_norms = torch.linalg.vector_norm(
        scene.quaternion_orientation,
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(scene.quaternion_orientation.dtype).eps)
    return scene.quaternion_orientation / quaternion_norms


def _import_fastgs_runtime() -> tuple[type[Any], type[Any]]:
    try:
        from diff_gaussian_rasterization_fastgs import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
    except ImportError as exc:
        raise ImportError(
            "The FastGS backend requires the diff_gaussian_rasterization_fastgs "
            'dependency. Install it with `pip install "splatkit-adapter-backends[fastgs]"`.'
        ) from exc
    return GaussianRasterizationSettings, GaussianRasterizer


@beartype
def _camera_to_world_to_view(cam_to_world: Tensor) -> Tensor:
    return torch.linalg.inv(cam_to_world).transpose(0, 1).contiguous()


@beartype
def _projection_matrix(
    tanfovx: float,
    tanfovy: float,
    znear: float,
    zfar: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    matrix = torch.zeros((4, 4), device=device, dtype=dtype)
    matrix[0, 0] = 1.0 / tanfovx
    matrix[1, 1] = 1.0 / tanfovy
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    matrix[3, 2] = 1.0
    return matrix.transpose(0, 1).contiguous()


@beartype
def _split_sh_coefficients(
    scene: GaussianScene3D,
) -> tuple[
    Float[Tensor, " num_splats 1 3"],
    Float[Tensor, " num_splats sh_coeffs_minus_one 3"],
]:
    if scene.feature.ndim != 3:
        raise ValueError(
            "FastGS expects spherical harmonics with shape "
            f"(num_splats, sh_coeffs, 3); got {tuple(scene.feature.shape)}."
        )
    if scene.feature.shape[1] < 1:
        raise ValueError("FastGS requires at least one SH basis.")
    return scene.feature[:, :1, :], scene.feature[:, 1:, :]


@beartype
def _validate_inputs(scene: GaussianScene3D, camera: CameraState) -> None:
    if scene.center_position.device.type != "cuda":
        raise ValueError("FastGS requires scene tensors on CUDA.")
    if camera.cam_to_world.device.type != "cuda":
        raise ValueError("FastGS requires camera tensors on CUDA.")
    if camera.camera_convention != "opencv":
        raise ValueError(
            "FastGS currently expects cameras in opencv convention; got "
            f"{camera.camera_convention!r}."
        )
    if scene.log_scales.shape[-1] != 3:
        raise ValueError(
            "FastGS only supports 3D Gaussian scales with shape "
            f"(num_splats, 3); got {tuple(scene.log_scales.shape)}."
        )
    if camera.width.shape[0] != camera.cam_to_world.shape[0]:
        raise ValueError("Camera batch fields must share the same batch size.")
    if camera.height.shape[0] != camera.cam_to_world.shape[0]:
        raise ValueError("Camera batch fields must share the same batch size.")
    if camera.fov_degrees.shape[0] != camera.cam_to_world.shape[0]:
        raise ValueError("Camera batch fields must share the same batch size.")


@beartype
def _build_rasterizer_settings(
    settings_type: type[Any],
    scene: GaussianScene3D,
    camera: CameraState,
    camera_index: int,
    options: FastGSRenderOptions,
    *,
    get_flag: bool,
    metric_map: Int[Tensor, " flattened_pixels"],
) -> Any:
    intrinsics = camera.get_intrinsics()[camera_index]
    width = int(camera.width[camera_index].item())
    height = int(camera.height[camera_index].item())
    fx = float(intrinsics[0, 0].item())
    fy = float(intrinsics[1, 1].item())
    tanfovx = (width * 0.5) / fx
    tanfovy = (height * 0.5) / fy
    cam_to_world = camera.cam_to_world[camera_index]
    viewmatrix = _camera_to_world_to_view(cam_to_world)
    projmatrix = _projection_matrix(
        tanfovx,
        tanfovy,
        0.01,
        1000.0,
        device=scene.center_position.device,
        dtype=scene.center_position.dtype,
    )
    full_proj_transform = viewmatrix @ projmatrix
    return settings_type(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=options.background_color.to(
            device=scene.center_position.device,
            dtype=scene.center_position.dtype,
        ),
        scale_modifier=options.scale_modifier,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_transform,
        sh_degree=scene.sh_degree,
        campos=cam_to_world[:3, 3].contiguous(),
        mult=options.mult,
        prefiltered=False,
        debug=options.debug,
        get_flag=get_flag,
        metric_map=metric_map,
    )


@beartype
def _render_single_camera(
    rasterizer_type: type[Any],
    settings_type: type[Any],
    scene: GaussianScene3D,
    camera: CameraState,
    camera_index: int,
    options: FastGSRenderOptions,
    sh_coefficients_0: Float[Tensor, " num_splats 1 3"],
    sh_coefficients_rest: Float[Tensor, " num_splats sh_coeffs_minus_one 3"],
    *,
    get_flag: bool = False,
    metric_map: Int[Tensor, " height width"] | None = None,
) -> tuple[
    Float[Tensor, " height width 3"],
    Float[Tensor, " num_splats 4"],
    Bool[Tensor, " num_splats"],
    Int[Tensor, " num_splats"],
    Int[Tensor, " num_splats"],
]:
    height = int(camera.height[camera_index].item())
    width = int(camera.width[camera_index].item())
    flattened_metric_map = (
        torch.zeros(
            height * width,
            dtype=torch.int32,
            device=scene.center_position.device,
        )
        if metric_map is None
        else metric_map.reshape(-1).to(
            device=scene.center_position.device,
            dtype=torch.int32,
        )
    )
    settings = _build_rasterizer_settings(
        settings_type,
        scene,
        camera,
        camera_index,
        options,
        get_flag=get_flag,
        metric_map=flattened_metric_map,
    )
    rasterizer = rasterizer_type(raster_settings=settings)
    num_splats = int(scene.center_position.shape[0])
    viewspace_points = torch.zeros(
        (num_splats, 4),
        dtype=scene.center_position.dtype,
        device=scene.center_position.device,
        requires_grad=True,
    )
    viewspace_points.retain_grad()
    rendered_image, radii, accum_metric_counts = rasterizer(
        means3D=scene.center_position,
        means2D=viewspace_points,
        dc=sh_coefficients_0.contiguous(),
        shs=sh_coefficients_rest.contiguous(),
        colors_precomp=None,
        opacities=torch.sigmoid(scene.logit_opacity)[:, None].contiguous(),
        scales=torch.exp(scene.log_scales).contiguous(),
        rotations=_normalized_quaternions(scene).contiguous(),
        cov3D_precomp=None,
    )
    image = rendered_image.permute(1, 2, 0).contiguous().clamp(0.0, 1.0)
    radii = radii.to(dtype=torch.int32).contiguous()
    visibility_filter = radii > 0
    if accum_metric_counts.numel() == 0:
        metric_counts = torch.zeros(
            (num_splats,),
            dtype=torch.int32,
            device=scene.center_position.device,
        )
    elif accum_metric_counts.numel() == num_splats:
        metric_counts = accum_metric_counts.reshape(num_splats).contiguous()
    else:
        raise RuntimeError(
            "FastGS returned an unexpected metric attribution size: "
            f"got {accum_metric_counts.numel()} values for "
            f"{num_splats} Gaussians."
        )
    return image, viewspace_points, visibility_filter, radii, metric_counts


@beartype
@dataclass(frozen=True)
class FastGSGaussianMetricAttribution(GaussianMetricAttribution):
    """FastGS metric-map attribution trait provider."""

    def attribute_metric_map(
        self,
        scene: GaussianScene3D,
        camera: CameraState,
        metric_map: Int[Tensor, " height width"],
        *,
        options: FastGSRenderOptions | None = None,
    ) -> Float[Tensor, " num_splats"]:
        if camera.cam_to_world.shape[0] != 1:
            raise ValueError(
                "FastGS metric attribution expects a single probe camera."
            )
        if metric_map.ndim != 2:
            raise ValueError(
                "FastGS metric attribution expects a 2D metric map with "
                f"shape (height, width); got {tuple(metric_map.shape)}."
            )
        resolved_options = options or FastGSRenderOptions()
        _validate_inputs(scene, camera)
        sh_coefficients_0, sh_coefficients_rest = _split_sh_coefficients(scene)
        settings_type, rasterizer_type = _import_fastgs_runtime()
        (
            _image,
            _viewspace_point_tensor,
            _visibility_filter,
            _radii,
            metric_counts,
        ) = _render_single_camera(
            rasterizer_type,
            settings_type,
            scene,
            camera,
            0,
            resolved_options,
            sh_coefficients_0,
            sh_coefficients_rest,
            get_flag=True,
            metric_map=metric_map,
        )
        return metric_counts.to(dtype=scene.center_position.dtype)


@beartype
def render_fastgs(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = False,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: FastGSRenderOptions | None = None,
) -> FastGSRenderOutput:
    """Render a scene with FastGS."""
    if return_alpha:
        raise ValueError("The FastGS backend does not expose alpha output.")
    if return_depth:
        raise ValueError("The FastGS backend does not expose depth output.")
    if return_gaussian_impact_score:
        raise ValueError(
            "The FastGS backend does not expose Gaussian impact scores."
        )
    if return_normals:
        raise ValueError("The FastGS backend does not expose normals.")
    if return_2d_projections:
        raise ValueError(
            "The FastGS backend does not expose 2D Gaussian projections."
        )
    if return_projective_intersection_transforms:
        raise ValueError(
            "The FastGS backend does not expose projective intersection "
            "transforms."
        )
    if options is not None and options.debug:
        raise ValueError(
            "The FastGS backend debug path is not supported in splatkit."
        )

    _validate_inputs(scene, camera)
    resolved_options = options or FastGSRenderOptions()
    sh_coefficients_0, sh_coefficients_rest = _split_sh_coefficients(scene)
    settings_type, rasterizer_type = _import_fastgs_runtime()

    renders: list[Float[Tensor, " height width 3"]] = []
    viewspace_points: list[Float[Tensor, " num_splats 4"]] = []
    visibility_filters: list[Bool[Tensor, " num_splats"]] = []
    radii_per_camera: list[Int[Tensor, " num_splats"]] = []

    for camera_index in range(camera.cam_to_world.shape[0]):
        (
            image,
            viewspace_point_tensor,
            visibility_filter,
            radii,
            _metric_counts,
        ) = _render_single_camera(
            rasterizer_type,
            settings_type,
            scene,
            camera,
            camera_index,
            resolved_options,
            sh_coefficients_0,
            sh_coefficients_rest,
        )
        renders.append(image)
        viewspace_points.append(viewspace_point_tensor)
        visibility_filters.append(visibility_filter)
        radii_per_camera.append(radii)

    return FastGSRenderOutput(
        render=torch.stack(renders, dim=0),
        viewspace_points=torch.stack(viewspace_points, dim=0),
        visibility_filter=torch.stack(visibility_filters, dim=0),
        radii=torch.stack(radii_per_camera, dim=0),
    )


def register() -> None:
    """Register the FastGS backend in the global splatkit registry."""
    register_backend(
        name="adapter.fastgs",
        default_options=FastGSRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=_SUPPORTED_OUTPUTS,
        trait_providers=(FastGSGaussianMetricAttribution(),),
    )(render_fastgs)
