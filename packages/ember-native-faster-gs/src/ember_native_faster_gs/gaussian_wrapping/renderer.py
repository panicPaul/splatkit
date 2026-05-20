"""Gaussian Wrapping backend adapter for the FasterGS family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from beartype import beartype
from ember_core.core.capabilities import HasAlpha, HasDepth, HasNormals
from ember_core.core.compaction import (
    RayCompactionLossMaps,
    RayCompactionTargets,
)
from ember_core.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
)
from ember_core.core.registry import output_set, register_backend
from ember_core.meshification import (
    MeshificationRequest,
    SurfacePointSamples,
    WrappingQueryResult,
    WrappingSurfaceEvidence,
    WrappingSurfaceProvider,
)
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ember_native_faster_gs.gaussian_wrapping.runtime import (
    ours_integrate_points_fwd_op,
    ours_render_op,
    radegs_integrate_points_fwd_op,
    radegs_render_op,
)

_SUPPORTED_OUTPUTS = output_set("alpha", "depth", "normals")

AlphaTensor = Float[Tensor, " num_cams height width"]
DepthTensor = Float[Tensor, " num_cams height width"]
NormalsTensor = Float[Tensor, "num_cams height width 3"]
RadiiTensor = Int[Tensor, "num_cams num_splats"]
VisibilityTensor = Bool[Tensor, "num_cams num_splats"]


@beartype
@dataclass(frozen=True)
class GaussianWrappingNativeRenderOutput(
    RenderOutput,
    HasAlpha,
    HasDepth,
    HasNormals,
):
    """Gaussian Wrapping render output."""

    alphas: AlphaTensor
    depth: DepthTensor
    normals: NormalsTensor
    median_depth: DepthTensor
    expected_depth: DepthTensor
    radii: RadiiTensor
    visibility_filter: VisibilityTensor


@beartype
@dataclass(frozen=True)
class GaussianWrappingNativeRenderOptions(RenderOptions):
    """Render configuration for Gaussian Wrapping."""

    rasterizer_mode: Literal["ours", "radegs"] = "ours"
    depth_mode: Literal["median", "expected"] = "median"
    near_plane: float = 0.01
    far_plane: float = 1000.0
    kernel_size: float = 0.0
    scale_modifier: float = 1.0
    color_source: Literal["spherical_harmonics", "direct_rgb"] = (
        "spherical_harmonics"
    )
    debug: bool = False


@dataclass(frozen=True)
class _CameraStageParams:
    viewmatrix: Float[Tensor, "4 4"]
    projmatrix: Float[Tensor, "4 4"]
    camera_position: Float[Tensor, " 3"]
    tan_fovx: float
    tan_fovy: float
    image_width: int
    image_height: int


@beartype
def _validate_inputs(scene: GaussianScene3D, camera: CameraState) -> None:
    if scene.center_position.device.type != "cuda":
        raise ValueError("Gaussian Wrapping requires scene tensors on CUDA.")
    if camera.cam_to_world.device.type != "cuda":
        raise ValueError("Gaussian Wrapping requires camera tensors on CUDA.")
    if scene.center_position.device != camera.cam_to_world.device:
        raise ValueError(
            "Gaussian Wrapping requires scene and camera tensors on the same "
            "device."
        )
    if camera.camera_convention != "opencv":
        raise ValueError(
            "faster_gs.gaussian_wrapping expects opencv cameras; got "
            f"{camera.camera_convention!r}."
        )
    if scene.log_scales.shape[-1] != 3:
        raise ValueError(
            "Gaussian Wrapping requires 3D Gaussian scales with shape "
            f"(num_splats, 3); got {tuple(scene.log_scales.shape)}."
        )


def _empty_float(scene: GaussianScene3D) -> Float[Tensor, " empty"]:
    return torch.empty(
        (0,),
        dtype=scene.center_position.dtype,
        device=scene.center_position.device,
    )


def _feature_inputs(
    scene: GaussianScene3D,
    options: GaussianWrappingNativeRenderOptions,
) -> tuple[Tensor, Tensor]:
    if options.color_source == "direct_rgb":
        if scene.feature.ndim != 2 or scene.feature.shape[-1] != 3:
            raise ValueError(
                "direct_rgb color_source expects scene.feature with shape "
                f"(num_splats, 3); got {tuple(scene.feature.shape)}."
            )
        return scene.feature.contiguous(), _empty_float(scene)
    if scene.feature.ndim != 3:
        raise ValueError(
            "spherical_harmonics color_source expects scene.feature with "
            f"shape (num_splats, sh_coeffs, 3); got {tuple(scene.feature.shape)}."
        )
    return _empty_float(scene), scene.feature.contiguous()


def _projection_matrix(
    *,
    tan_fovx: float,
    tan_fovy: float,
    near_plane: float,
    far_plane: float,
    reference: Tensor,
) -> Float[Tensor, "4 4"]:
    projection = torch.zeros(
        (4, 4),
        dtype=reference.dtype,
        device=reference.device,
    )
    projection[0, 0] = 1.0 / tan_fovx
    projection[1, 1] = 1.0 / tan_fovy
    projection[3, 2] = 1.0
    projection[2, 2] = far_plane / (far_plane - near_plane)
    projection[2, 3] = -(far_plane * near_plane) / (far_plane - near_plane)
    return projection.mT.contiguous()


def _camera_stage_params(
    camera: CameraState,
    camera_index: int,
    options: GaussianWrappingNativeRenderOptions,
) -> _CameraStageParams:
    intrinsics = camera.get_intrinsics()[camera_index]
    image_width = int(camera.width[camera_index].item())
    image_height = int(camera.height[camera_index].item())
    focal_x = float(intrinsics[0, 0].item())
    focal_y = float(intrinsics[1, 1].item())
    tan_fovx = image_width / (2.0 * focal_x)
    tan_fovy = image_height / (2.0 * focal_y)
    world_to_camera = torch.linalg.inv(camera.cam_to_world[camera_index])
    viewmatrix = world_to_camera.mT.contiguous()
    projection = _projection_matrix(
        tan_fovx=tan_fovx,
        tan_fovy=tan_fovy,
        near_plane=options.near_plane,
        far_plane=options.far_plane,
        reference=camera.cam_to_world,
    )
    return _CameraStageParams(
        viewmatrix=viewmatrix,
        projmatrix=(viewmatrix[None].bmm(projection[None])).squeeze(0),
        camera_position=camera.cam_to_world[camera_index, :3, 3].contiguous(),
        tan_fovx=tan_fovx,
        tan_fovy=tan_fovy,
        image_width=image_width,
        image_height=image_height,
    )


def _scene_stage_inputs(
    scene: GaussianScene3D,
    options: GaussianWrappingNativeRenderOptions,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    colors, spherical_harmonics = _feature_inputs(scene, options)
    empty = _empty_float(scene)
    return (
        scene.center_position.contiguous(),
        torch.sigmoid(scene.logit_opacity)[:, None].contiguous(),
        torch.exp(scene.log_scales).contiguous(),
        scene.quaternion_orientation.contiguous(),
        colors,
        spherical_harmonics,
        empty,
        empty,
    )


def _render_ours_single_camera(
    scene: GaussianScene3D,
    camera_params: _CameraStageParams,
    options: GaussianWrappingNativeRenderOptions,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    (
        center_positions,
        opacities,
        scales,
        rotations,
        colors,
        spherical_harmonics,
        empty,
        _empty_extra,
    ) = _scene_stage_inputs(scene, options)
    (
        _rendered_count,
        color,
        alpha,
        normal,
        median_depth,
        color_square,
        depth_sum,
        depth_square,
        radii,
        _geom_buffer,
        _binning_buffer,
        _image_buffer,
        _tile_buffer,
    ) = ours_render_op(
        options.background_color.to(
            device=scene.center_position.device,
            dtype=scene.center_position.dtype,
        ).contiguous(),
        center_positions,
        colors,
        opacities,
        scales,
        rotations,
        empty,
        spherical_harmonics,
        empty,
        empty,
        empty,
        scene.sh_degree,
        0,
        options.scale_modifier,
        camera_params.viewmatrix,
        camera_params.projmatrix,
        camera_params.tan_fovx,
        camera_params.tan_fovy,
        options.kernel_size,
        camera_params.image_height,
        camera_params.image_width,
        camera_params.camera_position,
        False,
        True,
        options.debug,
    )
    return (
        color,
        alpha,
        normal,
        median_depth,
        color_square,
        depth_sum,
        depth_square,
        radii,
    )


def _render_radegs_single_camera(
    scene: GaussianScene3D,
    camera_params: _CameraStageParams,
    options: GaussianWrappingNativeRenderOptions,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    (
        center_positions,
        opacities,
        scales,
        rotations,
        colors,
        spherical_harmonics,
        empty,
        _empty_extra,
    ) = _scene_stage_inputs(scene, options)
    (
        _rendered_count,
        color,
        _coord,
        _median_coord,
        alpha,
        normal,
        expected_depth,
        median_depth,
        color_square,
        depth_sum,
        depth_square,
        radii,
        _geom_buffer,
        _binning_buffer,
        _image_buffer,
    ) = radegs_render_op(
        options.background_color.to(
            device=scene.center_position.device,
            dtype=scene.center_position.dtype,
        ).contiguous(),
        center_positions,
        colors,
        opacities,
        scales,
        rotations,
        options.scale_modifier,
        empty,
        camera_params.viewmatrix,
        camera_params.projmatrix,
        camera_params.tan_fovx,
        camera_params.tan_fovy,
        options.kernel_size,
        camera_params.image_height,
        camera_params.image_width,
        spherical_harmonics,
        scene.sh_degree,
        camera_params.camera_position,
        False,
        True,
        True,
        options.debug,
    )
    return (
        color,
        alpha,
        normal,
        expected_depth,
        median_depth,
        color_square,
        depth_sum,
        depth_square,
        radii,
    )


def _chw_to_hwc(image: Tensor) -> Tensor:
    return image.permute(1, 2, 0).contiguous()


def _apply_loss_mask(loss: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return loss
    return loss * mask.to(device=loss.device, dtype=loss.dtype)


def _compaction_color_l2(
    *,
    color: Tensor,
    alpha: Tensor,
    color_square: Tensor,
    target_rgb: Tensor,
    background: Tensor,
    mask: Tensor | None,
) -> Tensor:
    alpha_map = alpha.squeeze(0).contiguous()
    target = target_rgb.to(device=color.device, dtype=color.dtype)
    foreground_color = _chw_to_hwc(color) - (
        (1.0 - alpha_map)[..., None] * background
    )
    loss = (
        color_square.squeeze(0)
        - 2.0 * (target * foreground_color).sum(dim=-1)
        + target.square().sum(dim=-1) * alpha_map
    )
    return _apply_loss_mask(loss.clamp_min(0.0), mask)


def _compaction_depth_l2(
    *,
    alpha: Tensor,
    depth_sum: Tensor,
    depth_square: Tensor,
    target_depth: Tensor,
    mask: Tensor | None,
) -> Tensor:
    alpha_map = alpha.squeeze(0).contiguous()
    target = target_depth.to(device=depth_sum.device, dtype=depth_sum.dtype)
    loss = (
        depth_square.squeeze(0)
        - 2.0 * target * depth_sum.squeeze(0)
        + target.square() * alpha_map
    )
    return _apply_loss_mask(loss.clamp_min(0.0), mask)


@beartype
def render_gaussian_wrapping_native(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = False,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: GaussianWrappingNativeRenderOptions | None = None,
) -> GaussianWrappingNativeRenderOutput:
    """Render a scene with the CUDA-only Gaussian Wrapping backend."""
    del return_alpha, return_depth, return_normals
    if return_gaussian_impact_score:
        raise ValueError(
            "faster_gs.gaussian_wrapping does not expose Gaussian impact "
            "scores."
        )
    if return_2d_projections:
        raise ValueError(
            "faster_gs.gaussian_wrapping does not expose 2D projections."
        )
    if return_projective_intersection_transforms:
        raise ValueError(
            "faster_gs.gaussian_wrapping does not expose projective "
            "intersection transforms."
        )
    resolved_options = options or GaussianWrappingNativeRenderOptions()
    _validate_inputs(scene, camera)

    renders: list[Tensor] = []
    alphas: list[Tensor] = []
    normals: list[Tensor] = []
    expected_depths: list[Tensor] = []
    median_depths: list[Tensor] = []
    radii: list[Tensor] = []
    for camera_index in range(camera.cam_to_world.shape[0]):
        camera_params = _camera_stage_params(
            camera,
            camera_index,
            resolved_options,
        )
        if resolved_options.rasterizer_mode == "ours":
            (
                color,
                alpha,
                normal,
                median_depth,
                _color_square,
                _depth_sum,
                _depth_square,
                camera_radii,
            ) = _render_ours_single_camera(
                scene, camera_params, resolved_options
            )
            expected_depth = median_depth
        else:
            (
                color,
                alpha,
                normal,
                expected_depth,
                median_depth,
                _color_square,
                _depth_sum,
                _depth_square,
                camera_radii,
            ) = _render_radegs_single_camera(
                scene,
                camera_params,
                resolved_options,
            )
        renders.append(_chw_to_hwc(color).clamp(0.0, 1.0))
        alphas.append(alpha.squeeze(0).contiguous())
        normals.append(_chw_to_hwc(normal))
        expected_depths.append(expected_depth.squeeze(0).contiguous())
        median_depths.append(median_depth.squeeze(0).contiguous())
        radii.append(camera_radii.contiguous())

    median_depth = torch.stack(median_depths, dim=0)
    expected_depth = torch.stack(expected_depths, dim=0)
    selected_depth = (
        median_depth
        if resolved_options.depth_mode == "median"
        else expected_depth
    )
    radii_tensor = torch.stack(radii, dim=0)
    return GaussianWrappingNativeRenderOutput(
        render=torch.stack(renders, dim=0),
        alphas=torch.stack(alphas, dim=0),
        depth=selected_depth,
        normals=torch.stack(normals, dim=0),
        median_depth=median_depth,
        expected_depth=expected_depth,
        radii=radii_tensor,
        visibility_filter=radii_tensor > 0,
    )


@beartype
def render_gaussian_wrapping_compaction_losses(
    scene: GaussianScene3D,
    camera: CameraState,
    targets: RayCompactionTargets,
    *,
    options: GaussianWrappingNativeRenderOptions | None = None,
) -> RayCompactionLossMaps:
    """Render unreduced per-pixel color/depth compaction loss maps."""
    resolved_options = options or GaussianWrappingNativeRenderOptions()
    _validate_inputs(scene, camera)

    color_losses: list[Tensor] | None = (
        [] if targets.rgb is not None else None
    )
    depth_losses: list[Tensor] | None = (
        [] if targets.depth is not None else None
    )
    alphas: list[Tensor] = []
    background = resolved_options.background_color.to(
        device=scene.center_position.device,
        dtype=scene.center_position.dtype,
    ).contiguous()

    for camera_index in range(camera.cam_to_world.shape[0]):
        camera_params = _camera_stage_params(
            camera,
            camera_index,
            resolved_options,
        )
        if resolved_options.rasterizer_mode == "ours":
            (
                color,
                alpha,
                _normal,
                _median_depth,
                color_square,
                depth_sum,
                depth_square,
                _camera_radii,
            ) = _render_ours_single_camera(
                scene,
                camera_params,
                resolved_options,
            )
        else:
            (
                color,
                alpha,
                _normal,
                _expected_depth,
                _median_depth,
                color_square,
                depth_sum,
                depth_square,
                _camera_radii,
            ) = _render_radegs_single_camera(
                scene,
                camera_params,
                resolved_options,
            )

        alpha_map = alpha.squeeze(0).contiguous()
        mask = (
            targets.mask[camera_index].to(
                device=alpha_map.device,
                dtype=alpha_map.dtype,
            )
            if targets.mask is not None
            else None
        )
        alphas.append(alpha_map)
        if color_losses is not None and targets.rgb is not None:
            color_losses.append(
                _compaction_color_l2(
                    color=color,
                    alpha=alpha,
                    color_square=color_square,
                    target_rgb=targets.rgb[camera_index],
                    background=background,
                    mask=mask,
                )
            )
        if depth_losses is not None and targets.depth is not None:
            depth_losses.append(
                _compaction_depth_l2(
                    alpha=alpha,
                    depth_sum=depth_sum,
                    depth_square=depth_square,
                    target_depth=targets.depth[camera_index],
                    mask=mask,
                )
            )

    return RayCompactionLossMaps(
        color_l2=(
            torch.stack(color_losses, dim=0)
            if color_losses is not None
            else None
        ),
        depth_l2=(
            torch.stack(depth_losses, dim=0)
            if depth_losses is not None
            else None
        ),
        weight_sum=torch.stack(alphas, dim=0),
        metadata={
            "backend": "faster_gs.gaussian_wrapping",
            "rasterizer_mode": resolved_options.rasterizer_mode,
        },
    )


def _quaternion_to_rotation_matrix(
    quaternion_orientation: Float[Tensor, "num_splats 4"],
) -> Float[Tensor, "num_splats 3 3"]:
    quaternion = torch.nn.functional.normalize(
        quaternion_orientation,
        dim=-1,
    )
    w, x, y, z = quaternion.unbind(dim=-1)
    one = torch.ones_like(w)
    two = one + one
    return torch.stack(
        (
            torch.stack(
                (
                    one - two * (y * y + z * z),
                    two * (x * y - z * w),
                    two * (x * z + y * w),
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    two * (x * y + z * w),
                    one - two * (x * x + z * z),
                    two * (y * z - x * w),
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    two * (x * z - y * w),
                    two * (y * z + x * w),
                    one - two * (x * x + y * y),
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )


@dataclass(frozen=True)
class GaussianWrappingSurfaceProvider(WrappingSurfaceProvider):
    """Wrapping trait provider backed by Gaussian Wrapping CUDA stages."""

    def surface_evidence(
        self,
        request: MeshificationRequest,
    ) -> WrappingSurfaceEvidence:
        """Render wrapping evidence for a meshification request."""
        if not isinstance(request.scene, GaussianScene3D):
            raise TypeError(
                "GaussianWrappingSurfaceProvider expects GaussianScene3D; "
                f"got {type(request.scene).__name__}."
            )
        options = (
            request.backend_options
            if isinstance(
                request.backend_options,
                GaussianWrappingNativeRenderOptions,
            )
            else GaussianWrappingNativeRenderOptions()
        )
        output = render_gaussian_wrapping_native(
            request.scene,
            request.camera,
            return_alpha=True,
            return_depth=True,
            return_normals=True,
            options=options,
        )
        return WrappingSurfaceEvidence(
            render=output.render,
            alpha=output.alphas,
            depth=output.depth,
            median_depth=output.median_depth,
            expected_depth=output.expected_depth,
            normals=output.normals,
            valid_mask=output.alphas > 0.0,
            metadata={
                "rasterizer_mode": options.rasterizer_mode,
                "backend": request.backend,
            },
        )

    def sample_surface_points(
        self,
        request: MeshificationRequest,
    ) -> SurfacePointSamples:
        """Return Gaussian center/corner pivots as wrapping samples."""
        if not isinstance(request.scene, GaussianScene3D):
            raise TypeError(
                "GaussianWrappingSurfaceProvider expects GaussianScene3D; "
                f"got {type(request.scene).__name__}."
            )
        scene = request.scene
        offsets = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=scene.center_position.dtype,
            device=scene.center_position.device,
        )
        rotations = _quaternion_to_rotation_matrix(scene.quaternion_orientation)
        scales = torch.exp(scene.log_scales)
        local_offsets = offsets[None, :, :] * scales[:, None, :]
        rotated_offsets = torch.einsum(
            "nij,npj->npi",
            rotations,
            local_offsets,
        )
        points = (scene.center_position[:, None, :] + rotated_offsets).reshape(
            -1,
            3,
        )
        point_scales = scales.mean(dim=-1)[:, None].expand(-1, 9).reshape(-1)
        return SurfacePointSamples(
            points=points,
            scales=point_scales,
            attributes={
                "primitive_index": torch.arange(
                    scene.center_position.shape[0],
                    device=scene.center_position.device,
                ).repeat_interleave(9),
            },
        )

    def query_wrapping_field(
        self,
        request: MeshificationRequest,
        points: Float[Tensor, " num_points 3"],
    ) -> WrappingQueryResult:
        """Evaluate the native wrapping field at world-space points."""
        if not isinstance(request.scene, GaussianScene3D):
            raise TypeError(
                "GaussianWrappingSurfaceProvider expects GaussianScene3D; "
                f"got {type(request.scene).__name__}."
            )
        if points.device != request.scene.center_position.device:
            raise ValueError(
                "Gaussian Wrapping field queries require points on the scene "
                "device."
            )
        options = (
            request.backend_options
            if isinstance(
                request.backend_options,
                GaussianWrappingNativeRenderOptions,
            )
            else GaussianWrappingNativeRenderOptions()
        )
        values: Tensor | None = None
        inside: Tensor | None = None
        for camera_index in range(request.camera.cam_to_world.shape[0]):
            camera_values, camera_inside = _query_single_camera(
                request.scene,
                request.camera,
                camera_index,
                points,
                options,
            )
            values = (
                camera_values
                if values is None
                else torch.maximum(values, camera_values)
            )
            inside = (
                camera_inside
                if inside is None
                else torch.logical_or(inside, camera_inside)
            )
        if values is None or inside is None:
            raise ValueError("Gaussian Wrapping field query needs a camera.")
        return WrappingQueryResult(
            values=values,
            inside=inside,
            metadata={"rasterizer_mode": options.rasterizer_mode},
        )


def _query_single_camera(
    scene: GaussianScene3D,
    camera: CameraState,
    camera_index: int,
    points: Float[Tensor, " num_points 3"],
    options: GaussianWrappingNativeRenderOptions,
) -> tuple[Float[Tensor, " num_points"], Bool[Tensor, " num_points"]]:
    camera_params = _camera_stage_params(camera, camera_index, options)
    (
        center_positions,
        opacities,
        scales,
        rotations,
        colors,
        spherical_harmonics,
        empty,
        _empty_extra,
    ) = _scene_stage_inputs(scene, options)
    if options.rasterizer_mode == "ours":
        (
            _rendered_count,
            transmittance,
            inside,
        ) = ours_integrate_points_fwd_op(
            points.contiguous(),
            center_positions,
            opacities,
            scales,
            rotations,
            options.scale_modifier,
            empty,
            empty,
            camera_params.viewmatrix,
            camera_params.projmatrix,
            camera_params.tan_fovx,
            camera_params.tan_fovy,
            options.kernel_size,
            camera_params.image_height,
            camera_params.image_width,
            camera_params.camera_position,
            False,
            options.debug,
        )
        return 1.0 - transmittance, inside

    subpixel_offset = torch.zeros(
        (camera_params.image_height, camera_params.image_width, 2),
        dtype=scene.center_position.dtype,
        device=scene.center_position.device,
    )
    (
        _rendered_count,
        _color,
        alpha_integrated,
        _color_integrated,
        _point_coordinate,
        point_sdf,
        _radii,
        _geom_buffer,
        _binning_buffer,
        _image_buffer,
    ) = radegs_integrate_points_fwd_op(
        options.background_color.to(
            device=scene.center_position.device,
            dtype=scene.center_position.dtype,
        ).contiguous(),
        points.contiguous(),
        center_positions,
        colors,
        opacities,
        scales,
        rotations,
        options.scale_modifier,
        empty,
        empty,
        camera_params.viewmatrix,
        camera_params.projmatrix,
        camera_params.tan_fovx,
        camera_params.tan_fovy,
        options.kernel_size,
        subpixel_offset,
        camera_params.image_height,
        camera_params.image_width,
        spherical_harmonics,
        scene.sh_degree,
        camera_params.camera_position,
        False,
        options.debug,
    )
    return alpha_integrated, point_sdf <= 0.0


def register() -> None:
    """Register the Gaussian Wrapping backend."""
    register_backend(
        name="faster_gs.gaussian_wrapping",
        default_options=GaussianWrappingNativeRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=_SUPPORTED_OUTPUTS,
        trait_providers=(GaussianWrappingSurfaceProvider(),),
    )(render_gaussian_wrapping_native)
