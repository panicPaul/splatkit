"""NHT 3DGUT renderer with FastGS densification traits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal  # noqa: F401

import torch
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

from ember_native_nht.threedgut.core.runtime import (
    RenderResult,
    rasterize_gaussian_indices,
)
from ember_native_nht.threedgut.core.runtime import (
    render as render_runtime,
)
from ember_native_nht.threedgut.renderer import (
    NHT3DGUTRenderOptions,
    NHT3DGUTRenderOutput,
    _validate_inputs,
)

_SUPPORTED_OUTPUTS = output_set("alpha", "depth")


@beartype
@dataclass(frozen=True)
class NHTFastGSRenderOptions(NHT3DGUTRenderOptions):
    """NHT 3DGUT render options with FastGS densification collection."""

    collect_densification_info: bool = False


@beartype
@dataclass(frozen=True)
class NHTFastGSRenderOutput(NHT3DGUTRenderOutput):
    """NHT 3DGUT output with projection metadata for FastGS traits."""

    radii: Int[Tensor, " num_cams num_splats 2"]
    projected_means: Float[Tensor, " num_cams num_splats 2"]
    primitive_depths: Float[Tensor, " num_cams num_splats"]
    conics: Float[Tensor, " num_cams num_splats 3"]
    tile_offsets: Int[Tensor, " num_cams tile_height tile_width"]
    flattened_gaussian_ids: Int[Tensor, " num_intersections"]
    mip_splatting_screen_filter_compensations: (
        Float[Tensor, " num_cams num_splats"] | None
    )


def _render_nht_result(
    scene: GaussianScene3D,
    camera: CameraState,
    options: NHTFastGSRenderOptions,
) -> RenderResult:
    return render_runtime(
        center_positions=scene.center_position.contiguous(),
        quaternions=torch.nn.functional.normalize(
            scene.quaternion_orientation, dim=-1
        ).contiguous(),
        scales=torch.exp(scene.log_scales).contiguous(),
        opacities=torch.sigmoid(scene.logit_opacity).contiguous(),
        features=scene.feature.contiguous(),
        world_to_camera_matrices=torch.linalg.inv(
            camera.cam_to_world
        ).contiguous(),
        camera_intrinsics=camera.get_intrinsics().contiguous(),
        image_width=int(camera.width[0].item()),
        image_height=int(camera.height[0].item()),
        tile_size=options.tile_size,
        mip_splatting_screen_filter=options.mip_splatting_screen_filter,
        render_mode="RGB+ED",
        near_plane=options.near_plane,
        far_plane=options.far_plane,
        radius_clip=options.radius_clip,
        eps2d=options.eps2d,
        camera_model=options.camera_model,
        center_ray_mode=options.center_ray_mode,
        ray_direction_scale=options.ray_dir_scale,
    )


def _output_from_result(
    scene: GaussianScene3D,
    result: RenderResult,
) -> NHTFastGSRenderOutput:
    features = result.renders[..., :-1]
    depth = result.renders[..., -1]
    num_splats = int(scene.center_position.shape[0])
    per_splat = result.intersections.tiles_per_gaussian.reshape(
        -1,
        num_splats,
    ).sum(dim=0)
    visibility = (per_splat > 0).to(
        dtype=scene.center_position.dtype,
        device=scene.center_position.device,
    )[:, None]
    weights = per_splat.to(
        dtype=scene.center_position.dtype,
        device=scene.center_position.device,
    )[:, None]
    render = features[..., :3] if features.shape[-1] >= 3 else features
    return NHTFastGSRenderOutput(
        render=render,
        features=features,
        alphas=result.alphas.squeeze(-1),
        depth=depth,
        visibility=visibility,
        weights=weights,
        radii=result.projection.radii,
        projected_means=result.projection.projected_means,
        primitive_depths=result.projection.primitive_depths,
        conics=result.projection.conics,
        tile_offsets=result.intersections.tile_offsets,
        flattened_gaussian_ids=result.intersections.flattened_gaussian_ids,
        mip_splatting_screen_filter_compensations=(
            result.projection.mip_splatting_screen_filter_compensations
        ),
    )


def _raw_output(output: object) -> object:
    return getattr(output, "raw_output", output)


def _camera_position(camera: CameraState) -> Tensor:
    return camera.cam_to_world[0, :3, 3]


def _max_screen_radii(output: object, scene: GaussianScene3D) -> Tensor:
    raw_output = _raw_output(output)
    radii = getattr(raw_output, "radii", None)
    if not isinstance(radii, Tensor):
        return torch.zeros(
            (int(scene.center_position.shape[0]),),
            dtype=scene.center_position.dtype,
            device=scene.center_position.device,
        )
    return radii.reshape(-1, radii.shape[-2], 2).amax(dim=(0, 2)).to(
        dtype=scene.center_position.dtype,
        device=scene.center_position.device,
    )


def _visible_weights(output: object, scene: GaussianScene3D) -> Tensor:
    raw_output = _raw_output(output)
    weights = getattr(raw_output, "weights", None)
    if not isinstance(weights, Tensor):
        return torch.zeros(
            (int(scene.center_position.shape[0]),),
            dtype=scene.center_position.dtype,
            device=scene.center_position.device,
        )
    return weights.detach().reshape(-1).to(
        dtype=scene.center_position.dtype,
        device=scene.center_position.device,
    )


def nht_fast_gs_metric_counts(
    *,
    output: NHTFastGSRenderOutput,
    opacities: Tensor,
    metric_map: Tensor,
    tile_size: int,
) -> Tensor:
    """Count native NHT Gaussian contributors on metric-map pixels."""
    if output.render.shape[0] != 1:
        raise ValueError("NHT-Fast-GS metric counting expects one camera.")
    height = int(output.render.shape[1])
    width = int(output.render.shape[2])
    transmittances = torch.ones(
        (1, height, width),
        dtype=opacities.dtype,
        device=opacities.device,
    )
    resolved_opacities = opacities[None]
    compensations = output.mip_splatting_screen_filter_compensations
    if compensations is not None:
        resolved_opacities = resolved_opacities * compensations
    gaussian_ids, pixel_ids = rasterize_gaussian_indices(
        transmittances=transmittances,
        projected_means=output.projected_means,
        conics=output.conics,
        opacities=resolved_opacities,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        tile_offsets=output.tile_offsets,
        flattened_gaussian_ids=output.flattened_gaussian_ids,
    )
    metric_flat = metric_map.reshape(-1).to(
        device=pixel_ids.device,
        dtype=torch.bool,
    )
    selected = metric_flat[pixel_ids.to(torch.long)]
    return torch.bincount(
        gaussian_ids[selected].to(torch.long),
        minlength=int(opacities.shape[0]),
    ).to(dtype=opacities.dtype, device=opacities.device)


@beartype
@dataclass(frozen=True)
class NHTFastGSSignalProvider(GaussianFastGSSignalProvider):
    """Extract FastGS-style density-control signals from NHT outputs."""

    def prepare_fastgs_signals(self, context: DensificationContext) -> None:
        """NHT gradients are available through scene tensors."""
        del context

    def collect_fastgs_signals(
        self,
        context: DensificationContext,
    ) -> GaussianFastGSDensificationSignals | None:
        """Collect NHT visibility, gradient, and screen-radius signals."""
        scene = context.state.model.scene
        if not isinstance(scene, GaussianScene3D):
            return None
        position_grad = scene.center_position.grad
        if position_grad is None:
            return None
        weights = _visible_weights(context.render_output, scene)
        visibility = weights > 0
        raw_output = _raw_output(context.render_output)
        if hasattr(raw_output, "visibility"):
            visibility |= raw_output.visibility.detach().reshape(-1).to(
                device=scene.center_position.device
            ) > 0
        visible_count = visibility.to(dtype=scene.center_position.dtype)
        camera_distance = (
            scene.center_position.detach() - _camera_position(context.batch.camera)
        ).norm(dim=-1)
        grad_sum = position_grad.detach().norm(dim=-1) * camera_distance
        grad_sum = grad_sum * visible_count
        return GaussianFastGSDensificationSignals(
            visible_count=visible_count,
            clone_grad_sum=grad_sum,
            split_grad_sum=grad_sum,
            max_screen_radii=(
                _max_screen_radii(context.render_output, scene) * visible_count
            ),
        )


@beartype
@dataclass(frozen=True)
class NHTFastGSMetricAttribution(GaussianMetricAttribution):
    """Attribute FastGS probe metric maps with native NHT contributor IDs."""

    def attribute_metric_map(
        self,
        scene: GaussianScene3D,
        camera: CameraState,
        metric_map: Int[Tensor, " *metric_dims"],
        *,
        options: NHTFastGSRenderOptions | None = None,
    ) -> Float[Tensor, " num_splats"]:
        """Attribute a single-camera metric map to NHT Gaussians exactly."""
        if camera.cam_to_world.shape[0] != 1:
            raise ValueError(
                "NHT-Fast-GS metric attribution expects a single probe camera."
            )
        if metric_map.ndim != 2:
            raise ValueError(
                "NHT-Fast-GS metric attribution expects a 2D metric map with "
                f"shape (height, width); got {tuple(metric_map.shape)}."
            )
        resolved_options = options or NHTFastGSRenderOptions()
        with torch.no_grad():
            output = render_nht_fast_gs(
                scene,
                camera,
                return_alpha=True,
                return_depth=True,
                options=resolved_options,
            )
            return nht_fast_gs_metric_counts(
                output=output,
                opacities=torch.sigmoid(scene.logit_opacity).to(
                    dtype=scene.center_position.dtype
                ),
                metric_map=metric_map,
                tile_size=resolved_options.tile_size,
            )


@beartype
def render_nht_fast_gs(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = True,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: NHTFastGSRenderOptions | None = None,
) -> NHTFastGSRenderOutput:
    """Render NHT features while accepting FastGS densification options."""
    del return_alpha, return_depth
    if return_gaussian_impact_score:
        raise ValueError("nht.3dgut_fast_gs does not expose impact scores.")
    if return_normals:
        raise ValueError("nht.3dgut_fast_gs does not expose normals.")
    if return_2d_projections:
        raise ValueError("nht.3dgut_fast_gs does not expose 2D projections.")
    if return_projective_intersection_transforms:
        raise ValueError(
            "nht.3dgut_fast_gs does not expose projective transforms."
        )
    _validate_inputs(scene, camera)
    result = _render_nht_result(
        scene,
        camera,
        options or NHTFastGSRenderOptions(),
    )
    return _output_from_result(scene, result)


def register() -> None:
    """Register the NHT 3DGUT FastGS backend."""
    register_backend(
        name="nht.3dgut_fast_gs",
        default_options=NHTFastGSRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=_SUPPORTED_OUTPUTS,
        trait_providers=(
            NHTFastGSSignalProvider(),
            NHTFastGSMetricAttribution(),
        ),
    )(render_nht_fast_gs)


__all__ = [
    "NHTFastGSMetricAttribution",
    "NHTFastGSRenderOptions",
    "NHTFastGSRenderOutput",
    "NHTFastGSSignalProvider",
    "nht_fast_gs_metric_counts",
    "register",
    "render_nht_fast_gs",
]
