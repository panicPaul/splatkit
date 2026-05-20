"""Rasterization-stage native ops for the NHT runtime."""

from __future__ import annotations

from typing import Any

import torch
from ember_native_nht.threedgut.core.runtime.ops._common import (
    CameraModelName,
    backend,
    camera_model_type,
    default_ftheta_distortion_parameters,
    default_unscented_transform_parameters,
    encoding_expansion_factor,
    feature_divisor,
    global_shutter_type,
)
from ember_native_nht.threedgut.core.runtime.packing import (
    parse_depth_rasterization_outputs,
    parse_feature_rasterization_outputs,
)
from ember_native_nht.threedgut.core.runtime.types import (
    DepthRasterizationResult,
    FeatureRasterizationResult,
)
from torch import Tensor

SUPPORTED_CHANNELS = (
    1,
    2,
    3,
    4,
    5,
    8,
    9,
    12,
    16,
    17,
    20,
    24,
    28,
    32,
    33,
    36,
    40,
    44,
    48,
    49,
    64,
    65,
    80,
    96,
    128,
    129,
    256,
    257,
    512,
    513,
)
SUPPORTED_DEPTH_CHANNELS = tuple(
    channel_count for channel_count in SUPPORTED_CHANNELS if channel_count != 49
)


def _find_next_supported_channel_count(
    channel_count: int,
    supported_channel_counts: tuple[int, ...],
) -> int:
    """Find the smallest supported channel count greater than or equal to input."""
    for supported_channel_count in supported_channel_counts:
        if supported_channel_count >= channel_count:
            return supported_channel_count
    return -1


def _pad_depth_channels(
    features: Tensor,
    backgrounds: Tensor | None,
) -> tuple[Tensor, Tensor | None, int]:
    """Pad eval3d features to a channel count supported by the native templates."""
    channel_count = int(features.shape[-1])
    if channel_count > 513 or channel_count == 0:
        raise ValueError(
            f"Unsupported number of depth channels: {channel_count}."
        )
    if channel_count in SUPPORTED_DEPTH_CHANNELS:
        return features, backgrounds, 0

    padded_channel_count = _find_next_supported_channel_count(
        channel_count,
        SUPPORTED_DEPTH_CHANNELS,
    )
    if padded_channel_count < 0:
        raise ValueError(
            f"Unsupported number of depth channels: {channel_count}."
        )
    padding_channel_count = padded_channel_count - channel_count
    padded_features = torch.cat(
        [
            features,
            torch.zeros(
                *features.shape[:-1],
                padding_channel_count,
                device=features.device,
                dtype=features.dtype,
            ),
        ],
        dim=-1,
    )
    padded_backgrounds = None
    if backgrounds is not None:
        padded_backgrounds = torch.cat(
            [
                backgrounds,
                torch.zeros(
                    *backgrounds.shape[:-1],
                    padding_channel_count,
                    device=backgrounds.device,
                    dtype=backgrounds.dtype,
                ),
            ],
            dim=-1,
        )
    return padded_features, padded_backgrounds, padding_channel_count


def _pad_nht_feature_channels(
    features: Tensor,
    backgrounds: Tensor | None,
) -> tuple[Tensor, Tensor | None, int, int]:
    """Pad NHT vertex features while preserving tetrahedron vertex boundaries."""
    input_channel_count = int(features.shape[-1])
    divisor = feature_divisor()
    expansion_factor = encoding_expansion_factor()
    base_feature_count = input_channel_count // divisor
    original_output_channel_count = base_feature_count * expansion_factor
    if input_channel_count in SUPPORTED_CHANNELS:
        return features, backgrounds, original_output_channel_count, 0

    padded_input_channel_count = _find_next_supported_channel_count(
        input_channel_count,
        SUPPORTED_CHANNELS,
    )
    if padded_input_channel_count < 0:
        raise ValueError(
            f"Unsupported NHT input channels: {input_channel_count}."
        )

    padded_base_feature_count = padded_input_channel_count // divisor
    padding_per_vertex = padded_base_feature_count - base_feature_count
    padded_features = features.unflatten(-1, (divisor, base_feature_count))
    padded_features = torch.nn.functional.pad(
        padded_features, (0, padding_per_vertex)
    )
    padded_features = padded_features.flatten(-2, -1)

    padded_backgrounds = backgrounds
    if backgrounds is not None:
        padded_output_channel_count = (
            padded_base_feature_count * expansion_factor
        )
        output_padding_count = (
            padded_output_channel_count - original_output_channel_count
        )
        padded_backgrounds = torch.cat(
            [
                backgrounds,
                torch.zeros(
                    *backgrounds.shape[:-1],
                    output_padding_count,
                    device=backgrounds.device,
                    dtype=backgrounds.dtype,
                ),
            ],
            dim=-1,
        )
    return (
        padded_features,
        padded_backgrounds,
        original_output_channel_count,
        padded_input_channel_count - input_channel_count,
    )


class _RasterizeFeatures(torch.autograd.Function):
    """Autograd wrapper for the native NHT feature rasterization stage."""

    @staticmethod
    def forward(
        context: Any,
        center_positions: Tensor,
        quaternions: Tensor,
        scales: Tensor,
        features: Tensor,
        opacities: Tensor,
        backgrounds: Tensor | None,
        masks: Tensor | None,
        world_to_camera_matrices: Tensor,
        camera_intrinsics: Tensor,
        image_width: int,
        image_height: int,
        tile_size: int,
        tile_offsets: Tensor,
        instance_primitive_indices: Tensor,
        camera_model: CameraModelName,
        center_ray_mode: bool,
        ray_direction_scale: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        native_camera_model = camera_model_type(camera_model)
        native_unscented_transform_parameters = (
            default_unscented_transform_parameters()
        )
        native_shutter_type = global_shutter_type()
        native_ftheta_parameters = default_ftheta_distortion_parameters()

        rendered_features, rendered_alphas, feature_square_sums, last_gaussian_ids = (
            backend().rasterize_features_fwd(
                center_positions,
                quaternions,
                scales,
                features,
                opacities,
                backgrounds,
                masks,
                image_width,
                image_height,
                tile_size,
                world_to_camera_matrices,
                None,
                camera_intrinsics,
                native_camera_model,
                native_unscented_transform_parameters,
                native_shutter_type,
                None,
                None,
                None,
                native_ftheta_parameters,
                tile_offsets,
                instance_primitive_indices,
                center_ray_mode,
                ray_direction_scale,
            )
        )

        context.save_for_backward(
            center_positions,
            quaternions,
            scales,
            features,
            opacities,
            backgrounds,
            masks,
            world_to_camera_matrices,
            camera_intrinsics,
            tile_offsets,
            instance_primitive_indices,
            rendered_alphas,
            last_gaussian_ids,
        )
        context.image_width = image_width
        context.image_height = image_height
        context.tile_size = tile_size
        context.native_camera_model = native_camera_model
        context.native_unscented_transform_parameters = (
            native_unscented_transform_parameters
        )
        context.native_shutter_type = native_shutter_type
        context.native_ftheta_parameters = native_ftheta_parameters
        return rendered_features, rendered_alphas, feature_square_sums

    @staticmethod
    def backward(
        context: Any,
        grad_rendered_features: Tensor,
        grad_rendered_alphas: Tensor,
        grad_feature_square_sums: Tensor,
    ) -> tuple[Any, ...]:
        (
            center_positions,
            quaternions,
            scales,
            features,
            opacities,
            backgrounds,
            masks,
            world_to_camera_matrices,
            camera_intrinsics,
            tile_offsets,
            instance_primitive_indices,
            rendered_alphas,
            last_gaussian_ids,
        ) = context.saved_tensors

        # The native NHT rasterizer appends three ray-direction channels. They are
        # shader inputs, not differentiable vertex features.
        grad_feature_channels = (
            grad_rendered_features[..., :-3].contiguous().float()
        )
        if grad_feature_square_sums is None:
            grad_feature_square_sums = torch.zeros_like(
                grad_rendered_features[..., :-3]
            )
        grad_feature_square_sums = grad_feature_square_sums.contiguous().float()
        (
            grad_center_positions,
            grad_quaternions,
            grad_scales,
            grad_features,
            grad_opacities,
        ) = backend().rasterize_features_bwd(
            center_positions,
            quaternions,
            scales,
            features,
            opacities,
            backgrounds,
            masks,
            context.image_width,
            context.image_height,
            context.tile_size,
            world_to_camera_matrices,
            None,
            camera_intrinsics,
            context.native_camera_model,
            context.native_unscented_transform_parameters,
            context.native_shutter_type,
            None,
            None,
            None,
            context.native_ftheta_parameters,
            tile_offsets,
            instance_primitive_indices,
            rendered_alphas,
            last_gaussian_ids,
            grad_feature_channels,
            grad_feature_square_sums,
            grad_rendered_alphas.contiguous(),
        )

        grad_backgrounds = None
        if context.needs_input_grad[5]:
            grad_background_features = grad_rendered_features[..., :-3]
            grad_backgrounds = (
                grad_background_features * (1.0 - rendered_alphas).float()
            ).sum(dim=(-3, -2))

        return (
            grad_center_positions,
            grad_quaternions,
            grad_scales,
            grad_features,
            grad_opacities,
            grad_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _RasterizeDepth(torch.autograd.Function):
    """Autograd wrapper for eval3d depth rasterization."""

    @staticmethod
    def forward(
        context: Any,
        center_positions: Tensor,
        quaternions: Tensor,
        scales: Tensor,
        depth_features: Tensor,
        opacities: Tensor,
        backgrounds: Tensor | None,
        masks: Tensor | None,
        world_to_camera_matrices: Tensor,
        camera_intrinsics: Tensor,
        image_width: int,
        image_height: int,
        tile_size: int,
        tile_offsets: Tensor,
        instance_primitive_indices: Tensor,
        camera_model: CameraModelName,
    ) -> tuple[Tensor, Tensor]:
        native_camera_model = camera_model_type(camera_model)
        native_unscented_transform_parameters = (
            default_unscented_transform_parameters()
        )
        native_shutter_type = global_shutter_type()
        native_ftheta_parameters = default_ftheta_distortion_parameters()

        rendered_depths, rendered_alphas, last_gaussian_ids = (
            backend().rasterize_depth_fwd(
                center_positions,
                quaternions,
                scales,
                depth_features,
                opacities,
                backgrounds,
                masks,
                image_width,
                image_height,
                tile_size,
                world_to_camera_matrices,
                None,
                camera_intrinsics,
                native_camera_model,
                native_unscented_transform_parameters,
                native_shutter_type,
                None,
                None,
                None,
                native_ftheta_parameters,
                tile_offsets,
                instance_primitive_indices,
            )
        )

        context.save_for_backward(
            center_positions,
            quaternions,
            scales,
            depth_features,
            opacities,
            backgrounds,
            masks,
            world_to_camera_matrices,
            camera_intrinsics,
            tile_offsets,
            instance_primitive_indices,
            rendered_alphas,
            last_gaussian_ids,
        )
        context.image_width = image_width
        context.image_height = image_height
        context.tile_size = tile_size
        context.native_camera_model = native_camera_model
        context.native_unscented_transform_parameters = (
            native_unscented_transform_parameters
        )
        context.native_shutter_type = native_shutter_type
        context.native_ftheta_parameters = native_ftheta_parameters
        return rendered_depths, rendered_alphas

    @staticmethod
    def backward(
        context: Any,
        grad_rendered_depths: Tensor,
        grad_rendered_alphas: Tensor,
    ) -> tuple[Any, ...]:
        (
            center_positions,
            quaternions,
            scales,
            depth_features,
            opacities,
            backgrounds,
            masks,
            world_to_camera_matrices,
            camera_intrinsics,
            tile_offsets,
            instance_primitive_indices,
            rendered_alphas,
            last_gaussian_ids,
        ) = context.saved_tensors

        (
            grad_center_positions,
            grad_quaternions,
            grad_scales,
            grad_depth_features,
            grad_opacities,
        ) = backend().rasterize_depth_bwd(
            center_positions,
            quaternions,
            scales,
            depth_features,
            opacities,
            backgrounds,
            masks,
            context.image_width,
            context.image_height,
            context.tile_size,
            world_to_camera_matrices,
            None,
            camera_intrinsics,
            context.native_camera_model,
            context.native_unscented_transform_parameters,
            context.native_shutter_type,
            None,
            None,
            None,
            context.native_ftheta_parameters,
            tile_offsets,
            instance_primitive_indices,
            rendered_alphas,
            last_gaussian_ids,
            grad_rendered_depths.contiguous(),
            grad_rendered_alphas.contiguous(),
        )

        grad_backgrounds = None
        if context.needs_input_grad[5]:
            grad_backgrounds = (
                grad_rendered_depths * (1.0 - rendered_alphas).float()
            ).sum(dim=(-3, -2))

        return (
            grad_center_positions,
            grad_quaternions,
            grad_scales,
            grad_depth_features,
            grad_opacities,
            grad_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rasterize_features(
    *,
    center_positions: Tensor,
    quaternions: Tensor,
    scales: Tensor,
    features: Tensor,
    opacities: Tensor,
    world_to_camera_matrices: Tensor,
    camera_intrinsics: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_offsets: Tensor,
    camera_model: CameraModelName,
    center_ray_mode: bool,
    ray_direction_scale: float,
    instance_primitive_indices: Tensor | None = None,
    flattened_gaussian_ids: Tensor | None = None,
    backgrounds: Tensor | None = None,
    masks: Tensor | None = None,
) -> FeatureRasterizationResult:
    """Rasterize NHT features and appended ray-direction channels."""
    if instance_primitive_indices is None:
        if flattened_gaussian_ids is None:
            raise TypeError(
                "rasterize_features requires instance_primitive_indices."
            )
        instance_primitive_indices = flattened_gaussian_ids
    original_output_channel_count = (
        int(features.shape[-1]) // feature_divisor()
    ) * encoding_expansion_factor()
    padded_features, padded_backgrounds, original_output_channel_count, _ = (
        _pad_nht_feature_channels(features, backgrounds)
    )
    rendered_features, rendered_alphas, feature_square_sums = (
        _RasterizeFeatures.apply(
            center_positions.contiguous(),
            quaternions.contiguous(),
            scales.contiguous(),
            padded_features.contiguous(),
            opacities.contiguous(),
            padded_backgrounds.contiguous()
            if padded_backgrounds is not None
            else None,
            masks.contiguous() if masks is not None else None,
            world_to_camera_matrices.contiguous(),
            camera_intrinsics.contiguous(),
            image_width,
            image_height,
            tile_size,
            tile_offsets.contiguous(),
            instance_primitive_indices.contiguous(),
            camera_model,
            center_ray_mode,
            ray_direction_scale,
        )
    )

    ray_direction_features = rendered_features[..., -3:]
    rendered_vertex_features = rendered_features[..., :-3]
    if rendered_vertex_features.shape[-1] > original_output_channel_count:
        rendered_vertex_features = rendered_vertex_features[
            ..., :original_output_channel_count
        ]
        feature_square_sums = feature_square_sums[
            ..., :original_output_channel_count
        ]
    rendered_features = torch.cat(
        [rendered_vertex_features, ray_direction_features],
        dim=-1,
    )
    return parse_feature_rasterization_outputs(
        (rendered_features, rendered_alphas, feature_square_sums)
    )


def rasterize_depth(
    *,
    center_positions: Tensor,
    quaternions: Tensor,
    scales: Tensor,
    depth_features: Tensor,
    opacities: Tensor,
    world_to_camera_matrices: Tensor,
    camera_intrinsics: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_offsets: Tensor,
    camera_model: CameraModelName,
    instance_primitive_indices: Tensor | None = None,
    flattened_gaussian_ids: Tensor | None = None,
    backgrounds: Tensor | None = None,
    masks: Tensor | None = None,
) -> DepthRasterizationResult:
    """Rasterize eval3d depth features."""
    if instance_primitive_indices is None:
        if flattened_gaussian_ids is None:
            raise TypeError(
                "rasterize_depth requires instance_primitive_indices."
            )
        instance_primitive_indices = flattened_gaussian_ids
    padded_depth_features, padded_backgrounds, padding_channel_count = (
        _pad_depth_channels(depth_features, backgrounds)
    )
    rendered_depths, rendered_alphas = _RasterizeDepth.apply(
        center_positions.contiguous(),
        quaternions.contiguous(),
        scales.contiguous(),
        padded_depth_features.contiguous(),
        opacities.contiguous(),
        padded_backgrounds.contiguous()
        if padded_backgrounds is not None
        else None,
        masks.contiguous() if masks is not None else None,
        world_to_camera_matrices.contiguous(),
        camera_intrinsics.contiguous(),
        image_width,
        image_height,
        tile_size,
        tile_offsets.contiguous(),
        instance_primitive_indices.contiguous(),
        camera_model,
    )
    if padding_channel_count > 0:
        rendered_depths = rendered_depths[..., :-padding_channel_count]
    return parse_depth_rasterization_outputs((rendered_depths, rendered_alphas))


def rasterize_gaussian_indices(
    *,
    transmittances: Tensor,
    projected_means: Tensor,
    conics: Tensor,
    opacities: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_offsets: Tensor,
    range_start: int = 0,
    range_end: int = 2**31 - 1,
    instance_primitive_indices: Tensor | None = None,
    flattened_gaussian_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return Gaussian and pixel ids for native 3DGS pixel contributors."""
    if instance_primitive_indices is None:
        if flattened_gaussian_ids is None:
            raise TypeError(
                "rasterize_gaussian_indices requires instance_primitive_indices."
            )
        instance_primitive_indices = flattened_gaussian_ids
    return backend().rasterize_to_indices_fwd(
        int(range_start),
        int(range_end),
        transmittances.contiguous(),
        projected_means.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        int(image_width),
        int(image_height),
        int(tile_size),
        tile_offsets.contiguous(),
        instance_primitive_indices.contiguous(),
    )
