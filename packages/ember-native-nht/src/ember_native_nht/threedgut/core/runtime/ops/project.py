"""Projection-stage native ops for the NHT runtime."""

from __future__ import annotations

from ember_native_nht.threedgut.core.runtime.ops._common import (
    CameraModelName,
    backend,
    camera_model_type,
    default_ftheta_distortion_parameters,
    default_unscented_transform_parameters,
    global_shutter_type,
)
from ember_native_nht.threedgut.core.runtime.packing import (
    parse_projection_outputs,
)
from ember_native_nht.threedgut.core.runtime.types import ProjectionResult
from torch import Tensor


def project(
    *,
    center_positions: Tensor,
    quaternions: Tensor,
    scales: Tensor,
    opacities: Tensor,
    world_to_camera_matrices: Tensor,
    camera_intrinsics: Tensor,
    image_width: int,
    image_height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    calculate_compensations: bool,
    camera_model: CameraModelName,
) -> ProjectionResult:
    """Run the 3DGUT unscented-transform projection stage."""
    raw_outputs = backend().project_3dgut_fwd(
        center_positions.contiguous(),
        quaternions.contiguous(),
        scales.contiguous(),
        opacities.contiguous(),
        world_to_camera_matrices.contiguous(),
        None,
        camera_intrinsics.contiguous(),
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        calculate_compensations,
        camera_model_type(camera_model),
        default_unscented_transform_parameters(),
        global_shutter_type(),
        None,
        None,
        None,
        default_ftheta_distortion_parameters(),
    )
    return parse_projection_outputs(raw_outputs)
