"""Optional scene-record post-processing helpers."""

from __future__ import annotations

from dataclasses import replace

import torch
from jaxtyping import Float
from torch import Tensor

from ember_core.data.contracts import (
    CameraSensorDataset,
    HorizonAdjustmentSpec,
    SceneRecord,
)
from ember_core.data.pipes import HorizonAlignPipeConfig, register_source_pipe


def _normalize(vector: Float[Tensor, " 3"]) -> Float[Tensor, " 3"]:
    return vector / torch.linalg.norm(vector)


def _rotation_aligning_vectors(
    source: Float[Tensor, " 3"],
    target: Float[Tensor, " 3"],
) -> Float[Tensor, " 3 3"]:
    source = _normalize(source)
    target = _normalize(target)
    cross = torch.cross(source, target, dim=0)
    dot = torch.clamp(torch.dot(source, target), min=-1.0, max=1.0)
    cross_norm = torch.linalg.norm(cross)
    if cross_norm < 1e-8:
        if dot > 0.0:
            return torch.eye(3, dtype=source.dtype, device=source.device)
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=source.dtype)
        if torch.abs(source[0]) > 0.9:
            axis = torch.tensor([0.0, 0.0, 1.0], dtype=source.dtype)
        cross = _normalize(torch.cross(source, axis, dim=0))
        cross_norm = torch.linalg.norm(cross)
    skew = torch.tensor(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=source.dtype,
        device=source.device,
    )
    identity = torch.eye(3, dtype=source.dtype, device=source.device)
    if cross_norm < 1e-8:
        return identity
    return (
        identity
        + skew
        + skew @ skew * ((1.0 - dot) / (cross_norm * cross_norm))
    )


def _estimate_world_up(
    camera_to_world: Float[Tensor, "num_cams 4 4"],
) -> Float[Tensor, " 3"]:
    up_vectors = camera_to_world[:, :3, 1]
    return _normalize(up_vectors.mean(dim=0))


def _estimate_focus_point(
    camera_to_world: Float[Tensor, "num_cams 4 4"],
) -> tuple[Float[Tensor, " 3"], bool]:
    origins = camera_to_world[:, :3, 3]
    directions = torch.nn.functional.normalize(
        camera_to_world[:, :3, 2], dim=-1
    )
    identity = torch.eye(3, dtype=origins.dtype, device=origins.device)
    lhs = torch.zeros((3, 3), dtype=origins.dtype, device=origins.device)
    rhs = torch.zeros(3, dtype=origins.dtype, device=origins.device)
    for origin, direction in zip(origins, directions, strict=True):
        projector = identity - torch.outer(direction, direction)
        lhs = lhs + projector
        rhs = rhs + projector @ origin
    try:
        focus_point = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        return origins.mean(dim=0), False
    return focus_point, bool(torch.isfinite(focus_point).all())


def adjust_scene_record_horizon(
    scene_record: SceneRecord,
    spec: HorizonAdjustmentSpec,
) -> SceneRecord:
    """Rotate and translate scene geometry into a canonical up-aligned frame."""
    if not spec.enabled:
        return scene_record
    if not scene_record.camera_sensors:
        return scene_record

    all_cam_to_world = torch.cat(
        [sensor.camera.cam_to_world for sensor in scene_record.camera_sensors],
        dim=0,
    )
    estimated_up = _estimate_world_up(all_cam_to_world)
    focus_point, focus_success = _estimate_focus_point(all_cam_to_world)
    if not focus_success:
        focus_point = all_cam_to_world[:, :3, 3].mean(dim=0)
    rotation = _rotation_aligning_vectors(
        estimated_up.to(spec.target_up.device),
        spec.target_up,
    )
    translation = -(rotation @ focus_point)

    world_transform = torch.eye(
        4,
        dtype=all_cam_to_world.dtype,
        device=all_cam_to_world.device,
    )
    world_transform[:3, :3] = rotation
    world_transform[:3, 3] = translation

    point_cloud = scene_record.point_cloud
    if point_cloud is not None:
        point_cloud = point_cloud.transformed(rotation, translation)

    transformed_sensors = tuple(
        replace(
            sensor,
            camera=replace(
                sensor.camera,
                cam_to_world=world_transform @ sensor.camera.cam_to_world,
            ),
        )
        if isinstance(sensor, CameraSensorDataset)
        else sensor
        for sensor in scene_record.sensors
    )

    return replace(
        scene_record,
        sensors=transformed_sensors,
        point_cloud=point_cloud,
        world_up=spec.target_up,
        focus_point=translation,
    )


@register_source_pipe(kind="horizon_align", spec_cls=HorizonAlignPipeConfig)
def apply_horizon_align_pipe(
    scene_record: SceneRecord,
    pipe: HorizonAlignPipeConfig,
) -> SceneRecord:
    """Registered source pipe wrapper for horizon alignment."""
    return adjust_scene_record_horizon(scene_record, pipe.to_spec())
