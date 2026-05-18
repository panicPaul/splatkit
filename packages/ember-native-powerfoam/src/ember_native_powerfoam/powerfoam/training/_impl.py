"""Training helpers for PowerFoam scenes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from ember_core.core.contracts import CameraState, PowerFoamScene
from ember_core.core.families import scene_family_id
from ember_core.data.contracts import PointCloudState, SceneRecord
from ember_core.densification.contracts import (
    BaseDensificationMethod,
    DensificationContext,
    DensificationRenderRequirements,
    Schedule,
)
from ember_core.initialization import InitializedModel
from ember_core.training import (
    CallableSpec,
    OptimizationConfig,
    ParameterGroupConfig,
    ParameterTargetSpec,
)
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import KDTree
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ember_native_powerfoam.powerfoam.native.warp.geometry import morton_sort
from ember_native_powerfoam.powerfoam.native.warp.metrics import ssim
from ember_native_powerfoam.powerfoam.runtime import (
    PowerFoamTopology,
    build_powerfoam_topology,
    powerfoam_interpenetration,
    rebuild_powerfoam_topology,
)

_POWERFOAM_PARAMETER_FIELDS = PowerFoamScene.parameter_field_names


class PowerFoamOptimizationRecipe(BaseModel):
    """PowerFoam optimizer group defaults from the reference implementation."""

    model_config = ConfigDict(extra="forbid")

    optimizer: str = "torch.optim.Adam"
    points_lr_init: float = Field(default=1e-3, gt=0.0)
    points_lr_final: float = Field(default=1e-4, gt=0.0)
    density_lr_init: float = Field(default=1.0, gt=0.0)
    density_lr_final: float = Field(default=1.0, gt=0.0)
    radii_lr_init: float = Field(default=1e-4, gt=0.0)
    radii_lr_final: float = Field(default=1e-5, gt=0.0)
    quaternions_lr_init: float = Field(default=1e-1, gt=0.0)
    quaternions_lr_final: float = Field(default=1e-2, gt=0.0)
    texel_sites_lr_init: float = Field(default=1e-2, gt=0.0)
    texel_sites_lr_final: float = Field(default=1e-3, gt=0.0)
    texel_sv_axis_lr_init: float = Field(default=1e-2, gt=0.0)
    texel_sv_axis_lr_final: float = Field(default=1e-3, gt=0.0)
    texel_sv_rgb_lr_init: float = Field(default=2e-3, gt=0.0)
    texel_sv_rgb_lr_final: float = Field(default=2e-4, gt=0.0)
    texel_height_lr_init: float = Field(default=1e-2, gt=0.0)
    texel_height_lr_final: float = Field(default=1e-3, gt=0.0)
    density_warmup_steps: int = Field(default=1_000, ge=0)
    radii_warmup_steps: int = Field(default=1_000, ge=0)
    texel_height_warmup_steps: int = Field(default=2_000, ge=0)
    adam_eps: float = Field(default=1e-15, gt=0.0)


def powerfoam_cosine_decay_to(
    optimizer: Optimizer,
    *,
    final_lr: float,
    max_steps: int,
    warmup_steps: int = 0,
    last_epoch: int = -1,
) -> LambdaLR:
    """Build the reference PowerFoam cosine LR schedule."""
    if final_lr <= 0.0:
        raise ValueError("final_lr must be > 0.")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")
    initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if any(lr <= 0.0 for lr in initial_lrs):
        raise ValueError("All optimizer learning rates must be > 0.")

    def lr_lambda(step: int) -> float:
        bounded_step = min(max(step, 0), max_steps)
        if warmup_steps and bounded_step < warmup_steps:
            return bounded_step / float(warmup_steps)
        if bounded_step >= max_steps:
            return final_lr / initial_lrs[0]
        denom = max(max_steps - warmup_steps, 1)
        ratio = (bounded_step - warmup_steps) / denom
        value = final_lr + 0.5 * (initial_lrs[0] - final_lr) * (
            1.0 + math.cos(math.pi * ratio)
        )
        return value / initial_lrs[0]

    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


def _scene_parameter_target(name: str) -> ParameterTargetSpec:
    return ParameterTargetSpec(scope="scene", name=name)


def _cosine_spec(
    *,
    final_lr: float,
    max_steps: int,
    warmup_steps: int = 0,
) -> CallableSpec:
    return CallableSpec(
        target="ember_native_powerfoam.powerfoam_cosine_decay_to",
        kwargs={
            "final_lr": final_lr,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
        },
    )


def powerfoam_parameter_groups(
    recipe: PowerFoamOptimizationRecipe | dict[str, Any],
    *,
    max_steps: int,
) -> list[ParameterGroupConfig]:
    """Build PowerFoam optimizer groups."""
    recipe = PowerFoamOptimizationRecipe.model_validate(recipe)
    optimizer_kwargs = {"eps": recipe.adam_eps}

    def group(
        name: str,
        lr_init: float,
        lr_final: float,
        *,
        warmup_steps: int = 0,
    ) -> ParameterGroupConfig:
        return ParameterGroupConfig(
            target=_scene_parameter_target(name),
            optimizer=recipe.optimizer,
            lr=lr_init,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=_cosine_spec(
                final_lr=lr_final,
                max_steps=max_steps,
                warmup_steps=warmup_steps,
            ),
        )

    return [
        group("points", recipe.points_lr_init, recipe.points_lr_final),
        group(
            "density",
            recipe.density_lr_init,
            recipe.density_lr_final,
            warmup_steps=recipe.density_warmup_steps,
        ),
        group(
            "radii",
            recipe.radii_lr_init,
            recipe.radii_lr_final,
            warmup_steps=recipe.radii_warmup_steps,
        ),
        group(
            "quaternions",
            recipe.quaternions_lr_init,
            recipe.quaternions_lr_final,
        ),
        group(
            "texel_sites",
            recipe.texel_sites_lr_init,
            recipe.texel_sites_lr_final,
        ),
        group(
            "texel_sv_axis",
            recipe.texel_sv_axis_lr_init,
            recipe.texel_sv_axis_lr_final,
        ),
        group(
            "texel_sv_rgb",
            recipe.texel_sv_rgb_lr_init,
            recipe.texel_sv_rgb_lr_final,
        ),
        group(
            "texel_height",
            recipe.texel_height_lr_init,
            recipe.texel_height_lr_final,
            warmup_steps=recipe.texel_height_warmup_steps,
        ),
    ]


def powerfoam_optimization_config(
    recipe: PowerFoamOptimizationRecipe | dict[str, Any],
    *,
    max_steps: int,
) -> OptimizationConfig:
    """Build an OptimizationConfig from PowerFoam optimizer groups."""
    return OptimizationConfig(
        parameter_groups=powerfoam_parameter_groups(
            recipe,
            max_steps=max_steps,
        )
    )


def _require_point_cloud(scene_record: SceneRecord) -> PointCloudState:
    if scene_record.point_cloud is None:
        raise ValueError(
            "PowerFoam initialization requires a scene-record point cloud."
        )
    return scene_record.point_cloud


def _make_generator(
    device: torch.device,
    seed: int | None,
) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=str(device))
    generator.manual_seed(seed)
    return generator


def _camera_basis(
    camera: CameraState,
    camera_index: int,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    intrinsics = camera.get_intrinsics()[camera_index].to(
        device=device,
        dtype=torch.float32,
    )
    cam_to_world = camera.cam_to_world[camera_index].to(
        device=device,
        dtype=torch.float32,
    )
    width = int(camera.width[camera_index].item())
    height = int(camera.height[camera_index].item())
    right_extent = (float(width) - 1.0) * 0.5 / float(intrinsics[0, 0])
    up_extent = (float(height) - 1.0) * 0.5 / float(intrinsics[1, 1])
    right = cam_to_world[:3, 0] * right_extent
    up = -cam_to_world[:3, 1] * up_extent
    return cam_to_world[:3, 3], right, up


def _sample_sfm_points(
    point_cloud: PointCloudState,
    *,
    init_points: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> Float[Tensor, "num_points 3"]:
    points = point_cloud.points.to(device=device, dtype=torch.float32)
    point_count = int(points.shape[0])
    if point_count == 0:
        raise ValueError("PowerFoam initialization needs at least one SfM point.")
    sample_points = min(init_points, int(0.95 * point_count))
    permutation = torch.randperm(
        point_count,
        device=device,
        generator=generator,
    )
    point_indices = permutation[:sample_points]
    if init_points < 0.9 * point_count:
        return points[point_indices]

    cpu_points = points.detach().cpu().numpy()
    k = min(6, point_count)
    kdtree = KDTree(cpu_points)
    _dist, knn_i = kdtree.query(cpu_points, k=k)
    extra_count = init_points - sample_points
    if extra_count <= 0:
        return points[point_indices]
    weights = torch.rand(
        (extra_count, k),
        device=device,
        generator=generator,
    )
    weights = weights / weights.sum(dim=-1, keepdim=True)
    source_indices = torch.randint(
        0,
        point_count,
        (extra_count,),
        device=device,
        generator=generator,
    )
    knn_tensor = torch.tensor(knn_i, dtype=torch.long, device=device)
    extra_points = (weights[:, :, None] * points[knn_tensor[source_indices]]).sum(
        dim=1
    )
    return torch.cat([points[point_indices], extra_points], dim=0)


def _random_unbounded_points(
    camera: CameraState,
    *,
    init_points: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> Float[Tensor, "num_points 3"]:
    centers = camera.cam_to_world[:, :3, 3].to(device=device, dtype=torch.float32)
    camera_mean = centers.mean(dim=0)
    camera_std = centers.std(dim=0)
    return (
        torch.randn(
            (init_points, 3),
            device=device,
            generator=generator,
        )
        * camera_std
        * 3.0
        + camera_mean[None, :]
    )


def _initial_radii(
    points: Float[Tensor, "num_points 3"],
    camera: CameraState,
) -> Float[Tensor, " num_points"]:
    device = points.device
    num_points = int(points.shape[0])
    max_radii = 100 * torch.ones(num_points, device=device)
    for camera_index in range(camera.cam_to_world.shape[0]):
        eye, right, up = _camera_basis(camera, camera_index, device=device)
        view = eye[None, :] - points
        z_axis = torch.cross(right, up, dim=-1)
        view_x = (view * right[None, :]).sum(dim=-1) / right.norm()
        view_y = (view * up[None, :]).sum(dim=-1) / up.norm()
        view_z = (view * z_axis[None, :]).sum(dim=-1) / z_axis.norm()
        max_radii_c = 0.1 * view_z * up.norm()
        mask = view_z > 0
        mask &= (view_x / view_z).abs() < right.norm()
        mask &= (view_y / view_z).abs() < up.norm()
        max_radii[mask] = torch.minimum(max_radii_c[mask], max_radii[mask])

    cpu_points = points.detach().cpu().numpy()
    k = min(8, num_points)
    kdtree = KDTree(cpu_points)
    knn_d, _knn_i = kdtree.query(cpu_points, k=k)
    radii = torch.tensor(knn_d, dtype=torch.float32, device=device)
    if radii.ndim == 2:
        radii = radii.mean(dim=1)
    return torch.minimum(radii, max_radii)


def _empty_topology(
    num_points: int,
    *,
    device: torch.device,
) -> PowerFoamTopology:
    return PowerFoamTopology(
        adjacency=torch.zeros((0,), dtype=torch.int32, device=device),
        adjacency_offsets=torch.zeros(
            (num_points + 1,),
            dtype=torch.int32,
            device=device,
        ),
    )


def initialize_powerfoam_scene_from_scene_record(
    scene_record: SceneRecord,
    *,
    device: torch.device | str = torch.device("cuda"),
    init_type: Literal["sfm", "random_bounded", "random_unbounded"] = "sfm",
    init_points: int = 100_000,
    render_objective: Literal["volume", "surface"] = "volume",
    sv_dof: int = 8,
    num_texel_sites: int = 8,
    attr_dtype: torch.dtype = torch.float32,
    seed: int | None = 0,
) -> PowerFoamScene:
    """Build a PowerFoamScene from an Ember scene record."""
    target_device = torch.device(device)
    camera = scene_record.camera.to(target_device)
    generator = _make_generator(target_device, seed)
    if init_type == "sfm":
        init_points_tensor = _sample_sfm_points(
            _require_point_cloud(scene_record),
            init_points=init_points,
            device=target_device,
            generator=generator,
        )
    elif init_type == "random_bounded":
        init_points_tensor = torch.rand(
            (init_points, 3),
            dtype=torch.float32,
            device=target_device,
            generator=generator,
        )
    elif init_type == "random_unbounded":
        init_points_tensor = _random_unbounded_points(
            camera,
            init_points=init_points,
            device=target_device,
            generator=generator,
        )
    else:
        raise ValueError(f"Unknown PowerFoam init_type {init_type!r}.")

    points = init_points_tensor.to(dtype=attr_dtype)
    radii = _initial_radii(points.float(), camera).to(dtype=attr_dtype)
    quaternions = torch.randn(
        (points.shape[0], 4),
        dtype=attr_dtype,
        device=target_device,
        generator=generator,
    )
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    density = torch.ones(
        points.shape[0],
        dtype=attr_dtype,
        device=target_device,
    ) * 1e-1
    texel_sites = torch.randn(
        (points.shape[0], num_texel_sites, 2),
        dtype=attr_dtype,
        device=target_device,
        generator=generator,
    ) * 0.1
    texel_sv_axis = torch.randn(
        (points.shape[0], num_texel_sites, 3 * sv_dof),
        dtype=attr_dtype,
        device=target_device,
        generator=generator,
    ) * 2.0
    texel_sv_rgb = torch.zeros(
        (points.shape[0], num_texel_sites, 3 * sv_dof),
        dtype=attr_dtype,
        device=target_device,
    )
    texel_height = torch.zeros(
        (points.shape[0], num_texel_sites),
        dtype=attr_dtype,
        device=target_device,
    )
    topology = (
        build_powerfoam_topology(points.float(), F.softplus(radii.float(), beta=100))
        if target_device.type == "cuda"
        else _empty_topology(int(points.shape[0]), device=target_device)
    )
    return PowerFoamScene(
        points=points.requires_grad_(True),
        radii=radii.requires_grad_(True),
        quaternions=quaternions.requires_grad_(True),
        density=density.requires_grad_(True),
        texel_sites=texel_sites.requires_grad_(True),
        texel_sv_axis=texel_sv_axis.requires_grad_(True),
        texel_sv_rgb=texel_sv_rgb.requires_grad_(True),
        texel_height=texel_height.requires_grad_(True),
        adjacency=topology.adjacency,
        adjacency_offsets=topology.adjacency_offsets,
        sv_dof=sv_dof,
        num_texel_sites=num_texel_sites,
        render_objective=render_objective,
        attr_dtype="half" if attr_dtype == torch.float16 else "float",
    )


def initialize_powerfoam_model_from_scene_record(
    scene_record: SceneRecord,
    *,
    modules: dict[str, nn.Module] | None = None,
    parameters: dict[str, nn.Parameter] | None = None,
    buffers: dict[str, Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> InitializedModel:
    """Build a training payload containing a PowerFoamScene."""
    return InitializedModel(
        scene=initialize_powerfoam_scene_from_scene_record(
            scene_record,
            **kwargs,
        ),
        modules=modules or {},
        parameters=parameters or {},
        buffers=buffers or {},
        metadata=metadata or {},
    )


def powerfoam_training_loss(
    state: Any,
    batch: Any,
    render_output: Any,
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, Tensor | float]:
    """Compute the PowerFoam RGB and SSIM training loss components."""
    weights = dict(weights or {})
    rgb_weight = float(weights.get("rgb", 1.0))
    ssim_weight = float(weights.get("ssim", 0.2))
    normal_weight = float(weights.get("normal", 0.0))
    contribution_weight = float(weights.get("contribution", 0.0))
    interpenetration_weight = float(weights.get("interpenetration", 0.0))
    predicted = render_output.render
    target = batch.images.to(device=predicted.device, dtype=predicted.dtype)
    rgb_loss = F.mse_loss(predicted, target, reduction="none").sum(dim=-1).mean()
    loss = rgb_weight * rgb_loss
    metrics: dict[str, Tensor | float] = {
        "loss": loss,
        "rgb_loss": float(rgb_loss.detach().item()),
    }
    if ssim_weight:
        ssim_loss = 1.0 - ssim(
            predicted.permute(0, 3, 1, 2).contiguous(),
            target.permute(0, 3, 1, 2).contiguous(),
        )
        loss = loss + ssim_weight * ssim_loss
        metrics["loss"] = loss
        metrics["ssim_loss"] = float(ssim_loss.detach().item())
    if normal_weight and render_output.normal_error is not None:
        normal_loss = render_output.normal_error.mean()
        loss = loss + normal_weight * normal_loss
        metrics["loss"] = loss
        metrics["normal_loss"] = float(normal_loss.detach().item())
    if contribution_weight and render_output.contrib is not None:
        contribution_loss = render_output.contrib.sum()
        loss = loss + contribution_weight * contribution_loss
        metrics["loss"] = loss
        metrics["contribution_loss"] = float(
            contribution_loss.detach().item()
        )
    if interpenetration_weight:
        scene = state.model.scene
        if isinstance(scene, PowerFoamScene):
            interpenetration_loss = powerfoam_interpenetration(scene).sum()
            loss = loss + interpenetration_weight * interpenetration_loss
            metrics["loss"] = loss
            metrics["interpenetration_loss"] = float(
                interpenetration_loss.detach().item()
            )
    return metrics


def powerfoam_training_backend_options(
    state: Any | None = None,
    batch: Any | None = None,
) -> dict[str, Tensor | bool]:
    """Pass RGB supervision into the renderer for native point-error stats."""
    del state
    if batch is None or not hasattr(batch, "images"):
        return {}
    return {"ray_gt": batch.images, "return_point_err": True}


def powerfoam_target_points(
    step: int,
    *,
    initial_num_points: int,
    final_points: int,
    densify_from: int,
    densify_until: int,
) -> int:
    """Return the upstream PowerFoam target point count for a train step."""
    if initial_num_points < 1:
        raise ValueError("initial_num_points must be >= 1.")
    if final_points < 1:
        raise ValueError("final_points must be >= 1.")
    if densify_until <= densify_from + 1:
        raise ValueError("densify_until must be greater than densify_from + 1.")
    if step < densify_from:
        return initial_num_points
    if step >= densify_until:
        return final_points
    growth = (final_points / initial_num_points) ** (
        1.0 / (densify_until - densify_from - 1)
    )
    return int(initial_num_points * (growth ** (step - densify_from)))


def _transform_powerfoam_optimizer_state(
    indices: Tensor,
    old_rows: int,
) -> Any:
    def transform(name: str, state: Tensor) -> Tensor:
        del name
        if state.ndim == 0 or int(state.shape[0]) != old_rows:
            return state
        return state[indices].contiguous()

    return transform


def _replace_powerfoam_fields_with_optimizer_state(
    scene: PowerFoamScene,
    optimizers: Sequence[Any],
    updates: dict[str, Tensor],
    *,
    indices: Tensor,
    old_rows: int,
) -> None:
    installed = scene.replace_fields_(**updates)
    transform = _transform_powerfoam_optimizer_state(indices, old_rows)
    for name, parameter in installed.items():
        for binding in optimizers:
            matches_target = getattr(binding, "matches_target", None)
            replace_parameter = getattr(binding, "replace_parameter", None)
            if not callable(matches_target) or not callable(
                replace_parameter
            ):
                continue
            if matches_target("scene", name):
                replace_parameter(parameter, transform)


def _mean_point_stat(value: Tensor | None, name: str) -> Tensor:
    if value is None:
        raise ValueError(
            "PowerFoamResampling requires render output "
            f"{name!r}; ensure powerfoam_training_backend_options is active."
        )
    detached = value.detach()
    if detached.ndim == 1:
        return detached.float()
    if detached.ndim == 2:
        return detached.float().mean(dim=0)
    raise ValueError(
        f"PowerFoam render output {name!r} must have shape [N] or [B, N], "
        f"got {tuple(detached.shape)}."
    )


def _visible_point_mask(value: Tensor | None) -> Tensor:
    if value is None:
        raise ValueError(
            "PowerFoamResampling requires render output "
            "'prim_visible_mask'; ensure the PowerFoam renderer is active."
        )
    detached = value.detach()
    if detached.ndim == 1:
        return detached.float()
    if detached.ndim == 2:
        return detached.float().amax(dim=0)
    raise ValueError(
        "PowerFoam render output 'prim_visible_mask' must have shape "
        f"[N] or [B, N], got {tuple(detached.shape)}."
    )


def _safe_resample_probabilities(prob: Tensor) -> Tensor:
    if prob.numel() == 0:
        return prob
    if torch.isfinite(prob).all() and float(prob.sum().item()) > 0.0:
        return prob
    return torch.ones_like(prob)


def _sample_powerfoam_indices(
    *,
    contrib_ema: Tensor,
    point_error_ema: Tensor,
    target_num_points: int,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, int]:
    device = contrib_ema.device
    num_points = int(contrib_ema.shape[0])
    contrib_quantiles = torch.quantile(
        contrib_ema,
        torch.tensor([0.1, 0.99], dtype=contrib_ema.dtype, device=device),
        dim=0,
    )
    contribution_floor = torch.tensor(
        1.0 / (target_num_points * 25),
        dtype=contrib_ema.dtype,
        device=device,
    )
    contrib_threshold = torch.minimum(contribution_floor, contrib_quantiles[0])
    valid_indices = torch.nonzero(
        contrib_ema > contrib_threshold,
        as_tuple=False,
    )[:, 0]
    if int(valid_indices.shape[0]) == 0:
        valid_indices = torch.arange(num_points, device=device)

    num_samples = target_num_points - int(valid_indices.shape[0])
    if num_samples <= 0:
        return None, None, None, 0

    point_error_quantile = torch.quantile(point_error_ema, 0.99, dim=0)
    prob = _safe_resample_probabilities(
        point_error_ema[valid_indices].clamp(max=point_error_quantile)
    )
    replacement = num_samples > int(valid_indices.shape[0])
    resample_valid_indices = torch.multinomial(
        prob,
        num_samples,
        replacement=replacement,
    )

    duplicate_count = torch.ones_like(valid_indices)
    duplicate_count.index_add_(
        0,
        resample_valid_indices,
        torch.ones_like(resample_valid_indices),
    )
    duplicate_mask = duplicate_count > 1

    resample_cell_indices = valid_indices[resample_valid_indices]
    resample_duplicate_mask = duplicate_mask[resample_valid_indices]
    resample_duplicate_count = duplicate_count[resample_valid_indices]

    new_indices = torch.cat([valid_indices, resample_cell_indices], dim=0)
    new_duplicate_mask = torch.cat(
        [duplicate_mask, resample_duplicate_mask],
        dim=0,
    ).bool()
    new_duplicate_count = torch.cat(
        [duplicate_count, resample_duplicate_count],
        dim=0,
    )
    return new_indices, new_duplicate_mask, new_duplicate_count, num_samples


def _resampled_powerfoam_updates(
    scene: PowerFoamScene,
    indices: Tensor,
    duplicate_mask: Tensor,
) -> dict[str, Tensor]:
    updates = {
        name: getattr(scene, name).detach()[indices].contiguous()
        for name in _POWERFOAM_PARAMETER_FIELDS
    }
    if bool(duplicate_mask.any().item()):
        normals = _powerfoam_normals_from_quaternions(
            updates["quaternions"]
        )
        cell_radius = F.softplus(updates["radii"].float(), beta=100.0)
        direction = torch.randn_like(normals)
        direction = direction - (direction * normals).sum(
            dim=-1,
            keepdim=True,
        )
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(
            1e-12
        )
        perturbation = 0.05 * cell_radius[:, None] * direction
        points = updates["points"].clone()
        points[duplicate_mask] += perturbation.to(dtype=points.dtype)[
            duplicate_mask
        ]
        updates["points"] = points
    for name, value in updates.items():
        value.requires_grad_(getattr(scene, name).requires_grad)
    return updates


def _powerfoam_normals_from_quaternions(
    quaternions: Tensor,
) -> Float[Tensor, "num_points 3"]:
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]
    normals = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
        ],
        dim=-1,
    )
    return normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass
class PowerFoamAdjacencyRefresh(BaseDensificationMethod):
    """Scheduled PowerFoam AABB and Cech-adjacency refresh stage."""

    refine_every: int = 1
    start_iter: int = 0
    stop_iter: int = -1
    expected_scene_families: tuple[str, ...] = ("foam",)

    def __post_init__(self) -> None:
        self.schedule = Schedule(
            start_iteration=self.start_iter,
            end_iteration=self.stop_iter,
            frequency=self.refine_every,
        )

    def bind(
        self,
        state: Any,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        """Validate that this refresh stage is bound to a foam scene."""
        del optimizers, family_ops
        scene = state.model.scene
        family = scene_family_id(scene.scene_family)
        expected = tuple(
            scene_family_id(item) for item in self.expected_scene_families
        )
        if family not in expected:
            raise TypeError(
                "PowerFoamAdjacencyRefresh expects foam scenes, got "
                f"{scene.scene_family!r}."
            )

    def post_optimizer_step(self, context: DensificationContext) -> None:
        """Refresh PowerFoam adjacency after scheduled optimizer steps."""
        step = context.step + 1
        if not self.schedule.includes(step):
            return
        scene = context.state.model.scene
        if not isinstance(scene, PowerFoamScene):
            return
        topology = rebuild_powerfoam_topology(scene)
        scene.replace_fields_(
            adjacency=topology.adjacency,
            adjacency_offsets=topology.adjacency_offsets,
        )


@dataclass
class PowerFoamResampling(BaseDensificationMethod):
    """PowerFoam contribution/error EMA resampling stage."""

    max_steps: int
    resample_every: int = 100
    resample_offset: int = 99
    densify_from: int = 1_000
    densify_until: int = 24_000
    final_points: int = 1_200_000
    stop_fraction: float = 0.95
    adjacency_max_interval: int = 20
    stats_epsilon: float = 1e-5
    sort_after_resample: bool = True
    expected_scene_families: tuple[str, ...] = ("foam",)

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if self.resample_every < 1:
            raise ValueError("resample_every must be >= 1.")
        if not 0 <= self.resample_offset < self.resample_every:
            raise ValueError(
                "resample_offset must satisfy "
                "0 <= resample_offset < resample_every."
            )
        if self.densify_until <= self.densify_from + 1:
            raise ValueError(
                "densify_until must be greater than densify_from + 1."
            )
        if self.final_points < 1:
            raise ValueError("final_points must be >= 1.")
        if not 0.0 <= self.stop_fraction <= 1.0:
            raise ValueError("stop_fraction must be between 0 and 1.")
        if self.adjacency_max_interval < 1:
            raise ValueError("adjacency_max_interval must be >= 1.")
        if self.stats_epsilon <= 0.0:
            raise ValueError("stats_epsilon must be > 0.")
        self._initial_num_points: int | None = None
        self._topology_interval = 1
        self._iters_since_topology = 0
        self._last_target_points = 0
        self._last_num_resampled = 0
        self._last_topology_refreshed = False

    def get_render_requirements(
        self,
        state: object,
    ) -> DensificationRenderRequirements:
        """Request point-error output for PowerFoam EMA statistics."""
        del state
        return DensificationRenderRequirements(
            backend_options={"return_point_err": True}
        )

    def bind(
        self,
        state: Any,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        """Validate and initialize the PowerFoam resampling state."""
        del optimizers, family_ops
        scene = state.model.scene
        family = scene_family_id(scene.scene_family)
        expected = tuple(
            scene_family_id(item) for item in self.expected_scene_families
        )
        if family not in expected or not isinstance(scene, PowerFoamScene):
            raise TypeError(
                "PowerFoamResampling expects PowerFoam scenes, got "
                f"{scene.scene_family!r}."
            )
        self._initial_num_points = int(scene.points.shape[0])
        self._last_target_points = self._initial_num_points

    def post_optimizer_step(self, context: DensificationContext) -> None:
        """Update stats, resample on schedule, and refresh topology."""
        scene = context.state.model.scene
        if not isinstance(scene, PowerFoamScene):
            return
        if self._initial_num_points is None:
            self._initial_num_points = int(scene.points.shape[0])

        self._last_num_resampled = 0
        self._last_topology_refreshed = False
        self._update_stats(context.render_output)
        self._last_target_points = self._target_points(context.step, scene)
        if self._should_resample(context.step):
            self._last_num_resampled = self._resample(
                scene,
                context.optimizers,
                self._last_target_points,
            )
            self._iters_since_topology = 0
            self._last_topology_refreshed = True
            return

        if self._topology_refresh_due():
            self._refresh_topology(scene)
            self._last_topology_refreshed = True

    def after_step(
        self,
        context: DensificationContext,
        metrics: dict[str, float],
    ) -> None:
        """Expose compact PowerFoam densification diagnostics."""
        scene = context.state.model.scene
        if not isinstance(scene, PowerFoamScene):
            return
        metrics["powerfoam/num_points"] = float(scene.points.shape[0])
        metrics["powerfoam/target_points"] = float(self._last_target_points)
        metrics["powerfoam/num_resampled"] = float(self._last_num_resampled)
        metrics["powerfoam/topology_refreshed"] = float(
            self._last_topology_refreshed
        )

    def _update_stats(self, render_output: Any) -> None:
        contrib = _mean_point_stat(
            getattr(render_output, "contrib", None),
            "contrib",
        )
        point_error = _mean_point_stat(
            getattr(render_output, "point_error", None),
            "point_error",
        )
        visible = _visible_point_mask(
            getattr(render_output, "prim_visible_mask", None)
        ).to(device=contrib.device)
        if contrib.shape != point_error.shape or contrib.shape != visible.shape:
            raise ValueError(
                "PowerFoam resampling stats must have matching point shapes: "
                f"contrib={tuple(contrib.shape)}, "
                f"point_error={tuple(point_error.shape)}, "
                f"visible={tuple(visible.shape)}."
            )
        if getattr(self, "contrib_ema", None) is None:
            self.contrib_ema = torch.ones_like(contrib) * self.stats_epsilon
        else:
            alpha = 0.99 * visible + (1.0 - visible)
            self.contrib_ema = (
                alpha * self.contrib_ema.to(contrib.device)
                + (1.0 - alpha) * contrib
            )
        if getattr(self, "point_error_ema", None) is None:
            self.point_error_ema = (
                torch.ones_like(point_error) * self.stats_epsilon
            )
        else:
            self.point_error_ema = (
                0.99 * self.point_error_ema.to(point_error.device)
                + 0.01 * point_error
            )

    def _target_points(self, step: int, scene: PowerFoamScene) -> int:
        if (
            self._initial_num_points is not None
            and self.densify_from <= step < self.densify_until
        ):
            return powerfoam_target_points(
                step,
                initial_num_points=self._initial_num_points,
                final_points=self.final_points,
                densify_from=self.densify_from,
                densify_until=self.densify_until,
            )
        return int(scene.points.shape[0])

    def _should_resample(self, step: int) -> bool:
        if step >= int(self.stop_fraction * self.max_steps):
            return False
        return step % self.resample_every == self.resample_offset

    def _topology_refresh_due(self) -> bool:
        self._iters_since_topology += 1
        if self._iters_since_topology % self._topology_interval != 0:
            return False
        self._iters_since_topology = 0
        self._topology_interval = min(
            self._topology_interval + 1,
            self.adjacency_max_interval,
        )
        return True

    def _resample(
        self,
        scene: PowerFoamScene,
        optimizers: Sequence[Any],
        target_num_points: int,
    ) -> int:
        contrib_ema = getattr(self, "contrib_ema", None)
        point_error_ema = getattr(self, "point_error_ema", None)
        if contrib_ema is None or point_error_ema is None:
            self._refresh_topology(scene)
            return 0

        old_rows = int(scene.points.shape[0])
        target_num_points = max(int(target_num_points), 1)
        indices, duplicate_mask, duplicate_count, num_resampled = (
            _sample_powerfoam_indices(
                contrib_ema=contrib_ema,
                point_error_ema=point_error_ema,
                target_num_points=target_num_points,
            )
        )
        if indices is None or duplicate_mask is None or duplicate_count is None:
            if self.sort_after_resample:
                permutation = morton_sort(scene.points.detach()).to(
                    device=scene.points.device,
                    dtype=torch.long,
                )
                self._sort_scene(scene, optimizers, permutation)
            else:
                self._refresh_topology(scene)
            return 0

        updates = _resampled_powerfoam_updates(scene, indices, duplicate_mask)
        scale = duplicate_count.to(device=contrib_ema.device, dtype=contrib_ema.dtype)
        new_contrib_ema = contrib_ema[indices] / scale
        new_point_error_ema = point_error_ema[indices] / scale
        optimizer_indices = indices
        if self.sort_after_resample:
            permutation = morton_sort(updates["points"].detach()).to(
                device=indices.device,
                dtype=torch.long,
            )
            sorted_updates = {}
            for name, value in updates.items():
                sorted_value = value[permutation].contiguous()
                sorted_updates[name] = sorted_value.requires_grad_(
                    getattr(scene, name).requires_grad
                )
            updates = sorted_updates
            optimizer_indices = optimizer_indices[permutation]
            new_contrib_ema = new_contrib_ema[permutation].contiguous()
            new_point_error_ema = new_point_error_ema[permutation].contiguous()

        topology = build_powerfoam_topology(
            updates["points"],
            F.softplus(updates["radii"].float(), beta=100.0),
        )
        updates["adjacency"] = topology.adjacency
        updates["adjacency_offsets"] = topology.adjacency_offsets
        _replace_powerfoam_fields_with_optimizer_state(
            scene,
            optimizers,
            updates,
            indices=optimizer_indices,
            old_rows=old_rows,
        )
        self.contrib_ema = new_contrib_ema
        self.point_error_ema = new_point_error_ema
        return int(num_resampled)

    def _sort_scene(
        self,
        scene: PowerFoamScene,
        optimizers: Sequence[Any],
        permutation: Tensor,
    ) -> None:
        old_rows = int(scene.points.shape[0])
        duplicate_mask = torch.zeros(
            old_rows,
            dtype=torch.bool,
            device=scene.points.device,
        )
        updates = _resampled_powerfoam_updates(
            scene,
            permutation,
            duplicate_mask,
        )
        topology = build_powerfoam_topology(
            updates["points"],
            F.softplus(updates["radii"].float(), beta=100.0),
        )
        updates["adjacency"] = topology.adjacency
        updates["adjacency_offsets"] = topology.adjacency_offsets
        _replace_powerfoam_fields_with_optimizer_state(
            scene,
            optimizers,
            updates,
            indices=permutation,
            old_rows=old_rows,
        )
        if getattr(self, "contrib_ema", None) is not None:
            self.contrib_ema = self.contrib_ema[permutation].contiguous()
        if getattr(self, "point_error_ema", None) is not None:
            self.point_error_ema = self.point_error_ema[
                permutation
            ].contiguous()

    def _refresh_topology(self, scene: PowerFoamScene) -> None:
        topology = rebuild_powerfoam_topology(scene)
        scene.replace_fields_(
            adjacency=topology.adjacency,
            adjacency_offsets=topology.adjacency_offsets,
        )


__all__ = [
    "PowerFoamAdjacencyRefresh",
    "PowerFoamOptimizationRecipe",
    "PowerFoamResampling",
    "initialize_powerfoam_model_from_scene_record",
    "initialize_powerfoam_scene_from_scene_record",
    "powerfoam_cosine_decay_to",
    "powerfoam_optimization_config",
    "powerfoam_parameter_groups",
    "powerfoam_target_points",
    "powerfoam_training_backend_options",
    "powerfoam_training_loss",
]
