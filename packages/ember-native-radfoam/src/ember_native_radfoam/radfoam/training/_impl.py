"""Training helpers for RADFOAM scenes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from ember_core.core.contracts import RadFoamScene
from ember_core.core.families import scene_family_id
from ember_core.data.contracts import PointCloudState, SceneRecord
from ember_core.densification.contracts import (
    BaseDensificationMethod,
    DensificationContext,
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
from torch import Tensor, nn

from ember_native_radfoam.radfoam.runtime import (
    MIN_RADFOAM_POINTS,
    build_radfoam_topology,
    farthest_neighbor,
)


class RadFoamOptimizationRecipe(BaseModel):
    """RADFOAM optimizer group defaults."""

    model_config = ConfigDict(extra="forbid")

    optimizer: str = "torch.optim.Adam"
    points_lr_init: float = Field(default=3e-2, gt=0.0)
    points_lr_final: float = Field(default=3e-4, gt=0.0)
    points_lr_max_steps: int | None = Field(default=None, ge=1)
    density_lr_init: float = Field(default=1e-2, gt=0.0)
    density_lr_final: float = Field(default=1e-3, gt=0.0)
    attributes_lr_init: float = Field(default=1e-2, gt=0.0)
    attributes_lr_final: float = Field(default=1e-3, gt=0.0)
    sh_factor: float = Field(default=0.2, gt=0.0)
    adam_eps: float = Field(default=1e-15, gt=0.0)


def _scene_parameter_target(name: str) -> ParameterTargetSpec:
    return ParameterTargetSpec(scope="scene", name=name)


def radfoam_parameter_groups(
    recipe: RadFoamOptimizationRecipe | dict[str, Any],
    *,
    max_steps: int,
) -> list[ParameterGroupConfig]:
    """Build RADFOAM optimizer groups."""
    recipe = RadFoamOptimizationRecipe.model_validate(recipe)
    points_max_steps = (
        recipe.points_lr_max_steps
        if recipe.points_lr_max_steps is not None
        else max_steps
    )
    optimizer_kwargs = {"eps": recipe.adam_eps}
    return [
        ParameterGroupConfig(
            target=_scene_parameter_target("primal_points"),
            optimizer=recipe.optimizer,
            lr=recipe.points_lr_init,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=CallableSpec(
                target="ember_core.training.exponential_decay_to",
                kwargs={
                    "final_lr": recipe.points_lr_final,
                    "max_steps": points_max_steps,
                },
            ),
        ),
        ParameterGroupConfig(
            target=_scene_parameter_target("density"),
            optimizer=recipe.optimizer,
            lr=recipe.density_lr_init,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=CallableSpec(
                target="ember_core.training.exponential_decay_to",
                kwargs={
                    "final_lr": recipe.density_lr_final,
                    "max_steps": max_steps,
                },
            ),
        ),
        ParameterGroupConfig(
            target=_scene_parameter_target("att_dc"),
            optimizer=recipe.optimizer,
            lr=recipe.attributes_lr_init,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=CallableSpec(
                target="ember_core.training.exponential_decay_to",
                kwargs={
                    "final_lr": recipe.attributes_lr_final,
                    "max_steps": max_steps,
                },
            ),
        ),
        ParameterGroupConfig(
            target=_scene_parameter_target("att_sh"),
            optimizer=recipe.optimizer,
            lr=recipe.sh_factor * recipe.attributes_lr_init,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=CallableSpec(
                target="ember_core.training.exponential_decay_to",
                kwargs={
                    "final_lr": recipe.sh_factor
                    * recipe.attributes_lr_final,
                    "max_steps": max_steps,
                },
            ),
        ),
    ]


def radfoam_optimization_config(
    recipe: RadFoamOptimizationRecipe | dict[str, Any],
    *,
    max_steps: int,
) -> OptimizationConfig:
    """Build an OptimizationConfig from RADFOAM optimizer groups."""
    return OptimizationConfig(
        parameter_groups=radfoam_parameter_groups(recipe, max_steps=max_steps)
    )


def _require_point_cloud(scene_record: SceneRecord) -> PointCloudState:
    if scene_record.point_cloud is None:
        raise ValueError(
            "RADFOAM initialization requires a scene-record point cloud."
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


def _initialize_points_from_point_cloud(
    point_cloud: PointCloudState,
    *,
    device: torch.device,
    dtype: torch.dtype,
    init_points: int,
    random_points: int,
    point_cloud_sample_ratio: float,
    jitter_std: float,
    random_point_scale: float,
    generator: torch.Generator | None,
) -> tuple[
    Float[Tensor, "num_points 3"],
    Float[Tensor, "num_points 1"],
]:
    points = point_cloud.points.to(device=device, dtype=torch.float32)
    point_count = int(points.shape[0])
    sampled_count = min(
        int(point_cloud_sample_ratio * point_count),
        max(init_points - random_points, 0),
    )
    if sampled_count <= 0:
        raise ValueError("RADFOAM initialization needs at least one SfM point.")
    indices = torch.randint(
        0,
        point_count,
        (sampled_count,),
        device=device,
        generator=generator,
    )
    sampled_points = points[indices]
    if jitter_std > 0.0:
        sampled_points = sampled_points + torch.randn(
            sampled_points.shape,
            dtype=sampled_points.dtype,
            device=sampled_points.device,
            generator=generator,
        ) * jitter_std
    random = torch.randn(
        (random_points, 3),
        device=device,
        generator=generator,
    ) * random_point_scale
    primal_points = torch.cat([sampled_points, random], dim=0)
    sampled_density = torch.rand(
        (sampled_count, 1),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    random_density = -0.5 * torch.ones(
        (random_points, 1),
        dtype=dtype,
        device=device,
    )
    return primal_points, torch.cat([sampled_density, random_density], dim=0)


def initialize_radfoam_scene_from_scene_record(
    scene_record: SceneRecord,
    *,
    device: torch.device | str = torch.device("cuda"),
    sh_degree: int = 3,
    init_points: int = 131_072,
    random_points: int = 5_000,
    point_cloud_sample_ratio: float = 0.9,
    activation_scale: float = 1.0,
    attr_dtype: torch.dtype = torch.float32,
    jitter_std: float = 1e-2,
    random_point_scale: float = 10.0,
    seed: int | None = 0,
) -> RadFoamScene:
    """Build a RadFoamScene from an SfM point cloud."""
    target_device = torch.device(device)
    point_cloud = _require_point_cloud(scene_record)
    generator = _make_generator(target_device, seed)
    primal_points, density = _initialize_points_from_point_cloud(
        point_cloud,
        device=target_device,
        dtype=attr_dtype,
        init_points=init_points,
        random_points=random_points,
        point_cloud_sample_ratio=point_cloud_sample_ratio,
        jitter_std=jitter_std,
        random_point_scale=random_point_scale,
        generator=generator,
    )
    if int(primal_points.shape[0]) < MIN_RADFOAM_POINTS:
        raise ValueError(
            "RADFOAM initialization requires at least "
            f"{MIN_RADFOAM_POINTS} points."
        )
    topology = build_radfoam_topology(primal_points)
    permutation = topology.permutation
    primal_points = primal_points[permutation]
    density = density[permutation]
    num_points = int(primal_points.shape[0])
    att_dc = torch.zeros(
        (num_points, 3),
        dtype=attr_dtype,
        device=target_device,
    )
    att_sh = torch.zeros(
        (num_points, 3 * ((1 + sh_degree) ** 2 - 1)),
        dtype=attr_dtype,
        device=target_device,
    )
    return RadFoamScene(
        primal_points=primal_points.requires_grad_(True),
        density=density.requires_grad_(True),
        att_dc=att_dc.requires_grad_(True),
        att_sh=att_sh.requires_grad_(True),
        point_adjacency=topology.point_adjacency,
        point_adjacency_offsets=topology.point_adjacency_offsets,
        sh_degree=sh_degree,
        activation_scale=activation_scale,
    )


def initialize_radfoam_model_from_scene_record(
    scene_record: SceneRecord,
    *,
    modules: dict[str, nn.Module] | None = None,
    parameters: dict[str, nn.Parameter] | None = None,
    buffers: dict[str, Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> InitializedModel:
    """Build a training payload containing a RadFoamScene."""
    return InitializedModel(
        scene=initialize_radfoam_scene_from_scene_record(
            scene_record,
            **kwargs,
        ),
        modules=modules or {},
        parameters=parameters or {},
        buffers=buffers or {},
        metadata=metadata or {},
    )


def radfoam_rgb_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    loss_type: str = "l1",
) -> Tensor:
    """Compute a photometric RGB loss for RADFOAM training."""
    if loss_type == "l1":
        return F.l1_loss(predicted, target)
    if loss_type == "mse":
        return F.mse_loss(predicted, target)
    raise ValueError(f"Unknown RADFOAM loss type {loss_type!r}.")


def _transform_optimizer_state(
    permutation: Tensor,
    old_rows: int,
) -> Any:
    def transform(name: str, state: Tensor) -> Tensor:
        del name
        if state.ndim == 0 or int(state.shape[0]) != old_rows:
            return state
        return state[permutation].contiguous()

    return transform


def _replace_scene_fields_with_optimizer_state(
    scene: RadFoamScene,
    optimizers: Sequence[Any],
    updates: dict[str, Tensor],
    *,
    permutation: Tensor,
    old_rows: int,
) -> None:
    installed = scene.replace_fields_(**updates)
    transform = _transform_optimizer_state(permutation, old_rows)
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


@dataclass
class RadFoamTopologyRefresh(BaseDensificationMethod):
    """Scheduled RADFOAM triangulation and AABB-tree refresh stage."""

    refine_every: int = 100
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
        """Validate that this refresh stage is bound to a RADFOAM scene."""
        del optimizers, family_ops
        scene = state.model.scene
        family = scene_family_id(scene.scene_family)
        expected = tuple(
            scene_family_id(item) for item in self.expected_scene_families
        )
        if family not in expected:
            raise TypeError(
                "RadFoamTopologyRefresh expects RADFOAM scenes, got "
                f"{scene.scene_family!r}."
            )

    def post_optimizer_step(self, context: DensificationContext) -> None:
        """Refresh RADFOAM topology after scheduled optimizer steps."""
        step = context.step + 1
        if not self.schedule.includes(step):
            return
        scene = context.state.model.scene
        if not isinstance(scene, RadFoamScene):
            return
        old_rows = int(scene.primal_points.shape[0])
        topology = build_radfoam_topology(scene.primal_points.detach())
        permutation = topology.permutation.to(scene.primal_points.device)
        updates = {
            "primal_points": scene.primal_points.detach()[permutation],
            "density": scene.density.detach()[permutation],
            "att_dc": scene.att_dc.detach()[permutation],
            "att_sh": scene.att_sh.detach()[permutation],
            "point_adjacency": topology.point_adjacency,
            "point_adjacency_offsets": topology.point_adjacency_offsets,
        }
        _replace_scene_fields_with_optimizer_state(
            scene,
            context.optimizers,
            updates,
            permutation=permutation,
            old_rows=old_rows,
        )


def radfoam_farthest_neighbor_radius(scene: RadFoamScene) -> Tensor:
    """Return each point's farthest-neighbor cell radius."""
    _neighbor, radius = farthest_neighbor(
        scene.primal_points,
        scene.point_adjacency,
        scene.point_adjacency_offsets,
    )
    return radius


__all__ = [
    "RadFoamOptimizationRecipe",
    "RadFoamTopologyRefresh",
    "initialize_radfoam_model_from_scene_record",
    "initialize_radfoam_scene_from_scene_record",
    "radfoam_farthest_neighbor_radius",
    "radfoam_optimization_config",
    "radfoam_parameter_groups",
    "radfoam_rgb_loss",
]
