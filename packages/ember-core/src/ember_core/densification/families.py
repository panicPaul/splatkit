"""Family-specific densification operations."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ember_core.core.contracts import GaussianScene, SparseVoxelScene
from ember_core.core.sparse_voxel import (
    _SUBTREE_SHIFTS,
    svraster_build_grid_points_link,
)


@dataclass(frozen=True)
class FieldBehavior:
    """Per-field behavior override for topology edits."""

    clone: Callable[[str, Tensor, Tensor, dict[str, Any]], Tensor] | None = None
    split: Callable[[str, Tensor, Tensor, dict[str, Any]], Tensor] | None = None


def copy_field() -> FieldBehavior:
    """Copy parent values for clone and split operations."""
    return FieldBehavior()


def zero_field() -> FieldBehavior:
    """Zero newly created children or clones."""

    def clone_fn(
        name: str, tensor: Tensor, mask: Tensor, context: dict[str, Any]
    ) -> Tensor:
        del name, context
        return torch.cat([tensor, torch.zeros_like(tensor[mask])], dim=0)

    def split_fn(
        name: str,
        tensor: Tensor,
        mask: Tensor,
        context: dict[str, Any],
    ) -> Tensor:
        del name
        repeats = int(context["num_children"])
        zeros = torch.zeros(
            (int(mask.sum()) * repeats, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor[~mask], zeros], dim=0)

    return FieldBehavior(clone=clone_fn, split=split_fn)


def custom_field(
    *,
    clone: Callable[[str, Tensor, Tensor, dict[str, Any]], Tensor]
    | None = None,
    split: Callable[[str, Tensor, Tensor, dict[str, Any]], Tensor]
    | None = None,
) -> FieldBehavior:
    """Build a custom per-field behavior."""
    return FieldBehavior(clone=clone, split=split)


class _SceneOptimizerAdapter:
    """Update scene selectors and optimizer state after topology edits."""

    def __init__(self, optimizers: list[Any]) -> None:
        self._bindings: dict[str, list[Any]] = {}
        for binding in optimizers:
            matches_target = getattr(binding, "matches_target", None)
            field_name = getattr(binding, "field_name", None)
            if not callable(matches_target) or field_name is None:
                continue
            if not matches_target("scene", field_name):
                continue
            self._bindings.setdefault(field_name, []).append(binding)

    def replace_scene_fields(
        self,
        scene: Any,
        updates: dict[str, Tensor],
        state_transforms: dict[str, Callable[[str, Tensor], Tensor]],
    ) -> Any:
        """Replace scene fields and update optimizer references/state."""
        for name, value in updates.items():
            bindings = self._bindings.get(name)
            if bindings is None:
                continue
            transform = state_transforms[name]
            for binding in bindings:
                replace_parameter = getattr(binding, "replace_parameter", None)
                if callable(replace_parameter):
                    replace_parameter(value, transform)
        return replace(scene, **updates)

    def reset_state(
        self,
        field_names: Sequence[str],
        indices: Tensor,
    ) -> None:
        for name in field_names:
            bindings = self._bindings.get(name)
            if bindings is None:
                continue
            for binding in bindings:
                reset_state = getattr(binding, "reset_state_for_indices", None)
                if callable(reset_state):
                    reset_state(indices)


def _is_per_splat_tensor(name: str, value: Any, num_splats: int) -> bool:
    return (
        isinstance(value, Tensor)
        and value.ndim > 0
        and name != "sh_degree"
        and int(value.shape[0]) == num_splats
    )


def _default_clone_tensor(tensor: Tensor, mask: Tensor) -> Tensor:
    return torch.cat([tensor, tensor[mask]], dim=0)


def _default_split_tensor(
    tensor: Tensor,
    mask: Tensor,
    num_children: int,
) -> Tensor:
    return torch.cat(
        [tensor[~mask], tensor[mask].repeat_interleave(num_children, dim=0)],
        dim=0,
    )


def _split_offsets(
    scene: GaussianScene,
    mask: Tensor,
    num_children: int,
) -> Tensor:
    repeated_scales = torch.exp(scene.log_scales[mask]).repeat_interleave(
        num_children,
        dim=0,
    )
    offsets = torch.normal(
        mean=torch.zeros_like(repeated_scales),
        std=repeated_scales,
    )
    rotations = _quaternion_to_rotation_matrix(
        scene.quaternion_orientation[mask]
    ).repeat_interleave(num_children, dim=0)
    return torch.bmm(
        rotations,
        offsets.unsqueeze(-1),
    ).squeeze(-1)


def _grown_clone_split_tensor(
    *,
    name: str,
    value: Tensor,
    clone_mask: Tensor,
    split_mask: Tensor,
    num_children: int,
    rotated_offsets: Tensor,
    scale_shrink: float,
) -> Tensor:
    retained = torch.cat([value[~split_mask], value[clone_mask]], dim=0)
    if name == "center_position":
        children = (
            value[split_mask].repeat_interleave(num_children, dim=0)
            + rotated_offsets
        )
    elif name == "log_scales":
        child_scale_multiplier = torch.log(
            torch.tensor(
                scale_shrink,
                dtype=value.dtype,
                device=value.device,
            )
        )
        children = (
            value[split_mask].repeat_interleave(num_children, dim=0)
            + child_scale_multiplier
        )
    else:
        children = value[split_mask].repeat_interleave(num_children, dim=0)
    return torch.cat([retained, children], dim=0)


def _clone_split_tensor_segments(
    *,
    name: str,
    value: Tensor,
    clone_mask: Tensor,
    split_mask: Tensor,
    num_children: int,
    rotated_offsets: Tensor,
    scale_shrink: float,
) -> tuple[Tensor, Tensor, Tensor]:
    retained = value[~split_mask]
    clones = value[clone_mask]
    if name == "center_position":
        children = (
            value[split_mask].repeat_interleave(num_children, dim=0)
            + rotated_offsets
        )
    elif name == "log_scales":
        child_scale_multiplier = torch.log(
            torch.tensor(
                scale_shrink,
                dtype=value.dtype,
                device=value.device,
            )
        )
        children = (
            value[split_mask].repeat_interleave(num_children, dim=0)
            + child_scale_multiplier
        )
    else:
        children = value[split_mask].repeat_interleave(num_children, dim=0)
    return retained, clones, children


def _clone_split_tensor_final(
    *,
    name: str,
    value: Tensor,
    clone_mask: Tensor,
    split_mask: Tensor,
    num_children: int,
    rotated_offsets: Tensor,
    scale_shrink: float,
    keep_mask: Tensor | None,
) -> Tensor:
    retained, clones, children = _clone_split_tensor_segments(
        name=name,
        value=value,
        clone_mask=clone_mask,
        split_mask=split_mask,
        num_children=num_children,
        rotated_offsets=rotated_offsets,
        scale_shrink=scale_shrink,
    )
    if keep_mask is None:
        return torch.cat([retained, clones, children], dim=0)
    retained_count = int(retained.shape[0])
    clone_count = int(clones.shape[0])
    keep_retained = keep_mask[:retained_count]
    keep_clones = keep_mask[retained_count : retained_count + clone_count]
    keep_children = keep_mask[retained_count + clone_count :]
    return torch.cat(
        [
            retained[keep_retained],
            clones[keep_clones],
            children[keep_children],
        ],
        dim=0,
    )


def _grown_clone_split_state(
    old_value: Tensor,
    clone_mask: Tensor,
    split_mask: Tensor,
    num_children: int,
) -> Tensor:
    clone_state = torch.zeros_like(old_value[clone_mask])
    split_shape = (
        int(split_mask.sum()) * num_children,
        *old_value.shape[1:],
    )
    split_state = torch.zeros(
        split_shape,
        dtype=old_value.dtype,
        device=old_value.device,
    )
    return torch.cat([old_value[~split_mask], clone_state, split_state], dim=0)


def _clone_split_state_final(
    old_value: Tensor,
    clone_mask: Tensor,
    split_mask: Tensor,
    num_children: int,
    keep_mask: Tensor | None,
) -> Tensor:
    retained = old_value[~split_mask]
    clone_state = torch.zeros_like(old_value[clone_mask])
    split_shape = (
        int(split_mask.sum()) * num_children,
        *old_value.shape[1:],
    )
    split_state = torch.zeros(
        split_shape,
        dtype=old_value.dtype,
        device=old_value.device,
    )
    if keep_mask is None:
        return torch.cat([retained, clone_state, split_state], dim=0)
    retained_count = int(retained.shape[0])
    clone_count = int(clone_state.shape[0])
    keep_retained = keep_mask[:retained_count]
    keep_clones = keep_mask[retained_count : retained_count + clone_count]
    keep_split = keep_mask[retained_count + clone_count :]
    return torch.cat(
        [
            retained[keep_retained],
            clone_state[keep_clones],
            split_state[keep_split],
        ],
        dim=0,
    )


def _record_metrics(state: Any, metrics: dict[str, float]) -> None:
    diagnostics = getattr(state, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        return
    diagnostics.setdefault("metrics", {}).update(metrics)


def _cuda_allocated(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.memory_allocated(device))


def _gaussian_scene_extent(scene: GaussianScene) -> float:
    positions = scene.center_position.detach()
    extent = positions.max(dim=0).values - positions.min(dim=0).values
    return float(extent.max().clamp_min(1e-6).item())


def _quaternion_to_rotation_matrix(
    quaternion: Float[Tensor, "num_splats 4"],
) -> Float[Tensor, "num_splats 3 3"]:
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    qw, qx, qy, qz = q.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qx * qw),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx * qx + qy * qy),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )


class GaussianFamilyOps:
    """Topology edits for Gaussian scenes."""

    def __init__(self, state: Any, optimizers: list[Any]) -> None:
        self._state = state
        self._optimizer_adapter = _SceneOptimizerAdapter(optimizers)

    @property
    def scene(self) -> GaussianScene:
        scene = self._state.model.scene
        if not isinstance(scene, GaussianScene):
            raise TypeError("GaussianFamilyOps requires a GaussianScene.")
        return scene

    def scene_extent(self) -> float:
        return _gaussian_scene_extent(self.scene)

    def _replace_scene(
        self,
        scene: GaussianScene,
        updates: dict[str, Tensor],
        state_transforms: dict[str, Callable[[str, Tensor], Tensor]],
    ) -> None:
        self._state.model = replace(
            self._state.model,
            scene=self._optimizer_adapter.replace_scene_fields(
                scene,
                updates,
                state_transforms,
            ),
        )

    def _tensor_field_names(self) -> tuple[str, ...]:
        scene = self.scene
        num_splats = int(scene.center_position.shape[0])
        return tuple(
            field_def.name
            for field_def in fields(scene)
            if _is_per_splat_tensor(
                field_def.name,
                getattr(scene, field_def.name),
                num_splats,
            )
        )

    def clone(
        self,
        mask: Bool[Tensor, " num_splats"],
        field_behaviors: dict[str, FieldBehavior] | None = None,
    ) -> None:
        scene = self.scene
        behaviors = field_behaviors or {}
        num_splats = int(scene.center_position.shape[0])
        updates: dict[str, Tensor] = {}
        state_transforms: dict[str, Callable[[str, Tensor], Tensor]] = {}
        for field_def in fields(scene):
            name = field_def.name
            value = getattr(scene, name)
            if not _is_per_splat_tensor(name, value, num_splats):
                continue
            behavior = behaviors.get(name)
            new_value = (
                behavior.clone(name, value, mask, {})
                if behavior is not None and behavior.clone is not None
                else _default_clone_tensor(value, mask)
            )
            updates[name] = new_value.detach().requires_grad_(
                value.requires_grad
            )
            state_transforms[name] = lambda _key, old_value, local_mask=mask: (
                torch.cat(
                    [old_value, torch.zeros_like(old_value[local_mask])],
                    dim=0,
                )
            )
        self._replace_scene(scene, updates, state_transforms)

    def split(
        self,
        mask: Bool[Tensor, " num_splats"],
        *,
        num_children: int = 2,
        field_behaviors: dict[str, FieldBehavior] | None = None,
        scale_shrink: float = 0.8,
    ) -> None:
        scene = self.scene
        if int(mask.sum()) == 0:
            return
        behaviors = field_behaviors or {}
        rotated_offsets = _split_offsets(scene, mask, num_children)
        child_scale_multiplier = torch.log(
            torch.tensor(
                scale_shrink,
                dtype=scene.log_scales.dtype,
                device=scene.log_scales.device,
            )
        )
        updates: dict[str, Tensor] = {}
        state_transforms: dict[str, Callable[[str, Tensor], Tensor]] = {}
        context = {
            "num_children": num_children,
            "offsets": rotated_offsets,
        }
        num_splats = int(scene.center_position.shape[0])
        for field_def in fields(scene):
            name = field_def.name
            value = getattr(scene, name)
            if not _is_per_splat_tensor(name, value, num_splats):
                continue
            behavior = behaviors.get(name)
            if behavior is not None and behavior.split is not None:
                new_value = behavior.split(name, value, mask, context)
            elif name == "center_position":
                new_value = torch.cat(
                    [
                        value[~mask],
                        value[mask].repeat_interleave(num_children, dim=0)
                        + rotated_offsets,
                    ],
                    dim=0,
                )
            elif name == "log_scales":
                new_value = torch.cat(
                    [
                        value[~mask],
                        value[mask].repeat_interleave(num_children, dim=0)
                        + child_scale_multiplier,
                    ],
                    dim=0,
                )
            else:
                new_value = _default_split_tensor(value, mask, num_children)
            updates[name] = new_value.detach().requires_grad_(
                value.requires_grad
            )
            state_transforms[name] = (
                lambda _key, old_value, local_mask=mask, repeats=num_children: (
                    torch.cat(
                        [
                            old_value[~local_mask],
                            torch.zeros(
                                (
                                    int(local_mask.sum()) * repeats,
                                    *old_value.shape[1:],
                                ),
                                dtype=old_value.dtype,
                                device=old_value.device,
                            ),
                        ],
                        dim=0,
                    )
                )
            )
        self._replace_scene(scene, updates, state_transforms)

    def clone_and_split(
        self,
        clone_mask: Bool[Tensor, " num_splats"],
        split_mask: Bool[Tensor, " num_splats"],
        *,
        num_children: int = 2,
        scale_shrink: float = 0.8,
        prune_fn: Callable[[GaussianScene], Tensor] | None = None,
        prune_field_names: Sequence[str] | None = None,
    ) -> None:
        """Apply clone, split, and optional prune in one scene replacement."""
        start = time.perf_counter()
        scene = self.scene
        device = scene.center_position.device
        memory_before = _cuda_allocated(device)
        primitive_count_before = int(scene.center_position.shape[0])
        clone_count = int(clone_mask.sum())
        split_count = int(split_mask.sum())
        if clone_count == 0 and split_count == 0:
            if prune_fn is not None:
                before_prune = int(scene.center_position.shape[0])
                self.prune(prune_fn(scene))
                after_prune = int(self.scene.center_position.shape[0])
                self._record_refinement_metrics(
                    clone_count=0,
                    split_count=0,
                    num_children=num_children,
                    primitive_count_before=before_prune,
                    primitive_count_after=after_prune,
                    memory_before=memory_before,
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                )
            return
        rotated_offsets = (
            _split_offsets(scene, split_mask, num_children)
            if split_count > 0
            else torch.empty(
                (0, scene.center_position.shape[1]),
                dtype=scene.center_position.dtype,
                device=scene.center_position.device,
            )
        )
        num_splats = int(scene.center_position.shape[0])
        grown_updates: dict[str, Tensor] = {}
        prune_field_name_set = (
            None if prune_field_names is None else set(prune_field_names)
        )
        for field_def in fields(scene):
            name = field_def.name
            value = getattr(scene, name)
            if not _is_per_splat_tensor(name, value, num_splats):
                continue
            if prune_field_name_set is not None and name not in prune_field_name_set:
                continue
            grown_updates[name] = _grown_clone_split_tensor(
                name=name,
                value=value,
                clone_mask=clone_mask,
                split_mask=split_mask,
                num_children=num_children,
                rotated_offsets=rotated_offsets,
                scale_shrink=scale_shrink,
            )
        keep_mask = None
        if prune_fn is not None:
            grown_scene = replace(scene, **grown_updates)
            keep_mask = prune_fn(grown_scene)
        updates: dict[str, Tensor] = {}
        state_transforms: dict[str, Callable[[str, Tensor], Tensor]] = {}
        for field_def in fields(scene):
            name = field_def.name
            value = getattr(scene, name)
            if not _is_per_splat_tensor(name, value, num_splats):
                continue
            grown_value = grown_updates.get(name)
            new_value = (
                _clone_split_tensor_final(
                    name=name,
                    value=value,
                    clone_mask=clone_mask,
                    split_mask=split_mask,
                    num_children=num_children,
                    rotated_offsets=rotated_offsets,
                    scale_shrink=scale_shrink,
                    keep_mask=keep_mask,
                )
                if grown_value is None
                else (
                    grown_value
                    if keep_mask is None
                    else grown_value[keep_mask]
                )
            )
            updates[name] = new_value.detach().requires_grad_(
                value.requires_grad
            )
            state_transforms[name] = (
                lambda _key,
                old_value,
                local_clone=clone_mask,
                local_split=split_mask,
                local_keep=keep_mask,
                repeats=num_children: (
                    _clone_split_state_final(
                        old_value,
                        local_clone,
                        local_split,
                        repeats,
                        local_keep,
                    )
                )
            )
        self._replace_scene(scene, updates, state_transforms)
        self._record_refinement_metrics(
            clone_count=clone_count,
            split_count=split_count,
            num_children=num_children,
            primitive_count_before=primitive_count_before,
            primitive_count_after=int(self.scene.center_position.shape[0]),
            memory_before=memory_before,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )

    def _record_refinement_metrics(
        self,
        *,
        clone_count: int,
        split_count: int,
        num_children: int,
        primitive_count_before: int,
        primitive_count_after: int,
        memory_before: int | None,
        elapsed_ms: float,
    ) -> None:
        memory_after = _cuda_allocated(self.scene.center_position.device)
        metrics = {
            "refinement_clone_count": float(clone_count),
            "refinement_split_count": float(split_count),
            "refinement_prune_count": float(
                primitive_count_before
                + clone_count
                + split_count * (num_children - 1)
                - primitive_count_after
            ),
            "refinement_primitives_before": float(primitive_count_before),
            "refinement_primitives_after": float(primitive_count_after),
            "refinement_elapsed_ms": float(elapsed_ms),
        }
        if memory_before is not None and memory_after is not None:
            metrics["refinement_cuda_allocated_delta_bytes"] = float(
                memory_after - memory_before
            )
        _record_metrics(self._state, metrics)

    def prune(self, keep_mask: Bool[Tensor, " num_splats"]) -> None:
        scene = self.scene
        updates: dict[str, Tensor] = {}
        state_transforms: dict[str, Callable[[str, Tensor], Tensor]] = {}
        num_splats = int(scene.center_position.shape[0])
        for field_def in fields(scene):
            name = field_def.name
            value = getattr(scene, name)
            if not _is_per_splat_tensor(name, value, num_splats):
                continue
            updates[name] = (
                value[keep_mask].detach().requires_grad_(value.requires_grad)
            )
            state_transforms[name] = (
                lambda _key, old_value, local_keep=keep_mask: old_value[
                    local_keep
                ]
            )
        self._replace_scene(scene, updates, state_transforms)

    def reset_opacity(self, max_post_sigmoid_value: float) -> None:
        scene = self.scene
        target = torch.tensor(
            max_post_sigmoid_value,
            dtype=scene.logit_opacity.dtype,
            device=scene.logit_opacity.device,
        ).clamp(1e-5, 1.0 - 1e-5)
        capped = torch.minimum(scene.logit_opacity, torch.logit(target))
        updates = {
            "logit_opacity": capped.detach().requires_grad_(
                scene.logit_opacity.requires_grad
            )
        }
        self._replace_scene(
            scene,
            updates,
            {
                "logit_opacity": lambda _key, old_value: torch.zeros_like(
                    old_value
                )
            },
        )

    def copy_from_indices(
        self,
        target_indices: Int[Tensor, " num_targets"],
        source_indices: Int[Tensor, " num_targets"],
        field_overrides: dict[str, Tensor] | None = None,
    ) -> None:
        scene = self.scene
        overrides = field_overrides or {}
        updates: dict[str, Tensor] = {}
        state_transforms: dict[str, Callable[[str, Tensor], Tensor]] = {}
        for name in self._tensor_field_names():
            value = getattr(scene, name)
            copied = value.detach().clone()
            copied[target_indices] = overrides.get(name, value[source_indices])
            updates[name] = copied.requires_grad_(value.requires_grad)
            state_transforms[name] = lambda _key, old_value: old_value
        self._replace_scene(scene, updates, state_transforms)

    def copy_to_indices(
        self,
        assignments: Sequence[tuple[Tensor, Tensor, dict[str, Tensor]]],
    ) -> None:
        """Copy several source-index selections into targets in one update."""
        scene = self.scene
        updates: dict[str, Tensor] = {}
        state_transforms: dict[str, Callable[[str, Tensor], Tensor]] = {}
        for name in self._tensor_field_names():
            value = getattr(scene, name)
            copied = value.detach().clone()
            for target_indices, source_indices, overrides in assignments:
                copied[target_indices] = overrides.get(
                    name,
                    value[source_indices],
                )
            updates[name] = copied.requires_grad_(value.requires_grad)
            state_transforms[name] = lambda _key, old_value: old_value
        self._replace_scene(scene, updates, state_transforms)

    def append_from_indices(
        self,
        source_indices: Int[Tensor, " num_sources"],
        field_overrides: dict[str, Tensor] | None = None,
    ) -> None:
        scene = self.scene
        overrides = field_overrides or {}
        updates: dict[str, Tensor] = {}
        state_transforms: dict[str, Callable[[str, Tensor], Tensor]] = {}
        for name in self._tensor_field_names():
            value = getattr(scene, name)
            appended = overrides.get(name, value[source_indices])
            updates[name] = (
                torch.cat([value, appended], dim=0)
                .detach()
                .requires_grad_(value.requires_grad)
            )
            state_transforms[name] = (
                lambda _key, old_value, local_indices=source_indices: torch.cat(
                    [old_value, torch.zeros_like(old_value[local_indices])],
                    dim=0,
                )
            )
        self._replace_scene(scene, updates, state_transforms)

    def copy_and_append_from_indices(
        self,
        source_indices: Int[Tensor, " num_sources"],
        field_overrides: dict[str, Tensor] | None = None,
    ) -> None:
        """Update source rows with overrides and append the same rows once."""
        scene = self.scene
        overrides = field_overrides or {}
        updates: dict[str, Tensor] = {}
        state_transforms: dict[str, Callable[[str, Tensor], Tensor]] = {}
        for name in self._tensor_field_names():
            value = getattr(scene, name)
            copied = value.detach().clone()
            appended = overrides.get(name, value[source_indices])
            copied[source_indices] = appended
            updates[name] = (
                torch.cat([copied, appended], dim=0)
                .detach()
                .requires_grad_(value.requires_grad)
            )
            state_transforms[name] = (
                lambda _key, old_value, local_indices=source_indices: torch.cat(
                    [old_value, torch.zeros_like(old_value[local_indices])],
                    dim=0,
                )
            )
        self._replace_scene(scene, updates, state_transforms)

    def reorder(self, order: Int[Tensor, " num_splats"]) -> None:
        scene = self.scene
        updates: dict[str, Tensor] = {}
        state_transforms: dict[str, Callable[[str, Tensor], Tensor]] = {}
        for name in self._tensor_field_names():
            value = getattr(scene, name)
            updates[name] = (
                value[order].detach().requires_grad_(value.requires_grad)
            )
            state_transforms[name] = lambda _key, old_value, local_order=order: (
                old_value[local_order]
            )
        self._replace_scene(scene, updates, state_transforms)

    def replace_fields(self, updates: dict[str, Tensor]) -> None:
        """Replace scene tensor fields while preserving optimizer state."""
        scene = self.scene
        replaced = {
            name: value.detach().requires_grad_(getattr(scene, name).requires_grad)
            for name, value in updates.items()
        }
        self._replace_scene(
            scene,
            replaced,
            {name: lambda _key, old_value: old_value for name in replaced},
        )

    def reset_optimizer_state(
        self,
        indices: Int[Tensor, " num_selected"],
        field_names: Sequence[str] | None = None,
    ) -> None:
        resolved_field_names = (
            tuple(field_names)
            if field_names is not None
            else self._tensor_field_names()
        )
        self._optimizer_adapter.reset_state(resolved_field_names, indices)

    def decay_opacity(self, gamma: float) -> None:
        scene = self.scene
        opacity = torch.sigmoid(scene.logit_opacity) * gamma
        capped = torch.logit(opacity.clamp(1e-5, 1.0 - 1e-5))
        updates = {
            "logit_opacity": capped.detach().requires_grad_(
                scene.logit_opacity.requires_grad
            )
        }
        self._state.model = replace(
            self._state.model,
            scene=self._optimizer_adapter.replace_scene_fields(
                scene,
                updates,
                {
                    "logit_opacity": lambda _key, old_value: torch.zeros_like(
                        old_value
                    )
                },
            ),
        )

    def jitter_positions(self, sigma: float) -> None:
        scene = self.scene
        updates = {
            "center_position": (
                scene.center_position
                + torch.randn_like(scene.center_position) * sigma
            )
            .detach()
            .requires_grad_(scene.center_position.requires_grad)
        }
        self._state.model = replace(
            self._state.model,
            scene=self._optimizer_adapter.replace_scene_fields(
                scene,
                updates,
                {
                    "center_position": lambda _key, old_value: torch.zeros_like(
                        old_value
                    )
                },
            ),
        )


def _encode_child_octpath(
    octpath: Int[Tensor, "num_voxels 1"],
    octlevel: Int[Tensor, "num_voxels 1"],
    max_num_levels: int,
) -> Int[Tensor, "num_voxels 8 1"]:
    child_level = octlevel.to(torch.int64) + 1
    bit_shift = 3 * (max_num_levels - child_level)
    child_ids = torch.arange(8, device=octpath.device, dtype=torch.int64).view(
        1, 8, 1
    )
    parent = octpath.to(torch.int64).reshape(-1, 1, 1).expand(-1, 8, -1)
    return parent | (child_ids << bit_shift.reshape(-1, 1, 1))


def _child_voxel_geometries(
    parent_geometries: Float[Tensor, "num_voxels 8"],
) -> Float[Tensor, "num_voxels*8 8"]:
    parent_corners = _SUBTREE_SHIFTS.to(
        device=parent_geometries.device,
        dtype=parent_geometries.dtype,
    )
    child_base = parent_corners[:, None, :]
    child_corner = parent_corners[None, :, :]
    coords = (child_base + child_corner) / 2.0
    weights = []
    for parent_corner in parent_corners:
        weights.append(
            torch.where(parent_corner.bool(), coords, 1.0 - coords).prod(dim=-1)
        )
    weight_tensor = torch.stack(weights, dim=-1)
    child = torch.einsum("vp,ckp->vck", parent_geometries, weight_tensor)
    return child.reshape(-1, 8)


def _rebuild_sparse_geo_grid(
    scene: SparseVoxelScene,
    octpath: Int[Tensor, "num_voxels 1"],
    octlevel: Int[Tensor, "num_voxels 1"],
    voxel_geometries: Float[Tensor, "num_voxels 8"],
) -> Float[Tensor, "num_grid_points 1"]:
    _, vox_key = svraster_build_grid_points_link(
        octpath,
        octlevel,
        backend_name=scene.backend_name,
        max_num_levels=scene.max_num_levels,
    )
    num_grid_points = (
        int(vox_key.max().item()) + 1 if vox_key.numel() > 0 else 0
    )
    sums = torch.zeros(
        (num_grid_points, 1),
        dtype=voxel_geometries.dtype,
        device=voxel_geometries.device,
    )
    counts = torch.zeros_like(sums)
    flat_key = vox_key.reshape(-1)
    flat_values = voxel_geometries.reshape(-1, 1)
    sums.index_add_(0, flat_key, flat_values)
    counts.index_add_(0, flat_key, torch.ones_like(flat_values))
    return sums / counts.clamp_min(1.0)


class SparseVoxelFamilyOps:
    """Topology edits for sparse-voxel scenes."""

    def __init__(self, state: Any, optimizers: list[Any]) -> None:
        self._state = state
        self._optimizer_adapter = _SceneOptimizerAdapter(optimizers)

    @property
    def scene(self) -> SparseVoxelScene:
        scene = self._state.model.scene
        if not isinstance(scene, SparseVoxelScene):
            raise TypeError("SparseVoxelFamilyOps requires a SparseVoxelScene.")
        return scene

    def prune(self, keep_mask: Bool[Tensor, " num_voxels"]) -> None:
        scene = self.scene
        voxel_geometries = scene.voxel_geometries[keep_mask]
        octpath = scene.octpath[keep_mask]
        octlevel = scene.octlevel[keep_mask]
        geo_grid_pts = _rebuild_sparse_geo_grid(
            scene,
            octpath,
            octlevel,
            voxel_geometries,
        )
        updates = {
            "octpath": octpath,
            "octlevel": octlevel,
            "geo_grid_pts": geo_grid_pts.detach().requires_grad_(
                scene.geo_grid_pts.requires_grad
            ),
            "sh0": scene.sh0[keep_mask]
            .detach()
            .requires_grad_(scene.sh0.requires_grad),
            "shs": scene.shs[keep_mask]
            .detach()
            .requires_grad_(scene.shs.requires_grad),
        }
        if scene.subdivision_priority is not None:
            updates["subdivision_priority"] = (
                scene.subdivision_priority[keep_mask]
                .detach()
                .requires_grad_(scene.subdivision_priority.requires_grad)
            )
        self._state.model = replace(
            self._state.model,
            scene=self._optimizer_adapter.replace_scene_fields(
                scene,
                updates,
                {
                    "sh0": lambda _key, old_value: old_value[keep_mask],
                    "shs": lambda _key, old_value: old_value[keep_mask],
                    "geo_grid_pts": lambda _key, old_value: torch.zeros_like(
                        geo_grid_pts
                    ),
                    "subdivision_priority": lambda _key, old_value: (
                        old_value[keep_mask]
                    ),
                },
            ),
        )

    def subdivide(self, mask: Bool[Tensor, " num_voxels"]) -> None:
        scene = self.scene
        if int(mask.sum()) == 0:
            return
        if torch.any(scene.octlevel[mask] >= scene.max_num_levels):
            raise ValueError("Cannot subdivide voxels beyond max_num_levels.")
        keep_mask = ~mask
        child_octpath = _encode_child_octpath(
            scene.octpath[mask],
            scene.octlevel[mask],
            scene.max_num_levels,
        ).reshape(-1, 1)
        child_octlevel = (
            scene.octlevel[mask].to(torch.int64) + 1
        ).repeat_interleave(8, dim=0)
        child_sh0 = scene.sh0[mask].repeat_interleave(8, dim=0)
        child_shs = scene.shs[mask].repeat_interleave(8, dim=0)
        child_geometries = _child_voxel_geometries(scene.voxel_geometries[mask])
        octpath = torch.cat([scene.octpath[keep_mask], child_octpath], dim=0)
        octlevel = torch.cat([scene.octlevel[keep_mask], child_octlevel], dim=0)
        sh0 = torch.cat([scene.sh0[keep_mask], child_sh0], dim=0)
        shs = torch.cat([scene.shs[keep_mask], child_shs], dim=0)
        if scene.subdivision_priority is not None:
            child_subdivision_priority = torch.ones(
                (int(child_octpath.shape[0]), 1),
                dtype=scene.subdivision_priority.dtype,
                device=scene.subdivision_priority.device,
            )
            subdivision_priority = torch.cat(
                [
                    scene.subdivision_priority[keep_mask],
                    child_subdivision_priority,
                ],
                dim=0,
            )
        voxel_geometries = torch.cat(
            [scene.voxel_geometries[keep_mask], child_geometries],
            dim=0,
        )
        geo_grid_pts = _rebuild_sparse_geo_grid(
            scene,
            octpath,
            octlevel,
            voxel_geometries,
        )
        updates = {
            "octpath": octpath,
            "octlevel": octlevel,
            "geo_grid_pts": geo_grid_pts.detach().requires_grad_(
                scene.geo_grid_pts.requires_grad
            ),
            "sh0": sh0.detach().requires_grad_(scene.sh0.requires_grad),
            "shs": shs.detach().requires_grad_(scene.shs.requires_grad),
        }
        if scene.subdivision_priority is not None:
            updates["subdivision_priority"] = (
                subdivision_priority.detach().requires_grad_(
                    scene.subdivision_priority.requires_grad
                )
            )
        self._state.model = replace(
            self._state.model,
            scene=self._optimizer_adapter.replace_scene_fields(
                scene,
                updates,
                {
                    "sh0": lambda _key, old_value: torch.cat(
                        [old_value[keep_mask], torch.zeros_like(child_sh0)],
                        dim=0,
                    ),
                    "shs": lambda _key, old_value: torch.cat(
                        [old_value[keep_mask], torch.zeros_like(child_shs)],
                        dim=0,
                    ),
                    "geo_grid_pts": lambda _key, old_value: torch.zeros_like(
                        geo_grid_pts
                    ),
                    "subdivision_priority": lambda _key, old_value: torch.cat(
                        [
                            old_value[keep_mask],
                            torch.zeros_like(child_subdivision_priority),
                        ],
                        dim=0,
                    ),
                },
            ),
        )

    def reset_subdivision_priority(self) -> None:
        """Reset persistent sparse-voxel subdivision priority to one."""
        scene = self.scene
        if scene.subdivision_priority is None:
            return
        subdivision_priority = torch.ones_like(scene.subdivision_priority)
        updates = {
            "subdivision_priority": subdivision_priority.detach().requires_grad_(
                scene.subdivision_priority.requires_grad
            )
        }
        self._state.model = replace(
            self._state.model,
            scene=self._optimizer_adapter.replace_scene_fields(
                scene,
                updates,
                {
                    "subdivision_priority": lambda _key, old_value: (
                        torch.zeros_like(old_value)
                    )
                },
            ),
        )


def build_family_ops(state: Any, optimizers: list[Any]) -> Any:
    """Build family ops for the current scene."""
    family = state.model.scene.scene_family
    if family == "gaussian":
        return GaussianFamilyOps(state, optimizers)
    if family == "sparse_voxel":
        return SparseVoxelFamilyOps(state, optimizers)
    raise ValueError(f"Unsupported densification scene family {family!r}.")


__all__ = [
    "FieldBehavior",
    "GaussianFamilyOps",
    "SparseVoxelFamilyOps",
    "build_family_ops",
    "copy_field",
    "custom_field",
    "zero_field",
]
