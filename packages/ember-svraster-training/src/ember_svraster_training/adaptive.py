"""SVRaster adaptive pruning and subdivision."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from ember_core.core.contracts import SparseVoxelScene
from ember_core.densification.contracts import (
    BaseDensificationMethod,
    DensificationContext,
    DensificationRenderRequirements,
    Schedule,
)
from ember_core.densification.families import SparseVoxelFamilyOps
from jaxtyping import Bool, Float
from torch import Tensor


def _scheduled_prune_threshold(
    step: int,
    *,
    start_step: int,
    end_step: int,
    initial_threshold: float,
    final_threshold: float,
) -> float:
    if end_step <= start_step:
        return final_threshold
    progress = (step - start_step) / float(end_step - start_step)
    progress = min(max(progress, 0.0), 1.0)
    return initial_threshold + progress * (final_threshold - initial_threshold)


def _max_voxel_weight(render_output: Any) -> Float[Tensor, " num_voxels 1"]:
    max_weight = getattr(render_output, "max_weight", None)
    if max_weight is None:
        raise RuntimeError(
            "SVRaster adaptive pruning requires render_output.max_weight. "
            "Enable SVRasterCoreRenderOptions.track_max_weight."
        )
    if max_weight.ndim == 3:
        return max_weight.detach().amax(dim=0)
    if max_weight.ndim == 2:
        return max_weight.detach()
    raise ValueError(
        "SVRaster max_weight must have shape (num_cams, num_voxels, 1) "
        f"or (num_voxels, 1), got {tuple(max_weight.shape)}."
    )


def _view_camera(view: Any) -> Any:
    camera = getattr(view, "camera", None)
    if camera is None:
        raise TypeError("SVRaster adaptive views must provide a camera field.")
    return camera


def _camera_pixel_size(camera: Any) -> float:
    intrinsics = camera.get_intrinsics()[0]
    focal_x = float(intrinsics[0, 0].item())
    image_width = float(camera.width[0].item())
    tanfovx = (image_width * 0.5) / focal_x
    return 2.0 * tanfovx / image_width


def _camera_forward(camera: Any) -> Float[Tensor, " 3"]:
    return camera.cam_to_world[0, :3, 2]


def _camera_position(camera: Any) -> Float[Tensor, " 3"]:
    return camera.cam_to_world[0, :3, 3]


def _current_render_statistics(
    scene: SparseVoxelScene,
    render_output: Any,
) -> tuple[
    Float[Tensor, " num_voxels 1"],
    Float[Tensor, " num_voxels 1"],
] | None:
    max_weight_tensor = getattr(render_output, "max_weight", None)
    if max_weight_tensor is None:
        return None
    max_weight = _max_voxel_weight(render_output)
    min_sample_interval = torch.zeros_like(scene.vox_size)
    return max_weight, min_sample_interval


def _training_view_statistics(
    *,
    scene: SparseVoxelScene,
    model: Any,
    runtime: Any,
) -> tuple[
    Float[Tensor, " num_voxels 1"],
    Float[Tensor, " num_voxels 1"],
] | None:
    max_weight = torch.zeros_like(scene.vox_size)
    min_sample_interval = torch.full_like(scene.vox_size, torch.inf)
    found_statistics = False
    for view in runtime.all_views():
        camera = _view_camera(view)
        render_output = runtime.render_raw(model, camera)
        view_max_weight_tensor = getattr(render_output, "max_weight", None)
        if view_max_weight_tensor is None:
            continue
        view_max_weight = _max_voxel_weight(render_output)
        found_statistics = True
        max_weight = torch.maximum(max_weight, view_max_weight)
        visible_mask = view_max_weight > 0.0
        voxel_to_camera = scene.vox_center - _camera_position(camera)
        view_distance = (voxel_to_camera * _camera_forward(camera)).sum(
            dim=-1,
            keepdim=True,
        )
        sample_interval = view_distance.clamp_min(0.0) * _camera_pixel_size(
            camera
        )
        min_sample_interval = torch.where(
            visible_mask,
            torch.minimum(min_sample_interval, sample_interval),
            min_sample_interval,
        )
    min_sample_interval = torch.where(
        torch.isfinite(min_sample_interval),
        min_sample_interval,
        torch.zeros_like(min_sample_interval),
    )
    if not found_statistics:
        return None
    return max_weight, min_sample_interval


def _subdivision_priority(
    scene: SparseVoxelScene,
) -> Float[Tensor, " num_voxels 1"]:
    priority = scene.resolved_subdivision_priority
    if priority.grad is not None:
        return priority.grad.detach()
    return priority.detach()


def _ensure_nonempty_keep_mask(
    keep_mask: Bool[Tensor, " num_voxels"],
    ranking: Float[Tensor, " num_voxels 1"],
) -> Bool[Tensor, " num_voxels"]:
    """Keep the best-ranked voxel if a prune step would empty the scene."""
    if keep_mask.numel() == 0 or bool(keep_mask.any()):
        return keep_mask
    guarded_mask = keep_mask.clone()
    guarded_mask[torch.argmax(ranking.reshape(-1))] = True
    return guarded_mask


@dataclass
class SVRasterAdaptivePruneSubdivide(BaseDensificationMethod):
    """Paper-style SVRaster adaptive pruning and voxel subdivision."""

    adapt_from: int = 1000
    adapt_every: int = 1000
    prune_until: int = 18000
    prune_threshold_initial: float = 0.0001
    prune_threshold_final: float = 0.05
    subdivide_until: int = 15000
    subdivide_all_until: int = 0
    subdivide_sample_threshold: float = 1.0
    subdivide_proportion: float = 0.05
    subdivide_max_voxels: int = 10_000_000
    final_step_margin: int = 500
    expected_scene_families: tuple[str, ...] = ("sparse_voxel",)
    _family_ops: SparseVoxelFamilyOps | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def bind(
        self,
        state: Any,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        """Bind sparse-voxel topology operations."""
        del state, optimizers
        if not isinstance(family_ops, SparseVoxelFamilyOps):
            raise TypeError(
                "SVRasterAdaptivePruneSubdivide requires "
                "SparseVoxelFamilyOps."
            )
        self._family_ops = family_ops

    def get_render_requirements(
        self,
        state: object,
    ) -> DensificationRenderRequirements:
        """Request native max-weight tracking for prune statistics."""
        del state
        return DensificationRenderRequirements(
            backend_options={"track_max_weight": True}
        )

    def _schedule(self, step: int, max_steps: int | None) -> Schedule:
        end_iteration = (
            -1
            if max_steps is None
            else max(0, max_steps - self.final_step_margin)
        )
        return Schedule(
            start_iteration=self.adapt_from,
            end_iteration=end_iteration,
            frequency=self.adapt_every,
        )

    def _subdivide_mask(
        self,
        scene: SparseVoxelScene,
        keep_mask: Bool[Tensor, " num_voxels"],
        min_sample_interval: Float[Tensor, " num_voxels 1"],
        step: int,
    ) -> Bool[Tensor, " num_voxels"]:
        valid_mask = keep_mask & (scene.octlevel.reshape(-1) < scene.max_num_levels)
        if valid_mask.sum() == 0:
            return valid_mask
        size_threshold = (
            min_sample_interval.reshape(-1) * self.subdivide_sample_threshold
        )
        valid_mask = valid_mask & (scene.vox_size.reshape(-1) * 0.5 > size_threshold)
        priority = _subdivision_priority(scene).reshape(-1)
        masked_priority = priority * valid_mask.to(priority.dtype)
        if step <= self.subdivide_all_until:
            selected_mask = valid_mask
        else:
            valid_priority = masked_priority[valid_mask]
            threshold = torch.quantile(
                valid_priority,
                1.0 - self.subdivide_proportion,
            )
            selected_mask = (masked_priority > threshold) & valid_mask
        max_selected = max(
            0,
            round((self.subdivide_max_voxels - scene.num_voxels) / 7),
        )
        if int(selected_mask.sum()) <= max_selected:
            return selected_mask
        selected_priority = masked_priority[selected_mask]
        selected_indices = torch.where(selected_mask)[0]
        top_indices = selected_indices[
            torch.topk(selected_priority, k=max_selected).indices
        ]
        capped_mask = torch.zeros_like(selected_mask)
        capped_mask[top_indices] = True
        return capped_mask

    def post_optimizer_step(self, context: DensificationContext) -> None:
        """Run scheduled pruning and subdivision after optimizer updates."""
        if self._family_ops is None:
            raise RuntimeError("SVRaster adaptive method is not bound.")
        scene = context.state.model.scene
        if not isinstance(scene, SparseVoxelScene):
            return
        max_steps = getattr(context.state, "max_steps", None)
        paper_iteration = context.step + 1
        if not self._schedule(paper_iteration, max_steps).includes(
            paper_iteration
        ):
            return

        statistics = _current_render_statistics(
            scene,
            context.render_output,
        )
        if context.runtime is not None and hasattr(context.runtime, "all_views"):
            statistics = _training_view_statistics(
                scene=scene,
                model=context.state.model,
                runtime=context.runtime,
            )
        if statistics is None:
            return
        max_weight, min_sample_interval = statistics

        keep_mask = torch.ones(
            (scene.num_voxels,),
            dtype=torch.bool,
            device=scene.octpath.device,
        )
        if paper_iteration <= self.prune_until:
            threshold = _scheduled_prune_threshold(
                paper_iteration,
                start_step=self.adapt_from,
                end_step=self.prune_until,
                initial_threshold=self.prune_threshold_initial,
                final_threshold=self.prune_threshold_final,
            )
            keep_mask = max_weight.reshape(-1) >= threshold
            keep_mask = _ensure_nonempty_keep_mask(keep_mask, max_weight)

        subdivide_mask = torch.zeros_like(keep_mask)
        if (
            paper_iteration <= self.subdivide_until
            and scene.num_voxels < self.subdivide_max_voxels
        ):
            subdivide_mask = self._subdivide_mask(
                scene,
                keep_mask,
                min_sample_interval,
                paper_iteration,
            )

        if int(keep_mask.sum()) != scene.num_voxels:
            self._family_ops.prune(keep_mask)
            scene = self._family_ops.scene
            subdivide_mask = subdivide_mask[keep_mask]
        if int(subdivide_mask.sum()) > 0:
            self._family_ops.subdivide(subdivide_mask)
            self._family_ops.reset_subdivision_priority()


__all__ = ["SVRasterAdaptivePruneSubdivide"]
