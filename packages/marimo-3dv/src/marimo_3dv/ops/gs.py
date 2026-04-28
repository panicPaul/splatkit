"""Declarative GS render-view nodes and post-render effects."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from math import isqrt
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from jaxtyping import Float
from plyfile import PlyData
from pydantic import BaseModel, Field
from torch import Tensor

from marimo_config_gui import form_gui
from marimo_3dv.pipeline.bundle import ViewerBackendBundle, backend_bundle
from marimo_3dv.pipeline.context import ViewerContext
from marimo_3dv.pipeline.gui import (
    AbstractRenderView,
    EffectNode,
    PipelineGroup,
    RenderNode,
    RenderResult,
    effect_node,
    render_node,
)
from marimo_3dv.viewer.widget import ViewerState, _best_effort_cuda_cleanup


@runtime_checkable
class SplatRenderData(Protocol):
    """Protocol for Gaussian splat scenes compatible with GS nodes."""

    @property
    def opacity_logits(self) -> torch.Tensor:
        """(N, 1) raw opacity logits before sigmoid."""
        ...

    @property
    def center_positions(self) -> torch.Tensor:
        """(N, 3) splat centers in world space."""
        ...

    @property
    def log_half_extents(self) -> torch.Tensor:
        """(N, 3) log-scale half-extents."""
        ...

    @property
    def quaternion_orientation(self) -> torch.Tensor:
        """(N, 4) splat orientations."""
        ...

    @property
    def spherical_harmonics(self) -> torch.Tensor:
        """(N, num_bases, 3) SH coefficients."""
        ...

    @property
    def sh_degree(self) -> int:
        """Maximum SH degree present in the data."""
        ...


@dataclass(frozen=True)
class SplatScene:
    """Minimal Gaussian splat scene used by the default GS backend."""

    center_positions: Float[Tensor, "num_splats 3"]
    log_half_extents: Float[Tensor, "num_splats 3"]
    quaternion_orientation: Float[Tensor, "num_splats 4"]
    spherical_harmonics: Float[Tensor, "num_splats num_bases 3"]
    opacity_logits: Float[Tensor, "num_splats 1"]
    sh_degree: int


class SplatLoadGpuConfig(BaseModel):
    """GPU cleanup options for notebook-driven scene replacement."""

    close_existing_viewer: bool = Field(
        default=True,
        description="Close the active viewer before loading a new scene.",
    )
    empty_cuda_cache: bool = Field(
        default=True,
        description="Release unused CUDA allocator cache before reload.",
    )


class SplatLoadConfig(BaseModel):
    """Configuration for loading a Gaussian splat PLY file."""

    ply_path: Path = Field(
        default=Path.cwd() / "point_cloud.ply",
        description="Path to a 3DGS-style `.ply` file.",
    )
    gpu: SplatLoadGpuConfig = Field(
        default_factory=SplatLoadGpuConfig,
        description="How to clean up GPU resources before replacing a scene.",
    )


def splat_load_form(*, default_path: Path | None = None) -> Any:
    """Build a reusable notebook form for loading splat scenes."""
    default_config = (
        SplatLoadConfig()
        if default_path is None
        else SplatLoadConfig(ply_path=default_path)
    )
    return form_gui(
        SplatLoadConfig,
        value=default_config,
    )


def pick_splat_load_config(
    *,
    title: str = "Open PLY file",
) -> SplatLoadConfig | None:
    """Prompt for a splat PLY file using the native desktop file picker."""
    import subprocess

    result = subprocess.run(
        [
            "zenity",
            "--file-selection",
            f"--title={title}",
            "--file-filter=*.ply",
        ],
        capture_output=True,
        text=True,
    )
    ply_path = Path(result.stdout.strip())
    if not ply_path.exists():
        return None
    return SplatLoadConfig(ply_path=ply_path)


def infer_sh_degree(num_bases: int) -> int:
    """Infer the SH degree from the number of basis functions."""
    degree = isqrt(num_bases) - 1
    if (degree + 1) ** 2 != num_bases:
        raise ValueError(f"Invalid SH basis count: {num_bases}")
    if not 0 <= degree <= 4:
        raise ValueError(f"Only SH degrees 0-4 are supported, got {degree}")
    return degree


def get_gsplat_device() -> torch.device:
    """Return the default device used for gsplat rasterization."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "gsplat viewer requires CUDA, but CUDA is unavailable."
        )
    return torch.device("cuda")


def cleanup_before_splat_reload(
    viewer_state: ViewerState,
    *,
    close_existing_viewer: bool,
    empty_cuda_cache: bool,
) -> None:
    """Release viewer-owned resources before replacing the active scene."""
    if close_existing_viewer:
        active_ref = viewer_state._active_marimo_viewer_ref
        active_viewer = None if active_ref is None else active_ref()
        if active_viewer is not None:
            active_viewer.close()

    gc.collect()

    if torch.cuda.is_available() and empty_cuda_cache:
        _best_effort_cuda_cleanup()


def load_splat_scene(path: Path) -> SplatScene:
    """Load a 3DGS-style `.ply` file into a GPU-resident splat scene."""
    device = get_gsplat_device()
    ply_data = PlyData.read(path)
    vertices = ply_data["vertex"]
    property_names = list(vertices.data.dtype.names)

    centers = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]],
        axis=1,
    ).astype(np.float32)

    dc_coefficients = np.stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]],
        axis=1,
    ).astype(np.float32)

    rest_feature_names = sorted(
        [name for name in property_names if name.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    num_rest_coefficients = len(rest_feature_names)
    if num_rest_coefficients % 3 != 0:
        raise ValueError(
            "Expected the number of `f_rest_*` attributes to be divisible by 3."
        )

    num_bases = 1 + num_rest_coefficients // 3
    sh_degree = infer_sh_degree(num_bases)
    sh_coefficients = np.zeros(
        (centers.shape[0], num_bases, 3),
        dtype=np.float32,
    )
    sh_coefficients[:, 0, :] = dc_coefficients
    if rest_feature_names:
        rest_coefficients = np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in rest_feature_names
            ],
            axis=1,
        )
        rest_coefficients = rest_coefficients.reshape(
            centers.shape[0],
            3,
            num_bases - 1,
        )
        sh_coefficients[:, 1:num_bases, :] = np.transpose(
            rest_coefficients,
            (0, 2, 1),
        )

    scale_feature_names = sorted(
        [name for name in property_names if name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    rotation_feature_names = sorted(
        [name for name in property_names if name.startswith("rot")],
        key=lambda name: int(name.split("_")[-1]),
    )
    log_scales = (
        np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in scale_feature_names
            ],
            axis=1,
        )
        if scale_feature_names
        else np.full((centers.shape[0], 3), np.log(0.01), dtype=np.float32)
    )
    rotations = (
        np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in rotation_feature_names
            ],
            axis=1,
        )
        if rotation_feature_names
        else np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (centers.shape[0], 1),
        )
    )
    opacity_logits = np.asarray(vertices["opacity"], dtype=np.float32)[:, None]

    return SplatScene(
        center_positions=torch.from_numpy(centers).to(device=device),
        log_half_extents=torch.from_numpy(log_scales).to(device=device),
        quaternion_orientation=torch.from_numpy(rotations).to(device=device),
        spherical_harmonics=torch.from_numpy(sh_coefficients).to(device=device),
        opacity_logits=torch.from_numpy(opacity_logits).to(device=device),
        sh_degree=sh_degree,
    )


def load_splat_scene_from_config(
    config: SplatLoadConfig | None,
    viewer_state: ViewerState,
) -> SplatScene | None:
    """Clean up viewer state and load the scene requested by a load form."""
    if config is None:
        return None
    cleanup_before_splat_reload(
        viewer_state,
        close_existing_viewer=config.gpu.close_existing_viewer,
        empty_cuda_cache=config.gpu.empty_cuda_cache,
    )
    if not config.ply_path.exists():
        return None
    return load_splat_scene(config.ply_path)


@dataclass(frozen=True)
class GsRenderView(AbstractRenderView[SplatRenderData | None]):
    """Immutable symbolic GS render view."""

    backend_key: str = "gsplat"
    keep_mask: torch.Tensor | None = None
    max_sh_degree: int | None = None

    def with_mask(self, keep_mask: torch.Tensor) -> GsRenderView:
        """Return a view with an additional symbolic keep-mask applied."""
        next_mask = keep_mask
        if self.keep_mask is not None:
            next_mask = self.keep_mask & keep_mask
        return GsRenderView(
            source_scene=self.source_scene,
            backend_key=self.backend_key,
            capabilities=self.capabilities,
            extensions=self.extensions,
            keep_mask=next_mask,
            max_sh_degree=self.max_sh_degree,
        )

    def with_max_sh_degree(self, max_sh_degree: int) -> GsRenderView:
        """Return a view with a capped active SH degree."""
        active_degree = max_sh_degree
        if self.max_sh_degree is not None:
            active_degree = min(self.max_sh_degree, max_sh_degree)
        return GsRenderView(
            source_scene=self.source_scene,
            backend_key=self.backend_key,
            capabilities=self.capabilities,
            extensions=self.extensions,
            keep_mask=self.keep_mask,
            max_sh_degree=active_degree,
        )


@dataclass(frozen=True)
class CompiledGsRenderView:
    """Materialized GS render view consumed by the rasterizer backend."""

    center_positions: torch.Tensor | None
    log_half_extents: torch.Tensor
    quaternion_orientation: torch.Tensor | None
    spherical_harmonics: torch.Tensor
    opacity_logits: torch.Tensor
    sh_degree: int
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name in self.extra_fields:
            return self.extra_fields[name]
        raise AttributeError(name)


def gs_render_view(scene: SplatRenderData | None) -> GsRenderView:
    """Create the initial immutable GS render view from a source scene."""
    return GsRenderView(source_scene=scene)


def _masked_value(value: Any, keep_mask: torch.Tensor | None) -> Any:
    if keep_mask is None:
        return value
    if not isinstance(value, torch.Tensor):
        return value
    if value.ndim == 0 or value.shape[0] != keep_mask.shape[0]:
        return value
    return value[keep_mask]


def compile_gs_render_view(view: GsRenderView) -> CompiledGsRenderView | None:
    """Compile a symbolic GS render view into concrete tensors."""
    scene = view.source_scene
    if scene is None:
        return None
    keep_mask = view.keep_mask

    center_positions = _masked_value(
        getattr(scene, "center_positions", None), keep_mask
    )
    log_half_extents = _masked_value(scene.log_half_extents, keep_mask)
    quaternion_orientation = _masked_value(
        getattr(scene, "quaternion_orientation", None), keep_mask
    )
    spherical_harmonics = _masked_value(scene.spherical_harmonics, keep_mask)
    opacity_logits = _masked_value(scene.opacity_logits, keep_mask)
    sh_degree = int(scene.sh_degree)

    if view.max_sh_degree is not None:
        sh_degree = min(sh_degree, view.max_sh_degree)
        num_bases = (sh_degree + 1) ** 2
        spherical_harmonics = spherical_harmonics[:, :num_bases, :]

    known_names = {
        "center_positions",
        "log_half_extents",
        "quaternion_orientation",
        "spherical_harmonics",
        "opacity_logits",
        "sh_degree",
    }
    extra_fields = {
        name: _masked_value(value, keep_mask)
        for name, value in vars(scene).items()
        if name not in known_names
    }
    return CompiledGsRenderView(
        center_positions=center_positions,
        log_half_extents=log_half_extents,
        quaternion_orientation=quaternion_orientation,
        spherical_harmonics=spherical_harmonics,
        opacity_logits=opacity_logits,
        sh_degree=sh_degree,
        extra_fields=extra_fields,
    )


class MaxShDegreeConfig(BaseModel):
    """Configuration for the max_sh_degree node."""

    max_sh_degree: int = Field(
        default=3,
        ge=0,
        le=4,
        description="Maximum SH degree to use during rendering (0 = diffuse).",
    )


def _max_sh_degree_apply(
    render_view: GsRenderView,
    config: MaxShDegreeConfig,
    context: ViewerContext,
) -> GsRenderView:
    del context
    if render_view.source_scene is None:
        return render_view
    return render_view.with_max_sh_degree(config.max_sh_degree)


def max_sh_degree_op(default_degree: int = 3) -> RenderNode[GsRenderView]:
    """Return a render-view node that caps the active SH degree."""
    return render_node(
        name="max_sh_degree",
        config_model=MaxShDegreeConfig,
        default_config=MaxShDegreeConfig(max_sh_degree=default_degree),
        apply=_max_sh_degree_apply,
    )


class FilterOpacityConfig(BaseModel):
    """Configuration for the filter_opacity node."""

    opacity_threshold: float = Field(
        default=0.005,
        ge=0.0,
        le=1.0,
        description="Splats with opacity below this threshold are removed.",
    )


def _filter_opacity_apply(
    render_view: GsRenderView,
    config: FilterOpacityConfig,
    context: ViewerContext,
) -> GsRenderView:
    del context
    if render_view.source_scene is None:
        return render_view
    logits = render_view.source_scene.opacity_logits.squeeze(-1)
    keep_mask = torch.sigmoid(logits) >= config.opacity_threshold
    return render_view.with_mask(keep_mask)


def filter_opacity_op(
    default_threshold: float = 0.005,
) -> RenderNode[GsRenderView]:
    """Return a render-view node that filters splats by opacity."""
    return render_node(
        name="filter_opacity",
        config_model=FilterOpacityConfig,
        default_config=FilterOpacityConfig(opacity_threshold=default_threshold),
        apply=_filter_opacity_apply,
    )


class FilterSizeConfig(BaseModel):
    """Configuration for the filter_size node."""

    max_log_extent: float = Field(
        default=3.0,
        description="Splats with any log-half-extent above this are removed.",
    )


def _filter_size_apply(
    render_view: GsRenderView,
    config: FilterSizeConfig,
    context: ViewerContext,
) -> GsRenderView:
    del context
    if render_view.source_scene is None:
        return render_view
    max_log_extents = render_view.source_scene.log_half_extents.amax(dim=-1)
    keep_mask = max_log_extents <= config.max_log_extent
    return render_view.with_mask(keep_mask)


def filter_size_op(
    default_max_log_extent: float = 3.0,
) -> RenderNode[GsRenderView]:
    """Return a render-view node that filters out oversized splats."""
    return render_node(
        name="filter_size",
        config_model=FilterSizeConfig,
        default_config=FilterSizeConfig(max_log_extent=default_max_log_extent),
        apply=_filter_size_apply,
    )


class ShowDistributionConfig(BaseModel):
    """Configuration for the distribution overlay effect."""

    show_distribution: bool = Field(
        default=False,
        description="Overlay a projected splat count heatmap on the image.",
    )
    distribution_alpha: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Alpha blend weight for the distribution overlay.",
    )


def _show_distribution_apply(
    result: RenderResult,
    config: ShowDistributionConfig,
    context: ViewerContext,
    runtime_state: None,
) -> RenderResult:
    del context, runtime_state
    if not config.show_distribution:
        return result

    projected_means = result.metadata.get("projected_means")
    if projected_means is None:
        return result

    image = result.image.copy()
    height, width = image.shape[:2]
    if isinstance(projected_means, torch.Tensor):
        means_np = projected_means.detach().cpu().numpy()
    else:
        means_np = np.asarray(projected_means)

    xs = np.clip(means_np[:, 0].astype(np.int32), 0, width - 1)
    ys = np.clip(means_np[:, 1].astype(np.int32), 0, height - 1)

    heatmap = np.zeros((height, width), dtype=np.float32)
    np.add.at(heatmap, (ys, xs), 1.0)
    max_count = float(heatmap.max())
    if max_count > 0.0:
        heatmap /= max_count

    import cv2

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)
    heatmap_rgb = heatmap_color[:, :, ::-1]

    alpha = config.distribution_alpha
    blended = (
        image.astype(np.float32) * (1.0 - alpha)
        + heatmap_rgb.astype(np.float32) * alpha
    )
    image = np.clip(blended, 0, 255).astype(np.uint8)
    return RenderResult(image=image, metadata=result.metadata)


def show_distribution_op() -> EffectNode[CompiledGsRenderView, None]:
    """Return a post-render effect that overlays projected splat density."""
    return effect_node(
        name="show_distribution",
        config_model=ShowDistributionConfig,
        default_config=ShowDistributionConfig(),
        apply=_show_distribution_apply,
    )


def gs_backend_bundle() -> ViewerBackendBundle[
    SplatRenderData | None, GsRenderView, CompiledGsRenderView | None
]:
    """Return a lightweight GS backend bundle with optional default groups."""
    return backend_bundle(
        name="gsplat",
        render_view_factory=gs_render_view,
        compile_view=compile_gs_render_view,
        default_render_items=(
            PipelineGroup("shading", max_sh_degree_op()),
            PipelineGroup(
                "filtering",
                filter_opacity_op(),
                filter_size_op(),
            ),
        ),
        default_effect_items=(
            PipelineGroup("diagnostics", show_distribution_op()),
        ),
    )


__all__ = [
    "CompiledGsRenderView",
    "FilterOpacityConfig",
    "FilterSizeConfig",
    "GsRenderView",
    "MaxShDegreeConfig",
    "ShowDistributionConfig",
    "SplatLoadConfig",
    "SplatLoadGpuConfig",
    "SplatRenderData",
    "SplatScene",
    "cleanup_before_splat_reload",
    "compile_gs_render_view",
    "filter_opacity_op",
    "filter_size_op",
    "get_gsplat_device",
    "gs_backend_bundle",
    "gs_render_view",
    "infer_sh_degree",
    "load_splat_scene",
    "load_splat_scene_from_config",
    "max_sh_degree_op",
    "pick_splat_load_config",
    "show_distribution_op",
    "splat_load_form",
]
