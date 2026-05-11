"""Pure viewer preparation helpers and allocation-stable caches."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch
from jaxtyping import Bool, Float
from pydantic import BaseModel
from torch import Tensor

from ember_core.core.contracts import GaussianScene

PreparedT = TypeVar("PreparedT")


@dataclass
class ViewerRenderResult:
    """Image and optional metadata produced by notebook viewer render code."""

    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ViewerPrepCache:
    """Cache for camera-independent viewer artifacts.

    The cache owns only derived artifacts. Clearing it must never modify the
    original scene objects that were used to produce those artifacts.
    """

    max_entries: int | None = None
    _entries: OrderedDict[Hashable, Any] = field(default_factory=OrderedDict)

    def __post_init__(self) -> None:
        """Validate cache bounds."""
        if self.max_entries is not None and self.max_entries < 1:
            raise ValueError("max_entries must be None or a positive integer.")

    def get_or_create(
        self,
        key: Hashable,
        factory: Callable[[], PreparedT],
    ) -> PreparedT:
        """Return a cached artifact or create it from ``factory``."""
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        value = factory()
        self._entries[key] = value
        if self.max_entries is not None:
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        """Drop all derived artifacts without touching source scenes."""
        self._entries.clear()

    def reset(self) -> None:
        """Alias for ``clear`` for viewer reset flows."""
        self.clear()

    def __len__(self) -> int:
        """Return the number of cached artifacts."""
        return len(self._entries)


def config_cache_key(config: BaseModel | Mapping[str, Any] | str | None) -> str:
    """Return a stable JSON-ish cache key for viewer prep config."""
    if config is None:
        return "{}"
    if isinstance(config, str):
        return config
    if isinstance(config, BaseModel):
        return config.model_dump_json()
    return json.dumps(config, sort_keys=True, default=str)


def viewer_prep_key(
    scene: object | None,
    config: BaseModel | Mapping[str, Any] | str | None,
    *,
    external_revision: Hashable = 0,
    stage: str = "scene",
) -> tuple[str, int | None, Hashable, str]:
    """Build a standard key for camera-independent viewer prep artifacts."""
    return (
        stage,
        None if scene is None else id(scene),
        external_revision,
        config_cache_key(config),
    )


def filter_gaussian_scene(
    scene: GaussianScene,
    keep_mask: Bool[Tensor, " num_splats"],
) -> GaussianScene:
    """Return a filtered Gaussian scene without mutating ``scene``."""
    mask = keep_mask.to(device=scene.center_position.device, dtype=torch.bool)
    return scene.with_fields(
        center_position=scene.center_position[mask],
        log_scales=scene.log_scales[mask],
        quaternion_orientation=scene.quaternion_orientation[mask],
        logit_opacity=scene.logit_opacity[mask],
        feature=scene.feature[mask],
    )


def replace_gaussian_features(
    scene: GaussianScene,
    feature: Float[Tensor, " num_splats feature_dim"]
    | Float[Tensor, " num_splats sh_coeffs 3"],
    *,
    sh_degree: int | None = None,
) -> GaussianScene:
    """Return a Gaussian scene with replaced features and shared geometry."""
    return scene.with_fields(
        feature=feature,
        sh_degree=scene.sh_degree if sh_degree is None else sh_degree,
    )


__all__ = [
    "ViewerPrepCache",
    "ViewerRenderResult",
    "config_cache_key",
    "filter_gaussian_scene",
    "replace_gaussian_features",
    "viewer_prep_key",
]
