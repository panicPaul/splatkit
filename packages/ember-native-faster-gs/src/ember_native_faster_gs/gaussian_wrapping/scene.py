"""Gaussian Wrapping scene extensions."""

from __future__ import annotations

from typing import ClassVar

import torch
from ember_core.core.contracts import GaussianScene3D
from jaxtyping import Float
from torch import Tensor, nn


class GaussianWrappingScene(GaussianScene3D):
    """3D Gaussian scene with wrapping-specific per-splat fields."""

    parameter_field_names: ClassVar[tuple[str, ...]] = (
        GaussianScene3D.parameter_field_names
        + ("normal_features", "occupancy_shift")
    )
    buffer_field_names: ClassVar[tuple[str, ...]] = ("base_occupancy",)
    metadata_field_names: ClassVar[tuple[str, ...]] = (
        GaussianScene3D.metadata_field_names
        + ("normal_feature_dim", "num_pivots")
    )
    topology_field_names: ClassVar[tuple[str, ...]] = (
        parameter_field_names + buffer_field_names
    )

    def __init__(
        self,
        *,
        center_position: Float[Tensor, "num_splats 3"],
        log_scales: Float[Tensor, "num_splats 3"],
        quaternion_orientation: Float[Tensor, "num_splats 4"],
        logit_opacity: Float[Tensor, " num_splats"],
        feature: (
            Float[Tensor, "num_splats feature_dim"]
            | Float[Tensor, "num_splats sh_coeffs 3"]
        ),
        sh_degree: int,
        normal_features: Float[Tensor, "num_splats normal_feature_dim"]
        | None = None,
        base_occupancy: Float[Tensor, "num_splats num_pivots"] | None = None,
        occupancy_shift: Float[Tensor, "num_splats num_pivots"] | None = None,
        normal_feature_dim: int = 3,
        num_pivots: int = 9,
    ) -> None:
        self.normal_feature_dim = normal_feature_dim
        self.num_pivots = num_pivots
        super().__init__(
            center_position=center_position,
            log_scales=log_scales,
            quaternion_orientation=quaternion_orientation,
            logit_opacity=logit_opacity,
            feature=feature,
            sh_degree=sh_degree,
        )
        num_splats = int(center_position.shape[0])
        if normal_features is None:
            normal_features = torch.zeros(
                (num_splats, normal_feature_dim),
                dtype=center_position.dtype,
                device=center_position.device,
            )
        if base_occupancy is None:
            base_occupancy = torch.zeros(
                (num_splats, num_pivots),
                dtype=center_position.dtype,
                device=center_position.device,
            )
        if occupancy_shift is None:
            occupancy_shift = torch.zeros_like(base_occupancy)

        self.register_parameter(
            "normal_features",
            self._to_parameter(normal_features),
        )
        self.register_buffer("base_occupancy", base_occupancy)
        self.register_parameter(
            "occupancy_shift",
            self._to_parameter(occupancy_shift),
        )
        self._validate()

    @staticmethod
    def _to_parameter(value: Tensor) -> nn.Parameter:
        if isinstance(value, nn.Parameter):
            return value
        return nn.Parameter(value, requires_grad=value.requires_grad)

    @property
    def occupancy_logits(self) -> Float[Tensor, "num_splats num_pivots"]:
        """Return the learned wrapping occupancy logits."""
        return self.base_occupancy + self.occupancy_shift

    @property
    def occupancy(self) -> Float[Tensor, "num_splats num_pivots"]:
        """Return pivot occupancy probabilities."""
        return torch.sigmoid(self.occupancy_logits)

    @property
    def sdf(self) -> Float[Tensor, "num_splats num_pivots"]:
        """Return the upstream-style negative occupancy logits."""
        return -self.occupancy_logits

    def _validate(self) -> None:
        super()._validate()
        if self.normal_feature_dim <= 0:
            raise ValueError(
                "GaussianWrappingScene.normal_feature_dim must be positive."
            )
        if self.num_pivots <= 0:
            raise ValueError(
                "GaussianWrappingScene.num_pivots must be positive."
            )

        normal_features = getattr(self, "normal_features", None)
        base_occupancy = getattr(self, "base_occupancy", None)
        occupancy_shift = getattr(self, "occupancy_shift", None)
        if (
            normal_features is None
            or base_occupancy is None
            or occupancy_shift is None
        ):
            return

        num_splats = int(self.center_position.shape[0])
        expected_normal_shape = (num_splats, self.normal_feature_dim)
        if tuple(normal_features.shape) != expected_normal_shape:
            raise ValueError(
                "GaussianWrappingScene.normal_features must have shape "
                f"{expected_normal_shape}; got {tuple(normal_features.shape)}."
            )
        expected_occupancy_shape = (num_splats, self.num_pivots)
        if tuple(base_occupancy.shape) != expected_occupancy_shape:
            raise ValueError(
                "GaussianWrappingScene.base_occupancy must have shape "
                f"{expected_occupancy_shape}; got {tuple(base_occupancy.shape)}."
            )
        if tuple(occupancy_shift.shape) != expected_occupancy_shape:
            raise ValueError(
                "GaussianWrappingScene.occupancy_shift must have shape "
                f"{expected_occupancy_shape}; got {tuple(occupancy_shift.shape)}."
            )
