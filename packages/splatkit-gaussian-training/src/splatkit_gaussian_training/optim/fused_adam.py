"""Reusable fused Adam optimizer for Gaussian training."""

from FasterGSCudaBackend import FusedAdam as _UpstreamFusedAdam


class FusedAdam(_UpstreamFusedAdam):
    """Thin package-local alias for the vendored FasterGS fused Adam."""


__all__ = ["FusedAdam"]
