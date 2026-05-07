"""Shared helpers for native NHT runtime ops."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from ember_native_nht.native_build.nht_rasterizer import (
    load_nht_rasterizer_runtime,
)

CameraModelName = Literal["pinhole", "ortho", "fisheye", "ftheta"]


@lru_cache(maxsize=1)
def backend() -> Any:
    """Return the staged vendored NHT rasterizer module."""
    return load_nht_rasterizer_runtime().module


def camera_model_type(camera_model: CameraModelName) -> Any:
    """Map a readable camera-model name to the native enum value."""
    return getattr(backend().CameraModelType, camera_model.upper())


def global_shutter_type() -> Any:
    """Return the native enum for global shutter rendering."""
    return backend().ShutterType.GLOBAL


def default_unscented_transform_parameters() -> Any:
    """Return native 3DGUT unscented-transform defaults."""
    parameters = backend().UnscentedTransformParameters()
    parameters.alpha = 0.1
    parameters.beta = 2.0
    parameters.kappa = 0.0
    parameters.in_image_margin_factor = 0.1
    parameters.require_all_sigma_points_valid = True
    return parameters


def default_ftheta_distortion_parameters() -> Any:
    """Return native default f-theta distortion parameters."""
    return backend().FThetaCameraDistortionParameters()


def encoding_expansion_factor() -> int:
    """Return the NHT harmonic encoding expansion factor."""
    return int(backend().encoding_expansion_factor)


def feature_divisor() -> int:
    """Return the number of tetrahedron vertices used per primitive."""
    return int(backend().feature_divisor)
