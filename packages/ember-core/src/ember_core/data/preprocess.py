"""Image decoding and preparation helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, UInt8
from torch import Tensor
from torch.nn import functional as F

from ember_core.core.contracts import CameraState
from ember_core.data.contracts import (
    ImagePreparationSpec,
    ResizeSpec,
    horizontal_fov_degrees,
)


def _require_pillow() -> object:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Image preparation requires Pillow. Install ember-core[data]."
        ) from exc
    return Image


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an RGB image as an HWC uint8 NumPy array."""
    image_module = _require_pillow()
    image = image_module.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def resolve_resize_shape(
    original_width: int,
    original_height: int,
    resize: ResizeSpec | None,
) -> tuple[int, int]:
    """Resolve the output image size for a resize spec."""
    if resize is None:
        return original_width, original_height
    if resize.width_target is not None:
        scale = resize.width_target / float(original_width)
    else:
        assert resize.width_scale is not None
        scale = resize.width_scale
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    return resized_width, resized_height


def resize_intrinsics(
    intrinsics: Float[Tensor, " 3 3"],
    original_width: int,
    original_height: int,
    resized_width: int,
    resized_height: int,
) -> Float[Tensor, " 3 3"]:
    """Resize a pinhole intrinsics matrix."""
    scale_x = resized_width / float(original_width)
    scale_y = resized_height / float(original_height)
    scaled_intrinsics = intrinsics.clone()
    scaled_intrinsics[0, 0] *= scale_x
    scaled_intrinsics[0, 2] *= scale_x
    scaled_intrinsics[1, 1] *= scale_y
    scaled_intrinsics[1, 2] *= scale_y
    return scaled_intrinsics


def _resize_image(
    image: np.ndarray,
    resized_width: int,
    resized_height: int,
    interpolation: str,
) -> np.ndarray:
    image_tensor = torch.from_numpy(np.array(image, copy=False))
    resized = _resize_image_tensor(
        image_tensor,
        resized_width,
        resized_height,
        interpolation,
    )
    return resized.cpu().numpy()


def _resize_image_tensor(
    image: UInt8[Tensor, "height width 3"],
    resized_width: int,
    resized_height: int,
    interpolation: str,
) -> UInt8[Tensor, "height width 3"]:
    image_nchw = image.permute(2, 0, 1)[None].to(torch.float32)
    resized = F.interpolate(
        image_nchw,
        size=(resized_height, resized_width),
        mode=interpolation,
        align_corners=(
            False if interpolation in {"bilinear", "bicubic"} else None
        ),
        antialias=interpolation in {"bilinear", "bicubic"},
    )
    resized_hwc = resized[0].permute(1, 2, 0)
    return torch.clamp(torch.round(resized_hwc), 0.0, 255.0).to(torch.uint8)


def prepare_image_and_camera(
    image_path: str | Path,
    camera: CameraState,
    preparation: ImagePreparationSpec,
) -> tuple[Float[Tensor, "height width 3"], CameraState]:
    """Load an image and update the single-frame camera for preprocessing."""
    image = torch.from_numpy(np.array(load_image_rgb(image_path), copy=True))
    return prepare_decoded_image_and_camera(
        image,
        camera,
        preparation,
    )


def prepare_decoded_image_and_camera(
    image: UInt8[Tensor, "height width 3"],
    camera: CameraState,
    preparation: ImagePreparationSpec,
) -> tuple[Float[Tensor, "height width 3"], CameraState]:
    """Prepare an already-decoded image tensor and camera."""
    image_array = image.cpu().numpy()
    original_height, original_width = image_array.shape[:2]
    resized_width, resized_height = resolve_resize_shape(
        original_width,
        original_height,
        preparation.resize,
    )
    if (resized_width, resized_height) != (original_width, original_height):
        resize = preparation.resize
        assert resize is not None
        image = _resize_image_tensor(
            image,
            resized_width,
            resized_height,
            interpolation=resize.interpolation,
        )
        image_array = image.cpu().numpy()

    intrinsics = camera.get_intrinsics()[0]
    resized_intrinsics = resize_intrinsics(
        intrinsics,
        original_width=original_width,
        original_height=original_height,
        resized_width=resized_width,
        resized_height=resized_height,
    )
    resized_camera = replace(
        camera,
        width=torch.tensor([resized_width], dtype=torch.int64),
        height=torch.tensor([resized_height], dtype=torch.int64),
        fov_degrees=torch.tensor(
            [horizontal_fov_degrees(resized_width, resized_intrinsics)],
            dtype=torch.float32,
        ),
        intrinsics=resized_intrinsics[None],
    )
    image_tensor = torch.from_numpy(np.array(image_array, copy=True))
    image_tensor = image_tensor.to(torch.float32)
    if preparation.normalize:
        image_tensor = image_tensor / 255.0
    return image_tensor, resized_camera
