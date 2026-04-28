"""Backend-independent image overlay pipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from marimo_3dv.pipeline.context import ViewerContext
from marimo_3dv.pipeline.gui import EffectNode, RenderResult, effect_node
from marimo_3dv.viewer.widget import CameraState


@dataclass
class _Ray:
    """A world-space ray stored by paint_ray."""

    origin: np.ndarray  # (3,)
    direction: np.ndarray  # (3,) unit vector


@dataclass
class _PaintRayState:
    """Mutable runtime state for the paint_ray op."""

    rays: list[_Ray] = field(default_factory=list)


class PaintRayConfig(BaseModel):
    """Configuration for the paint_ray overlay op."""

    paint_ray_enabled: bool = Field(
        default=True, description="Enable the paint-ray overlay."
    )
    paint_ray_color: tuple[int, int, int] = Field(
        default=(255, 80, 0),
        description="RGB color of painted rays.",
    )
    paint_ray_radius: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Radius in pixels of each ray endpoint dot.",
    )
    paint_ray_max_rays: int = Field(
        default=64,
        ge=1,
        description="Maximum number of stored rays before oldest are dropped.",
    )


def _pixel_to_world_ray(
    x: int,
    y: int,
    camera_state: CameraState,
) -> _Ray:
    """Compute the world-space ray through pixel (x, y).

    Args:
        x: Pixel column (0-indexed, left to right).
        y: Pixel row (0-indexed, top to bottom).
        camera_state: Camera state in any supported convention.

    Returns:
        A ``_Ray`` with world-space origin and unit direction.
    """
    cam = camera_state.with_convention("opencv")
    width = cam.width
    height = cam.height
    focal = (height / 2.0) / np.tan(np.radians(cam.fov_degrees / 2.0))
    cx, cy = width / 2.0, height / 2.0

    # Camera-space direction (OpenCV: +X right, +Y down, +Z forward)
    cam_dir = np.array(
        [(x + 0.5 - cx) / focal, (y + 0.5 - cy) / focal, 1.0],
        dtype=np.float64,
    )
    cam_dir /= np.linalg.norm(cam_dir)

    c2w = cam.cam_to_world
    world_dir = c2w[:3, :3] @ cam_dir
    world_dir /= np.linalg.norm(world_dir)
    origin = c2w[:3, 3].copy()
    return _Ray(origin=origin, direction=world_dir)


def _project_ray_to_pixel(
    ray: _Ray,
    camera_state: CameraState,
    depth: float = 1.0,
) -> tuple[int, int] | None:
    """Project a point along a world-space ray onto the image plane.

    Projects the point ``ray.origin + depth * ray.direction`` into screen
    space. Returns pixel coordinates (x, y) or None if the point is behind
    the camera or outside the image bounds.

    Args:
        ray: World-space ray (origin + unit direction).
        camera_state: Current camera state.
        depth: Distance along the ray to project (default 1.0).

    Returns:
        ``(x, y)`` pixel or ``None`` if not visible.
    """
    cam = camera_state.with_convention("opencv")
    c2w = cam.cam_to_world
    w2c = np.linalg.inv(c2w)

    # Project a point at `depth` along the ray into camera space.
    world_point = ray.origin + depth * ray.direction
    point_cam = w2c[:3, :3] @ world_point + w2c[:3, 3]
    if point_cam[2] <= 0.0:
        return None

    focal = (cam.height / 2.0) / np.tan(np.radians(cam.fov_degrees / 2.0))
    cx, cy = cam.width / 2.0, cam.height / 2.0

    x_px = int(point_cam[0] / point_cam[2] * focal + cx)
    y_px = int(point_cam[1] / point_cam[2] * focal + cy)

    if 0 <= x_px < cam.width and 0 <= y_px < cam.height:
        return x_px, y_px
    return None


def _paint_ray_hook(
    result: RenderResult,
    config: PaintRayConfig,
    context: ViewerContext,
    runtime_state: _PaintRayState,
) -> RenderResult:
    """image_overlay hook: store clicked rays and draw them on the image."""
    if not config.paint_ray_enabled:
        return result

    # Capture new ray from click if available.
    click = context.last_click
    if click is not None:
        new_ray = _pixel_to_world_ray(click.x, click.y, click.camera_state)
        runtime_state.rays.append(new_ray)
        # Drop oldest rays beyond the limit.
        while len(runtime_state.rays) > config.paint_ray_max_rays:
            runtime_state.rays.pop(0)

    if not runtime_state.rays:
        return result

    image = result.image.copy()
    color_bgr = (
        config.paint_ray_color[2],
        config.paint_ray_color[1],
        config.paint_ray_color[0],
    )
    camera_state = context.viewer_state.camera_state
    for ray in runtime_state.rays:
        pixel = _project_ray_to_pixel(ray, camera_state)
        if pixel is not None:
            cv2.circle(image, pixel, config.paint_ray_radius, color_bgr, -1)

    return RenderResult(image=image, metadata=result.metadata)


def paint_ray_op() -> EffectNode[Any, _PaintRayState]:
    """Return an effect node that stores clicked world-space rays and draws them.

    On each click, a world-space ray is computed from the clicked pixel and
    the current camera state. Rays are stored in runtime state and projected
    onto subsequent frames as dot overlays — without triggering any additional
    backend renders.

    Returns:
        An ``EffectNode`` configured for the post-render effect stage.
    """
    return effect_node(
        name="paint_ray",
        config_model=PaintRayConfig,
        default_config=PaintRayConfig(),
        apply=_paint_ray_hook,
        state_factory=_PaintRayState,
    )


__all__ = ["PaintRayConfig", "paint_ray_op"]
