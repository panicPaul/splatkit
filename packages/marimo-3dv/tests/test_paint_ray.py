"""Tests for the paint_ray overlay op."""

import numpy as np

from marimo_3dv.ops.overlay import (
    PaintRayConfig,
    _PaintRayState,
    _pixel_to_world_ray,
    _project_ray_to_pixel,
    paint_ray_op,
)
from marimo_3dv.viewer.widget import CameraState


def _forward_camera(width: int = 64, height: int = 48) -> CameraState:
    """Camera looking straight down +Z from z=3."""
    return CameraState.default(
        width=width, height=height, camera_convention="opencv"
    )


def test_pixel_to_world_ray_center_pixel():
    """Center pixel ray should point roughly along camera forward."""
    cam = _forward_camera()
    ray = _pixel_to_world_ray(cam.width // 2, cam.height // 2, cam)
    # Forward in the default camera is the +Z column of cam_to_world
    forward = cam.cam_to_world[:3, 2]
    dot = float(np.dot(ray.direction, forward))
    assert dot > 0.99, f"Center ray should align with forward, dot={dot}"


def test_pixel_to_world_ray_is_unit_length():
    cam = _forward_camera()
    ray = _pixel_to_world_ray(10, 10, cam)
    length = float(np.linalg.norm(ray.direction))
    assert abs(length - 1.0) < 1e-6


def test_pixel_to_world_ray_origin_is_camera_position():
    cam = _forward_camera()
    ray = _pixel_to_world_ray(0, 0, cam)
    expected_origin = cam.cam_to_world[:3, 3]
    assert np.allclose(ray.origin, expected_origin)


def test_project_ray_back_to_approx_same_pixel():
    """A ray shot from pixel (px, py) should project back near (px, py)."""
    cam = _forward_camera(width=128, height=96)
    px, py = 40, 30
    ray = _pixel_to_world_ray(px, py, cam)
    projected = _project_ray_to_pixel(ray, cam)
    assert projected is not None
    assert abs(projected[0] - px) <= 1
    assert abs(projected[1] - py) <= 1


def test_paint_ray_op_returns_effect_node():
    from marimo_3dv.pipeline.gui import EffectNode

    op = paint_ray_op()
    assert isinstance(op, EffectNode)
    assert op.name == "paint_ray"


def test_paint_ray_stores_ray_on_click():
    from marimo_3dv.pipeline.context import ViewerContext
    from marimo_3dv.pipeline.gui import RenderResult
    from marimo_3dv.viewer.widget import ViewerClick, ViewerState

    cam = _forward_camera()
    click = ViewerClick(
        x=32, y=24, width=cam.width, height=cam.height, camera_state=cam
    )

    viewer_state = ViewerState()
    viewer_state.last_click = click
    context = ViewerContext(viewer_state=viewer_state, last_click=click)

    runtime_state = _PaintRayState()
    config = PaintRayConfig()
    image = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
    result = RenderResult(image=image)

    op = paint_ray_op()
    op.apply(result, config, context, runtime_state)

    assert len(runtime_state.rays) == 1


def test_paint_ray_respects_max_rays():
    from marimo_3dv.pipeline.context import ViewerContext
    from marimo_3dv.pipeline.gui import RenderResult
    from marimo_3dv.viewer.widget import ViewerClick, ViewerState

    cam = _forward_camera()
    runtime_state = _PaintRayState()
    config = PaintRayConfig(paint_ray_max_rays=3)
    image = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)

    for i in range(5):
        click = ViewerClick(
            x=i % cam.width,
            y=0,
            width=cam.width,
            height=cam.height,
            camera_state=cam,
        )
        viewer_state = ViewerState()
        viewer_state.last_click = click
        context = ViewerContext(viewer_state=viewer_state, last_click=click)
        result = RenderResult(image=image.copy())
        paint_ray_op().apply(result, config, context, runtime_state)

    assert len(runtime_state.rays) == 3


def test_paint_ray_disabled_returns_unchanged():
    from marimo_3dv.pipeline.context import ViewerContext
    from marimo_3dv.pipeline.gui import RenderResult
    from marimo_3dv.viewer.widget import ViewerClick, ViewerState

    cam = _forward_camera()
    click = ViewerClick(
        x=0, y=0, width=cam.width, height=cam.height, camera_state=cam
    )
    viewer_state = ViewerState()
    viewer_state.last_click = click
    context = ViewerContext(viewer_state=viewer_state, last_click=click)
    runtime_state = _PaintRayState()
    config = PaintRayConfig(paint_ray_enabled=False)
    image = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
    result = RenderResult(image=image)

    returned = paint_ray_op().apply(result, config, context, runtime_state)
    assert returned is result
    assert len(runtime_state.rays) == 0
