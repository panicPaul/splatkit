"""Reusable viewer default controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import marimo as mo
from pydantic import BaseModel, Field, create_model

from marimo_config_gui import form_gui
from marimo_3dv.viewer.widget import ViewerState

if TYPE_CHECKING:
    from marimo_3dv.pipeline.gui import ViewerPipelineResult
    from marimo_3dv.viewer.controls import DesktopPydanticControls

PipelineConfigT = TypeVar("PipelineConfigT", bound=BaseModel)


class ViewerOriginConfig(BaseModel):
    """Viewer origin marker position."""

    x: float = Field(default=0.0)
    y: float = Field(default=0.0)
    z: float = Field(default=0.0)


class ViewerRotationConfig(BaseModel):
    """Viewer-frame rotation in degrees."""

    x_degrees: float = Field(default=0.0, ge=-180.0, le=180.0)
    y_degrees: float = Field(default=0.0, ge=-180.0, le=180.0)
    z_degrees: float = Field(default=0.0, ge=-180.0, le=180.0)


class ViewerCameraConfig(BaseModel):
    """Camera settings."""

    fov_degrees: float = Field(default=60.0, gt=0.0, lt=180.0)


class ViewerOverlayConfig(BaseModel):
    """Overlay visibility settings."""

    show_axes: bool = Field(default=True)
    show_horizon: bool = Field(default=False)
    show_origin: bool = Field(default=False)
    show_stats: bool = Field(default=True)


class ViewerRenderConfig(BaseModel):
    """Render quality settings."""

    interactive_quality: int = Field(default=50, ge=1, le=100)
    settled_quality: Literal["jpeg_95", "jpeg_100", "png"] = Field(
        default="jpeg_100"
    )
    interactive_max_side: int = Field(default=1980, ge=1)
    internal_render_max_side: int = Field(default=3840, ge=1)


class ViewerNavigationConfig(BaseModel):
    """Keyboard navigation tuning."""

    move_speed: float = Field(default=0.125, gt=0.0)
    sprint_multiplier: float = Field(default=4.0, ge=1.0)


class ViewerInteractionConfig(BaseModel):
    """Pointer interaction tuning."""

    orbit_invert_x: bool = Field(default=False)
    orbit_invert_y: bool = Field(default=False)
    pan_invert_x: bool = Field(default=False)
    pan_invert_y: bool = Field(default=False)


class ViewerTransformConfig(BaseModel):
    """Viewer-frame transform defaults."""

    rotation: ViewerRotationConfig = Field(default_factory=ViewerRotationConfig)
    origin: ViewerOriginConfig = Field(default_factory=ViewerOriginConfig)


class ViewerControlsConfig(BaseModel):
    """Reusable viewer controls schema."""

    camera: ViewerCameraConfig = Field(default_factory=ViewerCameraConfig)
    overlays: ViewerOverlayConfig = Field(default_factory=ViewerOverlayConfig)
    render: ViewerRenderConfig = Field(default_factory=ViewerRenderConfig)
    navigation: ViewerNavigationConfig = Field(
        default_factory=ViewerNavigationConfig
    )
    interaction: ViewerInteractionConfig = Field(
        default_factory=ViewerInteractionConfig
    )
    transform: ViewerTransformConfig = Field(
        default_factory=ViewerTransformConfig
    )


@dataclass(frozen=True)
class ViewerControlsHandle:
    """Notebook-ready viewer controls block."""

    config_model: type[ViewerControlsConfig]
    default_config: ViewerControlsConfig
    gui: Any

    @property
    def value(self) -> ViewerControlsConfig:
        """Return the latest controls value."""
        value = getattr(self.gui, "value", None)
        if value is None:
            return self.default_config
        return value


@dataclass(frozen=True)
class CombinedViewerPipelineControlsHandle(Generic[PipelineConfigT]):
    """Notebook-ready combined viewer and pipeline controls block."""

    config_model: type[BaseModel]
    default_config: BaseModel
    gui: Any
    pipeline_config_model: type[PipelineConfigT]
    pipeline_default_config: PipelineConfigT

    @property
    def value(self) -> BaseModel:
        """Return the latest combined controls value."""
        value = getattr(self.gui, "value", None)
        if value is None:
            return self.default_config
        return value


def viewer_controls_config(
    viewer_state: ViewerState,
) -> ViewerControlsConfig:
    """Return viewer controls config populated from a ViewerState."""
    return ViewerControlsConfig(
        camera=ViewerCameraConfig(
            fov_degrees=viewer_state.camera_state.fov_degrees,
        ),
        overlays=ViewerOverlayConfig(
            show_axes=viewer_state.show_axes,
            show_horizon=viewer_state.show_horizon,
            show_origin=viewer_state.show_origin,
            show_stats=viewer_state.show_stats,
        ),
        render=ViewerRenderConfig(
            interactive_quality=viewer_state.interactive_quality,
            settled_quality=viewer_state.settled_quality,
            interactive_max_side=viewer_state.interactive_max_side or 1980,
            internal_render_max_side=(
                viewer_state.internal_render_max_side or 3840
            ),
        ),
        navigation=ViewerNavigationConfig(
            move_speed=viewer_state.keyboard_move_speed,
            sprint_multiplier=viewer_state.keyboard_sprint_multiplier,
        ),
        interaction=ViewerInteractionConfig(
            orbit_invert_x=viewer_state.orbit_invert_x,
            orbit_invert_y=viewer_state.orbit_invert_y,
            pan_invert_x=viewer_state.pan_invert_x,
            pan_invert_y=viewer_state.pan_invert_y,
        ),
        transform=ViewerTransformConfig(
            rotation=ViewerRotationConfig(
                x_degrees=viewer_state.viewer_rotation_x_degrees,
                y_degrees=viewer_state.viewer_rotation_y_degrees,
                z_degrees=viewer_state.viewer_rotation_z_degrees,
            ),
            origin=ViewerOriginConfig(
                x=viewer_state.origin[0],
                y=viewer_state.origin[1],
                z=viewer_state.origin[2],
            ),
        ),
    )


def apply_viewer_config(
    viewer_state: ViewerState,
    config: ViewerControlsConfig,
) -> ViewerState:
    """Apply reusable viewer controls config onto a ViewerState."""
    viewer_state.set_fov_degrees(
        config.camera.fov_degrees,
        push_to_viewer=False,
    )
    viewer_state.interactive_quality = config.render.interactive_quality
    viewer_state.settled_quality = config.render.settled_quality
    viewer_state.interactive_max_side = config.render.interactive_max_side
    viewer_state.internal_render_max_side = (
        config.render.internal_render_max_side
    )
    return (
        viewer_state.set_show_axes(config.overlays.show_axes)
        .set_show_horizon(config.overlays.show_horizon)
        .set_show_origin(config.overlays.show_origin)
        .set_show_stats(config.overlays.show_stats)
        .set_keyboard_navigation(
            config.navigation.move_speed,
            config.navigation.sprint_multiplier,
        )
        .set_pointer_controls(
            config.interaction.orbit_invert_x,
            config.interaction.orbit_invert_y,
            config.interaction.pan_invert_x,
            config.interaction.pan_invert_y,
        )
        .set_viewer_rotation(
            config.transform.rotation.x_degrees,
            config.transform.rotation.y_degrees,
            config.transform.rotation.z_degrees,
        )
        .set_origin(
            config.transform.origin.x,
            config.transform.origin.y,
            config.transform.origin.z,
        )
    )


def viewer_controls_handle(
    viewer_state: ViewerState,
    *,
    label: str = "",
    default_config: ViewerControlsConfig | None = None,
) -> ViewerControlsHandle:
    """Build a live controls handle for the default viewer controls."""
    resolved_default_config = default_config or viewer_controls_config(
        viewer_state
    )
    if mo.running_in_notebook():
        gui = form_gui(
            ViewerControlsConfig,
            value=resolved_default_config,
            label=label,
            live_update=True,
        )
    else:
        from marimo_3dv.viewer.controls import DesktopPydanticControls

        gui = DesktopPydanticControls(
            ViewerControlsConfig,
            value=resolved_default_config,
            label=label,
        )
    return ViewerControlsHandle(
        config_model=ViewerControlsConfig,
        default_config=resolved_default_config,
        gui=gui,
    )
def viewer_controls_gui(
    viewer_state: ViewerState,
    *,
    label: str = "",
    default_config: ViewerControlsConfig | None = None,
) -> ViewerControlsHandle:
    """Build a live controls handle for the default viewer controls."""
    return viewer_controls_handle(
        viewer_state,
        label=label,
        default_config=default_config,
    )


def _combined_viewer_pipeline_model(
    pipeline_config_model: type[PipelineConfigT],
) -> type[BaseModel]:
    return create_model(
        "ViewerPipelineControlsConfig",
        viewer=(
            ViewerControlsConfig,
            Field(default_factory=ViewerControlsConfig),
        ),
        pipeline=(
            pipeline_config_model,
            Field(default_factory=pipeline_config_model),
        ),
    )


def viewer_pipeline_controls_handle(
    viewer_state: ViewerState,
    pipeline_result: ViewerPipelineResult[Any, Any, Any],
    *,
    label: str = "",
    viewer_default_config: ViewerControlsConfig | None = None,
) -> CombinedViewerPipelineControlsHandle[Any]:
    """Build one live config tree containing viewer and pipeline controls."""
    pipeline_config_model = pipeline_result.config_model
    combined_model = _combined_viewer_pipeline_model(pipeline_config_model)
    resolved_viewer_default = viewer_default_config or viewer_controls_config(
        viewer_state
    )
    default_config = combined_model(
        viewer=resolved_viewer_default,
        pipeline=pipeline_result.default_config,
    )
    if mo.running_in_notebook():
        gui = form_gui(
            combined_model,
            value=default_config,
            label=label,
            live_update=True,
        )
    else:
        from marimo_3dv.viewer.controls import DesktopPydanticControls

        gui = DesktopPydanticControls(
            combined_model,
            value=default_config,
            label=label,
        )
    return CombinedViewerPipelineControlsHandle(
        config_model=combined_model,
        default_config=default_config,
        gui=gui,
        pipeline_config_model=pipeline_config_model,
        pipeline_default_config=pipeline_result.default_config,
    )


def viewer_pipeline_controls_gui(
    viewer_state: ViewerState,
    pipeline_result: ViewerPipelineResult[Any, Any, Any],
    *,
    label: str = "",
    viewer_default_config: ViewerControlsConfig | None = None,
) -> CombinedViewerPipelineControlsHandle[Any]:
    """Build one live config tree containing viewer and pipeline controls."""
    return viewer_pipeline_controls_handle(
        viewer_state,
        pipeline_result,
        label=label,
        viewer_default_config=viewer_default_config,
    )


def apply_viewer_pipeline_config(
    viewer_state: ViewerState,
    config: BaseModel,
) -> BaseModel:
    """Apply combined viewer config and return the pipeline config subtree."""
    apply_viewer_config(viewer_state, config.viewer)
    pipeline_config = config.pipeline
    if not isinstance(pipeline_config, BaseModel):
        raise TypeError(
            "Expected combined config to expose a BaseModel pipeline."
        )
    return pipeline_config


__all__ = [
    "CombinedViewerPipelineControlsHandle",
    "ViewerCameraConfig",
    "ViewerControlsConfig",
    "ViewerControlsHandle",
    "ViewerNavigationConfig",
    "ViewerOriginConfig",
    "ViewerOverlayConfig",
    "ViewerRenderConfig",
    "ViewerRotationConfig",
    "ViewerTransformConfig",
    "apply_viewer_config",
    "apply_viewer_pipeline_config",
    "viewer_controls_config",
    "viewer_controls_gui",
    "viewer_controls_handle",
    "viewer_pipeline_controls_gui",
    "viewer_pipeline_controls_handle",
]
