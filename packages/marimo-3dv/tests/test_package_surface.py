"""Tests for the marimo-3dv public package surface."""

import pytest


def test_core_imports_work() -> None:
    from marimo_3dv import (
        CameraState,
        LinkedViewerStateField,
        MarimoViewer,
        NoopViewer,
        Viewer,
        ViewerCameraConfig,
        ViewerClick,
        ViewerControlsConfig,
        ViewerInteractionConfig,
        ViewerNavigationConfig,
        ViewerOverlayConfig,
        ViewerRenderConfig,
        ViewerState,
        ViewerStateLink,
        ViewerTransformConfig,
        apply_viewer_config,
        link_viewer_states,
        marimo_viewer,
        viewer_controls_config,
        viewer_controls_gui,
        viewer_controls_handle,
    )

    assert CameraState is not None
    assert LinkedViewerStateField is not None
    assert MarimoViewer is not None
    assert NoopViewer is not None
    assert Viewer is not None
    assert ViewerCameraConfig is not None
    assert ViewerClick is not None
    assert ViewerControlsConfig is not None
    assert ViewerInteractionConfig is not None
    assert ViewerNavigationConfig is not None
    assert ViewerOverlayConfig is not None
    assert ViewerRenderConfig is not None
    assert ViewerState is not None
    assert ViewerStateLink is not None
    assert ViewerTransformConfig is not None
    assert apply_viewer_config is not None
    assert link_viewer_states is not None
    assert marimo_viewer is not None
    assert viewer_controls_config is not None
    assert viewer_controls_handle is not None
    assert viewer_controls_gui is not None


def test_viewer_uses_marimo_backend_in_notebook_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marimo_3dv.viewer as viewer_module

    sentinel = object()
    monkeypatch.setattr(viewer_module.mo, "running_in_notebook", lambda: True)
    monkeypatch.setattr(
        viewer_module,
        "marimo_viewer",
        lambda render_fn, state=None: sentinel,
    )

    viewer = viewer_module.Viewer(lambda camera: camera)

    assert viewer is sentinel


def test_viewer_returns_noop_outside_notebook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marimo_3dv.viewer as viewer_module

    monkeypatch.setattr(viewer_module.mo, "running_in_notebook", lambda: False)

    viewer = viewer_module.Viewer(lambda camera: camera)

    assert isinstance(viewer, viewer_module.NoopViewer)


def test_removed_pipeline_and_desktop_exports_are_not_public() -> None:
    import marimo_3dv

    removed_names = [
        "ViewerPipeline",
        "ViewerPipelineResult",
        "RenderNode",
        "EffectNode",
        "RenderResult",
        "ViewerBackendBundle",
        "desktop_viewer",
        "gs_backend_bundle",
        "cleanup_before_splat_reload",
        "viewer_pipeline_controls_gui",
    ]

    for name in removed_names:
        assert not hasattr(marimo_3dv, name)


def test_gui_helpers_are_not_reexported() -> None:
    import marimo_3dv

    assert not hasattr(marimo_3dv, "form_gui")
    assert not hasattr(marimo_3dv, "json_gui")
    assert not hasattr(marimo_3dv, "config_gui")


def test_viser_exports_removed() -> None:
    import marimo_3dv

    assert not hasattr(marimo_3dv, "ViserMarimoWidget")
    assert not hasattr(marimo_3dv, "viser_marimo")
    assert not hasattr(marimo_3dv, "ViserCameraState")

    with pytest.raises(ImportError):
        from marimo_3dv import ViserMarimoWidget  # noqa: F401

    with pytest.raises(ImportError):
        from marimo_3dv import viser_marimo  # noqa: F401
