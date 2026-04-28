"""Tests for the marimo-3dv public package surface."""

import warnings

import pytest


def test_core_imports_work():
    from marimo_3dv import (
        CameraState,
        LinkedViewerStateField,
        MarimoViewer,
        Viewer,
        ViewerBackendBundle,
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
        apply_viewer_pipeline_config,
        backend_bundle,
        cleanup_before_splat_reload,
        gs_backend_bundle,
        link_viewer_states,
        load_splat_scene,
        load_splat_scene_from_config,
        pick_splat_load_config,
        splat_load_form,
        viewer_controls_config,
        viewer_controls_gui,
        viewer_controls_handle,
        viewer_pipeline_controls_gui,
        viewer_pipeline_controls_handle,
    )

    assert CameraState is not None
    assert LinkedViewerStateField is not None
    assert MarimoViewer is not None
    assert Viewer is not None
    assert ViewerBackendBundle is not None
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
    assert apply_viewer_pipeline_config is not None
    assert backend_bundle is not None
    assert cleanup_before_splat_reload is not None
    assert gs_backend_bundle is not None
    assert link_viewer_states is not None
    assert load_splat_scene is not None
    assert load_splat_scene_from_config is not None
    assert pick_splat_load_config is not None
    assert splat_load_form is not None
    assert viewer_controls_config is not None
    assert viewer_controls_handle is not None
    assert viewer_controls_gui is not None
    assert viewer_pipeline_controls_handle is not None
    assert viewer_pipeline_controls_gui is not None


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


def test_viewer_runs_desktop_backend_outside_notebook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marimo_3dv.viewer as viewer_module

    class _StubDesktopViewer:
        def __init__(self) -> None:
            self.ran = False

        def run(self) -> None:
            self.ran = True

    stub = _StubDesktopViewer()
    monkeypatch.setattr(viewer_module.mo, "running_in_notebook", lambda: False)
    monkeypatch.setattr(
        viewer_module,
        "desktop_viewer",
        lambda render_fn, state=None, controls=None, width=1280, height=720, title="": (
            stub
        ),
    )

    viewer = viewer_module.Viewer(lambda camera: camera)

    assert viewer is stub
    assert stub.ran is True


def test_pipeline_imports_work():
    from marimo_3dv import (
        AbstractRenderView,
        EffectNode,
        PipelineGroup,
        RenderNode,
        RenderResult,
        SetupPipeline,
        ViewerContext,
        ViewerPipeline,
        ViewerPipelineResult,
        effect_node,
        render_node,
    )

    assert AbstractRenderView is not None
    assert EffectNode is not None
    assert PipelineGroup is not None
    assert RenderNode is not None
    assert RenderResult is not None
    assert SetupPipeline is not None
    assert ViewerPipeline is not None
    assert ViewerPipelineResult is not None
    assert ViewerContext is not None
    assert effect_node is not None
    assert render_node is not None


def test_cleanup_before_splat_reload_ignores_cuda_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marimo_3dv.ops.gs as gs_module
    import marimo_3dv.viewer.widget as widget_module
    from marimo_3dv import ViewerState, cleanup_before_splat_reload

    monkeypatch.setattr(gs_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(widget_module.torch.cuda, "is_available", lambda: True)

    def _raise() -> None:
        raise RuntimeError("unspecified launch failure")

    monkeypatch.setattr(widget_module.torch.cuda, "empty_cache", _raise)
    monkeypatch.setattr(
        widget_module.torch.cuda,
        "ipc_collect",
        lambda: pytest.fail(
            "ipc_collect should not run after empty_cache fails"
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cleanup_before_splat_reload(
            ViewerState(),
            close_existing_viewer=True,
            empty_cuda_cache=True,
        )

    assert caught
    assert "CUDA cleanup during viewer teardown failed" in str(
        caught[0].message
    )


def test_gs_pipe_imports_work():
    from marimo_3dv import (
        filter_opacity_op,
        filter_size_op,
        max_sh_degree_op,
        paint_ray_op,
        show_distribution_op,
    )

    assert filter_opacity_op is not None
    assert filter_size_op is not None
    assert max_sh_degree_op is not None
    assert paint_ray_op is not None
    assert show_distribution_op is not None


def test_gui_helpers_are_not_reexported() -> None:
    import marimo_3dv

    assert not hasattr(marimo_3dv, "form_gui")
    assert not hasattr(marimo_3dv, "json_gui")
    assert not hasattr(marimo_3dv, "config_gui")


def test_backend_specific_viewer_imports_are_internal_only():
    from marimo_3dv.viewer.desktop import DesktopViewer, desktop_viewer

    assert DesktopViewer is not None
    assert desktop_viewer is not None


def test_viser_exports_removed():
    import marimo_3dv

    assert not hasattr(marimo_3dv, "ViserMarimoWidget")
    assert not hasattr(marimo_3dv, "viser_marimo")
    assert not hasattr(marimo_3dv, "ViserCameraState")

    with pytest.raises(ImportError):
        from marimo_3dv import ViserMarimoWidget  # noqa: F401

    with pytest.raises(ImportError):
        from marimo_3dv import viser_marimo  # noqa: F401
