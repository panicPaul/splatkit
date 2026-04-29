from __future__ import annotations

import pytest
import torch
from ember_core import (
    GaussianScene3D,
    ViewerPrepCache,
    ViewerStatsUpdateGate,
    filter_gaussian_scene,
    prepare_viewer_stats_series,
    replace_gaussian_features,
    viewer_prep_key,
)


def _scene() -> GaussianScene3D:
    return GaussianScene3D(
        center_position=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        log_scales=torch.zeros((4, 3), dtype=torch.float32),
        quaternion_orientation=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 4,
            dtype=torch.float32,
        ),
        logit_opacity=torch.arange(4, dtype=torch.float32),
        feature=torch.arange(12, dtype=torch.float32).reshape(4, 1, 3),
        sh_degree=0,
    )


def test_viewer_prep_cache_reuses_until_reset() -> None:
    cache = ViewerPrepCache()
    scene = _scene()
    key = viewer_prep_key(scene, {"threshold": 0.5})
    calls = 0

    def factory() -> object:
        nonlocal calls
        calls += 1
        return object()

    first = cache.get_or_create(key, factory)
    second = cache.get_or_create(key, factory)
    cache.reset()
    third = cache.get_or_create(key, factory)

    assert first is second
    assert third is not first
    assert calls == 2


def test_viewer_prep_cache_evicts_old_entries_when_bounded() -> None:
    cache = ViewerPrepCache(max_entries=2)
    calls = 0

    def factory() -> object:
        nonlocal calls
        calls += 1
        return object()

    first = cache.get_or_create("first", factory)
    second = cache.get_or_create("second", factory)
    cache.get_or_create("first", factory)
    third = cache.get_or_create("third", factory)
    second_again = cache.get_or_create("second", factory)

    assert len(cache) == 2
    assert list(cache._entries.keys()) == ["third", "second"]
    assert first not in cache._entries.values()
    assert third in cache._entries.values()
    assert second_again is not second
    assert calls == 4


def test_viewer_prep_cache_rejects_non_positive_bound() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        ViewerPrepCache(max_entries=0)


def test_filter_gaussian_scene_does_not_mutate_source() -> None:
    scene = _scene()
    original_center = scene.center_position.clone()
    original_feature = scene.feature.clone()
    filtered = filter_gaussian_scene(
        scene,
        torch.tensor([True, False, True, False]),
    )

    assert filtered is not scene
    assert filtered.center_position.shape == (2, 3)
    assert torch.equal(scene.center_position, original_center)
    assert torch.equal(scene.feature, original_feature)


def test_replace_gaussian_features_shares_geometry_without_mutating_source() -> None:
    scene = _scene()
    replacement = torch.ones((4, 1, 3), dtype=torch.float32)
    derived = replace_gaussian_features(scene, replacement, sh_degree=0)

    assert derived is not scene
    assert derived.center_position is scene.center_position
    assert derived.log_scales is scene.log_scales
    assert derived.feature is replacement
    assert scene.feature is not replacement


def test_prepare_viewer_stats_series_returns_bounded_histogram_rows() -> None:
    values = torch.arange(1000, dtype=torch.float32)
    series = prepare_viewer_stats_series(
        values,
        name="opacity",
        plot_kind="histogram",
        max_points=32,
    )

    assert series.name == "opacity"
    assert series.plot_kind == "histogram"
    assert len(series.rows) <= 32
    assert series.summary.total_count == 1000
    assert series.summary.selected_count == 1000
    assert series.summary.min_value == 0.0
    assert series.summary.max_value == 999.0


def test_prepare_viewer_stats_series_filters_without_mutating_source() -> None:
    values = torch.tensor([-1.0, 0.0, 1.0, 2.0, float("nan")])
    original = values.clone()
    series = prepare_viewer_stats_series(
        values,
        plot_kind="sorted_rank",
        max_points=4,
        positive_only=True,
        keep_mode="higher",
    )

    assert torch.allclose(values, original, equal_nan=True)
    assert series.summary.total_count == 5
    assert series.summary.finite_count == 4
    assert series.summary.selected_count == 2
    assert len(series.rows) <= 4
    assert series.rows[0]["value"] >= series.rows[-1]["value"]


def test_prepare_viewer_stats_series_bounds_top_count_without_full_payload() -> None:
    values = torch.arange(100, dtype=torch.float32)
    series = prepare_viewer_stats_series(
        values,
        plot_kind="histogram",
        max_points=8,
        keep_mode="higher",
        top_count=10,
    )

    assert series.summary.selected_count == 10
    assert series.summary.min_value == 90.0
    assert series.summary.max_value == 99.0
    assert len(series.rows) <= 8


def test_viewer_stats_update_gate_throttles_revision_updates() -> None:
    gate = ViewerStatsUpdateGate(min_interval_seconds=0.5)

    assert gate.should_update(1, now=10.0)
    assert not gate.should_update(1, now=10.6)
    assert not gate.should_update(2, now=10.2)
    assert gate.should_update(2, now=10.5)


def test_viewer_stats_update_gate_reset_allows_immediate_update() -> None:
    gate = ViewerStatsUpdateGate(min_interval_seconds=10.0)

    assert gate.should_update(1, now=10.0)
    gate.reset()

    assert gate.should_update(2, now=10.1)


def test_viewer_stats_update_gate_skips_active_updates_by_default() -> None:
    gate = ViewerStatsUpdateGate(min_interval_seconds=0.0)

    assert not gate.should_update(1, active=True, now=10.0)
    assert gate.should_update(1, active=False, now=10.0)


def test_viewer_stats_update_gate_can_update_while_active() -> None:
    gate = ViewerStatsUpdateGate(
        min_interval_seconds=0.0,
        update_while_active=True,
    )

    assert gate.should_update(1, active=True, now=10.0)
