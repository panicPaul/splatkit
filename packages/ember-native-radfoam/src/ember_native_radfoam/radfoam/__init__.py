"""Ember-native RADFOAM backend."""

from ember_native_radfoam.radfoam.renderer import (
    RadFoamAlphaDepthRenderOutput,
    RadFoamNativeRenderOptions,
    RadFoamNativeRenderOutput,
    register,
    render_radfoam_native,
)
from ember_native_radfoam.radfoam.runtime import (
    MIN_RADFOAM_POINTS,
    build_aabb_tree,
    build_radfoam_topology,
    farthest_neighbor,
    nearest_neighbor,
    trace,
)
from ember_native_radfoam.radfoam.training import (
    RadFoamOptimizationRecipe,
    RadFoamTopologyRefresh,
    initialize_radfoam_model_from_scene_record,
    initialize_radfoam_scene_from_scene_record,
    radfoam_farthest_neighbor_radius,
    radfoam_optimization_config,
    radfoam_parameter_groups,
    radfoam_rgb_loss,
)

__all__ = [
    "MIN_RADFOAM_POINTS",
    "RadFoamAlphaDepthRenderOutput",
    "RadFoamNativeRenderOptions",
    "RadFoamNativeRenderOutput",
    "RadFoamOptimizationRecipe",
    "RadFoamTopologyRefresh",
    "build_aabb_tree",
    "build_radfoam_topology",
    "farthest_neighbor",
    "initialize_radfoam_model_from_scene_record",
    "initialize_radfoam_scene_from_scene_record",
    "nearest_neighbor",
    "radfoam_farthest_neighbor_radius",
    "radfoam_optimization_config",
    "radfoam_parameter_groups",
    "radfoam_rgb_loss",
    "register",
    "render_radfoam_native",
    "trace",
]
