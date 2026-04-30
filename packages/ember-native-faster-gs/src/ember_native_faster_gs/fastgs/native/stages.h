#pragma once

#include <torch/extension.h>

#include <tuple>

namespace ember_core::fastgs_native {

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
preprocess_fwd_wrapper(
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coefficients_0,
    const torch::Tensor& sh_coefficients_rest,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    float near_plane,
    float far_plane,
    int width,
    int height,
    float focal_x,
    float focal_y,
    float center_x,
    float center_y,
    bool proper_antialiasing,
    int active_sh_bases,
    float compact_box_scale);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sort_fwd_wrapper(
    const torch::Tensor& depth_keys,
    const torch::Tensor& primitive_indices,
    const torch::Tensor& num_touched_tiles,
    const torch::Tensor& screen_bounds,
    const torch::Tensor& projected_means,
    const torch::Tensor& conic_opacity,
    const torch::Tensor& visible_count,
    const torch::Tensor& instance_count,
    int width,
    int height,
    float compact_box_scale);

}  // namespace ember_core::fastgs_native
