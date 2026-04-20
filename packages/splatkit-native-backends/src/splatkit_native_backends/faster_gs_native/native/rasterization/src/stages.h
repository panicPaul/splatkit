#pragma once

#include <torch/extension.h>

#include <tuple>

namespace splatkit::faster_gs_native {

std::tuple<
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
    int active_sh_bases);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
preprocess_bwd_wrapper(
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coefficients_rest,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& num_touched_tiles,
    const torch::Tensor& grad_projected_means,
    const torch::Tensor& grad_conic_opacity,
    const torch::Tensor& grad_colors_rgb,
    int width,
    int height,
    float focal_x,
    float focal_y,
    float center_x,
    float center_y,
    bool proper_antialiasing,
    int active_sh_bases);

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
    int height);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
blend_fwd_wrapper(
    const torch::Tensor& instance_primitive_indices,
    const torch::Tensor& tile_instance_ranges,
    const torch::Tensor& tile_bucket_offsets,
    const torch::Tensor& bucket_count,
    const torch::Tensor& projected_means,
    const torch::Tensor& conic_opacity,
    const torch::Tensor& colors_rgb,
    const torch::Tensor& bg_color,
    bool proper_antialiasing,
    int width,
    int height);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
blend_bwd_wrapper(
    const torch::Tensor& grad_image,
    const torch::Tensor& image,
    const torch::Tensor& instance_primitive_indices,
    const torch::Tensor& tile_instance_ranges,
    const torch::Tensor& tile_bucket_offsets,
    const torch::Tensor& projected_means,
    const torch::Tensor& conic_opacity,
    const torch::Tensor& colors_rgb,
    const torch::Tensor& bg_color,
    const torch::Tensor& tile_final_transmittances,
    const torch::Tensor& tile_max_n_processed,
    const torch::Tensor& tile_n_processed,
    const torch::Tensor& bucket_tile_index,
    const torch::Tensor& bucket_color_transmittance,
    bool proper_antialiasing,
    int width,
    int height);

}
