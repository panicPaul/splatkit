#pragma once

#include <torch/extension.h>

namespace faster_gs::mip_splatting {

void update_mip_splatting_3d_filter_wrapper(
    const torch::Tensor& positions,
    const torch::Tensor& world_to_camera_matrix,
    torch::Tensor& mip_splatting_3d_filter,
    torch::Tensor& visibility_mask,
    const int image_width,
    const int image_height,
    const float focal_length_x,
    const float focal_length_y,
    const float principal_point_x,
    const float principal_point_y,
    const float near_plane,
    const float clipping_tolerance,
    const float distance_to_filter_scale);

}
