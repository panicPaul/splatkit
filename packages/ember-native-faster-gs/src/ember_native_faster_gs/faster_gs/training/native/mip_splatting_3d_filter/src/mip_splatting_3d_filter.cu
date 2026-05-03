#include "mip_splatting_3d_filter.h"
#include "mip_splatting_3d_filter_config.h"
#include "helper_math.h"
#include "torch_utils.h"
#include "utils.h"

namespace faster_gs::mip_splatting {

__global__ void update_mip_splatting_3d_filter_cuda(
    const float3* positions,
    const float4* world_to_camera_matrix,
    float* mip_splatting_3d_filter,
    bool* visibility_mask,
    const int n_points,
    const float left,
    const float right,
    const float top,
    const float bottom,
    const float near_plane,
    const float distance_to_filter_scale)
{
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= n_points) return;

    const float3 position_world = positions[point_idx];
    const float4 world_to_camera_row_3 = world_to_camera_matrix[2];
    const float z = dot(make_float3(world_to_camera_row_3), position_world) + world_to_camera_row_3.w;
    if (z < near_plane) return;

    const float4 world_to_camera_row_1 = world_to_camera_matrix[0];
    const float x_clip = dot(make_float3(world_to_camera_row_1), position_world) + world_to_camera_row_1.w;
    if (x_clip < left * z || x_clip > right * z) return;

    const float4 world_to_camera_row_2 = world_to_camera_matrix[1];
    const float y_clip = dot(make_float3(world_to_camera_row_2), position_world) + world_to_camera_row_2.w;
    if (y_clip < top * z || y_clip > bottom * z) return;

    const float mip_splatting_3d_filter_new = distance_to_filter_scale * z;
    if (mip_splatting_3d_filter[point_idx] < mip_splatting_3d_filter_new) return;
    mip_splatting_3d_filter[point_idx] = mip_splatting_3d_filter_new;
    visibility_mask[point_idx] = true;
}

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
    const float distance_to_filter_scale)
{
    CHECK_INPUT(config::debug, positions, "positions");
    CHECK_INPUT(config::debug, world_to_camera_matrix, "world_to_camera_matrix");
    CHECK_INPUT(config::debug, mip_splatting_3d_filter, "mip_splatting_3d_filter");
    if (config::debug && (!visibility_mask.is_cuda() || !visibility_mask.is_contiguous() || visibility_mask.scalar_type() != torch::kBool)) {
        throw std::runtime_error("Input tensor 'visibility_mask' must be a contiguous CUDA bool tensor.");
    }

    const int n_points = positions.size(0);
    const float bounds_factor = clipping_tolerance + 0.5f;
    const float image_width_f = static_cast<float>(image_width);
    const float image_height_f = static_cast<float>(image_height);
    const float max_x_shifted = bounds_factor * image_width_f;
    const float max_y_shifted = bounds_factor * image_height_f;
    const float principal_offset_x = principal_point_x - 0.5f * image_width_f;
    const float principal_offset_y = principal_point_y - 0.5f * image_height_f;
    const float left = (-max_x_shifted - principal_offset_x) / focal_length_x;
    const float right = (max_x_shifted - principal_offset_x) / focal_length_x;
    const float top = (-max_y_shifted - principal_offset_y) / focal_length_y;
    const float bottom = (max_y_shifted - principal_offset_y) / focal_length_y;

    update_mip_splatting_3d_filter_cuda<<<
        div_round_up(n_points, config::block_size_update_mip_splatting_3d_filter),
        config::block_size_update_mip_splatting_3d_filter>>>(
        reinterpret_cast<const float3*>(positions.data_ptr<float>()),
        reinterpret_cast<const float4*>(world_to_camera_matrix.data_ptr<float>()),
        mip_splatting_3d_filter.data_ptr<float>(),
        visibility_mask.data_ptr<bool>(),
        n_points,
        left,
        right,
        top,
        bottom,
        near_plane,
        distance_to_filter_scale);
    CHECK_CUDA(config::debug, "update_mip_splatting_3d_filter_cuda");
}

}
