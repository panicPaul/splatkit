#include "stages.h"

#include "common.h"

#ifndef faster_gs
#define faster_gs ember_faster_gs_core_vendor
#define EMBER_FASTGS_UNDEF_FASTER_GS
#endif
#include "helper_math.h"
#include "kernels_forward.cuh"
#include "rasterization_config.h"
#ifdef EMBER_FASTGS_UNDEF_FASTER_GS
#undef faster_gs
#undef EMBER_FASTGS_UNDEF_FASTER_GS
#endif
#include "torch_utils.h"

namespace forward_kernels = ember_fastgs::rasterization::kernels::forward;
namespace config = ember_faster_gs_core_vendor::rasterization::config;

namespace ember_core::fastgs_native {

using namespace ember_core::faster_gs_native;

namespace {

__global__ void compute_primitive_depth_cu(
    const float3* __restrict__ means,
    const float4* __restrict__ w2c,
    float* __restrict__ primitive_depth,
    const uint n_primitives
) {
    const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (primitive_idx >= n_primitives) {
        return;
    }
    const float3 mean3d = means[primitive_idx];
    const float4 w2c_r3 = w2c[2];
    primitive_depth[primitive_idx] = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y +
                                     w2c_r3.z * mean3d.z + w2c_r3.w;
}

}  // namespace

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
    float compact_box_scale) {
    check_cuda_float_tensor(means, "means");
    check_cuda_float_tensor(scales, "scales");
    check_cuda_float_tensor(rotations, "rotations");
    check_cuda_float_tensor(opacities, "opacities");
    check_cuda_float_tensor(sh_coefficients_0, "sh_coefficients_0");
    check_cuda_float_tensor(sh_coefficients_rest, "sh_coefficients_rest");
    check_cuda_float_tensor(w2c, "w2c");
    check_cuda_float_tensor(cam_position, "cam_position");
    TORCH_CHECK(
        compact_box_scale > 0.0f,
        "compact_box_scale must be positive."
    );

    const int n_primitives = means.size(0);
    const int total_sh_bases = sh_coefficients_rest.size(1);
    const int grid_width = div_round_up(width, config::tile_width);
    const int grid_height = div_round_up(height, config::tile_height);
    auto float_options = means.options().dtype(torch::kFloat);
    auto int_options = means.options().dtype(torch::kInt32);
    auto ushort_options = means.options().dtype(torch::kUInt16);

    torch::Tensor projected_means = torch::empty({n_primitives, 2}, float_options);
    torch::Tensor conic_opacity = torch::empty({n_primitives, 4}, float_options);
    torch::Tensor colors_rgb = torch::empty({n_primitives, 3}, float_options);
    torch::Tensor primitive_depth = torch::zeros({n_primitives}, float_options);
    torch::Tensor depth_keys = torch::empty({n_primitives}, int_options);
    torch::Tensor primitive_indices = torch::empty({n_primitives}, int_options);
    torch::Tensor num_touched_tiles = torch::empty({n_primitives}, int_options);
    torch::Tensor screen_bounds = torch::empty({n_primitives, 4}, ushort_options);
    torch::Tensor visible_count = torch::zeros({1}, int_options);
    torch::Tensor instance_count = torch::zeros({1}, int_options);

    torch::Tensor means_c = means.contiguous();
    torch::Tensor scales_c = scales.contiguous();
    torch::Tensor rotations_c = rotations.contiguous();
    torch::Tensor opacities_c = opacities.reshape({-1}).contiguous();
    torch::Tensor sh0_c = sh_coefficients_0.squeeze(1).contiguous();
    torch::Tensor shrest_c = sh_coefficients_rest.contiguous();
    torch::Tensor w2c_c = w2c.contiguous();
    torch::Tensor cam_position_c = cam_position.contiguous();

    compute_primitive_depth_cu<<<div_round_up(n_primitives, config::block_size_preprocess),
                                 config::block_size_preprocess>>>(
        reinterpret_cast<float3*>(means_c.data_ptr<float>()),
        reinterpret_cast<float4*>(w2c_c.data_ptr<float>()),
        primitive_depth.data_ptr<float>(),
        static_cast<uint>(n_primitives)
    );
    CHECK_CUDA(config::debug, "fastgs_compute_primitive_depth")

    forward_kernels::preprocess_cu
        <<<div_round_up(n_primitives, config::block_size_preprocess),
           config::block_size_preprocess>>>(
            reinterpret_cast<float3*>(means_c.data_ptr<float>()),
            reinterpret_cast<float3*>(scales_c.data_ptr<float>()),
            reinterpret_cast<float4*>(rotations_c.data_ptr<float>()),
            opacities_c.data_ptr<float>(),
            reinterpret_cast<float3*>(sh0_c.data_ptr<float>()),
            reinterpret_cast<float3*>(shrest_c.data_ptr<float>()),
            reinterpret_cast<float4*>(w2c_c.data_ptr<float>()),
            reinterpret_cast<float3*>(cam_position_c.data_ptr<float>()),
            reinterpret_cast<uint*>(depth_keys.data_ptr<int>()),
            reinterpret_cast<uint*>(primitive_indices.data_ptr<int>()),
            reinterpret_cast<uint*>(num_touched_tiles.data_ptr<int>()),
            reinterpret_cast<ushort4*>(screen_bounds.data_ptr<uint16_t>()),
            reinterpret_cast<float2*>(projected_means.data_ptr<float>()),
            reinterpret_cast<float4*>(conic_opacity.data_ptr<float>()),
            reinterpret_cast<float3*>(colors_rgb.data_ptr<float>()),
            reinterpret_cast<uint*>(visible_count.data_ptr<int>()),
            reinterpret_cast<uint*>(instance_count.data_ptr<int>()),
            static_cast<uint>(n_primitives),
            static_cast<uint>(grid_width),
            static_cast<uint>(grid_height),
            static_cast<uint>(active_sh_bases),
            static_cast<uint>(total_sh_bases),
            static_cast<float>(width),
            static_cast<float>(height),
            focal_x,
            focal_y,
            center_x,
            center_y,
            near_plane,
            far_plane,
            proper_antialiasing,
            compact_box_scale
        );
    CHECK_CUDA(config::debug, "fastgs_preprocess_fwd")

    return {
        projected_means,
        conic_opacity,
        colors_rgb,
        primitive_depth,
        depth_keys,
        primitive_indices,
        num_touched_tiles,
        screen_bounds,
        visible_count,
        instance_count,
    };
}

}  // namespace ember_core::fastgs_native
