#include "stages.h"

// Implements the native FasterGS preprocess wrappers.

#include "common.h"

#include "helper_math.h"
#include "kernels_backward.cuh"
#include "kernels_forward.cuh"
#include "rasterization_config.h"
#include "torch_utils.h"

namespace forward_kernels = faster_gs::rasterization::kernels::forward;
namespace backward_kernels = faster_gs::rasterization::kernels::backward;
namespace config = faster_gs::rasterization::config;

namespace splatkit::faster_gs_native {

// Runs the vendored preprocess kernel after validating tensor dtypes and
// materializing contiguous views expected by the CUDA code.
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
    int active_sh_bases) {
    check_cuda_float_tensor(means, "means");
    check_cuda_float_tensor(scales, "scales");
    check_cuda_float_tensor(rotations, "rotations");
    check_cuda_float_tensor(opacities, "opacities");
    check_cuda_float_tensor(sh_coefficients_0, "sh_coefficients_0");
    check_cuda_float_tensor(sh_coefficients_rest, "sh_coefficients_rest");
    check_cuda_float_tensor(w2c, "w2c");
    check_cuda_float_tensor(cam_position, "cam_position");

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

    // The vendored kernel expects packed float2/float3/float4 views and writes
    // directly into the stage output tensors.
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
            proper_antialiasing
        );
    CHECK_CUDA(config::debug, "preprocess_fwd")

    return {
        projected_means,
        conic_opacity,
        colors_rgb,
        depth_keys,
        primitive_indices,
        num_touched_tiles,
        screen_bounds,
        visible_count,
        instance_count,
    };
}

// Runs the vendored preprocess backward kernel and reshapes the gradient
// tensors into the splatkit-facing layouts.
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
    int active_sh_bases) {
    check_cuda_float_tensor(means, "means");
    check_cuda_float_tensor(scales, "scales");
    check_cuda_float_tensor(rotations, "rotations");
    check_cuda_float_tensor(opacities, "opacities");
    check_cuda_float_tensor(sh_coefficients_rest, "sh_coefficients_rest");
    check_cuda_float_tensor(w2c, "w2c");
    check_cuda_float_tensor(cam_position, "cam_position");
    check_cuda_int_tensor(num_touched_tiles, "num_touched_tiles");
    check_cuda_float_tensor(grad_projected_means, "grad_projected_means");
    check_cuda_float_tensor(grad_conic_opacity, "grad_conic_opacity");
    check_cuda_float_tensor(grad_colors_rgb, "grad_colors_rgb");

    const int n_primitives = means.size(0);
    const int total_sh_bases = sh_coefficients_rest.size(1);
    auto float_options = means.options().dtype(torch::kFloat);

    torch::Tensor grad_means = torch::zeros({n_primitives, 3}, float_options);
    torch::Tensor grad_scales = torch::zeros({n_primitives, 3}, float_options);
    torch::Tensor grad_rotations = torch::zeros({n_primitives, 4}, float_options);
    torch::Tensor grad_opacities = grad_conic_opacity.narrow(1, 3, 1)
                                       .reshape({n_primitives, 1})
                                       .contiguous()
                                       .clone();
    torch::Tensor grad_sh_coefficients_0 = grad_colors_rgb.unsqueeze(1)
                                               .contiguous()
                                               .clone();
    torch::Tensor grad_sh_coefficients_rest = torch::zeros(
        {n_primitives, total_sh_bases, 3},
        float_options
    );
    torch::Tensor grad_mean2d = grad_projected_means.contiguous();
    torch::Tensor grad_conic = grad_conic_opacity.narrow(1, 0, 3)
                                   .transpose(0, 1)
                                   .contiguous();

    torch::Tensor means_c = means.contiguous();
    torch::Tensor scales_c = scales.contiguous();
    torch::Tensor rotations_c = rotations.contiguous();
    torch::Tensor opacities_c = opacities.reshape({-1}).contiguous();
    torch::Tensor shrest_c = sh_coefficients_rest.contiguous();
    torch::Tensor w2c_c = w2c.contiguous();
    torch::Tensor cam_position_c = cam_position.contiguous();
    torch::Tensor touched_c = num_touched_tiles.contiguous();

    // The backward kernel accumulates gradients into separate geometric and SH
    // buffers, which are then returned in the same order as the Python op
    // expects.
    backward_kernels::preprocess_backward_cu
        <<<div_round_up(n_primitives, config::block_size_preprocess_backward),
           config::block_size_preprocess_backward>>>(
            reinterpret_cast<float3*>(means_c.data_ptr<float>()),
            reinterpret_cast<float3*>(scales_c.data_ptr<float>()),
            reinterpret_cast<float4*>(rotations_c.data_ptr<float>()),
            opacities_c.data_ptr<float>(),
            reinterpret_cast<float3*>(shrest_c.data_ptr<float>()),
            reinterpret_cast<float4*>(w2c_c.data_ptr<float>()),
            reinterpret_cast<float3*>(cam_position_c.data_ptr<float>()),
            reinterpret_cast<const uint*>(touched_c.data_ptr<int>()),
            reinterpret_cast<float2*>(grad_mean2d.data_ptr<float>()),
            grad_conic.data_ptr<float>(),
            reinterpret_cast<float3*>(grad_means.data_ptr<float>()),
            reinterpret_cast<float3*>(grad_scales.data_ptr<float>()),
            reinterpret_cast<float4*>(grad_rotations.data_ptr<float>()),
            grad_opacities.data_ptr<float>(),
            reinterpret_cast<float3*>(grad_sh_coefficients_0.data_ptr<float>()),
            reinterpret_cast<float3*>(grad_sh_coefficients_rest.data_ptr<float>()),
            nullptr,
            static_cast<uint>(n_primitives),
            static_cast<uint>(active_sh_bases),
            static_cast<uint>(total_sh_bases),
            static_cast<float>(width),
            static_cast<float>(height),
            focal_x,
            focal_y,
            center_x,
            center_y,
            proper_antialiasing
        );
    CHECK_CUDA(config::debug, "preprocess_bwd")

    return {
        grad_means,
        grad_scales,
        grad_rotations,
        grad_opacities,
        grad_sh_coefficients_0,
        grad_sh_coefficients_rest,
    };
}

}  // namespace splatkit::faster_gs_native
