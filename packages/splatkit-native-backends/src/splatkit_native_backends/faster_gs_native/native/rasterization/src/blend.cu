#include "stages.h"

// Implements the native FasterGS blend wrappers.

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

// Runs the vendored blend kernel and returns the rendered image plus the
// auxiliary tensors needed for blend backward.
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
    int height) {
    check_cuda_int_tensor(instance_primitive_indices, "instance_primitive_indices");
    check_cuda_int_tensor(tile_instance_ranges, "tile_instance_ranges");
    check_cuda_int_tensor(tile_bucket_offsets, "tile_bucket_offsets");
    check_cuda_int_tensor(bucket_count, "bucket_count");
    check_cuda_float_tensor(projected_means, "projected_means");
    check_cuda_float_tensor(conic_opacity, "conic_opacity");
    check_cuda_float_tensor(colors_rgb, "colors_rgb");
    check_cuda_float_tensor(bg_color, "bg_color");

    const int tile_count = tile_instance_ranges.size(0);
    const int bucket_total = bucket_count.item<int>();
    const int grid_width = div_round_up(width, config::tile_width);
    const int grid_height = div_round_up(height, config::tile_height);
    auto float_options = projected_means.options().dtype(torch::kFloat);
    auto int_options = projected_means.options().dtype(torch::kInt32);

    torch::Tensor image = torch::empty({3, height, width}, float_options);
    torch::Tensor tile_final_transmittances = torch::empty(
        {tile_count * config::block_size_blend},
        float_options
    );
    torch::Tensor tile_max_n_processed = torch::empty({tile_count}, int_options);
    torch::Tensor tile_n_processed = torch::empty(
        {tile_count * config::block_size_blend},
        int_options
    );
    torch::Tensor bucket_tile_index = torch::empty({bucket_total}, int_options);
    torch::Tensor bucket_color_transmittance = torch::empty(
        {bucket_total * config::block_size_blend, 4},
        float_options
    );

    torch::Tensor instance_indices_c = instance_primitive_indices.contiguous();
    torch::Tensor tile_ranges_c = tile_instance_ranges.contiguous();
    torch::Tensor tile_offsets_c = tile_bucket_offsets.contiguous();
    torch::Tensor projected_means_c = projected_means.contiguous();
    torch::Tensor conic_opacity_c = conic_opacity.contiguous();
    torch::Tensor colors_rgb_c = colors_rgb.contiguous();
    torch::Tensor bg_color_c = bg_color.contiguous();
    (void)proper_antialiasing;

    // The vendored blend kernel writes both the image and the per-bucket
    // backward surfaces in a single pass.
    forward_kernels::blend_cu<<<dim3(grid_width, grid_height, 1),
                                dim3(config::tile_width, config::tile_height, 1)>>>(
        reinterpret_cast<const uint2*>(tile_ranges_c.data_ptr<int>()),
        reinterpret_cast<const uint*>(tile_offsets_c.data_ptr<int>()),
        reinterpret_cast<const uint*>(instance_indices_c.data_ptr<int>()),
        reinterpret_cast<const float2*>(projected_means_c.data_ptr<float>()),
        reinterpret_cast<const float4*>(conic_opacity_c.data_ptr<float>()),
        reinterpret_cast<const float3*>(colors_rgb_c.data_ptr<float>()),
        reinterpret_cast<const float3*>(bg_color_c.data_ptr<float>()),
        image.data_ptr<float>(),
        tile_final_transmittances.data_ptr<float>(),
        reinterpret_cast<uint*>(tile_max_n_processed.data_ptr<int>()),
        reinterpret_cast<uint*>(tile_n_processed.data_ptr<int>()),
        reinterpret_cast<uint*>(bucket_tile_index.data_ptr<int>()),
        reinterpret_cast<float4*>(bucket_color_transmittance.data_ptr<float>()),
        static_cast<uint>(width),
        static_cast<uint>(height),
        static_cast<uint>(grid_width)
    );
    CHECK_CUDA(config::debug, "blend_fwd")

    return {
        image,
        tile_final_transmittances,
        tile_max_n_processed,
        tile_n_processed,
        bucket_tile_index,
        bucket_color_transmittance,
    };
}

// Runs the vendored blend backward kernel and repacks its split opacity/conic
// gradients into the Python-facing conic-opacity tensor layout.
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
    int height) {
    check_cuda_float_tensor(grad_image, "grad_image");
    check_cuda_float_tensor(image, "image");
    check_cuda_int_tensor(instance_primitive_indices, "instance_primitive_indices");
    check_cuda_int_tensor(tile_instance_ranges, "tile_instance_ranges");
    check_cuda_int_tensor(tile_bucket_offsets, "tile_bucket_offsets");
    check_cuda_float_tensor(projected_means, "projected_means");
    check_cuda_float_tensor(conic_opacity, "conic_opacity");
    check_cuda_float_tensor(colors_rgb, "colors_rgb");
    check_cuda_float_tensor(bg_color, "bg_color");
    check_cuda_float_tensor(
        tile_final_transmittances,
        "tile_final_transmittances"
    );
    check_cuda_int_tensor(tile_max_n_processed, "tile_max_n_processed");
    check_cuda_int_tensor(tile_n_processed, "tile_n_processed");
    check_cuda_int_tensor(bucket_tile_index, "bucket_tile_index");
    check_cuda_float_tensor(
        bucket_color_transmittance,
        "bucket_color_transmittance"
    );

    const int n_primitives = projected_means.size(0);
    const int n_buckets = bucket_tile_index.size(0);
    const int grid_width = div_round_up(width, config::tile_width);
    auto float_options = projected_means.options().dtype(torch::kFloat);

    torch::Tensor grad_projected_means = torch::zeros({n_primitives, 2}, float_options);
    // Reuse the large helper buffers that the vendored backward kernel fills by
    // atomic accumulation.
    torch::Tensor grad_conic_helper = get_cached_workspace(
        "blend_grad_conic_helper",
        float_options,
        3LL * n_primitives
    ).view({3, n_primitives});
    torch::Tensor grad_opacity = get_cached_workspace(
        "blend_grad_opacity",
        float_options,
        n_primitives
    );
    grad_conic_helper.zero_();
    grad_opacity.zero_();
    torch::Tensor grad_sh_coefficients_0 = torch::zeros(
        {n_primitives, 1, 3},
        float_options
    );
    torch::Tensor tile_instance_ranges_c = tile_instance_ranges.contiguous();
    torch::Tensor tile_bucket_offsets_c = tile_bucket_offsets.contiguous();
    torch::Tensor instance_primitive_indices_c = instance_primitive_indices.contiguous();
    torch::Tensor projected_means_c = projected_means.contiguous();
    torch::Tensor conic_opacity_c = conic_opacity.contiguous();
    torch::Tensor colors_rgb_c = colors_rgb.contiguous();
    torch::Tensor bg_color_c = bg_color.contiguous();
    torch::Tensor grad_image_c = grad_image.contiguous();
    torch::Tensor image_c = image.contiguous();
    torch::Tensor tile_final_transmittances_c = tile_final_transmittances.contiguous();
    torch::Tensor tile_max_n_processed_c = tile_max_n_processed.contiguous();
    torch::Tensor tile_n_processed_c = tile_n_processed.contiguous();
    torch::Tensor bucket_tile_index_c = bucket_tile_index.contiguous();
    torch::Tensor bucket_color_transmittance_c = bucket_color_transmittance.contiguous();

    if (n_buckets > 0) {
        // Launch one block per bucket, matching the vendored FasterGS
        // implementation.
        backward_kernels::blend_backward_cu<<<n_buckets, 32>>>(
            reinterpret_cast<const uint2*>(tile_instance_ranges_c.data_ptr<int>()),
            reinterpret_cast<const uint*>(tile_bucket_offsets_c.data_ptr<int>()),
            reinterpret_cast<const uint*>(instance_primitive_indices_c.data_ptr<int>()),
            reinterpret_cast<const float2*>(projected_means_c.data_ptr<float>()),
            reinterpret_cast<const float4*>(conic_opacity_c.data_ptr<float>()),
            reinterpret_cast<const float3*>(colors_rgb_c.data_ptr<float>()),
            reinterpret_cast<const float3*>(bg_color_c.data_ptr<float>()),
            grad_image_c.data_ptr<float>(),
            image_c.data_ptr<float>(),
            tile_final_transmittances_c.data_ptr<float>(),
            reinterpret_cast<const uint*>(tile_max_n_processed_c.data_ptr<int>()),
            reinterpret_cast<const uint*>(tile_n_processed_c.data_ptr<int>()),
            reinterpret_cast<const uint*>(bucket_tile_index_c.data_ptr<int>()),
            reinterpret_cast<const float4*>(
                bucket_color_transmittance_c.data_ptr<float>()
            ),
            reinterpret_cast<float2*>(grad_projected_means.data_ptr<float>()),
            grad_conic_helper.data_ptr<float>(),
            grad_opacity.data_ptr<float>(),
            reinterpret_cast<float3*>(grad_sh_coefficients_0.data_ptr<float>()),
            static_cast<uint>(n_primitives),
            static_cast<uint>(width),
            static_cast<uint>(height),
            static_cast<uint>(grid_width),
            proper_antialiasing
        );
        CHECK_CUDA(config::debug, "blend_bwd")
    }

    // The vendored kernel accumulates conic and opacity gradients separately,
    // so concatenate them into the `(a, b, c, opacity)` layout expected by the
    // Python runtime.
    torch::Tensor grad_conic_opacity = torch::empty({n_primitives, 4}, float_options);
    grad_conic_opacity.narrow(1, 0, 3).copy_(grad_conic_helper.transpose(0, 1));
    grad_conic_opacity.narrow(1, 3, 1).copy_(
        grad_opacity.reshape({n_primitives, 1})
    );
    return {
        grad_projected_means,
        grad_conic_opacity,
        grad_sh_coefficients_0.squeeze(1),
    };
}

}  // namespace splatkit::faster_gs_native
