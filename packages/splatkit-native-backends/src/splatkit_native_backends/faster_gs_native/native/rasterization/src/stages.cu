#include "stages.h"

#include "buffer_utils.h"
#include "helper_math.h"
#include "kernels_backward.cuh"
#include "kernels_forward.cuh"
#include "rasterization_config.h"
#include "torch_utils.h"
#include "utils.h"

#include <cub/cub.cuh>
#include <type_traits>

namespace forward_kernels = faster_gs::rasterization::kernels::forward;
namespace backward_kernels = faster_gs::rasterization::kernels::backward;
namespace config = faster_gs::rasterization::config;

namespace splatkit::faster_gs_native {

namespace {

void check_cuda_float_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat32,
        name,
        " must be float32."
    );
}

void check_cuda_int_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kInt32,
        name,
        " must be int32."
    );
}

void check_cuda_uint16_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kUInt16,
        name,
        " must be uint16."
    );
}

template <typename KeyT>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int> run_sort_impl(
    const torch::Tensor& primitive_indices_sorted,
    const torch::Tensor& primitive_offsets,
    const torch::Tensor& screen_bounds,
    const torch::Tensor& projected_means,
    const torch::Tensor& conic_opacity,
    int grid_width,
    int tile_count,
    int end_bit,
    int n_visible,
    int n_instances) {
    auto int_options = primitive_indices_sorted.options().dtype(torch::kInt32);
    torch::Tensor tile_instance_ranges = torch::zeros(
        {tile_count, 2},
        int_options
    );
    torch::Tensor tile_bucket_offsets = torch::zeros({tile_count}, int_options);
    if (n_instances == 0) {
        return {
            torch::empty({0}, int_options),
            tile_instance_ranges,
            tile_bucket_offsets,
            0,
        };
    }

    auto key_options = torch::TensorOptions()
                           .dtype(
                               std::is_same_v<KeyT, ushort>
                                   ? torch::kUInt16
                                   : torch::kInt32
                           )
                           .device(primitive_indices_sorted.device());
    torch::Tensor instance_keys_current = torch::empty({n_instances}, key_options);
    torch::Tensor instance_keys_alternate = torch::empty({n_instances}, key_options);
    torch::Tensor instance_indices_current = torch::empty({n_instances}, int_options);
    torch::Tensor instance_indices_alternate = torch::empty({n_instances}, int_options);

    forward_kernels::create_instances_cu<KeyT>
        <<<div_round_up(n_visible, config::block_size_create_instances),
           config::block_size_create_instances>>>(
            reinterpret_cast<const uint*>(
                primitive_indices_sorted.data_ptr<int>()
            ),
            reinterpret_cast<const uint*>(primitive_offsets.data_ptr<int>()),
            reinterpret_cast<const ushort4*>(screen_bounds.data_ptr<uint16_t>()),
            reinterpret_cast<const float2*>(projected_means.data_ptr<float>()),
            reinterpret_cast<const float4*>(conic_opacity.data_ptr<float>()),
            reinterpret_cast<KeyT*>(instance_keys_current.data_ptr()),
            reinterpret_cast<uint*>(instance_indices_current.data_ptr<int>()),
            static_cast<uint>(grid_width),
            static_cast<uint>(primitive_indices_sorted.size(0))
        );
    CHECK_CUDA(config::debug, "create_instances")

    cub::DoubleBuffer<KeyT> instance_key_buffers(
        reinterpret_cast<KeyT*>(instance_keys_current.data_ptr()),
        reinterpret_cast<KeyT*>(instance_keys_alternate.data_ptr())
    );
    cub::DoubleBuffer<uint> instance_index_buffers(
        reinterpret_cast<uint*>(instance_indices_current.data_ptr<int>()),
        reinterpret_cast<uint*>(instance_indices_alternate.data_ptr<int>())
    );
    size_t sort_workspace_size = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr,
        sort_workspace_size,
        instance_key_buffers,
        instance_index_buffers,
        n_instances,
        0,
        end_bit
    );
    torch::Tensor sort_workspace = torch::empty(
        {static_cast<long long>(sort_workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(projected_means.device())
    );
    cub::DeviceRadixSort::SortPairs(
        sort_workspace.data_ptr(),
        sort_workspace_size,
        instance_key_buffers,
        instance_index_buffers,
        n_instances,
        0,
        end_bit
    );
    CHECK_CUDA(config::debug, "sort_instances")

    torch::Tensor instance_indices_sorted = torch::empty({n_instances}, int_options);
    torch::Tensor sorted_keys;
    if (instance_index_buffers.selector == 0) {
        instance_indices_sorted.copy_(instance_indices_current);
    } else {
        instance_indices_sorted.copy_(instance_indices_alternate);
    }
    if (instance_key_buffers.selector == 0) {
        sorted_keys = instance_keys_current;
    } else {
        sorted_keys = instance_keys_alternate;
    }

    forward_kernels::extract_instance_ranges_cu<KeyT>
        <<<div_round_up(n_instances, config::block_size_extract_instance_ranges),
           config::block_size_extract_instance_ranges>>>(
            reinterpret_cast<const KeyT*>(sorted_keys.data_ptr()),
            reinterpret_cast<uint2*>(tile_instance_ranges.data_ptr<int>()),
            static_cast<uint>(n_instances)
        );
    CHECK_CUDA(config::debug, "extract_instance_ranges")

    torch::Tensor tile_num_buckets = torch::empty({tile_count}, int_options);
    forward_kernels::extract_bucket_counts
        <<<div_round_up(tile_count, config::block_size_extract_bucket_counts),
           config::block_size_extract_bucket_counts>>>(
            reinterpret_cast<const uint2*>(tile_instance_ranges.data_ptr<int>()),
            reinterpret_cast<uint*>(tile_num_buckets.data_ptr<int>()),
            static_cast<uint>(tile_count)
        );
    CHECK_CUDA(config::debug, "extract_bucket_counts")

    size_t scan_workspace_size = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr,
        scan_workspace_size,
        reinterpret_cast<uint*>(tile_num_buckets.data_ptr<int>()),
        reinterpret_cast<uint*>(tile_bucket_offsets.data_ptr<int>()),
        tile_count
    );
    torch::Tensor scan_workspace = torch::empty(
        {static_cast<long long>(scan_workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(projected_means.device())
    );
    cub::DeviceScan::InclusiveSum(
        scan_workspace.data_ptr(),
        scan_workspace_size,
        reinterpret_cast<uint*>(tile_num_buckets.data_ptr<int>()),
        reinterpret_cast<uint*>(tile_bucket_offsets.data_ptr<int>()),
        tile_count
    );
    CHECK_CUDA(config::debug, "scan_bucket_counts")

    int bucket_count = 0;
    if (tile_count > 0) {
        bucket_count = tile_bucket_offsets[tile_count - 1].item<int>();
    }
    return {
        instance_indices_sorted,
        tile_instance_ranges,
        tile_bucket_offsets,
        bucket_count,
    };
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
    int height) {
    check_cuda_int_tensor(depth_keys, "depth_keys");
    check_cuda_int_tensor(primitive_indices, "primitive_indices");
    check_cuda_int_tensor(num_touched_tiles, "num_touched_tiles");
    check_cuda_uint16_tensor(screen_bounds, "screen_bounds");
    check_cuda_float_tensor(projected_means, "projected_means");
    check_cuda_float_tensor(conic_opacity, "conic_opacity");
    check_cuda_int_tensor(visible_count, "visible_count");
    check_cuda_int_tensor(instance_count, "instance_count");

    const int n_visible = visible_count.item<int>();
    const int n_instances = instance_count.item<int>();
    const int grid_width = div_round_up(width, config::tile_width);
    const int grid_height = div_round_up(height, config::tile_height);
    const int tile_count = grid_width * grid_height;
    const int end_bit = tile_count > 0
                            ? faster_gs::rasterization::extract_end_bit(tile_count - 1)
                            : 0;

    auto int_options = primitive_indices.options().dtype(torch::kInt32);
    torch::Tensor bucket_count = torch::zeros({1}, int_options);
    torch::Tensor screen_bounds_c = screen_bounds.contiguous();
    torch::Tensor projected_means_c = projected_means.contiguous();
    torch::Tensor conic_opacity_c = conic_opacity.contiguous();
    if (n_visible == 0) {
        return {
            torch::empty({0}, int_options),
            torch::zeros({tile_count, 2}, int_options),
            torch::zeros({tile_count}, int_options),
            bucket_count,
        };
    }

    torch::Tensor depth_keys_prefix = depth_keys.narrow(0, 0, n_visible).contiguous();
    torch::Tensor primitive_indices_prefix = primitive_indices.narrow(0, 0, n_visible)
                                                 .contiguous();
    torch::Tensor depth_keys_sorted = torch::empty({n_visible}, int_options);
    torch::Tensor primitive_indices_sorted_tmp = torch::empty({n_visible}, int_options);
    cub::DoubleBuffer<uint> depth_key_buffers(
        reinterpret_cast<uint*>(depth_keys_prefix.data_ptr<int>()),
        reinterpret_cast<uint*>(depth_keys_sorted.data_ptr<int>())
    );
    cub::DoubleBuffer<uint> primitive_index_buffers(
        reinterpret_cast<uint*>(primitive_indices_prefix.data_ptr<int>()),
        reinterpret_cast<uint*>(primitive_indices_sorted_tmp.data_ptr<int>())
    );
    size_t sort_workspace_size = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr,
        sort_workspace_size,
        depth_key_buffers,
        primitive_index_buffers,
        n_visible
    );
    torch::Tensor sort_workspace = torch::empty(
        {static_cast<long long>(sort_workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(depth_keys.device())
    );
    cub::DeviceRadixSort::SortPairs(
        sort_workspace.data_ptr(),
        sort_workspace_size,
        depth_key_buffers,
        primitive_index_buffers,
        n_visible
    );
    CHECK_CUDA(config::debug, "sort_depth")

    torch::Tensor primitive_indices_sorted = torch::empty({n_visible}, int_options);
    if (primitive_index_buffers.selector == 0) {
        primitive_indices_sorted.copy_(primitive_indices_prefix);
    } else {
        primitive_indices_sorted.copy_(primitive_indices_sorted_tmp);
    }

    torch::Tensor primitive_offsets = torch::empty({n_visible}, int_options);
    torch::Tensor num_touched_tiles_c = num_touched_tiles.contiguous();
    forward_kernels::apply_depth_ordering_cu
        <<<div_round_up(n_visible, config::block_size_apply_depth_ordering),
           config::block_size_apply_depth_ordering>>>(
            reinterpret_cast<const uint*>(primitive_indices_sorted.data_ptr<int>()),
            reinterpret_cast<const uint*>(num_touched_tiles_c.data_ptr<int>()),
            reinterpret_cast<uint*>(primitive_offsets.data_ptr<int>()),
            static_cast<uint>(n_visible)
        );
    CHECK_CUDA(config::debug, "apply_depth_ordering")

    size_t scan_workspace_size = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        scan_workspace_size,
        reinterpret_cast<uint*>(primitive_offsets.data_ptr<int>()),
        reinterpret_cast<uint*>(primitive_offsets.data_ptr<int>()),
        n_visible
    );
    torch::Tensor scan_workspace = torch::empty(
        {static_cast<long long>(scan_workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(depth_keys.device())
    );
    cub::DeviceScan::ExclusiveSum(
        scan_workspace.data_ptr(),
        scan_workspace_size,
        reinterpret_cast<uint*>(primitive_offsets.data_ptr<int>()),
        reinterpret_cast<uint*>(primitive_offsets.data_ptr<int>()),
        n_visible
    );
    CHECK_CUDA(config::debug, "scan_offsets")

    auto sort_result = end_bit <= 16
                           ? run_sort_impl<ushort>(
                                 primitive_indices_sorted,
                                 primitive_offsets,
                                 screen_bounds_c,
                                 projected_means_c,
                                 conic_opacity_c,
                                 grid_width,
                                 tile_count,
                                 end_bit,
                                 n_visible,
                                 n_instances
                             )
                           : run_sort_impl<uint>(
                                 primitive_indices_sorted,
                                 primitive_offsets,
                                 screen_bounds_c,
                                 projected_means_c,
                                 conic_opacity_c,
                                 grid_width,
                                 tile_count,
                                 end_bit,
                                 n_visible,
                                 n_instances
                             );
    bucket_count.fill_(std::get<3>(sort_result));
    return {
        std::get<0>(sort_result),
        std::get<1>(sort_result),
        std::get<2>(sort_result),
        bucket_count,
    };
}

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
    torch::Tensor grad_conic_helper = torch::zeros({3, n_primitives}, float_options);
    torch::Tensor grad_opacity = torch::zeros({n_primitives}, float_options);
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
