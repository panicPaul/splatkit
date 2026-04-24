#include "stages.h"

// Implements the native FasterGS sort wrapper.

#include "common.h"

#include "vendor_namespace_begin.h"
#include "buffer_utils.h"
#include "helper_math.h"
#include "kernels_forward.cuh"
#include "rasterization_config.h"
#include "vendor_namespace_end.h"
#include "torch_utils.h"
#include "utils.h"

#include <cub/cub.cuh>
#include <type_traits>

namespace forward_kernels = splatkit_faster_gs_core_vendor::rasterization::kernels::forward;
namespace config = splatkit_faster_gs_core_vendor::rasterization::config;

namespace splatkit::faster_gs_native {

namespace {

// Materializes tiled instance keys, radix-sorts them, and derives per-tile
// instance ranges for the blend stage.
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
    auto key_options = torch::TensorOptions()
                           .dtype(
                               std::is_same_v<KeyT, ushort>
                                   ? torch::kUInt16
                                   : torch::kInt32
                           )
                           .device(primitive_indices_sorted.device());
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

    // Reuse large instance and radix-sort buffers across frames while still
    // returning fresh output tensors to Python.
    torch::Tensor instance_keys_current = get_cached_workspace(
        "instance_keys_current",
        key_options,
        n_instances
    );
    torch::Tensor instance_keys_alternate = get_cached_workspace(
        "instance_keys_alternate",
        key_options,
        n_instances
    );
    torch::Tensor instance_indices_current = get_cached_workspace(
        "instance_indices_current",
        int_options,
        n_instances
    );
    torch::Tensor instance_indices_alternate = get_cached_workspace(
        "instance_indices_alternate",
        int_options,
        n_instances
    );

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

    // CUB alternates between the double buffers, so keep both around and copy
    // the selected output into a fresh tensor before returning to Python.
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
    torch::Tensor sort_workspace = get_cached_workspace(
        "instance_sort_workspace",
        torch::TensorOptions().dtype(torch::kUInt8).device(projected_means.device()),
        static_cast<int64_t>(sort_workspace_size)
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

    torch::Tensor tile_num_buckets = get_cached_workspace(
        "tile_num_buckets",
        int_options,
        tile_count
    );
    forward_kernels::extract_bucket_counts
        <<<div_round_up(tile_count, config::block_size_extract_bucket_counts),
           config::block_size_extract_bucket_counts>>>(
            reinterpret_cast<const uint2*>(tile_instance_ranges.data_ptr<int>()),
            reinterpret_cast<uint*>(tile_num_buckets.data_ptr<int>()),
            static_cast<uint>(tile_count)
        );
    CHECK_CUDA(config::debug, "extract_bucket_counts")

    // Convert bucket counts into inclusive offsets so the blend kernel can
    // index each tile's buckets directly.
    size_t scan_workspace_size = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr,
        scan_workspace_size,
        reinterpret_cast<uint*>(tile_num_buckets.data_ptr<int>()),
        reinterpret_cast<uint*>(tile_bucket_offsets.data_ptr<int>()),
        tile_count
    );
    torch::Tensor scan_workspace = get_cached_workspace(
        "bucket_scan_workspace",
        torch::TensorOptions().dtype(torch::kUInt8).device(projected_means.device()),
        static_cast<int64_t>(scan_workspace_size)
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

// Runs the vendored FasterGS depth/tile sort and returns the instance ordering
// consumed by the blend stage.
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
                            ? splatkit_faster_gs_core_vendor::rasterization::extract_end_bit(
                                  tile_count - 1
                              )
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

    // First sort visible primitives by depth so per-primitive tile expansion
    // preserves front-to-back ordering within each tile.
    torch::Tensor depth_keys_prefix = depth_keys.narrow(0, 0, n_visible).contiguous();
    torch::Tensor primitive_indices_prefix = primitive_indices.narrow(0, 0, n_visible)
                                                 .contiguous();
    torch::Tensor depth_keys_sorted = get_cached_workspace(
        "depth_keys_sorted",
        int_options,
        n_visible
    );
    torch::Tensor primitive_indices_sorted_tmp = get_cached_workspace(
        "primitive_indices_sorted_tmp",
        int_options,
        n_visible
    );
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
    torch::Tensor sort_workspace = get_cached_workspace(
        "depth_sort_workspace",
        torch::TensorOptions().dtype(torch::kUInt8).device(depth_keys.device()),
        static_cast<int64_t>(sort_workspace_size)
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

    // Then compute exclusive offsets into the flattened instance array and use
    // those offsets to expand each visible primitive into its touched tiles.
    torch::Tensor primitive_offsets = get_cached_workspace(
        "primitive_offsets",
        int_options,
        n_visible
    );
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
    torch::Tensor scan_workspace = get_cached_workspace(
        "offset_scan_workspace",
        torch::TensorOptions().dtype(torch::kUInt8).device(depth_keys.device()),
        static_cast<int64_t>(scan_workspace_size)
    );
    cub::DeviceScan::ExclusiveSum(
        scan_workspace.data_ptr(),
        scan_workspace_size,
        reinterpret_cast<uint*>(primitive_offsets.data_ptr<int>()),
        reinterpret_cast<uint*>(primitive_offsets.data_ptr<int>()),
        n_visible
    );
    CHECK_CUDA(config::debug, "scan_offsets")

    // The tile id bit-width determines whether the compact 16-bit or full
    // 32-bit key path is valid for the second radix sort.
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

}  // namespace splatkit::faster_gs_native
