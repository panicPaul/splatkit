#include <torch/extension.h>

#include <tuple>

#include "common.h"
#include "helper_math.h"
#include "vendor_namespace_begin.h"
#include "rasterization_config.h"
#include "vendor_namespace_end.h"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace config = splatkit_faster_gs_core_vendor::rasterization::config;

namespace splatkit::faster_gs_depth_native {

namespace {

constexpr float expected_depth_denom_eps = 1e-10f;

__global__ void __launch_bounds__(config::block_size_blend) depth_blend_forward_cu(
    const uint2* __restrict__ tile_instance_ranges,
    const uint* __restrict__ tile_bucket_offsets,
    const uint* __restrict__ instance_primitive_indices,
    const float2* __restrict__ primitive_mean2d,
    const float4* __restrict__ primitive_conic_opacity,
    const float* __restrict__ primitive_depth,
    float* __restrict__ depth,
    float* __restrict__ bucket_depth_prefix,
    const uint width,
    const uint height,
    const uint grid_width
) {
    auto block = cg::this_thread_block();
    const dim3 group_index = block.group_index();
    const dim3 thread_index = block.thread_index();
    const uint thread_rank = block.thread_rank();
    const uint2 pixel_coords = make_uint2(
        group_index.x * config::tile_width + thread_index.x,
        group_index.y * config::tile_height + thread_index.y
    );
    const bool inside = pixel_coords.x < width && pixel_coords.y < height;
    const float2 pixel = make_float2(
        __uint2float_rn(pixel_coords.x),
        __uint2float_rn(pixel_coords.y)
    ) +
                         0.5f;
    const uint tile_idx = group_index.y * grid_width + group_index.x;
    const uint2 tile_range = tile_instance_ranges[tile_idx];
    const int n_points_total = tile_range.y - tile_range.x;
    uint bucket_offset = (tile_idx == 0) ? 0 : tile_bucket_offsets[tile_idx - 1];

    __shared__ float2 collected_mean2d[config::block_size_blend];
    __shared__ float4 collected_conic_opacity[config::block_size_blend];
    __shared__ float collected_depth[config::block_size_blend];

    float depth_numerator = 0.0f;
    float transmittance = 1.0f;
    bool done = !inside;

    for (int n_points_remaining = n_points_total,
             current_fetch_idx = tile_range.x + thread_rank;
         n_points_remaining > 0;
         n_points_remaining -= config::block_size_blend,
            current_fetch_idx += config::block_size_blend) {
        if (__syncthreads_count(done) == config::block_size_blend) {
            break;
        }
        if (current_fetch_idx < tile_range.y) {
            const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
            collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
            collected_conic_opacity[thread_rank] =
                primitive_conic_opacity[primitive_idx];
            collected_depth[thread_rank] = primitive_depth[primitive_idx];
        }
        block.sync();
        const int current_batch_size =
            min(config::block_size_blend, n_points_remaining);
        for (int j = 0; !done && j < current_batch_size; ++j) {
            if (j % 32 == 0) {
                bucket_depth_prefix[bucket_offset * config::block_size_blend +
                                    thread_rank] = depth_numerator;
                bucket_offset++;
            }

            const float4 conic_opacity = collected_conic_opacity[j];
            const float3 conic = make_float3(conic_opacity);
            const float opacity = conic_opacity.w;
            const float2 delta = collected_mean2d[j] - pixel;
            const float exponent =
                -0.5f * (conic.x * delta.x * delta.x +
                         conic.z * delta.y * delta.y) -
                conic.y * delta.x * delta.y;
            const float gaussian = expf(fminf(exponent, 0.0f));
            if (!config::original_opacity_interpretation &&
                gaussian < config::min_alpha_threshold) {
                continue;
            }
            const float alpha = opacity * gaussian;
            if (config::original_opacity_interpretation &&
                alpha < config::min_alpha_threshold) {
                continue;
            }

            depth_numerator += transmittance * alpha * collected_depth[j];
            transmittance *= 1.0f - alpha;

            if (transmittance < config::transmittance_threshold) {
                done = true;
            }
        }
    }

    if (inside) {
        const float alpha_total = 1.0f - transmittance;
        const float depth_denom = fmaxf(alpha_total, expected_depth_denom_eps);
        depth[width * pixel_coords.y + pixel_coords.x] =
            depth_numerator / depth_denom;
    }
}

__global__ void depth_blend_backward_cu(
    const uint2* __restrict__ tile_instance_ranges,
    const uint* __restrict__ tile_bucket_offsets,
    const uint* __restrict__ instance_primitive_indices,
    const float2* __restrict__ primitive_mean2d,
    const float4* __restrict__ primitive_conic_opacity,
    const float* __restrict__ primitive_depth,
    const float* __restrict__ grad_depth,
    const float* __restrict__ depth,
    const float* __restrict__ tile_final_transmittances,
    const uint* __restrict__ tile_max_n_processed,
    const uint* __restrict__ tile_n_processed,
    const uint* __restrict__ bucket_tile_index,
    const float4* __restrict__ bucket_color_transmittance,
    const float* __restrict__ bucket_depth_prefix,
    float2* __restrict__ grad_mean2d,
    float* __restrict__ grad_conic,
    float* __restrict__ grad_opacity,
    float* __restrict__ grad_primitive_depth,
    const uint n_primitives,
    const uint width,
    const uint height,
    const uint grid_width,
    const bool proper_antialiasing
) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const uint bucket_idx = block.group_index().x;
    const uint lane_idx = warp.thread_rank();

    const uint tile_idx = bucket_tile_index[bucket_idx];
    const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
    const int tile_n_primitives = tile_instance_range.y - tile_instance_range.x;
    const uint tile_first_bucket_offset =
        (tile_idx == 0) ? 0 : tile_bucket_offsets[tile_idx - 1];
    const int tile_bucket_idx = bucket_idx - tile_first_bucket_offset;
    if (tile_bucket_idx * 32 >= tile_max_n_processed[tile_idx]) {
        return;
    }

    const int tile_primitive_idx = tile_bucket_idx * 32 + lane_idx;
    const int instance_idx = tile_instance_range.x + tile_primitive_idx;
    const bool valid_primitive = tile_primitive_idx < tile_n_primitives;

    uint primitive_idx = 0;
    float2 mean2d = {0.0f, 0.0f};
    float3 conic = {0.0f, 0.0f, 0.0f};
    float opacity = 0.0f;
    float depth_value = 0.0f;
    if (valid_primitive) {
        primitive_idx = instance_primitive_indices[instance_idx];
        mean2d = primitive_mean2d[primitive_idx];
        const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
        conic = make_float3(conic_opacity);
        opacity = conic_opacity.w;
        depth_value = primitive_depth[primitive_idx];
    }

    float2 dL_dmean2d_accum = {0.0f, 0.0f};
    float3 dL_dconic_accum = {0.0f, 0.0f, 0.0f};
    float dL_dopacity_accum = 0.0f;
    float dL_ddepth_accum = 0.0f;

    const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
    const uint2 start_pixel_coords = {
        tile_coords.x * config::tile_width,
        tile_coords.y * config::tile_height,
    };

    uint last_contributor = 0;
    float numerator_after = 0.0f;
    float transmittance = 1.0f;
    float final_transmittance = 1.0f;
    float expected_depth = 0.0f;
    float grad_depth_pixel = 0.0f;

    __shared__ uint collected_last_contributor[32];
    __shared__ float4 collected_depth_state[32];
    __shared__ float collected_grad_depth_pixel[32];

    for (int i = 0; i < config::block_size_blend + 31; ++i) {
        if (i % 32 == 0) {
            const uint local_idx = i + lane_idx;
            if (local_idx < config::block_size_blend) {
                const float4 color_transmittance =
                    bucket_color_transmittance[bucket_idx * config::block_size_blend +
                                               local_idx];
                const float depth_prefix =
                    bucket_depth_prefix[bucket_idx * config::block_size_blend +
                                        local_idx];
                const uint2 pixel_coords = {
                    start_pixel_coords.x + local_idx % config::tile_width,
                    start_pixel_coords.y + local_idx / config::tile_width,
                };
                const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
                float expected_depth_pixel = 0.0f;
                float grad_pixel = 0.0f;
                float final_transmittance_pixel = 1.0f;
                if (pixel_coords.x < width && pixel_coords.y < height) {
                    expected_depth_pixel = depth[pixel_idx];
                    grad_pixel = grad_depth[pixel_idx];
                    final_transmittance_pixel = tile_final_transmittances[pixel_idx];
                }
                const float alpha_total = 1.0f - final_transmittance_pixel;
                const float depth_denom =
                    fmaxf(alpha_total, expected_depth_denom_eps);
                const float depth_numerator =
                    expected_depth_pixel * depth_denom;
                collected_depth_state[lane_idx] = make_float4(
                    depth_numerator - depth_prefix,
                    color_transmittance.w,
                    final_transmittance_pixel,
                    expected_depth_pixel
                );
                collected_grad_depth_pixel[lane_idx] = grad_pixel;
                collected_last_contributor[lane_idx] = tile_n_processed[pixel_idx];
            }
            warp.sync();
        }

        if (i > 0) {
            last_contributor = warp.shfl_up(last_contributor, 1);
            numerator_after = warp.shfl_up(numerator_after, 1);
            transmittance = warp.shfl_up(transmittance, 1);
            final_transmittance = warp.shfl_up(final_transmittance, 1);
            expected_depth = warp.shfl_up(expected_depth, 1);
            grad_depth_pixel = warp.shfl_up(grad_depth_pixel, 1);
        }

        const int idx = i - static_cast<int>(lane_idx);
        const uint2 pixel_coords = {
            start_pixel_coords.x + idx % config::tile_width,
            start_pixel_coords.y + idx / config::tile_width,
        };
        const bool valid_pixel =
            pixel_coords.x < width && pixel_coords.y < height;

        if (valid_primitive && valid_pixel && lane_idx == 0 && idx >= 0 &&
            idx < config::block_size_blend) {
            const int current_shmem_index = i % 32;
            last_contributor = collected_last_contributor[current_shmem_index];
            const float4 depth_state = collected_depth_state[current_shmem_index];
            numerator_after = depth_state.x;
            transmittance = depth_state.y;
            final_transmittance = depth_state.z;
            expected_depth = depth_state.w;
            grad_depth_pixel = collected_grad_depth_pixel[current_shmem_index];
        }

        const bool skip = !valid_primitive || !valid_pixel || idx < 0 ||
                          idx >= config::block_size_blend ||
                          tile_primitive_idx >= last_contributor;
        if (skip) {
            continue;
        }

        const float2 pixel = make_float2(
            __uint2float_rn(pixel_coords.x),
            __uint2float_rn(pixel_coords.y)
        ) +
                             0.5f;
        const float2 delta = mean2d - pixel;
        const float exponent =
            -0.5f *
                (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) -
            conic.y * delta.x * delta.y;
        const float gaussian = expf(fminf(exponent, 0.0f));
        if (!config::original_opacity_interpretation &&
            gaussian < config::min_alpha_threshold) {
            continue;
        }
        const float alpha = opacity * gaussian;
        if (config::original_opacity_interpretation &&
            alpha < config::min_alpha_threshold) {
            continue;
        }

        const float blending_weight = transmittance * alpha;
        const float alpha_total = 1.0f - final_transmittance;
        const float depth_denom = fmaxf(alpha_total, expected_depth_denom_eps);
        const float grad_numerator = grad_depth_pixel / depth_denom;
        dL_ddepth_accum += blending_weight * grad_numerator;

        numerator_after -= blending_weight * depth_value;

        const float one_minus_alpha = 1.0f - alpha;
        const float one_minus_alpha_rcp =
            1.0f / fmaxf(one_minus_alpha, config::one_minus_alpha_eps);
        float dL_dalpha =
            (transmittance * depth_value -
             numerator_after * one_minus_alpha_rcp) *
            grad_numerator;
        if (alpha_total > expected_depth_denom_eps) {
            dL_dalpha -= expected_depth * final_transmittance *
                         one_minus_alpha_rcp * grad_depth_pixel / depth_denom;
        }
        dL_dopacity_accum += gaussian * dL_dalpha;

        const float gaussian_grad_helper = -alpha * dL_dalpha;
        const float3 dL_dconic = 0.5f * gaussian_grad_helper * make_float3(
            delta.x * delta.x,
            delta.x * delta.y,
            delta.y * delta.y
        );
        dL_dconic_accum += dL_dconic;
        const float2 dL_dmean2d = gaussian_grad_helper * make_float2(
            conic.x * delta.x + conic.y * delta.y,
            conic.y * delta.x + conic.z * delta.y
        );
        dL_dmean2d_accum += dL_dmean2d;

        transmittance *= one_minus_alpha;
    }

    if (valid_primitive) {
        atomicAdd(&grad_mean2d[primitive_idx].x, dL_dmean2d_accum.x);
        atomicAdd(&grad_mean2d[primitive_idx].y, dL_dmean2d_accum.y);
        atomicAdd(&grad_conic[primitive_idx], dL_dconic_accum.x);
        atomicAdd(&grad_conic[n_primitives + primitive_idx], dL_dconic_accum.y);
        atomicAdd(
            &grad_conic[2 * n_primitives + primitive_idx],
            dL_dconic_accum.z
        );
        const float dL_dopacity = proper_antialiasing
                                      ? dL_dopacity_accum
                                      : opacity * (1.0f - opacity) * dL_dopacity_accum;
        atomicAdd(&grad_opacity[primitive_idx], dL_dopacity);
        atomicAdd(&grad_primitive_depth[primitive_idx], dL_ddepth_accum);
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> depth_blend_fwd_wrapper(
    const torch::Tensor& instance_primitive_indices,
    const torch::Tensor& tile_instance_ranges,
    const torch::Tensor& tile_bucket_offsets,
    const torch::Tensor& bucket_count,
    const torch::Tensor& projected_means,
    const torch::Tensor& conic_opacity,
    const torch::Tensor& primitive_depth,
    bool proper_antialiasing,
    int width,
    int height
) {
    splatkit::faster_gs_native::check_cuda_int_tensor(
        instance_primitive_indices,
        "instance_primitive_indices"
    );
    splatkit::faster_gs_native::check_cuda_int_tensor(
        tile_instance_ranges,
        "tile_instance_ranges"
    );
    splatkit::faster_gs_native::check_cuda_int_tensor(
        tile_bucket_offsets,
        "tile_bucket_offsets"
    );
    splatkit::faster_gs_native::check_cuda_int_tensor(bucket_count, "bucket_count");
    splatkit::faster_gs_native::check_cuda_float_tensor(
        projected_means,
        "projected_means"
    );
    splatkit::faster_gs_native::check_cuda_float_tensor(
        conic_opacity,
        "conic_opacity"
    );
    splatkit::faster_gs_native::check_cuda_float_tensor(
        primitive_depth,
        "primitive_depth"
    );

    const int bucket_total = bucket_count.item<int>();
    const int grid_width = (width + config::tile_width - 1) / config::tile_width;
    const int grid_height = (height + config::tile_height - 1) / config::tile_height;
    auto float_options = projected_means.options().dtype(torch::kFloat32);

    torch::Tensor depth = torch::zeros({height, width}, float_options);
    torch::Tensor bucket_depth_prefix = torch::empty(
        {bucket_total * config::block_size_blend},
        float_options
    );

    torch::Tensor instance_indices_c = instance_primitive_indices.contiguous();
    torch::Tensor tile_ranges_c = tile_instance_ranges.contiguous();
    torch::Tensor tile_offsets_c = tile_bucket_offsets.contiguous();
    torch::Tensor projected_means_c = projected_means.contiguous();
    torch::Tensor conic_opacity_c = conic_opacity.contiguous();
    torch::Tensor primitive_depth_c = primitive_depth.contiguous();
    (void)proper_antialiasing;

    depth_blend_forward_cu<<<dim3(grid_width, grid_height, 1),
                             dim3(config::tile_width, config::tile_height, 1)>>>(
        reinterpret_cast<const uint2*>(tile_ranges_c.data_ptr<int>()),
        reinterpret_cast<const uint*>(tile_offsets_c.data_ptr<int>()),
        reinterpret_cast<const uint*>(instance_indices_c.data_ptr<int>()),
        reinterpret_cast<const float2*>(projected_means_c.data_ptr<float>()),
        reinterpret_cast<const float4*>(conic_opacity_c.data_ptr<float>()),
        primitive_depth_c.data_ptr<float>(),
        depth.data_ptr<float>(),
        bucket_depth_prefix.data_ptr<float>(),
        static_cast<uint>(width),
        static_cast<uint>(height),
        static_cast<uint>(grid_width)
    );

    return {
        depth,
        bucket_depth_prefix,
    };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> depth_blend_bwd_wrapper(
    const torch::Tensor& grad_depth,
    const torch::Tensor& depth,
    const torch::Tensor& instance_primitive_indices,
    const torch::Tensor& tile_instance_ranges,
    const torch::Tensor& tile_bucket_offsets,
    const torch::Tensor& projected_means,
    const torch::Tensor& conic_opacity,
    const torch::Tensor& primitive_depth,
    const torch::Tensor& tile_final_transmittances,
    const torch::Tensor& tile_max_n_processed,
    const torch::Tensor& tile_n_processed,
    const torch::Tensor& bucket_tile_index,
    const torch::Tensor& bucket_color_transmittance,
    const torch::Tensor& bucket_depth_prefix,
    bool proper_antialiasing,
    int width,
    int height
) {
    splatkit::faster_gs_native::check_cuda_float_tensor(grad_depth, "grad_depth");
    splatkit::faster_gs_native::check_cuda_float_tensor(depth, "depth");
    splatkit::faster_gs_native::check_cuda_int_tensor(
        instance_primitive_indices,
        "instance_primitive_indices"
    );
    splatkit::faster_gs_native::check_cuda_int_tensor(
        tile_instance_ranges,
        "tile_instance_ranges"
    );
    splatkit::faster_gs_native::check_cuda_int_tensor(
        tile_bucket_offsets,
        "tile_bucket_offsets"
    );
    splatkit::faster_gs_native::check_cuda_float_tensor(
        projected_means,
        "projected_means"
    );
    splatkit::faster_gs_native::check_cuda_float_tensor(
        conic_opacity,
        "conic_opacity"
    );
    splatkit::faster_gs_native::check_cuda_float_tensor(
        primitive_depth,
        "primitive_depth"
    );
    splatkit::faster_gs_native::check_cuda_float_tensor(
        tile_final_transmittances,
        "tile_final_transmittances"
    );
    splatkit::faster_gs_native::check_cuda_int_tensor(
        tile_max_n_processed,
        "tile_max_n_processed"
    );
    splatkit::faster_gs_native::check_cuda_int_tensor(
        tile_n_processed,
        "tile_n_processed"
    );
    splatkit::faster_gs_native::check_cuda_int_tensor(
        bucket_tile_index,
        "bucket_tile_index"
    );
    splatkit::faster_gs_native::check_cuda_float_tensor(
        bucket_color_transmittance,
        "bucket_color_transmittance"
    );
    splatkit::faster_gs_native::check_cuda_float_tensor(
        bucket_depth_prefix,
        "bucket_depth_prefix"
    );

    const int n_primitives = projected_means.size(0);
    const int n_buckets = bucket_tile_index.size(0);
    const int grid_width = (width + config::tile_width - 1) / config::tile_width;
    auto float_options = projected_means.options().dtype(torch::kFloat32);

    torch::Tensor grad_projected_means =
        torch::zeros({n_primitives, 2}, float_options);
    torch::Tensor grad_conic_helper = splatkit::faster_gs_native::get_cached_workspace(
        "depth_blend_grad_conic_helper",
        float_options,
        3LL * n_primitives
    ).view({3, n_primitives});
    torch::Tensor grad_opacity = splatkit::faster_gs_native::get_cached_workspace(
        "depth_blend_grad_opacity",
        float_options,
        n_primitives
    );
    torch::Tensor grad_primitive_depth =
        torch::zeros({n_primitives}, float_options);
    grad_conic_helper.zero_();
    grad_opacity.zero_();

    torch::Tensor grad_depth_c = grad_depth.contiguous();
    torch::Tensor depth_c = depth.contiguous();
    torch::Tensor tile_ranges_c = tile_instance_ranges.contiguous();
    torch::Tensor tile_offsets_c = tile_bucket_offsets.contiguous();
    torch::Tensor instance_indices_c = instance_primitive_indices.contiguous();
    torch::Tensor projected_means_c = projected_means.contiguous();
    torch::Tensor conic_opacity_c = conic_opacity.contiguous();
    torch::Tensor primitive_depth_c = primitive_depth.contiguous();
    torch::Tensor tile_final_transmittances_c = tile_final_transmittances.contiguous();
    torch::Tensor tile_max_n_processed_c = tile_max_n_processed.contiguous();
    torch::Tensor tile_n_processed_c = tile_n_processed.contiguous();
    torch::Tensor bucket_tile_index_c = bucket_tile_index.contiguous();
    torch::Tensor bucket_color_transmittance_c = bucket_color_transmittance.contiguous();
    torch::Tensor bucket_depth_prefix_c = bucket_depth_prefix.contiguous();

    if (n_buckets > 0) {
        depth_blend_backward_cu<<<n_buckets, 32>>>(
            reinterpret_cast<const uint2*>(tile_ranges_c.data_ptr<int>()),
            reinterpret_cast<const uint*>(tile_offsets_c.data_ptr<int>()),
            reinterpret_cast<const uint*>(instance_indices_c.data_ptr<int>()),
            reinterpret_cast<const float2*>(projected_means_c.data_ptr<float>()),
            reinterpret_cast<const float4*>(conic_opacity_c.data_ptr<float>()),
            primitive_depth_c.data_ptr<float>(),
            grad_depth_c.data_ptr<float>(),
            depth_c.data_ptr<float>(),
            tile_final_transmittances_c.data_ptr<float>(),
            reinterpret_cast<const uint*>(tile_max_n_processed_c.data_ptr<int>()),
            reinterpret_cast<const uint*>(tile_n_processed_c.data_ptr<int>()),
            reinterpret_cast<const uint*>(bucket_tile_index_c.data_ptr<int>()),
            reinterpret_cast<const float4*>(
                bucket_color_transmittance_c.data_ptr<float>()
            ),
            bucket_depth_prefix_c.data_ptr<float>(),
            reinterpret_cast<float2*>(grad_projected_means.data_ptr<float>()),
            grad_conic_helper.data_ptr<float>(),
            grad_opacity.data_ptr<float>(),
            grad_primitive_depth.data_ptr<float>(),
            static_cast<uint>(n_primitives),
            static_cast<uint>(width),
            static_cast<uint>(height),
            static_cast<uint>(grid_width),
            proper_antialiasing
        );
    }

    torch::Tensor grad_conic_opacity = torch::empty({n_primitives, 4}, float_options);
    grad_conic_opacity.narrow(1, 0, 3).copy_(grad_conic_helper.transpose(0, 1));
    grad_conic_opacity.narrow(1, 3, 1).copy_(
        grad_opacity.reshape({n_primitives, 1})
    );
    return {
        grad_projected_means,
        grad_conic_opacity,
        grad_primitive_depth,
    };
}

}  // namespace splatkit::faster_gs_depth_native
