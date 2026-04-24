#include <torch/extension.h>

#include <tuple>

#include "common.h"
#include "helper_math.h"
#include "utils.h"
#include "vendor_namespace_begin.h"
#include "rasterization_config.h"
#include "vendor_namespace_end.h"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace config = splatkit_faster_gs_core_vendor::rasterization::config;

namespace splatkit::gaussian_pop_native {

namespace {

constexpr float expected_depth_denom_eps = 1e-10f;
constexpr float gaussian_impact_eps = 1e-9f;

__global__ void __launch_bounds__(config::block_size_blend) pop_blend_forward_cu(
    const uint2* __restrict__ tile_instance_ranges,
    const uint* __restrict__ tile_bucket_offsets,
    const uint* __restrict__ instance_primitive_indices,
    const float2* __restrict__ primitive_mean2d,
    const float4* __restrict__ primitive_conic_opacity,
    const float3* __restrict__ primitive_color,
    const float* __restrict__ primitive_depth,
    const float3* __restrict__ bg_color,
    float* __restrict__ image,
    float* __restrict__ depth,
    float* __restrict__ tile_final_transmittances,
    uint* __restrict__ tile_max_n_processed,
    uint* __restrict__ tile_n_processed,
    uint* __restrict__ bucket_tile_index,
    float4* __restrict__ bucket_color_transmittance,
    float* __restrict__ bucket_depth_prefix,
    float* __restrict__ gaussian_impact_score,
    const uint width,
    const uint height,
    const uint grid_width,
    const bool return_depth,
    const bool return_gaussian_impact_score
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
    const int n_buckets = div_round_up(n_points_total, 32);
    uint bucket_offset = (tile_idx == 0) ? 0 : tile_bucket_offsets[tile_idx - 1];

    for (int n_buckets_remaining = n_buckets, current_bucket_idx = thread_rank;
         n_buckets_remaining > 0;
         n_buckets_remaining -= config::block_size_blend,
             current_bucket_idx += config::block_size_blend) {
        if (current_bucket_idx < n_buckets) {
            bucket_tile_index[bucket_offset + current_bucket_idx] = tile_idx;
        }
    }

    __shared__ uint collected_primitive_idx[config::block_size_blend];
    __shared__ float2 collected_mean2d[config::block_size_blend];
    __shared__ float4 collected_conic_opacity[config::block_size_blend];
    __shared__ float3 collected_color[config::block_size_blend];
    __shared__ float collected_depth[config::block_size_blend];
    __shared__ uint tile_processed_max[config::block_size_blend];

    float3 color_pixel = make_float3(0.0f);
    float depth_numerator = 0.0f;
    float transmittance = 1.0f;
    uint n_processed = 0;
    uint n_processed_and_used = 0;
    bool done = !inside;
    uint color_bucket_offset = bucket_offset;
    uint depth_bucket_offset = bucket_offset;

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
            collected_primitive_idx[thread_rank] = primitive_idx;
            collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
            collected_conic_opacity[thread_rank] =
                primitive_conic_opacity[primitive_idx];
            collected_color[thread_rank] =
                fmaxf(primitive_color[primitive_idx], 0.0f);
            if (return_depth) {
                collected_depth[thread_rank] = primitive_depth[primitive_idx];
            }
        }
        block.sync();

        const int current_batch_size =
            min(config::block_size_blend, n_points_remaining);
        for (int j = 0; !done && j < current_batch_size; ++j) {
            if (j % 32 == 0) {
                bucket_color_transmittance
                    [color_bucket_offset * config::block_size_blend + thread_rank] =
                        make_float4(color_pixel, transmittance);
                color_bucket_offset++;
                if (return_depth) {
                    bucket_depth_prefix
                        [depth_bucket_offset * config::block_size_blend +
                         thread_rank] = depth_numerator;
                    depth_bucket_offset++;
                }
            }

            n_processed++;

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

            const float blending_weight = transmittance * alpha;
            color_pixel += blending_weight * collected_color[j];
            if (return_depth) {
                depth_numerator += blending_weight * collected_depth[j];
            }
            transmittance *= 1.0f - alpha;
            n_processed_and_used = n_processed;

            if (transmittance < config::transmittance_threshold) {
                done = true;
            }
        }
    }

    float3 final_color = make_float3(0.0f);
    if (inside) {
        color_pixel += transmittance * bg_color[0];
        final_color = color_pixel;
        const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
        const uint n_pixels = width * height;
        image[pixel_idx] = color_pixel.x;
        image[n_pixels + pixel_idx] = color_pixel.y;
        image[2 * n_pixels + pixel_idx] = color_pixel.z;
        tile_final_transmittances[pixel_idx] = transmittance;
        tile_n_processed[pixel_idx] = n_processed_and_used;

        if (return_depth) {
            const float alpha_total = 1.0f - transmittance;
            const float depth_denom = fmaxf(alpha_total, expected_depth_denom_eps);
            depth[pixel_idx] = depth_numerator / depth_denom;
        }
    }

    tile_processed_max[thread_rank] = n_processed_and_used;
    block.sync();
    for (uint stride = config::block_size_blend / 2; stride > 0; stride /= 2) {
        if (thread_rank < stride) {
            tile_processed_max[thread_rank] = max(
                tile_processed_max[thread_rank],
                tile_processed_max[thread_rank + stride]
            );
        }
        block.sync();
    }
    if (thread_rank == 0) {
        tile_max_n_processed[tile_idx] = tile_processed_max[0];
    }

    if (!inside || !return_gaussian_impact_score) {
        return;
    }

    float3 prefix_color = make_float3(0.0f);
    transmittance = 1.0f;
    done = false;

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
            collected_primitive_idx[thread_rank] = primitive_idx;
            collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
            collected_conic_opacity[thread_rank] =
                primitive_conic_opacity[primitive_idx];
            collected_color[thread_rank] =
                fmaxf(primitive_color[primitive_idx], 0.0f);
        }
        block.sync();

        const int current_batch_size =
            min(config::block_size_blend, n_points_remaining);
        for (int j = 0; !done && j < current_batch_size; ++j) {
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

            const float effective_weight = transmittance * alpha;
            const float3 color = collected_color[j];
            prefix_color += effective_weight * color;
            const float transmittance_after = transmittance * (1.0f - alpha);
            const float3 background =
                (final_color - prefix_color) /
                (transmittance_after + gaussian_impact_eps);
            const float3 color_delta = color - background;
            const float3 weighted_delta = effective_weight * color_delta;
            atomicAdd(
                &gaussian_impact_score[collected_primitive_idx[j]],
                dot(weighted_delta, weighted_delta)
            );

            transmittance = transmittance_after;
            if (transmittance < config::transmittance_threshold) {
                done = true;
            }
        }
    }
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
pop_blend_fwd_wrapper(
    const torch::Tensor& instance_primitive_indices,
    const torch::Tensor& tile_instance_ranges,
    const torch::Tensor& tile_bucket_offsets,
    const torch::Tensor& bucket_count,
    const torch::Tensor& projected_means,
    const torch::Tensor& conic_opacity,
    const torch::Tensor& colors_rgb,
    const torch::Tensor& primitive_depth,
    const torch::Tensor& bg_color,
    bool proper_antialiasing,
    int width,
    int height,
    bool return_depth,
    bool return_gaussian_impact_score
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
    splatkit::faster_gs_native::check_cuda_float_tensor(colors_rgb, "colors_rgb");
    if (return_depth) {
        splatkit::faster_gs_native::check_cuda_float_tensor(
            primitive_depth,
            "primitive_depth"
        );
    }
    splatkit::faster_gs_native::check_cuda_float_tensor(bg_color, "bg_color");

    const int tile_count = tile_instance_ranges.size(0);
    const int bucket_total = bucket_count.item<int>();
    const int grid_width = div_round_up(width, config::tile_width);
    const int grid_height = div_round_up(height, config::tile_height);
    auto float_options = projected_means.options().dtype(torch::kFloat32);
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
    torch::Tensor depth = return_depth
                              ? torch::empty({height, width}, float_options)
                              : torch::empty({0}, float_options);
    torch::Tensor bucket_depth_prefix = return_depth
                                            ? torch::empty(
                                                  {bucket_total * config::block_size_blend},
                                                  float_options
                                              )
                                            : torch::empty({0}, float_options);
    torch::Tensor gaussian_impact_score = return_gaussian_impact_score
                                              ? torch::zeros(
                                                    {projected_means.size(0)},
                                                    float_options
                                                )
                                              : torch::empty({0}, float_options);

    torch::Tensor instance_indices_c = instance_primitive_indices.contiguous();
    torch::Tensor tile_ranges_c = tile_instance_ranges.contiguous();
    torch::Tensor tile_offsets_c = tile_bucket_offsets.contiguous();
    torch::Tensor projected_means_c = projected_means.contiguous();
    torch::Tensor conic_opacity_c = conic_opacity.contiguous();
    torch::Tensor colors_rgb_c = colors_rgb.contiguous();
    torch::Tensor primitive_depth_c = primitive_depth.contiguous();
    torch::Tensor bg_color_c = bg_color.contiguous();
    (void)proper_antialiasing;

    pop_blend_forward_cu<<<dim3(grid_width, grid_height, 1),
                           dim3(config::tile_width, config::tile_height, 1)>>>(
        reinterpret_cast<const uint2*>(tile_ranges_c.data_ptr<int>()),
        reinterpret_cast<const uint*>(tile_offsets_c.data_ptr<int>()),
        reinterpret_cast<const uint*>(instance_indices_c.data_ptr<int>()),
        reinterpret_cast<const float2*>(projected_means_c.data_ptr<float>()),
        reinterpret_cast<const float4*>(conic_opacity_c.data_ptr<float>()),
        reinterpret_cast<const float3*>(colors_rgb_c.data_ptr<float>()),
        primitive_depth_c.data_ptr<float>(),
        reinterpret_cast<const float3*>(bg_color_c.data_ptr<float>()),
        image.data_ptr<float>(),
        depth.data_ptr<float>(),
        tile_final_transmittances.data_ptr<float>(),
        reinterpret_cast<uint*>(tile_max_n_processed.data_ptr<int>()),
        reinterpret_cast<uint*>(tile_n_processed.data_ptr<int>()),
        reinterpret_cast<uint*>(bucket_tile_index.data_ptr<int>()),
        reinterpret_cast<float4*>(bucket_color_transmittance.data_ptr<float>()),
        bucket_depth_prefix.data_ptr<float>(),
        gaussian_impact_score.data_ptr<float>(),
        static_cast<uint>(width),
        static_cast<uint>(height),
        static_cast<uint>(grid_width),
        return_depth,
        return_gaussian_impact_score
    );
    CHECK_CUDA(config::debug, "pop_blend_fwd")

    return {
        image,
        depth,
        tile_final_transmittances,
        tile_max_n_processed,
        tile_n_processed,
        bucket_tile_index,
        bucket_color_transmittance,
        bucket_depth_prefix,
        gaussian_impact_score,
    };
}

}  // namespace splatkit::gaussian_pop_native
