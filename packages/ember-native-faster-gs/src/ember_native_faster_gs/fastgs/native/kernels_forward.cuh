#pragma once

#include "rasterization_config.h"
#include "kernel_utils.cuh"
#include "sh_utils.cuh"
#include "buffer_utils.h"
#include "helper_math.h"
#include "utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace ember_fastgs::rasterization::kernels::forward {

    using namespace ember_faster_gs_core_vendor::rasterization;
    using namespace ember_faster_gs_core_vendor::rasterization::kernels;
    namespace config = ember_faster_gs_core_vendor::rasterization::config;

    __device__ inline float2 compute_compact_box_intersection(
        const float4 conic_opacity,
        const float discriminant,
        const float compact_power_threshold,
        const float2 mean,
        const bool slice_y,
        const float coord)
    {
        const float mean_u = slice_y ? mean.y : mean.x;
        const float mean_v = slice_y ? mean.x : mean.y;
        const float coeff = slice_y ? conic_opacity.x : conic_opacity.z;
        const float h = coord - mean_u;
        const float sqrt_term = sqrtf(
            discriminant * h * h + compact_power_threshold * coeff
        );
        return make_float2(
            (-conic_opacity.y * h - sqrt_term) / coeff + mean_v,
            (-conic_opacity.y * h + sqrt_term) / coeff + mean_v
        );
    }

    __device__ inline uint4 compute_compact_box_bounds(
        const float2 bbox_min,
        const float2 bbox_max,
        const uint grid_width,
        const uint grid_height)
    {
        return make_uint4(
            min(
                grid_width,
                static_cast<uint>(max(
                    0,
                    __float2int_rd(
                        bbox_min.x / static_cast<float>(config::tile_width)
                    )
                ))
            ),
            min(
                grid_width,
                static_cast<uint>(max(
                    0,
                    __float2int_rd(
                        bbox_max.x / static_cast<float>(config::tile_width) + 1.0f
                    )
                ))
            ),
            min(
                grid_height,
                static_cast<uint>(max(
                    0,
                    __float2int_rd(
                        bbox_min.y / static_cast<float>(config::tile_height)
                    )
                ))
            ),
            min(
                grid_height,
                static_cast<uint>(max(
                    0,
                    __float2int_rd(
                        bbox_max.y / static_cast<float>(config::tile_height) + 1.0f
                    )
                ))
            )
        );
    }

    struct CompactBoxGeometry {
        bool active;
        float discriminant;
        float compact_power_threshold;
        float2 bbox_min;
        float2 bbox_max;
        float2 bbox_argmin;
        float2 bbox_argmax;
        uint4 screen_bounds;
    };

    __device__ inline CompactBoxGeometry compute_compact_box_geometry(
        const float2 mean,
        const float4 conic_opacity,
        const uint grid_width,
        const uint grid_height,
        const float compact_box_scale)
    {
        CompactBoxGeometry geometry{};
        geometry.active = false;
        const float discriminant =
            conic_opacity.y * conic_opacity.y -
            conic_opacity.x * conic_opacity.z;
        if (
            conic_opacity.x <= 0.0f ||
            conic_opacity.z <= 0.0f ||
            discriminant >= 0.0f
        ) {
            return geometry;
        }

        const float compact_power_threshold =
            compact_box_scale *
            2.0f *
            logf(conic_opacity.w * config::min_alpha_threshold_rcp);
        const float x_term = copysignf(
            sqrtf(
                -(conic_opacity.y * conic_opacity.y * compact_power_threshold) /
                (discriminant * conic_opacity.x)
            ),
            -conic_opacity.y
        );
        const float y_term = copysignf(
            sqrtf(
                -(conic_opacity.y * conic_opacity.y * compact_power_threshold) /
                (discriminant * conic_opacity.z)
            ),
            -conic_opacity.y
        );

        geometry.discriminant = discriminant;
        geometry.compact_power_threshold = compact_power_threshold;
        geometry.bbox_argmin = make_float2(mean.y - y_term, mean.x - x_term);
        geometry.bbox_argmax = make_float2(mean.y + y_term, mean.x + x_term);
        geometry.bbox_min = make_float2(
            compute_compact_box_intersection(
                conic_opacity,
                discriminant,
                compact_power_threshold,
                mean,
                true,
                geometry.bbox_argmin.x
            ).x,
            compute_compact_box_intersection(
                conic_opacity,
                discriminant,
                compact_power_threshold,
                mean,
                false,
                geometry.bbox_argmin.y
            ).x
        );
        geometry.bbox_max = make_float2(
            compute_compact_box_intersection(
                conic_opacity,
                discriminant,
                compact_power_threshold,
                mean,
                true,
                geometry.bbox_argmax.x
            ).y,
            compute_compact_box_intersection(
                conic_opacity,
                discriminant,
                compact_power_threshold,
                mean,
                false,
                geometry.bbox_argmax.y
            ).y
        );
        geometry.screen_bounds = compute_compact_box_bounds(
            geometry.bbox_min,
            geometry.bbox_max,
            grid_width,
            grid_height
        );
        geometry.active =
            geometry.screen_bounds.x < geometry.screen_bounds.y &&
            geometry.screen_bounds.z < geometry.screen_bounds.w;
        return geometry;
    }

    template <typename KeyT>
    __device__ inline uint process_compact_box_tiles(
        const CompactBoxGeometry geometry,
        const float2 mean,
        const float4 conic_opacity,
        const uint grid_width,
        const uint primitive_idx,
        uint write_offset,
        KeyT* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices)
    {
        if (!geometry.active) return 0;
        int2 rect_min = make_int2(
            static_cast<int>(geometry.screen_bounds.x),
            static_cast<int>(geometry.screen_bounds.z)
        );
        int2 rect_max = make_int2(
            static_cast<int>(geometry.screen_bounds.y),
            static_cast<int>(geometry.screen_bounds.w)
        );
        float2 bbox_min = geometry.bbox_min;
        float2 bbox_max = geometry.bbox_max;
        float2 bbox_argmin = geometry.bbox_argmin;
        float2 bbox_argmax = geometry.bbox_argmax;
        const int y_span = rect_max.y - rect_min.y;
        const int x_span = rect_max.x - rect_min.x;
        const bool slice_y = y_span < x_span;
        if (slice_y) {
            rect_min = make_int2(rect_min.y, rect_min.x);
            rect_max = make_int2(rect_max.y, rect_max.x);
            bbox_min = make_float2(bbox_min.y, bbox_min.x);
            bbox_max = make_float2(bbox_max.y, bbox_max.x);
            bbox_argmin = make_float2(bbox_argmin.y, bbox_argmin.x);
            bbox_argmax = make_float2(bbox_argmax.y, bbox_argmax.x);
        }

        uint tiles_count = 0;
        const float block_u = slice_y
                                  ? static_cast<float>(config::tile_height)
                                  : static_cast<float>(config::tile_width);
        const float block_v = slice_y
                                  ? static_cast<float>(config::tile_width)
                                  : static_cast<float>(config::tile_height);
        float2 intersect_max_line = make_float2(bbox_max.y, bbox_min.y);
        float min_line = static_cast<float>(rect_min.x) * block_u;
        float2 intersect_min_line =
            bbox_min.x <= min_line
                ? compute_compact_box_intersection(
                      conic_opacity,
                      geometry.discriminant,
                      geometry.compact_power_threshold,
                      mean,
                      slice_y,
                      min_line
                  )
                : intersect_max_line;

        for (int u = rect_min.x; u < rect_max.x; ++u) {
            const float max_line = min_line + block_u;
            if (max_line <= bbox_max.x) {
                intersect_max_line = compute_compact_box_intersection(
                    conic_opacity,
                    geometry.discriminant,
                    geometry.compact_power_threshold,
                    mean,
                    slice_y,
                    max_line
                );
            }
            const float ellipse_min =
                min_line <= bbox_argmin.y && bbox_argmin.y < max_line
                    ? bbox_min.y
                    : min(intersect_min_line.x, intersect_max_line.x);
            const float ellipse_max =
                min_line <= bbox_argmax.y && bbox_argmax.y < max_line
                    ? bbox_max.y
                    : max(intersect_min_line.y, intersect_max_line.y);
            const int min_tile_v = max(
                rect_min.y,
                min(rect_max.y, static_cast<int>(ellipse_min / block_v))
            );
            const int max_tile_v = min(
                rect_max.y,
                max(rect_min.y, static_cast<int>(ellipse_max / block_v + 1.0f))
            );

            tiles_count += static_cast<uint>(max_tile_v - min_tile_v);
            if (instance_keys != nullptr) {
                for (int v = min_tile_v; v < max_tile_v; ++v) {
                    const uint tile_idx =
                        slice_y
                            ? static_cast<uint>(u) * grid_width + static_cast<uint>(v)
                            : static_cast<uint>(v) * grid_width + static_cast<uint>(u);
                    instance_keys[write_offset] = static_cast<KeyT>(tile_idx);
                    instance_primitive_indices[write_offset] = primitive_idx;
                    write_offset++;
                }
            }
            intersect_min_line = intersect_max_line;
            min_line = max_line;
        }
        return tiles_count;
    }

    __global__ void preprocess_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ scales,
        const float4* __restrict__ rotations,
        const float* __restrict__ opacities,
        const float3* __restrict__ sh_coefficients_0,
        const float3* __restrict__ sh_coefficients_rest,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        uint* __restrict__ primitive_depth_keys,
        uint* __restrict__ primitive_indices,
        uint* __restrict__ primitive_n_touched_tiles,
        ushort4* __restrict__ primitive_screen_bounds,
        float2* __restrict__ primitive_mean2d,
        float4* __restrict__ primitive_conic_opacity,
        float3* __restrict__ primitive_color,
        uint* __restrict__ n_visible_primitives,
        uint* __restrict__ n_instances,
        const uint n_primitives,
        const uint grid_width,
        const uint grid_height,
        const uint active_sh_bases,
        const uint total_sh_bases,
        const float width,
        const float height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing,
        const float compact_box_scale)
    {
        constexpr uint warp_size = 32;
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<warp_size>(block);
        const uint thread_idx = cg::this_grid().thread_rank();

        bool active = true;
        uint primitive_idx = thread_idx;
        if (primitive_idx >= n_primitives) {
            active = false;
            primitive_idx = n_primitives - 1;
        }

        if (active) primitive_n_touched_tiles[primitive_idx] = 0;

        const float3 mean3d = means[primitive_idx];

        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        if (depth < near_plane || depth > far_plane) active = false;

        if (warp.ballot(active) == 0) return;

        const float raw_opacity = opacities[primitive_idx];
        float opacity = sigmoid(raw_opacity);
        if (config::original_opacity_interpretation && opacity < config::min_alpha_threshold) active = false;

        const float3 raw_scale = scales[primitive_idx];
        const float3 variance = expf(2.0f * raw_scale);
        const float4 raw_rotation = rotations[primitive_idx];
        float quaternion_norm_sq = 1.0f;
        const mat3x3 R = convert_quaternion_to_rotation_matrix(raw_rotation, quaternion_norm_sq);
        if (quaternion_norm_sq < 1e-8f) active = false;
        const mat3x3 RSS = {
            R.m11 * variance.x, R.m12 * variance.y, R.m13 * variance.z,
            R.m21 * variance.x, R.m22 * variance.y, R.m23 * variance.z,
            R.m31 * variance.x, R.m32 * variance.y, R.m33 * variance.z
        };
        const mat3x3_triu cov3d {
            RSS.m11 * R.m11 + RSS.m12 * R.m12 + RSS.m13 * R.m13,
            RSS.m11 * R.m21 + RSS.m12 * R.m22 + RSS.m13 * R.m23,
            RSS.m11 * R.m31 + RSS.m12 * R.m32 + RSS.m13 * R.m33,
            RSS.m21 * R.m21 + RSS.m22 * R.m22 + RSS.m23 * R.m23,
            RSS.m21 * R.m31 + RSS.m22 * R.m32 + RSS.m23 * R.m33,
            RSS.m31 * R.m31 + RSS.m32 * R.m32 + RSS.m33 * R.m33,
        };

        const float4 w2c_r1 = w2c[0];
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
        const float4 w2c_r2 = w2c[1];
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

        const float clip_left = (-0.15f * width - center_x) / focal_x;
        const float clip_right = (1.15f * width - center_x) / focal_x;
        const float clip_top = (-0.15f * height - center_y) / focal_y;
        const float clip_bottom = (1.15f * height - center_y) / focal_y;
        const float x_clipped = clamp(x, clip_left, clip_right);
        const float y_clipped = clamp(y, clip_top, clip_bottom);
        const float j11 = focal_x / depth;
        const float j13 = -j11 * x_clipped;
        const float j22 = focal_y / depth;
        const float j23 = -j22 * y_clipped;
        const float3 jw_r1 = make_float3(
            j11 * w2c_r1.x + j13 * w2c_r3.x,
            j11 * w2c_r1.y + j13 * w2c_r3.y,
            j11 * w2c_r1.z + j13 * w2c_r3.z
        );
        const float3 jw_r2 = make_float3(
            j22 * w2c_r2.x + j23 * w2c_r3.x,
            j22 * w2c_r2.y + j23 * w2c_r3.y,
            j22 * w2c_r2.z + j23 * w2c_r3.z
        );
        const float3 jwc_r1 = make_float3(
            jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
            jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
            jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33
        );
        const float3 jwc_r2 = make_float3(
            jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
            jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
            jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33
        );
        float3 cov2d = make_float3(
            dot(jwc_r1, jw_r1),
            dot(jwc_r1, jw_r2),
            dot(jwc_r2, jw_r2)
        );
        const float determinant_raw = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        const float kernel_size = proper_antialiasing ? config::dilation_proper_antialiasing : config::dilation;
        cov2d.x += kernel_size;
        cov2d.z += kernel_size;
        const float determinant = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        if (determinant < config::min_cov2d_determinant) active = false;
        const float3 conic = make_float3(
            cov2d.z / determinant,
            -cov2d.y / determinant,
            cov2d.x / determinant
        );
        if (proper_antialiasing) {
            opacity *= sqrtf(fmaxf(determinant_raw / determinant, 0.0f));
            if (config::original_opacity_interpretation && opacity < config::min_alpha_threshold) active = false;
        }

        const float2 mean2d = make_float2(
            x * focal_x + center_x,
            y * focal_y + center_y
        );

        const float4 conic_opacity = make_float4(conic, opacity);
        const CompactBoxGeometry compact_box = compute_compact_box_geometry(
            mean2d,
            conic_opacity,
            grid_width,
            grid_height,
            compact_box_scale
        );
        if (!compact_box.active) active = false;

        if (warp.ballot(active) == 0) return;

        const uint n_touched_tiles = process_compact_box_tiles<uint>(
            compact_box,
            mean2d,
            conic_opacity,
            grid_width,
            primitive_idx,
            0,
            nullptr,
            nullptr
        );

        if (n_touched_tiles == 0 || !active) return;

        primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
        primitive_screen_bounds[primitive_idx] = make_ushort4(
            static_cast<ushort>(compact_box.screen_bounds.x),
            static_cast<ushort>(compact_box.screen_bounds.y),
            static_cast<ushort>(compact_box.screen_bounds.z),
            static_cast<ushort>(compact_box.screen_bounds.w)
        );
        primitive_mean2d[primitive_idx] = mean2d;
        primitive_conic_opacity[primitive_idx] = conic_opacity;
        const float3 color = convert_sh_to_color(
            sh_coefficients_0, sh_coefficients_rest,
            mean3d, cam_position[0],
            primitive_idx, active_sh_bases, total_sh_bases
        );
        primitive_color[primitive_idx] = color;

        const uint offset = atomicAdd(n_visible_primitives, 1);
        const uint depth_key = __float_as_uint(depth);
        primitive_depth_keys[offset] = depth_key;
        primitive_indices[offset] = primitive_idx;
        atomicAdd(n_instances, n_touched_tiles);
    }

    template <typename KeyT>
    __global__ void create_instances_cu(
        const uint* __restrict__ primitive_indices_sorted,
        const uint* __restrict__ primitive_offsets,
        const ushort4* __restrict__ primitive_screen_bounds,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        KeyT* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        const uint grid_width,
        const uint grid_height,
        const uint n_visible_primitives,
        const float compact_box_scale)
    {
        constexpr uint warp_size = 32;
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<warp_size>(block);
        const uint thread_idx = cg::this_grid().thread_rank();

        uint original_idx = thread_idx;
        bool active = true;
        if (original_idx >= n_visible_primitives) {
            active = false;
            original_idx = n_visible_primitives - 1;
        }

        if (warp.ballot(active) == 0) return;

        const uint primitive_idx = primitive_indices_sorted[original_idx];

        (void)primitive_screen_bounds;
        const float2 mean2d = primitive_mean2d[primitive_idx];
        const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
        const CompactBoxGeometry compact_box = compute_compact_box_geometry(
            mean2d,
            conic_opacity,
            grid_width,
            grid_height,
            compact_box_scale
        );
        uint current_write_offset = primitive_offsets[original_idx];
        process_compact_box_tiles<KeyT>(
            compact_box,
            mean2d,
            conic_opacity,
            grid_width,
            primitive_idx,
            current_write_offset,
            instance_keys,
            instance_primitive_indices
        );
    }

}  // namespace ember_fastgs::rasterization::kernels::forward
