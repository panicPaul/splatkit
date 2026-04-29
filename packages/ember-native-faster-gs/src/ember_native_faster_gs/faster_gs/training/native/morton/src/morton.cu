#include "morton.h"
#include "morton_config.h"
#include "helper_math.h"
#include "torch_utils.h"
#include "utils.h"

#include <cstdint>

namespace faster_gs::morton {

__device__ __forceinline__ uint32_t expand_bits(uint32_t value) {
    value = (value * 0x00010001u) & 0xFF0000FFu;
    value = (value * 0x00000101u) & 0x0F00F00Fu;
    value = (value * 0x00000011u) & 0xC30C30C3u;
    value = (value * 0x00000005u) & 0x49249249u;
    return value;
}

__device__ __forceinline__ int64_t morton_3d(
    const uint32_t x,
    const uint32_t y,
    const uint32_t z)
{
    return static_cast<int64_t>(
        expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2));
}

__global__ void morton_codes_cu(
    const float3* positions,
    const float* scene_min,
    const float scene_extent,
    int64_t* codes,
    const int n_points)
{
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= n_points) return;

    const float3 position = positions[point_idx];
    const float3 min_corner = make_float3(scene_min[0], scene_min[1], scene_min[2]);
    const float inv_extent = 1.0f / fmaxf(scene_extent, 1.0e-12f);
    const float3 normalized = make_float3(
        fminf(fmaxf((position.x - min_corner.x) * inv_extent, 0.0f), 1.0f),
        fminf(fmaxf((position.y - min_corner.y) * inv_extent, 0.0f), 1.0f),
        fminf(fmaxf((position.z - min_corner.z) * inv_extent, 0.0f), 1.0f));
    const uint32_t x = static_cast<uint32_t>(normalized.x * 1023.0f);
    const uint32_t y = static_cast<uint32_t>(normalized.y * 1023.0f);
    const uint32_t z = static_cast<uint32_t>(normalized.z * 1023.0f);
    codes[point_idx] = morton_3d(x, y, z);
}

torch::Tensor morton_codes_wrapper(
    const torch::Tensor& positions,
    const torch::Tensor& scene_min,
    const float scene_extent)
{
    CHECK_INPUT(config::debug, positions, "positions");
    CHECK_INPUT(config::debug, scene_min, "scene_min");
    if (config::debug) {
        if (positions.dim() != 2 || positions.size(1) != 3) {
            throw std::runtime_error("positions must have shape (num_points, 3).");
        }
        if (scene_min.numel() != 3) {
            throw std::runtime_error("scene_min must contain exactly 3 values.");
        }
    }

    const int n_points = positions.size(0);
    const auto options = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(positions.device());
    torch::Tensor codes = torch::empty({n_points}, options);
    morton_codes_cu<<<
        div_round_up(n_points, config::block_size_morton_encode),
        config::block_size_morton_encode>>>(
        reinterpret_cast<const float3*>(positions.data_ptr<float>()),
        scene_min.data_ptr<float>(),
        scene_extent,
        codes.data_ptr<int64_t>(),
        n_points);
    CHECK_CUDA(config::debug, "morton_codes_cu");
    return codes;
}

}
