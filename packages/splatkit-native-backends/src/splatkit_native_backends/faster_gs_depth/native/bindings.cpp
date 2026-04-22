// Binds the native depth-only blend helpers used by the FasterGS depth backend.

#include <torch/extension.h>

namespace splatkit::faster_gs_depth_native {

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
depth_blend_fwd_wrapper(
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
    int height);

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
    int height);

}  // namespace splatkit::faster_gs_depth_native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "depth_blend_fwd",
        &splatkit::faster_gs_depth_native::depth_blend_fwd_wrapper
    );
    m.def(
        "depth_blend_bwd",
        &splatkit::faster_gs_depth_native::depth_blend_bwd_wrapper
    );
}
