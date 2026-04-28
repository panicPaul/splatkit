// Binds the native GaussianPOP blend helper used by the backend-owned blend
// stage.

#include <torch/extension.h>

namespace ember_core::gaussian_pop_native {

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
    bool return_gaussian_impact_score);

}  // namespace ember_core::gaussian_pop_native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pop_blend_fwd", &ember_core::gaussian_pop_native::pop_blend_fwd_wrapper);
}
