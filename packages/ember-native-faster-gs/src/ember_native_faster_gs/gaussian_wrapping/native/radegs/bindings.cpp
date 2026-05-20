// Gaussian Wrapping RaDe-GS CUDA stage wrappers.
//
// The implementation calls the package-local staged copy of the upstream
// GaussianWrapping diff-gaussian-rasterization kernels. The wrapper names and
// tensor packing follow the FasterGS native backend style.

#include <torch/extension.h>

#include "upstream/rasterize_points.h"

namespace ember_core::gaussian_wrapping::radegs_native {

torch::Tensor rendered_count_tensor(int rendered, const torch::Tensor& reference) {
    auto rendered_count = torch::empty(
        {1},
        reference.options().dtype(torch::kInt32)
    );
    rendered_count.fill_(rendered);
    return rendered_count;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
render_fwd_wrapper(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const int image_height,
    const int image_width,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool require_coord,
    const bool require_depth,
    const bool debug) {
    auto result = RasterizeGaussiansCUDA(
        background,
        means3D,
        colors,
        opacity,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        image_height,
        image_width,
        sh,
        degree,
        campos,
        prefiltered,
        require_coord,
        require_depth,
        debug
    );
    return std::make_tuple(
        rendered_count_tensor(std::get<0>(result), means3D),
        std::get<1>(result),
        std::get<2>(result),
        std::get<3>(result),
        std::get<4>(result),
        std::get<5>(result),
        std::get<6>(result),
        std::get<7>(result),
        std::get<8>(result),
        std::get<9>(result),
        std::get<10>(result),
        std::get<11>(result),
        std::get<12>(result),
        std::get<13>(result),
        std::get<14>(result)
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
render_bwd_wrapper(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const torch::Tensor& grad_color,
    const torch::Tensor& grad_coord,
    const torch::Tensor& grad_median_coord,
    const torch::Tensor& grad_depth,
    const torch::Tensor& grad_median_depth,
    const torch::Tensor& grad_color_square,
    const torch::Tensor& grad_depth_sum,
    const torch::Tensor& grad_depth_square,
    const torch::Tensor& grad_alpha,
    const torch::Tensor& grad_normal,
    const torch::Tensor& normal,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& geom_buffer,
    const torch::Tensor& rendered_count,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& image_buffer,
    const torch::Tensor& alpha,
    const bool require_coord,
    const bool require_depth,
    const bool debug) {
    return RasterizeGaussiansBackwardCUDA(
        background,
        means3D,
        radii,
        colors,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        grad_color,
        grad_coord,
        grad_median_coord,
        grad_depth,
        grad_median_depth,
        grad_color_square,
        grad_depth_sum,
        grad_depth_square,
        grad_alpha,
        grad_normal,
        normal,
        sh,
        degree,
        campos,
        geom_buffer,
        rendered_count.item<int>(),
        binning_buffer,
        image_buffer,
        alpha,
        require_coord,
        require_depth,
        debug
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
integrate_points_fwd_wrapper(
    const torch::Tensor& background,
    const torch::Tensor& points3D,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& view2gaussian_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const torch::Tensor& subpixel_offset,
    const int image_height,
    const int image_width,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug) {
    auto result = IntegrateGaussiansToPointsCUDA(
        background,
        points3D,
        means3D,
        colors,
        opacity,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        view2gaussian_precomp,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        subpixel_offset,
        image_height,
        image_width,
        sh,
        degree,
        campos,
        prefiltered,
        debug
    );
    return std::make_tuple(
        rendered_count_tensor(std::get<0>(result), means3D),
        std::get<1>(result),
        std::get<2>(result),
        std::get<3>(result),
        std::get<4>(result),
        std::get<5>(result),
        std::get<6>(result),
        std::get<7>(result),
        std::get<8>(result),
        std::get<9>(result)
    );
}

}  // namespace ember_core::gaussian_wrapping::radegs_native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace ember_core::gaussian_wrapping::radegs_native;
    m.def("render_fwd", &render_fwd_wrapper);
    m.def("render_bwd", &render_bwd_wrapper);
    m.def("integrate_points_fwd", &integrate_points_fwd_wrapper);
    m.def("mark_visible", &markVisible);
}
