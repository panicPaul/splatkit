// Gaussian Wrapping "ours" CUDA stage wrappers.
//
// The implementation calls the package-local staged copy of the upstream
// GaussianWrapping diff-gaussian-rasterization_ours kernels. The wrapper names
// and tensor packing follow the FasterGS native backend style.

#include <torch/extension.h>

#include "upstream/rasterize_points.h"

namespace ember_core::gaussian_wrapping::ours_native {

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
    torch::Tensor>
render_fwd_wrapper(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& sh,
    const torch::Tensor& sg_axis,
    const torch::Tensor& sg_sharpness,
    const torch::Tensor& sg_color,
    const int sh_degree,
    const int sg_degree,
    const float scale_modifier,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool require_depth,
    const bool debug) {
    auto result = RasterizeGaussiansCUDA(
        background,
        means3D,
        colors,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
        sh,
        sg_axis,
        sg_sharpness,
        sg_color,
        sh_degree,
        sg_degree,
        scale_modifier,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        image_height,
        image_width,
        campos,
        prefiltered,
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
        std::get<12>(result)
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
    torch::Tensor,
    torch::Tensor>
render_bwd_wrapper(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& sh,
    const torch::Tensor& sg_axis,
    const torch::Tensor& sg_sharpness,
    const torch::Tensor& sg_color,
    const int sh_degree,
    const int sg_degree,
    const float scale_modifier,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const torch::Tensor& grad_color,
    const torch::Tensor& grad_median_depth,
    const torch::Tensor& grad_color_square,
    const torch::Tensor& grad_depth,
    const torch::Tensor& grad_depth_square,
    const torch::Tensor& grad_alpha,
    const torch::Tensor& grad_normal,
    const torch::Tensor& alpha,
    const torch::Tensor& normal,
    const torch::Tensor& median_depth,
    const torch::Tensor& campos,
    const torch::Tensor& radii,
    const torch::Tensor& geom_buffer,
    const torch::Tensor& rendered_count,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& image_buffer,
    const torch::Tensor& tile_buffer,
    const bool require_depth,
    const bool debug) {
    return RasterizeGaussiansBackwardCUDA(
        background,
        means3D,
        colors,
        opacity,
        scales,
        rotations,
        cov3D_precomp,
        sh,
        sg_axis,
        sg_sharpness,
        sg_color,
        sh_degree,
        sg_degree,
        scale_modifier,
        viewmatrix,
        projmatrix,
        tan_fovx,
        tan_fovy,
        kernel_size,
        grad_color,
        grad_median_depth,
        grad_color_square,
        grad_depth,
        grad_depth_square,
        grad_alpha,
        grad_normal,
        alpha,
        normal,
        median_depth,
        campos,
        radii,
        geom_buffer,
        rendered_count.item<int>(),
        binning_buffer,
        image_buffer,
        tile_buffer,
        require_depth,
        debug
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
integrate_points_fwd_wrapper(
    const torch::Tensor& points3D,
    const torch::Tensor& means3D,
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
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug) {
    auto result = IntegrateGaussiansToPointsCUDA(
        points3D,
        means3D,
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
        image_height,
        image_width,
        campos,
        prefiltered,
        debug
    );
    return std::make_tuple(
        rendered_count_tensor(std::get<0>(result), means3D),
        std::get<1>(result),
        std::get<2>(result)
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sdf_fwd_wrapper(
    const torch::Tensor& points3D,
    const torch::Tensor& means3D,
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
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug) {
    auto result = evaluateSDFfromSingleView(
        points3D,
        means3D,
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
        image_height,
        image_width,
        campos,
        prefiltered,
        debug
    );
    return std::make_tuple(
        rendered_count_tensor(std::get<0>(result), means3D),
        std::get<1>(result),
        std::get<2>(result),
        std::get<3>(result)
    );
}

}  // namespace ember_core::gaussian_wrapping::ours_native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace ember_core::gaussian_wrapping::ours_native;
    m.def("render_fwd", &render_fwd_wrapper);
    m.def("render_bwd", &render_bwd_wrapper);
    m.def("integrate_points_fwd", &integrate_points_fwd_wrapper);
    m.def("sdf_fwd", &sdf_fwd_wrapper);
    m.def("sample_depth_fwd", &SampleRasterizedDepthCUDA);
    m.def("sample_depth_bwd", &SampleRasterizedDepthBackwardCUDA);
    m.def("mark_visible", &markVisible);
}
