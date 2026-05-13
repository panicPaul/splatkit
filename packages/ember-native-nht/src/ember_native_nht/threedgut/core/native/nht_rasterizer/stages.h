#pragma once

// Declares the NHT rasterizer stage wrappers exposed to Python.

#include <torch/extension.h>

#include "Ops.h"

namespace ember_native_nht::nht_rasterizer {

// Runs the 3DGUT unscented-transform projection stage.
inline auto project_fwd(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::optional<at::Tensor> opacities,
    const at::Tensor& viewmats0,
    const at::optional<at::Tensor> viewmats1,
    const at::Tensor& Ks,
    uint32_t image_width,
    uint32_t image_height,
    float eps2d,
    float near_plane,
    float far_plane,
    float radius_clip,
    bool mip_splatting_screen_filter,
    gsplat::CameraModelType camera_model,
    UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    FThetaCameraDistortionParameters ftheta_coeffs) {
    return gsplat::projection_ut_3dgs_fused(
        means,
        quats,
        scales,
        opacities,
        viewmats0,
        viewmats1,
        Ks,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        mip_splatting_screen_filter,
        camera_model,
        ut_params,
        rs_type,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs);
}

// Runs the tile-intersection stage and returns sorted primitive instances.
inline auto intersect_fwd(
    const at::Tensor& projected_means,
    const at::Tensor& radii,
    const at::Tensor& primitive_depth,
    const at::optional<at::Tensor> image_ids,
    const at::optional<at::Tensor> primitive_ids,
    uint32_t num_images,
    uint32_t tile_size,
    uint32_t tile_width,
    uint32_t tile_height,
    bool sort,
    bool segmented) {
    return gsplat::intersect_tile(
        projected_means,
        radii,
        primitive_depth,
        image_ids,
        primitive_ids,
        num_images,
        tile_size,
        tile_width,
        tile_height,
        sort,
        segmented);
}

// Builds per-tile instance offsets from sorted intersection ids.
inline auto intersect_offsets_fwd(
    const at::Tensor& intersection_ids,
    uint32_t num_images,
    uint32_t tile_width,
    uint32_t tile_height) {
    return gsplat::intersect_offset(
        intersection_ids, num_images, tile_width, tile_height);
}

// Rasterizes NHT vertex features plus ray-direction channels.
inline auto rasterize_features_fwd(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    uint32_t image_width,
    uint32_t image_height,
    uint32_t tile_size,
    const at::Tensor& viewmats0,
    const at::optional<at::Tensor> viewmats1,
    const at::Tensor& Ks,
    gsplat::CameraModelType camera_model,
    UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    FThetaCameraDistortionParameters ftheta_coeffs,
    const at::Tensor& tile_offsets,
    const at::Tensor& instance_primitive_indices,
    bool center_ray_mode,
    float ray_dir_scale) {
    return gsplat::rasterize_to_pixels_from_world_nht_3dgs_fwd(
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        viewmats0,
        viewmats1,
        Ks,
        camera_model,
        ut_params,
        rs_type,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        tile_offsets,
        instance_primitive_indices,
        center_ray_mode,
        ray_dir_scale);
}

// Runs the NHT feature-rasterization backward stage.
inline auto rasterize_features_bwd(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    uint32_t image_width,
    uint32_t image_height,
    uint32_t tile_size,
    const at::Tensor& viewmats0,
    const at::optional<at::Tensor> viewmats1,
    const at::Tensor& Ks,
    gsplat::CameraModelType camera_model,
    UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    FThetaCameraDistortionParameters ftheta_coeffs,
    const at::Tensor& tile_offsets,
    const at::Tensor& instance_primitive_indices,
    const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    const at::Tensor& grad_rendered_features,
    const at::Tensor& grad_rendered_alphas) {
    return gsplat::rasterize_to_pixels_from_world_nht_3dgs_bwd(
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        viewmats0,
        viewmats1,
        Ks,
        camera_model,
        ut_params,
        rs_type,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        tile_offsets,
        instance_primitive_indices,
        render_alphas,
        last_ids,
        grad_rendered_features,
        grad_rendered_alphas);
}

// Rasterizes scalar eval3d payloads, currently depth.
inline auto rasterize_depth_fwd(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    uint32_t image_width,
    uint32_t image_height,
    uint32_t tile_size,
    const at::Tensor& viewmats0,
    const at::optional<at::Tensor> viewmats1,
    const at::Tensor& Ks,
    gsplat::CameraModelType camera_model,
    UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    FThetaCameraDistortionParameters ftheta_coeffs,
    const at::Tensor& tile_offsets,
    const at::Tensor& instance_primitive_indices) {
    return gsplat::rasterize_to_pixels_from_world_3dgs_fwd(
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        viewmats0,
        viewmats1,
        Ks,
        camera_model,
        ut_params,
        rs_type,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        tile_offsets,
        instance_primitive_indices);
}

// Runs the eval3d payload-rasterization backward stage.
inline auto rasterize_depth_bwd(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    uint32_t image_width,
    uint32_t image_height,
    uint32_t tile_size,
    const at::Tensor& viewmats0,
    const at::optional<at::Tensor> viewmats1,
    const at::Tensor& Ks,
    gsplat::CameraModelType camera_model,
    UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    FThetaCameraDistortionParameters ftheta_coeffs,
    const at::Tensor& tile_offsets,
    const at::Tensor& instance_primitive_indices,
    const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    const at::Tensor& grad_rendered_depth,
    const at::Tensor& grad_rendered_alphas) {
    return gsplat::rasterize_to_pixels_from_world_3dgs_bwd(
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        viewmats0,
        viewmats1,
        Ks,
        camera_model,
        ut_params,
        rs_type,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        tile_offsets,
        instance_primitive_indices,
        render_alphas,
        last_ids,
        grad_rendered_depth,
        grad_rendered_alphas);
}

// Returns primitive and pixel ids for native contributor-attribution paths.
inline auto rasterize_to_indices_fwd(
    uint32_t range_start,
    uint32_t range_end,
    const at::Tensor& transmittances,
    const at::Tensor& projected_means,
    const at::Tensor& conics,
    const at::Tensor& opacities,
    uint32_t image_width,
    uint32_t image_height,
    uint32_t tile_size,
    const at::Tensor& tile_offsets,
    const at::Tensor& instance_primitive_indices) {
    return gsplat::rasterize_to_indices_3dgs(
        range_start,
        range_end,
        transmittances,
        projected_means,
        conics,
        opacities,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        instance_primitive_indices);
}

}  // namespace ember_native_nht::nht_rasterizer
