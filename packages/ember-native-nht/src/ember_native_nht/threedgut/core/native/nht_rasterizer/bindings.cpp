/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <torch/extension.h>

#include "Cameras.h"
#include "stages.h"
#include "upstream/csrc/Config.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::enum_<gsplat::CameraModelType>(
        module, "CameraModelType", py::module_local())
        .value("PINHOLE", gsplat::CameraModelType::PINHOLE)
        .value("ORTHO", gsplat::CameraModelType::ORTHO)
        .value("FISHEYE", gsplat::CameraModelType::FISHEYE)
        .value("FTHETA", gsplat::CameraModelType::FTHETA)
        .export_values();

    py::enum_<ShutterType>(module, "ShutterType", py::module_local())
        .value("ROLLING_TOP_TO_BOTTOM", ShutterType::ROLLING_TOP_TO_BOTTOM)
        .value("ROLLING_LEFT_TO_RIGHT", ShutterType::ROLLING_LEFT_TO_RIGHT)
        .value("ROLLING_BOTTOM_TO_TOP", ShutterType::ROLLING_BOTTOM_TO_TOP)
        .value("ROLLING_RIGHT_TO_LEFT", ShutterType::ROLLING_RIGHT_TO_LEFT)
        .value("GLOBAL", ShutterType::GLOBAL)
        .export_values();

    py::class_<UnscentedTransformParameters>(
        module, "UnscentedTransformParameters", py::module_local())
        .def(py::init<>())
        .def_readwrite("alpha", &UnscentedTransformParameters::alpha)
        .def_readwrite("beta", &UnscentedTransformParameters::beta)
        .def_readwrite("kappa", &UnscentedTransformParameters::kappa)
        .def_readwrite(
            "in_image_margin_factor",
            &UnscentedTransformParameters::in_image_margin_factor)
        .def_readwrite(
            "require_all_sigma_points_valid",
            &UnscentedTransformParameters::require_all_sigma_points_valid);

    py::enum_<FThetaCameraDistortionParameters::PolynomialType>(
        module, "FThetaPolynomialType", py::module_local())
        .value(
            "PIXELDIST_TO_ANGLE",
            FThetaCameraDistortionParameters::PolynomialType::PIXELDIST_TO_ANGLE)
        .value(
            "ANGLE_TO_PIXELDIST",
            FThetaCameraDistortionParameters::PolynomialType::ANGLE_TO_PIXELDIST)
        .export_values();
    py::class_<FThetaCameraDistortionParameters>(
        module, "FThetaCameraDistortionParameters", py::module_local())
        .def(py::init<>())
        .def_readwrite(
            "reference_poly", &FThetaCameraDistortionParameters::reference_poly)
        .def_readwrite(
            "pixeldist_to_angle_poly",
            &FThetaCameraDistortionParameters::pixeldist_to_angle_poly)
        .def_readwrite(
            "angle_to_pixeldist_poly",
            &FThetaCameraDistortionParameters::angle_to_pixeldist_poly)
        .def_readwrite("max_angle", &FThetaCameraDistortionParameters::max_angle)
        .def_readwrite("linear_cde", &FThetaCameraDistortionParameters::linear_cde);

    // Expose only the stage surface used by the Python runtime.
    module.def("project_fwd", &ember_native_nht::nht_rasterizer::project_fwd);
    module.def("intersect_fwd", &ember_native_nht::nht_rasterizer::intersect_fwd);
    module.def(
        "intersect_offsets_fwd",
        &ember_native_nht::nht_rasterizer::intersect_offsets_fwd);
    module.def(
        "rasterize_features_fwd",
        &ember_native_nht::nht_rasterizer::rasterize_features_fwd);
    module.def(
        "rasterize_features_bwd",
        &ember_native_nht::nht_rasterizer::rasterize_features_bwd);
    module.def(
        "rasterize_depth_fwd",
        &ember_native_nht::nht_rasterizer::rasterize_depth_fwd);
    module.def(
        "rasterize_depth_bwd",
        &ember_native_nht::nht_rasterizer::rasterize_depth_bwd);
    module.def(
        "rasterize_to_indices_fwd",
        &ember_native_nht::nht_rasterizer::rasterize_to_indices_fwd);

    module.attr("encoding_expansion_factor") = ENCF;
    // Compatibility name: this is encoded channels per base feature
    // (sin/cos), not the number of frequency bands.
    module.attr("num_encoding_frequencies") = ENCF;
    module.attr("feature_divisor") = VERTEX_PER_PRIM;
}
