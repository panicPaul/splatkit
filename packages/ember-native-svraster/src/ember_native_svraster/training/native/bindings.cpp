/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#include <torch/extension.h>
#include "src/adam_step.h"
#include "src/tv_compute.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("unbiased_adam_step", &ADAM_STEP::unbiased_adam_step);
    module.def("biased_adam_step", &ADAM_STEP::biased_adam_step);
    module.def("total_variation_bw", &TV_COMPUTE::total_variation_bw);
}
