#pragma once

#include <torch/extension.h>

namespace faster_gs::morton {

torch::Tensor morton_codes_wrapper(
    const torch::Tensor& positions,
    const torch::Tensor& scene_min,
    const float scene_extent);

}
