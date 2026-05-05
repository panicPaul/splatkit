#include <torch/extension.h>

#include "stages.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess_fwd", &ember_core::fastgs_native::preprocess_fwd_wrapper);
    m.def("sort_fwd", &ember_core::fastgs_native::sort_fwd_wrapper);
    m.def("blend_fwd", &ember_core::fastgs_native::blend_fwd_wrapper);
    m.def(
        "blend_metric_counts_fwd",
        &ember_core::fastgs_native::blend_metric_counts_fwd_wrapper
    );
    m.def("blend_bwd", &ember_core::fastgs_native::blend_bwd_wrapper);
}
