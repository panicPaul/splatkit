#include <torch/extension.h>

#include "stages.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess_fwd", &splatkit::faster_gs_native::preprocess_fwd_wrapper);
    m.def("preprocess_bwd", &splatkit::faster_gs_native::preprocess_bwd_wrapper);
    m.def("sort_fwd", &splatkit::faster_gs_native::sort_fwd_wrapper);
    m.def("blend_fwd", &splatkit::faster_gs_native::blend_fwd_wrapper);
    m.def("blend_bwd", &splatkit::faster_gs_native::blend_bwd_wrapper);
}
