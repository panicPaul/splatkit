// Binds the native FasterGS stage wrappers into a Python extension module used
// by the torch custom-op layer.

#include <torch/extension.h>

#include "stages.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Expose each stage separately so Python can register both raw and
    // composed custom ops on top of the same native kernels.
    m.def("preprocess_fwd", &splatkit::faster_gs_native::preprocess_fwd_wrapper);
    m.def("preprocess_bwd", &splatkit::faster_gs_native::preprocess_bwd_wrapper);
    m.def("sort_fwd", &splatkit::faster_gs_native::sort_fwd_wrapper);
    m.def("blend_fwd", &splatkit::faster_gs_native::blend_fwd_wrapper);
    m.def("blend_bwd", &splatkit::faster_gs_native::blend_bwd_wrapper);
}
