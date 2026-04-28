// Binds FasterGS training kernels into a Python extension module.

#include <torch/extension.h>

#include "adam.h"
#include "densification_api.h"

namespace adam_api = faster_gs::adam;
namespace densification_api = faster_gs::densification;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adam_step", &adam_api::adam_step_wrapper);
    m.def("relocation_adjustment", &densification_api::relocation_wrapper);
    m.def("add_noise", &densification_api::add_noise_wrapper);
}
