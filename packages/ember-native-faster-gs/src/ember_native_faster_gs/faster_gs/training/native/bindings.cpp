// Binds FasterGS training kernels into a Python extension module.

#include <torch/extension.h>

#include "adam.h"
#include "densification_api.h"
#include "mip_splatting_3d_filter.h"
#include "morton.h"

namespace adam_api = faster_gs::adam;
namespace densification_api = faster_gs::densification;
namespace mip_splatting_api = faster_gs::mip_splatting;
namespace morton_api = faster_gs::morton;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adam_step", &adam_api::adam_step_wrapper);
    m.def("relocation_adjustment", &densification_api::relocation_wrapper);
    m.def("add_noise", &densification_api::add_noise_wrapper);
    m.def("update_mip_splatting_3d_filter", &mip_splatting_api::update_mip_splatting_3d_filter_wrapper);
    m.def("morton_codes", &morton_api::morton_codes_wrapper);
}
