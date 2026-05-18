#include <torch/extension.h>

#include "pipeline_bindings.h"
#include "triangulation_bindings.h"
#include "utils/cuda_array.h"

namespace radfoam {

struct TorchBuffer : public OpaqueBuffer {
    torch::Tensor tensor;

    explicit TorchBuffer(size_t bytes) {
        // Allocate int64 words so CUDA scratch buffers keep natural alignment.
        size_t num_words = (bytes + sizeof(int64_t) - 1) / sizeof(int64_t);
        tensor = torch::empty({static_cast<int64_t>(num_words)},
                              torch::dtype(torch::kInt64).device(torch::kCUDA));
    }

    void *data() override { return tensor.data_ptr(); }
};

std::unique_ptr<OpaqueBuffer> allocate_buffer(size_t bytes) {
    return std::make_unique<TorchBuffer>(bytes);
}

} // namespace radfoam

namespace radfoam_bindings {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Ember RADFOAM native tracing and topology bindings";

    // Bind the topology stage before tracing so CUDA state is initialized once.
    init_triangulation_bindings(module);

    // Bind the render tracing stage without the upstream viewer/runtime package.
    init_pipeline_bindings(module);
}

} // namespace radfoam_bindings
