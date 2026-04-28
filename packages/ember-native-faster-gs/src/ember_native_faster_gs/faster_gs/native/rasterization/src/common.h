#pragma once

// Declares shared validation and scratch-buffer helpers for the native FasterGS
// CUDA wrapper entrypoints.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace ember_core::faster_gs_native {

// Verifies that `tensor` is a CUDA float32 tensor.
inline void check_cuda_float_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat32,
        name,
        " must be float32."
    );
}

// Verifies that `tensor` is a CUDA int32 tensor.
inline void check_cuda_int_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kInt32,
        name,
        " must be int32."
    );
}

// Verifies that `tensor` is a CUDA uint16 tensor.
inline void check_cuda_uint16_tensor(
    const torch::Tensor& tensor,
    const char* name
) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kUInt16,
        name,
        " must be uint16."
    );
}

// Returns a reusable scratch tensor named by `label` on the current CUDA
// stream. The returned view has exactly `size` elements and grows the cached
// backing storage on demand.
inline torch::Tensor get_cached_workspace(
    const char* label,
    const torch::TensorOptions& options,
    int64_t size
) {
    if (size <= 0) {
        return torch::empty({0}, options);
    }

    static std::mutex workspace_mutex;
    // Keeps one backing tensor per `(label, device, dtype, stream)` tuple so
    // repeated renders can reuse large temporary allocations.
    static std::unordered_map<std::string, torch::Tensor> workspace_cache;

    const auto device = options.device();
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    const std::string key = std::string(label) + ":" +
                            std::to_string(device.index()) + ":" +
                            std::to_string(static_cast<int>(options.dtype().toScalarType())) +
                            ":" + std::to_string(stream.id());

    std::lock_guard<std::mutex> guard(workspace_mutex);
    auto& workspace = workspace_cache[key];
    if (!workspace.defined() || workspace.numel() < size) {
        workspace = torch::empty({size}, options);
    }
    return workspace.narrow(0, 0, size);
}

}  // namespace ember_core::faster_gs_native
