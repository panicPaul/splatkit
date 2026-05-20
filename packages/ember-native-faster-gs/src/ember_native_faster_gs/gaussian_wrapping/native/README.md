Gaussian Wrapping Native Staging
================================

This directory stages the CUDA rasterizers used by Gaussian Wrapping into the
Ember-owned `ember_native_faster_gs.gaussian_wrapping` runtime. The staged
sources come from the pinned `third_party/GaussianWrapping` reference:

- `ours/upstream`: `submodules/diff-gaussian-rasterization_ours`
- `radegs/upstream`: `submodules/diff-gaussian-rasterization`

Do not import the upstream Python packages at runtime. The package-local
`bindings.cpp` files expose FasterGS-style stage wrappers around the staged
C++/CUDA functions, and Python runtime modules register thin
`torch.library.custom_op` dispatchers.
