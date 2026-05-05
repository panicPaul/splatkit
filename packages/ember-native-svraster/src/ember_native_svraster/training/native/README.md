# `training/native`

Vendored SVRaster CUDA sources used only for training utilities.

This native extension is separate from `core/native` so the registered
`svraster.core` render backend remains focused on rendering. Optimizers,
TV-density helpers, and other training-only kernels are exposed through
`ember_native_svraster.training.runtime`.
