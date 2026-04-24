# `core`

This package exposes the `faster_gs_mojo.core` backend.

Its public structure intentionally mirrors `splatkit_native_faster_gs.faster_gs`:

- `renderer.py` owns the splatkit-facing backend adapter
- `runtime/` owns staged torch custom ops and typed Python wrappers
- `operations/` is the package-local MAX/Mojo custom-op directory for the
  blend stage

For now, preprocess and sort delegate to the existing native FasterGS backend.
The blend stage is the dedicated Mojo seam.
