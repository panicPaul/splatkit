# `splatkit-native-3dgrt`

Native 3DGRT-family backends for `splatkit`.

This package owns the reusable `core` module and the `3dgrt.stoch3dgs`
backend.

## Upstream References

- 3DGRT: https://gaussiantracer.github.io/
- 3DGRT repository: https://github.com/nv-tlabs/3dgrut
- Stoch3DGS: https://xupaya.github.io/stoch3DGS/

The implementation folders under `src/splatkit_native_3dgrt/` include short
local READMEs pointing back to the specific upstream method or repository.

## License

Most package code is distributed under the Apache License 2.0. The vendored
OptiX development headers under
`src/splatkit_native_3dgrt/core/native/stoch3dgs/dependencies/optix-dev/` are
covered by NVIDIA DesignWorks/OptiX license terms included with those files.
