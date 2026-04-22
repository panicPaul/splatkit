# `splatkit-native-svraster`

Native SVRaster-family backends for `splatkit`.

This package is intentionally separate from `splatkit-adapter-backends` and the
other native families:

- it carries the restricted SVRaster licensing surface
- it keeps the `new-svraster-cuda` dependency opt-in
- it exposes the `svraster.core` backend and the reusable `core` module

The package-local [`LICENSE`](./LICENSE) is copied from the upstream SVRaster
source tree.

## Upstream Reference

- SVRaster: https://svraster.github.io/

The implementation folders under `src/splatkit_native_svraster/` include short
local READMEs pointing back to the SVRaster project.
