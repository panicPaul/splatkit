# `ember-native-svraster`

Native SVRaster-family backends for `ember-core`.

This package is intentionally separate from `ember-adapter-backends` and the
other native families:

- it carries the restricted SVRaster licensing surface
- it keeps the upstream `new-svraster-cuda` adapter path optional
- it exposes the `svraster.core` backend and the reusable `core` module

The package-local [`LICENSE`](./LICENSE) is copied from the upstream SVRaster
source tree.

## Upstream Reference

- SVRaster: https://svraster.github.io/

The implementation folders under `src/ember_native_svraster/` include short
local READMEs pointing back to the SVRaster project.

This package also carries the upstream SVRaster `LICENSE` and the related
`LICENSE_inria.md` text for the vendored render-kernel subset.

## License

This package is not Apache-2.0. It carries the upstream NVIDIA SVRaster license
and related Inria Gaussian-Splatting license terms. Those terms restrict use to
non-commercial research or evaluation purposes.
