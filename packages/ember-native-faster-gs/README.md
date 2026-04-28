# `ember-native-faster-gs`

Native FasterGS-family backends for `ember-core`.

This package owns the FasterGS-derived native family:

- `faster_gs.core`
- `faster_gs.depth`
- `faster_gs.gaussian_pop`
- `faster_gs.training`

## Upstream References

- FasterGS: https://fhahlbohm.github.io/faster-gaussian-splatting/
- GaussianPOP paper: https://arxiv.org/pdf/2602.06830

The implementation folders under `src/ember_native_faster_gs/` include
short local READMEs that point back to the specific upstream method each module
is derived from.

## License

This package is distributed under the Apache License 2.0. The FasterGS-derived
native kernels vendored in this package are derived from Apache-2.0 upstream
code.
