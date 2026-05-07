# `ember-native-nht`

Native-family Neural Harmonic Textures backends for `ember-core`.

This package owns the `nht.3dgut` backend surface. The implementation follows
the NHT reference semantics while keeping the backend independent from
`gsplat`.

## Upstream References

- Neural Harmonic Textures: https://github.com/nv-tlabs/neural-harmonic-textures
- NHT gsplat submodule: https://github.com/nerfstudio-project/gsplat

## Build Notes

The deferred shader uses `tinycudann` from upstream `bindings/torch`. On systems
where the default host compiler is GCC 14, build with GCC 13:

```bash
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDAHOSTCXX=/usr/bin/g++-13 uv sync --extra cu130-dev
```
