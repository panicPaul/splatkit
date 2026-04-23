# `splatkit-native-faster-gs-mojo`

Mojo-backed FasterGS-family native backends for `splatkit`.

This package currently owns the `faster_gs_mojo` family:

- `faster_gs_mojo.core`

The package mirrors the staged FasterGS runtime surface while reserving the
blend stage for MAX/Mojo custom ops. Until those custom ops are available in
the active environment, the runtime falls back to the existing native
FasterGS blend implementation so the backend remains usable and testable.
