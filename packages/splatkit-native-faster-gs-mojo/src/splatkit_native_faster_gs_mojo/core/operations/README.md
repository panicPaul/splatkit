# `operations`

This directory is reserved for the MAX/Mojo custom ops that will eventually own
the `faster_gs_mojo` blend stage.

The runtime loads this directory only when
`SPLATKIT_ENABLE_FASTER_GS_MOJO_CUSTOM_OPS=1` is set. That keeps the package
importable in environments that do not yet have Modular installed while still
providing a stable place for the upcoming Mojo kernels.
