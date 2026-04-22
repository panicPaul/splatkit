# `gaussian_pop_native.reuse`

This backend owns its blend-stage composition, but it does not re-export any
borrowed raw ops.

Downstream backends should still import shared root-owned stages from the root
owner:

- preprocess and sort from `faster_gs_native.reuse`
- expected-depth blend from `faster_gs_depth_native.reuse`

`gaussian_pop_native` is not a hub backend for re-exporting those borrowed
stages.
