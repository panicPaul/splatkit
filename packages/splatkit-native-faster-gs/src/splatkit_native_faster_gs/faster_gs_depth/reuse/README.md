# `faster_gs_depth.reuse`

This directory is the reuse surface for the depth backend.

## Ownership rule

Only raw ops that are actually implemented by `faster_gs_depth` belong here.
Borrowed raw ops from `faster_gs` are intentionally not re-exported.

That means downstream backends should:

- import shared preprocess/sort stages from `faster_gs.reuse`
- import depth-specific blend stages from `faster_gs_depth.reuse`

This keeps stage ownership explicit and avoids creating accidental "hub" backends that
re-export borrowed native stages.
