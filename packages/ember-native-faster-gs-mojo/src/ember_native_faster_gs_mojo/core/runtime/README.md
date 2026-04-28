# `runtime`

This directory mirrors the staged FasterGS runtime surface for the
`faster_gs_mojo` family.

Stage ownership in v1:

- `preprocess`: delegated to `ember_native_faster_gs.faster_gs`
- `sort`: delegated to `ember_native_faster_gs.faster_gs`
- `blend`: family-local seam that prefers MAX/Mojo custom ops and otherwise
  falls back to the existing FasterGS native blend stage
- `render`: family-local composition of delegated preprocess/sort plus the
  family-local blend stage
