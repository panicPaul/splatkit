# `runtime/ops`

This layer defines the torch custom-op namespace for `faster_gs_mojo`.

- `preprocess_*` delegates to root FasterGS ops
- `sort_*` delegates to root FasterGS ops
- `blend_*` is family-local and can switch between MAX/Mojo custom ops and the
  existing native FasterGS blend wrappers
- `render_*` composes the family-local stage surface into one autograd-carrying
  render op
