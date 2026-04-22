// Aggregates the stage-specific CUDA wrapper implementations into a single
// translation unit.
//
// The vendored FasterGS `.cuh` files contain concrete kernel definitions, so
// compiling the stage wrappers as separate CUDA translation units would produce
// duplicate symbols at link time.

#include "preprocess.cu"
#include "sort.cu"
#include "blend.cu"
