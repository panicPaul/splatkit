# `ember-splatting-training`

splatting training utilities for `ember-core`.

This package contains reusable training add-ons that are specific to Gaussian
scene families, including:

- MCMC densification helpers
- the FasterGS fused Adam optimizer wrapper
- splatting training presets as they become stable enough to share

The accelerated MCMC and optimizer utilities are provided by the
`ember-native-faster-gs` package. They do not depend on the external
`FasterGSCudaBackend` package; that package is only used in this repository as
a development/reference implementation for tests.

## Install

```bash
pip install ember-splatting-training
```

CUDA-specific torch wheels should be selected through the same environment
strategy used for the rest of `ember-core`.

## License

This package is distributed under the Apache License 2.0.
