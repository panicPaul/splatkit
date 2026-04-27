# `splatkit-gaussian-training`

Gaussian-specific training utilities for `splatkit`.

This package contains reusable training add-ons that are specific to Gaussian
scene families, including:

- MCMC densification helpers
- the FasterGS fused Adam optimizer wrapper
- Gaussian training presets as they become stable enough to share

The accelerated MCMC and optimizer utilities are provided by the
`splatkit-native-faster-gs` package. They do not depend on the external
`FasterGSCudaBackend` package; that package is only used in this repository as
a development/reference implementation for tests.

## Install

```bash
pip install splatkit-gaussian-training
```

CUDA-specific torch wheels should be selected through the same environment
strategy used for the rest of `splatkit`.

## License

This package is distributed under the Apache License 2.0.
