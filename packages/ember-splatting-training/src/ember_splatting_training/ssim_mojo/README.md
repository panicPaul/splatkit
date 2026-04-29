# SSIM Mojo

This package contains the Mojo/MAX custom-op implementation of fused SSIM for
splatting training. The Python layer registers separate PyTorch custom ops for
forward, backward, and mean-reduced backward execution, then wires autograd
through `torch.library.register_autograd`.

The Mojo kernels operate on NCHW float32 CUDA tensors and mirror the fixed
11-tap separable Gaussian window used by
[`rahul-goel/fused-ssim`](https://github.com/rahul-goel/fused-ssim). This
subpackage is a Mojo/MAX port of that fused CUDA SSIM implementation for use in
the splatting training loss path.
