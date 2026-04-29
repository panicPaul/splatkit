"""Profile fused-ssim and SSIM Mojo with CUDA profiler ranges."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import torch
from torch import Tensor

sys.path.insert(0, "packages/ember-splatting-training/src")

from fused_ssim import fused_ssim


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the SSIM profiler driver."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["fused", "mojo"], required=True)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=["forward", "forward-backward"],
        default="forward-backward",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def ssim_function(backend: str) -> Callable[[Tensor, Tensor], Tensor]:
    """Return the selected mean SSIM implementation."""
    if backend == "fused":
        return lambda prediction, target: fused_ssim(
            prediction, target, padding="same", train=True
        )
    if backend == "mojo":
        from ember_splatting_training.ssim_mojo import ssim_mojo

        return lambda prediction, target: ssim_mojo(
            prediction, target, padding="same"
        )
    raise ValueError(f"Unsupported SSIM backend: {backend!r}.")


def run_iteration(
    function: Callable[[Tensor, Tensor], Tensor],
    prediction: Tensor,
    target: Tensor,
    *,
    mode: str,
) -> Tensor:
    """Run one profiled SSIM iteration and return the scalar score."""
    prediction.grad = None
    score = function(prediction, target)
    if mode == "forward-backward":
        score.backward()
    return score


def main() -> None:
    """Run warmup outside profiler capture, then profile steady-state work."""
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    prediction = torch.rand(
        (args.batch, args.channels, args.height, args.width),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.rand_like(prediction)
    function = ssim_function(args.backend)

    for _ in range(args.warmup):
        run_iteration(function, prediction, target, mode=args.mode)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for iteration in range(args.iterations):
        torch.cuda.nvtx.range_push(f"{args.backend}_ssim_{args.mode}_iteration")
        score = run_iteration(function, prediction, target, mode=args.mode)
        torch.cuda.nvtx.range_pop()
        if iteration == args.iterations - 1:
            print(f"score={float(score.detach().cpu()):.9f}")
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
