"""SVRaster sparse Adam optimizer."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from ember_native_svraster.training import runtime
from jaxtyping import Float
from torch import Tensor


class SVRasterSparseAdam(torch.optim.Optimizer):
    """Sparse Adam optimizer backed by the native SVRaster training kernels."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.1, 0.99),
        eps: float = 1e-15,
        biased: bool = False,
        sparse: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})
        self.biased = biased
        self.sparse = sparse

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        """Run one native sparse Adam update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            learning_rate = float(group["lr"])
            beta1, beta2 = group["betas"]
            epsilon = float(group["eps"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.dtype != torch.float32:
                    raise TypeError(
                        "SVRasterSparseAdam expects float32 parameters."
                    )
                gradient: Float[Tensor, " *shape"] = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exponential_average"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )
                    state["exponential_average_squared"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )
                state["step"] += 1
                if self.biased:
                    runtime.biased_adam_step(
                        sparse=self.sparse,
                        parameter=parameter,
                        gradient=gradient,
                        exponential_average=state["exponential_average"],
                        exponential_average_squared=state[
                            "exponential_average_squared"
                        ],
                        learning_rate=learning_rate,
                        beta1=float(beta1),
                        beta2=float(beta2),
                        epsilon=epsilon,
                    )
                else:
                    runtime.unbiased_adam_step(
                        sparse=self.sparse,
                        parameter=parameter,
                        gradient=gradient,
                        exponential_average=state["exponential_average"],
                        exponential_average_squared=state[
                            "exponential_average_squared"
                        ],
                        step=float(state["step"]),
                        learning_rate=learning_rate,
                        beta1=float(beta1),
                        beta2=float(beta2),
                        epsilon=epsilon,
                    )
        return loss


__all__ = ["SVRasterSparseAdam"]
