"""Optimizer surfaces for splatting training."""

__all__ = ["FusedAdam"]


def __getattr__(name: str) -> object:
    """Load FasterGS-backed optimizers only when requested."""
    if name == "FusedAdam":
        from ember_splatting_training.optim.fused_adam import FusedAdam

        return FusedAdam
    raise AttributeError(name)
