"""PowerFoam backend registration helpers."""

from ember_native_powerfoam.powerfoam import register


def register_all() -> None:
    """Register all PowerFoam native backends."""
    register()


__all__ = ["register_all"]
