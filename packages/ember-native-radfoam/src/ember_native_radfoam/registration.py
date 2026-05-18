"""RADFOAM backend registration helpers."""

from ember_native_radfoam.radfoam import register


def register_all() -> None:
    """Register all RADFOAM native backends."""
    register()


__all__ = ["register_all"]
