"""Mojo-backed FasterGS-family native backends for Ember."""

from ember_native_faster_gs_mojo._version import __version__


def register() -> None:
    """Register all Mojo-backed FasterGS-family native backends."""
    from ember_native_faster_gs_mojo.core import register as register_core

    register_core()


__all__ = ["__version__", "register"]
