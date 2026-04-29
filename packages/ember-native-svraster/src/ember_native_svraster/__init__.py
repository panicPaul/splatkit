"""Official SVRaster-family native backends for Ember."""

from ember_native_svraster._version import __version__


def register() -> None:
    """Register all native SVRaster-family backends."""
    from ember_native_svraster.svraster import register as register_svraster

    register_svraster()


__all__ = ["__version__", "register"]
