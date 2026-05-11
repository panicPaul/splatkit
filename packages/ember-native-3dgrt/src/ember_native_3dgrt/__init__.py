"""Official 3DGRT-family native backends for Ember."""

from ember_native_3dgrt._version import __version__


def register() -> None:
    """Register all native 3DGRT-family backends."""
    from ember_native_3dgrt.stoch3dgs import register as register_stoch3dgs
    from ember_native_3dgrt.stoch_fast_gs import (
        register as register_stoch_fast_gs,
    )

    register_stoch3dgs()
    register_stoch_fast_gs()


__all__ = ["__version__", "register"]
