"""Official FasterGS-family native backends for Ember."""

from ember_native_faster_gs._version import __version__


def register() -> None:
    """Register all native FasterGS-family backends."""
    from ember_native_faster_gs.faster_gs import register as register_core
    from ember_native_faster_gs.faster_gs_depth import (
        register as register_depth,
    )
    from ember_native_faster_gs.fastgs import register as register_fastgs
    from ember_native_faster_gs.gaussian_pop import (
        register as register_gaussian_pop,
    )

    register_core()
    register_depth()
    register_gaussian_pop()
    register_fastgs()


__all__ = ["__version__", "register"]
