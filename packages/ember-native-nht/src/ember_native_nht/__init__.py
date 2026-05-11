"""Neural Harmonic Textures native-family backends for Ember."""

try:
    from ember_native_nht._version import __version__
except ModuleNotFoundError:  # pragma: no cover - generated in built wheels
    __version__ = "0.0.1a0"


def register() -> None:
    """Register all NHT-family native backends."""
    from ember_native_nht.threedgut import register as register_3dgut
    from ember_native_nht.threedgut_fast_gs import (
        register as register_3dgut_fast_gs,
    )

    register_3dgut()
    register_3dgut_fast_gs()


__all__ = ["__version__", "register"]
