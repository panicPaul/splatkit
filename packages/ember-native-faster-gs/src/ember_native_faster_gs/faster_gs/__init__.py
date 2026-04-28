"""Ember-native FasterGS backend."""

__all__ = [
    "FasterGSNativeRenderOptions",
    "FasterGSNativeRenderOutput",
    "register",
    "render_faster_gs_native",
]


def __getattr__(name: str) -> object:
    """Load renderer exports only when requested."""
    if name in __all__:
        from ember_native_faster_gs.faster_gs import renderer

        return getattr(renderer, name)
    raise AttributeError(name)
