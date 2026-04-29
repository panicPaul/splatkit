"""Ember-native FastGS experiment backend."""

__all__ = [
    "FastGSNativeDensificationRenderOutput",
    "FastGSNativeGaussianMetricAttribution",
    "FastGSNativeRenderOptions",
    "FastGSNativeRenderOutput",
    "register",
    "render_fastgs_native",
]


def __getattr__(name: str) -> object:
    """Load renderer exports only when requested."""
    if name in __all__:
        from ember_native_faster_gs.fastgs import renderer

        return getattr(renderer, name)
    raise AttributeError(name)
