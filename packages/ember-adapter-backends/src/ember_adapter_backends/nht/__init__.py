"""NHT reference adapter package for Ember."""

from ember_adapter_backends.nht.renderer import (
    NHTAdapterRenderOptions,
    NHTAdapterRenderOutput,
    register,
    render_nht_adapter,
)

register()

__all__ = [
    "NHTAdapterRenderOptions",
    "NHTAdapterRenderOutput",
    "register",
    "render_nht_adapter",
]
