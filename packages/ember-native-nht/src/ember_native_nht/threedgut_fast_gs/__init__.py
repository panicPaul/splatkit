"""NHT 3DGUT backend with FastGS densification traits."""

from ember_native_nht.threedgut_fast_gs.renderer import (
    NHTFastGSMetricAttribution,
    NHTFastGSRenderOptions,
    NHTFastGSRenderOutput,
    NHTFastGSSignalProvider,
    nht_fast_gs_metric_counts,
    register,
    render_nht_fast_gs,
)

__all__ = [
    "NHTFastGSMetricAttribution",
    "NHTFastGSRenderOptions",
    "NHTFastGSRenderOutput",
    "NHTFastGSSignalProvider",
    "nht_fast_gs_metric_counts",
    "register",
    "render_nht_fast_gs",
]
