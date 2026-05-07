"""3DGUT-style Neural Harmonic Textures backend."""

from ember_native_nht.threedgut.renderer import (
    NHT3DGUTRenderOptions,
    NHT3DGUTRenderOutput,
    barycentric_weights,
    harmonic_encode,
    render_nht_3dgut,
    tetrahedron_vertices,
)


def register() -> None:
    """Register the 3DGUT-style NHT backend."""
    from ember_core.core.contracts import GaussianScene3D
    from ember_core.core.registry import output_set, register_backend

    register_backend(
        name="nht.3dgut",
        default_options=NHT3DGUTRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=output_set("alpha", "depth"),
    )(render_nht_3dgut)


__all__ = [
    "NHT3DGUTRenderOptions",
    "NHT3DGUTRenderOutput",
    "barycentric_weights",
    "harmonic_encode",
    "register",
    "render_nht_3dgut",
    "tetrahedron_vertices",
]
