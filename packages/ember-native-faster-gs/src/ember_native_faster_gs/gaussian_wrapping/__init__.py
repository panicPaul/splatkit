"""Gaussian Wrapping backend surface."""

__all__ = [
    "GaussianWrappingNativeRenderOptions",
    "GaussianWrappingNativeRenderOutput",
    "GaussianWrappingScene",
    "GaussianWrappingSurfaceProvider",
    "register",
    "render_gaussian_wrapping_compaction_losses",
    "render_gaussian_wrapping_native",
]


def register() -> None:
    """Register the Gaussian Wrapping backend."""
    from ember_native_faster_gs.gaussian_wrapping.renderer import (
        register as register_backend,
    )

    register_backend()


def __getattr__(name: str) -> object:
    """Load Gaussian Wrapping exports only when requested."""
    if name == "GaussianWrappingScene":
        from ember_native_faster_gs.gaussian_wrapping.scene import (
            GaussianWrappingScene,
        )

        return GaussianWrappingScene
    if name in {
        "GaussianWrappingNativeRenderOptions",
        "GaussianWrappingNativeRenderOutput",
        "GaussianWrappingSurfaceProvider",
        "render_gaussian_wrapping_compaction_losses",
        "render_gaussian_wrapping_native",
    }:
        from ember_native_faster_gs.gaussian_wrapping import renderer

        return getattr(renderer, name)
    raise AttributeError(name)
