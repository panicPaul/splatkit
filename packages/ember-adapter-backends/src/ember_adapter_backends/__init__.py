"""Official backend adapters for Ember."""

from importlib.metadata import PackageNotFoundError, version

try:
    from ember_adapter_backends._version import __version__
except ImportError:
    try:
        __version__ = version("ember-adapter-backends")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = ["__version__"]
