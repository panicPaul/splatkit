"""Official backend adapters for splatkit."""

from importlib.metadata import PackageNotFoundError, version

try:
    from splatkit_adapter_backends._version import __version__
except ImportError:
    try:
        __version__ = version("splatkit-adapter-backends")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = ["__version__"]
