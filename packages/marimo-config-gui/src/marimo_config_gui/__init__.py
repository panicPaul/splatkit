"""Public package exports for marimo-config-gui."""

from importlib.metadata import PackageNotFoundError, version

from marimo_config_gui.api import create_config_gui
from marimo_config_gui.presets import (
    ConfigFile,
    ConfigPreset,
    ConfigPresetCatalog,
)

try:
    from marimo_config_gui._version import __version__
except ImportError:
    try:
        __version__ = version("marimo-config-gui")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = [
    "ConfigFile",
    "ConfigPreset",
    "ConfigPresetCatalog",
    "__version__",
    "create_config_gui",
]
