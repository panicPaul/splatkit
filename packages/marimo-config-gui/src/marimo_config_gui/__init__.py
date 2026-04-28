"""Public package exports for marimo-config-gui."""

from importlib.metadata import PackageNotFoundError, version

from marimo_config_gui.api import (
    config_commit_button,
    config_committed_value,
    config_error,
    config_form,
    config_gui,
    config_json,
    config_json_output,
    config_require_valid,
    config_value,
    create_committed_config_state,
    create_config_state,
    form_gui,
    json_gui,
    load_script_config,
)
from marimo_config_gui.elements import PydanticGui, PydanticJsonGui
from marimo_config_gui.state import ConfigBindings, ScriptConfigLoader

try:
    from marimo_config_gui._version import __version__
except ImportError:
    try:
        __version__ = version("marimo-config-gui")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = [
    "ConfigBindings",
    "PydanticGui",
    "PydanticJsonGui",
    "ScriptConfigLoader",
    "__version__",
    "config_commit_button",
    "config_committed_value",
    "config_error",
    "config_form",
    "config_gui",
    "config_json",
    "config_json_output",
    "config_require_valid",
    "config_value",
    "create_committed_config_state",
    "create_config_state",
    "form_gui",
    "json_gui",
    "load_script_config",
]
