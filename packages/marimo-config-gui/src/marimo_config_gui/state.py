"""Config state types shared by the public helpers and widgets."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Generic, TypeVar

import tyro
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class ConfigBindings(Generic[ModelT]):
    """Setter and metadata bindings for config GUI helpers.

    Attributes:
        model_cls: The Pydantic model type being edited.
        set_form_gui_state: Setter for the structured form GUI state.
        set_json_gui_state: Setter for the JSON draft text state.
    """

    model_cls: type[ModelT]
    set_form_gui_state: Callable[[dict[str, Any]], None]
    set_json_gui_state: Callable[[str], None]


@dataclass(frozen=True)
class JsonConfigSource:
    """Script-mode JSON config input."""

    path: Annotated[Path, tyro.conf.Positional]


ScriptConfigLoader = Callable[
    [type[BaseModel], BaseModel | dict[str, Any] | None, Sequence[str] | None],
    BaseModel,
]
