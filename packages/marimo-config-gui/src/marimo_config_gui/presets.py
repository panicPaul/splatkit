"""Named JSON preset helpers for config GUI state."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, create_model

from marimo_config_gui.widgets import (
    _order_payload_for_model,
    _resolve_materialized_payload,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class ConfigPreset:
    """A named JSON config preset."""

    name: str
    path: Path
    label: str | None = None
    base_dir: Path | None = None


@dataclass(frozen=True)
class ConfigPresetCatalog(Generic[ModelT]):
    """Collection of named JSON presets for one config model."""

    model_cls: type[ModelT]
    presets: Mapping[str, ConfigPreset]
    default: str
    preset_field: str | None = "preset"

    def __post_init__(self) -> None:
        if self.default not in self.presets:
            raise ValueError(
                f"Default preset {self.default!r} is not in the preset catalog."
            )
        for name, preset in self.presets.items():
            if name != preset.name:
                raise ValueError(
                    "Preset mapping keys must match ConfigPreset.name: "
                    f"{name!r} != {preset.name!r}."
                )
        if (
            self.preset_field is not None
            and self.preset_field not in self.model_cls.model_fields
        ):
            raise ValueError(
                f"Preset field {self.preset_field!r} is not a field on "
                f"{self.model_cls.__name__}."
            )


def _base_dir_for_json(path: Path, base_dir: Path | None) -> Path:
    return (
        base_dir.expanduser().resolve() if base_dir is not None else path.parent
    )


def _resolve_path_value(value: Any, *, base_dir: Path) -> Any:
    if isinstance(value, Path):
        expanded = value.expanduser()
        return expanded if expanded.is_absolute() else base_dir / expanded
    if isinstance(value, BaseModel):
        updates = {
            name: _resolve_path_value(
                getattr(value, name),
                base_dir=base_dir,
            )
            for name in type(value).model_fields
        }
        return value.model_copy(update=updates)
    if isinstance(value, tuple):
        return tuple(
            _resolve_path_value(item, base_dir=base_dir) for item in value
        )
    if isinstance(value, list):
        return [_resolve_path_value(item, base_dir=base_dir) for item in value]
    if isinstance(value, dict):
        return {
            key: _resolve_path_value(item, base_dir=base_dir)
            for key, item in value.items()
        }
    return value


def _with_preset_field(
    config: ModelT,
    *,
    preset_name: str | None,
    preset_field: str | None,
) -> ModelT:
    if preset_name is None or preset_field is None:
        return config
    return config.model_copy(update={preset_field: preset_name})


def load_json_config(
    model_cls: type[ModelT],
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
    resolve_relative_paths: bool = True,
) -> ModelT:
    """Load a JSON config and optionally resolve relative Path values."""
    resolved_path = Path(path).expanduser().resolve()
    payload = json.loads(resolved_path.read_text())
    config = model_cls.model_validate(
        _resolve_materialized_payload(model_cls, payload)
    )
    if not resolve_relative_paths:
        return config
    resolved_base_dir = _base_dir_for_json(
        resolved_path,
        None if base_dir is None else Path(base_dir),
    )
    resolved_config = _resolve_path_value(config, base_dir=resolved_base_dir)
    assert isinstance(resolved_config, model_cls)
    return resolved_config


def load_preset_config(
    catalog: ConfigPresetCatalog[ModelT],
    name: str | None = None,
) -> ModelT:
    """Load a named preset from a preset catalog."""
    preset_name = catalog.default if name is None else name
    try:
        preset = catalog.presets[preset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown config preset: {preset_name!r}.") from exc
    config = load_json_config(
        catalog.model_cls,
        preset.path,
        base_dir=preset.base_dir,
    )
    return _with_preset_field(
        config,
        preset_name=preset_name,
        preset_field=catalog.preset_field,
    )


def merge_config_override(
    base: ModelT, override_payload: Mapping[str, Any]
) -> ModelT:
    """Merge a sparse nested override payload into a Pydantic config model."""
    merged: BaseModel = base
    for field_name, value in override_payload.items():
        if value == {} or value is None:
            continue
        current = getattr(merged, field_name)
        next_value = (
            merge_config_override(current, value)
            if isinstance(current, BaseModel) and isinstance(value, Mapping)
            else value
        )
        merged = merged.model_copy(update={field_name: next_value})
    assert isinstance(merged, type(base))
    return merged


def payload_for_config(config: BaseModel | dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-style payload ordered according to a config model."""
    if isinstance(config, BaseModel):
        return _order_payload_for_model(
            type(config),
            config.model_dump(mode="json"),
        )
    return dict(config)


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    args = get_args(annotation)
    if type(None) not in args:
        return annotation
    non_none = tuple(arg for arg in args if arg is not type(None))
    if len(non_none) == 1:
        return non_none[0]
    return annotation


def _optional_annotation(annotation: Any) -> Any:
    return (
        annotation if type(None) in get_args(annotation) else annotation | None
    )


def _is_model_annotation(annotation: Any) -> bool:
    unwrapped = _unwrap_optional(annotation)
    return isinstance(unwrapped, type) and issubclass(unwrapped, BaseModel)


def _is_mapping_annotation(annotation: Any) -> bool:
    unwrapped = _unwrap_optional(annotation)
    origin = get_origin(unwrapped)
    return unwrapped is dict or origin in (dict, Mapping)


def _override_model_for(
    model_cls: type[BaseModel],
    *,
    name: str | None = None,
) -> type[BaseModel]:
    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field in model_cls.model_fields.items():
        annotation = field.annotation
        if _is_model_annotation(annotation):
            nested_model = _override_model_for(
                _unwrap_optional(annotation),
                name=f"{model_cls.__name__}{field_name.title()}Override",
            )
            fields[field_name] = (
                nested_model,
                Field(default_factory=nested_model),
            )
        elif _is_mapping_annotation(annotation):
            fields[field_name] = (Any, None)
        else:
            fields[field_name] = (_optional_annotation(annotation), None)
    return create_model(
        name or f"{model_cls.__name__}Override",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def override_model_for_catalog(
    catalog: ConfigPresetCatalog[ModelT],
) -> type[BaseModel]:
    """Build the internal sparse CLI override model for a preset catalog."""
    base_override_model = _override_model_for(catalog.model_cls)
    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field in base_override_model.model_fields.items():
        if field_name == catalog.preset_field:
            fields[field_name] = (str, catalog.default)
        else:
            default = (
                Field(default_factory=field.default_factory)
                if field.default_factory is not None
                else field.default
            )
            fields[field_name] = (field.annotation, default)
    if catalog.preset_field is None:
        fields["preset"] = (str, catalog.default)
    return create_model(
        f"{catalog.model_cls.__name__}PresetOverride",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def merge_preset_override(
    catalog: ConfigPresetCatalog[ModelT],
    override: BaseModel,
) -> ModelT:
    """Load the selected preset and merge an internal override model into it."""
    payload = override.model_dump(exclude_none=True)
    preset_field = catalog.preset_field or "preset"
    preset_name = payload.pop(preset_field, catalog.default)
    base = load_preset_config(catalog, str(preset_name))
    merged = merge_config_override(base, payload)
    return _with_preset_field(
        merged,
        preset_name=str(preset_name),
        preset_field=catalog.preset_field,
    )
