"""Named JSON preset helpers for config GUI state."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeAlias, TypeVar, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, create_model

from marimo_config_gui.widgets import (
    _order_payload_for_model,
    _resolve_materialized_payload,
)

ModelT = TypeVar("ModelT", bound=BaseModel)
PATH_DEFAULTS_FILENAME = ".path_defaults.json"


@dataclass(frozen=True)
class ConfigFile:
    """A JSON config layer file and whether it must exist."""

    path: Path
    required: bool


ConfigFileEntry: TypeAlias = str | Path | tuple[str | Path, bool] | ConfigFile


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
    overlays: Sequence[ConfigFileEntry] = ()
    path_defaults: Sequence[ConfigFileEntry] = ()

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


def _normalize_config_file_entry(
    entry: ConfigFileEntry,
    *,
    required_default: bool,
) -> ConfigFile:
    if isinstance(entry, ConfigFile):
        return entry
    if isinstance(entry, tuple):
        path, required = entry
        return ConfigFile(path=Path(path), required=required)
    return ConfigFile(path=Path(entry), required=required_default)


def _normalize_config_file_entries(
    entries: Sequence[ConfigFileEntry],
    *,
    required_default: bool,
) -> tuple[ConfigFile, ...]:
    return tuple(
        _normalize_config_file_entry(
            entry,
            required_default=required_default,
        )
        for entry in entries
    )


def _resolve_file_path(path: Path, *, base_dir: Path | None = None) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute() or base_dir is None:
        return expanded.resolve()
    return (base_dir / expanded).resolve()


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object.")
    return payload


def _load_json_layer(
    config_file: ConfigFile,
    *,
    base_dir: Path | None = None,
) -> tuple[dict[str, Any] | None, Path]:
    resolved_path = _resolve_file_path(config_file.path, base_dir=base_dir)
    if not resolved_path.exists():
        if config_file.required:
            raise FileNotFoundError(resolved_path)
        return None, resolved_path
    return _load_json_object(resolved_path), resolved_path


def _load_json_layers(
    entries: Sequence[ConfigFileEntry],
    *,
    required_default: bool,
    base_dir: Path | None = None,
) -> list[tuple[dict[str, Any], Path]]:
    layers: list[tuple[dict[str, Any], Path]] = []
    for config_file in _normalize_config_file_entries(
        entries,
        required_default=required_default,
    ):
        payload, resolved_path = _load_json_layer(
            config_file, base_dir=base_dir
        )
        if payload is not None:
            layers.append((payload, resolved_path))
    return layers


def _merge_payload_override(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if value == {} or value is None:
            continue
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_payload_override(current, value)
        else:
            merged[key] = value
    return merged


def _merge_payload_layers(
    base: Mapping[str, Any],
    layers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    merged = dict(base)
    for layer in layers:
        merged = _merge_payload_override(merged, layer)
    return merged


def _path_parts(path: Path) -> tuple[str, ...]:
    return tuple(part for part in path.parts if part not in ("", "."))


@dataclass(frozen=True)
class _PathDefaults:
    """Resolved local defaults for path-valued config fields."""

    paths: tuple[tuple[Path, Path], ...] = ()
    fields: Mapping[str, Path] | None = None


def _load_path_defaults(
    entries: Sequence[ConfigFileEntry],
    *,
    base_dir: Path | None = None,
) -> _PathDefaults:
    mappings: list[tuple[Path, Path]] = []
    fields: dict[str, Path] = {}
    for payload, resolved_path in _load_json_layers(
        entries,
        required_default=False,
        base_dir=base_dir,
    ):
        unknown_keys = set(payload) - {"path_prefixes", "fields"}
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(
                f"{resolved_path}: unknown path-default keys: {unknown}."
            )
        path_payload = payload.get("path_prefixes", {})
        field_payload = payload.get("fields", {})
        if not isinstance(path_payload, Mapping):
            raise ValueError(
                f"{resolved_path}: path_prefixes must be a JSON object."
            )
        if not isinstance(field_payload, Mapping):
            raise ValueError(f"{resolved_path}: fields must be a JSON object.")
        for source, target in path_payload.items():
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValueError(
                    f"{resolved_path}: path_prefixes must map strings to strings."
                )
            mappings.append(
                (Path(source).expanduser(), Path(target).expanduser())
            )
        for field_path, target in field_payload.items():
            if not isinstance(field_path, str) or not isinstance(target, str):
                raise ValueError(
                    f"{resolved_path}: fields must map strings to strings."
                )
            fields[field_path] = Path(target).expanduser()
    mappings.sort(key=lambda item: len(_path_parts(item[0])), reverse=True)
    return _PathDefaults(paths=tuple(mappings), fields=fields)


def _remap_path_default(
    path: Path, mappings: Sequence[tuple[Path, Path]]
) -> Path:
    expanded = path.expanduser()
    parts = _path_parts(expanded)
    for source, target in mappings:
        source_parts = _path_parts(source)
        if len(source_parts) > len(parts):
            continue
        if parts[: len(source_parts)] != source_parts:
            continue
        suffix = parts[len(source_parts) :]
        remapped = target
        for part in suffix:
            remapped /= part
        return remapped
    return expanded


def _resolve_path_value(
    value: Any,
    *,
    base_dir: Path,
    path_defaults: _PathDefaults = _PathDefaults(),
    field_path: str = "",
) -> Any:
    if isinstance(value, Path):
        field_default = (path_defaults.fields or {}).get(field_path)
        expanded = (
            field_default
            if field_default is not None
            else _remap_path_default(value, path_defaults.paths)
        )
        return expanded if expanded.is_absolute() else base_dir / expanded
    if isinstance(value, BaseModel):
        updates = {
            name: _resolve_path_value(
                getattr(value, name),
                base_dir=base_dir,
                path_defaults=path_defaults,
                field_path=name if not field_path else f"{field_path}.{name}",
            )
            for name in type(value).model_fields
        }
        return value.model_copy(update=updates)
    if isinstance(value, tuple):
        return tuple(
            _resolve_path_value(
                item,
                base_dir=base_dir,
                path_defaults=path_defaults,
                field_path=field_path,
            )
            for item in value
        )
    if isinstance(value, list):
        return [
            _resolve_path_value(
                item,
                base_dir=base_dir,
                path_defaults=path_defaults,
                field_path=field_path,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _resolve_path_value(
                item,
                base_dir=base_dir,
                path_defaults=path_defaults,
                field_path=f"{field_path}.{key}" if field_path else str(key),
            )
            for key, item in value.items()
        }
    return value


def _default_path_defaults_for_source(path: Path) -> tuple[ConfigFile, ...]:
    return (
        ConfigFile(
            path=path.parent / PATH_DEFAULTS_FILENAME,
            required=False,
        ),
    )


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
    overlays: Sequence[ConfigFileEntry] = (),
    path_defaults: Sequence[ConfigFileEntry] = (),
) -> ModelT:
    """Load a JSON config, sparse overlays, and local path defaults."""
    resolved_path = Path(path).expanduser().resolve()
    payload = _load_json_object(resolved_path)
    resolved_base_dir = _base_dir_for_json(
        resolved_path,
        None if base_dir is None else Path(base_dir),
    )
    overlay_layers = [
        layer
        for layer, _layer_path in _load_json_layers(
            overlays,
            required_default=True,
            base_dir=resolved_base_dir,
        )
    ]
    payload = _merge_payload_layers(payload, overlay_layers)
    config = model_cls.model_validate(
        _resolve_materialized_payload(model_cls, payload)
    )
    if not resolve_relative_paths:
        return config
    resolved_path_defaults = _default_path_defaults_for_source(
        resolved_path
    ) + _normalize_config_file_entries(
        path_defaults,
        required_default=False,
    )
    path_default_mappings = _load_path_defaults(
        resolved_path_defaults,
        base_dir=resolved_base_dir,
    )
    resolved_config = _resolve_path_value(
        config,
        base_dir=resolved_base_dir,
        path_defaults=path_default_mappings,
    )
    assert isinstance(resolved_config, model_cls)
    return resolved_config


def resolve_config_paths(
    config: ModelT,
    *,
    base_dir: str | Path | None = None,
    path_defaults: Sequence[ConfigFileEntry] = (),
) -> ModelT:
    """Apply path defaults and relative-path resolution to a config model."""
    resolved_base_dir = (
        Path.cwd()
        if base_dir is None
        else Path(base_dir).expanduser().resolve()
    )
    path_default_mappings = _load_path_defaults(
        path_defaults,
        base_dir=resolved_base_dir,
    )
    resolved_config = _resolve_path_value(
        config,
        base_dir=resolved_base_dir,
        path_defaults=path_default_mappings,
    )
    assert isinstance(resolved_config, type(config))
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
        overlays=catalog.overlays,
        path_defaults=catalog.path_defaults,
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
    fields["overlay"] = (tuple[Path, ...], ())
    return create_model(
        f"{catalog.model_cls.__name__}PresetOverride",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def override_model_for_config(model_cls: type[BaseModel]) -> type[BaseModel]:
    """Build a sparse CLI override model for a config model."""
    return _override_model_for(model_cls)


def merge_preset_override(
    catalog: ConfigPresetCatalog[ModelT],
    override: BaseModel,
    *,
    overlays: Sequence[ConfigFileEntry] = (),
    path_defaults: Sequence[ConfigFileEntry] = (),
) -> ModelT:
    """Load the selected preset and merge an internal override model into it."""
    payload = override.model_dump(exclude_none=True)
    overlay_entries = payload.pop("overlay", ())
    preset_field = catalog.preset_field or "preset"
    preset_name = payload.pop(preset_field, catalog.default)
    preset = catalog.presets[str(preset_name)]
    base = load_json_config(
        catalog.model_cls,
        preset.path,
        base_dir=preset.base_dir,
        overlays=(
            tuple(catalog.overlays)
            + tuple(overlays)
            + tuple(
                str(path.expanduser().resolve()) for path in overlay_entries
            )
        ),
        path_defaults=tuple(catalog.path_defaults) + tuple(path_defaults),
    )
    merged = merge_config_override(base, payload)
    return _with_preset_field(
        merged,
        preset_name=str(preset_name),
        preset_field=catalog.preset_field,
    )
