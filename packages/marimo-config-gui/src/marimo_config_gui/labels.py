"""Label helpers for generated config GUI text."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def humanize_model_name(model_cls: type[BaseModel]) -> str:
    """Return a display label for a Pydantic model class."""
    title = getattr(model_cls, "model_config", {}).get("title")
    if isinstance(title, str) and title:
        return title
    return humanize_identifier(_strip_model_suffix(model_cls.__name__))


def disambiguate_labels(labels: list[str]) -> list[str]:
    """Make repeated labels unique while preserving the first occurrence."""
    counts: dict[str, int] = {}
    result: list[str] = []
    for label in labels:
        count = counts.get(label, 0) + 1
        counts[label] = count
        result.append(label if count == 1 else f"{label} ({count})")
    return result


def humanize_identifier(name: str) -> str:
    """Split an identifier into display words without a dictionary."""
    parts: list[str] = []
    start = 0
    for index in range(1, len(name)):
        if _is_word_boundary(name, start=start, index=index):
            parts.append(name[start:index])
            start = index
    parts.append(name[start:])
    return " ".join(part for part in parts if part).strip()


def _strip_model_suffix(name: str) -> str:
    for suffix in ("Config", "Model"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _is_word_boundary(name: str, *, start: int, index: int) -> bool:
    previous = name[index - 1]
    current = name[index]
    next_char = _char_at(name, index + 1)
    if not current.isupper() or not next_char.islower():
        return False

    if previous.isdigit():
        return True
    if previous.isupper() and index >= 2 and name[index - 2].isdigit():
        return True

    prefix = name[start:index]
    if previous.isupper() and prefix.isupper() and len(prefix) >= 2:
        return True

    if not previous.islower():
        return False
    return not _starts_alphanumeric_tail(name, index)


def _starts_alphanumeric_tail(name: str, index: int) -> bool:
    tail_end = index + 1
    while tail_end < len(name) and name[tail_end].islower():
        tail_end += 1
    if tail_end >= len(name):
        return False
    if name[tail_end].isdigit():
        return True
    return (
        tail_end + 1 < len(name)
        and name[tail_end].isupper()
        and name[tail_end + 1].isdigit()
    )


def _char_at(value: str, index: int) -> str:
    if index >= len(value):
        return ""
    return value[index]


def _humanize_model_name(*args: Any, **kwargs: Any) -> Any:
    return humanize_model_name(*args, **kwargs)


def _disambiguate_labels(*args: Any, **kwargs: Any) -> Any:
    return disambiguate_labels(*args, **kwargs)
