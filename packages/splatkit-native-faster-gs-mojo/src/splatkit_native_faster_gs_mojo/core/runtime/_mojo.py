"""Lazy MAX/Mojo custom-op loading for the FasterGS Mojo runtime."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


def custom_op_library_path() -> Path:
    """Return the source package path for the FasterGS Mojo custom ops."""
    ops_root = Path(__file__).resolve().parent.parent / "operations"
    if not ops_root.exists():
        raise RuntimeError(
            "Missing FasterGS Mojo operations package at "
            f"{ops_root}."
        )
    return ops_root


@lru_cache(maxsize=1)
def load_custom_op_library() -> Any:
    """Load the MAX custom-op library for the FasterGS Mojo blend ops."""
    try:
        from max.experimental.torch import CustomOpLibrary
    except Exception as exc:
        raise RuntimeError(
            "Failed to import MAX/Mojo custom-op support required for "
            f"`faster_gs_mojo.core` ({exc!r})."
        )

    ops_root = custom_op_library_path()

    try:
        return CustomOpLibrary(ops_root)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load FasterGS Mojo custom-op package from "
            f"{ops_root} ({exc!r})."
        )
