"""Helpers for robust PyTorch JIT extension loading."""

from __future__ import annotations

from pathlib import Path

from torch.utils.cpp_extension import _get_build_directory


def clear_completed_build_lock(name: str) -> None:
    """Remove a stale JIT lock when the compiled extension already exists."""
    build_dir = Path(_get_build_directory(name, verbose=False))
    lock_path = build_dir / "lock"
    extension_path = build_dir / f"{name}.so"
    if not lock_path.exists() or not extension_path.exists():
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return
