"""Robust PyTorch JIT extension loading helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any

from torch.utils.cpp_extension import _get_build_directory, load

_BUILD_LOCKS_GUARD = Lock()
_BUILD_LOCKS: dict[str, Lock] = {}


def _lock_key(name: str, build_directory: str | None) -> str:
    if build_directory is None:
        return name
    return f"{Path(build_directory).resolve()}::{name}"


def _build_lock(name: str, build_directory: str | None) -> Lock:
    lock_key = _lock_key(name, build_directory)
    with _BUILD_LOCKS_GUARD:
        return _BUILD_LOCKS.setdefault(lock_key, Lock())


def _build_directory(name: str, build_directory: str | None) -> Path:
    if build_directory is not None:
        return Path(build_directory)
    return Path(_get_build_directory(name, verbose=False))


def clear_completed_build_lock(
    name: str,
    *,
    build_directory: str | None = None,
) -> None:
    """Remove a stale JIT lock when the compiled extension already exists."""
    resolved_build_directory = _build_directory(name, build_directory)
    lock_path = resolved_build_directory / "lock"
    extension_path = resolved_build_directory / f"{name}.so"
    if not lock_path.exists() or not extension_path.exists():
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def _load_torch_extension(
    *,
    name: str,
    sources: Sequence[str],
    extra_include_paths: Sequence[str],
    extra_cflags: Sequence[str],
    extra_cuda_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    build_directory: str | None,
    with_cuda: bool,
    verbose: bool,
) -> Any:
    load_kwargs: dict[str, Any] = {
        "name": name,
        "sources": list(sources),
        "extra_include_paths": list(extra_include_paths),
        "extra_cflags": list(extra_cflags),
        "extra_cuda_cflags": list(extra_cuda_cflags),
        "extra_ldflags": list(extra_ldflags),
        "with_cuda": with_cuda,
        "verbose": verbose,
    }
    if build_directory is not None:
        load_kwargs["build_directory"] = build_directory
    return load(**load_kwargs)


def load_torch_extension(
    *,
    name: str,
    sources: Sequence[str],
    extra_include_paths: Sequence[str] = (),
    extra_cflags: Sequence[str] = (),
    extra_cuda_cflags: Sequence[str] = (),
    extra_ldflags: Sequence[str] = (),
    build_directory: str | None = None,
    with_cuda: bool = True,
    verbose: bool = False,
) -> Any:
    """Load a PyTorch JIT extension with process-local lock-file protection."""
    with _build_lock(name, build_directory):
        clear_completed_build_lock(name, build_directory=build_directory)
        try:
            return _load_torch_extension(
                name=name,
                sources=sources,
                extra_include_paths=extra_include_paths,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=extra_ldflags,
                build_directory=build_directory,
                with_cuda=with_cuda,
                verbose=verbose,
            )
        except FileNotFoundError as exc:
            if Path(exc.filename or "").name != "lock":
                raise
            clear_completed_build_lock(name, build_directory=build_directory)
            return _load_torch_extension(
                name=name,
                sources=sources,
                extra_include_paths=extra_include_paths,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=extra_ldflags,
                build_directory=build_directory,
                with_cuda=with_cuda,
                verbose=verbose,
            )
