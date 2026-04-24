#!/usr/bin/env python3

"""Warn when a checked-out top-level submodule commit is not on its remote."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STRICT_ENV_VAR = "SPLATKIT_SUBMODULE_REMOTE_CHECK"


@dataclass(frozen=True)
class SubmoduleInfo:
    """Metadata required to validate one top-level submodule checkout."""

    name: str
    path: Path


def _run_git(*args: str, cwd: Path | None = None) -> str:
    """Run one git command and return its stdout as text."""
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd or REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _list_submodules() -> list[SubmoduleInfo]:
    """Read top-level submodules from `.gitmodules`."""
    output = _run_git(
        "config",
        "--file",
        ".gitmodules",
        "--get-regexp",
        r"^submodule\..*\.path$",
    )
    submodules: list[SubmoduleInfo] = []
    for line in output.splitlines():
        key, relative_path = line.split(maxsplit=1)
        name = key.removeprefix("submodule.").removesuffix(".path")
        submodules.append(
            SubmoduleInfo(name=name, path=REPO_ROOT / relative_path)
        )
    return submodules


def _remote_contains_commit(submodule: SubmoduleInfo, commit: str) -> bool:
    """Return whether the submodule's `origin` remote advertises the commit."""
    remote_output = _run_git("-C", str(submodule.path), "ls-remote", "origin")
    advertised_commits = {
        line.split(maxsplit=1)[0]
        for line in remote_output.splitlines()
        if line.strip()
    }
    return commit in advertised_commits


def _should_fail() -> bool:
    """Return whether missing remote commits should fail the hook."""
    mode = os.environ.get(STRICT_ENV_VAR, "warn").strip().lower()
    return mode in {"fail", "error", "strict", "1", "true", "yes"}


def main() -> int:
    """Validate that checked-out submodule commits are available on remotes."""
    try:
        submodules = _list_submodules()
    except subprocess.CalledProcessError as error:
        print(error.stderr.strip() or str(error), file=sys.stderr)
        return 1

    missing: list[tuple[SubmoduleInfo, str]] = []
    for submodule in submodules:
        try:
            commit = _run_git("-C", str(submodule.path), "rev-parse", "HEAD")
            if not _remote_contains_commit(submodule, commit):
                missing.append((submodule, commit))
        except subprocess.CalledProcessError as error:
            print(error.stderr.strip() or str(error), file=sys.stderr)
            return 1

    if not missing:
        return 0

    fail = _should_fail()
    stream = sys.stderr if fail else sys.stdout
    mode_text = (
        "Failing because "
        f"`{STRICT_ENV_VAR}` requests strict enforcement."
        if fail
        else "Continuing with a warning."
    )
    print(
        "The following submodule commits are checked out locally but are not "
        f"advertised by their `origin` remotes. {mode_text}",
        file=stream,
    )
    for submodule, commit in missing:
        print(
            f"- {submodule.path.relative_to(REPO_ROOT)}: {commit}",
            file=stream,
        )
        print(
            f"  Push it first: git -C {submodule.path.relative_to(REPO_ROOT)} "
            f"push origin {commit}:refs/heads/<branch>",
            file=stream,
        )
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
