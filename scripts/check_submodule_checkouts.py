#!/usr/bin/env python3

"""Validate that recursive submodules are initialized at recorded commits."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SubmoduleStatus:
    """Parsed status for one recursive submodule checkout."""

    state: str
    commit: str
    path: str
    description: str


def _run_git(*args: str) -> str:
    """Run one git command in the repository and return stdout."""
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _parse_status_line(line: str) -> SubmoduleStatus:
    """Parse one `git submodule status` line."""
    state = line[0]
    rest = line[1:].strip()
    commit, path, *description = rest.split(maxsplit=2)
    return SubmoduleStatus(
        state=state,
        commit=commit,
        path=path,
        description=description[0] if description else "",
    )


def _submodule_statuses() -> list[SubmoduleStatus]:
    """Return recursive submodule statuses recorded by git."""
    output = _run_git("submodule", "status", "--recursive")
    return [
        _parse_status_line(line) for line in output.splitlines() if line.strip()
    ]


def _status_message(status: SubmoduleStatus) -> str:
    if status.state == "-":
        return "not initialized"
    if status.state == "+":
        return "checked out at a different commit than the superproject records"
    if status.state == "U":
        return "has merge conflicts"
    return f"has unexpected status prefix {status.state!r}"


def main() -> int:
    """Fail when any submodule is missing, conflicted, or at the wrong SHA."""
    try:
        invalid = [
            status for status in _submodule_statuses() if status.state != " "
        ]
    except subprocess.CalledProcessError as error:
        print(error.stderr.strip() or str(error), file=sys.stderr)
        return 1
    except (IndexError, ValueError) as error:
        print(
            f"Failed to parse git submodule status output: {error}",
            file=sys.stderr,
        )
        return 1

    if not invalid:
        return 0

    print(
        "Submodules must be initialized recursively and checked out at the "
        "commits recorded by the superproject.",
        file=sys.stderr,
    )
    for status in invalid:
        print(
            f"- {status.path}: {_status_message(status)} ({status.commit})",
            file=sys.stderr,
        )
    print(
        "\nTry: git submodule update --init --recursive",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
