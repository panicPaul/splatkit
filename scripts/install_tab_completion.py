#!/usr/bin/env python3

"""Install shell completion for local tyro-based helper scripts."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

import tyro
from pydantic import BaseModel


ShellName = Literal["bash", "zsh"]

SCRIPT_SCAN_EXCLUDES = {
    ".git",
    ".venv",
    ".pixi",
    "__pycache__",
    "build",
}

def _bash_completion_dir() -> Path:
    """Return the user-local bash completion directory."""
    base = Path(
        os.environ.get(
            "BASH_COMPLETION_USER_DIR",
            Path(
                os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
            )
            / "bash-completion",
        )
    )
    return base / "completions"


def _zsh_completion_dir() -> Path:
    """Return the user-local zsh completion directory."""
    return Path.home() / ".zfunc"


class InstallTabCompletion(BaseModel):
    """Arguments for installing shell completion files."""

    shell: ShellName
    script: str | None = None

    def _iter_tyro_scripts(self, project_root: Path) -> list[Path]:
        """Find repository Python scripts that expose tyro CLIs."""
        candidates: list[Path] = []
        for path in project_root.rglob("*.py"):
            if any(part in SCRIPT_SCAN_EXCLUDES for part in path.parts):
                continue
            if not path.is_file() or not os.access(path, os.X_OK):
                continue
            try:
                contents = path.read_text()
            except UnicodeDecodeError:
                continue
            if re.search(r"\btyro\.cli\(", contents) is None:
                continue
            if "__main__" not in contents:
                continue
            candidates.append(path)
        return sorted(candidates)

    def run(self) -> None:
        """Generate and install completion files."""
        project_root = Path(__file__).resolve().parent.parent
        tyro_scripts = self._iter_tyro_scripts(project_root)

        if self.script is None:
            target_scripts = tyro_scripts
        else:
            normalized = self.script.removeprefix("./")
            requested = (project_root / normalized).resolve()
            if requested not in tyro_scripts:
                raise ValueError(
                    f"{normalized!r} is not a detected tyro CLI script in this repository."
                )
            target_scripts = [requested]

        if not target_scripts:
            raise ValueError("No tyro CLI scripts were found in this repository.")

        completion_dir = (
            _bash_completion_dir()
            if self.shell == "bash"
            else _zsh_completion_dir()
        )
        completion_dir.mkdir(parents=True, exist_ok=True)

        for script_path in target_scripts:
            relative = script_path.relative_to(project_root)
            flattened_name = "__".join(relative.with_suffix("").parts)
            output_name = (
                script_path.name if self.shell == "bash" else f"_{flattened_name}"
            )
            output_path = completion_dir / output_name

            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--tyro-write-completion",
                    self.shell,
                    str(output_path),
                ],
                check=True,
            )

            print(f"Wrote {self.shell} completion for {relative} to {output_path}")

        if self.shell == "zsh":
            print("Ensure ~/.zfunc is in fpath and run `autoload -Uz compinit && compinit`.")
        else:
            print(
                "Bash completion expects scripts to be invoked directly as commands."
            )


if __name__ == "__main__":
    tyro.cli(InstallTabCompletion).run()
