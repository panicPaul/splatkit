#!/usr/bin/env python3

"""Build fresh-clone smoke test images for supported CUDA flavors."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import tyro
from pydantic import BaseModel

IMPORT_SMOKE = (
    "import torch, diff_gaussian_rasterization, FasterGSCudaBackend, "
    "new_svraster_cuda"
)


class ProjectConfig(BaseModel):
    """Relevant project configuration for container smoke tests."""

    cuda_flavors: list[str]
    default_flavor: str

    @classmethod
    def load(cls, project_root: Path) -> ProjectConfig:
        """Load supported CUDA flavors from the repository config."""
        import tomllib

        pyproject = project_root / "pyproject.toml"
        contents = tomllib.loads(pyproject.read_text())
        extras = contents["project"]["optional-dependencies"]
        flavors = sorted(name for name in extras if name.startswith("cu"))
        default_envs = contents["tool"]["pixi"]["environments"]["default"]
        default_flavor = next(env for env in default_envs if env in flavors)
        ordered = [default_flavor] + [
            flavor for flavor in flavors if flavor != default_flavor
        ]
        return cls(cuda_flavors=ordered, default_flavor=default_flavor)


class SmokeFreshCloneCommand(BaseModel):
    """Arguments for smoke test image builds."""

    cuda: Literal["all", "cu128", "cu130"] = "all"
    source: Literal["remote", "worktree"] = "remote"
    repo: str | None = None
    ref: str | None = None
    image_prefix: str = "splatkit-smoke"

    def _run(self, *command: str, cwd: Path | None = None) -> None:
        """Run a subprocess and fail loudly on errors."""
        subprocess.run(command, check=True, cwd=cwd)

    def _capture(self, *command: str, cwd: Path | None = None) -> str:
        """Capture stdout from a subprocess."""
        result = subprocess.run(
            command,
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _resolve_repo(self, project_root: Path) -> str:
        """Resolve the clone source for remote smoke tests."""
        if self.repo is not None:
            return self.repo
        return self._capture(
            "git", "remote", "get-url", "origin", cwd=project_root
        )

    def _resolve_flavors(self, project_root: Path) -> list[str]:
        """Resolve the CUDA flavors that should be tested."""
        config = ProjectConfig.load(project_root)
        if self.cuda == "all":
            return config.cuda_flavors
        if self.cuda not in config.cuda_flavors:
            raise ValueError(
                f"Unsupported CUDA flavor {self.cuda!r}. "
                f"Available: {config.cuda_flavors!r}."
            )
        return [self.cuda]

    def _prepare_remote_clone(
        self, project_root: Path
    ) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        """Clone the repository recursively into a temporary directory."""
        temp_dir = tempfile.TemporaryDirectory(prefix="splatkit-smoke-")
        clone_root = Path(temp_dir.name) / "repo"
        repo = self._resolve_repo(project_root)
        self._run("git", "clone", "--recurse-submodules", repo, str(clone_root))
        if self.ref is not None:
            self._run("git", "checkout", self.ref, cwd=clone_root)
            self._run(
                "git",
                "submodule",
                "update",
                "--init",
                "--recursive",
                cwd=clone_root,
            )
        return temp_dir, clone_root

    def _build_image(self, project_root: Path, flavor: str) -> None:
        """Build one smoke image for the requested CUDA flavor."""
        tag = f"{self.image_prefix}:{flavor}"
        self._run(
            "docker",
            "build",
            "--file",
            "docker/smoke.Dockerfile",
            "--build-arg",
            f"CUDA_FLAVOR={flavor}",
            "--tag",
            tag,
            ".",
            cwd=project_root,
        )

    def _smoke_test_image(self, flavor: str) -> None:
        """Run the smoke test inside a GPU-backed container."""
        tag = f"{self.image_prefix}:{flavor}"
        command = (
            "uv sync --locked --extra "
            f"{flavor}"
            " && . .venv/bin/activate"
            " && python -c "
            f"{shlex.quote(IMPORT_SMOKE)}"
        )
        pixi_command = (
            "pixi install --frozen -e "
            f"{flavor}"
            " && pixi run -e "
            f"{flavor} "
            "bash -lc "
            f"{shlex.quote(command)}"
        )
        self._run(
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            tag,
            "bash",
            "-lc",
            pixi_command,
        )

    def run(self) -> None:
        """Build smoke images for the selected CUDA flavors."""
        project_root = Path(__file__).resolve().parent.parent
        flavors = self._resolve_flavors(project_root)

        if shutil.which("docker") is None:
            raise RuntimeError("docker is required for smoke image builds.")
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError(
                "A visible NVIDIA GPU is required for smoke tests."
            )

        if self.source == "worktree":
            build_root = project_root
            temp_dir = None
        else:
            temp_dir, build_root = self._prepare_remote_clone(project_root)

        try:
            for flavor in flavors:
                print(f"Building smoke image for {flavor} from {build_root}.")
                self._build_image(build_root, flavor)
                print(f"Running GPU smoke test for {flavor}.")
                self._smoke_test_image(flavor)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()


if __name__ == "__main__":
    tyro.cli(SmokeFreshCloneCommand).run()
