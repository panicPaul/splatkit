#!/usr/bin/env python3

"""Build SIF runtime images for supported CUDA flavors."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import tyro
from pydantic import BaseModel

IMPORT_SMOKE = (
    "import torch, diff_gaussian_rasterization, FasterGSCudaBackend, "
    "new_svraster_cuda"
)


class BuildSifCommand(BaseModel):
    """Arguments for SIF runtime image builds."""

    cuda: Literal["cu128", "cu130"] = "cu128"
    output: Path | None = None
    image_prefix: str = "ember-runtime"

    def _run(self, *command: str, cwd: Path | None = None) -> None:
        """Run a subprocess and fail loudly on errors."""
        subprocess.run(command, check=True, cwd=cwd)

    def _build_runtime_image(self, project_root: Path, image_tag: str) -> None:
        """Build the intermediate OCI image used for the SIF."""
        self._run(
            "docker",
            "build",
            "--file",
            "docker/runtime.Dockerfile",
            "--build-arg",
            f"CUDA_FLAVOR={self.cuda}",
            "--tag",
            image_tag,
            ".",
            cwd=project_root,
        )

    def _build_sif(self, image_tag: str, output_path: Path) -> None:
        """Convert the OCI image to a SIF artifact."""
        self._run(
            "apptainer",
            "build",
            str(output_path),
            f"docker-daemon://{image_tag}",
        )

    def _smoke_test(self, output_path: Path) -> None:
        """Run a GPU-backed import smoke test against the produced SIF."""
        command = "python -c " + shlex.quote(IMPORT_SMOKE)
        self._run(
            "apptainer",
            "exec",
            "--nv",
            str(output_path),
            "bash",
            "-lc",
            command,
        )

    def run(self) -> None:
        """Build one SIF runtime image."""
        project_root = Path(__file__).resolve().parent.parent
        output_path = self.output
        if output_path is None:
            output_path = project_root / "dist" / f"ember-{self.cuda}.sif"
        if output_path.is_absolute():
            target = output_path
        else:
            target = project_root / output_path
        target.parent.mkdir(parents=True, exist_ok=True)

        if shutil.which("docker") is None:
            raise RuntimeError("docker is required for runtime image builds.")
        if shutil.which("apptainer") is None:
            raise RuntimeError("apptainer is required for SIF builds.")

        image_tag = f"{self.image_prefix}:{self.cuda}"
        self._build_runtime_image(project_root, image_tag)
        self._build_sif(image_tag, target)
        self._smoke_test(target)


if __name__ == "__main__":
    tyro.cli(BuildSifCommand).run()
