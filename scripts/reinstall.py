#!/usr/bin/env python3

"""Reinstall script for development dependencies."""

import subprocess
from typing import Literal

import tyro
from pydantic import BaseModel

CudaFlavor = Literal["cu128", "cu130"]
BackendName = Literal["sv-raster", "inria", "faster-gs"]


class ReinstallCommand(BaseModel):
    """Arguments for the reinstall script."""

    backend: BackendName
    cu: CudaFlavor | None = None

    def _run(self, *command: str) -> None:
        """Run a subprocess and fail loudly on errors."""
        subprocess.run(command, check=True)

    def run(self) -> None:
        """Run the reinstall command."""
        if self.cu is not None:
            self._run("uv", "sync", "--extra", self.cu)

        match self.backend:
            case "sv-raster":
                self._run(
                    "uv",
                    "pip",
                    "install",
                    "--no-cache",
                    "--no-build-isolation",
                    "--reinstall",
                    "third_party/sv_raster/backends/new_cuda",
                )
            case "inria":
                self._run(
                    "uv",
                    "pip",
                    "install",
                    "--no-cache",
                    "--no-build-isolation",
                    "--reinstall",
                    "third_party/diff-gaussian-rasterization",
                )
            case "faster-gs":
                self._run(
                    "uv",
                    "pip",
                    "install",
                    "--no-cache",
                    "--no-build-isolation",
                    "--reinstall",
                    "third_party/faster-gaussian-splatting/FasterGSCudaBackend",
                )

        if self.cu is None:
            print(f"Reinstalled {self.backend}.")
        else:
            print(f"Reinstalled {self.backend} for {self.cu}.")


if __name__ == "__main__":
    tyro.cli(ReinstallCommand).run()
