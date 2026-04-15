#!/usr/bin/env python3

"""Clean up stale multiprocessing GPU workers owned by this repository."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import tyro
from pydantic import BaseModel


SYSTEMD_USER_NAMES = {"systemd", "(systemd)", "systemd --user"}


class CleanupCommand(BaseModel):
    """Arguments for cleaning up stale repository-owned GPU workers."""

    repo_root: Path = Path(__file__).resolve().parent.parent
    min_age_seconds: int = 300
    include_non_gpu: bool = False
    include_non_orphans: bool = False
    execute: bool = False
    force: bool = False
    quiet: bool = False

    def run(self) -> None:
        """Find and optionally terminate matching stale worker processes."""
        process_infos = [
            process_info
            for process_info in _iter_process_infos(self.repo_root)
            if _is_stale_worker(
                process_info,
                min_age_seconds=self.min_age_seconds,
                include_non_gpu=self.include_non_gpu,
                include_non_orphans=self.include_non_orphans,
            )
        ]

        if not process_infos:
            if not self.quiet:
                print("No stale repository worker processes found.")
            return

        for process_info in process_infos:
            if not self.quiet:
                print(_format_process_info(process_info))

        if not self.execute:
            if not self.quiet:
                print(
                    "Dry run only. Re-run with --execute to terminate these"
                    " worker processes."
                )
            return

        if not self.force:
            for process_info in process_infos:
                os.kill(process_info.pid, signal.SIGTERM)
            if not self.quiet:
                print(
                    f"Sent SIGTERM to {len(process_infos)} stale worker"
                    f"{'' if len(process_infos) == 1 else 's'}."
                )
            return

        for process_info in process_infos:
            os.kill(process_info.pid, signal.SIGKILL)

        if not self.quiet:
            print(
                f"Sent SIGKILL to {len(process_infos)} stale worker"
                f"{'' if len(process_infos) == 1 else 's'}."
            )


class ProcessInfo(BaseModel):
    """Minimal process state used to identify stale workers."""

    pid: int
    ppid: int
    elapsed_seconds: int
    command: str
    executable: Path
    cwd: Path | None
    parent_command: str | None
    uses_gpu: bool


def _iter_process_infos(repo_root: Path) -> list[ProcessInfo]:
    """Return process snapshots for children owned by the current user."""
    process_infos: list[ProcessInfo] = []
    for proc_path in Path("/proc").iterdir():
        if not proc_path.name.isdigit():
            continue
        process_info = _read_process_info(proc_path, repo_root)
        if process_info is not None:
            process_infos.append(process_info)
    return process_infos


def _read_process_info(proc_path: Path, repo_root: Path) -> ProcessInfo | None:
    """Read one process snapshot if it belongs to this repository."""
    pid = int(proc_path.name)
    try:
        ppid, elapsed_seconds = _read_stat_fields(proc_path / "stat")
        executable = Path(os.readlink(proc_path / "exe"))
        command = _read_cmdline(proc_path / "cmdline")
        cwd = _read_optional_link(proc_path / "cwd")
        parent_command = _read_parent_command(ppid)
        uses_gpu = any(
            target.startswith("nvidia")
            for target in _iter_fd_targets(proc_path / "fd")
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None

    if not _belongs_to_repo(executable, cwd, repo_root):
        return None

    return ProcessInfo(
        pid=pid,
        ppid=ppid,
        elapsed_seconds=elapsed_seconds,
        command=command,
        executable=executable,
        cwd=cwd,
        parent_command=parent_command,
        uses_gpu=uses_gpu,
    )


def _read_stat_fields(stat_path: Path) -> tuple[int, int]:
    """Read selected fields from `/proc/<pid>/stat`."""
    stat_text = stat_path.read_text()
    _, remainder = stat_text.rsplit(")", maxsplit=1)
    fields = remainder.strip().split()
    ppid = int(fields[1])
    elapsed_seconds = _read_elapsed_seconds(fields[19])
    return ppid, elapsed_seconds


def _read_elapsed_seconds(start_ticks: str) -> int:
    """Convert Linux clock ticks to elapsed seconds."""
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    uptime_seconds = float(Path("/proc/uptime").read_text().split()[0])
    return max(0, int(uptime_seconds - int(start_ticks) / clock_ticks))


def _read_cmdline(cmdline_path: Path) -> str:
    """Decode the null-delimited process command line."""
    raw = cmdline_path.read_bytes().replace(b"\x00", b" ").strip()
    return raw.decode() if raw else ""


def _read_optional_link(link_path: Path) -> Path | None:
    """Resolve a procfs symlink, returning None when unavailable."""
    try:
        return Path(os.readlink(link_path))
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _read_parent_command(ppid: int) -> str | None:
    """Return the parent command name if it still exists."""
    if ppid <= 0:
        return None
    comm_path = Path("/proc") / str(ppid) / "comm"
    try:
        return comm_path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _iter_fd_targets(fd_dir: Path) -> list[str]:
    """Collect basename targets from a process fd directory."""
    targets: list[str] = []
    try:
        for fd_path in fd_dir.iterdir():
            try:
                target = os.readlink(fd_path)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            targets.append(Path(target).name)
    except (FileNotFoundError, PermissionError, OSError):
        return []
    return targets


def _belongs_to_repo(
    executable: Path,
    cwd: Path | None,
    repo_root: Path,
) -> bool:
    """Return whether the process executable or cwd belongs to the repo."""
    repo_root = repo_root.resolve()
    return _is_relative_to(executable, repo_root) or (
        cwd is not None and _is_relative_to(cwd, repo_root)
    )


def _is_relative_to(path: Path, base: Path) -> bool:
    """Return whether path is located under base."""
    try:
        path.resolve().relative_to(base)
    except ValueError:
        return False
    return True


def _is_stale_worker(
    process_info: ProcessInfo,
    *,
    min_age_seconds: int,
    include_non_gpu: bool,
    include_non_orphans: bool,
) -> bool:
    """Return whether the process matches the stale-worker policy."""
    if "--multiprocessing-fork" not in process_info.command:
        return False
    if "spawn_main" not in process_info.command:
        return False
    if process_info.elapsed_seconds < min_age_seconds:
        return False
    if not include_non_gpu and not process_info.uses_gpu:
        return False
    if include_non_orphans:
        return True
    parent_name = process_info.parent_command
    return parent_name in SYSTEMD_USER_NAMES or process_info.ppid == 1


def _format_process_info(process_info: ProcessInfo) -> str:
    """Render a human-readable process summary line."""
    age = time.strftime("%H:%M:%S", time.gmtime(process_info.elapsed_seconds))
    gpu_marker = "gpu" if process_info.uses_gpu else "cpu"
    return (
        f"pid={process_info.pid} ppid={process_info.ppid} age={age} "
        f"type={gpu_marker} exe={process_info.executable} "
        f"cwd={process_info.cwd} cmd={process_info.command}"
    )


if __name__ == "__main__":
    tyro.cli(CleanupCommand).run()
