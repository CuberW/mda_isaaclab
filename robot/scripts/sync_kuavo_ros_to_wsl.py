#!/usr/bin/env python
"""Sync the official Kuavo ROS workspace into WSL native storage.

The Kuavo upstream stack is ROS1/Noetic.  On this machine the host WSL distro is
Ubuntu 24.04 with ROS2 Jazzy, so the official ROS1 nodes should still run inside
the vendor Docker image.  This script only moves the source/assets from the
slow Windows mount into a Linux-native WSL path, which is the right place for
inspection, Docker bind mounts, and future ROS-side work.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def run(cmd: list[str], *, timeout: float = 120.0, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def windows_path_to_wsl(path: str) -> str:
    text = str(path).strip()
    if text.startswith("/"):
        return text
    if len(text) >= 3 and text[1:3] in (":\\", ":/"):
        drive = text[0].lower()
        rest = text[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return text.replace("\\", "/")


def wslpath_to_windows(distro: str, path: str) -> str:
    proc = run(["wsl", "-d", distro, "--", "wslpath", "-w", path], timeout=20)
    return (proc.stdout or "").strip().splitlines()[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distro", default="UbuntuRobot")
    parser.add_argument("--source", default=r"E:\workspace\kuavo-ros-opensource")
    parser.add_argument("--target", default="/root/workspace/kuavo-ros-opensource")
    parser.add_argument("--delete", action="store_true", help="Delete files in target that vanished from source")
    parser.add_argument("--skip-copy", action="store_true", help="Only print/verify paths")
    args = parser.parse_args()

    source_wsl = windows_path_to_wsl(args.source)
    source_win = Path(args.source)
    if not source_win.exists() and not source_wsl.startswith("/mnt/"):
        raise SystemExit(f"Kuavo source does not exist: {args.source}")

    rsync_flags = "-a"
    delete_flag = "--delete" if args.delete else ""
    source = shlex.quote(source_wsl.rstrip("/") + "/")
    target = shlex.quote(args.target.rstrip("/") + "/")
    bash = (
        "set -e; "
        f"test -d {source}; "
        f"mkdir -p {shlex.quote(str(Path(args.target).parent).replace(chr(92), '/'))}; "
    )
    if args.skip_copy:
        bash += f"du -sh {source} 2>/dev/null || true; test -d {target.rstrip('/')} && du -sh {target.rstrip('/')} || true; "
    else:
        bash += f"rsync {rsync_flags} {delete_flag} --info=progress2 {source} {target}; "
    bash += (
        f"du -sh {target.rstrip('/')} 2>/dev/null; "
        f"test -d {target.rstrip('/')}/src/kuavo_assets; "
        f"test -d {target.rstrip('/')}/src/kuavo_msgs; "
        "echo KUAVO_WSL_COPY_OK"
    )
    run(["wsl", "-d", args.distro, "--", "bash", "-lc", bash], timeout=1800)
    unc = wslpath_to_windows(args.distro, args.target)
    print(f"WSL path: {args.target}")
    print(f"Windows UNC path: {unc}")

    docker_probe = run(
        ["wsl", "-d", args.distro, "--", "bash", "-lc", "docker ps >/dev/null 2>&1 && echo DOCKER_OK || echo DOCKER_NOT_READY"],
        timeout=30,
        check=False,
    )
    if "DOCKER_OK" not in (docker_probe.stdout or ""):
        print(
            "Docker is not ready inside this WSL distro. Enable Docker Desktop -> "
            "Settings -> Resources -> WSL integration -> UbuntuRobot, or start "
            "the official Docker services from Windows PowerShell."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
