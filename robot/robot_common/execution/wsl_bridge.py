"""Shared Windows-to-WSL bridge helpers for mature robot backends."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from robot_common.infra.config import PROJECT_ROOT


def bash_quote(text: str) -> str:
    """Quote a string for a single bash argument."""
    return "'" + str(text).replace("'", "'\"'\"'") + "'"


def windows_path_to_wsl(path: Path) -> str:
    """Convert a Windows path to the matching /mnt/<drive> WSL path."""
    text = str(Path(path).resolve())
    if len(text) >= 3 and text[1:3] == ":\\":
        drive = text[0].lower()
        rest = text[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return text.replace("\\", "/")


def wsl_distro() -> str:
    return os.environ.get("ROBOT_STACK_WSL_DISTRO", "UbuntuRobot")


def ros_distro() -> str:
    return os.environ.get("ROS_DISTRO", "jazzy")


def wsl_venv() -> str:
    return os.environ.get("ROBOT_STACK_VENV", "/opt/robot-stack/venv")


def wsl_bridge_enabled() -> bool:
    return (
        os.name == "nt"
        and os.environ.get("ROBOT_STACK_DISABLE_WSL_BRIDGE", "0") != "1"
    )


def project_root_wsl() -> str:
    return windows_path_to_wsl(PROJECT_ROOT)


def robot_stack_env_prefix(include_graspnet: bool = True) -> str:
    """Return a bash prefix that reliably exposes ROS2, MoveIt2, and venv deps.

    The ROS Python path is computed inside WSL, so it does not depend on the
    Windows conda interpreter version.
    """
    ros = ros_distro()
    venv = wsl_venv()
    root = project_root_wsl()

    py_parts = [
        "\\$PWD",
    ]
    if include_graspnet:
        py_parts.extend([
            "\\$PWD/third_party/graspnet-baseline",
            "\\$PWD/third_party/graspnet-baseline/models",
            "\\$PWD/third_party/graspnet-baseline/pointnet2",
            "\\$PWD/third_party/graspnet-baseline/knn",
            "\\$PWD/third_party/graspnetAPI",
        ])
    py_prefix = ":".join(py_parts)

    return (
        f"cd {bash_quote(root)} && "
        f"test -f /opt/ros/{ros}/setup.bash && "
        f"source /opt/ros/{ros}/setup.bash && "
        "if [ -f ros2_ws/install/setup.bash ]; then source ros2_ws/install/setup.bash; fi && "
        f"test -x {bash_quote(venv + '/bin/python')} && "
        f"source {bash_quote(venv + '/bin/activate')} && "
        f"for ROS_PY in /opt/ros/{ros}/lib/python*/site-packages; do "
        "if [ -d \"\\$ROS_PY\" ]; then break; fi; done && "
        "if [ ! -d \"\\${ROS_PY:-}\" ]; then ROS_PY=; fi && "
        f"for ROS_PY_DIST in /opt/ros/{ros}/lib/python*/dist-packages; do "
        "if [ -d \"\\$ROS_PY_DIST\" ]; then break; fi; done && "
        "if [ ! -d \"\\${ROS_PY_DIST:-}\" ]; then ROS_PY_DIST=; fi && "
        "export ROBOT_STACK_DISABLE_WSL_BRIDGE=1 && "
        "export CUDA_HOME=\\${CUDA_HOME:-/usr/local/cuda} && "
        "export MUJOCO_GL=\\${MUJOCO_GL:-egl} && "
        f"export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/ros/{ros}/lib:/opt/ros/{ros}/lib/x86_64-linux-gnu:\\${{LD_LIBRARY_PATH:-}} && "
        f"export PYTHONPATH=\"{py_prefix}:\\${{ROS_PY}}:\\${{ROS_PY_DIST}}:\\${{PYTHONPATH:-}}\""
    )


def run_wsl(command: str, timeout: int = 180) -> subprocess.CompletedProcess:
    """Run a bash command inside the configured WSL distro."""
    return subprocess.run(
        ["wsl", "-d", wsl_distro(), "--", "bash", "-lc", command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def run_wsl_with_retries(
    command: str,
    timeout: int = 180,
    attempts: int = 2,
    delay_s: float = 2.0,
) -> subprocess.CompletedProcess:
    """Run a WSL command, retrying cold-start failures."""
    last = None
    for attempt in range(max(1, attempts)):
        last = run_wsl(command, timeout=timeout)
        if last.returncode == 0:
            return last
        if attempt + 1 < attempts:
            time.sleep(delay_s)
    return last


def process_detail(proc: subprocess.CompletedProcess, limit: int = 700) -> str:
    text = (proc.stderr or proc.stdout or "").strip()
    if len(text) > limit:
        return text[:limit] + "..."
    return text
