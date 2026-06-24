"""ROS2/Nav2 discovery and command seam for mobile manipulation.

This module intentionally does not fake navigation.  It only reports whether
the ROS2/Nav2 graph exposes the nodes, topics, services, and actions required
for a real mobile-base plan/execution path.  The task pipeline may keep using
MuJoCo in light/debug mode, but full mode must pass this probe before running.
"""

from __future__ import annotations

import platform
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ROS2CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class ROS2GraphSnapshot:
    ros2_ready: bool
    nodes: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    services: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    command_results: tuple[ROS2CommandResult, ...] = ()
    message: str = ""

    def has_any_action(self, needles: Iterable[str]) -> bool:
        lower = [item.lower() for item in self.actions]
        return any(any(needle.lower() in item for item in lower) for needle in needles)

    def has_any_node(self, needles: Iterable[str]) -> bool:
        lower = [item.lower() for item in self.nodes]
        return any(any(needle.lower() in item for item in lower) for needle in needles)

    def has_topic(self, name: str) -> bool:
        return name in self.topics

    def has_service(self, name: str) -> bool:
        return name in self.services


@dataclass(frozen=True)
class Nav2Status:
    ready: bool
    ros2_ready: bool
    tf_ready: bool
    navigate_action_ready: bool
    lifecycle_ready: bool
    missing: tuple[str, ...] = ()
    snapshot: ROS2GraphSnapshot | None = None
    message: str = ""


@dataclass
class ROS2BridgeConfig:
    distro_setup: str = "/opt/ros/jazzy/setup.bash"
    workspace_setup: str = "~/ros2_ws/install/setup.bash"
    wsl_distro: str = "UbuntuRobot"
    use_wsl_on_windows: bool = True
    command_timeout_s: float = 8.0
    required_nav_actions: tuple[str, ...] = ("/navigate_to_pose", "navigate_to_pose")


class ROS2CLI:
    """Small subprocess-backed ROS2 CLI helper.

    The code runs on Windows or Linux.  On Windows it executes commands through
    the configured WSL distro so Python never depends on rclpy being installed
    in the Windows conda environment.
    """

    def __init__(self, config: ROS2BridgeConfig | None = None):
        self.config = config or ROS2BridgeConfig()

    def ros2(self, args: Sequence[str], timeout_s: float | None = None) -> ROS2CommandResult:
        quoted_args = " ".join(shlex.quote(str(arg)) for arg in args)
        distro_setup = _bash_path(self.config.distro_setup)
        workspace_setup = _bash_path(self.config.workspace_setup)
        setup = (
            f"test -f {distro_setup} "
            f"&& source {distro_setup}; "
            f"test -f {workspace_setup} "
            f"&& source {workspace_setup}; "
        )
        bash_cmd = f"{setup} ros2 {quoted_args}"
        return self.bash(bash_cmd, timeout_s=timeout_s)

    def bash(self, bash_cmd: str, timeout_s: float | None = None) -> ROS2CommandResult:
        timeout = float(timeout_s if timeout_s is not None else self.config.command_timeout_s)
        if platform.system().lower().startswith("win") and self.config.use_wsl_on_windows:
            cmd = ["wsl", "-d", self.config.wsl_distro, "--", "bash", "-lc", bash_cmd]
        else:
            cmd = ["bash", "-lc", bash_cmd]
        try:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            return ROS2CommandResult(
                command=" ".join(cmd),
                returncode=int(proc.returncode),
                stdout=proc.stdout.strip(),
                stderr=proc.stderr.strip(),
            )
        except subprocess.TimeoutExpired as exc:
            return ROS2CommandResult(
                command=" ".join(cmd),
                returncode=124,
                stdout=(exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
                stderr=f"timeout after {timeout:.1f}s",
            )
        except Exception as exc:
            return ROS2CommandResult(
                command=" ".join(cmd),
                returncode=1,
                stdout="",
                stderr=str(exc),
            )

    def snapshot(self) -> ROS2GraphSnapshot:
        checks = [
            self.ros2(["--help"]),
            self.ros2(["node", "list"]),
            self.ros2(["topic", "list"]),
            self.ros2(["service", "list"]),
            self.ros2(["action", "list"]),
        ]
        if not checks[0].ok:
            return ROS2GraphSnapshot(
                ros2_ready=False,
                command_results=tuple(checks),
                message=checks[0].stderr or checks[0].stdout or "ros2 command not found",
            )
        if any(not item.ok for item in checks[1:]):
            failed = [item.stderr or item.stdout for item in checks[1:] if not item.ok]
            return ROS2GraphSnapshot(
                ros2_ready=False,
                command_results=tuple(checks),
                message="; ".join(text for text in failed if text) or "ros2 graph query failed",
            )
        return ROS2GraphSnapshot(
            ros2_ready=True,
            nodes=tuple(_lines(checks[1].stdout)),
            topics=tuple(_lines(checks[2].stdout)),
            services=tuple(_lines(checks[3].stdout)),
            actions=tuple(_lines(checks[4].stdout)),
            command_results=tuple(checks),
            message="ready",
        )


class Nav2Client:
    """Probe seam for Nav2 base planning/execution."""

    def __init__(self, cli: ROS2CLI | None = None, config: ROS2BridgeConfig | None = None):
        self.config = config or ROS2BridgeConfig()
        self.cli = cli or ROS2CLI(self.config)

    def ready_status(self) -> Nav2Status:
        snap = self.cli.snapshot()
        missing: list[str] = []
        if not snap.ros2_ready:
            missing.append("ros2 CLI / graph")
        tf_ready = snap.has_topic("/tf") and snap.has_topic("/tf_static")
        if not tf_ready:
            missing.append("/tf and /tf_static topics")
        navigate_ready = snap.has_any_action(self.config.required_nav_actions)
        if not navigate_ready:
            missing.append("Nav2 navigate_to_pose action")
        lifecycle_ready = snap.has_any_node(("bt_navigator", "controller_server", "planner_server", "amcl", "map_server"))
        if not lifecycle_ready:
            missing.append("Nav2 lifecycle nodes")
        return Nav2Status(
            ready=bool(snap.ros2_ready and tf_ready and navigate_ready and lifecycle_ready),
            ros2_ready=bool(snap.ros2_ready),
            tf_ready=bool(tf_ready),
            navigate_action_ready=bool(navigate_ready),
            lifecycle_ready=bool(lifecycle_ready),
            missing=tuple(missing),
            snapshot=snap,
            message=snap.message,
        )

    def require_ready(self) -> None:
        status = self.ready_status()
        if status.ready:
            return
        raise RuntimeError(
            "ROS2/Nav2 is not ready for full mobile manipulation. "
            f"Missing: {', '.join(status.missing) or 'unknown'}. "
            f"Detail: {status.message}"
        )


def _lines(text: str) -> list[str]:
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def _bash_path(path: str) -> str:
    text = str(path)
    if text.startswith("~/"):
        return "$HOME/" + shlex.quote(text[2:]).strip("'")
    return shlex.quote(text)


__all__ = [
    "Nav2Client",
    "Nav2Status",
    "ROS2BridgeConfig",
    "ROS2CLI",
    "ROS2CommandResult",
    "ROS2GraphSnapshot",
]
