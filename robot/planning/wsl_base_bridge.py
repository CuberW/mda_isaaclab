"""Bidirectional MuJoCo <-> ROS2 base bridge via shared files.

- OdometryPublisher: writes MuJoCo base pose to .bridge/odom.json (Windows -> WSL)
- CmdVelReader: reads ROS2 /cmd_vel from .bridge/cmd_vel.json and applies to MuJoCo (WSL -> Windows)

The bridge directory is shared via /mnt/e/ on the WSL side.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class OdometryPublisher:
    """Publish MuJoCo base odometry to a shared file for WSL ROS2 consumption."""

    def __init__(self, bridge_dir: str = ".bridge", controller=None):
        self._root = _resolve_bridge_dir(bridge_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._controller = controller
        self._last_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_time = time.perf_counter()
        self._odom_file = "odom.json"

    def step(self) -> None:
        """Write current base pose and velocity to the bridge file.

        Call this every MuJoCo step during navigation.
        """
        if self._controller is None:
            return
        now = time.perf_counter()
        dt = max(now - self._last_time, 1e-6)
        self._last_time = now

        pose = self._controller.get_base_pose()
        dx = pose[0] - self._last_pose[0]
        dy = pose[1] - self._last_pose[1]
        dyaw = math.atan2(
            math.sin(pose[2] - self._last_pose[2]),
            math.cos(pose[2] - self._last_pose[2]),
        )
        vx = dx / dt
        vy = dy / dt
        vw = dyaw / dt
        self._last_pose = (float(pose[0]), float(pose[1]), float(pose[2]))

        data = {
            "x": float(pose[0]),
            "y": float(pose[1]),
            "yaw": float(pose[2]),
            "vx": float(vx),
            "vy": float(vy),
            "vw": float(vw),
            "timestamp": now,
        }
        _atomic_write(self._root / self._odom_file, data)


class CmdVelReader:
    """Read ROS2 /cmd_vel from the bridge file and apply to MuJoCo base."""

    def __init__(self, bridge_dir: str = ".bridge", controller=None):
        self._root = _resolve_bridge_dir(bridge_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._controller = controller
        self._cmd_vel_file = "cmd_vel.json"
        self._last_stale_check = 0.0

    def poll_and_apply(self) -> bool:
        """Read the latest cmd_vel and write to MuJoCo base motors.

        Returns True if a fresh command was available, False if stale.
        """
        data = _read_json(self._root / self._cmd_vel_file)
        if data is None:
            return False

        age = time.time() - float(data.get("timestamp", 0))
        if age > 0.5:
            # Stale — zero velocity for safety
            if self._controller is not None:
                self._controller.set_planar_motor_targets(0.0, 0.0, 0.0, max_delta=2.0)
            return False

        vx = float(data.get("vx", 0))
        vy = float(data.get("vy", 0))
        vw = float(data.get("vw", 0))

        if self._controller is not None:
            yaw = self._controller.get_base_pose()[2]
            self._controller.set_planar_motor_targets(
                80.0 * (vx * math.cos(yaw) - vy * math.sin(yaw)),
                80.0 * (vx * math.sin(yaw) + vy * math.cos(yaw)),
                45.0 * vw,
                max_delta=2.0,
            )

        return True


def _resolve_bridge_dir(bridge_dir: str) -> Path:
    p = Path(bridge_dir)
    if not p.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        p = project_root / bridge_dir
    return p


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON to path, handling Windows file locking gracefully."""
    text = json.dumps(data, ensure_ascii=False)
    # Try direct write first (most reliable on Windows with file locking)
    try:
        path.write_text(text, encoding="utf-8")
        return
    except PermissionError:
        pass
    # Fallback: write to temp file and rename
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        tmp.replace(path)
    except PermissionError:
        # Last resort: delete then write
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


__all__ = ["OdometryPublisher", "CmdVelReader"]
