"""ROS2 navigation executor — sends goals to WSL nav server via bridge files.

Protocol (all communication through .bridge/ directory):

  Windows -> .bridge/nav_goal.json    {"x","y","yaw","timestamp"}  (write once)
  WSL    -> .bridge/cmd_vel.json      {"vx","vy","vw","timestamp"} (20 Hz)
  Windows -> .bridge/odom.json         {"x","y","yaw","vx","vy","vw","timestamp"} (50 Hz)
  WSL    -> .bridge/nav_result.json   {"success","final_error_xy","final_error_yaw"} (write once at end)

The WSL nav server (scripts/kuavo_nav_server.py) polls nav_goal.json,
reads odom.json, writes cmd_vel.json and finally nav_result.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NavResult:
    success: bool
    final_error_xy: float
    final_error_yaw: float
    message: str


class Nav2Executor:
    """Send navigation goals to WSL via bridge file and wait for result."""

    def __init__(self, bridge_dir: str = ".bridge"):
        self._root = _resolve_bridge_dir(bridge_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._goal_file = self._root / "nav_goal.json"
        self._result_file = self._root / "nav_result.json"

    def write_goal(self, x: float, y: float, yaw: float) -> None:
        """Write a navigation goal to the bridge file (non-blocking).

        The caller must run the simulation loop, then call ``poll_result()``
        to check if navigation has completed.
        """
        # Clear stale result
        try:
            self._result_file.unlink()
        except FileNotFoundError:
            pass
        goal = {"x": x, "y": y, "yaw": yaw, "timestamp": time.time()}
        _atomic_write(self._goal_file, goal)

    def poll_result(self) -> NavResult | None:
        """Check if the navigation result is ready. Returns None if still in progress."""
        result = _read_json(self._result_file)
        if result is None:
            return None
        return NavResult(
            success=bool(result.get("success", False)),
            final_error_xy=float(result.get("final_error_xy", float("inf"))),
            final_error_yaw=float(result.get("final_error_yaw", float("inf"))),
            message=result.get("message", ""),
        )

    def navigate_to(
        self,
        x: float,
        y: float,
        yaw: float,
        timeout_s: float = 30.0,
    ) -> NavResult:
        """Write a goal to the bridge and poll for the result."""
        # Clear any stale result
        try:
            self._result_file.unlink()
        except FileNotFoundError:
            pass

        # Write goal
        goal = {"x": x, "y": y, "yaw": yaw, "timestamp": time.time()}
        _atomic_write(self._goal_file, goal)

        # Poll for result
        deadline = time.time() + timeout_s
        poll_interval = 0.1
        while time.time() < deadline:
            result = _read_json(self._result_file)
            if result is not None:
                return NavResult(
                    success=bool(result.get("success", False)),
                    final_error_xy=float(result.get("final_error_xy", float("inf"))),
                    final_error_yaw=float(result.get("final_error_yaw", float("inf"))),
                    message=result.get("message", ""),
                )
            time.sleep(poll_interval)

        return NavResult(
            success=False,
            final_error_xy=float("inf"),
            final_error_yaw=float("inf"),
            message="navigation timed out",
        )

    def cancel(self) -> bool:
        """Cancel by writing an empty goal."""
        goal = {"x": 0, "y": 0, "yaw": 0, "timestamp": time.time(), "cancel": True}
        try:
            _atomic_write(self._goal_file, goal)
            return True
        except Exception:
            return False


class BridgeFileClient:
    """Read/write shared bridge files between Windows and WSL.

    The bridge directory is accessible from both sides:
      Windows:  e:/workspace/bigdata/project/.bridge/
      WSL:      /mnt/e/workspace/bigdata/project/.bridge/
    """

    def __init__(self, bridge_dir: str = ".bridge"):
        self._root = _resolve_bridge_dir(bridge_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def write_json(self, filename: str, data: dict) -> None:
        _atomic_write(self._root / filename, data)

    def read_json(self, filename: str) -> dict | None:
        return _read_json(self._root / filename)

    @property
    def root_path(self) -> Path:
        return self._root


def _resolve_bridge_dir(bridge_dir: str) -> Path:
    p = Path(bridge_dir)
    if not p.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        p = project_root / bridge_dir
    return p


def _atomic_write(path: Path, data: dict) -> None:
    text = json.dumps(data, ensure_ascii=False)
    try:
        path.write_text(text, encoding="utf-8")
        return
    except PermissionError:
        pass
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        tmp.replace(path)
    except PermissionError:
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


__all__ = ["Nav2Executor", "NavResult", "BridgeFileClient"]
