"""Thin bridge to the official Kuavo ROS/SDK control stack.

The task code must not reimplement Kuavo's production control stack. This
adapter only discovers the official SDK and forwards high-level commands to it.
When the SDK/ROS services are not available, full-stack task runs fail before
execution instead of silently falling back to local MuJoCo servos.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class KuavoOfficialControlStatus:
    sdk_import: bool
    ros_ready: bool
    arm_trajectory_ready: bool
    base_control_ready: bool
    claw_ready: bool
    message: str = ""

    @property
    def ready(self) -> bool:
        return bool(
            self.sdk_import
            and self.ros_ready
            and self.arm_trajectory_ready
            and self.base_control_ready
            and self.claw_ready
        )


class KuavoOfficialControlBridge:
    """Adapter for kuavo_humanoid_sdk official motion/control calls."""

    SOURCE = "kuavo_official_sdk_control"

    def __init__(
        self,
        *,
        enabled: bool = True,
        required: bool = False,
        ros_workspace_env: str = "KUAVO_ROS_WS",
        workspace: str = "",
        log_level: str = "ERROR",
        use_docker: bool = False,
        docker_container: str = "kuavo_official_ros",
        docker_workspace: str = "/root/kuavo_ws_linux",
        ros_master_uri: str = "http://localhost:11311",
    ):
        self.enabled = bool(enabled)
        self.required = bool(required)
        self.ros_workspace_env = str(ros_workspace_env)
        self.workspace = str(workspace or "")
        self.log_level = str(log_level)
        self.use_docker = bool(use_docker)
        self.docker_container = str(docker_container)
        self.docker_workspace = str(docker_workspace)
        self.ros_master_uri = str(ros_master_uri)
        self._robot = None
        self._claw = None
        self._types = {}
        self._init_error = ""
        self._initialized = False
        self._external_workspace = self._resolve_workspace()

    def ready_status(self) -> KuavoOfficialControlStatus:
        if not self.enabled:
            return KuavoOfficialControlStatus(False, False, False, False, False, "disabled")
        if self.use_docker:
            return self._docker_ready_status()
        if not self._ensure_initialized():
            return KuavoOfficialControlStatus(False, False, False, False, False, self._init_error)

        arm_ok = all(
            hasattr(self._robot, name)
            for name in ("control_arm_joint_trajectory", "set_external_control_arm_mode")
        )
        base_ok = any(
            hasattr(self._robot, name)
            for name in ("control_command_pose_world", "control_command_pose", "walk")
        )
        claw_ok = self._claw is not None and all(
            hasattr(self._claw, name) for name in ("open", "close", "control_right")
        )
        return KuavoOfficialControlStatus(
            sdk_import=True,
            ros_ready=True,
            arm_trajectory_ready=bool(arm_ok),
            base_control_ready=bool(base_ok),
            claw_ready=bool(claw_ok),
            message="ready" if arm_ok and base_ok and claw_ok else "SDK object missing expected methods",
        )

    def require_ready(self) -> None:
        status = self.ready_status()
        if status.ready:
            return
        raise RuntimeError(
            "Kuavo official control stack is not ready. "
            f"sdk_import={status.sdk_import}, ros_ready={status.ros_ready}, "
            f"arm_trajectory={status.arm_trajectory_ready}, "
            f"base_control={status.base_control_ready}, claw={status.claw_ready}. "
            f"Detail: {status.message}. Install/source kuavo-ros-opensource and "
            f"kuavo_humanoid_sdk; set {self.ros_workspace_env} if needed."
        )

    def send_arm_joint_trajectory(
        self,
        joint_positions: np.ndarray,
        *,
        dt: float = 1.0 / 50.0,
    ) -> bool:
        """Send a time-parameterized arm trajectory to the official SDK.

        ``joint_positions`` is expected to be Nx14 in official Kuavo arm order
        (left 7 followed by right 7), matching KuavoRobot.control_arm_joint_trajectory.
        """
        status = self.ready_status()
        if not status.ready:
            if self.required:
                self.require_ready()
            return False
        q = np.asarray(joint_positions, dtype=float)
        if q.ndim != 2 or q.shape[1] != 14 or q.shape[0] == 0:
            raise ValueError(f"expected Nx14 Kuavo arm trajectory, got {q.shape}")
        if self.use_docker:
            return self._docker_publish_arm_joint_trajectory(q, dt=dt)
        times = [float((i + 1) * dt) for i in range(q.shape[0])]
        try:
            if hasattr(self._robot, "set_external_control_arm_mode"):
                self._robot.set_external_control_arm_mode()
            return bool(self._robot.control_arm_joint_trajectory(times, q.tolist()))
        except Exception as exc:
            if self.required:
                raise RuntimeError(f"official Kuavo arm trajectory execution failed: {exc}") from exc
            self._init_error = str(exc)
            return False

    def _docker_publish_arm_joint_trajectory(self, q: np.ndarray, *, dt: float) -> bool:
        q = np.asarray(q, dtype=float)
        if q.shape[0] > 16:
            keep = np.linspace(0, q.shape[0] - 1, 16).round().astype(int)
            q_send = q[keep]
        else:
            q_send = q
        payload = q_send.round(6).tolist()
        script = (
            "python3 - <<'PY'\n"
            "import rospy\n"
            "from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint\n"
            "rospy.init_node('mda_kuavo_arm_trajectory_command', anonymous=True, disable_signals=True)\n"
            "pub = rospy.Publisher('/bezier/arm_traj', JointTrajectory, queue_size=1)\n"
            "rospy.sleep(0.2)\n"
            f"points = {payload!r}\n"
            f"dt = {float(dt)!r}\n"
            "names = ['zarm_l1_joint','zarm_l2_joint','zarm_l3_joint','zarm_l4_joint','zarm_l5_joint','zarm_l6_joint','zarm_l7_joint',"
            "'zarm_r1_joint','zarm_r2_joint','zarm_r3_joint','zarm_r4_joint','zarm_r5_joint','zarm_r6_joint','zarm_r7_joint']\n"
            "for i, q in enumerate(points):\n"
            "    msg = JointTrajectory()\n"
            "    msg.header.stamp = rospy.Time.now()\n"
            "    msg.joint_names = names\n"
            "    pt = JointTrajectoryPoint()\n"
            "    pt.positions = list(q) + [0.0] * 14\n"
            "    pt.velocities = [0.0] * len(pt.positions)\n"
            "    pt.time_from_start = rospy.Duration.from_sec(max(dt, 0.02))\n"
            "    msg.points = [pt]\n"
            "    pub.publish(msg)\n"
            "    rospy.sleep(max(dt, 0.02))\n"
            "print('ARM_TRAJ_CMD_OK points=%d connections=%d' % (len(points), pub.get_num_connections()))\n"
            "PY"
        )
        result = self._docker_bash(script)
        ok = result.returncode == 0 and "ARM_TRAJ_CMD_OK" in result.stdout
        if not ok:
            self._init_error = result.stderr.strip() or result.stdout.strip() or "docker arm trajectory command failed"
        return bool(ok)

    def command_base_world(self, xy_yaw: Sequence[float]) -> bool:
        """Send a world-frame base/torso pose command through the official SDK."""
        if self.use_docker:
            goal = list(float(v) for v in xy_yaw)
            if len(goal) < 3:
                raise ValueError("base command requires [x, y, yaw]")
            vx = float(np.clip(goal[0], -0.22, 0.22))
            vy = float(np.clip(goal[1], -0.22, 0.22))
            wz = float(np.clip(goal[2], -0.55, 0.55))
            return self._docker_publish_base_twist(vx, vy, wz)
        status = self.ready_status()
        if not status.ready:
            if self.required:
                self.require_ready()
            return False
        goal = list(float(v) for v in xy_yaw)
        if len(goal) < 3:
            raise ValueError("base command requires [x, y, yaw]")
        try:
            if hasattr(self._robot, "control_command_pose_world"):
                return bool(self._robot.control_command_pose_world(goal[0], goal[1], 0.0, goal[2]))
            if hasattr(self._robot, "control_command_pose"):
                return bool(self._robot.control_command_pose(goal[0], goal[1], 0.0, goal[2]))
            return False
        except Exception as exc:
            if self.required:
                raise RuntimeError(f"official Kuavo base command failed: {exc}") from exc
            self._init_error = str(exc)
            return False

    def command_base_stop(self) -> bool:
        if self.use_docker:
            return self._docker_publish_base_twist(0.0, 0.0, 0.0)
        return True

    def _docker_publish_base_twist(self, vx: float, vy: float, wz: float) -> bool:
        script = (
            "python3 - <<'PY'\n"
            "import rospy\n"
            "from geometry_msgs.msg import Twist\n"
            "rospy.init_node('mda_kuavo_base_command', anonymous=True, disable_signals=True)\n"
            "pub = rospy.Publisher('/move_base/base_cmd_vel', Twist, queue_size=1)\n"
            "rospy.sleep(0.15)\n"
            "msg = Twist()\n"
            f"msg.linear.x = {float(vx)!r}\n"
            f"msg.linear.y = {float(vy)!r}\n"
            f"msg.angular.z = {float(wz)!r}\n"
            "for _ in range(3):\n"
            "    pub.publish(msg)\n"
            "    rospy.sleep(0.03)\n"
            "print('BASE_CMD_OK vx=%.4f vy=%.4f wz=%.4f connections=%d' % (msg.linear.x, msg.linear.y, msg.angular.z, pub.get_num_connections()))\n"
            "PY"
        )
        result = self._docker_bash(script)
        ok = result.returncode == 0 and "BASE_CMD_OK" in result.stdout
        if not ok:
            self._init_error = result.stderr.strip() or result.stdout.strip() or "docker base command failed"
        return bool(ok)

    def open_claw(self, side: str = "right") -> bool:
        return self._control_claw(opening=True, side=side)

    def close_claw(self, side: str = "right") -> bool:
        return self._control_claw(opening=False, side=side)

    def _control_claw(self, *, opening: bool, side: str) -> bool:
        if self.use_docker:
            return self._docker_publish_claw(opening=opening)
        status = self.ready_status()
        if not status.ready:
            if self.required:
                self.require_ready()
            return False
        side_enum = self._types.get("EndEffectorSide")
        sdk_side = None
        if side_enum is not None:
            sdk_side = side_enum.RIGHT if side == "right" else side_enum.LEFT
        try:
            if opening and hasattr(self._claw, "open"):
                return bool(self._claw.open(sdk_side) if sdk_side is not None else self._claw.open())
            if not opening and hasattr(self._claw, "close"):
                return bool(self._claw.close(sdk_side) if sdk_side is not None else self._claw.close())
            if side == "right" and hasattr(self._claw, "control_right"):
                return bool(self._claw.control_right([100.0 if opening else 0.0]))
            return False
        except Exception as exc:
            if self.required:
                raise RuntimeError(f"official Kuavo claw command failed: {exc}") from exc
            self._init_error = str(exc)
            return False

    def _docker_publish_claw(self, *, opening: bool) -> bool:
        position = [10.0, 10.0] if opening else [90.0, 90.0]
        velocity = [90.0, 90.0]
        effort = [1.0, 1.0]
        script = (
            "python3 - <<'PY'\n"
            "import time\n"
            "import rospy\n"
            "from kuavo_msgs.msg import lejuClawCommand\n"
            "rospy.init_node('mda_kuavo_claw_command', anonymous=True, disable_signals=True)\n"
            "pub = rospy.Publisher('/leju_claw_command', lejuClawCommand, queue_size=1)\n"
            "deadline = time.time() + 1.0\n"
            "while pub.get_num_connections() == 0 and time.time() < deadline and not rospy.is_shutdown():\n"
            "    rospy.sleep(0.05)\n"
            "msg = lejuClawCommand()\n"
            "msg.data.name = ['left_claw', 'right_claw']\n"
            f"msg.data.position = {position!r}\n"
            f"msg.data.velocity = {velocity!r}\n"
            f"msg.data.effort = {effort!r}\n"
            "for _ in range(3):\n"
            "    pub.publish(msg)\n"
            "    rospy.sleep(0.05)\n"
            "print('CLAW_CMD_OK connections=%d position=%s' % (pub.get_num_connections(), list(msg.data.position)))\n"
            "PY"
        )
        result = self._docker_bash(script)
        ok = result.returncode == 0 and "CLAW_CMD_OK" in result.stdout
        if not ok:
            self._init_error = result.stderr.strip() or result.stdout.strip() or "docker claw command failed"
        return bool(ok)

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._robot is not None
        self._initialized = True
        self._inject_python_paths()
        try:
            from kuavo_humanoid_sdk.kuavo.robot import KuavoRobot, KuavoSDK  # type: ignore
            from kuavo_humanoid_sdk.kuavo.leju_claw import LejuClaw  # type: ignore
            from kuavo_humanoid_sdk.interfaces.data_types import EndEffectorSide  # type: ignore
        except Exception as exc:
            self._init_error = (
                f"official SDK import failed: {exc}. "
                f"Checked workspace: {self._external_workspace or '<unset>'}"
            )
            return False

        workspace = os.environ.get(self.ros_workspace_env, "") or str(self._external_workspace or "")
        if self.required and not workspace:
            # The SDK may still work if the shell has already sourced the ROS
            # workspace. Keep this as a warning-level detail instead of a hard
            # import failure; ROS init below is the true readiness check.
            self._init_error = f"{self.ros_workspace_env} is not set"
        try:
            options = getattr(KuavoSDK, "Options").WithIK
            KuavoSDK.Init(options=options, log_level=self.log_level)
            self._robot = KuavoRobot()
            self._claw = LejuClaw()
            self._types["EndEffectorSide"] = EndEffectorSide
            self._init_error = ""
            return True
        except SystemExit as exc:
            self._init_error = f"official SDK exited during init: {exc}"
            return False
        except Exception as exc:
            detail = str(exc)
            if self._init_error:
                detail = f"{self._init_error}; {detail}"
            self._init_error = detail
            return False

    def _resolve_workspace(self) -> Path | None:
        candidates = []
        if self.workspace:
            candidates.append(Path(self.workspace))
        env_value = os.environ.get(self.ros_workspace_env, "")
        if env_value:
            candidates.append(Path(env_value))
        candidates.extend([
            Path(r"E:\workspace\kuavo-ros-opensource"),
            Path(r"\\wsl.localhost\UbuntuRobot\root\workspace\kuavo-ros-opensource"),
            Path(r"\\wsl$\UbuntuRobot\root\workspace\kuavo-ros-opensource"),
            Path("/root/workspace/kuavo-ros-opensource"),
            Path("/mnt/e/workspace/kuavo-ros-opensource"),
            Path.home() / "workspace" / "kuavo-ros-opensource",
        ])
        for candidate in candidates:
            try:
                if candidate.exists():
                    return candidate.resolve()
            except OSError:
                continue
        return None

    def _inject_python_paths(self) -> None:
        root = self._external_workspace
        if root is None:
            return
        candidates = [
            root / "src" / "kuavo_humanoid_sdk",
            root / "installed" / "lib" / "python3" / "dist-packages",
            root / "installed" / "lib" / "python3.8" / "dist-packages",
            root / "installed" / "lib" / "python3.10" / "dist-packages",
            root / "installed" / "lib" / "python3.12" / "dist-packages",
        ]
        for path in candidates:
            if path.exists():
                text = str(path)
                if text not in sys.path:
                    sys.path.insert(0, text)

    def _docker_ready_status(self) -> KuavoOfficialControlStatus:
        result = self._docker_bash(
            "printf '__NODES__\\n'; timeout 2s rosnode list; "
            "printf '__SERVICES__\\n'; timeout 2s rosservice list; "
            "printf '__TOPICS__\\n'; timeout 2s rostopic list"
        )
        if result.returncode != 0:
            return KuavoOfficialControlStatus(
                False,
                False,
                False,
                False,
                False,
                result.stderr.strip() or result.stdout.strip() or "docker ROS1 graph query failed",
            )
        text = result.stdout
        ros_ready = "__SERVICES__" in text and "__TOPICS__" in text and "ERROR:" not in text
        lower = text.lower()
        arm_ready = any(
            key in lower
            for key in (
                "plan_arm_trajectory",
                "planarmtrajectory",
                "execute_arm_action",
                "interrupt_arm_traj",
                "bezier/arm_traj",
                "kuavo_arm_traj",
                "change_arm_ctrl_mode",
                "changearmctrlmode",
                "two_arm_hand_pose",
                "follow_joint_trajectory",
            )
        )
        base_ready = any(
            key in lower
            for key in (
                "cmd_vel",
                "chassis",
                "navigate",
                "task_point",
                "base_link_pose",
                "control_command_pose",
            )
        )
        claw_ready = any(
            key in lower
            for key in ("claw", "gripper", "hand_position", "leju", "control_robot_hand_position")
        )
        sdk_import = bool(ros_ready)
        missing = []
        if not ros_ready:
            missing.append("roscore/ROS1 graph")
        if not arm_ready:
            missing.append("official arm trajectory service/topic")
        if not base_ready:
            missing.append("official base/navigation service/topic")
        if not claw_ready:
            missing.append("official LejuClaw/gripper service/topic")
        return KuavoOfficialControlStatus(
            sdk_import=bool(sdk_import),
            ros_ready=bool(ros_ready),
            arm_trajectory_ready=bool(arm_ready),
            base_control_ready=bool(base_ready),
            claw_ready=bool(claw_ready),
            message="ready" if not missing else "missing " + ", ".join(missing),
        )

    def _docker_sdk_import_ok(self) -> bool:
        sdk_path = shlex.quote(f"{self.docker_workspace}/src/kuavo_humanoid_sdk")
        ws_sdk_path = shlex.quote(f"{self.docker_workspace}/src/kuavo_humanoid_websocket_sdk")
        result = self._docker_bash(
            f"export PYTHONPATH={sdk_path}:{ws_sdk_path}:$PYTHONPATH"
            + "; python3 -c 'import kuavo_humanoid_sdk.kuavo.robot; import kuavo_humanoid_sdk.kuavo.leju_claw'"
        )
        return result.returncode == 0

    def _docker_bash(self, command: str) -> subprocess.CompletedProcess[str]:
        setup = (
            f"cd {shlex.quote(self.docker_workspace)} && "
            "source /opt/ros/noetic/setup.bash && "
            "test -f installed/setup.bash && source installed/setup.bash; "
            "test -f devel/setup.bash && source devel/setup.bash; "
            f"export ROS_MASTER_URI={shlex.quote(self.ros_master_uri)}; "
        )
        cmd = ["docker", "exec", self.docker_container, "bash", "-lc", setup + command]
        try:
            return subprocess.run(cmd, text=True, capture_output=True, timeout=8.0)
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                cmd,
                124,
                stdout=exc.stdout or "",
                stderr="timeout after 8.0s",
            )


__all__ = ["KuavoOfficialControlBridge", "KuavoOfficialControlStatus"]
