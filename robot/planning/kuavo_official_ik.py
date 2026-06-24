"""Official Kuavo ROS IK service adapter.

This module is intentionally thin: Kuavo's own ROS stack owns whole-body /
arm IK. The local MuJoCo controller may use a diagnostic fallback when ROS is
not running, but successful official calls are clearly marked as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class KuavoOfficialIKResult:
    available: bool
    success: bool
    q_arm: np.ndarray = field(default_factory=lambda: np.empty(0))
    q_torso: np.ndarray = field(default_factory=lambda: np.empty(0))
    q_full: np.ndarray = field(default_factory=lambda: np.empty(0))
    linear_error: float = float("inf")
    angular_error: float = float("inf")
    source: str = ""
    message: str = ""


class KuavoOfficialIKClient:
    """Client for Kuavo official ROS IK services.

    Preferred service:
      /mobile_manipulator_ik_accessibility_check (kuavo_msgs/accessIkSolve)

    The response qBest order is documented by Kuavo as:
      [lower_body_4, left_arm_7, right_arm_7]
    """

    def __init__(
        self,
        enabled: bool = True,
        service_name: str = "/mobile_manipulator_ik_accessibility_check",
        timeout_s: float = 0.2,
        total_time_desired: float = 1.0,
        max_attempts: int = 5,
        linear_error_max: float = 0.05,
        angular_error_max: float = 0.30,
        workspace: str = "",
        use_docker: bool = False,
        docker_container: str = "kuavo_official_ros",
        docker_workspace: str = "/root/kuavo_ws_linux",
        ros_master_uri: str = "http://localhost:11311",
    ):
        self.enabled = bool(enabled)
        self.service_name = service_name
        self.timeout_s = float(timeout_s)
        self.total_time_desired = float(total_time_desired)
        self.max_attempts = int(max_attempts)
        self.linear_error_max = float(linear_error_max)
        self.angular_error_max = float(angular_error_max)
        self.workspace = str(workspace or "")
        self.use_docker = bool(use_docker)
        self.docker_container = str(docker_container)
        self.docker_workspace = str(docker_workspace)
        self.ros_master_uri = str(ros_master_uri)
        self._rospy = None
        self._srv_type = None
        self._req_type = None
        self._import_error = ""
        self._checked_import = False
        self._external_workspace = self._resolve_workspace()

    def available(self) -> bool:
        if not self.enabled:
            return False
        if self.use_docker:
            return self._docker_service_available()
        if not self._ensure_imports():
            return False
        try:
            self._rospy.wait_for_service(self.service_name, timeout=self.timeout_s)
            return True
        except Exception:
            return False

    def solve(
        self,
        side: str,
        pose_xyz_rpy: Sequence[float],
        *,
        is_local: bool = False,
        is_whole_body: bool = False,
        joint_state: dict | None = None,
    ) -> KuavoOfficialIKResult:
        if not self.enabled:
            return self._unavailable("disabled")
        if self.use_docker:
            return self._solve_via_docker(
                side, pose_xyz_rpy,
                is_local=is_local, is_whole_body=is_whole_body,
                joint_state=joint_state,
            )
        if not self._ensure_imports():
            return self._unavailable(self._import_error or "rospy_or_kuavo_msgs_missing")

        pose = list(float(v) for v in pose_xyz_rpy)
        if len(pose) == 3:
            pose.extend([0.0, 0.0, 0.0])
        if len(pose) != 6:
            return KuavoOfficialIKResult(
                available=True,
                success=False,
                q_arm=np.empty(0),
                q_full=np.empty(0),
                linear_error=float("inf"),
                angular_error=float("inf"),
                source="kuavo_official_ik_invalid_pose",
                message=f"expected 3 or 6 pose values, got {len(pose)}",
            )

        try:
            self._rospy.wait_for_service(self.service_name, timeout=self.timeout_s)
            client = self._rospy.ServiceProxy(self.service_name, self._srv_type)
            req = self._req_type()
            req.isLeft = (side == "left")
            req.isLocal = bool(is_local)
            req.isWholeBody = bool(is_whole_body)
            req.poseDesired = pose
            req.totalTimeDesired = self.total_time_desired
            req.maxAttempts = self.max_attempts
            req.linearErrorMax = self.linear_error_max
            req.angularErrorMax = self.angular_error_max
            resp = client(req)
        except Exception as exc:
            return self._unavailable(str(exc))

        q_full = np.asarray(list(resp.qBest), dtype=float)
        if not bool(resp.success) and bool(getattr(resp, "posPriorityAccess", False)):
            q_full = np.asarray(list(resp.qPosPriorityBest), dtype=float)
        q_arm = self._extract_arm(side, q_full)
        success = bool(resp.success) or (q_arm.size == 7 and bool(getattr(resp, "posPriorityAccess", False)))
        source = "kuavo_official_access_ik"
        if success and not bool(resp.success):
            source = "kuavo_official_access_ik_position_priority"
        return KuavoOfficialIKResult(
            available=True,
            success=bool(success and q_arm.size == 7),
            q_arm=q_arm,
            q_full=q_full,
            linear_error=float(getattr(resp, "bestLinearError", float("inf"))),
            angular_error=float(getattr(resp, "bestAngularError", float("inf"))),
            source=source,
            message="",
        )

    def _ensure_imports(self) -> bool:
        if self._checked_import:
            return self._rospy is not None and self._srv_type is not None
        self._checked_import = True
        self._inject_python_paths()
        try:
            import rospy  # type: ignore
            from kuavo_msgs.srv import accessIkSolve, accessIkSolveRequest  # type: ignore
        except Exception as exc:
            self._import_error = (
                f"{exc}. Checked workspace: {self._external_workspace or '<unset>'}. "
                "Make sure kuavo-ros-opensource is built/source'd in a ROS1 environment "
                "so rospy and generated kuavo_msgs Python modules are available."
            )
            return False
        self._rospy = rospy
        self._srv_type = accessIkSolve
        self._req_type = accessIkSolveRequest
        try:
            if not self._rospy.core.is_initialized():
                self._rospy.init_node("mda_kuavo_official_ik_client", anonymous=True, disable_signals=True)
        except Exception:
            pass
        return True

    def _resolve_workspace(self) -> Path | None:
        candidates = []
        if self.workspace:
            candidates.append(Path(self.workspace))
        env_value = os.environ.get("KUAVO_ROS_WS", "")
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

    def _docker_service_available(self) -> bool:
        result = self._docker_bash(
            "rosservice list 2>/dev/null | grep -Fx "
            + shlex.quote(self.service_name)
        )
        return result.returncode == 0

    def _solve_via_docker(
        self,
        side: str,
        pose_xyz_rpy: Sequence[float],
        *,
        is_local: bool,
        is_whole_body: bool,
        joint_state: dict | None = None,
    ) -> KuavoOfficialIKResult:
        pose = list(float(v) for v in pose_xyz_rpy)
        if len(pose) == 3:
            pose.extend([0.0, 0.0, 0.0])
        if len(pose) != 6:
            return KuavoOfficialIKResult(
                available=True,
                success=False,
                q_arm=np.empty(0),
                q_full=np.empty(0),
                linear_error=float("inf"),
                angular_error=float("inf"),
                source="kuavo_official_ik_invalid_pose",
                message=f"expected 3 or 6 pose values, got {len(pose)}",
            )

        payload = {
            "service_name": self.service_name,
            "side": side,
            "pose": pose,
            "is_local": bool(is_local),
            "is_whole_body": bool(is_whole_body),
            "timeout_s": self.timeout_s,
            "total_time_desired": self.total_time_desired,
            "max_attempts": self.max_attempts,
            "linear_error_max": self.linear_error_max,
            "angular_error_max": self.angular_error_max,
        }
        if "two_arm_hand_pose" in self.service_name:
            payload["joint_state"] = joint_state or {}
            return self._solve_two_arm_pose_via_docker(side, pose, payload)
        script = r"""
import json
import rospy
from kuavo_msgs.srv import accessIkSolve, accessIkSolveRequest
payload = json.loads('PAYLOAD_JSON')
rospy.init_node('mda_kuavo_access_ik_cli', anonymous=True, disable_signals=True)
rospy.wait_for_service(payload['service_name'], timeout=float(payload['timeout_s']))
client = rospy.ServiceProxy(payload['service_name'], accessIkSolve)
req = accessIkSolveRequest()
req.isLeft = payload['side'] == 'left'
req.isLocal = bool(payload['is_local'])
req.isWholeBody = bool(payload['is_whole_body'])
req.poseDesired = [float(v) for v in payload['pose']]
req.totalTimeDesired = float(payload['total_time_desired'])
req.maxAttempts = int(payload['max_attempts'])
req.linearErrorMax = float(payload['linear_error_max'])
req.angularErrorMax = float(payload['angular_error_max'])
resp = client(req)
print(json.dumps({
    'available': True,
    'success': bool(resp.success),
    'bestLinearError': float(resp.bestLinearError),
    'bestAngularError': float(resp.bestAngularError),
    'qBest': list(resp.qBest),
    'posPriorityAccess': bool(resp.posPriorityAccess),
    'posPriorityLinearError': float(resp.posPriorityLinearError),
    'posPriorityAngularError': float(resp.posPriorityAngularError),
    'qPosPriorityBest': list(resp.qPosPriorityBest),
}))
""".replace("PAYLOAD_JSON", json.dumps(payload).replace("'", "\\'"))
        result = self._docker_bash("python3 - <<'PY'\n" + script + "\nPY")
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "docker IK call failed"
            return self._unavailable(detail)
        try:
            data = json.loads(result.stdout.strip().splitlines()[-1])
        except Exception as exc:
            return self._unavailable(f"invalid docker IK response: {exc}; stdout={result.stdout!r}")

        q_full = np.asarray(list(data.get("qBest", ())), dtype=float)
        if not bool(data.get("success", False)) and bool(data.get("posPriorityAccess", False)):
            q_full = np.asarray(list(data.get("qPosPriorityBest", ())), dtype=float)
        q_arm = self._extract_arm(side, q_full)
        success = bool(data.get("success", False)) or (
            q_arm.size == 7 and bool(data.get("posPriorityAccess", False))
        )
        source = "kuavo_official_ros1_access_ik"
        if success and not bool(data.get("success", False)):
            source = "kuavo_official_ros1_access_ik_position_priority"
        return KuavoOfficialIKResult(
            available=True,
            success=bool(success and q_arm.size == 7),
            q_arm=q_arm,
            q_full=q_full,
            linear_error=float(data.get("bestLinearError", float("inf"))),
            angular_error=float(data.get("bestAngularError", float("inf"))),
            source=source,
            message="",
        )

    def _solve_two_arm_pose_via_docker(
        self,
        side: str,
        pose: Sequence[float],
        payload: dict,
    ) -> KuavoOfficialIKResult:
        # pose = [x, y, z, roll, pitch, yaw] in base frame.
        # Convert RPY to quaternion using tf.transformations (same as official demo).
        target_pos = [float(pose[0]), float(pose[1]), float(pose[2])]
        rpy = [
            float(pose[3]) if len(pose) > 3 else -1.57,
            float(pose[4]) if len(pose) > 4 else 0.0,
            float(pose[5]) if len(pose) > 5 else 0.0,
        ]
        # Use identity quaternion for non-commanded arm (matching leju_claw_cylinder_pub.py).
        # The commanded arm's quaternion is computed inside Docker from rpy.
        payload = {
            **payload,
            "target_pos": target_pos,
            "target_side": side,
            "target_rpy": rpy,
            "joint_state": payload.get("joint_state", {}),
        }
        script = r"""
import json, math, time
import rospy
from sensor_msgs.msg import JointState
from kuavo_msgs.srv import twoArmHandPoseCmdSrv
from kuavo_msgs.msg import twoArmHandPoseCmd
from tf.transformations import quaternion_from_euler
payload = json.loads('PAYLOAD_JSON')
rospy.init_node('mda_kuavo_two_arm_ik_cli', anonymous=True, disable_signals=True)

# Publish current joint state so the IK knows the robot configuration
js_dict = payload.get('joint_state', {})
if js_dict:
    pub = rospy.Publisher('/joint_states', JointState, queue_size=1, latch=True)
    time.sleep(0.3)
    js = JointState()
    js.header.stamp = rospy.Time.now()
    # IK uses Drake/ROS joint names. MuJoCo names differ — map them.
    # Drake MJCF order for right arm: r_arm_pitch, r_arm_roll, r_arm_yaw,
    #   r_forearm_pitch, r_forearm_yaw, r_hand_roll, r_hand_pitch.
    # IK output order matches the ROS demo names.  For joint state publishing
    # we use the ROS names that the IK plant recognises.
    mj = js_dict
    js.name = [
        'l_arm_pitch','l_arm_roll','l_arm_yaw','l_forearm_pitch','l_hand_yaw','l_hand_pitch','l_hand_roll',
        'r_arm_pitch','r_arm_roll','r_arm_yaw','r_forearm_pitch','r_hand_yaw','r_hand_pitch','r_hand_roll',
        'waist_pitch_joint','waist_yaw_joint',
    ]
    # IK plant: r_hand_pitch (index 5) is physically X-axis=roll,
    #           r_hand_roll  (index 6) is physically Y-axis=pitch.
    # MuJoCo:   zarm_r6 (X=roll), zarm_r7 (Y=pitch).
    # So: z6 value → IK index 5 (r_hand_pitch), z7 value → IK index 6 (r_hand_roll)
    js.position = [
        float(mj.get('zarm_l1_joint',0.349)), float(mj.get('zarm_l2_joint',0)),
        float(mj.get('zarm_l3_joint',0)), float(mj.get('zarm_l4_joint',-0.524)),
        float(mj.get('zarm_l5_joint',0)), float(mj.get('zarm_l6_joint',0)),  # z6(X-roll)→IK r_hand_pitch(idx5)
        float(mj.get('zarm_l7_joint',0)),  # z7(Y-pitch)→IK r_hand_roll(idx6)
        float(mj.get('zarm_r1_joint',0.349)), float(mj.get('zarm_r2_joint',0)),
        float(mj.get('zarm_r3_joint',0)), float(mj.get('zarm_r4_joint',-0.524)),
        float(mj.get('zarm_r5_joint',0)), float(mj.get('zarm_r6_joint',0)),  # z6(X-roll)→IK r_hand_pitch(idx5)
        float(mj.get('zarm_r7_joint',0)),  # z7(Y-pitch)→IK r_hand_roll(idx6)
        float(mj.get('waist_pitch_joint',0)), float(mj.get('waist_yaw_joint',0)),
    ]
    js.velocity = [0.0]*16
    js.effort = [0.0]*16
    pub.publish(js)
    time.sleep(0.5)

rospy.wait_for_service(payload['service_name'], timeout=float(payload['timeout_s']))
client = rospy.ServiceProxy(payload['service_name'], twoArmHandPoseCmdSrv)

# Build request exactly as the official leju_claw_cylinder_pub.py demo does.
req = twoArmHandPoseCmd()
req.use_custom_ik_param = False
req.joint_angles_as_q0 = False
# Non-commanded arm: identity quaternion + default pose (from official demo).
req.hand_poses.left_pose.pos_xyz = [0.0, 0.3, -0.5]
req.hand_poses.left_pose.quat_xyzw = [0.0, 0.0, 0.0, 1.0]
req.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
req.hand_poses.right_pose.pos_xyz = [0.0, 0.3, -0.5]
req.hand_poses.right_pose.quat_xyzw = [0.0, 0.0, 0.0, 1.0]
req.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]

# Commanded arm: use caller's target position + RPY→quaternion.
pos = [float(v) for v in payload['target_pos']]
quat = list(quaternion_from_euler(*payload['target_rpy']))
if payload['target_side'] == 'left':
    req.hand_poses.left_pose.pos_xyz = pos
    req.hand_poses.left_pose.quat_xyzw = quat
else:
    req.hand_poses.right_pose.pos_xyz = pos
    req.hand_poses.right_pose.quat_xyzw = quat

resp = client(req)
print(json.dumps({
    'available': True,
    'success': bool(resp.success),
    'q_arm': list(resp.q_arm),
    'q_torso': list(resp.q_torso),
    'with_torso': bool(resp.with_torso),
    'time_cost': float(resp.time_cost),
    'error_reason': str(resp.error_reason),
}))
""".replace("PAYLOAD_JSON", json.dumps(payload).replace("'", "\\'"))
        result = self._docker_bash("python3 - <<'PY'\n" + script + "\nPY")
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "docker two-arm IK call failed"
            return self._unavailable(detail)
        try:
            data = json.loads(result.stdout.strip().splitlines()[-1])
        except Exception as exc:
            return self._unavailable(f"invalid docker two-arm IK response: {exc}; stdout={result.stdout!r}")
        q_full = np.asarray(list(data.get("q_arm", ())), dtype=float)
        q_torso = np.asarray(list(data.get("q_torso", ())), dtype=float)
        q_arm = self._extract_arm(side, q_full)
        return KuavoOfficialIKResult(
            available=True,
            success=bool(data.get("success", False) and q_arm.size == 7),
            q_arm=q_arm,
            q_torso=q_torso,
            q_full=q_full,
            linear_error=0.0 if data.get("success", False) else float("inf"),
            angular_error=0.0 if data.get("success", False) else float("inf"),
            source="kuavo_official_ros1_two_arm_ik",
            message=str(data.get("error_reason", "")),
        )

    def _docker_bash(self, command: str) -> subprocess.CompletedProcess[str]:
        setup = (
            f"cd {shlex.quote(self.docker_workspace)} && "
            "source /opt/ros/noetic/setup.bash && "
            "test -f installed/setup.bash && source installed/setup.bash; "
            "test -f devel/setup.bash && source devel/setup.bash; "
            f"export ROS_MASTER_URI={shlex.quote(self.ros_master_uri)}; "
        )
        cmd = [
            "docker",
            "exec",
            self.docker_container,
            "bash",
            "-lc",
            setup + command,
        ]
        try:
            return subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=max(self.timeout_s + 2.0, 5.0),
            )
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                cmd,
                124,
                stdout=exc.stdout or "",
                stderr=f"timeout after {max(self.timeout_s + 2.0, 5.0):.1f}s",
            )

    @staticmethod
    def _extract_arm(side: str, q_full: np.ndarray) -> np.ndarray:
        if q_full.size >= 18:
            return q_full[4:11].copy() if side == "left" else q_full[11:18].copy()
        if q_full.size == 14:
            return q_full[:7].copy() if side == "left" else q_full[7:14].copy()
        if q_full.size == 7:
            return q_full.copy()
        return np.empty(0)

    @staticmethod
    def _unavailable(message: str) -> KuavoOfficialIKResult:
        return KuavoOfficialIKResult(
            available=False,
            success=False,
            q_arm=np.empty(0),
            q_full=np.empty(0),
            linear_error=float("inf"),
            angular_error=float("inf"),
            source="kuavo_official_ik_unavailable",
            message=message,
        )


__all__ = ["KuavoOfficialIKClient", "KuavoOfficialIKResult"]
