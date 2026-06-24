"""Kuavo wheel-arm controller for Task 3.19 in MuJoCo."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from control import MuJoCoEnv
from robot_common.infra.logging import logger
from planning import (
    KuavoKinematicsSolver,
    KuavoOfficialControlBridge,
    KuavoOfficialIKClient,
    TrajectoryGenerator,
)

from .manipulation_controller import PlannedArmPath, RobotManipulationController


class KuavoWheelGarbageController(RobotManipulationController):
    """Name-mapped controller for the official Kuavo 5-W v62 wheel-arm model."""

    LIFT_IDX = 0
    ARM_EXTEND_IDX = 0
    OPEN_GRIP_CTRL = -80.0
    CLOSED_GRIP_CTRL = 0.0

    LEFT_ARM_JOINTS = [f"zarm_l{i}_joint" for i in range(1, 8)]
    RIGHT_ARM_JOINTS = [f"zarm_r{i}_joint" for i in range(1, 8)]
    FULL_BODY_IK_JOINTS = [
        "knee_joint",
        "leg_joint",
        "waist_pitch_joint",
        "waist_yaw_joint",
        *LEFT_ARM_JOINTS,
        *RIGHT_ARM_JOINTS,
    ]
    WHEEL_YAW_JOINTS = [
        "LF_wheel_yaw_joint",
        "RF_wheel_yaw_joint",
        "LB_wheel_yaw_joint",
        "RB_wheel_yaw_joint",
    ]
    WHEEL_PITCH_JOINTS = [
        "LF_wheel_pitch_joint",
        "RF_wheel_pitch_joint",
        "LB_wheel_pitch_joint",
        "RB_wheel_pitch_joint",
    ]
    WHEEL_POSITIONS = {
        "LF": np.array([0.253, 0.1785], dtype=float),
        "RF": np.array([0.253, -0.1785], dtype=float),
        "LB": np.array([-0.253, 0.1785], dtype=float),
        "RB": np.array([-0.253, -0.1785], dtype=float),
    }

    def __init__(
        self,
        env: MuJoCoEnv,
        official_ik_config: dict | None = None,
        official_control_config: dict | None = None,
    ):
        self.env = env
        official_ik_config = official_ik_config or {}
        official_control_config = official_control_config or {}
        self.official_ik = KuavoOfficialIKClient(
            enabled=bool(official_ik_config.get("enabled", True)),
            service_name=str(official_ik_config.get("service_name", "/mobile_manipulator_ik_accessibility_check")),
            timeout_s=float(official_ik_config.get("timeout_s", 0.2)),
            linear_error_max=float(official_ik_config.get("linear_error_max", 0.05)),
            angular_error_max=float(official_ik_config.get("angular_error_max", 0.30)),
            workspace=str(official_ik_config.get("workspace", "")),
            use_docker=bool(official_ik_config.get("use_docker", False)),
            docker_container=str(official_ik_config.get("docker_container", "kuavo_official_ros")),
            docker_workspace=str(official_ik_config.get("docker_workspace", "/root/kuavo_ws_linux")),
            ros_master_uri=str(official_ik_config.get("ros_master_uri", "http://localhost:11311")),
        )
        self.allow_local_ik_fallback = bool(official_ik_config.get("allow_local_fallback", True))
        self._standalone_ik_config = dict(official_ik_config.get("standalone_ik", {}) or {})
        self.require_official_control = bool(official_control_config.get("required", False))
        self.official_control = KuavoOfficialControlBridge(
            enabled=bool(official_control_config.get("enabled", True)),
            required=self.require_official_control,
            ros_workspace_env=str(official_control_config.get("ros_workspace_env", "KUAVO_ROS_WS")),
            workspace=str(official_control_config.get("workspace", "")),
            log_level=str(official_control_config.get("log_level", "ERROR")),
            use_docker=bool(official_control_config.get("use_docker", False)),
            docker_container=str(official_control_config.get("docker_container", "kuavo_official_ros")),
            docker_workspace=str(official_control_config.get("docker_workspace", "/root/kuavo_ws_linux")),
            ros_master_uri=str(official_control_config.get("ros_master_uri", "http://localhost:11311")),
        )
        self._official_ik_warned = False
        self._official_control_warned = False
        self._standalone_ik_warned = False
        self._standalone_ik_error = ""
        self._kinematics_solvers: dict[str, KuavoKinematicsSolver] = {}
        self._actuator_by_joint = self._build_actuator_by_joint()
        self.max_ctrl_delta = 2.0
        self.max_trace_ctrl_delta = 2.35
        self.max_gripper_ctrl_delta = 1.5
        self._last_planar_ctrl = np.zeros(3)
        self._held_arm_targets: dict[str, tuple[list[str], np.ndarray]] = {}
        self.arm_settle_steps = 20
        self._right_grip_open = self.OPEN_GRIP_CTRL
        self._right_grip_closed = self.CLOSED_GRIP_CTRL
        self._wheel_radius = 0.075
        self._posture_targets = {
            "knee_joint": 0.0,
            "leg_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "zhead_1_joint": 0.0,
            "zhead_2_joint": 0.0,
        }
        self.reset_to_home()

    def require_official_stack(self, required: bool = True) -> None:
        """Switch controller into full-stack mode.

        Full-stack mode forbids local DLS/MuJoCo-PD execution as a successful
        control source. It is intentionally harsh so task results cannot be
        mistaken for official Kuavo control when ROS/SDK services are absent.
        """
        self.require_official_control = bool(required)
        self.allow_local_ik_fallback = not bool(required)
        self.official_control.required = bool(required)

    def official_stack_status(self) -> dict:
        ik_available = self._standalone_ik_available("right")
        ik_message = self._standalone_ik_error
        control = self.official_control.ready_status()
        return {
            "ik_available": bool(ik_available),
            "ik_message": ik_message,
            "control_ready": bool(control.ready),
            "sdk_import": bool(control.sdk_import),
            "ros_ready": bool(control.ros_ready),
            "arm_trajectory_ready": bool(control.arm_trajectory_ready),
            "base_control_ready": bool(control.base_control_ready),
            "claw_ready": bool(control.claw_ready),
            "message": control.message,
            "allow_local_ik_fallback": bool(self.allow_local_ik_fallback),
            "require_official_control": bool(self.require_official_control),
        }

    def validate_official_stack(self) -> None:
        status = self.official_stack_status()
        if not status["ik_available"]:
            raise RuntimeError(
                "Kuavo standalone IK bridge is not ready: "
                f"{status['ik_message'] or 'unknown error'}"
            )
        self.official_control.require_ready()
        if status["allow_local_ik_fallback"]:
            raise RuntimeError("Full Kuavo execution forbids local IK fallback")

    def reset_to_home(self):
        """Reset the planar mobile base and set nominal Kuavo posture targets."""
        mujoco.mj_resetData(self.env.model, self.env.data)
        self._held_arm_targets.clear()
        self._last_planar_ctrl[:] = 0.0
        for joint_name, target in self._posture_targets.items():
            self._set_actuator_for_joint(joint_name, target)
        self._set_named_actuator("r_fingers_actuator", self._right_grip_open)
        mujoco.mj_forward(self.env.model, self.env.data)

    def _build_actuator_by_joint(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for a_id in range(self.env.model.nu):
            joint_id = int(self.env.model.actuator_trnid[a_id, 0])
            if joint_id < 0:
                continue
            name = self.env.model.joint(joint_id).name
            if name:
                mapping[name] = a_id
        return mapping

    def _actuator_ctrl_range(self, act: int) -> tuple[float, float]:
        return float(self.env.model.actuator_ctrlrange[act, 0]), float(self.env.model.actuator_ctrlrange[act, 1])

    def _joint_qvel_for_joint(self, joint_name: str) -> float:
        jid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return 0.0
        return float(self.env.data.qvel[int(self.env.model.jnt_dofadr[jid])])

    def _set_actuator_for_joint(self, joint_name: str, value: float):
        act = self._actuator_by_joint.get(joint_name)
        if act is None:
            return
        if int(self.env.model.actuator_biastype[act]) != int(mujoco.mjtBias.mjBIAS_NONE):
            lo, hi = self._actuator_ctrl_range(act)
            self.env.data.ctrl[act] = float(np.clip(value, lo, hi))
            return
        # Official biped_s60 arm/head/wheel-yaw actuators are torque motors.
        # Treat the incoming value as a joint-position target and close a small
        # MuJoCo-side PD loop. This mirrors the real stack's lower-level servo
        # layer without writing qpos.
        q = self._qpos_for_joint(joint_name)
        qd = self._joint_qvel_for_joint(joint_name)
        kp = 35.0 if joint_name.startswith("zarm_") else 35.0
        kd = 2.2 if joint_name.startswith("zarm_") else 2.0
        if joint_name.startswith("zhead_"):
            kp, kd = 20.0, 1.5
        if "wheel_yaw" in joint_name:
            kp, kd = 28.0, 2.0
        torque = kp * (float(value) - q) - kd * qd
        lo, hi = self._actuator_ctrl_range(act)
        self.env.data.ctrl[act] = float(np.clip(torque, lo, hi))

    def _hold_posture(self, exclude: set[str] | None = None):
        excluded = exclude or set()
        for joint_name, target in self._posture_targets.items():
            if joint_name not in excluded:
                self._set_actuator_for_joint(joint_name, target)

    def hold_arm_targets(self, side: str | None = None) -> None:
        """Keep the latest commanded arm joint targets active during base motion."""
        for held_side, (joint_names, q_targets) in self._held_arm_targets.items():
            if side is not None and held_side != side:
                continue
            for joint_name, value in zip(joint_names, q_targets):
                self._set_actuator_for_joint(joint_name, float(value))

    def _set_velocity_for_joint(self, joint_name: str, velocity: float):
        act = self._actuator_by_joint.get(joint_name)
        if act is None:
            return
        qd = self._joint_qvel_for_joint(joint_name)
        gain = 2.0 if "wheel_pitch" in joint_name else 4.5
        torque = gain * (float(velocity) - qd)
        lo, hi = self._actuator_ctrl_range(act)
        current = float(self.env.data.ctrl[act])
        delta_limit = 1.0 if "wheel_pitch" in joint_name else self.max_ctrl_delta
        self.env.data.ctrl[act] = current + float(
            np.clip(np.clip(torque, lo, hi) - current, -delta_limit, delta_limit)
        )

    def _set_named_actuator(self, actuator_name: str, value: float):
        act = self.env.actuator_index(actuator_name)
        if act is None:
            return
        lo, hi = self._actuator_ctrl_range(act)
        current = float(self.env.data.ctrl[act])
        target = float(np.clip(value, lo, hi))
        delta = float(np.clip(
            target - current,
            -self.max_trace_ctrl_delta,
            self.max_trace_ctrl_delta,
        ))
        self.env.data.ctrl[act] = current + delta

    def _named_ctrl_value(self, actuator_name: str) -> float:
        act = self.env.actuator_index(actuator_name)
        if act is None:
            return 0.0
        return float(self.env.data.ctrl[act])

    def set_planar_motor_targets(
        self,
        x_force: float,
        y_force: float,
        yaw_torque: float,
        max_delta: float = 2.0,
    ) -> None:
        """Ramp planar base motor commands to avoid visible control sign flips."""
        max_delta = min(float(max_delta), self.max_trace_ctrl_delta)
        desired = np.array([x_force, y_force, yaw_torque], dtype=float)
        desired = np.array([
            np.clip(desired[0], -1800.0, 1800.0),
            np.clip(desired[1], -1800.0, 1800.0),
            np.clip(desired[2], -1200.0, 1200.0),
        ], dtype=float)
        current = np.array([
            self._named_ctrl_value("base_x_motor"),
            self._named_ctrl_value("base_y_motor"),
            self._named_ctrl_value("base_yaw_motor"),
        ], dtype=float)
        self._last_planar_ctrl = current + np.clip(desired - current, -max_delta, max_delta)
        self._set_named_actuator("base_x_motor", float(self._last_planar_ctrl[0]))
        self._set_named_actuator("base_y_motor", float(self._last_planar_ctrl[1]))
        self._set_named_actuator("base_yaw_motor", float(self._last_planar_ctrl[2]))

    def _qpos_for_joint(self, joint_name: str) -> float:
        jid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return 0.0
        return float(self.env.data.qpos[int(self.env.model.jnt_qposadr[jid])])

    def get_base_pose(self) -> np.ndarray:
        pos = self.env.get_body_position("base_link")
        quat = self.env.get_body_quat("base_link")
        w, x, y, z = quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([pos[0], pos[1], yaw], dtype=float)

    def get_camera_frame(self) -> str:
        return "camera_rgb"

    def get_arm_end_effector_pos(self) -> np.ndarray:
        return self.env.get_body_position("r_pinch")

    def get_grasp_center_pos(self) -> np.ndarray:
        """Return the physical pad center used for official claw pinch checks."""
        try:
            f_gid = self.env.geom_index("r_f_fingers_pad")
            b_gid = self.env.geom_index("r_b_fingers_pad")
            if f_gid is not None and b_gid is not None:
                return 0.5 * (
                    self.env.data.geom_xpos[int(f_gid)]
                    + self.env.data.geom_xpos[int(b_gid)]
                )
        except Exception:
            pass
        try:
            return 0.5 * (
                self.env.get_body_position("r_f_fingers")
                + self.env.get_body_position("r_b_fingers")
            )
        except Exception:
            return self.get_arm_end_effector_pos()

    def finger_span(self) -> float:
        try:
            f_gid = self.env.geom_index("r_f_fingers_pad")
            b_gid = self.env.geom_index("r_b_fingers_pad")
            if f_gid is not None and b_gid is not None:
                left = self.env.data.geom_xpos[int(f_gid)]
                right = self.env.data.geom_xpos[int(b_gid)]
                return float(np.linalg.norm(left - right))
            left = self.env.get_body_position("r_f_fingers")
            right = self.env.get_body_position("r_b_fingers")
            return float(np.linalg.norm(left - right))
        except Exception:
            return float("nan")

    def hold_base_stationary(self):
        self.set_planar_motor_targets(0.0, 0.0, 0.0, max_delta=self.max_trace_ctrl_delta)
        for joint_name in self.WHEEL_PITCH_JOINTS:
            self._set_velocity_for_joint(joint_name, 0.0)

    def damp_base_motion(self) -> None:
        """Hold wheels still without injecting planar position corrections."""
        self.set_planar_motor_targets(0.0, 0.0, 0.0, max_delta=4.0)
        for joint_name in self.WHEEL_PITCH_JOINTS:
            self._set_velocity_for_joint(joint_name, 0.0)

    def _hold_base_mirror(self, reference_pose: np.ndarray) -> None:
        """Simulation mirror: resist base drift during arm motion.

        The official Kuavo WBC handles base stabilisation on the real robot.
        This MuJoCo-side PD hold only prevents simulation drift so contact
        physics remain valid — it is NOT a replacement for official control.
        """
        ref = np.asarray(reference_pose, dtype=float)
        pose = self.get_base_pose()
        x_vel = self._joint_qvel_for_joint("base_x_joint")
        y_vel = self._joint_qvel_for_joint("base_y_joint")
        yaw_vel = self._joint_qvel_for_joint("base_yaw_joint")
        yaw_err = math.atan2(math.sin(ref[2] - pose[2]), math.cos(ref[2] - pose[2]))
        self.set_planar_motor_targets(
            260.0 * float(ref[0] - pose[0]) - 70.0 * x_vel,
            260.0 * float(ref[1] - pose[1]) - 70.0 * y_vel,
            180.0 * yaw_err - 45.0 * yaw_vel,
            max_delta=28.0,
        )
        for joint_name in self.WHEEL_PITCH_JOINTS:
            self._set_velocity_for_joint(joint_name, 0.0)

    # Backward-compatible alias used by pipeline.py
    hold_planar_pose_pd = _hold_base_mirror

    def move_planar_to(self, goal_xy_yaw: np.ndarray, steps: int = 1800, tolerance: float = 0.08) -> bool:
        """Move planar base to goal via official SDK + MuJoCo PD mirror."""
        goal = np.asarray(goal_xy_yaw, dtype=float)
        if self.require_official_control and not self.official_control.command_base_world(goal[:3]):
            return False
        for _ in range(int(steps)):
            pose = self.get_base_pose()
            xy_err = goal[:2] - pose[:2]
            yaw_err = math.atan2(math.sin(goal[2] - pose[2]), math.cos(goal[2] - pose[2]))
            if float(np.linalg.norm(xy_err)) <= tolerance and abs(yaw_err) <= 0.14:
                self._ramp_planar_stop()
                return True
            self.set_planar_motor_targets(
                float(np.clip(4000.0 * xy_err[0], -1800.0, 1800.0)),
                float(np.clip(4000.0 * xy_err[1], -1800.0, 1800.0)),
                float(np.clip(1500.0 * yaw_err, -900.0, 900.0)),
                max_delta=60.0,
            )
            self.hold_arm_targets()
            self._hold_posture()
            self.env.step()
        self._ramp_planar_stop()
        return False

    def hold_base_pose(self, reference_pose: np.ndarray):
        """Hold the planar base at a reference pose (position-level only)."""
        ref = np.asarray(reference_pose, dtype=float)
        pose = self.get_base_pose()
        yaw_err = math.atan2(math.sin(ref[2] - pose[2]), math.cos(ref[2] - pose[2]))
        self.set_planar_motor_targets(
            90.0 * (ref[0] - pose[0]),
            90.0 * (ref[1] - pose[1]),
            55.0 * yaw_err,
            max_delta=2.0,
        )
        for joint_name in self.WHEEL_PITCH_JOINTS:
            self._set_velocity_for_joint(joint_name, 0.0)

    def move_base(self, forward_vel: float, turn_vel: float):
        """Command a short-horizon base velocity (simulation mirror).

        Official base control is handled by the ROS2 nav server or the
        official SDK. This method exists only to satisfy the abstract
        contract and support the Stretch fallback path.
        """
        forward = float(np.clip(forward_vel, -0.25, 0.25))
        turn = float(np.clip(turn_vel, -0.7, 0.7))
        self.set_planar_motor_targets(
            80.0 * forward,
            0.0,
            45.0 * turn,
            max_delta=2.0,
        )

    def move_base_to(self, goal_xy_yaw: np.ndarray, steps: int = 300) -> bool:
        """Drive the base toward a world-frame goal (delegates to ROS2 nav)."""
        return self.navigate_via_ros2(np.asarray(goal_xy_yaw, dtype=float))

    def stop_base(self):
        if self.require_official_control:
            self.official_control.command_base_stop()
        self.set_planar_motor_targets(0.0, 0.0, 0.0, max_delta=2.0)

    def _ramp_planar_stop(self, steps: int = 35) -> None:
        """Bring planar base motors down gradually to avoid a final jerk."""
        for _ in range(int(steps)):
            self.set_planar_motor_targets(0.0, 0.0, 0.0, max_delta=2.0)
            for joint_name in self.WHEEL_PITCH_JOINTS:
                self._set_velocity_for_joint(joint_name, 0.0)
            self.hold_arm_targets()
            self._hold_posture()
            self.env.step()

    def look_at(self, pan: float = 0.0, tilt: float = 0.0):
        self.stop_base()
        start_pan = self._qpos_for_joint("zhead_1_joint")
        start_tilt = self._qpos_for_joint("zhead_2_joint")
        target_pan = float(np.clip(pan, -1.2, 1.2))
        target_tilt = float(np.clip(tilt, -0.45, 0.45))
        for i in range(10):
            t = (i + 1) / 10.0
            s = 3.0 * t**2 - 2.0 * t**3  # smoothstep
            self._set_actuator_for_joint("zhead_1_joint", start_pan + s * (target_pan - start_pan))
            self._set_actuator_for_joint("zhead_2_joint", start_tilt + s * (target_tilt - start_tilt))
            self.env.step()

    def _arm_joints(self, side: str) -> list[str]:
        return self.LEFT_ARM_JOINTS if side == "left" else self.RIGHT_ARM_JOINTS

    @staticmethod
    def _arm_index(side: str) -> int:
        return 0 if side == "left" else 1

    def _standalone_ik_available(self, side: str) -> bool:
        try:
            self._get_kinematics_solver(side)
            self._standalone_ik_error = ""
            return True
        except Exception as exc:
            self._standalone_ik_error = str(exc)
            return False

    def _get_kinematics_solver(self, side: str) -> KuavoKinematicsSolver:
        key = "left" if side == "left" else "right"
        solver = self._kinematics_solvers.get(key)
        if solver is not None:
            return solver
        cfg = self._standalone_ik_config
        solver = KuavoKinematicsSolver(
            lib_path=cfg.get("lib_path"),
            urdf_path=cfg.get("urdf_path"),
            task_info_path=cfg.get("task_info_path"),
            arm_index=self._arm_index(key),
            is_whole_body=bool(cfg.get("is_whole_body", True)),
            linear_error_max=float(cfg.get("linear_error_max", 0.01)),
            angular_error_max=float(cfg.get("angular_error_max", 0.02)),
        )
        self._kinematics_solvers[key] = solver
        return solver

    def _current_whole_body_joint_positions(self) -> np.ndarray:
        return np.array([self._qpos_for_joint(j) for j in self.FULL_BODY_IK_JOINTS], dtype=float)

    def _current_ik_state(self, base_pose_override: np.ndarray | None = None) -> list[float]:
        base_pose = (
            np.asarray(base_pose_override, dtype=float)
            if base_pose_override is not None else self.get_base_pose().copy()
        )
        joint_positions = self._current_whole_body_joint_positions()
        return np.concatenate([base_pose[:3], joint_positions], axis=0).astype(float).tolist()

    def _default_target_pose_6d(
        self,
        target_xyz: np.ndarray,
        base_pose_override: np.ndarray | None = None,
    ) -> np.ndarray:
        base_pose = (
            np.asarray(base_pose_override, dtype=float)
            if base_pose_override is not None else self.get_base_pose().copy()
        )
        dx = float(target_xyz[0] - base_pose[0])
        dy = float(target_xyz[1] - base_pose[1])
        yaw = float(base_pose[2]) if abs(dx) + abs(dy) < 1e-9 else math.atan2(dy, dx)
        return np.array(
            [float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]), yaw, 0.0, -1.57],
            dtype=float,
        )

    def _normalize_target_pose_6d(
        self,
        target_pose_world: np.ndarray,
        base_pose_override: np.ndarray | None = None,
    ) -> np.ndarray:
        pose = np.asarray(target_pose_world, dtype=float).reshape(-1)
        if pose.size >= 6:
            return pose[:6].astype(float)
        if pose.size >= 3:
            return self._default_target_pose_6d(pose[:3], base_pose_override=base_pose_override)
        raise ValueError(f"target_pose_world must contain at least xyz, got shape={pose.shape}")

    def _solve_standalone_ik(
        self,
        side: str,
        target_pose_world: np.ndarray,
        base_pose_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float]:
        solver = self._get_kinematics_solver(side)
        target_6d = self._normalize_target_pose_6d(target_pose_world, base_pose_override=base_pose_override)
        current_q = self._current_ik_state(base_pose_override=base_pose_override)
        q_best = np.asarray(solver.solve_ik(target_6d.tolist(), current_q=current_q), dtype=float)
        if q_best.size != len(self.FULL_BODY_IK_JOINTS):
            raise RuntimeError(
                f"standalone IK returned {q_best.size} joints, expected {len(self.FULL_BODY_IK_JOINTS)}"
            )
        return q_best, float(solver.last_linear_error), float(solver.last_angular_error)

    def plan_arm_to(self, side: str, target_pose_world: np.ndarray) -> PlannedArmPath:
        """Plan a whole-body arm motion with the standalone Kuavo IK solver."""
        joints = list(self.FULL_BODY_IK_JOINTS)
        start = self._current_whole_body_joint_positions()
        normalized_target_6d = self._normalize_target_pose_6d(target_pose_world)
        solver = None
        try:
            solver = self._get_kinematics_solver(side)
            q_best, linear_error, angular_error = self._solve_standalone_ik(side, normalized_target_6d)
        except Exception as exc:
            self._standalone_ik_error = str(exc)
            if solver is None:
                try:
                    solver = self._get_kinematics_solver(side)
                except Exception:
                    solver = None
            if solver is not None:
                logger.warning(
                    "Kuavo standalone IK plan failed: "
                    f"side={side} raw_target={np.asarray(target_pose_world, dtype=float).round(5).tolist()} "
                    f"target_6d={normalized_target_6d.round(5).tolist()} "
                    f"qBest={np.asarray(getattr(solver, 'last_q', []), dtype=float).round(5).tolist()} "
                    f"linear_error={float(getattr(solver, 'last_linear_error', float('inf'))):.6f} "
                    f"angular_error={float(getattr(solver, 'last_angular_error', float('inf'))):.6f} "
                    f"seed={getattr(solver, 'last_seed_source', 'unknown')} "
                    f"status={int(getattr(solver, 'last_status', -1))}"
                )
            if not self._standalone_ik_warned:
                print(f"[KuavoStandaloneIK] planning failed: {exc}")
                self._standalone_ik_warned = True
            return PlannedArmPath(side, joints, np.empty((0, len(joints))), "kuavo_standalone_ik_failed")

        max_delta = float(np.max(np.abs(q_best - start))) if q_best.size else 0.0
        steps = int(np.clip(80 + 85 * max_delta, 90, 220))
        traj = TrajectoryGenerator.cubic_spline(start, q_best, steps)
        self._standalone_ik_error = ""
        logger.info(
            "Kuavo standalone IK plan: "
            f"side={side} raw_target={np.asarray(target_pose_world, dtype=float).round(5).tolist()} "
            f"target_6d={normalized_target_6d.round(5).tolist()} "
            f"qBest={np.asarray(q_best, dtype=float).round(5).tolist()} "
            f"linear_error={linear_error:.6f} angular_error={angular_error:.6f} "
            f"trajectory_shape={traj.shape}"
        )
        return PlannedArmPath(
            side,
            joints,
            traj,
            f"kuavo_standalone_ik_lin{linear_error:.4f}_ang{angular_error:.4f}",
        )

    def estimate_arm_reachability(
        self,
        side: str,
        target_pose_world: np.ndarray,
        base_pose_override: np.ndarray | None = None,
    ) -> dict:
        """Estimate reachability with the same standalone IK used for execution planning.

        For current base pose: uses official IK service.
        For scratch base poses: uses lightweight local Jacobian IK for
        planning-phase estimation only. This is NOT execution IK — official
        IK always owns the execution path.
        """
        joints = list(self.FULL_BODY_IK_JOINTS)
        try:
            _, linear_error, _ = self._solve_standalone_ik(
                side,
                target_pose_world,
                base_pose_override=base_pose_override,
            )
            ok = True
        except Exception as exc:
            self._standalone_ik_error = str(exc)
            ok = False
            linear_error = float("inf")
        return {
            "reachable": bool(ok),
            "source": "kuavo_standalone_ik_reachability" if ok else "kuavo_standalone_ik_unreachable",
            "ik_error": float(linear_error),
            "joint_names": joints,
        }

    def _estimate_reachability_scratch(
        self,
        side: str,
        target_pose_world: np.ndarray,
        base_pose_override: np.ndarray,
    ) -> dict:
        """Planning-phase reachability check for a hypothetical base pose.

        Uses a lightweight local Jacobian IK that does NOT depend on the
        official ROS1 service. Results are advisory only — execution always
        routes through official IK.
        """
        raise RuntimeError("legacy scratch reachability path removed; use standalone IK reachability instead")

    def _try_official_ik(self, side: str, target_pose_world: np.ndarray) -> PlannedArmPath | None:
        base = self.get_base_pose()
        # world → base-local
        return None

    def _jacobian_position_step(
        self,
        body_name: str,
        target_pos: np.ndarray,
        current_q: np.ndarray,
        active_joint_names: list[str],
    ) -> tuple[np.ndarray | None, float]:
        """Single Jacobian pseudo-inverse step toward target position."""
        return None, float("inf")

    def navigate_via_ros2(self, target_pose: np.ndarray, timeout_s: float = 30.0) -> bool:
        """Start ROS2 navigation by writing a goal to the bridge file.

        Does NOT block — the caller must run the simulation loop and call
        ``ros2_nav_step()`` each frame until the result is ready.

        Returns True if the goal was written successfully.
        """
        from planning.ros2_nav_executor import Nav2Executor

        nav = Nav2Executor()
        nav.write_goal(
            float(target_pose[0]),
            float(target_pose[1]),
            float(target_pose[2]),
        )
        self._ros2_nav_deadline = __import__('time').perf_counter() + timeout_s
        self._ros2_nav_active = True
        return True

    def ros2_nav_step(self) -> tuple[bool, float, float]:
        """Call each simulation frame during ROS2 navigation.

        Returns (done, error_xy, error_yaw).
        """
        from planning.ros2_nav_executor import Nav2Executor

        nav = Nav2Executor()
        result = nav.poll_result()
        if result is not None:
            self._ros2_nav_active = False
            return True, result.final_error_xy, result.final_error_yaw

        now = __import__('time').perf_counter()
        if now > getattr(self, '_ros2_nav_deadline', now + 30.0):
            self._ros2_nav_active = False
            return True, float("inf"), float("inf")

        return False, 0.0, 0.0

    def _planning_jacobian_ik(
        self,
        body_name: str,
        target_pos: np.ndarray,
        seed_q: np.ndarray,
        active_joint_names: list[str],
    ) -> tuple[np.ndarray | None, float]:
        """Lightweight MuJoCo Jacobian IK for planning-phase reachability estimation.

        WARNING: This is NOT execution IK. It is deliberately limited (position-only,
        no orientation, simplified seed) and used only to prune unreachable
        base-pose candidates during reachability search. Execution always goes
        through official Kuavo IK.
        """
        return None, float("inf")

    def _write_base_pose_to_qpos(self, qpos: np.ndarray, base_pose: np.ndarray) -> None:
        for joint_name, value in {
            "base_x_joint": float(base_pose[0]),
            "base_y_joint": float(base_pose[1]),
            "base_yaw_joint": float(base_pose[2]),
        }.items():
            jid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid >= 0:
                qpos[int(self.env.model.jnt_qposadr[jid])] = value

    def _estimate_terminal_error(
        self,
        body_name: str,
        joint_names: list[str],
        q_target: np.ndarray,
        target_pos: np.ndarray,
        base_pose_override: np.ndarray | None = None,
    ) -> float:
        body_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return float("inf")
        scratch = mujoco.MjData(self.env.model)
        scratch.qpos[:] = self.env.data.qpos
        if base_pose_override is not None:
            self._write_base_pose_to_qpos(scratch.qpos, np.asarray(base_pose_override, dtype=float))
        for joint_name, value in zip(joint_names, q_target):
            jid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid >= 0:
                scratch.qpos[int(self.env.model.jnt_qposadr[jid])] = float(value)
        mujoco.mj_forward(self.env.model, scratch)
        return float(np.linalg.norm(scratch.xpos[body_id].copy() - np.asarray(target_pos[:3], dtype=float)))

    def execute_arm_trajectory(self, side: str, trajectory: PlannedArmPath) -> bool:
        if trajectory.positions.size == 0:
            return False
        official_mirror = False
        if self.require_official_control:
            if not trajectory.source.startswith(("kuavo_official_", "kuavo_standalone_ik")):
                raise RuntimeError(
                    f"Full Kuavo execution requires an approved trajectory source, got {trajectory.source}"
                )
            q14 = self._to_official_arm_trajectory(side, trajectory)
            if not self.official_control.send_arm_joint_trajectory(q14, dt=self.env.dt):
                return False
            official_mirror = True
        controlled = set(trajectory.joint_names)
        base_reference = self.get_base_pose().copy()
        for q in trajectory.positions:
            self._hold_base_mirror(base_reference)
            for joint, value in zip(trajectory.joint_names, q):
                self._set_actuator_for_joint(joint, float(value))
            self._hold_posture(exclude=controlled | set(self.RIGHT_ARM_JOINTS))
            self.env.step()
        final = trajectory.positions[-1]
        self._held_arm_targets[side] = (
            list(trajectory.joint_names),
            np.asarray(final, dtype=float).copy(),
        )
        for _ in range(int(self.arm_settle_steps)):
            self._hold_base_mirror(base_reference)
            for joint, value in zip(trajectory.joint_names, final):
                self._set_actuator_for_joint(joint, float(value))
            self._hold_posture(exclude=controlled | set(self.RIGHT_ARM_JOINTS))
            self.env.step()
        if official_mirror:
            # The official SDK owns the real command. These MuJoCo steps mirror
            # the accepted trajectory for visualization and contact validation.
            pass
        return True

    def _to_official_arm_trajectory(self, side: str, trajectory: PlannedArmPath) -> np.ndarray:
        """Convert a local trajectory into Kuavo official 14-DoF arm order."""
        q = np.asarray(trajectory.positions, dtype=float)
        if q.ndim != 2:
            raise ValueError(f"invalid trajectory shape {q.shape}")
        if q.shape[1] == 18:
            return q[:, 4:18].copy()
        if q.shape[1] == 14:
            return q
        if q.shape[1] != 7:
            raise ValueError(f"expected 7, 14, or 18 arm joints, got {q.shape[1]}")
        current_left = np.array([self._qpos_for_joint(j) for j in self.LEFT_ARM_JOINTS], dtype=float)
        current_right = np.array([self._qpos_for_joint(j) for j in self.RIGHT_ARM_JOINTS], dtype=float)
        out = np.tile(np.concatenate([current_left, current_right]), (q.shape[0], 1))
        if side == "left":
            out[:, :7] = q
        else:
            out[:, 7:14] = q
        return out

    def move_arm(
        self,
        lift: float | None = None,
        extend: float | None = None,
        wrist_yaw: float | None = None,
        grip: float | None = None,
        preserve_base: bool = False,
    ):
        """Compatibility method used by the Stretch-oriented pipeline."""
        if not preserve_base:
            self.stop_base()
        if wrist_yaw is not None:
            self._set_actuator_for_joint("zarm_r5_joint", float(np.clip(wrist_yaw, -1.2, 1.2)))
        if lift is not None:
            self._set_actuator_for_joint("zarm_r1_joint", float(np.clip(-0.7 + lift, -1.2, 0.8)))
            self._set_actuator_for_joint("zarm_r4_joint", float(np.clip(-1.4 + lift, -2.2, -0.2)))
        if extend is not None:
            self._set_actuator_for_joint("zarm_r2_joint", float(np.clip(-0.4 + extend, -1.4, 0.3)))
            self._set_actuator_for_joint("zarm_r7_joint", float(np.clip(0.15 - extend, -0.5, 0.5)))
        if grip is not None:
            if grip <= -0.001:
                self.close_gripper()
            else:
                self.open_gripper()
        self._hold_posture(exclude={"waist_pitch_joint", "waist_yaw_joint"})

    def open_gripper(self, side: str = "right"):
        if self.require_official_control and not self.official_control.open_claw(side):
            raise RuntimeError("official Kuavo claw open command failed")
        base_reference = self.get_base_pose().copy()
        current = self._named_ctrl_value("r_fingers_actuator")
        for cmd in np.linspace(current, self._right_grip_open, 60):
            self._set_named_actuator("r_fingers_actuator", float(cmd))
            self.hold_arm_targets(side)
            self._hold_base_mirror(base_reference)
            self._hold_posture()
            self.env.step()

    def hold_gripper_closed(self, side: str = "right") -> None:
        """Maintain the pinch command without advancing simulation internally."""
        current = self._named_ctrl_value("r_fingers_actuator")
        delta = np.clip(
            self._right_grip_closed - current,
            -self.max_gripper_ctrl_delta,
            self.max_gripper_ctrl_delta,
        )
        self._set_named_actuator("r_fingers_actuator", float(current + delta))

    def close_gripper(self, side: str = "right"):
        if self.require_official_control and not self.official_control.close_claw(side):
            raise RuntimeError("official Kuavo claw close command failed")
        base_reference = self.get_base_pose().copy()
        current = self._named_ctrl_value("r_fingers_actuator")
        for cmd in np.linspace(current, self._right_grip_closed, 70):
            self._set_named_actuator("r_fingers_actuator", float(cmd))
            self.hold_arm_targets(side)
            self._hold_base_mirror(base_reference)
            self._hold_posture()
            self.env.step()

    def verify_grasp_contact(self, object_id: str) -> bool:
        return False

    def carry_to(self, target_pose_world: np.ndarray) -> bool:
        target = np.asarray(target_pose_world, dtype=float)
        planned = self.plan_arm_to("right", target[:3].copy())
        if planned.positions.size == 0:
            return False
        return self.execute_arm_trajectory("right", planned)

    def release(self):
        self.open_gripper()
