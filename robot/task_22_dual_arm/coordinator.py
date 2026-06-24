"""DualArmCoordinator — synchronized dual-arm kinematics, IK, and collision checks."""

from typing import List, Tuple

import numpy as np
import mujoco

from control import MuJoCoEnv
from robot_common.infra.logging import logger
from planning import (
    MinkIKSolver, TrajectoryGenerator, CollisionMonitor, IKSolution,
)
from planning.ik import _init_qpos_from_midrange

from .state import DualArmState


class DualArmCoordinator:
    """Coordinates dual-arm movements with synchronization constraints.

    Key features:
      - Synchronized trajectory execution
      - Pose maintenance for carried objects (tilt < 5°)
      - Self-collision avoidance between arms
      - Master-slave and sync role management
    """

    def __init__(self, env: MuJoCoEnv,
                 left_joints: List[str], right_joints: List[str],
                 arm_base_distance: float = 0.8,
                 max_sync_error: float = 0.02,
                 max_tilt_angle: float = 5.0):
        self.env = env
        self.left_joints = left_joints
        self.right_joints = right_joints
        self.arm_base_distance = arm_base_distance
        self.max_sync_error = max_sync_error
        self.max_tilt_angle = np.deg2rad(max_tilt_angle)
        self.nominal_gripper_separation = 0.1
        self.max_ctrl_delta = 0.25

        # IK solver — Mink handles dual-arm natively
        self.ik = MinkIKSolver(env.model, env.data)

        # Collision monitor
        self.collision_monitor = CollisionMonitor(env)
        self.wbc_controller = None

    def set_mature_backends(self, wbc_controller=None):
        """Attach mature planning/control adapters without changing the DAG API."""
        self.wbc_controller = wbc_controller

    def get_state(self) -> DualArmState:
        """Get current dual-arm state."""
        return DualArmState(
            left_joints=self._get_joint_positions("left"),
            right_joints=self._get_joint_positions("right"),
            left_ee_pos=self._get_ee_position("left"),
            left_ee_quat=self._get_ee_rotation("left"),
            right_ee_pos=self._get_ee_position("right"),
            right_ee_quat=self._get_ee_rotation("right"),
            nominal_separation=self.nominal_gripper_separation,
        )

    def _get_joint_positions(self, side: str) -> np.ndarray:
        joints = self.left_joints if side == "left" else self.right_joints
        pos = []
        for j in joints:
            jid = self.env.joint_index(j)
            if jid is not None:
                pos.append(float(self.env.data.qpos[jid]))
            else:
                pos.append(0.0)
        return np.array(pos)

    def _get_ee_position(self, side: str) -> np.ndarray:
        short = "l" if side == "left" else "r"
        candidates = [
            f"{short}_pinch",
            f"{short}_gripper",
            f"{short}_finger_l",
            f"{side}_hand",
            f"{side}_gripper",
        ]
        ee_name = next((name for name in candidates if name in self.env._body_names), "")
        try:
            return self.env.get_body_position(ee_name)
        except Exception:
            # Try site
            try:
                return self.env.get_site_position(f"{short}_grip_site")
            except Exception:
                joints = self.left_joints if side == "left" else self.right_joints
                if joints:
                    return self.env.get_body_position(f"{short}_link7")
                return np.zeros(3)

    def _get_ee_rotation(self, side: str) -> np.ndarray:
        short = "l" if side == "left" else "r"
        candidates = [
            f"{short}_pinch",
            f"{short}_gripper",
            f"{short}_finger_l",
            f"{side}_hand",
            f"{side}_gripper",
        ]
        ee_name = next((name for name in candidates if name in self.env._body_names), "")
        try:
            return self.env.get_body_quat(ee_name)
        except Exception:
            return np.array([1.0, 0.0, 0.0, 0.0])

    def plan_synchronized_trajectory(self,
                                      left_target: np.ndarray,
                                      right_target: np.ndarray,
                                      num_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Plan synchronized dual-arm trajectory.

        Args:
            left_target: Target EE pose for left arm [x, y, z, roll, pitch, yaw, grip]
            right_target: Target EE pose for right arm
            num_steps: Number of trajectory steps

        Returns:
            (left_trajectory, right_trajectory) each (num_steps, n_joints)
        """
        state = self.get_state()

        wbc_command = None
        if (
            self.wbc_controller is not None
            and getattr(self.wbc_controller, "available", False)
            and hasattr(self.wbc_controller, "make_dual_arm_command")
        ):
            wbc_command = self.wbc_controller.make_dual_arm_command(
                left_target[:3],
                right_target[:3],
                state.left_ee_pos,
                state.right_ee_pos,
            )
            logger.info(f"Dual-arm command source: {wbc_command.source}")

        # Solve IK for both arms using Mink (dual-arm aware). In full mode the
        # preceding WBC command must be available; Mink maps its target to the
        # current MuJoCo scene for execution.
        sol = self.ik.solve_dual_ik(
            targets=[
                {'pos': left_target[:3], 'body': self._ee_body("left")},
                {'pos': right_target[:3], 'body': self._ee_body("right")},
            ],
            orientation_cost=0.05,
            posture_cost=1e-4,
            max_steps=450,
        )

        if not sol.success:
            logger.warning(f"Dual-arm IK failed: error={sol.position_error:.4f}m")

        # Extract per-arm joint positions from the full qpos solution
        left_target_joints = self.ik.qpos_for_joints(sol.joint_positions, self.left_joints)
        right_target_joints = self.ik.qpos_for_joints(sol.joint_positions, self.right_joints)

        # Generate synchronized trajectory (same number of steps for both arms)
        max_step = 0.12
        joint_delta = max(
            float(np.max(np.abs(left_target_joints - state.left_joints))) if len(left_target_joints) else 0.0,
            float(np.max(np.abs(right_target_joints - state.right_joints))) if len(right_target_joints) else 0.0,
        )
        num_steps = max(num_steps, int(np.ceil(joint_delta / max_step)) + 1)
        left_traj = TrajectoryGenerator.cubic_spline(
            state.left_joints, left_target_joints, num_steps
        )
        right_traj = TrajectoryGenerator.cubic_spline(
            state.right_joints, right_target_joints, num_steps
        )
        self.last_plan_debug = {
            "source": "mink_cubic_time_parameterized",
            "ik_success": bool(sol.success),
            "ik_error_m": float(sol.position_error),
            "steps": int(num_steps),
            "left_target_world": left_target[:3].copy(),
            "right_target_world": right_target[:3].copy(),
            "left_joint_delta_max": float(np.max(np.abs(left_target_joints - state.left_joints))) if len(left_target_joints) else 0.0,
            "right_joint_delta_max": float(np.max(np.abs(right_target_joints - state.right_joints))) if len(right_target_joints) else 0.0,
        }

        return left_traj, right_traj

    def _ee_body(self, side: str) -> str:
        short = "l" if side == "left" else "r"
        for name in (f"{short}_pinch", f"{short}_gripper"):
            if name in self.env._body_names:
                return name
        return f"{short}_gripper"

    def _set_joints(self, side: str, positions: np.ndarray):
        """Set arm joint position targets with bounded command deltas."""
        joints = self.left_joints if side == "left" else self.right_joints
        # Map joint names to motor actuator indices
        actuator_map = self._get_actuator_map()
        for i, j_name in enumerate(joints):
            if i < len(positions) and j_name in actuator_map:
                act_id = actuator_map[j_name]
                jid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
                qadr = self.env.model.jnt_qposadr[jid]
                desired = float(positions[i])
                if self.env.model.jnt_limited[jid]:
                    lo, hi = self.env.model.jnt_range[jid]
                    desired = float(np.clip(desired, lo, hi))
                current = float(self.env.data.ctrl[act_id])
                self.env.data.ctrl[act_id] = current + np.clip(
                    desired - current, -self.max_ctrl_delta, self.max_ctrl_delta
                )

    def _get_actuator_map(self) -> dict:
        """Build a mapping from joint name → actuator index."""
        if not hasattr(self, '_actuator_map_cache'):
            self._actuator_map_cache = {}
            for a_id in range(self.env.model.nu):
                name = self.env.model.actuator(a_id).name
                if name:
                    # Motor actuators have their target joint accessible
                    trnid = self.env.model.actuator_trnid[a_id]
                    jnt_id = trnid[0]  # transmission joint id
                    jnt_name = self.env.model.joint(jnt_id).name
                    if jnt_name:
                        self._actuator_map_cache[jnt_name] = a_id
        return self._actuator_map_cache

    def object_tilt(self, body_name: str, reference_quat: np.ndarray) -> float:
        """Return carried-object tilt angle in radians from its initial pose."""
        if body_name not in self.env._body_names:
            return 0.0
        current = self.env.get_body_quat(body_name)
        if body_name == "long_rod":
            ref_mat = np.zeros(9)
            cur_mat = np.zeros(9)
            mujoco.mju_quat2Mat(ref_mat, np.asarray(reference_quat, dtype=float))
            mujoco.mju_quat2Mat(cur_mat, np.asarray(current, dtype=float))
            ref_axis = ref_mat.reshape(3, 3)[:, 1]
            cur_axis = cur_mat.reshape(3, 3)[:, 1]
            dot = abs(float(np.dot(ref_axis, cur_axis)))
            return float(np.arccos(np.clip(dot, -1.0, 1.0)))
        dot = abs(float(np.dot(current, reference_quat)))
        return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))

    def is_safe(self) -> bool:
        """Check if current configuration is collision-free."""
        return self.collision_monitor.is_safe()
