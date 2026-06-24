"""
Torque-level dual-arm whole-body controller for the MuJoCo dual Panda scene.

The controller solves a damped operational-space least-squares QP each control
cycle and writes torque commands directly to arm motor actuators. Position
servos are intentionally not used for the 14 arm joints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import mujoco
import numpy as np


WBC_SOURCE = "dual_arm_wbc_qp_torque"


@dataclass
class WBCStepInfo:
    source: str
    sync_error: float
    tilt_error: float
    left_error: float
    right_error: float
    torque_norm: float
    saturated: bool
    metadata: dict = field(default_factory=dict)


class DualArmWBCController:
    """Dual end-effector WBC with a hard left-right relative task."""

    def __init__(
        self,
        env,
        left_joints: Iterable[str],
        right_joints: Iterable[str],
        left_body: str = "l_gripper",
        right_body: str = "r_gripper",
        object_body: str = "long_rod",
        kp_pos: float = 120.0,
        kd_pos: float = 24.0,
        kp_rel: float = 260.0,
        kd_rel: float = 38.0,
        kp_ori: float = 60.0,
        kd_ori: float = 14.0,
        damping: float = 2e-4,
        joint_damping: float = 2.0,
        posture_gain: float = 1.5,
        torque_limit: float = 80.0,
        max_torque_delta: float = 3.0,
    ):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.left_joints = list(left_joints)
        self.right_joints = list(right_joints)
        self.arm_joints = self.left_joints + self.right_joints
        self.left_body = left_body
        self.right_body = right_body
        self.object_body = object_body
        self.kp_pos = float(kp_pos)
        self.kd_pos = float(kd_pos)
        self.kp_rel = float(kp_rel)
        self.kd_rel = float(kd_rel)
        self.kp_ori = float(kp_ori)
        self.kd_ori = float(kd_ori)
        self.damping = float(damping)
        self.joint_damping = float(joint_damping)
        self.posture_gain = float(posture_gain)
        self.torque_limit = float(torque_limit)
        self.max_torque_delta = float(max_torque_delta)
        self.left_body_id = self._body_id(left_body)
        self.right_body_id = self._body_id(right_body)
        self.object_body_id = self._body_id(object_body, required=False)
        self.dof_ids = [self._joint_dof_id(name) for name in self.arm_joints]
        self.qpos_ids = [self._joint_qpos_id(name) for name in self.arm_joints]
        self.motor_ids = [self._motor_actuator_for_joint(name) for name in self.arm_joints]
        self.home_q = self.data.qpos[self.qpos_ids].copy()
        self._last_tau = np.zeros(len(self.arm_joints), dtype=float)
        self.available = self._check_torque_actuators()

    def _body_id(self, name: str, required: bool = True) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0 and required:
            raise RuntimeError(f"WBC body missing: {name}")
        return int(bid)

    def _joint_dof_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"WBC joint missing: {name}")
        return int(self.model.jnt_dofadr[jid])

    def _joint_qpos_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"WBC joint missing: {name}")
        return int(self.model.jnt_qposadr[jid])

    def _motor_actuator_for_joint(self, joint_name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        for act_id in range(self.model.nu):
            if int(self.model.actuator_trnid[act_id, 0]) == jid:
                return int(act_id)
        raise RuntimeError(f"WBC motor actuator missing for joint: {joint_name}")

    def _check_torque_actuators(self) -> bool:
        for act_id in self.motor_ids:
            # Position actuators have affine bias/gain parameters. Motor
            # actuators in this scene use fixed gain and no affine bias.
            bias_type = int(self.model.actuator_biastype[act_id])
            if bias_type != int(mujoco.mjtBias.mjBIAS_NONE):
                return False
        return True

    def validate(self) -> bool:
        return bool(self.available and len(self.motor_ids) == 14)

    def reset(self):
        self._last_tau = np.zeros(len(self.arm_joints), dtype=float)
        self.home_q = self.data.qpos[self.qpos_ids].copy()

    def current_left_pos(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[self.left_body_id].copy()

    def current_right_pos(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[self.right_body_id].copy()

    def step(
        self,
        left_target_world: np.ndarray,
        right_target_world: np.ndarray,
        relative_target_world: Optional[np.ndarray] = None,
        object_quat_target: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        object_body_name: Optional[str] = None,
    ) -> WBCStepInfo:
        if not self.available:
            raise RuntimeError("DualArmWBCController requires torque motor actuators")

        mujoco.mj_forward(self.model, self.data)
        left_target = np.asarray(left_target_world[:3], dtype=float)
        right_target = np.asarray(right_target_world[:3], dtype=float)
        x_l = self.data.xpos[self.left_body_id].copy()
        x_r = self.data.xpos[self.right_body_id].copy()
        rel_target = (
            np.asarray(relative_target_world[:3], dtype=float)
            if relative_target_world is not None else left_target - right_target
        )
        qd = self.data.qvel[self.dof_ids].copy()
        q = self.data.qpos[self.qpos_ids].copy()

        jacp_l, jacr_l = self._body_jacobians(self.left_body_id)
        jacp_r, jacr_r = self._body_jacobians(self.right_body_id)
        j_l = jacp_l[:, self.dof_ids]
        j_r = jacp_r[:, self.dof_ids]
        jr_mid = 0.5 * (jacr_l[:, self.dof_ids] + jacr_r[:, self.dof_ids])
        j_rel = j_l - j_r

        e_l = left_target - x_l
        e_r = right_target - x_r
        e_rel = rel_target - (x_l - x_r)
        v_l = j_l @ qd
        v_r = j_r @ qd
        v_rel = j_rel @ qd

        task_accels = [
            self.kp_pos * e_l - self.kd_pos * v_l,
            self.kp_pos * e_r - self.kd_pos * v_r,
            self.kp_rel * e_rel - self.kd_rel * v_rel,
        ]
        task_jacs = [j_l, j_r, j_rel]
        task_weights = [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 120.0, 120.0, 120.0]
        ori_error = self._orientation_error_vector(object_quat_target, object_body_name)
        if ori_error is not None:
            omega_mid = jr_mid @ qd
            task_accels.append(self.kp_ori * ori_error - self.kd_ori * omega_mid)
            task_jacs.append(jr_mid)
            task_weights.extend([25.0, 25.0, 25.0])
        xdd = np.concatenate(task_accels)
        j_task = np.vstack(task_jacs)
        weights = np.diag(task_weights)
        lhs = j_task.T @ weights @ j_task + self.damping * np.eye(len(self.dof_ids))
        rhs = j_task.T @ weights @ xdd

        qdd_task = self._solve_qp_lstsq(lhs, rhs)
        posture = -self.posture_gain * (q - self.home_q) - 0.25 * qd
        null = np.eye(len(self.dof_ids)) - np.linalg.pinv(j_task, rcond=1e-3) @ j_task
        qdd = qdd_task + null @ posture

        mass = np.zeros((self.model.nv, self.model.nv), dtype=float)
        mujoco.mj_fullM(self.model, mass, self.data.qM)
        m_arm = mass[np.ix_(self.dof_ids, self.dof_ids)]
        bias = self.data.qfrc_bias[self.dof_ids].copy()
        tau = m_arm @ qdd + bias - self.joint_damping * qd
        tau = np.asarray(tau, dtype=float)
        saturated = bool(np.any(np.abs(tau) > self.torque_limit))
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        if self.max_torque_delta > 0:
            tau = self._last_tau + np.clip(
                tau - self._last_tau,
                -self.max_torque_delta,
                self.max_torque_delta,
            )
        self._last_tau = tau.copy()

        ctrl = self.data.ctrl.copy()
        for act_id, value in zip(self.motor_ids, tau):
            lo, hi = self.model.actuator_ctrlrange[act_id]
            if self.model.actuator_ctrllimited[act_id]:
                value = float(np.clip(value, lo, hi))
            ctrl[act_id] = value
        self.env.set_control(ctrl)

        tilt_error = self._tilt_error(object_quat_target, object_body_name)
        return WBCStepInfo(
            source=WBC_SOURCE,
            sync_error=float(np.linalg.norm(e_rel)),
            tilt_error=float(tilt_error),
            left_error=float(np.linalg.norm(e_l)),
            right_error=float(np.linalg.norm(e_r)),
            torque_norm=float(np.linalg.norm(tau)),
            saturated=saturated,
            metadata={
                "dt": float(dt if dt is not None else self.env.dt),
                "left_target_world": left_target.copy(),
                "right_target_world": right_target.copy(),
                "relative_target_world": rel_target.copy(),
            },
        )

    def _body_jacobians(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        return jacp, jacr

    def _solve_qp_lstsq(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(lhs, rhs, rcond=1e-6)[0]

    def _tilt_error(self, reference_quat: Optional[np.ndarray],
                    object_body_name: Optional[str]) -> float:
        if reference_quat is None:
            return 0.0
        body_id = self.object_body_id
        if object_body_name:
            body_id = self._body_id(object_body_name, required=False)
        if body_id is None or body_id < 0:
            return 0.0
        current = self.data.xquat[body_id].copy()
        ref = np.asarray(reference_quat, dtype=float)
        if ref.shape[0] != 4:
            return 0.0
        ref_mat = np.zeros(9)
        cur_mat = np.zeros(9)
        mujoco.mju_quat2Mat(ref_mat, ref)
        mujoco.mju_quat2Mat(cur_mat, current)
        ref_axis = ref_mat.reshape(3, 3)[:, 1]
        cur_axis = cur_mat.reshape(3, 3)[:, 1]
        dot = abs(float(np.dot(ref_axis, cur_axis)))
        return float(np.arccos(np.clip(dot, -1.0, 1.0)))

    def _orientation_error_vector(self, reference_quat: Optional[np.ndarray],
                                  object_body_name: Optional[str]) -> Optional[np.ndarray]:
        if reference_quat is None:
            return None
        body_id = self.object_body_id
        if object_body_name:
            body_id = self._body_id(object_body_name, required=False)
        if body_id is None or body_id < 0:
            return None
        ref = np.asarray(reference_quat, dtype=float)
        if ref.shape[0] != 4:
            return None
        ref_mat = np.zeros(9)
        mujoco.mju_quat2Mat(ref_mat, ref)
        cur_mat = self.data.xmat[body_id].copy()
        r_ref = ref_mat.reshape(3, 3)
        r_cur = cur_mat.reshape(3, 3)
        return 0.5 * (
            np.cross(r_cur[:, 0], r_ref[:, 0])
            + np.cross(r_cur[:, 1], r_ref[:, 1])
            + np.cross(r_cur[:, 2], r_ref[:, 2])
        )
