"""
Mink IK Solver — a wrapper around the Mink library for MuJoCo-native IK.

Mink (https://github.com/kevinzakka/mink) is a mature MuJoCo-native
differential IK library that uses QP-based task prioritization. It supports
single-arm, dual-arm, humanoid, and collision avoidance out of the box.

IMPORTANT: solve_ik() returns the FULL qpos array (data.qpos format).
Callers must execute the returned target through controllers/actuators.

Install: pip install mink
Requires: mujoco >= 3.1.0
"""

from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field

import numpy as np
import mujoco

from mink import (
    Configuration,
    DofFreezingTask,
    FrameTask,
    PostureTask,
    ConfigurationLimit,
    solve_ik,
    SE3,
    SO3,
)
from robot_common.infra.logging import logger


@dataclass
class IKSolution:
    """Result of inverse kinematics computation.

    joint_positions: The FULL qpos array (data.qpos format).
        Extract actuated joints from this target and track them with robot
        controllers. Do not assign it directly to live data.qpos during task
        execution.
    """
    success: bool = False
    joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    position_error: float = 0.0
    iterations: int = 0


def _init_qpos_from_midrange(model: mujoco.MjModel) -> np.ndarray:
    """Create a valid initial qpos from joint mid-range values."""
    qpos = np.zeros(model.nq)
    for jid in range(model.njnt):
        jnt_type = model.jnt_type[jid]
        qadr = model.jnt_qposadr[jid]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            qpos[qadr + 3] = 1  # identity quaternion w=1
        elif jnt_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            lo, hi = model.jnt_range[jid]
            qpos[qadr] = (lo + hi) / 2.0
    return qpos


def _clip_qpos_to_joint_limits(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Return a scratch qpos clipped to MuJoCo joint limits."""
    clipped = np.asarray(qpos, dtype=float).copy()
    for jid in range(model.njnt):
        jnt_type = int(model.jnt_type[jid])
        if jnt_type not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            continue
        if not bool(model.jnt_limited[jid]):
            continue
        qadr = int(model.jnt_qposadr[jid])
        lo, hi = model.jnt_range[jid]
        clipped[qadr] = float(np.clip(clipped[qadr], lo, hi))
    return clipped


class MinkIKSolver:
    """Inverse kinematics solver using the Mink library.

    Usage:
        solver = MinkIKSolver(model, data)
        sol = solver.solve_ik(target_pos=np.array([0.5, 0, 0.8]),
                              ee_body_name="l_gripper")
        if sol.success:
            command_actuators_toward(sol.joint_positions)
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

    def solve_ik(
        self,
        target_pos: np.ndarray,
        ee_body_name: str,
        target_quat: Optional[np.ndarray] = None,
        current_q: Optional[np.ndarray] = None,
        position_cost: float = 1.0,
        orientation_cost: float = 0.5,
        posture_cost: float = 1e-3,
        dt: float = 0.01,
        max_steps: int = 200,
        tolerance: float = 1e-3,
        active_joint_names: Optional[List[str]] = None,
    ) -> IKSolution:
        """Solve IK for a single end-effector to reach a target pose.

        Returns IKSolution with full qpos in joint_positions.
        """
        scratch_data = mujoco.MjData(self.model)
        try:
            config = Configuration(self.model)

            if current_q is not None and len(current_q) == self.model.nq:
                config.update(_clip_qpos_to_joint_limits(self.model, current_q))
            elif np.all(self.data.qpos == 0):
                config.update(_init_qpos_from_midrange(self.model))
            else:
                config.update(_clip_qpos_to_joint_limits(self.model, self.data.qpos))

            tasks = []
            ee_task = FrameTask(
                frame_name=ee_body_name,
                frame_type="body",
                position_cost=position_cost,
                orientation_cost=orientation_cost,
            )

            if target_quat is not None:
                target_pose = SE3.from_rotation_and_translation(
                    SO3(w=float(target_quat[0]), x=float(target_quat[1]),
                        y=float(target_quat[2]), z=float(target_quat[3])),
                    target_pos.copy(),
                )
            else:
                target_pose = SE3.from_translation(target_pos.copy())

            ee_task.set_target(target_pose)
            tasks.append(ee_task)

            posture = PostureTask(self.model, cost=posture_cost)
            posture.set_target_from_configuration(config)
            tasks.append(posture)

            limits = [ConfigurationLimit(self.model)]
            constraints = None
            if active_joint_names is not None:
                active_dofs = set()
                for joint_name in active_joint_names:
                    jid = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if jid < 0:
                        continue
                    dof_adr = int(self.model.jnt_dofadr[jid])
                    jnt_type = int(self.model.jnt_type[jid])
                    width = 6 if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE) else 1
                    active_dofs.update(range(dof_adr, dof_adr + width))
                frozen_dofs = [
                    dof for dof in range(self.model.nv)
                    if dof not in active_dofs
                ]
                if frozen_dofs:
                    constraints = [DofFreezingTask(self.model, frozen_dofs)]

            best_error = float('inf')
            best_q = config.q.copy()

            for step in range(max_steps):
                vel = solve_ik(
                    config, tasks, dt, solver="quadprog",
                    limits=limits, constraints=constraints,
                )
                config.integrate_inplace(vel, dt)

                scratch_data.qpos[:] = config.q
                mujoco.mj_forward(self.model, scratch_data)
                ee_pos = scratch_data.body(ee_body_name).xpos.copy()
                error = np.linalg.norm(ee_pos - target_pos)
                if error < best_error:
                    best_error = error
                    best_q = config.q.copy()
                if error < tolerance:
                    break

            return IKSolution(
                success=best_error < 0.05,
                joint_positions=best_q,
                position_error=float(best_error),
                iterations=step + 1,
            )

        except Exception as e:
            logger.error(f"Mink IK failed for '{ee_body_name}': {e}")
            import traceback
            traceback.print_exc()
            return IKSolution(success=False, position_error=float('inf'))

    def solve_dual_ik(
        self,
        targets: List[Dict],
        current_q: Optional[np.ndarray] = None,
        position_cost: float = 1.0,
        orientation_cost: float = 0.5,
        posture_cost: float = 1e-3,
        dt: float = 0.01,
        max_steps: int = 200,
        tolerance: float = 1e-3,
        active_joint_names: Optional[List[str]] = None,
    ) -> IKSolution:
        """Solve IK for dual-arm with multiple end-effector targets.

        Args:
            targets: List of dicts with 'pos' (np.ndarray), 'body' (str),
                    and optional 'quat' (np.ndarray, wxyz).
        """
        scratch_data = mujoco.MjData(self.model)
        try:
            config = Configuration(self.model)

            if current_q is not None and len(current_q) == self.model.nq:
                config.update(_clip_qpos_to_joint_limits(self.model, current_q))
            elif np.all(self.data.qpos == 0):
                config.update(_init_qpos_from_midrange(self.model))
            else:
                config.update(_clip_qpos_to_joint_limits(self.model, self.data.qpos))

            tasks = []
            for target in targets:
                ee_body = target["body"]
                pos = np.asarray(target["pos"])
                quat = target.get("quat", None)

                ee_task = FrameTask(
                    frame_name=ee_body,
                    frame_type="body",
                    position_cost=position_cost,
                    orientation_cost=orientation_cost,
                )

                if quat is not None:
                    target_pose = SE3.from_rotation_and_translation(
                        SO3(w=float(quat[0]), x=float(quat[1]),
                            y=float(quat[2]), z=float(quat[3])),
                        pos.copy(),
                    )
                else:
                    target_pose = SE3.from_translation(pos.copy())

                ee_task.set_target(target_pose)
                tasks.append(ee_task)

            posture = PostureTask(self.model, cost=posture_cost)
            posture.set_target_from_configuration(config)
            tasks.append(posture)

            limits = [ConfigurationLimit(self.model)]
            constraints = None
            if active_joint_names is not None:
                active_dofs = set()
                for joint_name in active_joint_names:
                    jid = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if jid < 0:
                        continue
                    dof_adr = int(self.model.jnt_dofadr[jid])
                    jnt_type = int(self.model.jnt_type[jid])
                    width = 6 if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE) else 1
                    active_dofs.update(range(dof_adr, dof_adr + width))
                frozen_dofs = [
                    dof for dof in range(self.model.nv)
                    if dof not in active_dofs
                ]
                if frozen_dofs:
                    constraints = [DofFreezingTask(self.model, frozen_dofs)]

            best_error = float('inf')
            best_q = config.q.copy()

            for step in range(max_steps):
                vel = solve_ik(
                    config, tasks, dt, solver="quadprog",
                    limits=limits, constraints=constraints,
                )
                config.integrate_inplace(vel, dt)

                scratch_data.qpos[:] = config.q
                mujoco.mj_forward(self.model, scratch_data)
                max_err = 0.0
                for target in targets:
                    ee_body = target["body"]
                    pos = np.asarray(target["pos"])
                    ee_pos = scratch_data.body(ee_body).xpos.copy()
                    err = np.linalg.norm(ee_pos - pos)
                    max_err = max(max_err, err)

                if max_err < best_error:
                    best_error = max_err
                    best_q = config.q.copy()
                if max_err < tolerance:
                    break

            return IKSolution(
                success=best_error < 0.05,
                joint_positions=best_q,
                position_error=float(best_error),
                iterations=step + 1,
            )

        except Exception as e:
            logger.error(f"Mink dual IK failed: {e}")
            import traceback
            traceback.print_exc()
            return IKSolution(success=False, position_error=float('inf'))

    def qpos_for_joints(self, qpos: np.ndarray, joint_names: List[str]) -> np.ndarray:
        """Extract positions for specific joints from a full qpos array."""
        result = np.zeros(len(joint_names))
        for i, name in enumerate(joint_names):
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                qadr = self.model.jnt_qposadr[jid]
                value = qpos[qadr]
                if self.model.jnt_limited[jid]:
                    lo, hi = self.model.jnt_range[jid]
                    value = np.clip(value, lo, hi)
                result[i] = value
            except Exception:
                result[i] = 0.0
        return result

    def get_ee_pose(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get current end-effector position and orientation."""
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.body(body_name).xpos.copy()
        quat = self.data.body(body_name).xquat.copy()
        return pos, quat
