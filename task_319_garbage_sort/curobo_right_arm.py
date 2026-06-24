"""cuRobo helpers for Task 319 Kuavo right-arm grasp motion.

This module intentionally exposes only the Kuavo right arm.  The mobile base,
legs, torso, waist, head, left arm, and gripper fingers are locked in cuRobo's
robot model; the active cspace is zarm_r1_joint..zarm_r7_joint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import torch

from task_319_garbage_sort.gripper_robot_urdf import ensure_kuavo_with_gripper_urdf


RIGHT_ARM_JOINT_NAMES = [
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
    "zarm_r5_joint",
    "zarm_r6_joint",
    "zarm_r7_joint",
]

RIGHT_ARM_LOCK_JOINTS = {
    "knee_joint": 0.0,
    "leg_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_finger_joint": 0.030,
    "right_finger_joint": 0.030,
}

RIGHT_ARM_CARRY_CONFIG = [-0.95, 0.0, 0.0, -1.15, 0.0, 0.0, 0.15]

RIGHT_ARM_COLLISION_LINKS = [
    "zarm_r1_link",
    "zarm_r2_link",
    "zarm_r3_link",
    "zarm_r4_link",
    "zarm_r5_link",
    "zarm_r6_link",
    "zarm_r7_link",
    "gripper_base",
    "left_finger",
    "right_finger",
]


def _read_joint_limits_from_urdf(urdf_path: Path, joint_names: list[str]) -> dict[str, tuple[float, float]]:
    root = ET.parse(urdf_path).getroot()
    wanted = set(joint_names)
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in wanted:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        try:
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
        except Exception:
            continue
        if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
            limits[name] = (lower, upper)
    return limits


def clamp_right_arm_start_q_to_limits(
    q: np.ndarray,
    limits: dict[str, tuple[float, float]],
    *,
    margin_rad: float = 1.0e-4,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(q, dtype=np.float32).reshape(len(RIGHT_ARM_JOINT_NAMES))
    clamped = raw.copy()
    records: list[dict[str, Any]] = []
    max_abs_delta = 0.0
    for i, name in enumerate(RIGHT_ARM_JOINT_NAMES):
        if name not in limits:
            continue
        lower, upper = limits[name]
        lo = lower + float(margin_rad)
        hi = upper - float(margin_rad)
        if lo >= hi:
            lo, hi = lower, upper
        before = float(raw[i])
        after = float(min(max(before, lo), hi))
        clamped[i] = after
        delta = after - before
        max_abs_delta = max(max_abs_delta, abs(delta))
        records.append(
            {
                "joint": name,
                "raw": before,
                "clamped": after,
                "lower": float(lower),
                "upper": float(upper),
                "margin_rad": float(margin_rad),
                "delta": float(delta),
                "raw_below_lower": bool(before < lower),
                "raw_above_upper": bool(before > upper),
            }
        )
    return clamped, {
        "margin_rad": float(margin_rad),
        "max_abs_delta": float(max_abs_delta),
        "changed": bool(max_abs_delta > 0.0),
        "records": records,
    }


def _sphere(center: tuple[float, float, float], radius: float) -> dict[str, Any]:
    return {"center": [float(center[0]), float(center[1]), float(center[2])], "radius": float(radius)}


def kuavo_right_arm_collision_spheres() -> dict[str, list[dict[str, Any]]]:
    """Return a conservative first-pass collision-sphere model.

    The spheres are deliberately simple and slightly padded.  They are good
    enough for table avoidance; fine self-collision tuning can be done later
    from recorded cuRobo debug outputs.
    """

    return {
        "zarm_r1_link": [_sphere((0.0, 0.0, 0.0), 0.075), _sphere((0.0, 0.0, -0.08), 0.060)],
        "zarm_r2_link": [_sphere((0.0, 0.0, 0.0), 0.070), _sphere((0.01, 0.0, -0.10), 0.055)],
        "zarm_r3_link": [
            _sphere((0.02, 0.0, -0.06), 0.060),
            _sphere((0.02, 0.0, -0.16), 0.055),
            _sphere((0.02, 0.0, -0.27), 0.055),
        ],
        "zarm_r4_link": [
            _sphere((0.0, 0.0, 0.0), 0.060),
            _sphere((-0.01, 0.0, -0.06), 0.052),
            _sphere((-0.02, 0.0, -0.12), 0.050),
        ],
        "zarm_r5_link": [_sphere((0.0, 0.0, -0.03), 0.050), _sphere((0.0, 0.0, -0.10), 0.045)],
        "zarm_r6_link": [_sphere((0.0, 0.0, -0.02), 0.045)],
        "zarm_r7_link": [
            _sphere((0.0, 0.0, -0.04), 0.043),
            _sphere((0.0, 0.0, -0.12), 0.040),
            _sphere((0.0, 0.0, -0.17), 0.036),
        ],
        "gripper_base": [_sphere((0.0, 0.0, 0.0), 0.045), _sphere((0.04, 0.0, 0.0), 0.035)],
        "left_finger": [_sphere((0.035, 0.0, 0.0), 0.024), _sphere((0.075, 0.0, 0.0), 0.020)],
        "right_finger": [_sphere((0.035, 0.0, 0.0), 0.024), _sphere((0.075, 0.0, 0.0), 0.020)],
    }


def build_kuavo_right_arm_robot_cfg(
    kuavo_base_urdf: Path,
    gripper_urdf: Path,
    *,
    tcp_offset_m: tuple[float, float, float] = (0.115, 0.0, 0.0),
    joint_limit_clip_rad: float = 0.0,
) -> dict[str, Any]:
    """Build an in-memory cuRobo robot config for the Kuavo right arm."""

    merged_urdf = ensure_kuavo_with_gripper_urdf(kuavo_base_urdf, gripper_urdf)
    tcp_offset = [float(v) for v in tcp_offset_m]
    collision_links = list(RIGHT_ARM_COLLISION_LINKS)
    collision_spheres = kuavo_right_arm_collision_spheres()
    return {
        "robot_cfg": {
            "kinematics": {
                "use_usd_kinematics": False,
                "usd_path": "",
                "usd_robot_root": "/robot",
                "isaac_usd_path": "",
                "usd_flip_joints": {},
                "usd_flip_joint_limits": [],
                "urdf_path": str(merged_urdf),
                "asset_root_path": str(merged_urdf.parent),
                "base_link": "base_link",
                "ee_link": "right_gripper_tcp",
                "link_names": ["right_gripper_tcp"],
                "lock_joints": dict(RIGHT_ARM_LOCK_JOINTS),
                "extra_links": {
                    "right_gripper_tcp": {
                        "parent_link_name": "gripper_base",
                        "link_name": "right_gripper_tcp",
                        "fixed_transform": [tcp_offset[0], tcp_offset[1], tcp_offset[2], 1.0, 0.0, 0.0, 0.0],
                        "joint_type": "FIXED",
                        "joint_name": "right_gripper_tcp_fixed_joint",
                    }
                },
                "collision_link_names": collision_links,
                "collision_spheres": collision_spheres,
                "collision_sphere_buffer": 0.008,
                "extra_collision_spheres": {},
                "self_collision_ignore": {
                    "zarm_r1_link": ["zarm_r2_link", "zarm_r3_link"],
                    "zarm_r2_link": ["zarm_r3_link", "zarm_r4_link"],
                    "zarm_r3_link": ["zarm_r4_link", "zarm_r5_link"],
                    "zarm_r4_link": ["zarm_r5_link", "zarm_r6_link"],
                    "zarm_r5_link": ["zarm_r6_link", "zarm_r7_link"],
                    "zarm_r6_link": ["zarm_r7_link", "gripper_base", "left_finger", "right_finger"],
                    "zarm_r7_link": ["gripper_base", "left_finger", "right_finger"],
                    "gripper_base": ["left_finger", "right_finger"],
                    "left_finger": ["right_finger"],
                },
                "self_collision_buffer": {name: 0.0 for name in collision_links},
                "use_global_cumul": True,
                "mesh_link_names": collision_links,
                "external_asset_path": None,
                "cspace": {
                    "joint_names": list(RIGHT_ARM_JOINT_NAMES),
                    "retract_config": list(RIGHT_ARM_CARRY_CONFIG),
                    "null_space_weight": [1.0, 1.0, 0.8, 1.0, 0.5, 0.5, 0.4],
                    "cspace_distance_weight": [1.0, 1.0, 0.8, 1.0, 0.4, 0.4, 0.3],
                    "max_acceleration": 6.0,
                    "max_jerk": 180.0,
                    "position_limit_clip": float(max(0.0, joint_limit_clip_rad)),
                },
            }
        }
    }


def pose_matrix_to_curobo_pose(pose_b: np.ndarray, tensor_args: Any):
    """Convert a 4x4 base-frame pose matrix to cuRobo Pose."""

    from curobo.types.math import Pose

    rot = np.asarray(pose_b[:3, :3], dtype=np.float32)
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rot[2, 1] - rot[1, 2]) / scale,
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[1, 0] - rot[0, 1]) / scale,
            ],
            dtype=np.float32,
        )
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            scale = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            quat = np.array([(rot[2, 1] - rot[1, 2]) / scale, 0.25 * scale, (rot[0, 1] + rot[1, 0]) / scale, (rot[0, 2] + rot[2, 0]) / scale], dtype=np.float32)
        elif idx == 1:
            scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quat = np.array([(rot[0, 2] - rot[2, 0]) / scale, (rot[0, 1] + rot[1, 0]) / scale, 0.25 * scale, (rot[1, 2] + rot[2, 1]) / scale], dtype=np.float32)
        else:
            scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quat = np.array([(rot[1, 0] - rot[0, 1]) / scale, (rot[0, 2] + rot[2, 0]) / scale, (rot[1, 2] + rot[2, 1]) / scale, 0.25 * scale], dtype=np.float32)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
    return Pose(
        position=tensor_args.to_device(np.asarray(pose_b[:3, 3], dtype=np.float32).reshape(1, 3)),
        quaternion=tensor_args.to_device(quat.reshape(1, 4)),
    )


@dataclass
class CuroboPlanResult:
    success: bool
    reason: str
    joint_positions: np.ndarray
    metadata: dict[str, Any]


class KuavoRightArmCuroboPlanner:
    """Thin lazy-loaded wrapper around cuRobo MotionGen for the right arm."""

    def __init__(
        self,
        kuavo_base_urdf: Path,
        gripper_urdf: Path,
        *,
        tcp_offset_m: tuple[float, float, float] = (0.115, 0.0, 0.0),
        world_model: Any | None = None,
        device: str = "cuda:0",
        warmup: bool = True,
        use_cuda_graph: bool = True,
        collision_activation_distance_m: float = 0.025,
        joint_limit_clip_rad: float = 0.0,
    ) -> None:
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

        self.tensor_args = TensorDeviceType(device=torch.device(device))
        self.plan_config_cls = MotionGenPlanConfig
        self.joint_limit_clip_rad = float(max(0.0, joint_limit_clip_rad))
        self.robot_cfg = build_kuavo_right_arm_robot_cfg(
            kuavo_base_urdf,
            gripper_urdf,
            tcp_offset_m=tcp_offset_m,
            joint_limit_clip_rad=self.joint_limit_clip_rad,
        )
        merged_urdf = Path(self.robot_cfg["robot_cfg"]["kinematics"]["urdf_path"])
        self.joint_limits = _read_joint_limits_from_urdf(merged_urdf, RIGHT_ARM_JOINT_NAMES)
        self.motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            world_model,
            self.tensor_args,
            trajopt_tsteps=32,
            interpolation_steps=240,
            interpolation_dt=0.02,
            num_ik_seeds=48,
            num_trajopt_seeds=8,
            grad_trajopt_iters=220,
            js_trajopt_tsteps=32,
            js_trajopt_dt=0.35,
            evaluate_interpolated_trajectory=True,
            self_collision_check=True,
            self_collision_opt=True,
            use_cuda_graph=bool(use_cuda_graph),
            collision_activation_distance=float(collision_activation_distance_m),
            position_threshold=0.012,
            rotation_threshold=0.75,
            optimize_dt=True,
            project_pose_to_goal_frame=True,
        )
        self.motion_gen = MotionGen(self.motion_gen_cfg)
        if warmup:
            self.motion_gen.warmup(warmup_js_trajopt=False)
        self.joint_names = list(RIGHT_ARM_JOINT_NAMES)

    def update_world(self, world_model: Any) -> None:
        self.motion_gen.update_world(world_model)

    def fk_summary(self, joint_positions: np.ndarray) -> dict[str, Any]:
        q = np.asarray(joint_positions, dtype=np.float32).reshape(1, len(RIGHT_ARM_JOINT_NAMES))
        state = self.motion_gen.kinematics.get_state(self.tensor_args.to_device(q))
        pos = state.ee_position.detach().cpu().numpy().reshape(-1, 3)[0]
        quat = state.ee_quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
        return {
            "ee_position_base_m": pos.astype(float).tolist(),
            "ee_quaternion_wxyz": quat.astype(float).tolist(),
        }

    def plan_to_pose(
        self,
        start_joint_positions: np.ndarray,
        target_pose_in_base: np.ndarray,
        *,
        max_attempts: int = 3,
        enable_graph: bool = True,
        timeout_s: float | None = None,
        check_start_validity: bool = True,
        position_only: bool = False,
    ) -> CuroboPlanResult:
        from curobo.types.robot import JointState

        raw_q = np.asarray(start_joint_positions, dtype=np.float32).reshape(len(RIGHT_ARM_JOINT_NAMES))
        clamp_margin = max(1.0e-4, float(self.joint_limit_clip_rad) + 1.0e-4) if self.joint_limit_clip_rad > 0.0 else 1.0e-4
        clamped_q, start_limit_audit = clamp_right_arm_start_q_to_limits(raw_q, self.joint_limits, margin_rad=clamp_margin)
        q = clamped_q.reshape(1, len(RIGHT_ARM_JOINT_NAMES))
        start = JointState.from_position(self.tensor_args.to_device(q), joint_names=list(RIGHT_ARM_JOINT_NAMES))
        target_pose = np.asarray(target_pose_in_base, dtype=np.float32).reshape(4, 4)
        goal = pose_matrix_to_curobo_pose(target_pose, self.tensor_args)
        start_fk = self.fk_summary(clamped_q)
        target_pos = target_pose[:3, 3].astype(np.float32)
        start_pos = np.asarray(start_fk["ee_position_base_m"], dtype=np.float32)
        pose_cost_metric = None
        if bool(position_only):
            from curobo.rollout.cost.pose_cost import PoseCostMetric

            pose_cost_metric = PoseCostMetric(
                reach_partial_pose=True,
                reach_vec_weight=self.tensor_args.to_device([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
                project_to_goal_frame=False,
            )
        plan_config = self.plan_config_cls(
            max_attempts=max(1, int(max_attempts)),
            enable_graph=bool(enable_graph),
            enable_graph_attempt=2,
            timeout=10.0 if timeout_s is None else float(timeout_s),
            check_start_validity=bool(check_start_validity),
            pose_cost_metric=pose_cost_metric,
            use_start_state_as_retract=True,
        )
        try:
            result = self.motion_gen.plan_single(start, goal, plan_config=plan_config)
        except Exception as exc:  # cuRobo raises rich exceptions for bad configs/targets.
            return CuroboPlanResult(False, f"cuRobo plan exception: {exc}", np.empty((0, len(RIGHT_ARM_JOINT_NAMES)), dtype=np.float32), {"exception": repr(exc)})

        success = bool(result.success.item())
        metadata: dict[str, Any] = {
            "success": success,
            "status": str(getattr(result, "status", "")),
            "solve_time_s": float(getattr(result, "solve_time", 0.0) or 0.0),
            "valid_query": bool(getattr(result, "valid_query", torch.tensor([True])).item()) if hasattr(getattr(result, "valid_query", None), "item") else None,
            "joint_names": list(RIGHT_ARM_JOINT_NAMES),
            "start_q_raw": raw_q.astype(float).tolist(),
            "start_q_used": clamped_q.astype(float).tolist(),
            "start_limit_audit": start_limit_audit,
            "start_fk": start_fk,
            "target_position_base_m": target_pos.astype(float).tolist(),
            "start_to_target_distance_m": float(np.linalg.norm(target_pos - start_pos)),
            "position_only": bool(position_only),
            "pose_cost_metric": {
                "type": "reach_partial_pose" if bool(position_only) else "full_pose",
                "reach_vec_weight_rot_xyz_pos_xyz": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0] if bool(position_only) else [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "project_to_goal_frame": False if bool(position_only) else None,
                "use_start_state_as_retract": True,
            },
            "plan_config": {
                "max_attempts": max(1, int(max_attempts)),
                "enable_graph": bool(enable_graph),
                "timeout_s": 10.0 if timeout_s is None else float(timeout_s),
                "check_start_validity": bool(check_start_validity),
                "position_only": bool(position_only),
            },
        }
        if not success:
            return CuroboPlanResult(False, metadata["status"] or "cuRobo plan failed", np.empty((0, len(RIGHT_ARM_JOINT_NAMES)), dtype=np.float32), metadata)
        traj = result.get_interpolated_plan()
        positions = traj.position.detach().cpu().numpy().astype(np.float32)
        if positions.ndim == 3:
            positions = positions[0]
        metadata["interpolated_steps"] = int(positions.shape[0])
        metadata["interpolation_dt_s"] = float(result.interpolation_dt)
        if positions.shape[0] > 0:
            planned_final_fk = self.fk_summary(positions[-1])
            planned_final_pos = np.asarray(planned_final_fk["ee_position_base_m"], dtype=np.float32)
            metadata["planned_final_fk"] = planned_final_fk
            metadata["planned_final_target_pos_error_m"] = float(np.linalg.norm(planned_final_pos - target_pos))
        return CuroboPlanResult(True, "", positions, metadata)

    def plan_to_joint_positions(
        self,
        start_joint_positions: np.ndarray,
        target_joint_positions: np.ndarray,
        *,
        max_attempts: int = 3,
        enable_graph: bool = True,
        timeout_s: float | None = None,
        check_start_validity: bool = True,
    ) -> CuroboPlanResult:
        """Plan a collision-aware joint-space path to a supplied right-arm posture.

        This is used to let an external Kuavo IK implementation choose the
        redundant arm posture while cuRobo remains responsible for smooth,
        collision-aware trajectory generation.
        """

        from curobo.types.robot import JointState

        raw_start_q = np.asarray(start_joint_positions, dtype=np.float32).reshape(len(RIGHT_ARM_JOINT_NAMES))
        raw_goal_q = np.asarray(target_joint_positions, dtype=np.float32).reshape(len(RIGHT_ARM_JOINT_NAMES))
        clamp_margin = max(1.0e-4, float(self.joint_limit_clip_rad) + 1.0e-4) if self.joint_limit_clip_rad > 0.0 else 1.0e-4
        start_q, start_limit_audit = clamp_right_arm_start_q_to_limits(raw_start_q, self.joint_limits, margin_rad=clamp_margin)
        goal_q, goal_limit_audit = clamp_right_arm_start_q_to_limits(raw_goal_q, self.joint_limits, margin_rad=clamp_margin)
        start = JointState.from_position(
            self.tensor_args.to_device(start_q.reshape(1, len(RIGHT_ARM_JOINT_NAMES))),
            joint_names=list(RIGHT_ARM_JOINT_NAMES),
        )
        goal = JointState.from_position(
            self.tensor_args.to_device(goal_q.reshape(1, len(RIGHT_ARM_JOINT_NAMES))),
            joint_names=list(RIGHT_ARM_JOINT_NAMES),
        )
        plan_config = self.plan_config_cls(
            max_attempts=max(1, int(max_attempts)),
            enable_graph=bool(enable_graph),
            enable_graph_attempt=2,
            timeout=10.0 if timeout_s is None else float(timeout_s),
            check_start_validity=bool(check_start_validity),
        )
        try:
            result = self.motion_gen.plan_single_js(start, goal, plan_config=plan_config)
        except Exception as exc:
            return CuroboPlanResult(
                False,
                f"cuRobo joint-space plan exception: {exc}",
                np.empty((0, len(RIGHT_ARM_JOINT_NAMES)), dtype=np.float32),
                {"exception": repr(exc)},
            )

        success = bool(result.success.item())
        metadata: dict[str, Any] = {
            "success": success,
            "status": str(getattr(result, "status", "")),
            "solve_time_s": float(getattr(result, "solve_time", 0.0) or 0.0),
            "valid_query": bool(getattr(result, "valid_query", torch.tensor([True])).item()) if hasattr(getattr(result, "valid_query", None), "item") else None,
            "joint_names": list(RIGHT_ARM_JOINT_NAMES),
            "start_q_raw": raw_start_q.astype(float).tolist(),
            "start_q_used": start_q.astype(float).tolist(),
            "target_q_raw": raw_goal_q.astype(float).tolist(),
            "target_q_used": goal_q.astype(float).tolist(),
            "start_limit_audit": start_limit_audit,
            "target_limit_audit": goal_limit_audit,
            "start_fk": self.fk_summary(start_q),
            "target_fk": self.fk_summary(goal_q),
            "start_to_target_joint_l2_rad": float(np.linalg.norm(goal_q - start_q)),
            "plan_config": {
                "max_attempts": max(1, int(max_attempts)),
                "enable_graph": bool(enable_graph),
                "timeout_s": 10.0 if timeout_s is None else float(timeout_s),
                "check_start_validity": bool(check_start_validity),
            },
        }
        if not success:
            return CuroboPlanResult(False, metadata["status"] or "cuRobo joint-space plan failed", np.empty((0, len(RIGHT_ARM_JOINT_NAMES)), dtype=np.float32), metadata)
        traj = result.get_interpolated_plan()
        positions = traj.position.detach().cpu().numpy().astype(np.float32)
        if positions.ndim == 3:
            positions = positions[0]
        metadata["interpolated_steps"] = int(positions.shape[0])
        metadata["interpolation_dt_s"] = float(result.interpolation_dt)
        if positions.shape[0] > 0:
            planned_final_q = positions[-1]
            planned_final_fk = self.fk_summary(planned_final_q)
            metadata["planned_final_fk"] = planned_final_fk
            metadata["planned_final_joint_l2_error_rad"] = float(np.linalg.norm(planned_final_q - goal_q))
        return CuroboPlanResult(True, "", positions, metadata)


def build_world_cuboids(cuboids: list[dict[str, Any]]):
    """Build a cuRobo WorldConfig from cuboid dictionaries."""

    from curobo.geom.types import Cuboid, WorldConfig

    return WorldConfig(
        cuboid=[
            Cuboid(
                name=str(c["name"]),
                pose=[float(v) for v in c["pose"]],
                dims=[float(v) for v in c["dims"]],
            )
            for c in cuboids
        ]
    )
