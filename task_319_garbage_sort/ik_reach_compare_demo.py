"""Compare official Kuavo IK and cuRobo reaching the same hardcoded TCP target.

This script intentionally bypasses perception, navigation, grasping, and object
physics. It starts the right arm from a visible natural-down posture, reads the
current simulated wrist/TCP pose, adds a hardcoded low base-frame offset, checks
that both official Kuavo IK and cuRobo can reach that same target, then records
one video per backend.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
for source_dir in (
    WORKSPACE_ROOT / "IsaacLab/source/isaaclab",
    WORKSPACE_ROOT / "IsaacLab/source/isaaclab_assets",
    WORKSPACE_ROOT / "IsaacLab/source/isaaclab_tasks",
):
    if source_dir.exists() and str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from isaaclab.app import AppLauncher


def parse_xyz(value: str) -> tuple[float, float, float]:
    parts = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected x,y,z triplet, got {value!r}")
    return (parts[0], parts[1], parts[2])


parser = argparse.ArgumentParser(description="Task319 official IK vs cuRobo fixed-target reach comparison.")
parser.add_argument("--target_delta_b", type=parse_xyz, default=(0.30, -0.04, 0.16), help="Preferred wrist target offset in robot base frame, meters.")
parser.add_argument("--min_target_delta_norm_m", type=float, default=0.12, help="Reject fallback target offsets shorter than this distance.")
parser.add_argument("--start_arm_pose", choices=("natural_down", "carry"), default="natural_down", help="Initial right-arm posture before solving and recording.")
parser.add_argument("--trajectory_steps", type=int, default=260, help="Official IK min-jerk execution steps.")
parser.add_argument("--hold_steps", type=int, default=160, help="Hold steps after reaching the target.")
parser.add_argument("--settle_steps", type=int, default=240, help="Extra final-target settling steps for actuator tracking.")
parser.add_argument("--joint_error_threshold_rad", type=float, default=0.025, help="Stop final settling once controlled joints track this closely.")
parser.add_argument("--precheck_trajectory_steps", type=int, default=180, help="Fast no-video execution steps used while selecting a reachable target.")
parser.add_argument("--precheck_hold_steps", type=int, default=120, help="Fast no-video hold/settle steps used while selecting a reachable target.")
parser.add_argument("--force_joint_state_tracking", action=argparse.BooleanOptionalAction, default=True, help="For this solver comparison, directly write each smooth joint command to the simulator for ideal tracking.")
parser.add_argument("--warmup_steps", type=int, default=80)
parser.add_argument("--max_joint_step", type=float, default=0.012, help="Per-step right-arm joint command clamp in radians.")
parser.add_argument("--success_position_error_m", type=float, default=0.03)
parser.add_argument("--record_video", action="store_true")
parser.add_argument("--save_video_frames", action="store_true")
parser.add_argument("--video_width", type=int, default=1280)
parser.add_argument("--video_height", type=int, default=720)
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--video_sample_stride", type=int, default=4)
parser.add_argument("--observer_camera_pos", type=parse_xyz, default=(1.35, -1.65, 1.05))
parser.add_argument("--observer_camera_target", type=parse_xyz, default=(0.34, -0.23, 0.55))
parser.add_argument("--obstacle_demo", action=argparse.BooleanOptionalAction, default=True, help="Spawn a medium obstacle between the natural-down right arm and target.")
parser.add_argument("--obstacle_pos_b", type=parse_xyz, default=(0.285, -0.273, 0.415), help="Obstacle center in robot base/world frame, meters.")
parser.add_argument("--obstacle_size_b", type=parse_xyz, default=(0.14, 0.22, 0.12), help="Obstacle cuboid dimensions, meters.")
parser.add_argument("--curobo_obstacle_world", action=argparse.BooleanOptionalAction, default=True, help="Add the demo obstacle to cuRobo collision checking.")
parser.add_argument("--curobo_collision_activation_distance_m", type=float, default=0.03)
parser.add_argument("--output_dir", type=Path, default=WORKSPACE_ROOT / "task_319_garbage_sort/output/ik_reach_compare")
parser.add_argument("--curobo_device", default="cuda:0")
parser.add_argument("--kuavo_ik_host", default="127.0.0.1")
parser.add_argument("--kuavo_ik_port", type=int, default=31975)
parser.add_argument("--kuavo_ik_timeout_s", type=float, default=4.0)
parser.add_argument("--kuavo_ik_connect_timeout_s", type=float, default=25.0)
parser.add_argument("--kuavo_ik_auto_start", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--kuavo_ik_auto_start_mode", choices=("docker", "host_sidecar"), default="docker")
parser.add_argument("--kuavo_ik_docker_container", default="kuavo_official_ros")
parser.add_argument("--kuavo_ik_docker_workspace", default="/root/kuavo_ws_linux")
parser.add_argument("--kuavo_ik_python", default="python3")
parser.add_argument("--kuavo_ik_sidecar_script", type=Path, default=WORKSPACE_ROOT / "task_319_garbage_sort/scripts/start_kuavo_official_ik_sidecar.sh")
parser.add_argument("--kuavo_ik_service", default="/ik/two_arm_hand_pose_cmd_srv")
parser.add_argument("--kuavo_ik_fk_service", default="/ik/fk_srv")
parser.add_argument("--kuavo_ik_reset_node", action=argparse.BooleanOptionalAction, default=True, help="Restart the official ROS IK node in docker mode so its stateful q0 seed is clean.")
parser.add_argument("--kuavo_ik_robot_version", type=int, default=62)
parser.add_argument("--kuavo_ik_frame", type=int, default=2)
parser.add_argument("--kuavo_ik_pos_tol_m", type=float, default=0.008)
parser.add_argument("--kuavo_ik_ori_tol_rad", type=float, default=0.35)
parser.add_argument("--kuavo_ik_pos_cost_weight", type=float, default=80.0)
parser.add_argument(
    "--official_apply_upper_body",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply the official IK q_torso prefix before the right-arm joints. Disabled by default for right-arm-only comparison.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat

from task_319_garbage_sort.curobo_right_arm import (  # noqa: E402
    RIGHT_ARM_CARRY_CONFIG,
    RIGHT_ARM_JOINT_NAMES,
    CuroboPlanResult,
    KuavoRightArmCuroboPlanner,
    build_world_cuboids,
)
from task_319_garbage_sort.gripper_robot_urdf import ensure_kuavo_with_gripper_urdf  # noqa: E402


ROOT_DIR = WORKSPACE_ROOT
GRIPPER_URDF = ROOT_DIR / "task_319_garbage_sort/two_finger_gripper.urdf"
KUAVO_BASE_URDF = ROOT_DIR / "kuavo-ros-opensource/src/kuavo_assets/models/biped_s62/urdf/biped_s62.urdf"
KUAVO_WITH_GRIPPER_URDF = ensure_kuavo_with_gripper_urdf(KUAVO_BASE_URDF, GRIPPER_URDF)

ROBOT_ROOT_Z_ON_WHEELS = 0.0
RIGHT_ARM_JOINT_EXPR = "zarm_r[1-7]_joint"


def rpy_matrix_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


GRIPPER_LOCAL_TCP_OFFSET = np.array([0.115, 0.0, 0.0], dtype=np.float32)
RIGHT_GRIPPER_INLINE_MOUNT_ROT = rpy_matrix_xyz(0.0, math.pi / 2.0, 0.0)
RIGHT_WRIST_INLINE_TCP_OFFSET = (RIGHT_GRIPPER_INLINE_MOUNT_ROT @ GRIPPER_LOCAL_TCP_OFFSET).astype(np.float32)
OFFICIAL_REFERENCE_RIGHT_WRIST_POS_B = np.array([0.45, -0.25, 0.11988012], dtype=np.float32)
OFFICIAL_REFERENCE_RIGHT_WRIST_QUAT_XYZW = np.array([0.0, -0.70682518, 0.0, 0.70738827], dtype=np.float32)
RIGHT_ARM_NATURAL_DOWN_CONFIG = [0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0]
KUAVO_IK_TORSO_JOINT_NAMES = ["knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint"]
KUAVO_IK_LEFT_ARM_JOINT_NAMES = [
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_l5_joint",
    "zarm_l6_joint",
    "zarm_l7_joint",
]
KUAVO_IK_Q_ARM_JOINT_NAMES = [*KUAVO_IK_LEFT_ARM_JOINT_NAMES, *RIGHT_ARM_JOINT_NAMES]


def rigid_props(*, kinematic: bool = False) -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        kinematic_enabled=bool(kinematic),
        disable_gravity=bool(kinematic),
        linear_damping=0.2,
        angular_damping=0.2,
        max_depenetration_velocity=1.0,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=6,
    )


def collision_props(contact_offset: float = 0.003, rest_offset: float = 0.0) -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=max(float(contact_offset), float(rest_offset) + 1.0e-4),
        rest_offset=max(0.0, float(rest_offset)),
    )


def material(static_friction: float, dynamic_friction: float) -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=float(static_friction),
        dynamic_friction=float(dynamic_friction),
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )


def surface(color: tuple[float, float, float]) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(v) for v in color))


def active_obstacle_pos_b() -> tuple[float, float, float]:
    if not bool(args_cli.obstacle_demo):
        return (20.0, 20.0, 20.0)
    return tuple(float(v) for v in args_cli.obstacle_pos_b)


def active_obstacle_size_b() -> tuple[float, float, float]:
    if not bool(args_cli.obstacle_demo):
        return (0.01, 0.01, 0.01)
    return tuple(float(v) for v in args_cli.obstacle_size_b)


def kuavo_robot_cfg() -> ArticulationCfg:
    right_config = RIGHT_ARM_NATURAL_DOWN_CONFIG if str(args_cli.start_arm_pose) == "natural_down" else RIGHT_ARM_CARRY_CONFIG
    right_home = dict(zip(RIGHT_ARM_JOINT_NAMES, right_config, strict=True))
    left_home = {"zarm_l2_joint": 0.0, "zarm_l4_joint": 0.0} if str(args_cli.start_arm_pose) == "natural_down" else {"zarm_l2_joint": 0.30, "zarm_l4_joint": -0.60}
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo62",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(KUAVO_WITH_GRIPPER_URDF),
            fix_base=True,
            merge_fixed_joints=False,
            link_density=1000.0,
            collision_from_visuals=True,
            collider_type="convex_hull",
            self_collision=False,
            rigid_props=rigid_props(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=6,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=180.0, damping=20.0),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, ROBOT_ROOT_Z_ON_WHEELS),
            joint_pos={
                "knee_joint": 0.0,
                "leg_joint": 0.0,
                "waist_pitch_joint": 0.0,
                "waist_yaw_joint": 0.0,
                "zhead_1_joint": 0.0,
                "zhead_2_joint": 0.0,
                **left_home,
                **right_home,
                "left_finger_joint": 0.055,
                "right_finger_joint": 0.055,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "locked_wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_.*_joint"],
                effort_limit_sim=300.0,
                velocity_limit_sim=0.05,
                stiffness=5000.0,
                damping=600.0,
            ),
            "locked_stance": ImplicitActuatorCfg(
                joint_names_expr=["knee_joint", "leg_joint"],
                effort_limit_sim=900.0,
                velocity_limit_sim=0.1,
                stiffness=4500.0,
                damping=500.0,
            ),
            "locked_torso": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*_joint"],
                effort_limit_sim=600.0,
                velocity_limit_sim=1.2,
                stiffness=3200.0,
                damping=360.0,
            ),
            "locked_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["zarm_l[1-7]_joint"],
                effort_limit_sim=180.0,
                velocity_limit_sim=0.1,
                stiffness=2400.0,
                damping=260.0,
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=[RIGHT_ARM_JOINT_EXPR],
                effort_limit_sim=600.0,
                velocity_limit_sim=3.2,
                stiffness=2200.0,
                damping=220.0,
            ),
            "locked_head": ImplicitActuatorCfg(
                joint_names_expr=["zhead_[12]_joint"],
                effort_limit_sim=60.0,
                velocity_limit_sim=0.1,
                stiffness=900.0,
                damping=120.0,
            ),
            "locked_gripper": ImplicitActuatorCfg(
                joint_names_expr=[".*finger_joint"],
                effort_limit_sim=60.0,
                velocity_limit_sim=0.1,
                stiffness=1200.0,
                damping=120.0,
            ),
        },
    )


def quat_wxyz_from_matrix(rot: np.ndarray) -> tuple[float, float, float, float]:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rot[2, 1] - rot[1, 2]) / scale,
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[1, 0] - rot[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            scale = math.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 1.0e-12)) * 2.0
            quat = np.array([(rot[2, 1] - rot[1, 2]) / scale, 0.25 * scale, (rot[0, 1] + rot[1, 0]) / scale, (rot[0, 2] + rot[2, 0]) / scale])
        elif idx == 1:
            scale = math.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 1.0e-12)) * 2.0
            quat = np.array([(rot[0, 2] - rot[2, 0]) / scale, (rot[0, 1] + rot[1, 0]) / scale, 0.25 * scale, (rot[1, 2] + rot[2, 1]) / scale])
        else:
            scale = math.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 1.0e-12)) * 2.0
            quat = np.array([(rot[1, 0] - rot[0, 1]) / scale, (rot[0, 2] + rot[2, 0]) / scale, (rot[1, 2] + rot[2, 1]) / scale, 0.25 * scale])
    quat /= max(float(np.linalg.norm(quat)), 1.0e-9)
    return tuple(float(v) for v in quat)


def quat_xyzw_from_wxyz(quat: np.ndarray | list[float] | tuple[float, ...]) -> list[float]:
    q = np.asarray(quat, dtype=np.float32).reshape(4)
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def quat_wxyz_to_matrix(quat: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quat, dtype=np.float64).reshape(4)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def look_at_quat_world(pos: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[float, float, float, float]:
    forward = np.asarray(target, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1.0e-9)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1.0e-6:
        right = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right /= max(np.linalg.norm(right), 1.0e-9)
    up = np.cross(forward, right)
    up /= max(np.linalg.norm(up), 1.0e-9)
    return quat_wxyz_from_matrix(np.stack([forward, right, up], axis=1))


@configclass
class IkReachCompareSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(5.0, 5.0),
            physics_material=material(1.2, 1.0),
            color=(0.31, 0.32, 0.31),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2400.0, color=(0.82, 0.86, 0.90)),
    )
    robot: ArticulationCfg = kuavo_robot_cfg()
    mid_path_obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MidPathObstacle",
        spawn=sim_utils.CuboidCfg(
            size=active_obstacle_size_b(),
            rigid_props=rigid_props(kinematic=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=collision_props(0.004),
            physics_material=material(1.0, 0.8),
            visual_material=surface((0.90, 0.12, 0.08)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=active_obstacle_pos_b()),
    )
    observer_rgb = CameraCfg(
        prim_path="{ENV_REGEX_NS}/observer_rgb",
        update_period=0.0,
        width=int(args_cli.video_width),
        height=int(args_cli.video_height),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=3.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 8.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=args_cli.observer_camera_pos,
            rot=look_at_quat_world(args_cli.observer_camera_pos, args_cli.observer_camera_target),
            convention="world",
        ),
    )


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def sanitize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


class ObserverVideoRecorder:
    def __init__(self, run_dir: Path, label: str) -> None:
        self.enabled = bool(args_cli.record_video)
        self.path = run_dir / f"{label}.mp4"
        self.frame_dir = run_dir / f"{label}_frames"
        self.fps = max(1.0, float(args_cli.video_fps))
        self.sample_stride = max(1, int(args_cli.video_sample_stride))
        self._step_count = 0
        self.frame_count = 0
        self._writer: Any | None = None
        self._writer_failed = False
        self.error = ""
        if self.enabled and bool(args_cli.save_video_frames):
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, scene: InteractiveScene) -> None:
        if not self.enabled:
            return
        self._step_count += 1
        if self._step_count % self.sample_stride != 0:
            return
        if "observer_rgb" not in scene.keys():
            self.error = "observer_rgb camera is not available."
            return
        try:
            frame = sanitize_rgb(tensor_to_numpy(scene["observer_rgb"].data.output["rgb"])[0])
            self._write_frame(frame)
        except Exception as exc:
            self.error = repr(exc)

    def _write_frame(self, frame: np.ndarray) -> None:
        if self._writer is None and not self._writer_failed:
            try:
                import cv2  # type: ignore

                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
                if not self._writer.isOpened():
                    self._writer = None
                    self._writer_failed = True
                    self.error = "OpenCV VideoWriter could not open mp4 output."
            except Exception as exc:
                self._writer = None
                self._writer_failed = True
                self.error = repr(exc)
        if self._writer is not None:
            import cv2  # type: ignore

            self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if bool(args_cli.save_video_frames) or self._writer_failed:
            from PIL import Image

            self.frame_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(self.frame_dir / f"frame_{self.frame_count:06d}.png")
        self.frame_count += 1

    def finalize(self) -> None:
        if self._writer is not None:
            self._writer.release()


def robot_root_pose_matrix(robot: Articulation) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = robot.data.root_pose_w[0, :3].detach().cpu().numpy().astype(np.float32)
    pose[:3, :3] = matrix_from_quat(robot.data.root_pose_w[:, 3:7])[0].detach().cpu().numpy().astype(np.float32)
    return pose


def body_pose_matrix_by_name(robot: Articulation, body_name: str) -> np.ndarray | None:
    body_names = getattr(robot, "body_names", None)
    if body_names is None or body_name not in body_names:
        return None
    body_id = int(body_names.index(body_name))
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = robot.data.body_pose_w[0, body_id, 0:3].detach().cpu().numpy().astype(np.float32)
    pose[:3, :3] = matrix_from_quat(robot.data.body_pose_w[:, body_id, 3:7])[0].detach().cpu().numpy().astype(np.float32)
    return pose


def tcp_pose_matrix(robot: Articulation) -> np.ndarray:
    gripper_base_pose = body_pose_matrix_by_name(robot, "gripper_base")
    if gripper_base_pose is None:
        raise RuntimeError("Cannot resolve gripper_base body for TCP pose.")
    tcp_pose = gripper_base_pose.copy()
    tcp_pose[:3, 3] = gripper_base_pose[:3, 3] + gripper_base_pose[:3, :3] @ GRIPPER_LOCAL_TCP_OFFSET
    return tcp_pose.astype(np.float32)


def wrist_pose_matrix(robot: Articulation) -> np.ndarray:
    wrist_pose = body_pose_matrix_by_name(robot, "zarm_r7_end_effector")
    if wrist_pose is None:
        raise RuntimeError("Cannot resolve zarm_r7_end_effector body for wrist pose.")
    return wrist_pose.astype(np.float32)


def tcp_pose_to_official_wrist_pose(tcp_pose: np.ndarray) -> np.ndarray:
    tcp_pose = np.asarray(tcp_pose, dtype=np.float32).reshape(4, 4)
    wrist_pose = tcp_pose.copy()
    wrist_pose[:3, :3] = tcp_pose[:3, :3] @ RIGHT_GRIPPER_INLINE_MOUNT_ROT.T
    wrist_pose[:3, 3] = tcp_pose[:3, 3] - wrist_pose[:3, :3] @ RIGHT_WRIST_INLINE_TCP_OFFSET
    return wrist_pose.astype(np.float32)


def official_wrist_pose_to_tcp_pose(wrist_pose: np.ndarray) -> np.ndarray:
    wrist_pose = np.asarray(wrist_pose, dtype=np.float32).reshape(4, 4)
    tcp_pose = wrist_pose.copy()
    tcp_pose[:3, :3] = wrist_pose[:3, :3] @ RIGHT_GRIPPER_INLINE_MOUNT_ROT
    tcp_pose[:3, 3] = wrist_pose[:3, 3] + wrist_pose[:3, :3] @ RIGHT_WRIST_INLINE_TCP_OFFSET
    return tcp_pose.astype(np.float32)


def kuavo_hand_pose_matrix(hand_pose: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(hand_pose, dict):
        return None
    pos = np.asarray(hand_pose.get("pos_xyz", []), dtype=np.float32).reshape(-1)
    quat_xyzw = np.asarray(hand_pose.get("quat_xyzw", []), dtype=np.float32).reshape(-1)
    if pos.size < 3 or quat_xyzw.size < 4:
        return None
    if not np.isfinite(pos[:3]).all() or not np.isfinite(quat_xyzw[:4]).all():
        return None
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = pos[:3]
    pose[:3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    return pose


def pose_matrix_from_pos_quat_xyzw(pos_xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos_xyz, dtype=np.float32).reshape(3)
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = pos
    pose[:3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    return pose


def kuavo_response_right_tcp_pose_base(response: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(response, dict):
        return None
    hand_poses = response.get("hand_poses")
    if not isinstance(hand_poses, dict):
        return None
    wrist_pose = kuavo_hand_pose_matrix(hand_poses.get("right_pose"))
    if wrist_pose is None:
        return None
    return official_wrist_pose_to_tcp_pose(wrist_pose)


def world_pose_to_robot_base_pose(robot: Articulation, pose_w: np.ndarray) -> np.ndarray:
    return (np.linalg.inv(robot_root_pose_matrix(robot)) @ np.asarray(pose_w, dtype=np.float32).reshape(4, 4)).astype(np.float32)


def robot_base_pose_to_world_pose(robot: Articulation, pose_b: np.ndarray) -> np.ndarray:
    return (robot_root_pose_matrix(robot) @ np.asarray(pose_b, dtype=np.float32).reshape(4, 4)).astype(np.float32)


def pose_matrix_to_curobo_list(pose_b: np.ndarray) -> list[float]:
    pose_b = np.asarray(pose_b, dtype=np.float32).reshape(4, 4)
    quat = quat_wxyz_from_matrix(pose_b[:3, :3])
    return [
        float(pose_b[0, 3]),
        float(pose_b[1, 3]),
        float(pose_b[2, 3]),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    ]


def curobo_obstacle_world_model(robot: Articulation):
    if not bool(args_cli.obstacle_demo) or not bool(args_cli.curobo_obstacle_world):
        return None
    obstacle_pose_w = np.eye(4, dtype=np.float32)
    obstacle_pose_w[:3, 3] = np.asarray(active_obstacle_pos_b(), dtype=np.float32)
    obstacle_pose_b = world_pose_to_robot_base_pose(robot, obstacle_pose_w)
    return build_world_cuboids(
        [
            {
                "name": "ik_reach_mid_path_obstacle",
                "pose": pose_matrix_to_curobo_list(obstacle_pose_b),
                "dims": [float(v) for v in active_obstacle_size_b()],
            }
        ]
    )


def signed_distance_point_to_aabb(point_w: np.ndarray, center_w: np.ndarray, size: tuple[float, float, float]) -> float:
    point = np.asarray(point_w, dtype=np.float32).reshape(3)
    center = np.asarray(center_w, dtype=np.float32).reshape(3)
    half = 0.5 * np.asarray(size, dtype=np.float32).reshape(3)
    delta = np.abs(point - center) - half
    outside = np.maximum(delta, 0.0)
    inside = min(float(np.max(delta)), 0.0)
    return float(np.linalg.norm(outside) + inside)


def obstacle_pose_summary(robot: Articulation | None = None) -> dict[str, Any]:
    center_w = np.asarray(active_obstacle_pos_b(), dtype=np.float32)
    pose_b = np.eye(4, dtype=np.float32)
    pose_b[:3, 3] = center_w
    if robot is not None:
        pose_b = world_pose_to_robot_base_pose(robot, pose_b)
    return {
        "enabled": bool(args_cli.obstacle_demo),
        "curobo_world_enabled": bool(args_cli.obstacle_demo) and bool(args_cli.curobo_obstacle_world),
        "center_world_m": center_w.astype(float).tolist(),
        "center_base_m": pose_b[:3, 3].astype(float).tolist(),
        "size_m": [float(v) for v in active_obstacle_size_b()],
    }


def right_arm_joint_ids(robot: Articulation) -> list[int]:
    joint_ids, joint_names = robot.find_joints(RIGHT_ARM_JOINT_NAMES, preserve_order=True)
    if len(joint_ids) != len(RIGHT_ARM_JOINT_NAMES):
        raise RuntimeError(f"Expected 7 right-arm joints, got {joint_names}.")
    return list(joint_ids)


def q_arm_joint_ids(robot: Articulation) -> list[int]:
    joint_ids, joint_names = robot.find_joints(KUAVO_IK_Q_ARM_JOINT_NAMES, preserve_order=True)
    if len(joint_ids) != len(KUAVO_IK_Q_ARM_JOINT_NAMES):
        raise RuntimeError(f"Expected 14 official q_arm joints, got {joint_names}.")
    return list(joint_ids)


def current_joint_q(robot: Articulation, joint_ids: list[int]) -> np.ndarray:
    return robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().astype(np.float32)


def current_q_arm(robot: Articulation) -> np.ndarray:
    return current_joint_q(robot, q_arm_joint_ids(robot))


def current_right_q(robot: Articulation, joint_ids: list[int]) -> np.ndarray:
    return current_joint_q(robot, joint_ids)


def set_joint_q_target(robot: Articulation, joint_ids: list[int], q: np.ndarray, locked_target: torch.Tensor) -> None:
    q_tensor = torch.tensor(np.asarray(q, dtype=np.float32).reshape(1, -1), dtype=torch.float32, device=robot.device)
    robot.set_joint_position_target(locked_target)
    robot.set_joint_position_target(q_tensor, joint_ids=joint_ids)
    if bool(args_cli.force_joint_state_tracking):
        zero_vel = torch.zeros_like(q_tensor)
        robot.write_joint_state_to_sim(q_tensor, zero_vel, joint_ids=joint_ids)


def set_right_q_target(robot: Articulation, joint_ids: list[int], q: np.ndarray, locked_target: torch.Tensor) -> None:
    set_joint_q_target(robot, joint_ids, q, locked_target)


def reset_robot(scene: InteractiveScene, robot: Articulation) -> None:
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    root_state[:, 2] = ROBOT_ROOT_Z_ON_WHEELS
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    scene.reset()


def step_scene(
    sim: SimulationContext,
    scene: InteractiveScene,
    recorder: ObserverVideoRecorder | None = None,
    current_marker: VisualizationMarkers | None = None,
    goal_marker: VisualizationMarkers | None = None,
    target_pose_w: np.ndarray | None = None,
    tracked_pose_fn: Any = tcp_pose_matrix,
) -> None:
    robot: Articulation = scene["robot"]
    if current_marker is not None:
        tcp_w = tracked_pose_fn(robot)
        current_marker.visualize(
            torch.tensor(tcp_w[:3, 3], dtype=torch.float32, device=robot.device).reshape(1, 3),
            torch.tensor(quat_wxyz_from_matrix(tcp_w[:3, :3]), dtype=torch.float32, device=robot.device).reshape(1, 4),
        )
    if goal_marker is not None and target_pose_w is not None:
        goal_marker.visualize(
            torch.tensor(target_pose_w[:3, 3], dtype=torch.float32, device=robot.device).reshape(1, 3),
            torch.tensor(quat_wxyz_from_matrix(target_pose_w[:3, :3]), dtype=torch.float32, device=robot.device).reshape(1, 4),
        )
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(sim.get_physics_dt())
    if recorder is not None:
        recorder.capture(scene)


def min_jerk(alpha: float) -> float:
    alpha = max(0.0, min(1.0, float(alpha)))
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def clamp_joint_step(desired: np.ndarray, previous: np.ndarray, max_step: float) -> np.ndarray:
    return previous + np.clip(desired - previous, -float(max_step), float(max_step))


def target_deltas() -> list[tuple[float, float, float]]:
    preferred = tuple(float(v) for v in args_cli.target_delta_b)
    preferred_np = np.asarray(preferred, dtype=np.float32)
    candidates = [
        preferred,
        tuple(float(v) for v in 0.85 * preferred_np),
        tuple(float(v) for v in 0.70 * preferred_np),
        tuple(float(v) for v in 0.55 * preferred_np),
        (0.24, -0.04, 0.13),
        (0.22, -0.03, 0.11),
        (0.18, -0.03, 0.09),
        (0.16, -0.02, 0.08),
        (0.035, -0.02, 0.025),
        (0.03, -0.02, 0.03),
        (0.025, -0.015, 0.02),
    ]
    unique: list[tuple[float, float, float]] = []
    min_norm = max(0.0, float(args_cli.min_target_delta_norm_m))
    for cand in candidates:
        normalized = tuple(float(v) for v in cand)
        if float(np.linalg.norm(np.asarray(normalized, dtype=np.float32))) < min_norm:
            continue
        if normalized not in unique:
            unique.append(normalized)
    return unique


class KuavoIkBridgeClient:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.process: subprocess.Popen[Any] | None = None
        self.sock: socket.socket | None = None
        self.stream: Any | None = None
        self.health: dict[str, Any] = {}
        self.last_error = ""

    def start(self) -> bool:
        if str(args_cli.kuavo_ik_auto_start_mode) == "docker" and bool(args_cli.kuavo_ik_reset_node):
            if not self._reset_docker_ik_node():
                return False
        if self._connect_once():
            return True
        if not bool(args_cli.kuavo_ik_auto_start):
            return False
        if str(args_cli.kuavo_ik_auto_start_mode) == "docker":
            if not self._start_docker_bridge():
                return False
        else:
            if not self._start_host_sidecar():
                return False
        deadline = time.time() + float(args_cli.kuavo_ik_connect_timeout_s)
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                self.last_error = f"IK bridge exited early with code {self.process.returncode}; see {self.run_dir / 'kuavo_ik_sidecar'}"
                return False
            if self._connect_once():
                return True
            time.sleep(0.2)
        self.last_error = f"Timed out waiting for Kuavo IK bridge at {args_cli.kuavo_ik_host}:{args_cli.kuavo_ik_port}; see {self.run_dir / 'kuavo_ik_sidecar'}"
        return False

    def _start_host_sidecar(self) -> bool:
        script = Path(args_cli.kuavo_ik_sidecar_script)
        if not script.exists():
            self.last_error = f"Kuavo IK sidecar script missing: {script}"
            return False
        log_dir = self.run_dir / "kuavo_ik_sidecar"
        log_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["KUAVO_IK_BRIDGE_HOST"] = str(args_cli.kuavo_ik_host)
        env["KUAVO_IK_BRIDGE_PORT"] = str(args_cli.kuavo_ik_port)
        env["KUAVO_IK_SERVICE"] = str(args_cli.kuavo_ik_service)
        env["KUAVO_IK_LOG_DIR"] = str(log_dir)
        log_file = (log_dir / "sidecar.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            ["bash", str(script)],
            cwd=str(WORKSPACE_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_file.close()
        return True

    def _start_docker_bridge(self) -> bool:
        script = WORKSPACE_ROOT / "task_319_garbage_sort/scripts/kuavo_ik_socket_bridge.py"
        if not script.exists():
            self.last_error = f"Kuavo IK socket bridge script missing: {script}"
            return False
        container = str(args_cli.kuavo_ik_docker_container)
        container_script = "/tmp/task319_kuavo_ik_socket_bridge.py"
        log_dir = self.run_dir / "kuavo_ik_sidecar"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["docker", "cp", str(script), f"{container}:{container_script}"],
                text=True,
                capture_output=True,
                timeout=10,
                check=True,
            )
        except Exception as exc:
            self.last_error = f"Could not copy IK bridge into docker container {container}: {exc!r}"
            return False
        workspace = str(args_cli.kuavo_ik_docker_workspace).rstrip("/")
        setup = (
            f"cd {workspace}; "
            "source /opt/ros/noetic/setup.bash; "
            f"test -f {workspace}/devel/setup.bash && source {workspace}/devel/setup.bash; "
            f"test -f {workspace}/devel_mda319/setup.bash && source {workspace}/devel_mda319/setup.bash; "
            "export ROS_MASTER_URI=${ROS_MASTER_URI:-http://127.0.0.1:11311}; "
            "export ROS_IP=${ROS_IP:-127.0.0.1}; "
            "export ROS_HOSTNAME=${ROS_HOSTNAME:-localhost}; "
        )
        argv = [
            str(args_cli.kuavo_ik_python),
            container_script,
            "--host",
            str(args_cli.kuavo_ik_host),
            "--port",
            str(args_cli.kuavo_ik_port),
            "--service",
            str(args_cli.kuavo_ik_service),
            "--fk-service",
            str(args_cli.kuavo_ik_fk_service),
        ]
        cmd = setup + "exec " + " ".join("'" + part.replace("'", "'\"'\"'") + "'" for part in argv)
        log_file = (log_dir / "docker_bridge.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            ["docker", "exec", container, "bash", "-lc", cmd],
            cwd=str(WORKSPACE_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_file.close()
        return True

    def _reset_docker_ik_node(self) -> bool:
        container = str(args_cli.kuavo_ik_docker_container)
        workspace = str(args_cli.kuavo_ik_docker_workspace).rstrip("/")
        log_dir = self.run_dir / "kuavo_ik_sidecar"
        log_dir.mkdir(parents=True, exist_ok=True)
        robot_version = int(args_cli.kuavo_ik_robot_version)
        model_version = 14 if robot_version == 15 else robot_version
        service_name = str(args_cli.kuavo_ik_service)
        reset_cmd = f"""
set -e
cd '{workspace}'
source /opt/ros/noetic/setup.bash
test -f '{workspace}/devel/setup.bash' && source '{workspace}/devel/setup.bash'
test -f '{workspace}/devel_mda319/setup.bash' && source '{workspace}/devel_mda319/setup.bash'
export ROS_MASTER_URI=${{ROS_MASTER_URI:-http://127.0.0.1:11311}}
export ROS_IP=${{ROS_IP:-127.0.0.1}}
export ROS_HOSTNAME=${{ROS_HOSTNAME:-localhost}}
pids=$(pgrep -x arms_ik_node || true)
if [ -n "$pids" ]; then
  kill $pids || true
  sleep 0.6
fi
rosparam set /model_path '{workspace}/src/kuavo_assets/models/biped_s{model_version}/urdf/drake/biped_v3_arm.urdf'
rosparam set /robot_version '{robot_version}'
rosparam set /control_hand_side 1
rosparam set /print_ik_info false
rosparam set /enable_ik_vis false
if [ -x '{workspace}/devel/lib/motion_capture_ik/arms_ik_node' ]; then
  ik_bin='{workspace}/devel/lib/motion_capture_ik/arms_ik_node'
elif [ -x '{workspace}/devel_mda319/.private/motion_capture_ik/lib/motion_capture_ik/arms_ik_node' ]; then
  ik_bin='{workspace}/devel_mda319/.private/motion_capture_ik/lib/motion_capture_ik/arms_ik_node'
else
  echo 'arms_ik_node binary not found' >&2
  exit 1
fi
nohup "$ik_bin" __name:=arms_ik_node >/tmp/task319_arms_ik_node_restart.log 2>&1 &
python3 - <<'PY'
import rospy
import sys
rospy.init_node('task319_wait_arms_ik_node', anonymous=True, disable_signals=True)
try:
    rospy.wait_for_service({service_name!r}, timeout=12.0)
except Exception as exc:
    print(repr(exc), file=sys.stderr)
    sys.exit(1)
PY
"""
        log_path = log_dir / "docker_ik_node_reset.log"
        try:
            completed = subprocess.run(
                ["docker", "exec", container, "bash", "-lc", reset_cmd],
                cwd=str(WORKSPACE_ROOT),
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            log_path.write_text(
                "STDOUT:\n" + completed.stdout + "\nSTDERR:\n" + completed.stderr,
                encoding="utf-8",
            )
            if completed.returncode != 0:
                self.last_error = f"Could not reset official IK node in docker container {container}; see {log_path}"
                return False
            return True
        except Exception as exc:
            self.last_error = f"Could not reset official IK node in docker container {container}: {exc!r}"
            return False

    def _connect_once(self) -> bool:
        self.close_socket_only()
        try:
            self.sock = socket.create_connection((str(args_cli.kuavo_ik_host), int(args_cli.kuavo_ik_port)), timeout=0.5)
            self.sock.settimeout(max(0.5, float(args_cli.kuavo_ik_timeout_s)))
            self.stream = self.sock.makefile("rw", encoding="utf-8", newline="\n")
            health = self.exchange({"type": "health"})
            if bool(health.get("ok")):
                self.health = dict(health)
                return True
            self.last_error = json.dumps(health, ensure_ascii=False)
        except Exception as exc:
            self.last_error = repr(exc)
        self.close_socket_only()
        return False

    def exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.stream is None:
            raise RuntimeError("Kuavo IK bridge is not connected.")
        self.stream.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")
        self.stream.flush()
        line = self.stream.readline()
        if not line:
            raise RuntimeError("Kuavo IK bridge closed the socket.")
        return json.loads(line)

    def fk(self, label: str, q: np.ndarray | list[float]) -> dict[str, Any]:
        try:
            return self.exchange({"type": "fk", "label": label, "q": [float(v) for v in np.asarray(q, dtype=np.float32).reshape(-1)]})
        except Exception as exc:
            return {"type": "fk_result", "label": label, "success": False, "error": repr(exc)}

    def solve_right_pose(self, label: str, target_pose_b: np.ndarray, right_q: np.ndarray, q_arm_seed: np.ndarray | list[float] | None = None) -> dict[str, Any]:
        target_pose_b = np.asarray(target_pose_b, dtype=np.float32).reshape(4, 4)
        quat_wxyz = np.asarray(quat_wxyz_from_matrix(target_pose_b[:3, :3]), dtype=np.float32)
        q_right = np.asarray(right_q, dtype=np.float32).reshape(-1).tolist()
        if q_arm_seed is not None and len(np.asarray(q_arm_seed).reshape(-1)) >= 14:
            q_arm = [float(v) for v in np.asarray(q_arm_seed, dtype=np.float32).reshape(-1)[:14]]
            q_arm[7:14] = [float(v) for v in q_right[:7]]
        else:
            q_arm = [0.0] * 7 + [float(v) for v in q_right[:7]]
        payload = {
            "type": "solve",
            "label": label,
            "service": str(args_cli.kuavo_ik_service),
            "frame": int(args_cli.kuavo_ik_frame),
            "q_arm": q_arm,
            "left_pose": {"pos_xyz": [0.0, 0.25, 0.45], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "right_pose": {
                "pos_xyz": [float(v) for v in np.asarray(target_pose_b[:3, 3], dtype=np.float32).reshape(3)],
                "quat_xyzw": quat_xyzw_from_wxyz(quat_wxyz),
                "joint_angles": [float(v) for v in q_right[:7]],
            },
            "params": {
                "constraint_mode": 2,
                "pos_constraint_tol": float(args_cli.kuavo_ik_pos_tol_m),
                "oritation_constraint_tol": float(args_cli.kuavo_ik_ori_tol_rad),
                "pos_cost_weight": float(args_cli.kuavo_ik_pos_cost_weight),
                "major_iterations_limit": 100,
            },
        }
        try:
            response = self.exchange(payload)
        except Exception as exc:
            return {"label": label, "success": False, "error": repr(exc)}
        right_solution = None
        q_arm_resp = response.get("q_arm")
        if isinstance(q_arm_resp, list) and len(q_arm_resp) >= 14:
            right_solution = [float(v) for v in q_arm_resp[7:14]]
        predicted_tcp = kuavo_response_right_tcp_pose_base(response)
        return {
            "label": label,
            "success": bool(response.get("success", False)),
            "error_reason": response.get("error_reason", ""),
            "time_cost_ms": response.get("time_cost_ms"),
            "right_q": right_solution,
            "q_torso": [float(v) for v in response.get("q_torso", [])] if isinstance(response.get("q_torso"), list) else [],
            "with_torso": bool(response.get("with_torso", False)),
            "commanded_official_wrist_pose_b": target_pose_b.astype(float).tolist(),
            "predicted_tcp_pose_b": predicted_tcp.astype(float).tolist() if predicted_tcp is not None else None,
            "raw": response,
        }

    def close_socket_only(self) -> None:
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def close(self) -> None:
        try:
            if self.stream is not None:
                try:
                    self.stream.write(json.dumps({"type": "shutdown"}, separators=(",", ":")) + "\n")
                    self.stream.flush()
                except Exception:
                    pass
        finally:
            self.close_socket_only()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except Exception:
                self.process.kill()
        self.process = None


@dataclass
class SelectedTarget:
    delta_b: tuple[float, float, float]
    pose_b: np.ndarray
    pose_w: np.ndarray
    curobo_tcp_pose_b: np.ndarray
    official_solution: dict[str, Any]
    curobo_plan: CuroboPlanResult
    selection_mode: str = "strict_both_reach"


def simulate_official_solution(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    target_pose_b: np.ndarray,
    official_solution: dict[str, Any],
) -> dict[str, Any]:
    try:
        reset_robot(scene, robot)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim.get_physics_dt())
        target_pose_w = robot_base_pose_to_world_pose(robot, target_pose_b)
        joint_ids, joint_names, goal_q, control_meta = official_control_goal(robot, official_solution)
        locked_target = robot.data.default_joint_pos.clone()
        metrics = drive_joint_sequence(
            sim,
            scene,
            robot,
            joint_ids,
            goal_q.reshape(1, -1),
            locked_target,
            target_pose_w,
            interpolate_single_goal=True,
            trajectory_steps=max(1, int(args_cli.precheck_trajectory_steps)),
            hold_steps=max(0, int(args_cli.precheck_hold_steps)),
            settle_steps=max(0, int(args_cli.precheck_hold_steps)),
            tracked_pose_fn=wrist_pose_matrix,
        )
        metrics.update(
            {
                "success": bool(metrics["position_error_m"] <= float(args_cli.success_position_error_m)),
                "joint_names": list(joint_names),
                **control_meta,
            }
        )
        return metrics
    except Exception as exc:
        return {"success": False, "error": repr(exc)}
    finally:
        try:
            reset_robot(scene, robot)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim.get_physics_dt())
        except Exception:
            pass


def simulate_curobo_plan(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    target_pose_b: np.ndarray,
    plan: CuroboPlanResult,
) -> dict[str, Any]:
    try:
        reset_robot(scene, robot)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim.get_physics_dt())
        target_pose_w = robot_base_pose_to_world_pose(robot, target_pose_b)
        joint_ids = right_arm_joint_ids(robot)
        locked_target = robot.data.default_joint_pos.clone()
        metrics = drive_joint_sequence(
            sim,
            scene,
            robot,
            joint_ids,
            np.asarray(plan.joint_positions, dtype=np.float32).reshape(-1, len(joint_ids)),
            locked_target,
            target_pose_w,
            interpolate_single_goal=False,
            hold_steps=max(0, int(args_cli.precheck_hold_steps)),
            settle_steps=max(0, int(args_cli.precheck_hold_steps)),
            tracked_pose_fn=wrist_pose_matrix,
        )
        metrics.update(
            {
                "success": bool(metrics["position_error_m"] <= float(args_cli.success_position_error_m)),
                "joint_names": list(RIGHT_ARM_JOINT_NAMES),
            }
        )
        return metrics
    except Exception as exc:
        return {"success": False, "error": repr(exc)}
    finally:
        try:
            reset_robot(scene, robot)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim.get_physics_dt())
        except Exception:
            pass


def find_best_official_right_q_mapping(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    target_pose_b: np.ndarray,
    official_solution: dict[str, Any],
) -> dict[str, Any]:
    raw_q = official_solution.get("right_q")
    if not isinstance(raw_q, list) or len(raw_q) != len(RIGHT_ARM_JOINT_NAMES):
        return {"success": False, "reason": "official solution has no 7-DOF right_q"}
    raw = np.asarray(raw_q, dtype=np.float32).reshape(7)
    joint_ids = right_arm_joint_ids(robot)
    locked_target = robot.data.default_joint_pos.clone()
    target_pose_w = robot_base_pose_to_world_pose(robot, target_pose_b)
    best: dict[str, Any] | None = None
    top: list[dict[str, Any]] = []
    try:
        for signs in itertools.product((-1.0, 1.0), repeat=7):
            q = raw * np.asarray(signs, dtype=np.float32)
            reset_robot(scene, robot)
            set_joint_q_target(robot, joint_ids, q, locked_target)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            wrist_w = wrist_pose_matrix(robot)
            err = float(np.linalg.norm(wrist_w[:3, 3] - target_pose_w[:3, 3]))
            record = {
                "signs": [int(v) for v in signs],
                "position_error_m": err,
                "mapped_q": q.astype(float).tolist(),
                "wrist_world_m": wrist_w[:3, 3].astype(float).tolist(),
            }
            if best is None or err < float(best["position_error_m"]):
                best = record
            top.append(record)
    finally:
        try:
            reset_robot(scene, robot)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
        except Exception:
            pass
    top = sorted(top, key=lambda item: float(item["position_error_m"]))[:5]
    if best is None:
        return {"success": False, "reason": "no sign mapping evaluated"}
    return {
        "success": bool(float(best["position_error_m"]) <= float(args_cli.success_position_error_m)),
        "best": best,
        "top5": top,
        "raw_q": raw.astype(float).tolist(),
        "policy": "searched_all_2^7_right_arm_sign_combinations_for_isaac_joint_convention_against_wrist_pose",
    }


def select_reachable_target(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    joint_ids: list[int],
    planner: KuavoRightArmCuroboPlanner,
    ik_client: KuavoIkBridgeClient,
) -> tuple[SelectedTarget | None, list[dict[str, Any]]]:
    start_q = current_right_q(robot, joint_ids)
    start_q_arm = current_q_arm(robot)
    start_wrist_b = world_pose_to_robot_base_pose(robot, wrist_pose_matrix(robot))
    start_tcp_b = world_pose_to_robot_base_pose(robot, tcp_pose_matrix(robot))
    wrist_to_tcp = (np.linalg.inv(start_wrist_b) @ start_tcp_b).astype(np.float32)
    attempts: list[dict[str, Any]] = []
    best_demo_fallback: tuple[float, SelectedTarget] | None = None
    for delta in target_deltas():
        target_pose_b = start_wrist_b.copy()
        target_pose_b[:3, 3] += np.asarray(delta, dtype=np.float32)
        curobo_tcp_pose_b = (target_pose_b @ wrist_to_tcp).astype(np.float32)
        official_command_pose_b = target_pose_b.copy()
        official = ik_client.solve_right_pose(f"candidate_{len(attempts)}", official_command_pose_b, start_q, q_arm_seed=start_q_arm)
        predicted_wrist = kuavo_hand_pose_matrix((((official.get("raw") or {}).get("hand_poses") or {}).get("right_pose") if isinstance((official.get("raw") or {}).get("hand_poses"), dict) else None))
        predicted_wrist_error = None
        if predicted_wrist is not None:
            predicted_wrist_error = float(np.linalg.norm(predicted_wrist[:3, 3] - official_command_pose_b[:3, 3]))
        official_service_success = bool(official.get("success")) and isinstance(official.get("right_q"), list) and len(official.get("right_q") or []) == 7
        official_for_execution = dict(official)
        official_mapping: dict[str, Any] = {}
        if official_service_success:
            official_mapping = find_best_official_right_q_mapping(sim, scene, robot, target_pose_b, official)
            official_for_execution["right_q_mapping"] = official_mapping
            best_mapping = official_mapping.get("best") if isinstance(official_mapping, dict) else None
            if isinstance(best_mapping, dict) and isinstance(best_mapping.get("mapped_q"), list) and len(best_mapping["mapped_q"]) == len(RIGHT_ARM_JOINT_NAMES):
                official_for_execution["right_q"] = [float(v) for v in best_mapping["mapped_q"]]
        curobo_plan = planner.plan_to_pose(
            start_q,
            curobo_tcp_pose_b,
            max_attempts=4,
            enable_graph=True,
            timeout_s=10.0,
            check_start_validity=False,
            position_only=False,
            rotation_threshold_rad=0.35,
        )
        attempt = {
            "delta_b_m": [float(v) for v in delta],
            "target_wrist_position_b_m": target_pose_b[:3, 3].astype(float).tolist(),
            "curobo_command_tcp_position_b_m": curobo_tcp_pose_b[:3, 3].astype(float).tolist(),
            "official_command_wrist_position_b_m": official_command_pose_b[:3, 3].astype(float).tolist(),
            "official_success": official_service_success,
            "official_error_reason": str(official.get("error_reason") or official.get("error") or ""),
            "official_time_cost_ms": official.get("time_cost_ms"),
            "official_predicted_wrist_error_m": predicted_wrist_error,
            "official_with_torso": bool(official.get("with_torso")),
            "official_right_q_mapping": official_mapping,
            "curobo_success": bool(curobo_plan.success),
            "curobo_reason": curobo_plan.reason,
            "curobo_plan_steps": int(curobo_plan.joint_positions.shape[0]) if curobo_plan.joint_positions is not None else 0,
        }
        if attempt["official_success"] and attempt["curobo_success"] and curobo_plan.joint_positions.shape[0] > 0:
            official_sim = simulate_official_solution(sim, scene, robot, target_pose_b, official_for_execution)
            curobo_sim = simulate_curobo_plan(sim, scene, robot, target_pose_b, curobo_plan)
            attempt["official_sim_precheck"] = official_sim
            attempt["curobo_sim_precheck"] = curobo_sim
            attempts.append(attempt)
            if bool(curobo_sim.get("success")):
                demo_score = float(official_sim.get("position_error_m", float("inf")))
                demo_target = SelectedTarget(
                    delta_b=delta,
                    pose_b=target_pose_b,
                    pose_w=robot_base_pose_to_world_pose(robot, target_pose_b),
                    curobo_tcp_pose_b=curobo_tcp_pose_b,
                    official_solution=official_for_execution,
                    curobo_plan=curobo_plan,
                    selection_mode="demo_fallback_curobo_reaches_official_right_arm_only_fails",
                )
                if best_demo_fallback is None or demo_score < best_demo_fallback[0]:
                    best_demo_fallback = (demo_score, demo_target)
            if bool(official_sim.get("success")) and bool(curobo_sim.get("success")):
                return (
                    SelectedTarget(
                        delta_b=delta,
                        pose_b=target_pose_b,
                        pose_w=robot_base_pose_to_world_pose(robot, target_pose_b),
                        curobo_tcp_pose_b=curobo_tcp_pose_b,
                        official_solution=official_for_execution,
                        curobo_plan=curobo_plan,
                        selection_mode="strict_both_reach",
                    ),
                    attempts,
                )
        else:
            attempts.append(attempt)
    if best_demo_fallback is not None:
        return best_demo_fallback[1], attempts
    return None, attempts


def make_markers(label: str) -> tuple[VisualizationMarkers, VisualizationMarkers]:
    frame_cfg = FRAME_MARKER_CFG.copy()
    frame_cfg.markers["frame"].scale = (0.075, 0.075, 0.075)
    current_marker = VisualizationMarkers(frame_cfg.replace(prim_path=f"/Visuals/{label}_tcp_current"))
    goal_marker = VisualizationMarkers(frame_cfg.replace(prim_path=f"/Visuals/{label}_tcp_goal"))
    return current_marker, goal_marker


def official_control_goal(robot: Articulation, official_solution: dict[str, Any]) -> tuple[list[int], list[str], np.ndarray, dict[str, Any]]:
    names: list[str] = []
    values: list[float] = []
    q_torso = [float(v) for v in (official_solution.get("q_torso") or [])]
    applied_torso: dict[str, float] = {}
    if bool(args_cli.official_apply_upper_body) and bool(official_solution.get("with_torso")) and len(q_torso) >= len(KUAVO_IK_TORSO_JOINT_NAMES):
        torso_by_name = dict(zip(KUAVO_IK_TORSO_JOINT_NAMES, q_torso[: len(KUAVO_IK_TORSO_JOINT_NAMES)], strict=True))
        for name in KUAVO_IK_TORSO_JOINT_NAMES:
            names.append(name)
            values.append(float(torso_by_name[name]))
            applied_torso[name] = float(torso_by_name[name])
    right_q = official_solution.get("right_q")
    if not isinstance(right_q, list) or len(right_q) != len(RIGHT_ARM_JOINT_NAMES):
        raise RuntimeError("Official IK did not return a 7-DOF right-arm solution.")
    names.extend(RIGHT_ARM_JOINT_NAMES)
    values.extend(float(v) for v in right_q)
    joint_ids, resolved = robot.find_joints(names, preserve_order=True)
    if len(joint_ids) != len(names) or list(resolved) != names:
        raise RuntimeError(f"Could not resolve official control joints: requested={names}, resolved={resolved}.")
    return list(joint_ids), names, np.asarray(values, dtype=np.float32), {
        "official_execution_scope": "upper_body" if bool(applied_torso) else "right_arm_only",
        "official_apply_upper_body": bool(args_cli.official_apply_upper_body),
        "official_with_torso": bool(official_solution.get("with_torso")),
        "applied_torso": applied_torso,
        "q_torso_order": list(KUAVO_IK_TORSO_JOINT_NAMES),
        "q_arm_order": list(KUAVO_IK_Q_ARM_JOINT_NAMES),
    }


def drive_joint_sequence(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    joint_ids: list[int],
    waypoints: np.ndarray,
    locked_target: torch.Tensor,
    target_pose_w: np.ndarray,
    *,
    recorder: ObserverVideoRecorder | None = None,
    current_marker: VisualizationMarkers | None = None,
    goal_marker: VisualizationMarkers | None = None,
    interpolate_single_goal: bool = False,
    trajectory_steps: int | None = None,
    hold_steps: int | None = None,
    settle_steps: int | None = None,
    tracked_pose_fn: Any = tcp_pose_matrix,
) -> dict[str, Any]:
    waypoints = np.asarray(waypoints, dtype=np.float32).reshape(-1, len(joint_ids))
    previous = current_joint_q(robot, joint_ids)
    start_q = previous.copy()
    max_step = max(1.0e-5, float(args_cli.max_joint_step))
    max_command_step = 0.0
    steps_executed = 0
    min_tcp_obstacle_signed_distance = math.inf
    min_tracked_obstacle_signed_distance = math.inf

    def update_obstacle_metrics() -> None:
        nonlocal min_tcp_obstacle_signed_distance, min_tracked_obstacle_signed_distance
        if not bool(args_cli.obstacle_demo):
            return
        center_w = np.asarray(active_obstacle_pos_b(), dtype=np.float32)
        size = active_obstacle_size_b()
        tcp_w = tcp_pose_matrix(robot)
        tracked_w = tracked_pose_fn(robot)
        min_tcp_obstacle_signed_distance = min(
            min_tcp_obstacle_signed_distance,
            signed_distance_point_to_aabb(tcp_w[:3, 3], center_w, size),
        )
        min_tracked_obstacle_signed_distance = min(
            min_tracked_obstacle_signed_distance,
            signed_distance_point_to_aabb(tracked_w[:3, 3], center_w, size),
        )

    update_obstacle_metrics()

    def command_until(desired: np.ndarray) -> None:
        nonlocal previous, max_command_step, steps_executed
        desired = np.asarray(desired, dtype=np.float32).reshape(len(joint_ids))
        while True:
            command = clamp_joint_step(desired, previous, max_step)
            max_command_step = max(max_command_step, float(np.max(np.abs(command - previous))))
            set_joint_q_target(robot, joint_ids, command, locked_target)
            step_scene(sim, scene, recorder, current_marker, goal_marker, target_pose_w, tracked_pose_fn)
            update_obstacle_metrics()
            steps_executed += 1
            previous = command
            if float(np.max(np.abs(desired - previous))) <= 1.0e-5:
                break

    if bool(interpolate_single_goal):
        goal_q = waypoints[-1]
        total_steps = max(1, int(args_cli.trajectory_steps if trajectory_steps is None else trajectory_steps))
        for idx in range(total_steps):
            alpha = min_jerk((idx + 1) / total_steps)
            command_until((1.0 - alpha) * start_q + alpha * goal_q)
    else:
        for waypoint in waypoints:
            command_until(waypoint)

    final_command = previous.copy()
    for _ in range(max(0, int(args_cli.hold_steps if hold_steps is None else hold_steps))):
        set_joint_q_target(robot, joint_ids, final_command, locked_target)
        step_scene(sim, scene, recorder, current_marker, goal_marker, target_pose_w, tracked_pose_fn)
        update_obstacle_metrics()

    settle_executed = 0
    joint_error_threshold = float(args_cli.joint_error_threshold_rad)
    for _ in range(max(0, int(args_cli.settle_steps if settle_steps is None else settle_steps))):
        set_joint_q_target(robot, joint_ids, final_command, locked_target)
        step_scene(sim, scene, recorder, current_marker, goal_marker, target_pose_w, tracked_pose_fn)
        update_obstacle_metrics()
        settle_executed += 1
        actual = current_joint_q(robot, joint_ids)
        if float(np.max(np.abs(actual - final_command))) <= joint_error_threshold:
            break

    final_tcp_w = tracked_pose_fn(robot)
    actual_q = current_joint_q(robot, joint_ids)
    joint_error = float(np.max(np.abs(actual_q - final_command))) if actual_q.size else 0.0
    pos_error = float(np.linalg.norm(final_tcp_w[:3, 3] - target_pose_w[:3, 3]))
    metrics = {
        "position_error_m": pos_error,
        "steps_executed": int(steps_executed),
        "settle_executed_steps": int(settle_executed),
        "max_command_step_rad": float(max_command_step),
        "final_command_q": final_command.astype(float).tolist(),
        "final_actual_q": actual_q.astype(float).tolist(),
        "final_joint_tracking_error_rad": joint_error,
        "final_tcp_world_m": final_tcp_w[:3, 3].astype(float).tolist(),
        "target_tcp_world_m": target_pose_w[:3, 3].astype(float).tolist(),
    }
    if bool(args_cli.obstacle_demo):
        min_tcp = float(min_tcp_obstacle_signed_distance)
        min_tracked = float(min_tracked_obstacle_signed_distance)
        metrics["obstacle_clearance"] = {
            "aabb_center_world_m": [float(v) for v in active_obstacle_pos_b()],
            "aabb_size_m": [float(v) for v in active_obstacle_size_b()],
            "min_tcp_aabb_signed_distance_m": min_tcp,
            "min_tracked_aabb_signed_distance_m": min_tracked,
            "tcp_entered_aabb": bool(min_tcp < 0.0),
            "tracked_pose_entered_aabb": bool(min_tracked < 0.0),
        }
    return metrics


def warmup_scene(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    joint_ids: list[int],
    locked_target: torch.Tensor,
    recorder: ObserverVideoRecorder | None,
    target_pose_w: np.ndarray,
    current_marker: VisualizationMarkers,
    goal_marker: VisualizationMarkers,
    tracked_pose_fn: Any = tcp_pose_matrix,
) -> None:
    q = current_joint_q(robot, joint_ids)
    for _ in range(max(0, int(args_cli.warmup_steps))):
        set_joint_q_target(robot, joint_ids, q, locked_target)
        step_scene(sim, scene, recorder, current_marker, goal_marker, target_pose_w, tracked_pose_fn)


def execute_official_ik(
    sim: SimulationContext,
    scene: InteractiveScene,
    run_dir: Path,
    target: SelectedTarget,
) -> dict[str, Any]:
    robot: Articulation = scene["robot"]
    locked_target = robot.data.default_joint_pos.clone()
    recorder = ObserverVideoRecorder(run_dir, "official_ik_reach")
    current_marker, goal_marker = make_markers("official_ik")
    reset_robot(scene, robot)
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(sim.get_physics_dt())
    target_pose_w = robot_base_pose_to_world_pose(robot, target.pose_b)
    joint_ids, joint_names, goal_q, control_meta = official_control_goal(robot, target.official_solution)
    warmup_scene(sim, scene, robot, joint_ids, locked_target, recorder, target_pose_w, current_marker, goal_marker, wrist_pose_matrix)
    metrics = drive_joint_sequence(
        sim,
        scene,
        robot,
        joint_ids,
        goal_q.reshape(1, -1),
        locked_target,
        target_pose_w,
        recorder=recorder,
        current_marker=current_marker,
        goal_marker=goal_marker,
        interpolate_single_goal=True,
        trajectory_steps=max(1, int(args_cli.trajectory_steps)),
        hold_steps=max(0, int(args_cli.hold_steps)),
        settle_steps=max(0, int(args_cli.settle_steps)),
        tracked_pose_fn=wrist_pose_matrix,
    )
    pos_error = float(metrics["position_error_m"])
    recorder.finalize()
    return {
        "backend": "official_kuavo_ik",
        "success": bool(pos_error <= float(args_cli.success_position_error_m)),
        "position_error_m": pos_error,
        "steps_executed": int(metrics["steps_executed"]),
        "settle_executed_steps": int(metrics["settle_executed_steps"]),
        "max_command_step_rad": float(metrics["max_command_step_rad"]),
        "final_tcp_world_m": metrics["final_tcp_world_m"],
        "target_tcp_world_m": metrics["target_tcp_world_m"],
        "commanded_joint_names": list(joint_names),
        "commanded_joint_q": metrics["final_command_q"],
        "actual_joint_q": metrics["final_actual_q"],
        "final_joint_tracking_error_rad": float(metrics["final_joint_tracking_error_rad"]),
        "obstacle_clearance": metrics.get("obstacle_clearance"),
        "video": str(recorder.path) if recorder.enabled else None,
        "video_frames": int(recorder.frame_count),
        "video_error": recorder.error,
        "tracked_pose": "zarm_r7_end_effector",
        "final_wrist_world_m": metrics["final_tcp_world_m"],
        "target_wrist_world_m": metrics["target_tcp_world_m"],
        "ik_solution": {
            "time_cost_ms": target.official_solution.get("time_cost_ms"),
            "error_reason": target.official_solution.get("error_reason", ""),
            "predicted_tcp_pose_b": target.official_solution.get("predicted_tcp_pose_b"),
            "right_q_mapping": target.official_solution.get("right_q_mapping"),
            **control_meta,
        },
    }


def execute_curobo_plan(
    sim: SimulationContext,
    scene: InteractiveScene,
    run_dir: Path,
    target: SelectedTarget,
) -> dict[str, Any]:
    robot: Articulation = scene["robot"]
    joint_ids = right_arm_joint_ids(robot)
    locked_target = robot.data.default_joint_pos.clone()
    recorder = ObserverVideoRecorder(run_dir, "curobo_reach")
    current_marker, goal_marker = make_markers("curobo")
    reset_robot(scene, robot)
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(sim.get_physics_dt())
    target_pose_w = robot_base_pose_to_world_pose(robot, target.pose_b)
    warmup_scene(sim, scene, robot, joint_ids, locked_target, recorder, target_pose_w, current_marker, goal_marker, wrist_pose_matrix)
    metrics = drive_joint_sequence(
        sim,
        scene,
        robot,
        joint_ids,
        np.asarray(target.curobo_plan.joint_positions, dtype=np.float32).reshape(-1, len(RIGHT_ARM_JOINT_NAMES)),
        locked_target,
        target_pose_w,
        recorder=recorder,
        current_marker=current_marker,
        goal_marker=goal_marker,
        interpolate_single_goal=False,
        hold_steps=max(0, int(args_cli.hold_steps)),
        settle_steps=max(0, int(args_cli.settle_steps)),
        tracked_pose_fn=wrist_pose_matrix,
    )
    pos_error = float(metrics["position_error_m"])
    recorder.finalize()
    return {
        "backend": "curobo_right_arm",
        "success": bool(pos_error <= float(args_cli.success_position_error_m)),
        "position_error_m": pos_error,
        "steps_executed": int(metrics["steps_executed"]),
        "settle_executed_steps": int(metrics["settle_executed_steps"]),
        "max_command_step_rad": float(metrics["max_command_step_rad"]),
        "final_tcp_world_m": metrics["final_tcp_world_m"],
        "target_tcp_world_m": metrics["target_tcp_world_m"],
        "commanded_joint_names": list(RIGHT_ARM_JOINT_NAMES),
        "commanded_joint_q": metrics["final_command_q"],
        "actual_joint_q": metrics["final_actual_q"],
        "final_joint_tracking_error_rad": float(metrics["final_joint_tracking_error_rad"]),
        "obstacle_clearance": metrics.get("obstacle_clearance"),
        "video": str(recorder.path) if recorder.enabled else None,
        "video_frames": int(recorder.frame_count),
        "video_error": recorder.error,
        "tracked_pose": "zarm_r7_end_effector",
        "final_wrist_world_m": metrics["final_tcp_world_m"],
        "target_wrist_world_m": metrics["target_tcp_world_m"],
        "curobo_command_tcp_pose_base": target.curobo_tcp_pose_b.astype(float).tolist(),
        "curobo_plan": {
            "reason": target.curobo_plan.reason,
            "plan_steps": int(target.curobo_plan.joint_positions.shape[0]),
            "metadata": target.curobo_plan.metadata,
        },
    }


def main() -> int:
    run_dir = Path(args_cli.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "success": False,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args_cli).items() if k != "kit_args"},
    }
    ik_client: KuavoIkBridgeClient | None = None
    try:
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 120.0)
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view(list(args_cli.observer_camera_pos), list(args_cli.observer_camera_target))
        scene_cfg = IkReachCompareSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        robot: Articulation = scene["robot"]
        joint_ids = right_arm_joint_ids(robot)
        reset_robot(scene, robot)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim.get_physics_dt())

        for _ in range(max(1, int(args_cli.warmup_steps))):
            robot.set_joint_position_target(robot.data.default_joint_pos.clone())
            step_scene(sim, scene)

        ik_client = KuavoIkBridgeClient(run_dir)
        if not ik_client.start():
            summary.update({"success": False, "reason": f"Kuavo IK bridge unavailable: {ik_client.last_error}"})
            (run_dir / "ik_reach_compare_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 2
        summary["kuavo_ik_bridge_health"] = ik_client.health

        world_model = curobo_obstacle_world_model(robot)
        summary["obstacle_demo"] = obstacle_pose_summary(robot)
        planner = KuavoRightArmCuroboPlanner(
            KUAVO_BASE_URDF,
            GRIPPER_URDF,
            tcp_offset_m=tuple(float(v) for v in GRIPPER_LOCAL_TCP_OFFSET),
            world_model=world_model,
            device=str(args_cli.curobo_device),
            warmup=True,
            use_cuda_graph=False,
            collision_activation_distance_m=float(args_cli.curobo_collision_activation_distance_m),
        )
        selected, attempts = select_reachable_target(sim, scene, robot, joint_ids, planner, ik_client)
        summary["target_selection_attempts"] = attempts
        if selected is None:
            summary.update({"success": False, "reason": "No target delta was reachable by both official IK and cuRobo."})
            (run_dir / "ik_reach_compare_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 3

        summary["selected_target"] = {
            "selection_mode": selected.selection_mode,
            "delta_b_m": [float(v) for v in selected.delta_b],
            "target_wrist_pose_base": selected.pose_b.astype(float).tolist(),
            "target_wrist_pose_world": selected.pose_w.astype(float).tolist(),
            "curobo_command_tcp_pose_base": selected.curobo_tcp_pose_b.astype(float).tolist(),
        }
        summary["official_ik"] = execute_official_ik(sim, scene, run_dir, selected)
        summary["curobo"] = execute_curobo_plan(sim, scene, run_dir, selected)
        summary["strict_both_reach_success"] = bool(summary["official_ik"]["success"] and summary["curobo"]["success"])
        summary["success"] = bool(summary["strict_both_reach_success"] or (selected.selection_mode != "strict_both_reach" and summary["curobo"]["success"]))
        if not summary["strict_both_reach_success"]:
            summary["reason"] = (
                "Demo fallback: cuRobo reached the target with obstacle constraints; official right-arm-only IK "
                "did not meet the final wrist error threshold."
                if selected.selection_mode != "strict_both_reach"
                else "At least one backend exceeded the final wrist error threshold."
            )
        out_json = run_dir / "ik_reach_compare_summary.json"
        out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary["success"] else 4
    finally:
        if ik_client is not None:
            ik_client.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
