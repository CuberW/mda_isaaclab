"""Simple Kuavo right-arm top-down cube grasp test.

This intentionally bypasses the Task 319 sorting state machine.  It keeps only:

- Kuavo fixed-base robot with the attached parallel gripper.
- A small rectangular cuboid on a table.
- PCA-derived short-axis jaw alignment.
- cuRobo right-arm motion generation / Cartesian IK.
- Optional Kuavo official IK socket audit/seed.
- Physical lift verification.

Usage:
    python task_319_garbage_sort/topdown_cube_grasp_test.py --record_video
"""

from __future__ import annotations

import argparse
import json
import math
import socket
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
        raise argparse.ArgumentTypeError(f"Expected x,y,z triplet, got: {value!r}")
    return (parts[0], parts[1], parts[2])


parser = argparse.ArgumentParser(description="Task 319 simple top-down physical cube grasp test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--debug_cube_simple_topdown", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--object_size", type=parse_xyz, default=(0.050, 0.032, 0.040), help="Cuboid dimensions sx,sy,sz in meters.")
parser.add_argument("--object_pos", type=parse_xyz, default=(0.70, -0.16, 0.560), help="Cuboid center world x,y,z in meters.")
parser.add_argument("--object_yaw_deg", type=float, default=25.0, help="Cuboid yaw angle on the table.")
parser.add_argument("--random_yaw", action="store_true", help="Randomize object yaw for each trial.")
parser.add_argument("--num_trials", type=int, default=1)
parser.add_argument("--table_size", type=parse_xyz, default=(0.70, 0.50, 0.060), help="Table dimensions sx,sy,sz in meters.")
parser.add_argument("--table_surface_z", type=float, default=0.540)
parser.add_argument("--object_mass", type=float, default=0.035)
parser.add_argument("--max_object_grasp_width_m", type=float, default=0.085, help="Reject objects whose XY grasp span exceeds this.")
parser.add_argument("--grasp_clearance_m", type=float, default=0.010)
parser.add_argument("--closed_width_m", type=float, default=-0.012, help="Unclamped close command; negative overclose increases simulated grip.")
parser.add_argument("--hover_height_m", type=float, default=0.120)
parser.add_argument("--lift_height_m", type=float, default=0.140)
parser.add_argument("--lift_success_height_m", type=float, default=0.025)
parser.add_argument("--max_object_tcp_distance_m", type=float, default=0.120)
parser.add_argument("--hover_xy_tolerance_m", type=float, default=0.018)
parser.add_argument("--max_joint_step", type=float, default=0.018)
parser.add_argument("--warmup_steps", type=int, default=80)
parser.add_argument("--open_steps", type=int, default=30)
parser.add_argument("--close_steps", type=int, default=110)
parser.add_argument("--settle_steps", type=int, default=80)
parser.add_argument("--hold_steps", type=int, default=80)
parser.add_argument("--cartesian_step_m", type=float, default=0.003)
parser.add_argument("--curobo_device", default="cuda:0")
parser.add_argument("--curobo_position_only_hover", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--position_only_cartesian_fallback", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--kuavo_ik_audit", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--kuavo_ik_host", default="127.0.0.1")
parser.add_argument("--kuavo_ik_port", type=int, default=31975)
parser.add_argument("--kuavo_ik_timeout_s", type=float, default=2.5)
parser.add_argument("--use_kuavo_seed", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--record_video", action="store_true")
parser.add_argument("--save_video_frames", action="store_true")
parser.add_argument("--video_width", type=int, default=1280)
parser.add_argument("--video_height", type=int, default=720)
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--video_sample_stride", type=int, default=4)
parser.add_argument("--observer_camera_pos", type=parse_xyz, default=(1.45, -1.80, 1.55))
parser.add_argument("--observer_camera_target", type=parse_xyz, default=(0.65, -0.16, 0.68))
parser.add_argument("--output_dir", type=Path, default=WORKSPACE_ROOT / "task_319_garbage_sort/output/topdown_cube_grasp_tests")
parser.add_argument("--exit_after_trials", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat

from task_319_garbage_sort.curobo_right_arm import (
    RIGHT_ARM_CARRY_CONFIG,
    RIGHT_ARM_JOINT_NAMES,
    CuroboPlanResult,
    KuavoRightArmCuroboPlanner,
    build_world_cuboids,
)
from task_319_garbage_sort.grasp_pipeline.execution.gripper_control import AttachedParallelGripper, GripperLimits
from task_319_garbage_sort.gripper_robot_urdf import ensure_kuavo_with_gripper_urdf


ROOT_DIR = WORKSPACE_ROOT
GRIPPER_URDF = ROOT_DIR / "task_319_garbage_sort/two_finger_gripper.urdf"
KUAVO_BASE_URDF = ROOT_DIR / "kuavo-ros-opensource/src/kuavo_assets/models/biped_s62/urdf/biped_s62.urdf"
KUAVO_WITH_GRIPPER_URDF = ensure_kuavo_with_gripper_urdf(KUAVO_BASE_URDF, GRIPPER_URDF)

ROBOT_ROOT_Z_ON_WHEELS = 0.0
RIGHT_ARM_JOINT_EXPR = "zarm_r[1-7]_joint"
RIGHT_EE_BODY_CANDIDATES = ("gripper_base", "zarm_r7_end_effector")
GRIPPER_LOCAL_TCP_OFFSET = np.array([0.115, 0.0, 0.0], dtype=np.float32)
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
TABLE_SURFACE_Z = float(args_cli.table_surface_z)
TABLE_SIZE = tuple(float(v) for v in args_cli.table_size)
TABLE_CENTER = (float(args_cli.object_pos[0]), float(args_cli.object_pos[1]), TABLE_SURFACE_Z - 0.5 * TABLE_SIZE[2])
OBJECT_SIZE = tuple(float(v) for v in args_cli.object_size)
OBJECT_POS = tuple(float(v) for v in args_cli.object_pos)


def rigid_props(*, kinematic: bool = False) -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        kinematic_enabled=bool(kinematic),
        disable_gravity=bool(kinematic),
        linear_damping=0.08,
        angular_damping=0.12,
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
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.78)


def yaw_quat_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def kuavo_robot_cfg() -> ArticulationCfg:
    right_home = dict(zip(RIGHT_ARM_JOINT_NAMES, RIGHT_ARM_CARRY_CONFIG, strict=True))
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
                "zarm_l2_joint": 0.30,
                "zarm_l4_joint": -0.60,
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
                effort_limit_sim=450.0,
                velocity_limit_sim=0.1,
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
                effort_limit_sim=260.0,
                velocity_limit_sim=2.6,
                stiffness=1100.0,
                damping=120.0,
            ),
            "locked_head": ImplicitActuatorCfg(
                joint_names_expr=["zhead_[12]_joint"],
                effort_limit_sim=60.0,
                velocity_limit_sim=0.1,
                stiffness=900.0,
                damping=120.0,
            ),
            "right_gripper": ImplicitActuatorCfg(
                joint_names_expr=[".*finger_joint"],
                effort_limit_sim=650.0,
                velocity_limit_sim=0.35,
                stiffness=12000.0,
                damping=850.0,
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
class TopDownCubeGraspSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(5.0, 5.0),
            physics_material=material(1.3, 1.0),
            color=(0.31, 0.32, 0.31),
        ),
    )
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2400.0, color=(0.82, 0.86, 0.90)))
    robot: ArticulationCfg = kuavo_robot_cfg()
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TableTop",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=rigid_props(kinematic=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=80.0),
            collision_props=collision_props(0.003),
            physics_material=material(1.0, 0.8),
            visual_material=surface((0.48, 0.36, 0.24)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_CENTER),
    )
    grasp_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspCube",
        spawn=sim_utils.CuboidCfg(
            size=OBJECT_SIZE,
            rigid_props=rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(args_cli.object_mass)),
            collision_props=collision_props(0.002),
            physics_material=material(2.2, 1.7),
            visual_material=surface((0.92, 0.16, 0.08)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_POS, rot=yaw_quat_wxyz(math.radians(float(args_cli.object_yaw_deg)))),
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
        self.run_dir = run_dir
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


def apply_gripper_contact_material(num_envs: int) -> dict[str, Any]:
    from pxr import Usd, UsdPhysics

    stage = sim_utils.get_current_stage()
    material_path = "/World/physicsScene/task319_topdown_gripper_rubber"
    cfg = material(2.3, 1.7)
    cfg.func(material_path, cfg)
    name_tokens = ("gripper_base", "left_finger", "right_finger")
    matched = 0
    collision_candidates = 0
    applied = 0
    for env_id in range(int(num_envs)):
        root_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Kuavo62")
        if not root_prim.IsValid():
            continue
        for prim in Usd.PrimRange(root_prim):
            prim_path = str(prim.GetPath())
            if prim.IsInstanceProxy() or not any(token in prim_path for token in name_tokens):
                continue
            matched += 1
            lower_path = prim_path.casefold()
            if not (prim.HasAPI(UsdPhysics.CollisionAPI) or "/collisions" in lower_path or "collision" in lower_path):
                continue
            collision_candidates += 1
            sim_utils.define_collision_properties(prim_path, collision_props(0.0025, 0.0), stage=stage)
            sim_utils.bind_physics_material(prim_path, material_path, stage=stage)
            applied += 1
    return {
        "material_path": material_path,
        "matched_prims": int(matched),
        "collision_candidates": int(collision_candidates),
        "applied_prims": int(applied),
        "static_friction": 2.3,
        "dynamic_friction": 1.7,
    }


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


def resolve_ee_body_id(robot: Articulation) -> int:
    body_names = getattr(robot, "body_names", None) or []
    for name in RIGHT_EE_BODY_CANDIDATES:
        if name in body_names:
            return int(body_names.index(name))
    raise RuntimeError(f"Cannot find any right end-effector body from {RIGHT_EE_BODY_CANDIDATES}.")


def tcp_pose_matrix(robot: Articulation) -> np.ndarray:
    gripper_base_pose = body_pose_matrix_by_name(robot, "gripper_base")
    if gripper_base_pose is None:
        raise RuntimeError("Cannot resolve gripper_base body for TCP pose.")
    tcp_pose = gripper_base_pose.copy()
    tcp_pose[:3, 3] = gripper_base_pose[:3, 3] + gripper_base_pose[:3, :3] @ GRIPPER_LOCAL_TCP_OFFSET
    return tcp_pose.astype(np.float32)


def world_pose_to_robot_base_pose(robot: Articulation, pose_w: np.ndarray) -> np.ndarray:
    return (np.linalg.inv(robot_root_pose_matrix(robot)) @ np.asarray(pose_w, dtype=np.float32).reshape(4, 4)).astype(np.float32)


def pose_matrix(pos_w: np.ndarray, rot_w: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.asarray(pos_w, dtype=np.float32).reshape(3)
    pose[:3, :3] = np.asarray(rot_w, dtype=np.float32).reshape(3, 3)
    return pose


def quat_xyzw_from_wxyz(quat: np.ndarray | list[float] | tuple[float, ...]) -> list[float]:
    q = np.asarray(quat, dtype=np.float32).reshape(4)
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def yaw_rot_z(yaw_rad: float) -> np.ndarray:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def cuboid_sample_points_xy(center: np.ndarray, size: tuple[float, float, float], yaw_rad: float) -> np.ndarray:
    sx, sy, sz = [float(v) for v in size]
    xs = np.linspace(-0.5 * sx, 0.5 * sx, 5)
    ys = np.linspace(-0.5 * sy, 0.5 * sy, 5)
    zs = np.array([-0.5 * sz, 0.0, 0.5 * sz], dtype=np.float32)
    local = np.asarray([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)
    return center.reshape(1, 3) + local @ yaw_rot_z(yaw_rad).T


@dataclass
class GraspGeometry:
    short_axis_xy: np.ndarray
    long_axis_xy: np.ndarray
    short_extent_m: float
    long_extent_m: float
    jaw_open_width_m: float
    target_depth_m: float
    tcp_rotation_w: np.ndarray
    hover_pose_w: np.ndarray
    grasp_pose_w: np.ndarray
    lift_pose_w: np.ndarray
    metadata: dict[str, Any]


def estimate_topdown_grasp_geometry(object_center_w: np.ndarray, object_size: tuple[float, float, float], yaw_rad: float) -> GraspGeometry:
    points = cuboid_sample_points_xy(object_center_w, object_size, yaw_rad)
    xy = points[:, :2]
    centered = xy - np.mean(xy, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    long_axis = eigvecs[:, 0].astype(np.float32)
    short_axis = eigvecs[:, 1].astype(np.float32)
    if np.linalg.norm(long_axis) < 1.0e-6 or np.linalg.norm(short_axis) < 1.0e-6:
        rot = yaw_rot_z(yaw_rad)
        if object_size[0] >= object_size[1]:
            long_axis = rot[:2, 0].astype(np.float32)
            short_axis = rot[:2, 1].astype(np.float32)
        else:
            long_axis = rot[:2, 1].astype(np.float32)
            short_axis = rot[:2, 0].astype(np.float32)
    long_axis /= max(float(np.linalg.norm(long_axis)), 1.0e-8)
    short_axis /= max(float(np.linalg.norm(short_axis)), 1.0e-8)

    projections_short = centered @ short_axis.reshape(2, 1)
    projections_long = centered @ long_axis.reshape(2, 1)
    short_extent = float(np.max(projections_short) - np.min(projections_short))
    long_extent = float(np.max(projections_long) - np.min(projections_long))
    if short_extent > long_extent:
        short_axis, long_axis = long_axis.copy(), short_axis.copy()
        short_extent, long_extent = long_extent, short_extent

    if short_axis[1] < 0.0:
        short_axis = -short_axis
    long_axis = np.array([-short_axis[1], short_axis[0]], dtype=np.float32)

    sx, sy, sz = [float(v) for v in object_size]
    top_z = float(object_center_w[2] + 0.5 * sz)
    if sz > 0.024:
        target_depth = min(max(0.60 * sz, 0.018), sz - 0.012)
    else:
        target_depth = max(0.5 * sz, sz - 0.008)
    target_depth = float(max(0.0, min(target_depth, sz - 0.004)))
    grasp_pos = np.asarray([object_center_w[0], object_center_w[1], top_z - target_depth], dtype=np.float32)
    hover_pos = grasp_pos + np.array([0.0, 0.0, float(args_cli.hover_height_m)], dtype=np.float32)
    lift_pos = grasp_pos + np.array([0.0, 0.0, float(args_cli.lift_height_m)], dtype=np.float32)

    x_axis_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    y_axis_w = np.array([short_axis[0], short_axis[1], 0.0], dtype=np.float32)
    y_axis_w /= max(float(np.linalg.norm(y_axis_w)), 1.0e-8)
    z_axis_w = np.cross(x_axis_w, y_axis_w)
    z_axis_w /= max(float(np.linalg.norm(z_axis_w)), 1.0e-8)
    y_axis_w = np.cross(z_axis_w, x_axis_w)
    y_axis_w /= max(float(np.linalg.norm(y_axis_w)), 1.0e-8)
    tcp_rot_w = np.stack([x_axis_w, y_axis_w, z_axis_w], axis=1).astype(np.float32)

    jaw_open_width = min(0.110, max(0.030, float(short_extent) + float(args_cli.grasp_clearance_m)))
    metadata = {
        "object_center_w_m": object_center_w.astype(float).tolist(),
        "object_size_m": [sx, sy, sz],
        "object_yaw_deg": float(math.degrees(yaw_rad)),
        "pca_eigenvalues": eigvals.astype(float).tolist(),
        "short_axis_xy": short_axis.astype(float).tolist(),
        "long_axis_xy": long_axis.astype(float).tolist(),
        "short_extent_m": float(short_extent),
        "long_extent_m": float(long_extent),
        "jaw_open_width_m": float(jaw_open_width),
        "target_depth_m": float(target_depth),
        "tcp_local_x_axis_world": x_axis_w.astype(float).tolist(),
        "tcp_local_y_jaw_axis_world": y_axis_w.astype(float).tolist(),
        "tcp_local_z_axis_world": z_axis_w.astype(float).tolist(),
        "grasp_policy": "vertical_down_tcp_plus_x_with_jaw_axis_aligned_to_pca_short_axis",
    }
    return GraspGeometry(
        short_axis_xy=short_axis,
        long_axis_xy=long_axis,
        short_extent_m=short_extent,
        long_extent_m=long_extent,
        jaw_open_width_m=jaw_open_width,
        target_depth_m=target_depth,
        tcp_rotation_w=tcp_rot_w,
        hover_pose_w=pose_matrix(hover_pos, tcp_rot_w),
        grasp_pose_w=pose_matrix(grasp_pos, tcp_rot_w),
        lift_pose_w=pose_matrix(lift_pos, tcp_rot_w),
        metadata=metadata,
    )


class KuavoIkAuditClient:
    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.available = False
        self.health: dict[str, Any] = {}
        self.error = ""
        self._conn: socket.socket | None = None
        self._stream: Any | None = None

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._conn is None or self._stream is None:
            self._conn = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            self._conn.settimeout(self.timeout_s)
            self._stream = self._conn.makefile("rw", encoding="utf-8", newline="\n")
        stream = self._stream
        stream.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")
        stream.flush()
        line = stream.readline()
        if not line:
            raise RuntimeError("empty response from Kuavo IK socket bridge")
        return json.loads(line)

    def close(self) -> None:
        try:
            if self._stream is not None:
                try:
                    self._stream.write(json.dumps({"type": "shutdown"}, separators=(",", ":")) + "\n")
                    self._stream.flush()
                except Exception:
                    pass
                self._stream.close()
        finally:
            self._stream = None
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def probe(self) -> dict[str, Any]:
        try:
            self.health = self._request({"type": "health"})
            self.available = bool(self.health.get("ok"))
        except Exception as exc:
            self.available = False
            self.error = repr(exc)
            self.health = {"ok": False, "error": self.error}
        return self.health

    def fk(self, label: str, q: list[float]) -> dict[str, Any]:
        return self._request({"type": "fk", "label": label, "q": [float(item) for item in q]})

    def fk_mapping_audit(self, label: str, solve_response: dict[str, Any]) -> dict[str, Any]:
        q_arm = [float(item) for item in (solve_response.get("q_arm") or [])]
        q_torso = [float(item) for item in (solve_response.get("q_torso") or [])]
        audit: dict[str, Any] = {
            "q_torso_order": list(KUAVO_IK_TORSO_JOINT_NAMES),
            "q_arm_order": list(KUAVO_IK_Q_ARM_JOINT_NAMES),
            "q_torso_len": len(q_torso),
            "q_arm_len": len(q_arm),
            "with_torso": bool(solve_response.get("with_torso")),
        }
        if not bool(solve_response.get("success")):
            audit.update({"success": False, "reason": "skip FK audit because solve did not succeed"})
            return audit
        if len(q_arm) < 14:
            audit.update({"success": False, "reason": "official IK returned fewer than 14 arm joints"})
            return audit
        if bool(solve_response.get("with_torso")):
            if len(q_torso) < len(KUAVO_IK_TORSO_JOINT_NAMES):
                audit.update({"success": False, "reason": "official IK returned with_torso=true but q_torso has fewer than 4 values"})
                return audit
            q_for_fk = [*q_torso[: len(KUAVO_IK_TORSO_JOINT_NAMES)], *q_arm[:14]]
            audit["q_for_fk_order"] = [*KUAVO_IK_TORSO_JOINT_NAMES, *KUAVO_IK_Q_ARM_JOINT_NAMES]
        else:
            q_for_fk = q_arm[:14]
            audit["q_for_fk_order"] = list(KUAVO_IK_Q_ARM_JOINT_NAMES)
        audit["q_for_fk_len"] = len(q_for_fk)
        try:
            fk_response = self.fk(f"{label}_fk_audit", q_for_fk)
        except Exception as exc:
            audit.update({"success": False, "reason": repr(exc)})
            return audit
        audit["fk_response"] = fk_response
        audit["success"] = bool(fk_response.get("success"))
        if not bool(fk_response.get("success")):
            audit["reason"] = str(fk_response.get("error_reason") or fk_response.get("error") or "official FK returned success=false")
        return audit

    def solve_right_pose(self, label: str, target_pose_b: np.ndarray, right_q: np.ndarray) -> dict[str, Any]:
        if not self.available:
            return {"label": label, "available": False, "error": self.error or "Kuavo IK bridge unavailable"}
        quat_wxyz = np.asarray(quat_wxyz_from_matrix(target_pose_b[:3, :3]), dtype=np.float32)
        q_right = np.asarray(right_q, dtype=np.float32).reshape(-1).tolist()
        q_arm = [0.0] * 7 + [float(v) for v in q_right[:7]]
        payload = {
            "type": "solve",
            "label": label,
            "frame": 2,
            "q_arm": q_arm,
            "left_pose": {"pos_xyz": [0.0, 0.25, 0.45], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "right_pose": {
                "pos_xyz": [float(v) for v in np.asarray(target_pose_b[:3, 3], dtype=np.float32).reshape(3)],
                "quat_xyzw": quat_xyzw_from_wxyz(quat_wxyz),
                "joint_angles": [float(v) for v in q_right[:7]],
            },
            "params": {
                "constraint_mode": 2,
                "pos_constraint_tol": 0.008,
                "oritation_constraint_tol": 0.35,
                "pos_cost_weight": 80.0,
                "major_iterations_limit": 100,
            },
        }
        try:
            response = self._request(payload)
        except Exception as exc:
            return {"label": label, "available": True, "success": False, "error": repr(exc)}
        right_solution = None
        q_arm_resp = response.get("q_arm")
        if isinstance(q_arm_resp, list) and len(q_arm_resp) >= 14:
            right_solution = [float(v) for v in q_arm_resp[7:14]]
        fk_audit = self.fk_mapping_audit(label, response)
        return {
            "label": label,
            "available": True,
            "success": bool(response.get("success", False)),
            "error_reason": response.get("error_reason", ""),
            "time_cost_ms": response.get("time_cost_ms"),
            "right_q_seed": right_solution,
            "fk_audit": fk_audit,
            "raw": response,
        }


def reset_scene(scene: InteractiveScene, object_yaw_rad: float) -> None:
    robot: Articulation = scene["robot"]
    obj: RigidObject = scene["grasp_object"]
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.default_root_state[:, 7:]))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    object_pose = torch.tensor([[OBJECT_POS[0], OBJECT_POS[1], OBJECT_POS[2], *yaw_quat_wxyz(object_yaw_rad)]], dtype=torch.float32, device=obj.device)
    obj.write_root_pose_to_sim(object_pose)
    obj.write_root_velocity_to_sim(torch.zeros((obj.num_instances, 6), dtype=torch.float32, device=obj.device))
    scene.reset()


def right_arm_joint_ids(robot: Articulation) -> list[int]:
    joint_ids, joint_names = robot.find_joints(RIGHT_ARM_JOINT_NAMES, preserve_order=True)
    if len(joint_ids) != len(RIGHT_ARM_JOINT_NAMES):
        raise RuntimeError(f"Expected 7 right-arm joints, got {joint_names}.")
    return list(joint_ids)


def current_right_q(robot: Articulation, joint_ids: list[int]) -> np.ndarray:
    return robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().astype(np.float32)


def set_right_q_target(robot: Articulation, joint_ids: list[int], q: np.ndarray, locked_target: torch.Tensor) -> None:
    q_tensor = torch.tensor(np.asarray(q, dtype=np.float32).reshape(1, -1), dtype=torch.float32, device=robot.device)
    robot.set_joint_position_target(locked_target)
    robot.set_joint_position_target(q_tensor, joint_ids=joint_ids)


def step_scene(sim: SimulationContext, scene: InteractiveScene, recorder: ObserverVideoRecorder | None = None) -> None:
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(sim.get_physics_dt())
    if recorder is not None:
        recorder.capture(scene)


def hold_steps(
    sim: SimulationContext,
    scene: InteractiveScene,
    steps: int,
    robot: Articulation,
    joint_ids: list[int],
    q_target: np.ndarray,
    locked_target: torch.Tensor,
    gripper: AttachedParallelGripper,
    width_m: float,
    recorder: ObserverVideoRecorder | None,
) -> None:
    for _ in range(max(0, int(steps))):
        set_right_q_target(robot, joint_ids, q_target, locked_target)
        if width_m < 0.0:
            gripper.set_width_unclamped(width_m)
        else:
            gripper.set_width(width_m)
        step_scene(sim, scene, recorder)


def execute_joint_trajectory(
    sim: SimulationContext,
    scene: InteractiveScene,
    label: str,
    plan: CuroboPlanResult,
    robot: Articulation,
    joint_ids: list[int],
    locked_target: torch.Tensor,
    gripper: AttachedParallelGripper,
    width_m: float,
    recorder: ObserverVideoRecorder | None,
) -> dict[str, Any]:
    if not plan.success or plan.joint_positions.shape[0] == 0:
        return {"label": label, "success": False, "reason": plan.reason, "metadata": plan.metadata}
    previous = current_right_q(robot, joint_ids)
    max_step = max(1.0e-5, float(args_cli.max_joint_step))
    max_command_step = 0.0
    steps_executed = 0
    for q in np.asarray(plan.joint_positions, dtype=np.float32).reshape(-1, len(RIGHT_ARM_JOINT_NAMES)):
        desired = np.asarray(q, dtype=np.float32)
        while True:
            delta = np.clip(desired - previous, -max_step, max_step)
            command = previous + delta
            max_command_step = max(max_command_step, float(np.max(np.abs(delta))))
            set_right_q_target(robot, joint_ids, command, locked_target)
            if width_m < 0.0:
                gripper.set_width_unclamped(width_m)
            else:
                gripper.set_width(width_m)
            step_scene(sim, scene, recorder)
            steps_executed += 1
            previous = command
            if float(np.max(np.abs(desired - previous))) <= 1.0e-5:
                break
    return {
        "label": label,
        "success": True,
        "steps_executed": int(steps_executed),
        "max_command_step_rad": float(max_command_step),
        "final_right_q": previous.astype(float).tolist(),
        "metadata": plan.metadata,
    }


def interpolate_cartesian_poses(start_pose_w: np.ndarray, target_pose_w: np.ndarray, max_step_m: float) -> list[np.ndarray]:
    start = np.asarray(start_pose_w, dtype=np.float32).reshape(4, 4)
    target = np.asarray(target_pose_w, dtype=np.float32).reshape(4, 4)
    dist = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
    count = max(2, int(math.ceil(dist / max(1.0e-4, float(max_step_m)))))
    poses: list[np.ndarray] = []
    for idx in range(1, count + 1):
        alpha = float(idx) / float(count)
        pose = target.copy()
        pose[:3, 3] = (1.0 - alpha) * start[:3, 3] + alpha * target[:3, 3]
        poses.append(pose.astype(np.float32))
    return poses


def curobo_world_for_table(robot: Articulation):
    table_pose_w = pose_matrix(np.asarray(TABLE_CENTER, dtype=np.float32), np.eye(3, dtype=np.float32))
    table_pose_b = world_pose_to_robot_base_pose(robot, table_pose_w)
    quat = quat_wxyz_from_matrix(table_pose_b[:3, :3])
    return build_world_cuboids(
        [
            {
                "name": "topdown_test_table",
                "pose": [
                    float(table_pose_b[0, 3]),
                    float(table_pose_b[1, 3]),
                    float(table_pose_b[2, 3]),
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                ],
                "dims": [float(TABLE_SIZE[0]), float(TABLE_SIZE[1]), float(TABLE_SIZE[2] + 0.018)],
            }
        ]
    )


def maybe_plan_with_kuavo_seed(
    planner: KuavoRightArmCuroboPlanner,
    start_q: np.ndarray,
    target_pose_b: np.ndarray,
    kuavo_audit: dict[str, Any],
) -> tuple[CuroboPlanResult | None, dict[str, Any]]:
    seed = kuavo_audit.get("right_q_seed") if isinstance(kuavo_audit, dict) else None
    if not bool(args_cli.use_kuavo_seed) or not kuavo_audit.get("success") or not isinstance(seed, list) or len(seed) != 7:
        return None, {"used": False, "reason": "no_valid_kuavo_right_q_seed"}
    seed_q = np.asarray(seed, dtype=np.float32).reshape(7)
    plan = planner.plan_to_joint_positions(
        start_q,
        seed_q,
        max_attempts=3,
        enable_graph=True,
        timeout_s=8.0,
        check_start_validity=False,
    )
    if not plan.success:
        return plan, {"used": False, "attempted": True, "reason": plan.reason, "metadata": plan.metadata}
    final_fk = planner.fk_summary(plan.joint_positions[-1])
    final_pos = np.asarray(final_fk["ee_position_base_m"], dtype=np.float32)
    target_pos = np.asarray(target_pose_b[:3, 3], dtype=np.float32)
    error = float(np.linalg.norm(final_pos - target_pos))
    if error > 0.030:
        return plan, {"used": False, "attempted": True, "reason": f"seed FK target error {error:.4f}m too large", "target_error_m": error}
    return plan, {"used": True, "attempted": True, "target_error_m": error, "metadata": plan.metadata}


def run_trial(
    sim: SimulationContext,
    scene: InteractiveScene,
    planner: KuavoRightArmCuroboPlanner,
    trial_index: int,
    object_yaw_rad: float,
    run_dir: Path,
    gripper_material_meta: dict[str, Any],
) -> dict[str, Any]:
    robot: Articulation = scene["robot"]
    obj: RigidObject = scene["grasp_object"]
    joint_ids = right_arm_joint_ids(robot)
    gripper = AttachedParallelGripper(robot, limits=GripperLimits(max_width_m=0.120, default_open_width_m=0.110, grasp_extra_clearance_m=float(args_cli.grasp_clearance_m)))
    locked_target = robot.data.default_joint_pos.clone()
    recorder = ObserverVideoRecorder(run_dir, f"trial_{trial_index:02d}") if bool(args_cli.record_video) else None
    result: dict[str, Any] = {
        "trial_index": int(trial_index),
        "success": False,
        "reason": "",
        "stages": [],
        "object": {},
        "gripper_material": gripper_material_meta,
        "video": None,
    }
    start_time = time.time()
    kuavo_client: KuavoIkAuditClient | None = None
    try:
        reset_scene(scene, object_yaw_rad)
        open_width = 0.110
        for _ in range(max(1, int(args_cli.warmup_steps))):
            gripper.set_width(open_width)
            step_scene(sim, scene, recorder)

        object_center = obj.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
        object_initial_z = float(object_center[2])
        geom = estimate_topdown_grasp_geometry(object_center, OBJECT_SIZE, object_yaw_rad)
        result["object"] = {
            "initial_center_w_m": object_center.astype(float).tolist(),
            "size_m": [float(v) for v in OBJECT_SIZE],
            "mass_kg": float(args_cli.object_mass),
            "yaw_deg": float(math.degrees(object_yaw_rad)),
            "table_surface_z_m": TABLE_SURFACE_Z,
        }
        result["grasp_geometry"] = geom.metadata

        if max(geom.short_extent_m, geom.long_extent_m) > float(args_cli.max_object_grasp_width_m):
            result["reason"] = (
                f"object XY extent exceeds max_object_grasp_width_m: "
                f"{max(geom.short_extent_m, geom.long_extent_m):.3f}m > {float(args_cli.max_object_grasp_width_m):.3f}m"
            )
            return result
        if geom.jaw_open_width_m > gripper.limits.max_width_m:
            result["reason"] = f"required open width {geom.jaw_open_width_m:.3f}m exceeds gripper limit {gripper.limits.max_width_m:.3f}m"
            return result

        frame_cfg = FRAME_MARKER_CFG.copy()
        frame_cfg.markers["frame"].scale = (0.065, 0.065, 0.065)
        goal_marker = VisualizationMarkers(frame_cfg.replace(prim_path=f"/Visuals/topdown_grasp_goal_{trial_index}"))
        hover_marker = VisualizationMarkers(frame_cfg.replace(prim_path=f"/Visuals/topdown_hover_goal_{trial_index}"))
        hover_marker.visualize(
            torch.tensor(geom.hover_pose_w[:3, 3], device=robot.device).reshape(1, 3),
            torch.tensor(quat_wxyz_from_matrix(geom.hover_pose_w[:3, :3]), device=robot.device).reshape(1, 4),
        )
        goal_marker.visualize(
            torch.tensor(geom.grasp_pose_w[:3, 3], device=robot.device).reshape(1, 3),
            torch.tensor(quat_wxyz_from_matrix(geom.grasp_pose_w[:3, :3]), device=robot.device).reshape(1, 4),
        )

        for _ in range(max(1, int(args_cli.open_steps))):
            gripper.set_width(geom.jaw_open_width_m)
            step_scene(sim, scene, recorder)

        kuavo_client = KuavoIkAuditClient(args_cli.kuavo_ik_host, args_cli.kuavo_ik_port, args_cli.kuavo_ik_timeout_s) if bool(args_cli.kuavo_ik_audit) else None
        kuavo_meta: dict[str, Any] = {"enabled": bool(args_cli.kuavo_ik_audit)}
        if kuavo_client is not None:
            kuavo_meta["health"] = kuavo_client.probe()
        start_q = current_right_q(robot, joint_ids)
        hover_pose_b = world_pose_to_robot_base_pose(robot, geom.hover_pose_w)
        grasp_pose_b = world_pose_to_robot_base_pose(robot, geom.grasp_pose_w)
        lift_pose_b = world_pose_to_robot_base_pose(robot, geom.lift_pose_w)
        if kuavo_client is not None:
            kuavo_meta["hover_solve"] = kuavo_client.solve_right_pose("hover", hover_pose_b, start_q)

        seed_plan, seed_meta = maybe_plan_with_kuavo_seed(planner, start_q, hover_pose_b, kuavo_meta.get("hover_solve", {}))
        kuavo_meta["seed_plan"] = seed_meta
        if seed_plan is not None and seed_meta.get("used"):
            hover_plan = seed_plan
        else:
            hover_plan = planner.plan_to_pose(
                start_q,
                hover_pose_b,
                max_attempts=4,
                enable_graph=True,
                timeout_s=10.0,
                check_start_validity=False,
                position_only=bool(args_cli.curobo_position_only_hover),
                rotation_threshold_rad=0.35,
            )
        result["kuavo_ik_audit"] = kuavo_meta
        result["stages"].append(execute_joint_trajectory(sim, scene, "plan_hover", hover_plan, robot, joint_ids, locked_target, gripper, geom.jaw_open_width_m, recorder))
        if not hover_plan.success:
            result["reason"] = f"hover plan failed: {hover_plan.reason}"
            return result

        hold_steps(sim, scene, 20, robot, joint_ids, current_right_q(robot, joint_ids), locked_target, gripper, geom.jaw_open_width_m, recorder)
        actual_hover_tcp = tcp_pose_matrix(robot)
        hover_xy_error = float(np.linalg.norm(actual_hover_tcp[:2, 3] - np.asarray(geom.hover_pose_w[:2, 3], dtype=np.float32)))
        result["hover_error"] = {
            "xy_error_m": hover_xy_error,
            "z_error_m": float(abs(actual_hover_tcp[2, 3] - geom.hover_pose_w[2, 3])),
            "target_tcp_w_m": geom.hover_pose_w[:3, 3].astype(float).tolist(),
            "actual_tcp_w_m": actual_hover_tcp[:3, 3].astype(float).tolist(),
        }
        if hover_xy_error > float(args_cli.hover_xy_tolerance_m):
            result["reason"] = f"hover XY error {hover_xy_error:.4f}m exceeds tolerance {float(args_cli.hover_xy_tolerance_m):.4f}m"
            return result

        descend_start = tcp_pose_matrix(robot)
        descend_start[:3, :3] = geom.tcp_rotation_w
        descend_waypoints_w = interpolate_cartesian_poses(descend_start, geom.grasp_pose_w, float(args_cli.cartesian_step_m))
        descend_waypoints_b = [world_pose_to_robot_base_pose(robot, pose) for pose in descend_waypoints_w]
        descend_plan = planner.solve_ik_chain_for_poses_sequential(
            current_right_q(robot, joint_ids),
            descend_waypoints_b,
            return_seeds=64,
            max_waypoint_joint_l2_rad=1.20,
            newton_iters=40,
            position_only=False,
        )
        if not descend_plan.success and bool(args_cli.position_only_cartesian_fallback):
            result["stages"].append({"label": "descend_full_pose_failed", "success": False, "reason": descend_plan.reason, "metadata": descend_plan.metadata})
            descend_plan = planner.solve_ik_chain_for_poses_sequential(
                current_right_q(robot, joint_ids),
                descend_waypoints_b,
                return_seeds=64,
                max_waypoint_joint_l2_rad=1.20,
                newton_iters=40,
                position_only=True,
            )
        result["stages"].append(execute_joint_trajectory(sim, scene, "cartesian_descend", descend_plan, robot, joint_ids, locked_target, gripper, geom.jaw_open_width_m, recorder))
        if not descend_plan.success:
            result["reason"] = f"descend IK failed: {descend_plan.reason}"
            return result

        grasp_tcp = tcp_pose_matrix(robot)
        result["grasp_error"] = {
            "tcp_target_w_m": geom.grasp_pose_w[:3, 3].astype(float).tolist(),
            "tcp_actual_w_m": grasp_tcp[:3, 3].astype(float).tolist(),
            "xy_error_m": float(np.linalg.norm(grasp_tcp[:2, 3] - geom.grasp_pose_w[:2, 3])),
            "z_error_m": float(abs(grasp_tcp[2, 3] - geom.grasp_pose_w[2, 3])),
        }

        q_at_grasp = current_right_q(robot, joint_ids)
        close_start_width = geom.jaw_open_width_m
        for idx in range(max(1, int(args_cli.close_steps))):
            alpha = (idx + 1) / max(1, int(args_cli.close_steps))
            width = (1.0 - alpha) * close_start_width + alpha * float(args_cli.closed_width_m)
            set_right_q_target(robot, joint_ids, q_at_grasp, locked_target)
            if width < 0.0:
                gripper.set_width_unclamped(width)
            else:
                gripper.set_width(width)
            step_scene(sim, scene, recorder)
        hold_steps(sim, scene, max(0, int(args_cli.settle_steps)), robot, joint_ids, q_at_grasp, locked_target, gripper, float(args_cli.closed_width_m), recorder)
        result["stages"].append(
            {
                "label": "close_gripper",
                "success": True,
                "open_width_m": float(close_start_width),
                "closed_width_command_m": float(args_cli.closed_width_m),
                "measured_width_m": float(gripper.current_width()),
            }
        )

        lift_start = tcp_pose_matrix(robot)
        lift_start[:3, :3] = geom.tcp_rotation_w
        lift_waypoints_w = interpolate_cartesian_poses(lift_start, geom.lift_pose_w, float(args_cli.cartesian_step_m))
        lift_waypoints_b = [world_pose_to_robot_base_pose(robot, pose) for pose in lift_waypoints_w]
        lift_plan = planner.solve_ik_chain_for_poses_sequential(
            current_right_q(robot, joint_ids),
            lift_waypoints_b,
            return_seeds=64,
            max_waypoint_joint_l2_rad=1.20,
            newton_iters=40,
            position_only=False,
        )
        if not lift_plan.success and bool(args_cli.position_only_cartesian_fallback):
            result["stages"].append({"label": "lift_full_pose_failed", "success": False, "reason": lift_plan.reason, "metadata": lift_plan.metadata})
            lift_plan = planner.solve_ik_chain_for_poses_sequential(
                current_right_q(robot, joint_ids),
                lift_waypoints_b,
                return_seeds=64,
                max_waypoint_joint_l2_rad=1.20,
                newton_iters=40,
                position_only=True,
            )
        result["stages"].append(execute_joint_trajectory(sim, scene, "cartesian_lift", lift_plan, robot, joint_ids, locked_target, gripper, float(args_cli.closed_width_m), recorder))
        if not lift_plan.success:
            result["reason"] = f"lift IK failed: {lift_plan.reason}"
            return result

        max_object_z = float(obj.data.root_pos_w[0, 2].detach().cpu().item())
        for _ in range(max(0, int(args_cli.hold_steps))):
            hold_q = current_right_q(robot, joint_ids)
            set_right_q_target(robot, joint_ids, hold_q, locked_target)
            gripper.set_width_unclamped(float(args_cli.closed_width_m))
            step_scene(sim, scene, recorder)
            max_object_z = max(max_object_z, float(obj.data.root_pos_w[0, 2].detach().cpu().item()))

        final_object_pos = obj.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
        final_tcp = tcp_pose_matrix(robot)
        object_lift = float(final_object_pos[2] - object_initial_z)
        max_object_lift = float(max_object_z - object_initial_z)
        object_tcp_dist = float(np.linalg.norm(final_object_pos - final_tcp[:3, 3]))
        success = bool(object_lift >= float(args_cli.lift_success_height_m) or max_object_lift >= float(args_cli.lift_success_height_m))
        if success and object_tcp_dist > float(args_cli.max_object_tcp_distance_m):
            success = False
            result["reason"] = f"object lifted but not retained near TCP: dist={object_tcp_dist:.4f}m"
        else:
            result["reason"] = "physical_lift_success" if success else f"object lift {object_lift:.4f}m below threshold {float(args_cli.lift_success_height_m):.4f}m"
        result["success"] = success
        result["verification"] = {
            "object_initial_z_m": float(object_initial_z),
            "object_final_pos_w_m": final_object_pos.astype(float).tolist(),
            "object_final_lift_m": object_lift,
            "object_max_lift_m": max_object_lift,
            "object_tcp_distance_m": object_tcp_dist,
            "lift_success_height_m": float(args_cli.lift_success_height_m),
            "max_object_tcp_distance_m": float(args_cli.max_object_tcp_distance_m),
        }
        return result
    finally:
        if kuavo_client is not None:
            kuavo_client.close()
        result["elapsed_s"] = float(time.time() - start_time)
        if recorder is not None:
            recorder.finalize()
            result["video"] = {
                "enabled": True,
                "path": str(recorder.path),
                "frames": int(recorder.frame_count),
                "error": recorder.error,
            }


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("This standalone grasp test currently supports --num_envs 1 only.")
    object_z_expected = TABLE_SURFACE_Z + 0.5 * float(OBJECT_SIZE[2])
    if abs(float(OBJECT_POS[2]) - object_z_expected) > 0.012:
        print(
            f"[WARN] object_pos.z={float(OBJECT_POS[2]):.4f}m is not near table_surface_z + size_z/2 = {object_z_expected:.4f}m.",
            flush=True,
        )

    run_dir = Path(args_cli.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1.0 / 120.0,
        render_interval=1,
        physics_material=material(1.4, 1.1),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(list(args_cli.observer_camera_pos), list(args_cli.observer_camera_target))
    scene = InteractiveScene(TopDownCubeGraspSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False))
    gripper_material_meta = apply_gripper_contact_material(args_cli.num_envs)
    sim.reset()
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(sim.get_physics_dt())
    robot: Articulation = scene["robot"]
    _ = resolve_ee_body_id(robot)
    world_model = curobo_world_for_table(robot)
    planner = KuavoRightArmCuroboPlanner(
        KUAVO_BASE_URDF,
        GRIPPER_URDF,
        tcp_offset_m=tuple(float(v) for v in GRIPPER_LOCAL_TCP_OFFSET),
        world_model=world_model,
        device=str(args_cli.curobo_device),
        warmup=True,
        use_cuda_graph=True,
        collision_activation_distance_m=0.018,
        joint_limit_clip_rad=0.001,
        rotation_threshold_rad=0.35,
        strict_rotation_threshold_rad=0.12,
    )

    rng = np.random.default_rng(319)
    summary: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "run_dir": str(run_dir),
        "args": vars(args_cli).copy(),
        "trials": [],
        "success_count": 0,
    }
    summary["args"]["output_dir"] = str(summary["args"]["output_dir"])
    for trial in range(max(1, int(args_cli.num_trials))):
        if bool(args_cli.random_yaw):
            yaw = float(rng.uniform(-math.pi, math.pi))
        else:
            yaw = math.radians(float(args_cli.object_yaw_deg))
        print(f"[TOPDOWN] trial={trial} yaw_deg={math.degrees(yaw):.2f}", flush=True)
        trial_result = run_trial(sim, scene, planner, trial, yaw, run_dir, gripper_material_meta)
        summary["trials"].append(trial_result)
        summary["success_count"] = int(summary["success_count"]) + int(bool(trial_result.get("success")))
        print(
            f"[TOPDOWN] trial={trial} success={bool(trial_result.get('success'))} reason={trial_result.get('reason', '')}",
            flush=True,
        )
        if not simulation_app.is_running():
            break

    summary["success_rate"] = float(summary["success_count"]) / max(1, len(summary["trials"]))
    result_path = run_dir / "topdown_cube_grasp_result.json"
    result_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[TOPDOWN] wrote {result_path}", flush=True)
    if not bool(args_cli.exit_after_trials):
        while simulation_app.is_running():
            sim.step(render=True)
            scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
