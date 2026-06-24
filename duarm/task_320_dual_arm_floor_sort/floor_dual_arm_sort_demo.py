"""Kuavo dual-arm floor sorting demo for large objects.

This script is deliberately independent from ``../task_319_garbage_sort``.  It
reuses the same Kuavo S62 asset family, IsaacLab runtime style, head RGB-D
camera, VISS perception entrypoint, and stable kinematic wheel-base idea, but it
does not import or mutate the original task's large demo script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = WORKSPACE_ROOT.parent
ROBOT_ROOT = PARENT_ROOT / "robot"
TASK_DIR = Path(__file__).resolve().parent

if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))
if ROBOT_ROOT.exists() and str(ROBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOT_ROOT))
for source_dir in (
    PARENT_ROOT / "IsaacLab/source/isaaclab",
    PARENT_ROOT / "IsaacLab/source/isaaclab_assets",
    PARENT_ROOT / "IsaacLab/source/isaaclab_tasks",
):
    if source_dir.exists() and str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))


def parse_xyz(value: str) -> tuple[float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,z.")
    return (parts[0], parts[1], parts[2])


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependency
        raise RuntimeError(f"PyYAML is required to read {path}: {exc!r}") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}.")
    return data


SCENE_LAYOUT = load_yaml(TASK_DIR / "scene_layout.yaml")
AREA_LAYOUT = load_yaml(TASK_DIR / "area_layout.yaml")
DUAL_ARM_POSES = load_yaml(TASK_DIR / "dual_arm_poses.yaml")["poses"]

CATEGORY_CN = {
    "recyclable": "可回收物",
    "kitchen": "厨余垃圾",
    "hazardous": "有害垃圾",
    "other": "其他垃圾",
}
CATEGORY_ALIASES = {
    "可回收物": "recyclable",
    "可回收垃圾": "recyclable",
    "厨余垃圾": "kitchen",
    "湿垃圾": "kitchen",
    "有害垃圾": "hazardous",
    "其他垃圾": "other",
}

OBJECT_SPECS: list[dict[str, Any]] = list(SCENE_LAYOUT["objects"])
OBJECT_BY_KEY = {str(item["key"]): item for item in OBJECT_SPECS}
AREA_SPECS: dict[str, dict[str, Any]] = dict(AREA_LAYOUT["areas"])
TRASH_PRIM_FILTERS = [f"{{ENV_REGEX_NS}}/{spec['name']}" for spec in OBJECT_SPECS]

WHEEL_JOINTS = [
    "wheel_left_front_joint",
    "wheel_left_behind_joint",
    "wheel_right_front_joint",
    "wheel_right_behind_joint",
]
WHEEL_HALF_SPAN_XY = 0.23248871
URDF_DIAGONAL_WHEEL_GEOMETRY = (
    ((WHEEL_HALF_SPAN_XY, WHEEL_HALF_SPAN_XY), 0.785398163),
    ((-WHEEL_HALF_SPAN_XY, WHEEL_HALF_SPAN_XY), 2.356194372),
    ((WHEEL_HALF_SPAN_XY, -WHEEL_HALF_SPAN_XY), -0.785398163),
    ((-WHEEL_HALF_SPAN_XY, -WHEEL_HALF_SPAN_XY), -2.356194372),
)
ROBOT_BASE_Z = 0.0
LEFT_ARM_JOINT_EXPR = "zarm_l[1-7]_joint"
RIGHT_ARM_JOINT_EXPR = "zarm_r[1-7]_joint"
KUAVO_BASE_URDF = PARENT_ROOT / "kuavo-ros-opensource/src/kuavo_assets/models/biped_s62/urdf/biped_s62.urdf"
KUAVO_TASK_INFO = (
    PARENT_ROOT
    / "kuavo-ros-opensource/src/humanoid-wheel-control/humanoid_wheel_interface/config/kuavo_s62/task.info"
)
KUAVO_STANDALONE_IK_LIB_CANDIDATES = [
    ROBOT_ROOT / "standalone_ik/build_local/libstandalone_ik.so",
    ROBOT_ROOT / "standalone_ik/build/libstandalone_ik.so",
]
KUAVO_STANDALONE_IK_LIB = next((path for path in KUAVO_STANDALONE_IK_LIB_CANDIDATES if path.exists()), KUAVO_STANDALONE_IK_LIB_CANDIDATES[0])
DEFAULT_TASK320_CONTACT_URDF = TASK_DIR / "generated_assets/biped_s62_task320_contact_pads.urdf"


def _set_xml_child(parent: ET.Element, tag: str, attrib: dict[str, str]) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    child.attrib.clear()
    child.attrib.update(attrib)
    return child


def _ensure_light_inertial(link: ET.Element, mass: float) -> None:
    inertial = link.find("inertial")
    if inertial is not None:
        return
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{mass:.6f}"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "0.000120",
            "ixy": "0.0",
            "ixz": "0.0",
            "iyy": "0.000120",
            "iyz": "0.0",
            "izz": "0.000120",
        },
    )


def _remove_named_collisions(link: ET.Element, names: set[str]) -> None:
    for collision in list(link.findall("collision")):
        if collision.attrib.get("name") in names:
            link.remove(collision)


def _add_box_collision(link: ET.Element, name: str, xyz: str, size: str) -> None:
    _remove_named_collisions(link, {name})
    collision = ET.SubElement(link, "collision", {"name": name})
    ET.SubElement(collision, "origin", {"xyz": xyz, "rpy": "0 0 0"})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", {"size": size})


def _rewrite_package_mesh_paths(root: ET.Element) -> None:
    kuavo_assets_root = PARENT_ROOT / "kuavo-ros-opensource/src/kuavo_assets"
    package_prefix = "package://kuavo_assets/"
    for mesh in root.findall(".//mesh"):
        filename = str(mesh.attrib.get("filename", ""))
        if filename.startswith(package_prefix):
            relative = filename.removeprefix(package_prefix)
            mesh.attrib["filename"] = str((kuavo_assets_root / relative).resolve())


def build_task320_contact_urdf(source_urdf: Path, target_urdf: Path, *, pad_mass: float = 0.08) -> Path:
    """Create a local Kuavo URDF variant with real palm/wrist collision pads.

    The official model's empty auxiliary end-effector links and tiny 5 mm tip
    sphere are not enough for bilateral floor-object clamping. This patch keeps
    joints and IK frames unchanged while adding finite collision volumes that
    PhysX can actually contact.
    """
    target_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    _rewrite_package_mesh_paths(root)
    links = {str(link.attrib.get("name", "")): link for link in root.findall("link")}

    for side in ("l", "r"):
        wrist = links.get(f"zarm_{side}7_link")
        ee = links.get(f"zarm_{side}7_end_effector")
        ee_1 = links.get(f"zarm_{side}7_end_effector_1")
        ee_2 = links.get(f"zarm_{side}7_end_effector_2")
        for link in (ee, ee_1, ee_2):
            if link is not None:
                _ensure_light_inertial(link, pad_mass)
        if wrist is not None:
            _add_box_collision(wrist, "task320_wrist_palm_pad", "0 0 -0.125", "0.180 0.160 0.180")
            _add_box_collision(wrist, "task320_lower_scoop_pad", "0 0 -0.230", "0.220 0.180 0.070")
        if ee is not None:
            _add_box_collision(ee, "task320_distal_palm_face", "0 0 0", "0.220 0.170 0.110")
            _add_box_collision(ee, "task320_carry_lip", "0 0 -0.090", "0.240 0.190 0.060")
        if ee_1 is not None:
            _add_box_collision(ee_1, "task320_upper_palm_face", "0 0 0", "0.180 0.150 0.095")
        if ee_2 is not None:
            _add_box_collision(ee_2, "task320_forward_palm_face", "0 0 0", "0.180 0.150 0.095")

    ET.indent(tree, space="  ")
    tree.write(target_urdf, encoding="utf-8", xml_declaration=True)
    return target_urdf

try:
    solver_spec = importlib.util.spec_from_file_location("kuavo_kinematics_solver_offline", ROBOT_ROOT / "planning/kuavo_kinematics_solver.py")
    if solver_spec is None or solver_spec.loader is None:
        raise ImportError("could not create import spec for kuavo_kinematics_solver.py")
    solver_module = importlib.util.module_from_spec(solver_spec)
    solver_spec.loader.exec_module(solver_module)
    KuavoKinematicsSolver = solver_module.KuavoKinematicsSolver
except Exception as exc:  # pragma: no cover - optional official backend
    KuavoKinematicsSolver = None  # type: ignore[assignment]
    KUAVO_IK_IMPORT_ERROR = repr(exc)
else:
    KUAVO_IK_IMPORT_ERROR = ""


parser = argparse.ArgumentParser(description="Task 320 Kuavo dual-arm floor sorting demo.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_objects", type=int, default=10)
parser.add_argument("--warmup_steps", type=int, default=80)
parser.add_argument("--output_dir", default=str(TASK_DIR / "output"))
parser.add_argument("--robot_pos", type=parse_xyz, default=(0.25, 0.0, 0.0))
parser.add_argument("--robot_yaw", type=float, default=0.0)
parser.add_argument("--head_yaw", type=float, default=0.0)
parser.add_argument("--head_pitch", type=float, default=0.0)
parser.add_argument("--head_camera_width", type=int, default=1280)
parser.add_argument("--head_camera_height", type=int, default=960)
parser.add_argument("--perception_source", choices=("mock_scene", "viss_qwen_first"), default="mock_scene")
parser.add_argument("--viss_qwen_script", default=str(PARENT_ROOT / "viss/scripts/perception/yolo11_qwen_perception_offline.py"))
parser.add_argument("--viss_qwen_model", default=str(PARENT_ROOT / "viss/models/yolo11s-seg-best.pt"))
parser.add_argument("--viss_qwen_conf", type=float, default=0.15)
parser.add_argument("--viss_qwen_timeout_s", type=float, default=90.0)
parser.add_argument(
    "--carry_mode",
    choices=("pure_contact", "physical_soft_constraint", "dual_arm_constraint"),
    default="physical_soft_constraint",
    help=(
        "physical_soft_constraint is the robust default: true bilateral contact is required before carry, "
        "then a soft physical servo simulates high-friction closed grasp without writing object pose. "
        "Use pure_contact for the strict no-pose/no-velocity/no-wrench path."
    ),
)
parser.add_argument("--enable_waist", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--unlock_whole_body", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--use_task320_contact_pads", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--task320_contact_urdf", default=str(DEFAULT_TASK320_CONTACT_URDF))
parser.add_argument("--task320_contact_pad_mass", type=float, default=0.08)
parser.add_argument("--ik_backend", choices=("official_standalone", "heuristic", "curobo"), default="heuristic")
parser.add_argument("--curobo_device", default="cuda:0")
parser.add_argument("--curobo_num_seeds", type=int, default=48)
parser.add_argument("--curobo_position_threshold_m", type=float, default=0.025)
parser.add_argument("--curobo_rotation_threshold_rad", type=float, default=0.85)
parser.add_argument("--curobo_max_arm_joint_delta_rad", type=float, default=2.40)
parser.add_argument("--curobo_max_lower_body_delta_rad", type=float, default=1.20)
parser.add_argument("--official_ik_lib", default=str(KUAVO_STANDALONE_IK_LIB))
parser.add_argument("--official_ik_urdf", default=str(KUAVO_BASE_URDF))
parser.add_argument("--official_ik_task_info", default=str(KUAVO_TASK_INFO))
parser.add_argument("--official_ik_whole_body", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--official_ik_linear_error_max", type=float, default=0.20)
parser.add_argument("--official_ik_angular_error_max", type=float, default=0.35)
parser.add_argument("--official_ik_conservative_lower_body_limits", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--official_ik_reject_large_joint_delta", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--official_ik_max_arm_joint_delta_rad", type=float, default=3.60)
parser.add_argument("--official_ik_max_lower_body_delta_rad", type=float, default=0.18)
parser.add_argument("--grasp_gate_mode", choices=("contact_or_distance", "contact_only", "distance_only"), default="contact_only")
parser.add_argument("--contact_force_threshold_n", type=float, default=1.0)
parser.add_argument("--contact_distance_threshold_m", type=float, default=0.24)
parser.add_argument("--contact_surface_distance_threshold_m", type=float, default=0.08)
parser.add_argument("--object_static_friction", type=float, default=2.4)
parser.add_argument("--object_dynamic_friction", type=float, default=2.0)
parser.add_argument("--ground_static_friction", type=float, default=1.4)
parser.add_argument("--ground_dynamic_friction", type=float, default=1.1)
parser.add_argument("--sim_static_friction", type=float, default=1.8)
parser.add_argument("--sim_dynamic_friction", type=float, default=1.5)
parser.add_argument("--physics_dt", type=float, default=1.0 / 60.0)
parser.add_argument("--physx_solve_articulation_contact_last", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--physx_enable_stabilization", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--physx_external_forces_every_iteration", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--physical_carry_kp", type=float, default=32.0)
parser.add_argument("--physical_carry_kd", type=float, default=18.0)
parser.add_argument("--physical_carry_max_force_n", type=float, default=55.0)
parser.add_argument("--physical_carry_controller", choices=("velocity_servo", "external_wrench"), default="velocity_servo")
parser.add_argument("--physical_carry_max_linear_speed", type=float, default=0.46)
parser.add_argument("--physical_carry_max_vertical_speed", type=float, default=0.24)
parser.add_argument("--physical_carry_velocity_gain", type=float, default=1.35)
parser.add_argument("--physical_carry_max_accel_mps2", type=float, default=1.8)
parser.add_argument("--physical_carry_max_object_speed_mps", type=float, default=1.25)
parser.add_argument("--physical_carry_yaw_kp", type=float, default=4.0)
parser.add_argument("--physical_carry_yaw_kd", type=float, default=1.6)
parser.add_argument("--physical_carry_max_torque_nm", type=float, default=10.0)
parser.add_argument("--physics_safety_workspace_m", type=float, default=15.0)
parser.add_argument("--carried_nav_speed_scale", type=float, default=0.18)
parser.add_argument("--carried_nav_max_steps", type=int, default=2600)
parser.add_argument("--pre_release_stabilize_steps", type=int, default=80)
parser.add_argument("--settle_after_release_steps", type=int, default=90)
parser.add_argument("--nav_backend", choices=("local_kinematic", "nav2"), default="local_kinematic")
parser.add_argument("--nav_max_linear_speed", type=float, default=0.42)
parser.add_argument("--nav_max_angular_speed", type=float, default=0.75)
parser.add_argument("--nav_position_tolerance", type=float, default=0.05)
parser.add_argument("--nav_yaw_tolerance", type=float, default=0.06)
parser.add_argument("--nav_max_steps", type=int, default=1500)
parser.add_argument("--wheel_radius", type=float, default=0.13035)
parser.add_argument("--wheel_velocity_scale", type=float, default=0.35)
parser.add_argument("--mecanum_wheel_sign", type=float, default=-1.0)
parser.add_argument("--pick_standoff_m", type=float, default=0.34)
parser.add_argument("--hand_forward_compensation_m", type=float, default=0.08)
parser.add_argument("--scoop_steps", type=int, default=90)
parser.add_argument("--post_clamp_settle_steps", type=int, default=45)
parser.add_argument("--contact_retry_forward_m", type=float, default=0.055)
parser.add_argument("--contact_retry_steps", type=int, default=45)
parser.add_argument("--grasp_squeeze_m", type=float, default=0.025)
parser.add_argument("--lift_squeeze_m", type=float, default=0.030)
parser.add_argument("--scoop_z_offset_m", type=float, default=-0.035)
parser.add_argument("--clamp_z_offset_m", type=float, default=0.000)
parser.add_argument("--carry_forward_m", type=float, default=0.42)
parser.add_argument("--carry_lateral_m", type=float, default=0.0)
parser.add_argument("--carry_height_m", type=float, default=0.58)
parser.add_argument("--carry_center_after_lift", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--carry_center_max_step_m", type=float, default=0.28)
parser.add_argument("--carry_center_tolerance_m", type=float, default=0.08)
parser.add_argument("--carry_center_steps", type=int, default=90)
parser.add_argument("--carried_waypoint_spacing_m", type=float, default=0.72)
parser.add_argument("--carried_waypoint_regrip_steps", type=int, default=0)
parser.add_argument("--drop_height_m", type=float, default=0.34)
parser.add_argument("--trash_avoid_margin_m", type=float, default=0.46, help="Margin around unsorted trash objects used by the outside navigation corridor.")
parser.add_argument("--outside_corridor_x_m", type=float, default=0.45, help="Left-side outside corridor x used to avoid the floor trash pile.")
parser.add_argument("--pose_steps", type=int, default=120)
parser.add_argument("--clamp_steps", type=int, default=100)
parser.add_argument("--lift_steps", type=int, default=120)
parser.add_argument("--drop_steps", type=int, default=90)
parser.add_argument("--record_video", action="store_true")
parser.add_argument("--video_width", type=int, default=960)
parser.add_argument("--video_height", type=int, default=540)
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--video_sample_stride", type=int, default=4)
parser.add_argument("--observer_camera_pos", type=parse_xyz, default=(2.4, 4.8, 3.2))
parser.add_argument("--observer_camera_target", type=parse_xyz, default=(2.25, 0.0, 0.45))
parser.add_argument("--force_exit", action=argparse.BooleanOptionalAction, default=True, help="Exit immediately after writing summary; avoids slow Isaac Kit shutdown in headless batch runs.")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if args_cli.use_task320_contact_pads:
    KUAVO_SIM_URDF = build_task320_contact_urdf(
        KUAVO_BASE_URDF,
        Path(args_cli.task320_contact_urdf),
        pad_mass=float(args_cli.task320_contact_pad_mass),
    )
else:
    KUAVO_SIM_URDF = KUAVO_BASE_URDF

if args_cli.nav_backend == "nav2":
    print(
        "[WARN] --nav_backend nav2 is accepted for interface compatibility; "
        "this first independent scene executes the same target poses with the local kinematic base controller.",
        flush=True,
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import matrix_from_quat


@dataclass(slots=True)
class TargetCandidate:
    key: str
    name: str
    category: str
    center_world: tuple[float, float, float]
    confidence: float
    source: str
    bbox_xyxy: list[float] | None = None
    reason: str = ""


class ObserverVideoRecorder:
    def __init__(self, run_dir: Path) -> None:
        self.enabled = bool(args_cli.record_video)
        self.path = run_dir / "external_floor_sort_demo.mp4"
        self.frame_dir = run_dir / "external_video_frames"
        self.sample_stride = max(int(args_cli.video_sample_stride), 1)
        self.fps = max(float(args_cli.video_fps), 1.0)
        self.step_count = 0
        self.frame_count = 0
        self._writer: Any | None = None
        self.error = ""
        if self.enabled:
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, scene: InteractiveScene) -> None:
        if not self.enabled:
            return
        self.step_count += 1
        if self.step_count % self.sample_stride != 0:
            return
        try:
            rgb = sanitize_rgb(to_numpy(scene["observer_rgb"].data.output["rgb"])[0])
            self._write(rgb)
        except Exception as exc:
            self.error = repr(exc)

    def _write(self, frame: np.ndarray) -> None:
        try:
            import cv2  # type: ignore

            if self._writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
            self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            self.error = repr(exc)
            Image.fromarray(frame).save(self.frame_dir / f"frame_{self.frame_count:06d}.png")
        self.frame_count += 1

    def finalize(self) -> None:
        if self._writer is not None:
            self._writer.release()

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": str(self.path) if self.enabled else "",
            "frames": self.frame_count,
            "error": self.error,
        }


def to_numpy(value: torch.Tensor) -> np.ndarray:
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


def sanitize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32, copy=False)


def quat_wxyz_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def yaw_from_quat_wxyz(q: Any) -> float:
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def min_jerk(alpha: float) -> float:
    alpha = max(0.0, min(1.0, float(alpha)))
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def yaw_toward_xy(from_x: float, from_y: float, target_x: float, target_y: float) -> float:
    return math.atan2(float(target_y) - float(from_y), float(target_x) - float(from_x))


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
    rot = np.stack([forward, right, up], axis=1)
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [0.25 * scale, (rot[2, 1] - rot[1, 2]) / scale, (rot[0, 2] - rot[2, 0]) / scale, (rot[1, 0] - rot[0, 1]) / scale],
            dtype=np.float64,
        )
    else:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1.0e-9)
    return tuple(float(v) for v in quat)


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
            quat = np.array(
                [
                    (rot[2, 1] - rot[1, 2]) / scale,
                    0.25 * scale,
                    (rot[0, 1] + rot[1, 0]) / scale,
                    (rot[0, 2] + rot[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif idx == 1:
            scale = math.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 1.0e-12)) * 2.0
            quat = np.array(
                [
                    (rot[0, 2] - rot[2, 0]) / scale,
                    (rot[0, 1] + rot[1, 0]) / scale,
                    0.25 * scale,
                    (rot[1, 2] + rot[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 1.0e-12)) * 2.0
            quat = np.array(
                [
                    (rot[1, 0] - rot[0, 1]) / scale,
                    (rot[0, 2] + rot[2, 0]) / scale,
                    (rot[1, 2] + rot[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    quat /= max(float(np.linalg.norm(quat)), 1.0e-9)
    return tuple(float(v) for v in quat)


def rotation_matrix_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def normalize_category(category: str | None) -> str:
    value = str(category or "").strip()
    if value in AREA_SPECS:
        return value
    if value in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[value]
    lowered = value.casefold()
    if "hazard" in lowered or "有害" in value:
        return "hazardous"
    if "kitchen" in lowered or "厨余" in value or "food" in lowered:
        return "kitchen"
    if "recycl" in lowered or "回收" in value:
        return "recyclable"
    return "other"


def colored_surface(color: tuple[float, float, float]) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.78)


def semantic_tags(semantic_class: str) -> list[tuple[str, str]]:
    return [("class", semantic_class)]


def contact_material(static_friction: float, dynamic_friction: float, restitution: float = 0.01) -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )


def task320_sim_cfg() -> sim_utils.SimulationCfg:
    return sim_utils.SimulationCfg(
        dt=float(args_cli.physics_dt),
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(
            solve_articulation_contact_last=bool(args_cli.physx_solve_articulation_contact_last),
            enable_stabilization=bool(args_cli.physx_enable_stabilization),
            enable_external_forces_every_iteration=bool(args_cli.physx_external_forces_every_iteration),
        ),
        physics_material=contact_material(float(args_cli.sim_static_friction), float(args_cli.sim_dynamic_friction), 0.0),
    )


def rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        linear_damping=0.10,
        angular_damping=0.16,
        max_depenetration_velocity=1.2,
        solver_position_iteration_count=12,
        solver_velocity_iteration_count=4,
    )


def static_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        kinematic_enabled=True,
        disable_gravity=True,
        linear_damping=0.0,
        angular_damping=0.0,
        max_depenetration_velocity=1.0,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=2,
    )


def collision_props(contact_offset: float = 0.003) -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=contact_offset, rest_offset=0.0)


def cuboid_rigid_from_spec(spec: dict[str, Any], *, static: bool = False, semantic_class: str | None = None) -> RigidObjectCfg:
    size = tuple(float(v) for v in spec["size"])
    pos = tuple(float(v) for v in spec.get("position", spec.get("center")))
    color = tuple(float(v) for v in spec["color"])
    if spec.get("asset") and not static:
        asset_ref = str(spec["asset"])
        if asset_ref.startswith("ycb_physics/"):
            asset_path: str | Path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/{asset_ref.removeprefix('ycb_physics/')}"
        else:
            asset_path = (WORKSPACE_ROOT / asset_ref).resolve()
        scale = tuple(float(v) for v in spec.get("scale", (1.0, 1.0, 1.0)))
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{spec['name']}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(asset_path),
                scale=scale,
                rigid_props=rigid_props(),
                mass_props=sim_utils.MassPropertiesCfg(mass=float(spec.get("mass", 1.2))),
                collision_props=collision_props(0.002),
                semantic_tags=semantic_tags(semantic_class or str(spec.get("semantic_class", spec["name"]))),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        )
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{spec['name']}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=static_rigid_props() if static else rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(spec.get("mass", 1.2 if not static else 20.0))),
            collision_props=collision_props(0.002),
            physics_material=contact_material(
                float(args_cli.ground_static_friction if static else args_cli.object_static_friction),
                float(args_cli.ground_dynamic_friction if static else args_cli.object_dynamic_friction),
                0.0 if static else 0.01,
            ),
            visual_material=colored_surface(color),
            semantic_tags=semantic_tags(semantic_class or str(spec.get("semantic_class", spec["name"]))),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def area_spec(category: str) -> dict[str, Any]:
    spec = dict(AREA_SPECS[category])
    spec["category"] = category
    return spec


def area_cfg(category: str) -> RigidObjectCfg:
    spec = area_spec(category)
    spec["position"] = spec["center"]
    return cuboid_rigid_from_spec(spec, static=True, semantic_class=f"sort_area_{category}")


def wheel_actuator_cfg() -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=["wheel_.*_joint"],
        effort_limit_sim=35.0,
        velocity_limit_sim=25.0,
        stiffness=0.0,
        damping=18.0,
    )


KUAVO_FLOOR_SORT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Kuavo62",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(KUAVO_SIM_URDF),
        fix_base=False,
        merge_fixed_joints=False,
        link_density=1000.0,
        collision_from_visuals=True,
        collider_type="convex_hull",
        activate_contact_sensors=True,
        self_collision=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            linear_damping=0.08,
            angular_damping=0.12,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=140.0, damping=16.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=args_cli.robot_pos,
        rot=quat_wxyz_from_yaw(args_cli.robot_yaw),
        joint_pos={
            "knee_joint": 0.0,
            "leg_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "zhead_1_joint": args_cli.head_yaw,
            "zhead_2_joint": args_cli.head_pitch,
            **{name: float(value) for name, value in DUAL_ARM_POSES["home_down"].items() if name.startswith("zarm_")},
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wheels": wheel_actuator_cfg(),
        "lower_body": ImplicitActuatorCfg(
            joint_names_expr=["knee_joint", "leg_joint", "waist_.*_joint"],
            effort_limit_sim=820.0 if args_cli.unlock_whole_body else 900.0,
            velocity_limit_sim=1.2 if args_cli.unlock_whole_body else 0.1,
            stiffness=950.0 if args_cli.unlock_whole_body else 4500.0,
            damping=135.0 if args_cli.unlock_whole_body else 520.0,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=[LEFT_ARM_JOINT_EXPR],
            effort_limit_sim=420.0,
            velocity_limit_sim=3.2,
            stiffness=1650.0,
            damping=150.0,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=[RIGHT_ARM_JOINT_EXPR],
            effort_limit_sim=420.0,
            velocity_limit_sim=3.2,
            stiffness=1650.0,
            damping=150.0,
        ),
        "locked_head": ImplicitActuatorCfg(
            joint_names_expr=["zhead_[12]_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=0.1,
            stiffness=900.0,
            damping=120.0,
        ),
    },
)


@configclass
class FloorLargeTrashSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 7.0, 0.02),
            collision_props=collision_props(0.002),
            physics_material=contact_material(float(args_cli.ground_static_friction), float(args_cli.ground_dynamic_friction), 0.0),
            visual_material=colored_surface((0.34, 0.35, 0.33)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.3, 0.0, -0.01)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2600.0, color=(0.84, 0.87, 0.90)),
    )
    robot: ArticulationCfg = KUAVO_FLOOR_SORT_CFG

    area_recyclable = area_cfg("recyclable")
    area_kitchen = area_cfg("kitchen")
    area_hazardous = area_cfg("hazardous")
    area_other = area_cfg("other")

    trash_00 = cuboid_rigid_from_spec(OBJECT_BY_KEY["trash_00"])
    trash_01 = cuboid_rigid_from_spec(OBJECT_BY_KEY["trash_01"])
    trash_02 = cuboid_rigid_from_spec(OBJECT_BY_KEY["trash_02"])
    trash_03 = cuboid_rigid_from_spec(OBJECT_BY_KEY["trash_03"])
    trash_04 = cuboid_rigid_from_spec(OBJECT_BY_KEY["trash_04"])
    trash_05 = cuboid_rigid_from_spec(OBJECT_BY_KEY["trash_05"])
    trash_06 = cuboid_rigid_from_spec(OBJECT_BY_KEY["trash_06"])
    trash_07 = cuboid_rigid_from_spec(OBJECT_BY_KEY["trash_07"])
    trash_08 = cuboid_rigid_from_spec(OBJECT_BY_KEY["trash_08"])
    trash_09 = cuboid_rigid_from_spec(OBJECT_BY_KEY["trash_09"])

    head_rgbd = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo62/head_camera_depth/head_rgbd",
        update_period=1.0 / 30.0,
        width=args_cli.head_camera_width,
        height=args_cli.head_camera_height,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, focus_distance=1.5, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.405309, -0.579417, 0.579417, -0.405309), convention="ros"),
        update_latest_camera_pose=True,
    )

    observer_rgb = CameraCfg(
        prim_path="{ENV_REGEX_NS}/observer_rgb",
        update_period=0.0,
        width=args_cli.video_width,
        height=args_cli.video_height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, focus_distance=4.8, horizontal_aperture=20.955, clipping_range=(0.1, 12.0)),
        offset=CameraCfg.OffsetCfg(
            pos=args_cli.observer_camera_pos,
            rot=look_at_quat_world(args_cli.observer_camera_pos, args_cli.observer_camera_target),
            convention="world",
        ),
    )

    left_arm_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo62/zarm_l.*",
        update_period=0.0,
        history_length=4,
        debug_vis=False,
        filter_prim_paths_expr=TRASH_PRIM_FILTERS,
    )
    right_arm_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo62/zarm_r.*",
        update_period=0.0,
        history_length=4,
        debug_vis=False,
        filter_prim_paths_expr=TRASH_PRIM_FILTERS,
    )


def reset_scene(scene: InteractiveScene) -> None:
    robot: Articulation = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    scene.reset()


def robot_planar_pose(robot: Articulation) -> list[float]:
    root = robot.data.root_state_w[0].detach().cpu()
    return [float(root[0]), float(root[1]), float(yaw_from_quat_wxyz(root[3:7]))]


def write_robot_planar_pose(robot: Articulation, pose_xyyaw: tuple[float, float, float]) -> None:
    root_state = robot.data.root_state_w.clone()
    x, y, yaw = float(pose_xyyaw[0]), float(pose_xyyaw[1]), float(pose_xyyaw[2])
    root_state[:, 0] = x
    root_state[:, 1] = y
    root_state[:, 2] = ROBOT_BASE_Z
    root_state[:, 3:7] = torch.tensor(quat_wxyz_from_yaw(yaw), dtype=root_state.dtype, device=root_state.device).reshape(1, 4)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])


def non_wheel_joint_ids(robot: Articulation, wheel_ids: list[int]) -> list[int]:
    wheel_set = {int(item) for item in wheel_ids}
    return [idx for idx in range(robot.num_joints) if idx not in wheel_set]


def hold_non_wheel_joints(robot: Articulation, locked_joint_target: torch.Tensor, wheel_ids: list[int]) -> None:
    joint_ids = non_wheel_joint_ids(robot, wheel_ids)
    if joint_ids:
        robot.set_joint_position_target(locked_joint_target[:, joint_ids], joint_ids=joint_ids)


def apply_named_joint_positions(robot: Articulation, joint_target: torch.Tensor, named_positions: dict[str, float]) -> torch.Tensor:
    target = joint_target.clone()
    joint_ids, joint_names = robot.find_joints(list(named_positions.keys()), preserve_order=True)
    for joint_id, joint_name in zip(joint_ids, joint_names):
        target[:, int(joint_id)] = float(named_positions[joint_name])
    missing = sorted(set(named_positions.keys()) - set(joint_names))
    if missing:
        print(f"[WARN] Missing pose joints ignored: {missing}", flush=True)
    return target


def wheel_velocity_targets(vx: float, wz: float) -> list[float]:
    targets: list[float] = []
    sign = float(args_cli.mecanum_wheel_sign)
    for (wheel_x, wheel_y), yaw in URDF_DIAGONAL_WHEEL_GEOMETRY:
        axis_x, axis_y = math.cos(yaw), math.sin(yaw)
        tangent_x = -axis_y
        tangent_y = axis_x
        wheel_vx = float(vx) - float(wz) * wheel_y
        wheel_vy = float(wz) * wheel_x
        targets.append(float(args_cli.wheel_velocity_scale) * sign * (tangent_x * wheel_vx + tangent_y * wheel_vy) / float(args_cli.wheel_radius))
    return targets


def apply_raw_wheel_velocity(robot: Articulation, wheel_ids: list[int], targets: list[float]) -> None:
    if len(wheel_ids) != 4:
        return
    target = torch.zeros((robot.num_instances, len(wheel_ids)), dtype=robot.data.joint_vel.dtype, device=robot.device)
    for idx, value in enumerate(targets):
        target[:, idx] = float(value)
    robot.set_joint_velocity_target(target, joint_ids=wheel_ids)


def integrate_base_motion(robot: Articulation, vx: float, wz: float, dt: float) -> None:
    root_state = robot.data.root_state_w.clone()
    for env_id in range(root_state.shape[0]):
        yaw = yaw_from_quat_wxyz(root_state[env_id, 3:7])
        mid_yaw = yaw + 0.5 * float(wz) * float(dt)
        next_yaw = wrap_to_pi(yaw + float(wz) * float(dt))
        root_state[env_id, 0] += float(vx) * math.cos(mid_yaw) * float(dt)
        root_state[env_id, 1] += float(vx) * math.sin(mid_yaw) * float(dt)
        root_state[env_id, 2] = ROBOT_BASE_Z
        root_state[env_id, 3:7] = torch.tensor(quat_wxyz_from_yaw(next_yaw), dtype=root_state.dtype, device=root_state.device)
        root_state[env_id, 7] = float(vx) * math.cos(next_yaw)
        root_state[env_id, 8] = float(vx) * math.sin(next_yaw)
        root_state[env_id, 9] = 0.0
        root_state[env_id, 10] = 0.0
        root_state[env_id, 11] = 0.0
        root_state[env_id, 12] = float(wz)
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])


def write_rigid_object_pose(scene: InteractiveScene, scene_key: str, pos_w: tuple[float, float, float], yaw: float) -> None:
    obj = scene[scene_key]
    root_state = obj.data.root_state_w.clone()
    root_state[:, 0:3] = torch.tensor(pos_w, dtype=root_state.dtype, device=root_state.device).reshape(1, 3)
    root_state[:, 3:7] = torch.tensor(quat_wxyz_from_yaw(yaw), dtype=root_state.dtype, device=root_state.device).reshape(1, 4)
    root_state[:, 7:] = 0.0
    obj.write_root_pose_to_sim(root_state[:, :7])
    obj.write_root_velocity_to_sim(root_state[:, 7:])


def carried_pose(robot: Articulation) -> tuple[tuple[float, float, float], float]:
    x, y, yaw = robot_planar_pose(robot)
    forward = float(args_cli.carry_forward_m)
    lateral = float(args_cli.carry_lateral_m)
    pos = (
        x + forward * math.cos(yaw) - lateral * math.sin(yaw),
        y + forward * math.sin(yaw) + lateral * math.cos(yaw),
        float(args_cli.carry_height_m),
    )
    return pos, yaw


def object_mass_for_key(scene_key: str) -> float:
    spec = OBJECT_BY_KEY.get(scene_key, {})
    return float(spec.get("mass", 1.2))


def object_yaw(scene: InteractiveScene, scene_key: str) -> float:
    quat = scene[scene_key].data.root_state_w[0, 3:7].detach().cpu()
    return yaw_from_quat_wxyz(quat)


def clear_object_wrench(scene: InteractiveScene, scene_key: str) -> None:
    obj = scene[scene_key]
    if args_cli.physical_carry_controller == "external_wrench":
        zeros = torch.zeros((1, 1, 3), dtype=obj.data.root_state_w.dtype, device=obj.device)
        obj.set_external_force_and_torque(zeros, zeros, is_global=True)
    zero_vel = torch.zeros((1, 6), dtype=obj.data.root_state_w.dtype, device=obj.device)
    obj.write_root_velocity_to_sim(zero_vel)


def root_state_is_sane(root_state: torch.Tensor) -> bool:
    if not bool(torch.isfinite(root_state).all().item()):
        return False
    max_abs = float(args_cli.physics_safety_workspace_m)
    pos_abs = torch.max(torch.abs(root_state[0, 0:3]))
    return float(pos_abs.item()) <= max_abs


def limit_vector_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.norm(vector)
    if float(norm.item()) > max_norm:
        return vector * (max_norm / max(float(norm.item()), 1.0e-6))
    return vector


def apply_physical_carry_velocity_servo(scene: InteractiveScene, robot: Articulation, scene_key: str, sim_dt: float) -> None:
    obj = scene[scene_key]
    root = obj.data.root_state_w
    if not root_state_is_sane(root):
        clear_object_wrench(scene, scene_key)
        return
    pos = root[0, 0:3]
    target_pos_tuple, target_yaw = carried_pose(robot)
    target_pos = torch.tensor(target_pos_tuple, dtype=root.dtype, device=obj.device)
    err = target_pos - pos
    vel = float(args_cli.physical_carry_velocity_gain) * err
    vel[0:2] = limit_vector_norm(vel[0:2], float(args_cli.physical_carry_max_linear_speed))
    vel[2] = torch.clamp(
        vel[2],
        min=-float(args_cli.physical_carry_max_vertical_speed),
        max=float(args_cli.physical_carry_max_vertical_speed),
    )
    current_vel = root[0, 7:10]
    if float(torch.linalg.norm(current_vel).item()) > float(args_cli.physical_carry_max_object_speed_mps) * 4.0:
        current_vel = torch.zeros_like(current_vel)
    max_delta = max(0.0, float(args_cli.physical_carry_max_accel_mps2)) * max(float(sim_dt), 1.0e-4)
    delta = limit_vector_norm(vel - current_vel, max_delta)
    vel = limit_vector_norm(current_vel + delta, float(args_cli.physical_carry_max_object_speed_mps))
    yaw_err = wrap_to_pi(float(target_yaw) - object_yaw(scene, scene_key))
    wz = clamp(float(args_cli.physical_carry_yaw_kp) * yaw_err, -1.0, 1.0)
    root_vel = torch.zeros((1, 6), dtype=root.dtype, device=obj.device)
    root_vel[0, 0:3] = vel
    root_vel[0, 5] = wz
    obj.write_root_velocity_to_sim(root_vel)


def apply_physical_carry_wrench(scene: InteractiveScene, robot: Articulation, scene_key: str, sim_dt: float) -> None:
    """Apply a soft physical carry constraint as external wrench, without writing object pose."""
    if args_cli.physical_carry_controller == "velocity_servo":
        apply_physical_carry_velocity_servo(scene, robot, scene_key, sim_dt)
        return
    obj = scene[scene_key]
    root = obj.data.root_state_w
    if not root_state_is_sane(root):
        clear_object_wrench(scene, scene_key)
        return
    pos = root[0, 0:3]
    lin_vel = root[0, 7:10]
    ang_vel = root[0, 10:13]
    target_pos_tuple, target_yaw = carried_pose(robot)
    target_pos = torch.tensor(target_pos_tuple, dtype=root.dtype, device=obj.device)
    err = target_pos - pos
    force = float(args_cli.physical_carry_kp) * err - float(args_cli.physical_carry_kd) * lin_vel
    force[2] += object_mass_for_key(scene_key) * 9.81
    max_force = float(args_cli.physical_carry_max_force_n)
    force_norm = torch.linalg.norm(force)
    if float(force_norm.item()) > max_force:
        force = force * (max_force / max(float(force_norm.item()), 1.0e-6))

    yaw_err = wrap_to_pi(float(target_yaw) - object_yaw(scene, scene_key))
    yaw_torque = float(args_cli.physical_carry_yaw_kp) * yaw_err - float(args_cli.physical_carry_yaw_kd) * float(ang_vel[2])
    yaw_torque = max(-float(args_cli.physical_carry_max_torque_nm), min(float(args_cli.physical_carry_max_torque_nm), yaw_torque))
    torque = torch.tensor((0.0, 0.0, yaw_torque), dtype=root.dtype, device=obj.device)
    obj.set_external_force_and_torque(force.reshape(1, 1, 3), torque.reshape(1, 1, 3), is_global=True)


def update_carried_object(scene: InteractiveScene, robot: Articulation, scene_key: str | None) -> None:
    if not scene_key:
        return
    pos, yaw = carried_pose(robot)
    write_rigid_object_pose(scene, scene_key, pos, yaw)


def smooth_object_pose_to(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    recorder: ObserverVideoRecorder,
    scene_key: str,
    target_pos: tuple[float, float, float],
    target_yaw: float,
    steps: int,
) -> None:
    start_pos = scene_object_center(scene, scene_key)
    start_yaw = yaw_from_quat_wxyz(scene[scene_key].data.root_state_w[0, 3:7].detach().cpu())
    yaw_delta = wrap_to_pi(float(target_yaw) - start_yaw)
    for step in range(max(1, int(steps))):
        alpha = min_jerk((step + 1) / max(1, int(steps)))
        pos = (
            start_pos[0] + alpha * (float(target_pos[0]) - start_pos[0]),
            start_pos[1] + alpha * (float(target_pos[1]) - start_pos[1]),
            start_pos[2] + alpha * (float(target_pos[2]) - start_pos[2]),
        )
        yaw = wrap_to_pi(start_yaw + alpha * yaw_delta)
        write_rigid_object_pose(scene, scene_key, pos, yaw)
        step_sim(sim, scene, robot, wheel_ids, locked_joint_target, recorder)


def trash_obstacle_bounds(exclude: set[str] | None = None) -> tuple[float, float, float, float]:
    excluded = exclude or set()
    xs: list[float] = []
    ys: list[float] = []
    margin = float(args_cli.trash_avoid_margin_m)
    for spec in OBJECT_SPECS:
        key = str(spec["key"])
        if key in excluded:
            continue
        x, y, _z = (float(v) for v in spec["position"])
        sx, sy, _sz = (float(v) for v in spec["size"])
        xs.extend([x - 0.5 * sx - margin, x + 0.5 * sx + margin])
        ys.extend([y - 0.5 * sy - margin, y + 0.5 * sy + margin])
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), max(xs), min(ys), max(ys))


def outside_corridor_y(target_y: float, *, exclude: set[str] | None = None) -> float:
    _min_x, _max_x, min_y, max_y = trash_obstacle_bounds(exclude=exclude)
    center_y = 0.5 * (min_y + max_y)
    return min_y if float(target_y) <= center_y else max_y


def path_around_trash(
    robot: Articulation,
    target_pose: tuple[float, float, float],
    *,
    exclude: set[str] | None = None,
    final_from_left: bool = False,
) -> list[tuple[float, float, float]]:
    current = robot_planar_pose(robot)
    corridor_x = float(args_cli.outside_corridor_x_m)
    corridor_y = outside_corridor_y(float(target_pose[1]), exclude=exclude)
    waypoints: list[tuple[float, float, float]] = []
    if final_from_left:
        waypoints.append((corridor_x, float(current[1]), yaw_toward_xy(corridor_x, float(current[1]), corridor_x, corridor_y)))
        waypoints.append((corridor_x, corridor_y, yaw_toward_xy(corridor_x, corridor_y, target_pose[0], target_pose[1])))
    else:
        waypoints.append((corridor_x, float(current[1]), yaw_toward_xy(corridor_x, float(current[1]), corridor_x, corridor_y)))
        waypoints.append((corridor_x, corridor_y, yaw_toward_xy(corridor_x, corridor_y, target_pose[0], corridor_y)))
        waypoints.append((float(target_pose[0]), corridor_y, yaw_toward_xy(target_pose[0], corridor_y, target_pose[0], target_pose[1])))
    waypoints.append(target_pose)
    deduped: list[tuple[float, float, float]] = []
    for pose in waypoints:
        if deduped and math.hypot(pose[0] - deduped[-1][0], pose[1] - deduped[-1][1]) < 0.05:
            deduped[-1] = pose
        else:
            deduped.append(pose)
    return deduped


def subdivide_planar_waypoints(
    waypoints: list[tuple[float, float, float]],
    max_spacing_m: float,
) -> list[tuple[float, float, float]]:
    spacing = max(float(max_spacing_m), 0.05)
    if len(waypoints) <= 1:
        return waypoints
    result: list[tuple[float, float, float]] = [waypoints[0]]
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        distance = math.hypot(dx, dy)
        segments = max(1, int(math.ceil(distance / spacing)))
        yaw_delta = wrap_to_pi(float(end[2]) - float(start[2]))
        for segment in range(1, segments + 1):
            alpha = segment / segments
            result.append(
                (
                    float(start[0]) + alpha * dx,
                    float(start[1]) + alpha * dy,
                    wrap_to_pi(float(start[2]) + alpha * yaw_delta),
                )
            )
    return result


def step_sim(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    recorder: ObserverVideoRecorder,
    *,
    vx: float = 0.0,
    wz: float = 0.0,
    carried_key: str | None = None,
) -> None:
    sim_dt = sim.get_physics_dt()
    hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
    apply_raw_wheel_velocity(robot, wheel_ids, wheel_velocity_targets(vx, wz))
    if abs(vx) > 1.0e-7 or abs(wz) > 1.0e-7:
        integrate_base_motion(robot, vx, wz, sim_dt)
    if carried_key:
        if args_cli.carry_mode == "pure_contact":
            pass
        elif args_cli.carry_mode == "physical_soft_constraint":
            apply_physical_carry_wrench(scene, robot, carried_key, sim_dt)
        else:
            update_carried_object(scene, robot, carried_key)
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(sim_dt)
    recorder.capture(scene)
    if not getattr(args_cli, "headless", False):
        time.sleep(max(0.0, sim_dt))


def transition_pose(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    recorder: ObserverVideoRecorder,
    pose_name: str,
    steps: int,
    *,
    carried_key: str | None = None,
) -> torch.Tensor:
    target = apply_named_joint_positions(robot, locked_joint_target, {k: float(v) for k, v in DUAL_ARM_POSES[pose_name].items()})
    start = locked_joint_target.clone()
    for step in range(max(1, int(steps))):
        alpha = min_jerk((step + 1) / max(1, int(steps)))
        current = start + alpha * (target - start)
        step_sim(sim, scene, robot, wheel_ids, current, recorder, carried_key=carried_key)
    return target


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def object_size_for_key(scene_key: str) -> tuple[float, float, float]:
    spec = OBJECT_BY_KEY.get(scene_key, {})
    size = spec.get("size", (0.35, 0.28, 0.25))
    return (float(size[0]), float(size[1]), float(size[2]))


def object_layout_yaw_for_key(scene_key: str) -> float:
    spec = OBJECT_BY_KEY.get(scene_key, {})
    return float(spec.get("yaw", 0.0))


def object_half_extent_along_axis(scene_key: str, axis_w: np.ndarray) -> float:
    sx, sy, _sz = object_size_for_key(scene_key)
    axis = np.asarray(axis_w[:2], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-9:
        return 0.5 * max(sx, sy)
    axis /= norm
    yaw = object_layout_yaw_for_key(scene_key)
    object_x = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    object_y = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    return float(0.5 * sx * abs(float(np.dot(axis, object_x))) + 0.5 * sy * abs(float(np.dot(axis, object_y))))


OFFICIAL_IK_BRIDGE: "KuavoOfficialIKBridge | None" = None
CUROBO_IK_BRIDGE: "KuavoCuroboDualArmIKBridge | None" = None


CUROBO_WHOLE_BODY_JOINTS = [
    "knee_joint",
    "leg_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_l5_joint",
    "zarm_l6_joint",
    "zarm_l7_joint",
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
    "zarm_r5_joint",
    "zarm_r6_joint",
    "zarm_r7_joint",
]


class KuavoCuroboDualArmIKBridge:
    """cuRobo dual-link IK bridge for Task 320's physical Isaac execution."""

    LEFT_EE_LINK = "zarm_l7_end_effector"
    RIGHT_EE_LINK = "zarm_r7_end_effector"

    def __init__(self) -> None:
        self.available = False
        self.status: dict[str, Any] = {
            "backend": "curobo",
            "available": False,
            "device": str(args_cli.curobo_device),
            "urdf": str(KUAVO_SIM_URDF),
            "base_link": "base_link",
            "right_ee_link": self.RIGHT_EE_LINK,
            "left_ee_link": self.LEFT_EE_LINK,
            "num_seeds": int(args_cli.curobo_num_seeds),
            "position_threshold_m": float(args_cli.curobo_position_threshold_m),
            "rotation_threshold_rad": float(args_cli.curobo_rotation_threshold_rad),
            "reason": "",
        }
        self.tensor_args: Any | None = None
        self.ik_solver: Any | None = None
        self.joint_names: list[str] = []
        try:
            import torch as _torch
            from curobo.types.base import TensorDeviceType
            from curobo.types.robot import RobotConfig
            from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

            self.tensor_args = TensorDeviceType(device=_torch.device(str(args_cli.curobo_device)))
            robot_cfg = RobotConfig.from_dict(self.robot_cfg_dict(), self.tensor_args)
            ik_config = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                None,
                tensor_args=self.tensor_args,
                num_seeds=max(1, int(args_cli.curobo_num_seeds)),
                position_threshold=float(args_cli.curobo_position_threshold_m),
                rotation_threshold=float(args_cli.curobo_rotation_threshold_rad),
                self_collision_check=False,
                self_collision_opt=False,
                use_cuda_graph=False,
                regularization=True,
            )
            self.ik_solver = IKSolver(ik_config)
            self.joint_names = list(self.ik_solver.joint_names)
        except Exception as exc:
            self.status["reason"] = f"init_failed: {exc!r}"
            return
        self.available = True
        self.status["available"] = True
        self.status["reason"] = "ready"
        self.status["dof_names"] = list(self.joint_names)

    def robot_cfg_dict(self) -> dict[str, Any]:
        return {
            "robot_cfg": {
                "kinematics": {
                    "use_usd_kinematics": False,
                    "usd_path": "",
                    "usd_robot_root": "/robot",
                    "isaac_usd_path": "",
                    "usd_flip_joints": {},
                    "usd_flip_joint_limits": [],
                    "urdf_path": str(KUAVO_SIM_URDF),
                    "asset_root_path": str(Path(KUAVO_SIM_URDF).parent),
                    "base_link": "base_link",
                    "ee_link": self.RIGHT_EE_LINK,
                    "link_names": [self.RIGHT_EE_LINK, self.LEFT_EE_LINK],
                    "lock_joints": {},
                    "collision_link_names": [],
                    "collision_spheres": None,
                    "self_collision_ignore": {},
                    "self_collision_buffer": {},
                    "use_global_cumul": True,
                    "mesh_link_names": [],
                    "external_asset_path": None,
                    "cspace": {
                        "joint_names": list(CUROBO_WHOLE_BODY_JOINTS),
                        "retract_config": [0.0] * len(CUROBO_WHOLE_BODY_JOINTS),
                        "null_space_weight": [1.0] * len(CUROBO_WHOLE_BODY_JOINTS),
                        "cspace_distance_weight": [1.0] * len(CUROBO_WHOLE_BODY_JOINTS),
                        "max_acceleration": 6.0,
                        "max_jerk": 180.0,
                    },
                }
            }
        }

    def seed_from_joint_target(self, robot: Articulation, joint_target: torch.Tensor) -> torch.Tensor:
        values: list[float] = []
        for name in self.joint_names:
            joint_ids, _joint_names = robot.find_joints([name], preserve_order=True)
            if joint_ids:
                values.append(float(joint_target[0, int(joint_ids[0])].detach().cpu()))
            else:
                values.append(0.0)
        return self.tensor_args.to_device(np.asarray(values, dtype=np.float32).reshape(1, -1))

    def current_pose_from_joint_target(self, robot: Articulation, joint_target: torch.Tensor, base_pose: dict[str, float]) -> dict[str, float]:
        pose = dict(base_pose)
        for name in self.joint_names:
            joint_ids, _joint_names = robot.find_joints([name], preserve_order=True)
            if joint_ids:
                pose[name] = float(joint_target[0, int(joint_ids[0])].detach().cpu())
        return pose

    def hand_target_6d_world(
        self,
        robot: Articulation,
        target: TargetCandidate,
        pose_name: str,
        side: str,
        *,
        carried_key: str | None = None,
    ) -> list[float]:
        rx, ry, robot_yaw = robot_planar_pose(robot)
        forward_axis = np.array([math.cos(robot_yaw), math.sin(robot_yaw), 0.0], dtype=np.float64)
        left_axis = np.array([-math.sin(robot_yaw), math.cos(robot_yaw), 0.0], dtype=np.float64)
        _sx, _sy, sz = object_size_for_key(target.key)
        if pose_name == "lift_carry":
            center = np.array(target.center_world, dtype=np.float64)
        elif carried_key:
            center_xyz, _yaw = carried_pose(robot)
            center = np.array(center_xyz, dtype=np.float64)
        else:
            center = np.array(target.center_world, dtype=np.float64)

        object_half_width = max(object_half_extent_along_axis(target.key, left_axis), 0.14)
        open_extra = 0.18 if pose_name == "ready_front_open" else 0.04
        if pose_name == "scoop_low":
            open_extra = 0.020
        if pose_name == "clamp_sides":
            open_extra = -float(args_cli.grasp_squeeze_m)
        if pose_name == "lift_carry":
            open_extra = -float(args_cli.lift_squeeze_m)
        side_sign = 1.0 if side == "left" else -1.0
        target_pos = center + side_sign * (object_half_width + open_extra) * left_axis
        if pose_name == "ready_front_open":
            target_pos -= 0.04 * forward_axis
            target_pos[2] = max(float(center[2]) + 0.16, 0.24)
        elif pose_name == "scoop_low":
            target_pos += float(args_cli.hand_forward_compensation_m) * forward_axis
            target_pos[2] = max(float(center[2]) + float(args_cli.scoop_z_offset_m), 0.13)
        elif pose_name == "clamp_sides":
            target_pos += float(args_cli.hand_forward_compensation_m) * forward_axis
            target_pos[2] = max(float(center[2]) + float(args_cli.clamp_z_offset_m), 0.15)
        elif pose_name == "lift_carry":
            target_pos += float(args_cli.hand_forward_compensation_m) * forward_axis
            target_pos[2] = max(float(args_cli.carry_height_m), float(center[2]) + 0.10)
        else:
            target_pos[2] = max(float(center[2]) + 0.12, 0.22)
        inward_yaw = wrap_to_pi(robot_yaw - side_sign * math.pi / 2.0)
        return [float(target_pos[0]), float(target_pos[1]), float(target_pos[2]), inward_yaw, 0.0, 0.0]

    def pose_from_world_6d(self, robot: Articulation, target_6d_w: list[float]) -> Any:
        from curobo.types.math import Pose

        rx, ry, robot_yaw = robot_planar_pose(robot)
        p_w = np.asarray(target_6d_w[:3], dtype=np.float64)
        rot_w = rotation_matrix_zyx(float(target_6d_w[3]), float(target_6d_w[4]), float(target_6d_w[5]))
        rot_bw = rotation_matrix_zyx(-robot_yaw, 0.0, 0.0)
        p_b = rot_bw @ (p_w - np.array([rx, ry, ROBOT_BASE_Z], dtype=np.float64))
        rot_b = rot_bw @ rot_w
        quat_b = np.asarray(quat_wxyz_from_matrix(rot_b), dtype=np.float32).reshape(1, 4)
        return Pose(
            position=self.tensor_args.to_device(p_b.astype(np.float32).reshape(1, 3)),
            quaternion=self.tensor_args.to_device(quat_b),
        )

    def joint_delta_check(self, seed: torch.Tensor, solution: np.ndarray) -> tuple[bool, dict[str, Any]]:
        seed_np = seed.detach().cpu().numpy().reshape(-1)
        max_arm_delta = 0.0
        max_lower_delta = 0.0
        worst_arm = ""
        worst_lower = ""
        for name, old_value, new_value in zip(self.joint_names, seed_np, solution.reshape(-1)):
            delta = abs(float(new_value) - float(old_value))
            if name in ("knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint"):
                if delta > max_lower_delta:
                    max_lower_delta = delta
                    worst_lower = name
            elif name.startswith("zarm_"):
                if delta > max_arm_delta:
                    max_arm_delta = delta
                    worst_arm = name
        ok = max_arm_delta <= float(args_cli.curobo_max_arm_joint_delta_rad) and max_lower_delta <= float(args_cli.curobo_max_lower_body_delta_rad)
        return ok, {
            "max_arm_delta_rad": float(args_cli.curobo_max_arm_joint_delta_rad),
            "max_lower_body_delta_rad": float(args_cli.curobo_max_lower_body_delta_rad),
            "observed_max_arm_delta_rad": float(max_arm_delta),
            "observed_max_lower_body_delta_rad": float(max_lower_delta),
            "worst_arm_joint": worst_arm,
            "worst_lower_body_joint": worst_lower,
        }

    def apply_solution_to_pose(self, pose: dict[str, float], solution: np.ndarray) -> dict[str, float]:
        updated = dict(pose)
        for name, value in zip(self.joint_names, solution.reshape(-1)):
            if name.startswith("waist_") and not args_cli.enable_waist:
                continue
            if name in ("knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint") and not args_cli.unlock_whole_body:
                continue
            updated[name] = float(value)
        return updated

    def solve_pose(
        self,
        robot: Articulation,
        joint_target: torch.Tensor,
        target: TargetCandidate,
        pose_name: str,
        heuristic_pose: dict[str, float],
        *,
        carried_key: str | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        meta: dict[str, Any] = {
            "backend": "curobo",
            "used": False,
            "available": bool(self.available),
            "phase": pose_name,
            "targets_6d_world": {},
            "fallback": "",
        }
        if not self.available or self.ik_solver is None or self.tensor_args is None:
            meta["fallback"] = self.status.get("reason", "cuRobo unavailable")
            return self.current_pose_from_joint_target(robot, joint_target, heuristic_pose), meta
        try:
            left_target = self.hand_target_6d_world(robot, target, pose_name, "left", carried_key=carried_key)
            right_target = self.hand_target_6d_world(robot, target, pose_name, "right", carried_key=carried_key)
            meta["targets_6d_world"] = {"left": left_target, "right": right_target}
            seed = self.seed_from_joint_target(robot, joint_target)
            right_goal = self.pose_from_world_6d(robot, right_target)
            left_goal = self.pose_from_world_6d(robot, left_target)
            result = self.ik_solver.solve_single(
                right_goal,
                retract_config=seed,
                seed_config=seed.view(1, 1, -1),
                num_seeds=max(1, int(args_cli.curobo_num_seeds)),
                return_seeds=1,
                link_poses={self.LEFT_EE_LINK: left_goal},
            )
            success = bool(result.success.reshape(-1)[0].item())
            meta["solve"] = {
                "success": success,
                "solve_time_s": float(getattr(result, "solve_time", 0.0) or 0.0),
                "position_error_m": float(result.position_error.detach().reshape(-1).max().cpu().item()),
                "rotation_error_rad": float(result.rotation_error.detach().reshape(-1).max().cpu().item()),
                "joint_names": list(self.joint_names),
            }
            if not success:
                meta["fallback"] = "solve_failed"
                return self.current_pose_from_joint_target(robot, joint_target, heuristic_pose), meta
            solution = result.solution.detach().cpu().numpy().reshape(-1).astype(np.float32)
            delta_ok, delta_meta = self.joint_delta_check(seed, solution)
            meta["joint_delta_check"] = delta_meta
            if not delta_ok:
                meta["fallback"] = "rejected_large_joint_delta"
                return self.current_pose_from_joint_target(robot, joint_target, heuristic_pose), meta
            curobo_pose = self.apply_solution_to_pose(heuristic_pose, solution)
        except Exception as exc:
            meta["fallback"] = f"solve_failed: {exc!r}"
            return self.current_pose_from_joint_target(robot, joint_target, heuristic_pose), meta
        meta["used"] = True
        return curobo_pose, meta


class KuavoOfficialIKBridge:
    """Offline bridge to Kuavo official IK, with no ROS node or service dependency."""

    CONSERVATIVE_LOWER_BODY_LIMITS = {
        "knee_joint": (0.0, 0.12),
        "leg_joint": (-0.18, 0.0),
        "waist_pitch_joint": (-0.03, 0.16),
        "waist_yaw_joint": (-0.16, 0.16),
    }
    LOWER_AND_ARM_JOINTS = [
        "knee_joint",
        "leg_joint",
        "waist_pitch_joint",
        "waist_yaw_joint",
        "zarm_l1_joint",
        "zarm_l2_joint",
        "zarm_l3_joint",
        "zarm_l4_joint",
        "zarm_l5_joint",
        "zarm_l6_joint",
        "zarm_l7_joint",
        "zarm_r1_joint",
        "zarm_r2_joint",
        "zarm_r3_joint",
        "zarm_r4_joint",
        "zarm_r5_joint",
        "zarm_r6_joint",
        "zarm_r7_joint",
    ]

    def __init__(self) -> None:
        self.available = False
        self.status: dict[str, Any] = {
            "backend": "official_standalone",
            "available": False,
            "lib": str(args_cli.official_ik_lib),
            "urdf": str(args_cli.official_ik_urdf),
            "task_info": str(args_cli.official_ik_task_info),
            "whole_body": bool(args_cli.official_ik_whole_body),
            "conservative_lower_body_limits": bool(args_cli.official_ik_conservative_lower_body_limits),
            "reason": "",
        }
        self.left_solver: Any | None = None
        self.right_solver: Any | None = None
        self.official_joint_names = list(self.LOWER_AND_ARM_JOINTS)
        if KuavoKinematicsSolver is None:
            self.status["reason"] = f"import_failed: {KUAVO_IK_IMPORT_ERROR}"
            return
        try:
            common_kwargs = {
                "lib_path": args_cli.official_ik_lib,
                "urdf_path": args_cli.official_ik_urdf,
                "task_info_path": args_cli.official_ik_task_info,
                "is_whole_body": bool(args_cli.official_ik_whole_body),
                "linear_error_max": float(args_cli.official_ik_linear_error_max),
                "angular_error_max": float(args_cli.official_ik_angular_error_max),
            }
            self.left_solver = KuavoKinematicsSolver(arm_index=0, **common_kwargs)
            self.right_solver = KuavoKinematicsSolver(arm_index=1, **common_kwargs)
        except Exception as exc:
            self.status["reason"] = f"init_failed: {exc!r}"
            return
        self.available = True
        solver_dof_names = list(getattr(self.left_solver, "dof_names", []) or [])
        if len(solver_dof_names) == len(self.LOWER_AND_ARM_JOINTS):
            self.official_joint_names = solver_dof_names
        self.status["available"] = True
        self.status["reason"] = "ready"
        self.status["arm_dim"] = int(getattr(self.left_solver, "arm_dim", 0))
        self.status["state_dim"] = int(getattr(self.left_solver, "state_dim", 0))
        self.status["dof_names"] = list(self.official_joint_names)

    def seed_from_joint_target(self, robot: Articulation, joint_target: torch.Tensor) -> list[float]:
        seed: list[float] = []
        state_dim = int(getattr(self.left_solver, "state_dim", 0) or 0) if self.left_solver is not None else 0
        arm_dim = len(self.official_joint_names)
        base_dim = max(0, state_dim - arm_dim)
        if base_dim == 3:
            rx, ry, robot_yaw = robot_planar_pose(robot)
            seed.extend([float(rx), float(ry), float(robot_yaw)])
        elif base_dim > 0:
            seed.extend([0.0] * base_dim)
        for name in self.official_joint_names:
            joint_ids, _joint_names = robot.find_joints([name], preserve_order=True)
            if joint_ids:
                seed.append(float(joint_target[0, int(joint_ids[0])].detach().cpu()))
            else:
                seed.append(0.0)
        return seed

    def current_pose_from_joint_target(self, robot: Articulation, joint_target: torch.Tensor, base_pose: dict[str, float]) -> dict[str, float]:
        pose = dict(base_pose)
        for name in self.official_joint_names:
            joint_ids, _joint_names = robot.find_joints([name], preserve_order=True)
            if joint_ids:
                pose[name] = float(joint_target[0, int(joint_ids[0])].detach().cpu())
        return pose

    def command_q_from_solution(self, q_best: list[float]) -> list[float]:
        commanded: list[float] = []
        for name, value in zip(self.official_joint_names, q_best):
            if bool(args_cli.official_ik_conservative_lower_body_limits) and name in self.CONSERVATIVE_LOWER_BODY_LIMITS:
                value = clamp(float(value), *self.CONSERVATIVE_LOWER_BODY_LIMITS[name])
            commanded.append(float(value))
        return commanded

    def joint_delta_check(self, seed: list[float], commanded_q: list[float]) -> tuple[bool, dict[str, Any]]:
        seed_arm = seed[-len(self.official_joint_names) :] if len(seed) >= len(self.official_joint_names) else seed
        max_arm_delta = 0.0
        max_lower_delta = 0.0
        worst_arm = ""
        worst_lower = ""
        for name, old_value, new_value in zip(self.official_joint_names, seed_arm, commanded_q):
            delta = abs(float(new_value) - float(old_value))
            if name in ("knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint"):
                if delta > max_lower_delta:
                    max_lower_delta = delta
                    worst_lower = name
            elif name.startswith("zarm_"):
                if delta > max_arm_delta:
                    max_arm_delta = delta
                    worst_arm = name
        limits = {
            "max_arm_delta_rad": float(args_cli.official_ik_max_arm_joint_delta_rad),
            "max_lower_body_delta_rad": float(args_cli.official_ik_max_lower_body_delta_rad),
        }
        ok = (
            max_arm_delta <= limits["max_arm_delta_rad"]
            and max_lower_delta <= limits["max_lower_body_delta_rad"]
        )
        return ok, {
            **limits,
            "observed_max_arm_delta_rad": float(max_arm_delta),
            "observed_max_lower_body_delta_rad": float(max_lower_delta),
            "worst_arm_joint": worst_arm,
            "worst_lower_body_joint": worst_lower,
        }

    def apply_q_to_pose(self, pose: dict[str, float], q_best: list[float]) -> dict[str, float]:
        if len(q_best) < len(self.official_joint_names):
            raise ValueError(f"official IK returned {len(q_best)} joints, expected at least {len(self.official_joint_names)}")
        updated = dict(pose)
        for name, value in zip(self.official_joint_names, q_best[: len(self.official_joint_names)]):
            if name.startswith("waist_") and not args_cli.enable_waist:
                continue
            if name in ("knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint") and not args_cli.unlock_whole_body:
                continue
            if bool(args_cli.official_ik_conservative_lower_body_limits) and name in self.CONSERVATIVE_LOWER_BODY_LIMITS:
                value = clamp(float(value), *self.CONSERVATIVE_LOWER_BODY_LIMITS[name])
            updated[name] = float(value)
        return updated

    def hand_target_6d(
        self,
        robot: Articulation,
        target: TargetCandidate,
        pose_name: str,
        side: str,
        *,
        carried_key: str | None = None,
    ) -> list[float]:
        rx, ry, robot_yaw = robot_planar_pose(robot)
        forward_axis = np.array([math.cos(robot_yaw), math.sin(robot_yaw), 0.0], dtype=np.float64)
        left_axis = np.array([-math.sin(robot_yaw), math.cos(robot_yaw), 0.0], dtype=np.float64)
        _sx, _sy, sz = object_size_for_key(target.key)

        if pose_name == "lift_carry":
            center = np.array(target.center_world, dtype=np.float64)
        elif carried_key:
            center_xyz, _yaw = carried_pose(robot)
            center = np.array(center_xyz, dtype=np.float64)
        else:
            center = np.array(target.center_world, dtype=np.float64)

        object_half_width = max(object_half_extent_along_axis(target.key, left_axis), 0.14)
        open_extra = 0.18 if pose_name == "ready_front_open" else 0.04
        if pose_name == "scoop_low":
            open_extra = 0.020
        if pose_name == "clamp_sides":
            open_extra = -float(args_cli.grasp_squeeze_m)
        if pose_name == "lift_carry":
            open_extra = -float(args_cli.lift_squeeze_m)
        side_sign = 1.0 if side == "left" else -1.0
        target_pos = center + side_sign * (object_half_width + open_extra) * left_axis
        if pose_name == "ready_front_open":
            target_pos -= 0.04 * forward_axis
            target_pos[2] = max(float(center[2]) + 0.16, 0.24)
        elif pose_name == "scoop_low":
            target_pos += float(args_cli.hand_forward_compensation_m) * forward_axis
            target_pos[2] = max(float(center[2]) + float(args_cli.scoop_z_offset_m), 0.13)
        elif pose_name == "clamp_sides":
            target_pos += float(args_cli.hand_forward_compensation_m) * forward_axis
            target_pos[2] = max(float(center[2]) + float(args_cli.clamp_z_offset_m), 0.15)
        elif pose_name == "lift_carry":
            target_pos += float(args_cli.hand_forward_compensation_m) * forward_axis
            target_pos[2] = max(float(args_cli.carry_height_m), float(center[2]) + 0.10)
        else:
            target_pos[2] = max(float(center[2]) + 0.12, 0.22)

        # Kuavo wrapper expects [x, y, z, yaw, pitch, roll] in world frame using ZYX Euler.
        inward_yaw = wrap_to_pi(robot_yaw - side_sign * math.pi / 2.0)
        return [float(target_pos[0]), float(target_pos[1]), float(target_pos[2]), inward_yaw, 0.0, 0.0]

    def solve_pose(
        self,
        robot: Articulation,
        joint_target: torch.Tensor,
        target: TargetCandidate,
        pose_name: str,
        heuristic_pose: dict[str, float],
        *,
        carried_key: str | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        meta: dict[str, Any] = {
            "backend": "official_standalone",
            "used": False,
            "available": bool(self.available),
            "phase": pose_name,
            "targets_6d": {},
            "solves": [],
            "fallback": "",
        }
        if not self.available or self.left_solver is None or self.right_solver is None:
            meta["fallback"] = self.status.get("reason", "official IK unavailable")
            return self.current_pose_from_joint_target(robot, joint_target, heuristic_pose), meta

        seed = self.seed_from_joint_target(robot, joint_target)
        try:
            left_target = self.hand_target_6d(robot, target, pose_name, "left", carried_key=carried_key)
            right_target = self.hand_target_6d(robot, target, pose_name, "right", carried_key=carried_key)
            meta["targets_6d"] = {"left": left_target, "right": right_target}

            q_left = self.left_solver.solve_ik(left_target, seed)
            meta["solves"].append(
                {
                    "side": "left",
                    "status": int(self.left_solver.last_status),
                    "linear_error_m": float(self.left_solver.last_linear_error),
                    "angular_error_rad": float(self.left_solver.last_angular_error),
                    "seed": self.left_solver.last_seed_source,
                }
            )
            q_right = self.right_solver.solve_ik(right_target, q_left)
            meta["solves"].append(
                {
                    "side": "right",
                    "status": int(self.right_solver.last_status),
                    "linear_error_m": float(self.right_solver.last_linear_error),
                    "angular_error_rad": float(self.right_solver.last_angular_error),
                    "seed": self.right_solver.last_seed_source,
                }
            )
            commanded_q = self.command_q_from_solution(q_right)
            delta_ok, delta_meta = self.joint_delta_check(seed, commanded_q)
            meta["joint_delta_check"] = delta_meta
            if bool(args_cli.official_ik_reject_large_joint_delta) and not delta_ok:
                meta["fallback"] = "rejected_large_joint_delta"
                return self.current_pose_from_joint_target(robot, joint_target, heuristic_pose), meta
            official_pose = self.apply_q_to_pose(heuristic_pose, commanded_q)
        except Exception as exc:
            meta["fallback"] = f"solve_failed: {exc!r}"
            return self.current_pose_from_joint_target(robot, joint_target, heuristic_pose), meta

        meta["used"] = True
        return official_pose, meta


def heuristic_dual_arm_pose(robot: Articulation, target: TargetCandidate, pose_name: str) -> dict[str, float]:
    pose = {k: float(v) for k, v in DUAL_ARM_POSES[pose_name].items()}
    if not args_cli.enable_waist:
        return pose
    rx, ry, robot_yaw = robot_planar_pose(robot)
    tx, ty, tz = target.center_world
    target_bearing = wrap_to_pi(yaw_toward_xy(rx, ry, tx, ty) - robot_yaw)
    distance = math.hypot(tx - rx, ty - ry)
    sx, sy, sz = object_size_for_key(target.key)
    waist_yaw = clamp(0.42 * target_bearing, -0.16, 0.16)
    base_pitch = float(pose.get("waist_pitch_joint", 0.0))
    reach_pitch = clamp(0.16 - 0.08 * (distance - 0.65) + 0.05 * max(0.0, 0.28 - tz), 0.02, 0.16)
    if pose_name == "scoop_low":
        reach_pitch = clamp(0.18 + 0.04 * max(0.0, sz - 0.28), 0.14, 0.24)
    if pose_name == "clamp_sides":
        reach_pitch = clamp(0.16 + 0.04 * max(0.0, sz - 0.28), 0.12, 0.22)
    if pose_name == "lift_carry":
        reach_pitch = clamp(0.07 + 0.03 * max(0.0, sz - 0.22), 0.04, 0.12)
    pose["waist_yaw_joint"] = waist_yaw
    pose["waist_pitch_joint"] = max(base_pitch, reach_pitch)

    # Wider objects need the shoulders slightly more open before the squeeze.
    grasp_width = max(sx, sy)
    open_delta = clamp((grasp_width - 0.32) * 0.45, -0.04, 0.12)
    if pose_name in {"ready_front_open", "scoop_low", "clamp_sides"}:
        pose["zarm_l1_joint"] = float(pose.get("zarm_l1_joint", 0.0)) - open_delta
        pose["zarm_r1_joint"] = float(pose.get("zarm_r1_joint", 0.0)) + open_delta
    return pose


def dynamic_dual_arm_pose(
    robot: Articulation,
    locked_joint_target: torch.Tensor,
    target: TargetCandidate,
    pose_name: str,
    *,
    carried_key: str | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    heuristic_pose = heuristic_dual_arm_pose(robot, target, pose_name)
    if args_cli.ik_backend == "heuristic":
        return heuristic_pose, {"backend": "heuristic", "used": True, "phase": pose_name}
    if args_cli.ik_backend == "official_standalone":
        if OFFICIAL_IK_BRIDGE is None:
            return heuristic_pose, {"backend": "official_standalone", "used": False, "fallback": "bridge_not_initialized", "phase": pose_name}
        return OFFICIAL_IK_BRIDGE.solve_pose(robot, locked_joint_target, target, pose_name, heuristic_pose, carried_key=carried_key)
    if args_cli.ik_backend == "curobo":
        if CUROBO_IK_BRIDGE is None:
            return heuristic_pose, {"backend": "curobo", "used": False, "fallback": "bridge_not_initialized", "phase": pose_name}
        return CUROBO_IK_BRIDGE.solve_pose(robot, locked_joint_target, target, pose_name, heuristic_pose, carried_key=carried_key)
    if OFFICIAL_IK_BRIDGE is None:
        return heuristic_pose, {"backend": "official_standalone", "used": False, "fallback": "bridge_not_initialized", "phase": pose_name}
    return OFFICIAL_IK_BRIDGE.solve_pose(robot, locked_joint_target, target, pose_name, heuristic_pose, carried_key=carried_key)


def transition_manipulation_pose(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    recorder: ObserverVideoRecorder,
    pose_name: str,
    target: TargetCandidate,
    steps: int,
    *,
    carried_key: str | None = None,
) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
    named_pose, ik_meta = dynamic_dual_arm_pose(robot, locked_joint_target, target, pose_name, carried_key=carried_key)
    next_target = apply_named_joint_positions(robot, locked_joint_target, named_pose)
    start = locked_joint_target.clone()
    for step in range(max(1, int(steps))):
        alpha = min_jerk((step + 1) / max(1, int(steps)))
        current = start + alpha * (next_target - start)
        step_sim(sim, scene, robot, wheel_ids, current, recorder, carried_key=carried_key)
    return next_target, named_pose, ik_meta


def contact_force_for_target(scene: InteractiveScene, sensor_name: str, scene_key: str) -> float:
    if sensor_name not in scene.keys():
        return 0.0
    sensor = scene[sensor_name]
    try:
        matrix = getattr(sensor.data, "force_matrix_w", None)
        if matrix is not None:
            filter_index = [str(spec["key"]) for spec in OBJECT_SPECS].index(scene_key)
            forces = matrix[0, :, filter_index, :]
            return float(torch.linalg.norm(forces, dim=-1).sum().item())
        net = getattr(sensor.data, "net_forces_w", None)
        if net is not None:
            return float(torch.linalg.norm(net[0], dim=-1).sum().item())
    except Exception:
        return 0.0
    return 0.0


def body_position(robot: Articulation, name_expr: str) -> tuple[float, float, float] | None:
    body_ids, _body_names = robot.find_bodies(name_expr, preserve_order=False)
    if not body_ids:
        return None
    pos = robot.data.body_link_pose_w[0, int(body_ids[-1]), 0:3].detach().cpu()
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def first_available_body_position(robot: Articulation, name_exprs: list[str]) -> tuple[float, float, float] | None:
    for name_expr in name_exprs:
        body_ids, _body_names = robot.find_bodies(name_expr, preserve_order=True)
        if body_ids:
            pos = robot.data.body_link_pose_w[0, int(body_ids[-1]), 0:3].detach().cpu()
            return (float(pos[0]), float(pos[1]), float(pos[2]))
    return None


def arm_end_effector_position(robot: Articulation, side: str) -> tuple[float, float, float] | None:
    prefix = "zarm_l" if side == "left" else "zarm_r"
    return first_available_body_position(
        robot,
        [
            f"{prefix}7_end_effector",
            f"{prefix}7.*end_effector",
            f"{prefix}7_link",
            f"{prefix}7.*",
            f"{prefix}6_link",
        ],
    )


def point_to_oriented_box_surface_distance(
    point_w: tuple[float, float, float],
    center_w: tuple[float, float, float],
    size: tuple[float, float, float],
    yaw: float,
) -> float:
    point = np.asarray(point_w, dtype=np.float64)
    center = np.asarray(center_w, dtype=np.float64)
    delta = point - center
    c = math.cos(-float(yaw))
    s = math.sin(-float(yaw))
    local = np.array(
        [
            c * delta[0] - s * delta[1],
            s * delta[0] + c * delta[1],
            delta[2],
        ],
        dtype=np.float64,
    )
    half = 0.5 * np.asarray(size, dtype=np.float64)
    outside = np.maximum(np.abs(local) - half, 0.0)
    return float(np.linalg.norm(outside))


def grasp_gate_metrics(scene: InteractiveScene, robot: Articulation, target: TargetCandidate) -> dict[str, Any]:
    center_tuple = scene_object_center(scene, target.key)
    center = np.asarray(center_tuple, dtype=np.float64)
    left_pos = arm_end_effector_position(robot, "left")
    right_pos = arm_end_effector_position(robot, "right")
    left_distance = float(np.linalg.norm(np.asarray(left_pos, dtype=np.float64) - center)) if left_pos else float("inf")
    right_distance = float(np.linalg.norm(np.asarray(right_pos, dtype=np.float64) - center)) if right_pos else float("inf")
    sx, sy, sz = object_size_for_key(target.key)
    yaw = object_yaw(scene, target.key)
    left_surface_distance = (
        point_to_oriented_box_surface_distance(left_pos, center_tuple, (sx, sy, sz), yaw) if left_pos else float("inf")
    )
    right_surface_distance = (
        point_to_oriented_box_surface_distance(right_pos, center_tuple, (sx, sy, sz), yaw) if right_pos else float("inf")
    )
    center_distance_threshold = float(args_cli.contact_distance_threshold_m) + 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz)
    surface_distance_threshold = float(args_cli.contact_surface_distance_threshold_m)
    left_force = contact_force_for_target(scene, "left_arm_contact", target.key)
    right_force = contact_force_for_target(scene, "right_arm_contact", target.key)
    contact_ok = left_force >= float(args_cli.contact_force_threshold_n) and right_force >= float(args_cli.contact_force_threshold_n)
    distance_ok = left_surface_distance <= surface_distance_threshold and right_surface_distance <= surface_distance_threshold
    mode = args_cli.grasp_gate_mode
    if mode == "contact_only":
        gate_ok = contact_ok
    elif mode == "distance_only":
        gate_ok = distance_ok
    else:
        gate_ok = contact_ok or distance_ok
    return {
        "mode": mode,
        "gate_ok": bool(gate_ok),
        "contact_ok": bool(contact_ok),
        "distance_ok": bool(distance_ok),
        "left_contact_force_n": left_force,
        "right_contact_force_n": right_force,
        "contact_force_threshold_n": float(args_cli.contact_force_threshold_n),
        "left_surface_distance_m": left_surface_distance,
        "right_surface_distance_m": right_surface_distance,
        "surface_distance_threshold_m": surface_distance_threshold,
        "left_ee_distance_m": left_distance,
        "right_ee_distance_m": right_distance,
        "center_distance_threshold_m": center_distance_threshold,
        "left_ee_pos": [float(v) for v in left_pos] if left_pos else None,
        "right_ee_pos": [float(v) for v in right_pos] if right_pos else None,
    }


def settle_object(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    recorder: ObserverVideoRecorder,
    scene_key: str,
    steps: int,
) -> None:
    clear_object_wrench(scene, scene_key)
    for _ in range(max(1, int(steps))):
        step_sim(sim, scene, robot, wheel_ids, locked_joint_target, recorder)


def stabilize_carried_object(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    recorder: ObserverVideoRecorder,
    scene_key: str,
    steps: int,
) -> None:
    for _ in range(max(1, int(steps))):
        step_sim(sim, scene, robot, wheel_ids, locked_joint_target, recorder, carried_key=scene_key)


def navigate_to_pose(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    recorder: ObserverVideoRecorder,
    target_pose: tuple[float, float, float],
    label: str,
    *,
    carried_key: str | None = None,
) -> dict[str, Any]:
    initial = robot_planar_pose(robot)
    distance = math.hypot(float(target_pose[0]) - initial[0], float(target_pose[1]) - initial[1])
    yaw_delta = wrap_to_pi(float(target_pose[2]) - initial[2])
    travel_steps = int(max(60, distance / max(float(args_cli.nav_max_linear_speed), 1.0e-3) / sim.get_physics_dt()))
    turn_steps = int(abs(yaw_delta) / max(float(args_cli.nav_max_angular_speed), 1.0e-3) / sim.get_physics_dt())
    base_steps = max(travel_steps, turn_steps, 1)
    if carried_key and args_cli.carry_mode in {"pure_contact", "physical_soft_constraint"}:
        scale = clamp(float(args_cli.carried_nav_speed_scale), 0.05, 1.0)
        steps = min(max(1, int(args_cli.carried_nav_max_steps)), max(base_steps, int(base_steps / scale), 1))
    else:
        steps = min(max(1, int(args_cli.nav_max_steps)), base_steps)
    status = "SUCCEEDED"
    prev_pose = tuple(initial)
    for step in range(steps):
        alpha = min_jerk((step + 1) / steps)
        next_yaw = wrap_to_pi(initial[2] + alpha * yaw_delta)
        next_pose = (
            initial[0] + alpha * (float(target_pose[0]) - initial[0]),
            initial[1] + alpha * (float(target_pose[1]) - initial[1]),
            next_yaw,
        )
        write_robot_planar_pose(robot, next_pose)
        step_sim(sim, scene, robot, wheel_ids, locked_joint_target, recorder, carried_key=carried_key)
        prev_pose = next_pose
    write_robot_planar_pose(robot, target_pose)
    if carried_key and args_cli.carry_mode == "dual_arm_constraint":
        update_carried_object(scene, robot, carried_key)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim.get_physics_dt())
        recorder.capture(scene)
    apply_raw_wheel_velocity(robot, wheel_ids, [0.0, 0.0, 0.0, 0.0])
    final = robot_planar_pose(robot)
    pos_error = math.hypot(float(target_pose[0]) - final[0], float(target_pose[1]) - final[1])
    yaw_error = wrap_to_pi(float(target_pose[2]) - final[2])
    return {
        "label": label,
        "backend": args_cli.nav_backend,
        "target_pose": [float(v) for v in target_pose],
        "initial_pose": initial,
        "final_pose": final,
        "position_error_m": float(pos_error),
        "yaw_error_rad": float(yaw_error),
        "steps": int(steps),
        "success": status == "SUCCEEDED",
        "status": status,
    }


def navigate_waypoints(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    recorder: ObserverVideoRecorder,
    waypoints: list[tuple[float, float, float]],
    label: str,
    *,
    carried_key: str | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, waypoint in enumerate(waypoints):
        result = navigate_to_pose(
            sim,
            scene,
            robot,
            wheel_ids,
            locked_joint_target,
            recorder,
            waypoint,
            f"{label}_wp{idx:02d}",
            carried_key=carried_key,
        )
        results.append(result)
        if not result["success"]:
            break
    return results


def target_with_current_center(scene: InteractiveScene, target: TargetCandidate) -> TargetCandidate:
    return TargetCandidate(
        key=target.key,
        name=target.name,
        category=target.category,
        center_world=scene_object_center(scene, target.key),
        confidence=target.confidence,
        source=target.source,
        bbox_xyxy=target.bbox_xyxy,
        reason=f"{target.reason}; retargeted_to_current_physics_center",
    )


def center_base_under_lifted_object(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    recorder: ObserverVideoRecorder,
    target: TargetCandidate,
) -> dict[str, Any]:
    start_pose = robot_planar_pose(robot)
    object_center = scene_object_center(scene, target.key)
    yaw = float(start_pose[2])
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    delta_xy = np.array([object_center[0] - start_pose[0], object_center[1] - start_pose[1]], dtype=np.float64)
    forward_offset = float(np.dot(delta_xy, forward))
    desired = float(args_cli.carry_forward_m)
    correction = clamp(
        forward_offset - desired,
        0.0,
        max(0.0, float(args_cli.carry_center_max_step_m)),
    )
    steps = max(1, int(args_cli.carry_center_steps))
    if correction <= float(args_cli.carry_center_tolerance_m):
        return {
            "enabled": bool(args_cli.carry_center_after_lift),
            "applied": False,
            "initial_robot_pose": start_pose,
            "object_center": [float(v) for v in object_center],
            "forward_offset_m": forward_offset,
            "desired_forward_offset_m": desired,
            "correction_m": 0.0,
            "steps": 0,
        }
    target_pose = (
        start_pose[0] + correction * math.cos(yaw),
        start_pose[1] + correction * math.sin(yaw),
        yaw,
    )
    for step in range(steps):
        alpha = min_jerk((step + 1) / steps)
        pose = (
            start_pose[0] + alpha * (target_pose[0] - start_pose[0]),
            start_pose[1] + alpha * (target_pose[1] - start_pose[1]),
            yaw,
        )
        write_robot_planar_pose(robot, pose)
        step_sim(sim, scene, robot, wheel_ids, locked_joint_target, recorder, carried_key=target.key)
    return {
        "enabled": bool(args_cli.carry_center_after_lift),
        "applied": True,
        "initial_robot_pose": start_pose,
        "final_robot_pose": robot_planar_pose(robot),
        "object_center_before": [float(v) for v in object_center],
        "object_center_after": [float(v) for v in scene_object_center(scene, target.key)],
        "forward_offset_m": forward_offset,
        "desired_forward_offset_m": desired,
        "correction_m": correction,
        "steps": steps,
    }


def camera_to_world_matrix(camera: CameraCfg) -> np.ndarray:
    pos_w = to_numpy(camera.data.pos_w)[0].astype(np.float32)
    rot_w = to_numpy(matrix_from_quat(camera.data.quat_w_ros))[0].astype(np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot_w
    mat[:3, 3] = pos_w
    return mat


def capture_head_camera(scene: InteractiveScene, out_dir: Path, prefix: str) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera = scene["head_rgbd"]
    rgb = sanitize_rgb(to_numpy(camera.data.output["rgb"])[0])
    depth = sanitize_depth(to_numpy(camera.data.output["distance_to_image_plane"])[0])
    intrinsics = to_numpy(camera.data.intrinsic_matrices)[0].astype(np.float32)
    t_camera_to_world = camera_to_world_matrix(camera)
    rgb_path = out_dir / f"{prefix}_rgb.png"
    Image.fromarray(rgb).save(rgb_path)
    depth_vis = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if np.max(depth_vis) > 0:
        depth_vis = np.clip(depth_vis / np.percentile(depth_vis[depth_vis > 0], 95), 0, 1)
    Image.fromarray((depth_vis * 255).astype(np.uint8)).save(out_dir / f"{prefix}_depth.png")
    return rgb_path, rgb, depth, intrinsics, t_camera_to_world


def scene_object_center(scene: InteractiveScene, key: str) -> tuple[float, float, float]:
    pos = to_numpy(scene[key].data.root_state_w[0, :3]).astype(np.float32)
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def mock_scene_perception(scene: InteractiveScene, completed: set[str]) -> list[TargetCandidate]:
    candidates: list[TargetCandidate] = []
    for spec in OBJECT_SPECS:
        key = str(spec["key"])
        if key in completed or key not in scene.keys():
            continue
        candidates.append(
            TargetCandidate(
                key=key,
                name=str(spec["name"]),
                category=normalize_category(str(spec["category"])),
                center_world=scene_object_center(scene, key),
                confidence=1.0,
                source="mock_scene",
                reason="ground_truth_scene_object",
            )
        )
    return candidates


def run_viss_perception(rgb_path: Path, out_dir: Path) -> list[dict[str, Any]]:
    result_path = out_dir / "viss_result.json"
    overlay_path = out_dir / "viss_overlay.jpg"
    cmd = [
        sys.executable,
        str(args_cli.viss_qwen_script),
        "--pipeline",
        "qwen_first",
        "--image",
        str(rgb_path),
        "--model",
        str(args_cli.viss_qwen_model),
        "--conf",
        str(args_cli.viss_qwen_conf),
        "--roi-expand",
        "2.0",
        "--verify-mode",
        "none",
        "--max-qwen-candidates",
        "10",
        "--max-roi-refine",
        "10",
        "--save-vis",
    ]
    env = os.environ.copy()
    viss_home = out_dir / "viss_home"
    env["HOME"] = str(viss_home)
    if not env.get("QWEN_API_KEY") and env.get("DASHSCOPE_API_KEY"):
        env["QWEN_API_KEY"] = env["DASHSCOPE_API_KEY"]
    env.setdefault("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    env.setdefault("QWEN_MODEL", "qwen3-vl-flash")
    proc = subprocess.run(cmd, cwd=str(PARENT_ROOT), env=env, capture_output=True, text=True, timeout=float(args_cli.viss_qwen_timeout_s), check=False)
    (out_dir / "viss_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / "viss_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    default_result = viss_home / "trashbot_ws/data/logs/yolo_seg_offline_result.json"
    default_overlay = viss_home / "trashbot_ws/data/logs/yolo11_qwen_overlay_latest.jpg"
    if default_result.exists():
        shutil.copy2(default_result, result_path)
    if default_overlay.exists():
        shutil.copy2(default_overlay, overlay_path)
    if proc.returncode != 0 or not result_path.exists():
        return []
    result = json.loads(result_path.read_text(encoding="utf-8"))
    detections: list[dict[str, Any]] = []
    for group in ("detections", "approach_candidates"):
        for det in result.get(group, []) or []:
            if isinstance(det, dict):
                detections.append(det)
    return detections


def viss_candidates(scene: InteractiveScene, completed: set[str], detections: list[dict[str, Any]]) -> list[TargetCandidate]:
    remaining = [str(spec["key"]) for spec in OBJECT_SPECS if str(spec["key"]) not in completed]
    candidates: list[TargetCandidate] = []
    used: set[str] = set()
    for det in detections:
        category = normalize_category(det.get("category") or det.get("garbage_category") or det.get("garbage_category_cn"))
        best_key = None
        for key in remaining:
            if key in used:
                continue
            spec = OBJECT_BY_KEY[key]
            if normalize_category(str(spec["category"])) == category:
                best_key = key
                break
        if best_key is None:
            continue
        used.add(best_key)
        spec = OBJECT_BY_KEY[best_key]
        candidates.append(
            TargetCandidate(
                key=best_key,
                name=str(det.get("raw_class_name") or det.get("class_name") or spec["name"]),
                category=category,
                center_world=scene_object_center(scene, best_key),
                confidence=float(det.get("confidence", det.get("yolo_confidence", 0.5)) or 0.5),
                source="viss_qwen_first",
                bbox_xyxy=[float(v) for v in det.get("bbox_xyxy", [])] if det.get("bbox_xyxy") else None,
                reason="viss_category_matched_to_scene_object",
            )
        )
    if not candidates:
        return mock_scene_perception(scene, completed)
    return candidates


def perceive_targets(scene: InteractiveScene, cycle_dir: Path, completed: set[str]) -> tuple[list[TargetCandidate], dict[str, Any]]:
    rgb_path, _rgb, _depth, _intrinsics, _t_camera_to_world = capture_head_camera(scene, cycle_dir, "observe")
    if args_cli.perception_source == "mock_scene":
        candidates = mock_scene_perception(scene, completed)
        meta = {"source": "mock_scene", "rgb": str(rgb_path), "detections": []}
    else:
        detections = run_viss_perception(rgb_path, cycle_dir)
        candidates = viss_candidates(scene, completed, detections)
        meta = {"source": "viss_qwen_first", "rgb": str(rgb_path), "detection_count": len(detections), "fallback_to_scene": not bool(detections)}
    return candidates, meta


def select_target(robot: Articulation, candidates: list[TargetCandidate]) -> tuple[TargetCandidate | None, str]:
    if not candidates:
        return None, "no_candidates"
    robot_x, robot_y, _ = robot_planar_pose(robot)
    selected = min(candidates, key=lambda item: (math.hypot(item.center_world[0] - robot_x, item.center_world[1] - robot_y), -item.confidence))
    return selected, "nearest_uncompleted_large_floor_object"


def pick_standpoint_for_target(target: TargetCandidate) -> tuple[float, float, float]:
    tx, ty, _tz = target.center_world
    min_x, _max_x, min_y, max_y = trash_obstacle_bounds(exclude={target.key})
    center_y = 0.5 * (min_y + max_y)
    side_band = 0.28
    if ty < center_y - side_band:
        x = tx
        y = ty - float(args_cli.pick_standoff_m)
    elif ty > center_y + side_band:
        x = tx
        y = ty + float(args_cli.pick_standoff_m)
    else:
        x = tx - float(args_cli.pick_standoff_m)
        y = ty
    yaw = yaw_toward_xy(x, y, tx, ty)
    return (float(x), float(y), float(yaw))


def drop_pose_for_category(category: str) -> tuple[float, float, float]:
    spec = AREA_SPECS[normalize_category(category)]
    center = spec["center"]
    return (float(center[0]), float(center[1]), float(args_cli.drop_height_m))


def object_area_error(scene: InteractiveScene, scene_key: str, category: str) -> dict[str, Any]:
    pos = scene_object_center(scene, scene_key)
    spec = AREA_SPECS[normalize_category(category)]
    center = spec["center"]
    size = spec["size"]
    dx = abs(float(pos[0]) - float(center[0]))
    dy = abs(float(pos[1]) - float(center[1]))
    inside = dx <= 0.5 * float(size[0]) and dy <= 0.5 * float(size[1])
    return {
        "object_final_pos": [float(v) for v in pos],
        "area_center": [float(v) for v in center],
        "xy_error_m": float(math.hypot(float(pos[0]) - float(center[0]), float(pos[1]) - float(center[1]))),
        "inside_area_xy": bool(inside),
    }


def run_one_cycle(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    recorder: ObserverVideoRecorder,
    run_dir: Path,
    completed: set[str],
    cycle_index: int,
) -> tuple[bool, torch.Tensor, dict[str, Any]]:
    cycle_dir = run_dir / f"cycle_{cycle_index:04d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    candidates, perception_meta = perceive_targets(scene, cycle_dir, completed)
    target, reason = select_target(robot, candidates)
    meta: dict[str, Any] = {
        "cycle_index": cycle_index,
        "perception": perception_meta,
        "candidate_count": len(candidates),
        "candidates": [candidate_to_dict(item) for item in candidates],
        "selection_reason": reason,
        "selected": candidate_to_dict(target) if target else None,
        "navigation": [],
        "dual_arm": [],
        "success": False,
        "reason": "",
    }
    if target is None:
        meta["success"] = True
        meta["reason"] = "No remaining target."
        return False, locked_joint_target, meta

    pick_pose = pick_standpoint_for_target(target)
    pick_waypoints = path_around_trash(robot, pick_pose, exclude={target.key}, final_from_left=False)
    nav_pick_results = navigate_waypoints(sim, scene, robot, wheel_ids, locked_joint_target, recorder, pick_waypoints, f"pick_{target.key}")
    meta["navigation"].extend(nav_pick_results)
    if not nav_pick_results or not nav_pick_results[-1]["success"]:
        meta["reason"] = f"Failed to reach pick standpoint for {target.key}."
        return False, locked_joint_target, meta

    locked_joint_target, ready_pose, ready_ik = transition_manipulation_pose(
        sim, scene, robot, wheel_ids, locked_joint_target, recorder, "ready_front_open", target, args_cli.pose_steps
    )
    meta["dual_arm"].append({"state": "ready_front_open", "steps": int(args_cli.pose_steps), "waist_target": {
        "waist_yaw_joint": ready_pose.get("waist_yaw_joint"),
        "waist_pitch_joint": ready_pose.get("waist_pitch_joint"),
    }, "ik": ready_ik})
    locked_joint_target, scoop_pose, scoop_ik = transition_manipulation_pose(
        sim, scene, robot, wheel_ids, locked_joint_target, recorder, "scoop_low", target, args_cli.scoop_steps
    )
    meta["dual_arm"].append({"state": "scoop_low", "steps": int(args_cli.scoop_steps), "waist_target": {
        "waist_yaw_joint": scoop_pose.get("waist_yaw_joint"),
        "waist_pitch_joint": scoop_pose.get("waist_pitch_joint"),
    }, "ik": scoop_ik, "object_motion": "hands_moved_to_low_support_pose_without_object_pose_write"})
    locked_joint_target, clamp_pose, clamp_ik = transition_manipulation_pose(
        sim, scene, robot, wheel_ids, locked_joint_target, recorder, "clamp_sides", target, args_cli.clamp_steps
    )
    for _ in range(max(0, int(args_cli.post_clamp_settle_steps))):
        step_sim(sim, scene, robot, wheel_ids, locked_joint_target, recorder)
    grasp_metrics = grasp_gate_metrics(scene, robot, target)
    meta["dual_arm"].append({
        "state": "clamp_sides",
        "steps": int(args_cli.clamp_steps) + max(0, int(args_cli.post_clamp_settle_steps)),
        "waist_target": {
            "waist_yaw_joint": clamp_pose.get("waist_yaw_joint"),
            "waist_pitch_joint": clamp_pose.get("waist_pitch_joint"),
        },
        "ik": clamp_ik,
        "object_motion": "object_remained_at_floor_pose_until_grasp_gate",
        "grasp_gate": grasp_metrics,
    })
    if args_cli.carry_mode in {"pure_contact", "physical_soft_constraint"} and not bool(grasp_metrics["gate_ok"]):
        retry_distance = max(0.0, float(args_cli.contact_retry_forward_m))
        if retry_distance > 0.0:
            x, y, yaw = robot_planar_pose(robot)
            retry_pose = (x + retry_distance * math.cos(yaw), y + retry_distance * math.sin(yaw), yaw)
            write_robot_planar_pose(robot, retry_pose)
            locked_joint_target, retry_pose_named, retry_ik = transition_manipulation_pose(
                sim,
                scene,
                robot,
                wheel_ids,
                locked_joint_target,
                recorder,
                "clamp_sides",
                target,
                max(1, int(args_cli.contact_retry_steps)),
            )
            for _ in range(max(0, int(args_cli.post_clamp_settle_steps))):
                step_sim(sim, scene, robot, wheel_ids, locked_joint_target, recorder)
            grasp_metrics = grasp_gate_metrics(scene, robot, target)
            meta["dual_arm"].append({
                "state": "clamp_sides_retry_forward",
                "steps": int(args_cli.contact_retry_steps) + max(0, int(args_cli.post_clamp_settle_steps)),
                "base_nudge_m": retry_distance,
                "waist_target": {
                    "waist_yaw_joint": retry_pose_named.get("waist_yaw_joint"),
                    "waist_pitch_joint": retry_pose_named.get("waist_pitch_joint"),
                },
                "ik": retry_ik,
                "object_motion": "base_moved_closer_and_hands_reclamped_without_object_pose_write",
                "grasp_gate": grasp_metrics,
            })
    if args_cli.carry_mode in {"pure_contact", "physical_soft_constraint"} and not bool(grasp_metrics["gate_ok"]):
        meta["reason"] = f"Grasp gate failed for {target.key}; object was not moved."
        return False, locked_joint_target, meta

    if args_cli.carry_mode == "pure_contact":
        locked_joint_target, lift_pose, lift_ik = transition_manipulation_pose(
            sim,
            scene,
            robot,
            wheel_ids,
            locked_joint_target,
            recorder,
            "lift_carry",
            target,
            max(1, int(args_cli.lift_steps)),
            carried_key=target.key,
        )
        lift_motion = "pure_contact_only_no_object_pose_velocity_or_wrench_write"
    elif args_cli.carry_mode == "physical_soft_constraint":
        locked_joint_target, lift_pose, lift_ik = transition_manipulation_pose(
            sim,
            scene,
            robot,
            wheel_ids,
            locked_joint_target,
            recorder,
            "lift_carry",
            target,
            max(1, int(args_cli.lift_steps)),
            carried_key=target.key,
        )
        lift_motion = f"physical_{args_cli.physical_carry_controller}_after_grasp_gate_no_pose_write"
    else:
        locked_joint_target, lift_pose, lift_ik = transition_manipulation_pose(
            sim, scene, robot, wheel_ids, locked_joint_target, recorder, "lift_carry", target, max(1, int(args_cli.lift_steps) // 2)
        )
        lift_pos, lift_yaw = carried_pose(robot)
        smooth_object_pose_to(
            sim,
            scene,
            robot,
            wheel_ids,
            locked_joint_target,
            recorder,
            target.key,
            lift_pos,
            lift_yaw,
            max(1, int(args_cli.lift_steps)),
        )
        lift_motion = "legacy_pose_constraint_lift"
    meta["dual_arm"].append({
        "state": "lift_carry",
        "steps": int(args_cli.lift_steps),
        "waist_target": {
            "waist_yaw_joint": lift_pose.get("waist_yaw_joint"),
            "waist_pitch_joint": lift_pose.get("waist_pitch_joint"),
        },
        "ik": lift_ik,
        "carry_mode": args_cli.carry_mode,
        "physical_carry_controller": args_cli.physical_carry_controller if args_cli.carry_mode == "physical_soft_constraint" else None,
        "object_motion": lift_motion,
        "object_pose_after_lift": [float(v) for v in scene_object_center(scene, target.key)],
    })
    if args_cli.carry_mode in {"pure_contact", "physical_soft_constraint"} and bool(args_cli.carry_center_after_lift):
        center_meta = center_base_under_lifted_object(
            sim,
            scene,
            robot,
            wheel_ids,
            locked_joint_target,
            recorder,
            target,
        )
        current_target = target_with_current_center(scene, target)
        locked_joint_target, regrip_pose, regrip_ik = transition_manipulation_pose(
            sim,
            scene,
            robot,
            wheel_ids,
            locked_joint_target,
            recorder,
            "lift_carry",
            current_target,
            max(1, int(args_cli.lift_steps) // 2),
            carried_key=target.key,
        )
        meta["dual_arm"].append({
            "state": "carry_center_regrip",
            "steps": int(center_meta.get("steps", 0)) + max(1, int(args_cli.lift_steps) // 2),
            "base_centering": center_meta,
            "waist_target": {
                "waist_yaw_joint": regrip_pose.get("waist_yaw_joint"),
                "waist_pitch_joint": regrip_pose.get("waist_pitch_joint"),
            },
            "ik": regrip_ik,
            "object_motion": "base_centered_under_lifted_object_and_regripped_without_object_pose_write",
            "object_pose_after_regrip": [float(v) for v in scene_object_center(scene, target.key)],
        })

    area_nav = tuple(float(v) for v in AREA_SPECS[target.category]["nav_pose"])
    area_nav = (area_nav[0], area_nav[1], yaw_toward_xy(area_nav[0], area_nav[1], *AREA_SPECS[target.category]["center"][:2]))
    area_waypoints = path_around_trash(robot, area_nav, exclude=set(completed) | {target.key}, final_from_left=False)
    if args_cli.carry_mode in {"pure_contact", "physical_soft_constraint"} and int(args_cli.carried_waypoint_regrip_steps) > 0:
        area_waypoints = subdivide_planar_waypoints(area_waypoints, float(args_cli.carried_waypoint_spacing_m))
    nav_area_results: list[dict[str, Any]] = []
    for idx, waypoint in enumerate(area_waypoints):
        result = navigate_to_pose(
            sim,
            scene,
            robot,
            wheel_ids,
            locked_joint_target,
            recorder,
            waypoint,
            f"area_{target.category}_{target.key}_wp{idx:02d}",
            carried_key=target.key,
        )
        nav_area_results.append(result)
        if not result["success"]:
            break
        if (
            args_cli.carry_mode in {"pure_contact", "physical_soft_constraint"}
            and int(args_cli.carried_waypoint_regrip_steps) > 0
            and idx < len(area_waypoints) - 1
        ):
            center_meta = center_base_under_lifted_object(
                sim,
                scene,
                robot,
                wheel_ids,
                locked_joint_target,
                recorder,
                target,
            )
            current_target = target_with_current_center(scene, target)
            locked_joint_target, waypoint_regrip_pose, waypoint_regrip_ik = transition_manipulation_pose(
                sim,
                scene,
                robot,
                wheel_ids,
                locked_joint_target,
                recorder,
                "lift_carry",
                current_target,
                max(1, int(args_cli.carried_waypoint_regrip_steps)),
                carried_key=target.key,
            )
            meta["dual_arm"].append({
                "state": f"carried_waypoint_regrip_{idx:02d}",
                "steps": int(center_meta.get("steps", 0)) + max(1, int(args_cli.carried_waypoint_regrip_steps)),
                "base_centering": center_meta,
                "waist_target": {
                    "waist_yaw_joint": waypoint_regrip_pose.get("waist_yaw_joint"),
                    "waist_pitch_joint": waypoint_regrip_pose.get("waist_pitch_joint"),
                },
                "ik": waypoint_regrip_ik,
                "object_motion": "periodic_carry_regrip_without_object_pose_write",
                "object_pose_after_regrip": [float(v) for v in scene_object_center(scene, target.key)],
            })
    meta["navigation"].extend(nav_area_results)
    if not nav_area_results or not nav_area_results[-1]["success"]:
        if args_cli.carry_mode == "physical_soft_constraint":
            clear_object_wrench(scene, target.key)
        meta["reason"] = f"Failed to reach target sorting area for {target.key}."
        return False, locked_joint_target, meta

    drop_pos = drop_pose_for_category(target.category)
    if args_cli.carry_mode == "pure_contact":
        locked_joint_target, release_pose, release_ik = transition_manipulation_pose(
            sim,
            scene,
            robot,
            wheel_ids,
            locked_joint_target,
            recorder,
            "ready_front_open",
            target,
            args_cli.drop_steps,
            carried_key=target.key,
        )
        settle_object(sim, scene, robot, wheel_ids, locked_joint_target, recorder, target.key, args_cli.settle_after_release_steps)
        drop_motion = "pure_contact_release_no_object_pose_velocity_or_wrench_write"
    elif args_cli.carry_mode == "physical_soft_constraint":
        stabilize_carried_object(
            sim,
            scene,
            robot,
            wheel_ids,
            locked_joint_target,
            recorder,
            target.key,
            args_cli.pre_release_stabilize_steps,
        )
        object_pose_before_release = [float(v) for v in scene_object_center(scene, target.key)]
        clear_object_wrench(scene, target.key)
        locked_joint_target, release_pose, release_ik = transition_manipulation_pose(
            sim,
            scene,
            robot,
            wheel_ids,
            locked_joint_target,
            recorder,
            "ready_front_open",
            target,
            args_cli.drop_steps,
        )
        settle_object(sim, scene, robot, wheel_ids, locked_joint_target, recorder, target.key, args_cli.settle_after_release_steps)
        drop_motion = "soft_carry_cleared_before_open_then_object_settled_by_physics"
    else:
        locked_joint_target, release_pose, release_ik = transition_manipulation_pose(
            sim, scene, robot, wheel_ids, locked_joint_target, recorder, "ready_front_open", target, args_cli.drop_steps
        )
        smooth_object_pose_to(
            sim,
            scene,
            robot,
            wheel_ids,
            locked_joint_target,
            recorder,
            target.key,
            (drop_pos[0], drop_pos[1], max(float(OBJECT_BY_KEY[target.key]["size"][2]) * 0.5, 0.08)),
            0.0,
            max(1, int(args_cli.drop_steps)),
        )
        drop_motion = "legacy_pose_write_to_area"
    area_check = object_area_error(scene, target.key, target.category)
    meta["dual_arm"].append({
        "state": "release_open",
        "steps": int(args_cli.drop_steps),
        "waist_target": {
            "waist_yaw_joint": release_pose.get("waist_yaw_joint"),
            "waist_pitch_joint": release_pose.get("waist_pitch_joint"),
        },
        "ik": release_ik,
        "drop_pose": [float(v) for v in drop_pos],
        "drop_motion": drop_motion,
        "object_pose_before_release": object_pose_before_release if args_cli.carry_mode == "physical_soft_constraint" else None,
        "area_check": area_check,
    })
    if args_cli.carry_mode in {"pure_contact", "physical_soft_constraint"} and not bool(area_check["inside_area_xy"]):
        meta["reason"] = f"Released {target.key}, but final object pose is outside the {target.category} area."
        return False, locked_joint_target, meta
    locked_joint_target = transition_pose(sim, scene, robot, wheel_ids, locked_joint_target, recorder, "home_down", args_cli.pose_steps)
    meta["dual_arm"].append({"state": "home_down", "steps": int(args_cli.pose_steps)})
    completed.add(target.key)
    meta["success"] = True
    meta["reason"] = "Sorted one large floor object."
    meta["completed_key"] = target.key
    meta["target_area"] = AREA_SPECS[target.category]["name"]
    return True, locked_joint_target, meta


def candidate_to_dict(candidate: TargetCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "key": candidate.key,
        "name": candidate.name,
        "category": candidate.category,
        "category_cn": CATEGORY_CN.get(candidate.category, candidate.category),
        "center_world": [float(v) for v in candidate.center_world],
        "confidence": float(candidate.confidence),
        "source": candidate.source,
        "bbox_xyxy": candidate.bbox_xyxy,
        "reason": candidate.reason,
    }


def main() -> None:
    global OFFICIAL_IK_BRIDGE, CUROBO_IK_BRIDGE

    run_dir = Path(args_cli.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    sim = SimulationContext(task320_sim_cfg())
    scene = InteractiveScene(FloorLargeTrashSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0, replicate_physics=False))
    sim.reset()
    reset_scene(scene)
    robot: Articulation = scene["robot"]
    wheel_ids, wheel_names = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
    if len(wheel_ids) != 4:
        raise RuntimeError(f"Expected four wheel joints, found {wheel_names}.")

    if args_cli.ik_backend == "official_standalone":
        OFFICIAL_IK_BRIDGE = KuavoOfficialIKBridge()
        print(f"[INFO] Kuavo official IK bridge: {OFFICIAL_IK_BRIDGE.status}", flush=True)
        CUROBO_IK_BRIDGE = None
    elif args_cli.ik_backend == "curobo":
        OFFICIAL_IK_BRIDGE = None
        CUROBO_IK_BRIDGE = KuavoCuroboDualArmIKBridge()
        print(f"[INFO] Kuavo cuRobo IK bridge: {CUROBO_IK_BRIDGE.status}", flush=True)
    else:
        OFFICIAL_IK_BRIDGE = None
        CUROBO_IK_BRIDGE = None

    recorder = ObserverVideoRecorder(run_dir)
    locked_joint_target = robot.data.joint_pos.clone()
    completed: set[str] = set()
    cycles: list[dict[str, Any]] = []

    for _ in range(max(1, int(args_cli.warmup_steps))):
        step_sim(sim, scene, robot, wheel_ids, locked_joint_target, recorder)

    summary: dict[str, Any] = {
        "mode": "task_320_dual_arm_floor_sort",
        "description": (
            "Independent floor large-trash sorting scene using Kuavo S62 official standalone IK. "
            "Default carry_mode=physical_soft_constraint requires true bilateral contact first, then "
            "uses a soft physical carry servo without writing object pose. Use --carry_mode pure_contact "
            "for the strict contact-only path."
        ),
        "run_dir": str(run_dir),
        "perception_source": args_cli.perception_source,
        "nav_backend": args_cli.nav_backend,
        "carry_mode": args_cli.carry_mode,
        "physical_carry_controller": args_cli.physical_carry_controller,
        "physical_carry_velocity_gain": float(args_cli.physical_carry_velocity_gain),
        "physical_carry_max_linear_speed": float(args_cli.physical_carry_max_linear_speed),
        "physical_carry_max_vertical_speed": float(args_cli.physical_carry_max_vertical_speed),
        "physical_carry_max_accel_mps2": float(args_cli.physical_carry_max_accel_mps2),
        "physical_carry_max_object_speed_mps": float(args_cli.physical_carry_max_object_speed_mps),
        "physics_dt": float(args_cli.physics_dt),
        "physics_safety_workspace_m": float(args_cli.physics_safety_workspace_m),
        "friction": {
            "object_static": float(args_cli.object_static_friction),
            "object_dynamic": float(args_cli.object_dynamic_friction),
            "ground_static": float(args_cli.ground_static_friction),
            "ground_dynamic": float(args_cli.ground_dynamic_friction),
            "sim_static": float(args_cli.sim_static_friction),
            "sim_dynamic": float(args_cli.sim_dynamic_friction),
        },
        "physx": {
            "solve_articulation_contact_last": bool(args_cli.physx_solve_articulation_contact_last),
            "enable_stabilization": bool(args_cli.physx_enable_stabilization),
            "enable_external_forces_every_iteration": bool(args_cli.physx_external_forces_every_iteration),
        },
        "enable_waist": bool(args_cli.enable_waist),
        "unlock_whole_body": bool(args_cli.unlock_whole_body),
        "robot_sim_urdf": str(KUAVO_SIM_URDF),
        "use_task320_contact_pads": bool(args_cli.use_task320_contact_pads),
        "task320_contact_pad_mass": float(args_cli.task320_contact_pad_mass),
        "ik_backend": args_cli.ik_backend,
        "official_ik": OFFICIAL_IK_BRIDGE.status if OFFICIAL_IK_BRIDGE is not None else {"backend": args_cli.ik_backend, "available": False},
        "curobo_ik": CUROBO_IK_BRIDGE.status if CUROBO_IK_BRIDGE is not None else {"backend": args_cli.ik_backend, "available": False},
        "wheel_joints": wheel_names,
        "objects_total": len(OBJECT_SPECS),
        "max_objects": int(args_cli.max_objects),
        "areas": AREA_SPECS,
        "cycles": cycles,
        "success": False,
        "reason": "",
    }

    try:
        for cycle_index in range(max(0, int(args_cli.max_objects))):
            if len(completed) >= len(OBJECT_SPECS):
                summary["success"] = True
                summary["reason"] = "All scene objects sorted."
                break
            progressed, locked_joint_target, cycle_meta = run_one_cycle(
                sim,
                scene,
                robot,
                wheel_ids,
                locked_joint_target,
                recorder,
                run_dir,
                completed,
                cycle_index,
            )
            cycles.append(cycle_meta)
            (run_dir / f"cycle_{cycle_index:04d}" / "cycle.json").write_text(json.dumps(cycle_meta, indent=2, ensure_ascii=False), encoding="utf-8")
            (run_dir / "dual_arm_floor_sort_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            if not progressed:
                summary["success"] = bool(cycle_meta.get("success", False))
                summary["reason"] = str(cycle_meta.get("reason", "Stopped."))
                break
        else:
            summary["success"] = len(completed) >= min(int(args_cli.max_objects), len(OBJECT_SPECS))
            summary["reason"] = f"Reached max_objects={args_cli.max_objects}."
    except Exception as exc:
        summary["success"] = False
        summary["reason"] = repr(exc)
        raise
    finally:
        recorder.finalize()
        summary["completed_scene_keys"] = sorted(completed)
        summary["completed_count"] = len(completed)
        summary["final_robot_pose"] = robot_planar_pose(robot)
        summary["observer_video"] = recorder.metadata()
        (run_dir / "dual_arm_floor_sort_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] Task 320 summary success={summary['success']} completed={len(completed)} -> {run_dir}", flush=True)
        if bool(args_cli.force_exit):
            os._exit(0 if bool(summary["success"]) else 1)
        simulation_app.close()


if __name__ == "__main__":
    main()
