"""Phase 1 IsaacLab scene and ROS 2 bridge for task 319 garbage sorting.

Usage:
    ./IsaacLab/isaaclab.sh -p task_319_garbage_sort/phase1_scene_ros2_bridge.py

The script builds a single IsaacLab scene, exposes Nav2-compatible ROS 2 topics
(`/cmd_vel`, `/odom`, `/scan`, `/tf`), and drives the Kuavo wheel joints from
incoming velocity commands.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


parser = argparse.ArgumentParser(description="Task 319 phase-one IsaacLab scene with ROS 2 bridge.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--ros2", action="store_true", help="Enable the rclpy ROS 2 bridge.")
parser.add_argument("--disable_cameras", action="store_true", help="Skip RGBD camera sensors for headless smoke tests.")
parser.add_argument("--disable_lidar", action="store_true", help="Skip RayCaster LiDAR for headless smoke tests.")
parser.add_argument("--smoke_steps", type=int, default=0, help="Exit after this many simulation steps. 0 runs forever.")
parser.add_argument("--preview_only", action="store_true", help="Load the scene and keep the viewport alive without stepping physics.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
CAMERA_SENSORS_ENABLED = (not args_cli.disable_cameras) and bool(getattr(args_cli, "enable_cameras", False))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from task_319_garbage_sort.gripper_robot_urdf import ensure_kuavo_with_gripper_urdf


ROOT_DIR = WORKSPACE_ROOT
GRIPPER_URDF = ROOT_DIR / "task_319_garbage_sort/two_finger_gripper.urdf"
KUAVO_BASE_URDF = ROOT_DIR / "kuavo-ros-opensource/src/kuavo_assets/models/biped_s62/urdf/biped_s62.urdf"
KUAVO_URDF = ensure_kuavo_with_gripper_urdf(KUAVO_BASE_URDF, GRIPPER_URDF)
YCB_AXIS_ALIGNED_DIR = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned"
YCB_AXIS_ALIGNED_PHYSICS_DIR = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics"

FRAME_ODOM = "odom"
FRAME_BASE = "base_link"
FRAME_LIDAR = "lidar_link"
FRAME_HEAD_CAMERA = "head_camera_depth"
FRAME_WRIST_CAMERA = "right_wrist_camera"

WHEEL_JOINTS = [
    "wheel_left_front_joint",
    "wheel_left_behind_joint",
    "wheel_right_front_joint",
    "wheel_right_behind_joint",
]

STATIC_SCAN_OBSTACLES = [
    (1.8, 0.0, 1.05),
    (3.0, -0.72, 0.34),
    (3.0, -0.24, 0.34),
    (3.0, 0.24, 0.34),
    (3.0, 0.72, 0.34),
]

TABLE_TOP_CENTER_Z = 0.51
TABLE_TOP_THICKNESS = 0.06
TABLE_SURFACE_Z = TABLE_TOP_CENTER_Z + TABLE_TOP_THICKNESS / 2.0
TRASH_VISUAL_CLEARANCE = 0.002
ROBOT_STABILIZED_BASE_Z = 0.0
ROT_X_90 = (0.70710678, 0.70710678, 0.0, 0.0)

YCB_LOCAL_MIN_Y = {
    "003_cracker_box": -0.106719,
    "004_sugar_box": -0.088127,
    "005_tomato_soup_can": -0.050927,
    "006_mustard_bottle": -0.095650,
    "010_potted_meat_can": -0.041772,
    "011_banana": -0.019325,
    "021_bleach_cleanser": -0.125290,
    "025_mug": -0.040652,
    "040_large_marker": -0.010500,
    "061_foam_brick": -0.025596,
}

def uniform_scale(scale: float) -> tuple[float, float, float]:
    return (scale, scale, scale)


def ycb_table_pos_from_min_y(x: float, y: float, name: str, scale: float = 1.0) -> tuple[float, float, float]:
    return (x, y, TABLE_SURFACE_Z + TRASH_VISUAL_CLEARANCE - YCB_LOCAL_MIN_Y[name] * scale)



def high_friction_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=1.45,
        dynamic_friction=1.10,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )


def contact_material(static_friction: float, dynamic_friction: float, restitution: float = 0.01) -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )


WOOD_MATERIAL = contact_material(0.72, 0.55, 0.02)
BIN_PLASTIC_MATERIAL = contact_material(0.85, 0.62, 0.02)
PAPER_MATERIAL = contact_material(1.05, 0.85, 0.01)
PLASTIC_MATERIAL = contact_material(0.62, 0.45, 0.035)
METAL_MATERIAL = contact_material(0.42, 0.30, 0.05)
FOOD_SKIN_MATERIAL = contact_material(0.95, 0.72, 0.02)


def colored_surface(color: tuple[float, float, float]) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.78)


def semantic_tags(semantic_class: str) -> list[tuple[str, str]]:
    return [("class", semantic_class)] if CAMERA_SENSORS_ENABLED else []


def rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        linear_damping=0.08,
        angular_damping=0.12,
        max_depenetration_velocity=1.5,
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


def cuboid_rigid(
    name: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
    semantic_class: str,
    *,
    visible: bool = True,
    static: bool = False,
    material: sim_utils.RigidBodyMaterialCfg | None = None,
    contact_offset: float = 0.003,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            visible=visible,
            rigid_props=static_rigid_props() if static else rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=collision_props(contact_offset),
            physics_material=material or high_friction_material(),
            visual_material=colored_surface(color),
            semantic_tags=semantic_tags(semantic_class),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def cylinder_rigid(
    name: str,
    radius: float,
    height: float,
    pos: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
    semantic_class: str,
    *,
    visible: bool = True,
    material: sim_utils.RigidBodyMaterialCfg | None = None,
    contact_offset: float = 0.003,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            visible=visible,
            rigid_props=rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=collision_props(contact_offset),
            physics_material=material or high_friction_material(),
            visual_material=colored_surface(color),
            semantic_tags=semantic_tags(semantic_class),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def sphere_rigid(
    name: str,
    radius: float,
    pos: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
    semantic_class: str,
    *,
    visible: bool = True,
    material: sim_utils.RigidBodyMaterialCfg | None = None,
    contact_offset: float = 0.003,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.SphereCfg(
            radius=radius,
            visible=visible,
            rigid_props=rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=collision_props(contact_offset),
            physics_material=material or high_friction_material(),
            visual_material=colored_surface(color),
            semantic_tags=semantic_tags(semantic_class),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def capsule_rigid(
    name: str,
    radius: float,
    height: float,
    pos: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
    semantic_class: str,
    *,
    visible: bool = True,
    material: sim_utils.RigidBodyMaterialCfg | None = None,
    contact_offset: float = 0.003,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CapsuleCfg(
            radius=radius,
            height=height,
            visible=visible,
            rigid_props=rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=collision_props(contact_offset),
            physics_material=material or high_friction_material(),
            visual_material=colored_surface(color),
            semantic_tags=semantic_tags(semantic_class),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def visual_usd(
    name: str,
    usd_path: Path | str,
    pos: tuple[float, float, float],
    scale: tuple[float, float, float],
    semantic_class: str,
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            scale=scale,
            semantic_tags=semantic_tags(semantic_class),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def usd_rigid(
    name: str,
    usd_path: Path | str,
    pos: tuple[float, float, float],
    scale: tuple[float, float, float],
    mass: float,
    material: sim_utils.RigidBodyMaterialCfg,
    semantic_class: str = "trash",
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    contact_offset: float = 0.002,
    apply_runtime_physics: bool = True,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            scale=scale,
            rigid_props=rigid_props() if apply_runtime_physics else None,
            mass_props=sim_utils.MassPropertiesCfg(mass=mass) if apply_runtime_physics else None,
            collision_props=collision_props(contact_offset) if apply_runtime_physics else None,
            semantic_tags=semantic_tags(semantic_class),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


TRASH_USD_OBJECT_NAMES = [
    "trash_cracker_box_0",
    "trash_sugar_box_0",
    "trash_tomato_soup_can_0",
    "trash_mustard_bottle_0",
    "trash_banana_0",
    "trash_potted_meat_can_0",
    "trash_bleach_cleanser_0",
    "trash_large_marker_0",
    "trash_foam_brick_0",
    "trash_mug_0",
]
TRASH_SCENE_KEYS = [f"trash_{idx:02d}" for idx in range(len(TRASH_USD_OBJECT_NAMES))]
TRASH_CATEGORY_NAMES = [
    "recyclable",
    "recyclable",
    "recyclable",
    "recyclable",
    "kitchen",
    "kitchen",
    "hazard",
    "hazard",
    "other",
    "other",
]
YCB_VISUAL_RUNTIME_MASSES = {
    "trash_banana_0": 0.08,
    "trash_potted_meat_can_0": 0.09,
    "trash_bleach_cleanser_0": 0.18,
    "trash_large_marker_0": 0.02,
    "trash_foam_brick_0": 0.02,
    "trash_mug_0": 0.12,
}
REQUIRED_TRASH_CATEGORIES = {"recyclable", "kitchen", "hazard", "other"}
MIN_TRASH_OBJECT_COUNT = 10
def validate_trash_inventory() -> None:
    object_count = len(TRASH_USD_OBJECT_NAMES)
    categories = set(TRASH_CATEGORY_NAMES)
    missing_categories = REQUIRED_TRASH_CATEGORIES - categories
    if object_count < MIN_TRASH_OBJECT_COUNT or missing_categories:
        raise RuntimeError(
            f"Task 319 requires at least {MIN_TRASH_OBJECT_COUNT} trash objects covering "
            f"{sorted(REQUIRED_TRASH_CATEGORIES)}; got {object_count} objects and missing {sorted(missing_categories)}."
        )


validate_trash_inventory()


KUAVO_62_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Kuavo62",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(KUAVO_URDF),
        fix_base=False,
        merge_fixed_joints=False,
        link_density=1000.0,
        collision_from_visuals=True,
        collider_type="convex_hull",
        self_collision=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            linear_damping=0.04,
            angular_damping=0.08,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=120.0, damping=12.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, -2.0, 0.0),
        joint_pos={
            "knee_joint": 0.0,
            "leg_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "zhead_1_joint": 0.0,
            "zhead_2_joint": 0.0,
            "zarm_l2_joint": 0.30,
            "zarm_l4_joint": -0.6,
            "zarm_r2_joint": 0.30,
            "zarm_r4_joint": -0.6,
            "left_finger_joint": 0.030,
            "right_finger_joint": 0.030,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*_joint"], effort_limit_sim=35.0, velocity_limit_sim=25.0, stiffness=0.0, damping=18.0
        ),
        "stance": ImplicitActuatorCfg(
            joint_names_expr=["knee_joint", "leg_joint"],
            effort_limit_sim=650.0,
            velocity_limit_sim=12.0,
            stiffness=1400.0,
            damping=140.0,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"], effort_limit_sim=280.0, velocity_limit_sim=8.0, stiffness=800.0, damping=90.0
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["zarm_l[1-7]_joint"], effort_limit_sim=120.0, velocity_limit_sim=8.0, stiffness=700.0, damping=60.0
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["zarm_r[1-7]_joint"], effort_limit_sim=120.0, velocity_limit_sim=8.0, stiffness=900.0, damping=70.0
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["zhead_[12]_joint"], effort_limit_sim=40.0, velocity_limit_sim=5.0, stiffness=150.0, damping=20.0
        ),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*finger_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=0.35,
            stiffness=1200.0,
            damping=90.0,
        ),
    },
)


@configclass
class TrashSortingSceneCfg(InteractiveSceneCfg):
    """Single-env task scene with Kuavo, trash objects, sensors, and Nav2 lidar."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(12.0, 12.0),
            physics_material=high_friction_material(),
            color=(0.35, 0.36, 0.34),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.82, 0.86, 0.9))
    )

    robot: ArticulationCfg = KUAVO_62_CFG
    table_top = cuboid_rigid(
        "sorting_table_top",
        (1.80, 0.90, TABLE_TOP_THICKNESS),
        (1.80, 0.0, TABLE_TOP_CENTER_Z),
        60.0,
        (0.48, 0.36, 0.24),
        "table",
        static=True,
        material=WOOD_MATERIAL,
    )
    table_leg_fl = cuboid_rigid("sorting_table_leg_fl", (0.07, 0.07, 0.48), (0.96, -0.38, 0.24), 8.0, (0.32, 0.25, 0.18), "table", static=True, material=WOOD_MATERIAL)
    table_leg_fr = cuboid_rigid("sorting_table_leg_fr", (0.07, 0.07, 0.48), (2.64, -0.38, 0.24), 8.0, (0.32, 0.25, 0.18), "table", static=True, material=WOOD_MATERIAL)
    table_leg_bl = cuboid_rigid("sorting_table_leg_bl", (0.07, 0.07, 0.48), (0.96, 0.38, 0.24), 8.0, (0.32, 0.25, 0.18), "table", static=True, material=WOOD_MATERIAL)
    table_leg_br = cuboid_rigid("sorting_table_leg_br", (0.07, 0.07, 0.48), (2.64, 0.38, 0.24), 8.0, (0.32, 0.25, 0.18), "table", static=True, material=WOOD_MATERIAL)

    bin_recycle_bottom = cuboid_rigid("bin_recycle_bottom", (0.50, 0.50, 0.04), (3.0, -0.72, 0.02), 16.0, (0.04, 0.20, 0.72), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_recycle_left = cuboid_rigid("bin_recycle_left", (0.04, 0.50, 0.62), (2.77, -0.72, 0.33), 16.0, (0.04, 0.20, 0.72), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_recycle_right = cuboid_rigid("bin_recycle_right", (0.04, 0.50, 0.62), (3.23, -0.72, 0.33), 16.0, (0.04, 0.20, 0.72), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_recycle_front = cuboid_rigid("bin_recycle_front", (0.50, 0.04, 0.62), (3.0, -0.95, 0.33), 16.0, (0.04, 0.20, 0.72), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_recycle_back = cuboid_rigid("bin_recycle_back", (0.50, 0.04, 0.62), (3.0, -0.49, 0.33), 16.0, (0.04, 0.20, 0.72), "bin", static=True, material=BIN_PLASTIC_MATERIAL)

    bin_kitchen_bottom = cuboid_rigid("bin_kitchen_bottom", (0.50, 0.50, 0.04), (3.0, -0.24, 0.02), 16.0, (0.04, 0.48, 0.16), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_kitchen_left = cuboid_rigid("bin_kitchen_left", (0.04, 0.50, 0.62), (2.77, -0.24, 0.33), 16.0, (0.04, 0.48, 0.16), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_kitchen_right = cuboid_rigid("bin_kitchen_right", (0.04, 0.50, 0.62), (3.23, -0.24, 0.33), 16.0, (0.04, 0.48, 0.16), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_kitchen_front = cuboid_rigid("bin_kitchen_front", (0.50, 0.04, 0.62), (3.0, -0.47, 0.33), 16.0, (0.04, 0.48, 0.16), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_kitchen_back = cuboid_rigid("bin_kitchen_back", (0.50, 0.04, 0.62), (3.0, -0.01, 0.33), 16.0, (0.04, 0.48, 0.16), "bin", static=True, material=BIN_PLASTIC_MATERIAL)

    bin_hazard_bottom = cuboid_rigid("bin_hazard_bottom", (0.50, 0.50, 0.04), (3.0, 0.24, 0.02), 16.0, (0.72, 0.07, 0.06), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_hazard_left = cuboid_rigid("bin_hazard_left", (0.04, 0.50, 0.62), (2.77, 0.24, 0.33), 16.0, (0.72, 0.07, 0.06), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_hazard_right = cuboid_rigid("bin_hazard_right", (0.04, 0.50, 0.62), (3.23, 0.24, 0.33), 16.0, (0.72, 0.07, 0.06), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_hazard_front = cuboid_rigid("bin_hazard_front", (0.50, 0.04, 0.62), (3.0, 0.01, 0.33), 16.0, (0.72, 0.07, 0.06), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_hazard_back = cuboid_rigid("bin_hazard_back", (0.50, 0.04, 0.62), (3.0, 0.47, 0.33), 16.0, (0.72, 0.07, 0.06), "bin", static=True, material=BIN_PLASTIC_MATERIAL)

    bin_other_bottom = cuboid_rigid("bin_other_bottom", (0.50, 0.50, 0.04), (3.0, 0.72, 0.02), 16.0, (0.28, 0.28, 0.28), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_other_left = cuboid_rigid("bin_other_left", (0.04, 0.50, 0.62), (2.77, 0.72, 0.33), 16.0, (0.28, 0.28, 0.28), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_other_right = cuboid_rigid("bin_other_right", (0.04, 0.50, 0.62), (3.23, 0.72, 0.33), 16.0, (0.28, 0.28, 0.28), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_other_front = cuboid_rigid("bin_other_front", (0.50, 0.04, 0.62), (3.0, 0.49, 0.33), 16.0, (0.28, 0.28, 0.28), "bin", static=True, material=BIN_PLASTIC_MATERIAL)
    bin_other_back = cuboid_rigid("bin_other_back", (0.50, 0.04, 0.62), (3.0, 0.95, 0.33), 16.0, (0.28, 0.28, 0.28), "bin", static=True, material=BIN_PLASTIC_MATERIAL)

    trash_00 = usd_rigid(
        "trash_cracker_box_0", f"{YCB_AXIS_ALIGNED_PHYSICS_DIR}/003_cracker_box.usd",
        ycb_table_pos_from_min_y(1.30, -0.31, "003_cracker_box"),
        uniform_scale(1.0), 0.08, PAPER_MATERIAL, semantic_class="recyclable_paper", rot=ROT_X_90,
    )
    trash_01 = usd_rigid(
        "trash_sugar_box_0", f"{YCB_AXIS_ALIGNED_PHYSICS_DIR}/004_sugar_box.usd",
        ycb_table_pos_from_min_y(1.49, -0.19, "004_sugar_box"),
        uniform_scale(1.0), 0.10, PAPER_MATERIAL, semantic_class="recyclable_paper", rot=ROT_X_90,
    )
    trash_02 = usd_rigid(
        "trash_tomato_soup_can_0", f"{YCB_AXIS_ALIGNED_PHYSICS_DIR}/005_tomato_soup_can.usd",
        ycb_table_pos_from_min_y(1.68, -0.33, "005_tomato_soup_can"),
        uniform_scale(1.0), 0.08, METAL_MATERIAL, semantic_class="recyclable_metal", rot=ROT_X_90,
    )
    trash_03 = usd_rigid(
        "trash_mustard_bottle_0", f"{YCB_AXIS_ALIGNED_PHYSICS_DIR}/006_mustard_bottle.usd",
        ycb_table_pos_from_min_y(1.86, -0.20, "006_mustard_bottle"),
        uniform_scale(1.0), 0.08, PLASTIC_MATERIAL, semantic_class="recyclable_plastic", rot=ROT_X_90,
    )
    trash_04 = usd_rigid(
        "trash_banana_0", f"{YCB_AXIS_ALIGNED_DIR}/011_banana.usd",
        ycb_table_pos_from_min_y(2.04, -0.32, "011_banana"),
        uniform_scale(1.0), 0.08, FOOD_SKIN_MATERIAL, semantic_class="kitchen_food", rot=ROT_X_90, apply_runtime_physics=False,
    )
    trash_05 = usd_rigid(
        "trash_potted_meat_can_0", f"{YCB_AXIS_ALIGNED_DIR}/010_potted_meat_can.usd",
        ycb_table_pos_from_min_y(1.04, -0.16, "010_potted_meat_can"),
        uniform_scale(1.0), 0.09, METAL_MATERIAL, semantic_class="kitchen_food_residue", rot=ROT_X_90, apply_runtime_physics=False,
    )
    trash_06 = usd_rigid(
        "trash_bleach_cleanser_0", f"{YCB_AXIS_ALIGNED_DIR}/021_bleach_cleanser.usd",
        ycb_table_pos_from_min_y(1.52, 0.25, "021_bleach_cleanser", scale=0.85),
        uniform_scale(0.85), 0.18, PLASTIC_MATERIAL, semantic_class="hazard_chemical", rot=ROT_X_90, apply_runtime_physics=False,
    )
    trash_07 = usd_rigid(
        "trash_large_marker_0", f"{YCB_AXIS_ALIGNED_DIR}/040_large_marker.usd",
        ycb_table_pos_from_min_y(1.06, 0.06, "040_large_marker"),
        uniform_scale(1.0), 0.02, PLASTIC_MATERIAL, semantic_class="hazard_marker", rot=ROT_X_90, apply_runtime_physics=False,
    )
    trash_08 = usd_rigid(
        "trash_foam_brick_0", f"{YCB_AXIS_ALIGNED_DIR}/061_foam_brick.usd",
        ycb_table_pos_from_min_y(1.10, 0.31, "061_foam_brick"),
        uniform_scale(1.0), 0.02, PLASTIC_MATERIAL, semantic_class="other_waste", rot=ROT_X_90, apply_runtime_physics=False,
    )
    trash_09 = usd_rigid(
        "trash_mug_0", f"{YCB_AXIS_ALIGNED_DIR}/025_mug.usd",
        ycb_table_pos_from_min_y(1.76, 0.11, "025_mug"),
        uniform_scale(1.0), 0.12, METAL_MATERIAL, semantic_class="other_waste", rot=ROT_X_90, apply_runtime_physics=False,
    )

    head_rgbd = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo62/head_camera_depth/head_rgbd",
        update_period=1.0 / 30.0,
        width=640,
        height=480,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, focus_distance=1.2, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.405309, -0.579417, 0.579417, -0.405309), convention="ros"),
        update_latest_camera_pose=True,
    ) if CAMERA_SENSORS_ENABLED else None
    wrist_rgbd = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo62/zarm_r7_end_effector/right_wrist_rgbd",
        update_period=1.0 / 30.0,
        width=320,
        height=240,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, focus_distance=0.6, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=(0.06, 0.0, 0.02), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    ) if CAMERA_SENSORS_ENABLED else None
    lidar = None if args_cli.disable_lidar else RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo62/base_link",
        update_period=1.0 / 10.0,
        mesh_prim_paths=["/World/defaultGroundPlane"],
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.32), rot=(1.0, 0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        max_distance=8.0,
        debug_vis=False,
    )


def yaw_from_quat_wxyz(q: Any) -> float:
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def roll_pitch_from_quat_wxyz(q: Any) -> tuple[float, float]:
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_arg)
    return roll, pitch


def quat_wxyz_from_yaw(yaw: float, device: torch.device | str) -> torch.Tensor:
    half_yaw = 0.5 * yaw
    return torch.tensor([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], device=device)


@dataclass
class CmdVel:
    linear_x: float = 0.0
    angular_z: float = 0.0
    stamp: float = 0.0


class Ros2NavBridge:
    """Minimal rclpy bridge for Nav2-compatible base topics."""

    def __init__(self, robot: Articulation, lidar: Any, wheel_radius: float = 0.09, track_width: float = 0.52):
        self.robot = robot
        self.lidar = lidar
        self.wheel_radius = wheel_radius
        self.track_width = track_width
        self.cmd = CmdVel(stamp=time.time())
        self.enabled = False
        self._wheel_ids, self._wheel_names = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
        if len(self._wheel_ids) != 4:
            print(f"[WARN] Expected 4 wheel joints, found {self._wheel_names}; /cmd_vel will be ignored.")

        try:
            import rclpy
            from geometry_msgs.msg import TransformStamped, Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import LaserScan
            from tf2_ros import TransformBroadcaster

            self.rclpy = rclpy
            self.TransformStamped = TransformStamped
            self.Twist = Twist
            self.Odometry = Odometry
            self.LaserScan = LaserScan
            self.rclpy.init(args=None)
            self.node = self.rclpy.create_node("isaaclab_task319_bridge")
            self.odom_pub = self.node.create_publisher(Odometry, "/odom", 10)
            self.scan_pub = self.node.create_publisher(LaserScan, "/scan", 10)
            self.tf_pub = TransformBroadcaster(self.node)
            self.node.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
            self.enabled = True
        except Exception as exc:
            print(f"[WARN] ROS 2 bridge disabled: {exc}")

    def _on_cmd_vel(self, msg: Any) -> None:
        self.cmd = CmdVel(float(msg.linear.x), float(msg.angular.z), time.time())

    def apply_cmd_vel(self) -> None:
        if time.time() - self.cmd.stamp > 0.5:
            vx = wz = 0.0
        else:
            vx, wz = self.cmd.linear_x, self.cmd.angular_z

        if len(self._wheel_ids) != 4:
            return
        left = (vx - 0.5 * self.track_width * wz) / self.wheel_radius
        right = (vx + 0.5 * self.track_width * wz) / self.wheel_radius
        target = torch.zeros((1, len(self._wheel_ids)), device=self.robot.device)
        target[0, 0] = left
        target[0, 1] = left
        target[0, 2] = right
        target[0, 3] = right
        self.robot.set_joint_velocity_target(target, joint_ids=self._wheel_ids)

    def spin_once(self) -> None:
        if self.enabled:
            self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish(self) -> None:
        if not self.enabled:
            return
        stamp = self.node.get_clock().now().to_msg()
        root = self.robot.data.root_state_w[0].detach().cpu()
        pos = root[0:3]
        quat = root[3:7]
        lin_vel = root[7:10]
        ang_vel = root[10:13]
        self._publish_odom(stamp, pos, quat, lin_vel, ang_vel)
        self._publish_tf(stamp, pos, quat)
        self._publish_scan(stamp)

    def _publish_odom(self, stamp: Any, pos: Any, quat: Any, lin_vel: Any, ang_vel: Any) -> None:
        msg = self.Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = FRAME_ODOM
        msg.child_frame_id = FRAME_BASE
        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation.w = float(quat[0])
        msg.pose.pose.orientation.x = float(quat[1])
        msg.pose.pose.orientation.y = float(quat[2])
        msg.pose.pose.orientation.z = float(quat[3])
        msg.twist.twist.linear.x = float(lin_vel[0])
        msg.twist.twist.linear.y = float(lin_vel[1])
        msg.twist.twist.angular.z = float(ang_vel[2])
        self.odom_pub.publish(msg)

    def _publish_tf(self, stamp: Any, pos: Any, quat: Any) -> None:
        transforms = []
        transforms.append(self._transform(stamp, FRAME_ODOM, FRAME_BASE, pos, quat))
        transforms.append(self._transform(stamp, FRAME_BASE, FRAME_LIDAR, (0.0, 0.0, 0.32), (1.0, 0.0, 0.0, 0.0)))
        transforms.append(self._transform(stamp, FRAME_BASE, FRAME_HEAD_CAMERA, (0.18, 0.0, 1.35), (1.0, 0.0, 0.0, 0.0)))
        transforms.append(self._transform(stamp, FRAME_BASE, FRAME_WRIST_CAMERA, (0.45, -0.2, 1.05), (1.0, 0.0, 0.0, 0.0)))
        self.tf_pub.sendTransform(transforms)

    def _transform(self, stamp: Any, parent: str, child: str, pos: Any, quat: Any) -> Any:
        tf = self.TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.translation.x = float(pos[0])
        tf.transform.translation.y = float(pos[1])
        tf.transform.translation.z = float(pos[2])
        tf.transform.rotation.w = float(quat[0])
        tf.transform.rotation.x = float(quat[1])
        tf.transform.rotation.y = float(quat[2])
        tf.transform.rotation.z = float(quat[3])
        return tf

    def _publish_scan(self, stamp: Any) -> None:
        msg = self.LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = FRAME_LIDAR
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = math.radians(1.0)
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = 0.05
        msg.range_max = 8.0
        msg.ranges = self._extract_lidar_ranges()
        self.scan_pub.publish(msg)

    def _extract_lidar_ranges(self) -> list[float]:
        count = 361
        try:
            if self.lidar is None:
                raise RuntimeError("LiDAR disabled")
            ray_hits = self.lidar.data.ray_hits_w[0].detach().cpu()
            ray_starts = self.lidar.data.pos_w[0].detach().cpu()
            ranges = torch.linalg.norm(ray_hits - ray_starts, dim=-1).tolist()
            ranges = [float(r) if math.isfinite(float(r)) else float("inf") for r in ranges]
            if any(math.isfinite(r) and 0.05 < r < 8.0 for r in ranges):
                return ranges[:count] if len(ranges) >= count else ranges + [float("inf")] * (count - len(ranges))
        except Exception:
            pass
        return self._synthetic_static_scan(count)

    def _synthetic_static_scan(self, count: int) -> list[float]:
        root = self.robot.data.root_state_w[0].detach().cpu()
        base_x, base_y = float(root[0]), float(root[1])
        base_yaw = yaw_from_quat_wxyz(root[3:7])
        ranges = [float("inf")] * count
        for obs_x, obs_y, radius in STATIC_SCAN_OBSTACLES:
            dx = obs_x - base_x
            dy = obs_y - base_y
            distance = math.hypot(dx, dy) - radius
            if distance <= 0.05 or distance >= 8.0:
                continue
            bearing = math.atan2(math.sin(math.atan2(dy, dx) - base_yaw), math.cos(math.atan2(dy, dx) - base_yaw))
            beam = int(round(math.degrees(bearing))) + 180
            spread = max(1, int(math.degrees(math.atan2(radius, max(distance, 0.05)))))
            for i in range(max(0, beam - spread), min(count, beam + spread + 1)):
                ranges[i] = min(ranges[i], distance)
        return ranges

    def shutdown(self) -> None:
        if self.enabled:
            self.node.destroy_node()
            self.rclpy.shutdown()


def apply_ycb_visual_runtime_physics() -> None:
    from pxr import Usd, UsdPhysics

    stage = sim_utils.get_current_stage()
    convex_cfg = sim_utils.ConvexHullPropertiesCfg(hull_vertex_limit=64)
    collider_cfg = collision_props(contact_offset=0.002)
    for env_id in range(args_cli.num_envs):
        for name, mass in YCB_VISUAL_RUNTIME_MASSES.items():
            root_path = f"/World/envs/env_{env_id}/{name}"
            root_prim = stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                print(f"[WARN] Runtime physics skipped missing trash prim: {root_path}", flush=True)
                continue
            try:
                sim_utils.define_rigid_body_properties(root_path, rigid_props(), stage=stage)
                sim_utils.define_mass_properties(root_path, sim_utils.MassPropertiesCfg(mass=mass), stage=stage)
            except Exception as exc:
                print(f"[WARN] Could not define rigid-body physics for {root_path}: {exc}", flush=True)
                continue

            mesh_paths = [str(prim.GetPath()) for prim in Usd.PrimRange(root_prim) if prim.GetTypeName() == "Mesh"]
            if not mesh_paths:
                print(f"[WARN] No Mesh prims found under {root_path}; object will not collide correctly.", flush=True)
                continue
            for mesh_path in mesh_paths:
                try:
                    sim_utils.define_collision_properties(mesh_path, collider_cfg, stage=stage)
                    sim_utils.modify_mesh_collision_properties(mesh_path, convex_cfg, stage=stage)
                except Exception as exc:
                    print(f"[WARN] Could not apply YCB convexHull collision to {mesh_path}: {exc}", flush=True)


def apply_gripper_contact_material() -> None:
    from pxr import Usd, UsdPhysics

    stage = sim_utils.get_current_stage()
    material_path = "/World/physicsScene/task319_gripper_rubber"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.9,
        dynamic_friction=1.45,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    material_cfg.func(material_path, material_cfg)
    matched = 0
    bound = 0
    name_tokens = ("gripper_base", "left_finger", "right_finger")
    for env_id in range(args_cli.num_envs):
        root_path = f"/World/envs/env_{env_id}/Kuavo62"
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            continue
        for prim in Usd.PrimRange(root_prim):
            prim_path = str(prim.GetPath())
            if not any(token in prim_path for token in name_tokens):
                continue
            if prim.IsInstanceProxy() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            matched += 1
            try:
                if sim_utils.bind_physics_material(prim_path, material_path, stage=stage):
                    bound += 1
            except Exception as exc:
                print(f"[WARN] Could not bind gripper rubber material to {prim_path}: {exc}", flush=True)
    if matched == 0:
        print(
            "[INFO] Gripper collision prims are instanced; using SimulationCfg default high-friction material.",
            flush=True,
        )
    else:
        print(f"[INFO] Gripper contact material bound to {bound}/{matched} collision prims.", flush=True)


def reset_scene(scene: InteractiveScene) -> None:
    robot = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    scene.reset()



def stabilize_robot_base(robot: Articulation) -> None:
    # Kuavo's real platform relies on a low-level balance controller; keep that attitude loop closed in phase one.
    root_state = robot.data.root_state_w.clone()
    for env_id in range(root_state.shape[0]):
        yaw = yaw_from_quat_wxyz(root_state[env_id, 3:7])
        root_state[env_id, 2] = ROBOT_STABILIZED_BASE_Z
        root_state[env_id, 3:7] = quat_wxyz_from_yaw(yaw, robot.device)
        root_state[env_id, 9] = 0.0
        root_state[env_id, 10] = 0.0
        root_state[env_id, 11] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    lidar = scene["lidar"] if not args_cli.disable_lidar else None
    bridge = Ros2NavBridge(robot, lidar) if args_cli.ros2 else None
    sim_dt = sim.get_physics_dt()
    reset_scene(scene)
    count = 0
    try:
        while simulation_app.is_running():
            if bridge is not None:
                bridge.spin_once()
                bridge.apply_cmd_vel()
            stabilize_robot_base(robot)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            if bridge is not None:
                bridge.publish()

            count += 1
            if args_cli.smoke_steps > 0 and count >= args_cli.smoke_steps:
                robot_root = robot.data.root_state_w[0].detach().cpu()
                roll, pitch = roll_pitch_from_quat_wxyz(robot_root[3:7])
                trash_z_values = []
                trash_diagnostics = []
                for object_name, key in zip(TRASH_USD_OBJECT_NAMES, TRASH_SCENE_KEYS, strict=True):
                    if key in scene.keys():
                        obj = scene[key]
                        current_z = float(obj.data.root_state_w[0, 2].detach().cpu())
                        initial_z = float(obj.data.default_root_state[0, 2].detach().cpu())
                        trash_z_values.append(current_z)
                        trash_diagnostics.append((object_name, current_z, current_z - initial_z))
                print(f"[INFO] Smoke completed {count} physics steps.", flush=True)
                print(
                    f"[INFO] Robot base z={float(robot_root[2]):.3f} m, "
                    f"roll={math.degrees(roll):.2f} deg, pitch={math.degrees(pitch):.2f} deg.",
                    flush=True,
                )
                if trash_z_values:
                    print(
                        f"[INFO] Trash object z range after smoke: "
                        f"{min(trash_z_values):.3f}..{max(trash_z_values):.3f} m.",
                        flush=True,
                    )
                    for object_name, current_z, delta_z in trash_diagnostics:
                        print(f"[INFO]   {object_name}: z={current_z:.3f} m, dz={delta_z:+.3f} m", flush=True)
                break
    finally:
        if bridge is not None:
            bridge.shutdown()


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 120.0, physics_material=high_friction_material())
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([4.2, -4.0, 3.0], [1.4, -0.1, 0.8])
    scene_cfg = TrashSortingSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)
    apply_ycb_visual_runtime_physics()
    apply_gripper_contact_material()
    sim.reset()
    print("[INFO] Task 319 phase-one scene ready.", flush=True)
    if args_cli.preview_only:
        if args_cli.smoke_steps > 0:
            return
        while simulation_app.is_running():
            simulation_app.update()
        return
    if args_cli.ros2:
        print("[INFO] ROS 2 topics: /cmd_vel, /odom, /scan, /tf", flush=True)
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
