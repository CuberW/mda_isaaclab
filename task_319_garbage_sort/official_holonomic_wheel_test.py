"""Minimal official Isaac Sim holonomic wheel-controller test for Kuavo S62.

This intentionally bypasses Nav2, perception, grasping, and the hand-written
Task319 wheel inverse kinematics. It uses NVIDIA's
``isaacsim.robot.wheeled_robots.controllers.HolonomicController`` to convert a
body velocity command into four wheel joint velocity targets, then applies only
those targets to the Kuavo wheel joints in IsaacLab.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
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


parser = argparse.ArgumentParser(description="Official Isaac Sim wheeled-controller sanity test for Task319 Kuavo S62.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=40)
parser.add_argument("--drive_steps", type=int, default=240)
parser.add_argument("--cmd_vx", type=float, default=0.18, help="Body-frame x velocity command in m/s.")
parser.add_argument("--cmd_vy", type=float, default=0.0, help="Body-frame y velocity command in m/s.")
parser.add_argument("--cmd_wz", type=float, default=0.0, help="Body yaw velocity command in rad/s.")
parser.add_argument("--controller_backend", choices=("differential", "holonomic"), default="differential")
parser.add_argument("--mecanum_angle_deg", type=float, default=45.0)
parser.add_argument("--joint_velocity_sign", type=float, default=1.0, help="Optional global sign for controller output before applying wheel joint targets.")
parser.add_argument("--max_linear_speed", type=float, default=0.45)
parser.add_argument("--max_angular_speed", type=float, default=0.75)
parser.add_argument("--max_wheel_speed", type=float, default=25.0)
parser.add_argument("--min_translation_m", type=float, default=0.03)
parser.add_argument("--output_dir", default="task_319_garbage_sort/output/official_holonomic_wheel_tests")
parser.add_argument("--record_debug", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.robot.wheeled_robots")
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from task_319_garbage_sort.gripper_robot_urdf import ensure_kuavo_with_gripper_urdf


ROOT_DIR = WORKSPACE_ROOT
GRIPPER_URDF = ROOT_DIR / "task_319_garbage_sort/two_finger_gripper.urdf"
KUAVO_BASE_URDF = ROOT_DIR / "kuavo-ros-opensource/src/kuavo_assets/models/biped_s62/urdf/biped_s62.urdf"
KUAVO_URDF = ensure_kuavo_with_gripper_urdf(KUAVO_BASE_URDF, GRIPPER_URDF)

WHEEL_RADIUS_M = 0.13035
WHEEL_JOINTS = [
    "wheel_left_front_joint",
    "wheel_left_behind_joint",
    "wheel_right_front_joint",
    "wheel_right_behind_joint",
]
WHEEL_POSITIONS = np.array(
    [
        [0.23248871, 0.23248871, WHEEL_RADIUS_M],
        [-0.23248871, 0.23248871, WHEEL_RADIUS_M],
        [0.23248871, -0.23248871, WHEEL_RADIUS_M],
        [-0.23248871, -0.23248871, WHEEL_RADIUS_M],
    ],
    dtype=np.float64,
)
WHEEL_YAWS_RAD = [0.785398163, 2.356194372, -0.785398163, -2.356194372]


def xyzw_from_yaw(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)]


def quat_wxyz_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def yaw_from_quat_wxyz(q: Any) -> float:
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def robot_pose_xy_yaw(robot: Articulation) -> list[float]:
    root = robot.data.root_state_w[0].detach().cpu()
    return [float(root[0]), float(root[1]), yaw_from_quat_wxyz(root[3:7])]


def wheel_actuator_cfg() -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=["wheel_.*_joint"],
        effort_limit_sim=80.0,
        velocity_limit_sim=max(float(args_cli.max_wheel_speed), 1.0),
        stiffness=0.0,
        damping=18.0,
    )


def locked_actuator_cfg(expr: str, effort: float = 300.0) -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=[expr],
        effort_limit_sim=effort,
        velocity_limit_sim=0.1,
        stiffness=3000.0,
        damping=300.0,
    )


KUAVO_WHEEL_TEST_CFG = ArticulationCfg(
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
        pos=(0.0, 0.0, 0.0),
        rot=quat_wxyz_from_yaw(0.0),
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
        "wheels": wheel_actuator_cfg(),
        "stance": locked_actuator_cfg("knee_joint|leg_joint", 900.0),
        "torso": locked_actuator_cfg("waist_.*_joint", 450.0),
        "left_arm": locked_actuator_cfg("zarm_l[1-7]_joint", 180.0),
        "right_arm": locked_actuator_cfg("zarm_r[1-7]_joint", 180.0),
        "head": locked_actuator_cfg("zhead_[12]_joint", 60.0),
        "gripper": locked_actuator_cfg(".*finger_joint", 60.0),
    },
)


@configclass
class OfficialHolonomicWheelSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(8.0, 8.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.45,
                dynamic_friction=1.10,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
            color=(0.32, 0.33, 0.32),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2200.0, color=(0.82, 0.86, 0.9)),
    )
    robot: ArticulationCfg = KUAVO_WHEEL_TEST_CFG


def reset_robot(scene: InteractiveScene) -> Articulation:
    robot: Articulation = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    root_state[:, 2] = 0.0
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    scene.reset()
    return robot


def make_holonomic_controller() -> HolonomicController:
    return HolonomicController(
        name="task319_official_holonomic",
        wheel_radius=np.array([WHEEL_RADIUS_M] * len(WHEEL_JOINTS), dtype=np.float64),
        wheel_positions=WHEEL_POSITIONS,
        wheel_orientations=np.array([xyzw_from_yaw(yaw) for yaw in WHEEL_YAWS_RAD], dtype=np.float64),
        mecanum_angles=np.array([float(args_cli.mecanum_angle_deg)] * len(WHEEL_JOINTS), dtype=np.float64),
        max_linear_speed=max(float(args_cli.max_linear_speed), 1.0e-6),
        max_angular_speed=max(float(args_cli.max_angular_speed), 1.0e-6),
        max_wheel_speed=max(float(args_cli.max_wheel_speed), 1.0e-6),
    )


def make_official_wheel_targets(command: np.ndarray) -> tuple[list[float], str]:
    if args_cli.controller_backend == "differential":
        controller = DifferentialController(
            name="task319_official_differential",
            wheel_radius=WHEEL_RADIUS_M,
            wheel_base=0.52,
            max_linear_speed=max(float(args_cli.max_linear_speed), 1.0e-6),
            max_angular_speed=max(float(args_cli.max_angular_speed), 1.0e-6),
            max_wheel_speed=max(float(args_cli.max_wheel_speed), 1.0e-6),
        )
        actions = controller.forward(np.array([float(command[0]), float(command[2])], dtype=np.float64))
        left, right = [float(value) for value in actions.joint_velocities]
        return [left, left, right, right], "DifferentialController"

    controller = make_holonomic_controller()
    actions = controller.forward(command)
    return [float(value) for value in actions.joint_velocities], "HolonomicController"


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (WORKSPACE_ROOT / args_cli.output_dir / timestamp).resolve()
    if args_cli.record_debug:
        run_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Creating SimulationContext.", flush=True)
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.4, -2.0, 1.5], [0.0, 0.0, 0.35])
    print("[INFO] Creating official holonomic wheel test scene.", flush=True)
    scene = InteractiveScene(OfficialHolonomicWheelSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False))
    print("[INFO] Resetting simulation.", flush=True)
    sim.reset()
    print("[INFO] Resetting robot state.", flush=True)
    robot = reset_robot(scene)
    sim_dt = sim.get_physics_dt()

    wheel_ids, wheel_names = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
    if len(wheel_ids) != len(WHEEL_JOINTS):
        raise RuntimeError(f"Expected {len(WHEEL_JOINTS)} wheel joints, got {wheel_names}.")
    print(f"[INFO] Wheel joints: {wheel_names} -> {wheel_ids}", flush=True)
    wheel_id_set = {int(item) for item in wheel_ids}
    non_wheel_ids = [idx for idx in range(robot.num_joints) if idx not in wheel_id_set]

    print(f"[INFO] Creating official {args_cli.controller_backend} wheel controller.", flush=True)
    command = np.array([float(args_cli.cmd_vx), float(args_cli.cmd_vy), float(args_cli.cmd_wz)], dtype=np.float64)
    raw_targets, controller_name = make_official_wheel_targets(command)
    wheel_targets = np.asarray(raw_targets, dtype=np.float64) * float(args_cli.joint_velocity_sign)
    target_tensor = torch.tensor(wheel_targets, dtype=torch.float32, device=robot.device).unsqueeze(0)
    locked_joint_target = robot.data.default_joint_pos.clone()

    print(f"[INFO] Warmup steps: {int(args_cli.warmup_steps)}.", flush=True)
    for _ in range(max(int(args_cli.warmup_steps), 0)):
        robot.set_joint_position_target(locked_joint_target[:, non_wheel_ids], joint_ids=non_wheel_ids)
        robot.set_joint_velocity_target(torch.zeros_like(target_tensor), joint_ids=wheel_ids)
        scene.write_data_to_sim()
        sim.step(render=not bool(getattr(args_cli, "headless", False)))
        scene.update(sim_dt)

    initial_pose = robot_pose_xy_yaw(robot)
    samples: list[dict[str, Any]] = []
    print(
        f"[INFO] Official {controller_name} command "
        f"vx={command[0]:.3f}, vy={command[1]:.3f}, wz={command[2]:.3f}; "
        f"wheel_targets_radps={wheel_targets.tolist()}",
        flush=True,
    )
    print(f"[INFO] Drive steps: {int(args_cli.drive_steps)}.", flush=True)
    for step in range(max(int(args_cli.drive_steps), 0)):
        robot.set_joint_position_target(locked_joint_target[:, non_wheel_ids], joint_ids=non_wheel_ids)
        robot.set_joint_velocity_target(target_tensor, joint_ids=wheel_ids)
        scene.write_data_to_sim()
        sim.step(render=not bool(getattr(args_cli, "headless", False)))
        scene.update(sim_dt)
        if step % 30 == 0:
            samples.append({"step": step, "pose": robot_pose_xy_yaw(robot)})
            print(f"[INFO] step={step} pose={samples[-1]['pose']}", flush=True)

    robot.set_joint_velocity_target(torch.zeros_like(target_tensor), joint_ids=wheel_ids)
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(sim_dt)

    final_pose = robot_pose_xy_yaw(robot)
    translation = math.hypot(final_pose[0] - initial_pose[0], final_pose[1] - initial_pose[1])
    yaw_delta = math.atan2(math.sin(final_pose[2] - initial_pose[2]), math.cos(final_pose[2] - initial_pose[2]))
    success = translation >= float(args_cli.min_translation_m) or abs(yaw_delta) >= 0.05
    metadata = {
        "mode": "official_holonomic_wheel_test",
        "controller_backend": args_cli.controller_backend,
        "success": bool(success),
        "command": command.tolist(),
        "wheel_joint_names": wheel_names,
        "wheel_joint_ids": [int(i) for i in wheel_ids],
        "wheel_targets_radps": wheel_targets.tolist(),
        "mecanum_angle_deg": float(args_cli.mecanum_angle_deg),
        "joint_velocity_sign": float(args_cli.joint_velocity_sign),
        "initial_pose": initial_pose,
        "final_pose": final_pose,
        "translation_m": float(translation),
        "yaw_delta_rad": float(yaw_delta),
        "samples": samples,
    }
    if args_cli.record_debug:
        (run_dir / "official_holonomic_wheel_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
