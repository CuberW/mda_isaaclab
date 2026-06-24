"""Isolated Kuavo right-arm control-chain test for task 319.

This script intentionally excludes navigation, obstacles, trash objects, lidar,
and ROS bridges.  It validates the manipulation chain only:

- Kuavo is spawned in an empty area at the wheel-contact height.
- The base is fixed and the wheels/lower body are position locked.
- The right arm tracks a smooth 6D end-effector pose trajectory using IsaacLab's
  DifferentialIKController.

Usage:
    /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
        task_319_garbage_sort/arm_chain_isolated_test.py --headless --smoke_steps 360
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

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


parser = argparse.ArgumentParser(description="Task 319 isolated Kuavo arm-chain test.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of isolated test environments.")
parser.add_argument("--smoke_steps", type=int, default=360, help="Exit after this many physics steps. 0 runs forever.")
parser.add_argument("--trajectory_steps", type=int, default=240, help="Min-jerk interpolation duration in physics steps.")
parser.add_argument("--hold_steps", type=int, default=120, help="Hold duration at the target in physics steps.")
parser.add_argument("--max_joint_step", type=float, default=0.015, help="Per-step joint target clamp in radians.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, subtract_frame_transforms

from task_319_garbage_sort.gripper_robot_urdf import ensure_kuavo_with_gripper_urdf

ROOT_DIR = WORKSPACE_ROOT
GRIPPER_URDF = ROOT_DIR / "task_319_garbage_sort/two_finger_gripper.urdf"
KUAVO_BASE_URDF = ROOT_DIR / "kuavo-ros-opensource/src/kuavo_assets/models/biped_s62/urdf/biped_s62.urdf"
KUAVO_URDF = ensure_kuavo_with_gripper_urdf(KUAVO_BASE_URDF, GRIPPER_URDF)

# From biped_s62.urdf: wheel joint origin z == wheel collision radius == 0.13035 m.
# Therefore base_link root z=0 places the wheels exactly on the ground plane.
ROBOT_ROOT_Z_ON_WHEELS = 0.0
RIGHT_ARM_JOINT_EXPR = "zarm_r[1-7]_joint"
RIGHT_EE_BODY = "zarm_r7_end_effector"
WHEEL_JOINT_EXPR = "wheel_.*_joint"

# Smooth 6D tracking command: target is the measured initial EE pose in base frame
# plus this fixed, test-only 6D offset.  Keeping it local to the initial pose makes
# the test robust to minor URDF/import frame changes while still sending a full
# position + orientation target to the IK controller.
TARGET_EE_DELTA_POS_B = (0.06, -0.05, 0.05)
TARGET_EE_DELTA_ANGLE_RAD = math.radians(6.0)
TARGET_EE_DELTA_AXIS_B = (0.0, 0.0, 1.0)


def rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        linear_damping=0.2,
        angular_damping=0.2,
        max_depenetration_velocity=1.0,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=4,
    )


KUAVO_ARM_TEST_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Kuavo62",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(KUAVO_URDF),
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
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=180.0, damping=18.0),
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
            "zarm_l4_joint": -0.6,
            "zarm_r2_joint": 0.30,
            "zarm_r4_joint": -0.6,
            "left_finger_joint": 0.030,
            "right_finger_joint": 0.030,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "locked_wheels": ImplicitActuatorCfg(
            joint_names_expr=[WHEEL_JOINT_EXPR],
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
            effort_limit_sim=180.0,
            velocity_limit_sim=2.0,
            stiffness=950.0,
            damping=95.0,
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
            effort_limit_sim=60.0,
            velocity_limit_sim=0.35,
            stiffness=1200.0,
            damping=90.0,
        ),
    },
)


@configclass
class IsolatedArmSceneCfg(InteractiveSceneCfg):
    """Empty scene for manipulation-only validation."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(6.0, 6.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0, restitution=0.0),
            color=(0.32, 0.33, 0.32),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2200.0, color=(0.82, 0.86, 0.9))
    )
    robot: ArticulationCfg = KUAVO_ARM_TEST_CFG


def min_jerk(alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def clamp_joint_step(desired: torch.Tensor, previous: torch.Tensor, max_step: float) -> torch.Tensor:
    return previous + torch.clamp(desired - previous, min=-max_step, max=max_step)


def interpolate_quat_shortest_path(q0: torch.Tensor, q1: torch.Tensor, alpha: float) -> torch.Tensor:
    q1 = torch.where(torch.sum(q0 * q1, dim=1, keepdim=True) < 0.0, -q1, q1)
    q = (1.0 - alpha) * q0 + alpha * q1
    return torch.nn.functional.normalize(q, dim=1)


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


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=sim.device)

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[RIGHT_ARM_JOINT_EXPR], body_names=[RIGHT_EE_BODY])
    robot_entity_cfg.resolve(scene)
    ee_body_id = robot_entity_cfg.body_ids[0]
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    reset_robot(scene, robot)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
    root_pose_w = robot.data.root_pose_w
    start_pos_b, start_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    delta_pos = torch.tensor(TARGET_EE_DELTA_POS_B, device=sim.device).repeat(scene.num_envs, 1)
    delta_axis = torch.tensor(TARGET_EE_DELTA_AXIS_B, device=sim.device).repeat(scene.num_envs, 1)
    delta_angle = torch.full((scene.num_envs,), TARGET_EE_DELTA_ANGLE_RAD, device=sim.device)
    delta_quat = quat_from_angle_axis(delta_angle, delta_axis)
    target_pos_b = start_pos_b + delta_pos
    target_quat_b = quat_mul(delta_quat, start_quat_b)

    ik_commands = torch.zeros(scene.num_envs, ik_controller.action_dim, device=sim.device)
    joint_pos_des = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
    previous_joint_target = joint_pos_des.clone()
    locked_joint_target = robot.data.default_joint_pos.clone()
    initial_root_xy = robot.data.root_pos_w[:, 0:2].clone()

    ik_controller.reset()
    max_joint_jump = 0.0
    max_joint_speed = 0.0
    max_base_xy_drift = 0.0
    max_measured_arm_velocity = 0.0
    final_pos_error = float("nan")
    final_rot_error = float("nan")
    count = 0
    total_steps = args_cli.smoke_steps if args_cli.smoke_steps > 0 else args_cli.trajectory_steps + args_cli.hold_steps
    total_steps = max(total_steps, args_cli.trajectory_steps + args_cli.hold_steps)

    print("[INFO] Task 319 isolated manipulation test ready.", flush=True)
    print(f"[INFO] Robot fixed_base={robot.is_fixed_base}; root_z={ROBOT_ROOT_Z_ON_WHEELS:.5f} m.", flush=True)
    print(f"[INFO] Controlled joints: {robot_entity_cfg.joint_names}", flush=True)
    print(
        f"[INFO] Target EE pose in base frame: pos={target_pos_b[0].detach().cpu().tolist()}, "
        f"quat={target_quat_b[0].detach().cpu().tolist()}",
        flush=True,
    )

    while simulation_app.is_running():
        alpha = min_jerk(count / max(1, args_cli.trajectory_steps))
        command_pos_b = (1.0 - alpha) * start_pos_b + alpha * target_pos_b
        command_quat_b = interpolate_quat_shortest_path(start_quat_b, target_quat_b, alpha)
        ik_commands[:, 0:3] = command_pos_b
        ik_commands[:, 3:7] = command_quat_b
        ik_controller.set_command(ik_commands)

        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        joint_vel = robot.data.joint_vel[:, robot_entity_cfg.joint_ids]
        max_measured_arm_velocity = max(max_measured_arm_velocity, float(torch.max(torch.abs(joint_vel)).item()))
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        joint_pos_des = clamp_joint_step(joint_pos_des, previous_joint_target, args_cli.max_joint_step)

        joint_step = torch.max(torch.abs(joint_pos_des - previous_joint_target)).item()
        max_joint_jump = max(max_joint_jump, joint_step)
        max_joint_speed = max(max_joint_speed, joint_step / sim_dt)
        previous_joint_target = joint_pos_des.clone()

        robot.set_joint_position_target(locked_joint_target)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        root_xy = robot.data.root_pos_w[:, 0:2]
        max_base_xy_drift = max(max_base_xy_drift, float(torch.linalg.norm(root_xy - initial_root_xy, dim=1).max().item()))
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        root_pose_w = robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        final_pos_error = float(torch.linalg.norm(target_pos_b - ee_pos_b, dim=1).max().item())
        quat_dot = torch.abs(torch.sum(target_quat_b * ee_quat_b, dim=1)).clamp(max=1.0)
        final_rot_error = float((2.0 * torch.acos(quat_dot)).max().item())
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(target_pos_b + scene.env_origins, target_quat_b)

        count += 1
        if args_cli.smoke_steps > 0 and count >= total_steps:
            print(f"[INFO] Completed {count} isolated manipulation steps.", flush=True)
            print(f"[INFO] Final EE position error: {final_pos_error:.4f} m.", flush=True)
            print(f"[INFO] Final EE orientation error: {math.degrees(final_rot_error):.2f} deg.", flush=True)
            print(f"[INFO] Max right-arm joint target step: {max_joint_jump:.5f} rad.", flush=True)
            print(f"[INFO] Max right-arm joint target speed: {max_joint_speed:.3f} rad/s.", flush=True)
            print(f"[INFO] Max measured right-arm joint velocity: {max_measured_arm_velocity:.3f} rad/s.", flush=True)
            print(f"[INFO] Max base XY drift: {max_base_xy_drift:.6f} m.", flush=True)
            break


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.4, -2.8, 1.7], [0.0, 0.0, 0.7])
    scene_cfg = IsolatedArmSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
