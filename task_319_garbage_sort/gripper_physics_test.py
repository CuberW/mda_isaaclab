"""Attached gripper physics smoke test for task 319.

This script validates the gripper in isolation from navigation and perception:

- Kuavo and the gripper are one articulation through a fixed wrist joint.
- The base is fixed and all non-gripper joints are locked.
- A small grasp block is placed between the finger pads and the gripper closes.

Usage:
    /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
        task_319_garbage_sort/gripper_physics_test.py --headless --smoke_steps 240
"""

from __future__ import annotations

import argparse
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


parser = argparse.ArgumentParser(description="Task 319 attached gripper physics smoke test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--smoke_steps", type=int, default=240)
parser.add_argument("--open_width", type=float, default=0.060)
parser.add_argument("--closed_width", type=float, default=0.0)
parser.add_argument("--close_start_step", type=int, default=40, help="Step at which the block is inserted and the gripper closes.")
parser.add_argument("--hold_assess_after_steps", type=int, default=100, help="Steps after close command before measuring hold drift.")
parser.add_argument("--max_hold_drop", type=float, default=0.015, help="Allowed vertical drop after the grasp has settled.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from task_319_garbage_sort.gripper_robot_urdf import ensure_kuavo_with_gripper_urdf
from task_319_garbage_sort.grasp_pipeline.execution.gripper_control import AttachedParallelGripper

ROOT_DIR = WORKSPACE_ROOT
GRIPPER_URDF = ROOT_DIR / "task_319_garbage_sort/two_finger_gripper.urdf"
KUAVO_BASE_URDF = ROOT_DIR / "kuavo-ros-opensource/src/kuavo_assets/models/biped_s62/urdf/biped_s62.urdf"
KUAVO_URDF = ensure_kuavo_with_gripper_urdf(KUAVO_BASE_URDF, GRIPPER_URDF)


def rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        linear_damping=0.15,
        angular_damping=0.15,
        max_depenetration_velocity=0.5,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=4,
    )


ROBOT_CFG = ArticulationCfg(
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
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "knee_joint": 0.0,
            "leg_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "zhead_1_joint": 0.0,
            "zhead_2_joint": 0.0,
            "zarm_r2_joint": 0.30,
            "zarm_r4_joint": -0.6,
            "left_finger_joint": 0.030,
            "right_finger_joint": 0.030,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "locked_body": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*_joint", "knee_joint", "leg_joint", "waist_.*_joint", "zarm_[lr][1-7]_joint", "zhead_[12]_joint"],
            effort_limit_sim=600.0,
            velocity_limit_sim=0.1,
            stiffness=3000.0,
            damping=350.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*finger_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=0.35,
            stiffness=1200.0,
            damping=90.0,
        ),
    },
)


@configclass
class GripperPhysicsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(4.0, 4.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.3, dynamic_friction=1.0, restitution=0.0, friction_combine_mode="max", restitution_combine_mode="min"),
            color=(0.30, 0.31, 0.30),
        ),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot: ArticulationCfg = ROBOT_CFG
    grasp_block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/grasp_block",
        spawn=sim_utils.CuboidCfg(
            size=(0.030, 0.030, 0.050),
            rigid_props=rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.035),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.35, dynamic_friction=1.05, restitution=0.0, friction_combine_mode="max", restitution_combine_mode="min"),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.18, 0.10), roughness=0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 1.0)),
    )


def apply_gripper_contact_material(num_envs: int) -> None:
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
    for env_id in range(num_envs):
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
    robot: Articulation = scene["robot"]
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.default_root_state[:, 7:]))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    scene.reset()


def place_block_between_fingers(scene: InteractiveScene) -> torch.Tensor:
    robot: Articulation = scene["robot"]
    block: RigidObject = scene["grasp_block"]
    body_ids, _ = robot.find_bodies(["gripper_base"], preserve_order=True)
    if not body_ids:
        raise RuntimeError("Merged gripper body 'gripper_base' was not found in the Kuavo articulation.")
    pose = robot.data.body_link_pose_w[:, body_ids[0], :].clone()
    offset = torch.tensor([[0.115, 0.0, 0.0]], device=robot.device).repeat(robot.num_instances, 1)
    block_pos = pose[:, :3] + quat_apply(pose[:, 3:7], offset)
    block_pose = torch.cat([block_pos, pose[:, 3:7]], dim=1)
    block.write_root_pose_to_sim(block_pose)
    block.write_root_velocity_to_sim(torch.zeros((robot.num_instances, 6), device=robot.device))
    return block_pos[:, 2].clone()


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    robot: Articulation = scene["robot"]
    block: RigidObject = scene["grasp_block"]
    reset_scene(scene)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    initial_block_z = None
    hold_reference_z = None
    gripper = AttachedParallelGripper(robot)
    count = 0
    close_start_step = max(1, args_cli.close_start_step)
    max_block_drop = 0.0
    max_hold_drop = 0.0
    while simulation_app.is_running():
        if count < close_start_step:
            gripper.set_width(args_cli.open_width)
        else:
            if initial_block_z is None:
                initial_block_z = place_block_between_fingers(scene)
            gripper.set_width(args_cli.closed_width)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        if initial_block_z is not None:
            block_z = block.data.root_pos_w[:, 2].clone()
            block_drop = torch.max(initial_block_z - block_z).item()
            max_block_drop = max(max_block_drop, float(block_drop))
            if hold_reference_z is None and count >= close_start_step + args_cli.hold_assess_after_steps:
                hold_reference_z = block_z.clone()
            if hold_reference_z is not None:
                hold_drop = torch.max(hold_reference_z - block_z).item()
                max_hold_drop = max(max_hold_drop, float(hold_drop))
        count += 1
        if args_cli.smoke_steps > 0 and count >= args_cli.smoke_steps:
            joint_pos = robot.data.joint_pos[:, gripper.joint_ids][0].detach().cpu().tolist()
            final_hold_drop = float("nan")
            if hold_reference_z is not None:
                final_hold_drop = float(torch.max(hold_reference_z - block.data.root_pos_w[:, 2]).item())
            print(f"[INFO] Attached gripper joints: {gripper.joint_names}", flush=True)
            print(f"[INFO] Final finger joint positions: {joint_pos}", flush=True)
            print(f"[INFO] Max grasp block drop during close/settle: {max_block_drop:.4f} m", flush=True)
            print(f"[INFO] Max grasp block drop after hold reference: {max_hold_drop:.4f} m", flush=True)
            print(f"[INFO] Final hold drop: {final_hold_drop:.4f} m", flush=True)
            if hold_reference_z is not None and max_hold_drop > args_cli.max_hold_drop:
                raise RuntimeError(
                    f"Gripper hold drift {max_hold_drop:.4f} m exceeds limit {args_cli.max_hold_drop:.4f} m."
                )
            break


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1.0 / 120.0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.45,
            dynamic_friction=1.10,
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="min",
        ),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.15, -1.35, 1.45], [0.46, -0.05, 1.02])
    scene = InteractiveScene(GripperPhysicsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=False))
    apply_gripper_contact_material(args_cli.num_envs)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
