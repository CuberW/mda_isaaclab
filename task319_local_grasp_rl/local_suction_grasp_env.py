"""Local RL environment for Task 319 hover-to-grasp control.

The environment is deliberately scoped to the final 10 cm after the main
Task319 stack has already selected the object, navigated to a grasp standpoint,
and moved the right gripper to a hover pose.  The policy only performs local
TCP corrections, the final descent, physical gripper closure, and lift.

Actor observations are limited to signals the 3.19 mainline can provide at
runtime: current TCP/proprioception, a noisy RGB-D-style estimate of the grasp
target relative to the TCP, gripper opening, and previous action.  Rewards may
still use simulator truth during training, but the policy input is not the
object root pose or object velocity.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG, FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix, quat_inv, skew_symmetric_matrix, subtract_frame_transforms

from task_319_garbage_sort.curobo_right_arm import RIGHT_ARM_CARRY_CONFIG, RIGHT_ARM_JOINT_NAMES
from task_319_garbage_sort.gripper_robot_urdf import ensure_kuavo_with_gripper_urdf


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
KUAVO_BASE_URDF = WORKSPACE_ROOT / "kuavo-ros-opensource/src/kuavo_assets/models/biped_s62/urdf/biped_s62.urdf"
GRIPPER_URDF = WORKSPACE_ROOT / "task_319_garbage_sort/two_finger_gripper.urdf"
KUAVO_WITH_GRIPPER_URDF = ensure_kuavo_with_gripper_urdf(KUAVO_BASE_URDF, GRIPPER_URDF)

ROBOT_ROOT_Z_ON_WHEELS = 0.0
TABLE_SURFACE_Z = 0.54
TABLE_CENTER_Z = 0.51
TABLE_THICKNESS = 0.06
OBJECT_SIZE = (0.050, 0.050, 0.050)
OBJECT_HALF_Z = OBJECT_SIZE[2] * 0.5
OBJECT_SPAWN_XY = (0.36, -0.22)
# Right-arm pose whose TCP starts near the reachable grasp workspace at z=0.70 m.
# The object is randomized around a lower-x reachable band, so the policy must
# search laterally instead of always finding the object directly below the TCP.
# Computed once with cuRobo from RIGHT_ARM_CARRY_CONFIG using position-only IK.
RIGHT_ARM_OBJECT_HOVER_CONFIG = (
    0.6359651684761047,
    -0.07164611667394638,
    0.009798305109143257,
    -1.8747950792312622,
    -0.5749091506004333,
    0.052881497889757156,
    0.17181889712810516,
)
RIGHT_EE_BODY_CANDIDATES = ("gripper_base", "zarm_r7_end_effector")
GRIPPER_LOCAL_TCP_OFFSET = (0.115, 0.0, 0.0)
GRIPPER_OPEN_M = 0.055
GRIPPER_CLOSED_M = 0.0


def _rigid_props(*, kinematic: bool = False) -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        kinematic_enabled=kinematic,
        disable_gravity=kinematic,
        linear_damping=0.08,
        angular_damping=0.12,
        max_depenetration_velocity=1.0,
        solver_position_iteration_count=12,
        solver_velocity_iteration_count=4,
    )


def _collision_props(contact_offset: float = 0.003, rest_offset: float = 0.0) -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=max(contact_offset, rest_offset + 1.0e-4),
        rest_offset=max(0.0, rest_offset),
    )


def _material(static_friction: float, dynamic_friction: float) -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )


def _surface(color: tuple[float, float, float]) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.75)


def _kuavo_robot_cfg() -> ArticulationCfg:
    right_home = dict(zip(RIGHT_ARM_JOINT_NAMES, RIGHT_ARM_CARRY_CONFIG, strict=True))
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Kuavo62",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(KUAVO_WITH_GRIPPER_URDF),
            fix_base=True,
            merge_fixed_joints=False,
            link_density=1000.0,
            collision_from_visuals=True,
            collider_type="convex_hull",
            self_collision=False,
            rigid_props=_rigid_props(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=160.0, damping=18.0),
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
                "left_finger_joint": GRIPPER_OPEN_M,
                "right_finger_joint": GRIPPER_OPEN_M,
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
                joint_names_expr=["zarm_r[1-7]_joint"],
                effort_limit_sim=220.0,
                velocity_limit_sim=2.4,
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
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[".*finger_joint"],
                effort_limit_sim=70.0,
                velocity_limit_sim=0.35,
                stiffness=1200.0,
                damping=90.0,
            ),
        },
    )


class Task319LocalGraspWindow(BaseEnvWindow):
    """Small debug panel for the local grasp environment."""

    def __init__(self, env: "Task319LocalSuctionGraspEnv", window_name: str = "Task319 Local Grasp"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("target", self.env)


@configclass
class Task319LocalSuctionGraspEnvCfg(DirectRLEnvCfg):
    episode_length_s = 5.0
    decimation = 2
    action_space = 4
    observation_space = 28
    state_space = 0
    debug_vis = True
    ui_window_class_type = Task319LocalGraspWindow

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="min",
            static_friction=4.0,
            dynamic_friction=3.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    robot: ArticulationCfg = _kuavo_robot_cfg()

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(8.0, 8.0),
            physics_material=_material(1.0, 0.8),
            color=(0.32, 0.33, 0.32),
        ),
    )

    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TableTop",
        spawn=sim_utils.CuboidCfg(
            size=(1.20, 0.80, TABLE_THICKNESS),
            rigid_props=_rigid_props(kinematic=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=80.0),
            collision_props=_collision_props(0.003),
            physics_material=_material(0.8, 0.6),
            visual_material=_surface((0.48, 0.36, 0.24)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.72, -0.16, TABLE_CENTER_Z)),
    )

    grasp_object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GraspObject",
        spawn=sim_utils.CuboidCfg(
            size=OBJECT_SIZE,
            rigid_props=_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.035),
            collision_props=_collision_props(0.002),
            physics_material=_material(4.0, 3.0),
            visual_material=_surface((0.12, 0.32, 0.72)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*OBJECT_SPAWN_XY, TABLE_SURFACE_Z + OBJECT_HALF_Z)),
    )

    # Local controller parameters. The main 3.19 stack should already be at a
    # hover pose; this policy only handles the final TCP descent, close, lift.
    max_delta_pos_m = 0.012
    max_delta_rot_rad = 0.080
    max_joint_step_rad = 0.032
    dls_damping = 0.018
    ik_orientation_axis_weight = 0.10
    enforce_topdown_grasp_orientation = True
    align_topdown_jaw_to_object_short_axis = False
    grasp_depth_m = 0.020
    lift_success_height_m = 0.025
    workspace_radius_m = 0.12
    tcp_min_height_m = TABLE_SURFACE_Z + 0.010
    tcp_max_height_m = TABLE_SURFACE_Z + 0.22
    object_spawn_xy_range_m = 0.040
    initial_right_arm_joint_noise_rad = 0.008
    target_estimate_xy_noise_std_m = 0.006
    target_estimate_z_noise_std_m = 0.004

    teacher_enabled = False
    teacher_only = False
    teacher_warmup_steps = 10000
    teacher_decay_steps = 80000
    teacher_mix_start = 0.80
    teacher_mix_end = 0.0
    teacher_hover_height_m = 0.12
    teacher_close_xy_tolerance_m = 0.040
    teacher_close_z_tolerance_m = 0.035
    teacher_close_hold_steps = 24

    stage_reach_xy_tolerance_m = 0.045
    stage_descend_z_tolerance_m = 0.040
    stage_lift_height_m = 0.018

    # Debug-only latch. Strict training/evaluation must keep this disabled so
    # success can only come from contact physics and gripper closure.
    grasp_latch_enabled = False
    grasp_latch_xy_tolerance_m = 0.045
    grasp_latch_z_tolerance_m = 0.040
    grasp_latch_min_close_cmd = 0.75
    grasp_latch_release_cmd = 0.35
    grasp_latch_max_offset_m = 0.070

    success_close_cmd_threshold = 0.75
    strict_physical_success = True
    success_tcp_object_dist_m = 0.065
    success_min_actual_finger_width_m = 0.005
    object_push_penalty_start_m = 0.03

    reward_xy_align_scale = 1.2
    reward_xy_progress_scale = 70.0
    reward_hover_height_scale = 0.35
    reward_z_align_scale = 0.8
    reward_z_progress_scale = 65.0
    reward_near_grasp_scale = 0.6
    reward_stage_reach_bonus = 0.35
    reward_stage_descend_bonus = 0.75
    reward_stage_close_bonus = 1.0
    reward_stage_lift_bonus = 2.0
    reward_grasp_latch_scale = 1.2
    reward_close_near_scale = 1.2
    reward_lift_progress_scale = 120.0
    reward_success = 800.0
    reward_downward_open_scale = 0.5
    reward_upward_closed_scale = 1.0
    reward_teacher_action_scale = 0.15
    premature_close_penalty_scale = 3.0
    action_penalty_scale = 0.03
    object_push_penalty_scale = 16.0
    table_penalty = 10.0
    time_penalty = 0.01


class Task319LocalSuctionGraspEnv(DirectRLEnv):
    cfg: Task319LocalSuctionGraspEnvCfg

    def __init__(self, cfg: Task319LocalSuctionGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._last_actions = torch.zeros_like(self._actions)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._object_initial_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._object_initial_z = torch.zeros(self.num_envs, device=self.device)
        self._target_estimate_offset_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_tcp_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._gripper_close_cmd = torch.zeros(self.num_envs, device=self.device)
        self._min_tcp_target_dist = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._max_object_lift = torch.zeros(self.num_envs, device=self.device)
        self._max_close_cmd = torch.zeros(self.num_envs, device=self.device)
        self._max_grasp_retained = torch.zeros(self.num_envs, device=self.device)
        self._max_stage_reach = torch.zeros(self.num_envs, device=self.device)
        self._max_stage_descend = torch.zeros(self.num_envs, device=self.device)
        self._max_stage_close = torch.zeros(self.num_envs, device=self.device)
        self._max_stage_lift = torch.zeros(self.num_envs, device=self.device)
        self._teacher_used_steps = torch.zeros(self.num_envs, device=self.device)
        self._teacher_total_steps = torch.zeros(self.num_envs, device=self.device)
        self._prev_xy_dist = torch.full((self.num_envs,), float(self.cfg.workspace_radius_m), device=self.device)
        self._prev_z_abs = torch.full((self.num_envs,), float(self.cfg.teacher_hover_height_m), device=self.device)
        self._prev_object_lift = torch.zeros(self.num_envs, device=self.device)
        self._last_teacher_action = torch.zeros_like(self._actions)
        self._last_teacher_mix_probability = torch.tensor(0.0, device=self.device)
        self._teacher_close_steps = torch.zeros(self.num_envs, device=self.device)
        self._teacher_lift_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_latch_offset_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._max_grasp_latched = torch.zeros(self.num_envs, device=self.device)
        self._global_policy_step = 0
        self._joint_pos_target = self._robot.data.default_joint_pos.clone()
        self._tcp_offset_local = torch.tensor(GRIPPER_LOCAL_TCP_OFFSET, dtype=torch.float32, device=self.device).reshape(1, 3, 1)

        self._right_joint_ids, self._right_joint_names = self._robot.find_joints(RIGHT_ARM_JOINT_NAMES, preserve_order=True)
        self._right_joint_ids = list(self._right_joint_ids)
        if len(self._right_joint_ids) != len(RIGHT_ARM_JOINT_NAMES):
            raise RuntimeError(f"Expected 7 right-arm joints, got {self._right_joint_names}")

        self._finger_joint_ids, _ = self._robot.find_joints(".*finger_joint")
        self._finger_joint_ids = list(self._finger_joint_ids)

        self._ee_body_id = self._resolve_ee_body_id()
        self._ee_jacobi_idx = self._ee_body_id - 1 if self._robot.is_fixed_base else self._ee_body_id
        self._jacobi_joint_ids = self._right_joint_ids if self._robot.is_fixed_base else [joint_id + 6 for joint_id in self._right_joint_ids]
        self._joint_lower = self._robot.data.soft_joint_pos_limits[0, :, 0].clone()
        self._joint_upper = self._robot.data.soft_joint_pos_limits[0, :, 1].clone()
        self._episode_sums = {
            "xy_align": torch.zeros(self.num_envs, device=self.device),
            "z_align": torch.zeros(self.num_envs, device=self.device),
            "near_grasp": torch.zeros(self.num_envs, device=self.device),
            "stage_reach": torch.zeros(self.num_envs, device=self.device),
            "stage_descend": torch.zeros(self.num_envs, device=self.device),
            "close_near": torch.zeros(self.num_envs, device=self.device),
            "stage_close": torch.zeros(self.num_envs, device=self.device),
            "grasp_latch": torch.zeros(self.num_envs, device=self.device),
            "stage_lift": torch.zeros(self.num_envs, device=self.device),
            "lift": torch.zeros(self.num_envs, device=self.device),
            "teacher_action": torch.zeros(self.num_envs, device=self.device),
            "success_bonus": torch.zeros(self.num_envs, device=self.device),
            "penalty": torch.zeros(self.num_envs, device=self.device),
        }

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._table = RigidObject(self.cfg.table)
        self._object = RigidObject(self.cfg.grasp_object)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["grasp_object"] = self._object

        self.cfg.ground.spawn.func(self.cfg.ground.prim_path, self.cfg.ground.spawn)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.ground.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.82, 0.86, 0.9))
        light_cfg.func("/World/Light", light_cfg)

    def _resolve_ee_body_id(self) -> int:
        for body_name in RIGHT_EE_BODY_CANDIDATES:
            body_ids, _ = self._robot.find_bodies(body_name)
            if body_ids:
                return int(body_ids[0])
        raise RuntimeError(f"Cannot find any right end-effector body from {RIGHT_EE_BODY_CANDIDATES}")

    def _gripper_base_rot_w(self) -> torch.Tensor:
        return matrix_from_quat(self._robot.data.body_quat_w[:, self._ee_body_id])

    def _gripper_base_pos_w(self) -> torch.Tensor:
        return self._robot.data.body_pos_w[:, self._ee_body_id]

    def _tcp_pos_w(self) -> torch.Tensor:
        offset = self._tcp_offset_local.repeat(self.num_envs, 1, 1)
        return self._gripper_base_pos_w() + torch.bmm(self._gripper_base_rot_w(), offset).squeeze(-1)

    def _tcp_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pose_w = self._robot.data.root_pose_w
        return subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            self._tcp_pos_w(),
            self._robot.data.body_quat_w[:, self._ee_body_id],
        )

    def _desired_tcp_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(self.cfg.enforce_topdown_grasp_orientation):
            desired_tcp_quat_w = quat_from_matrix(self._desired_grasp_rot_w())
        else:
            desired_tcp_quat_w = self._robot.data.body_quat_w[:, self._ee_body_id]
        root_pose_w = self._robot.data.root_pose_w
        return subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            self._desired_tcp_pos_w,
            desired_tcp_quat_w,
        )

    def _tcp_jacobian_b(self) -> torch.Tensor:
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, 0:6, :][:, :, self._jacobi_joint_ids].clone()
        root_quat_w = self._robot.data.root_pose_w[:, 3:7]
        root_rot_matrix = matrix_from_quat(quat_inv(root_quat_w))
        jacobian[:, 0:3, :] = torch.bmm(root_rot_matrix, jacobian[:, 0:3, :])
        jacobian[:, 3:6, :] = torch.bmm(root_rot_matrix, jacobian[:, 3:6, :])

        offset = self._tcp_offset_local.reshape(1, 3).repeat(self.num_envs, 1)
        jacobian[:, 0:3, :] += torch.bmm(-skew_symmetric_matrix(offset), jacobian[:, 3:6, :])
        return jacobian

    def _object_top_w(self) -> torch.Tensor:
        return self._object.data.root_pos_w + torch.tensor((0.0, 0.0, OBJECT_HALF_Z), device=self.device)

    def _object_grasp_target_w(self) -> torch.Tensor:
        target = self._object_top_w().clone()
        target[:, 2] -= float(self.cfg.grasp_depth_m)
        return target

    def _estimated_grasp_target_w(self) -> torch.Tensor:
        return self._object_grasp_target_w() + self._target_estimate_offset_w

    def _desired_grasp_rot_w(self) -> torch.Tensor:
        """Official-style top-down grasp frame.

        Gripper/TCP local x points downward into the object. By default, local
        y keeps the current reachable horizontal jaw yaw, matching the official
        Lift-Cube style of using a stable fixed top-down grasp orientation
        instead of re-solving a new yaw from the object every step.
        """
        x_axis = torch.zeros((self.num_envs, 3), device=self.device)
        x_axis[:, 2] = -1.0

        if bool(self.cfg.align_topdown_jaw_to_object_short_axis):
            object_rot = matrix_from_quat(self._object.data.root_quat_w)
            short_local_axis = torch.tensor((0.0, 1.0, 0.0), device=self.device)
            if float(OBJECT_SIZE[0]) <= float(OBJECT_SIZE[1]):
                short_local_axis = torch.tensor((1.0, 0.0, 0.0), device=self.device)
            y_axis = torch.matmul(object_rot, short_local_axis.reshape(1, 3, 1).repeat(self.num_envs, 1, 1)).squeeze(-1)
        else:
            y_axis = self._gripper_base_rot_w()[:, :, 1].clone()
        y_axis[:, 2] = 0.0
        fallback_y = torch.zeros_like(y_axis)
        fallback_y[:, 1] = 1.0
        y_norm = torch.linalg.norm(y_axis, dim=1, keepdim=True)
        y_axis = torch.where(y_norm > 1.0e-6, y_axis, fallback_y)
        y_axis = y_axis / torch.linalg.norm(y_axis, dim=1, keepdim=True).clamp_min(1.0e-6)

        z_axis = torch.cross(x_axis, y_axis, dim=1)
        z_axis = z_axis / torch.linalg.norm(z_axis, dim=1, keepdim=True).clamp_min(1.0e-6)
        y_axis = torch.cross(z_axis, x_axis, dim=1)
        y_axis = y_axis / torch.linalg.norm(y_axis, dim=1, keepdim=True).clamp_min(1.0e-6)
        return torch.stack((x_axis, y_axis, z_axis), dim=2)

    def _orientation_error_w(self, current_rot: torch.Tensor, desired_rot: torch.Tensor) -> torch.Tensor:
        err = (
            torch.cross(current_rot[:, :, 0], desired_rot[:, :, 0], dim=1)
            + torch.cross(current_rot[:, :, 1], desired_rot[:, :, 1], dim=1)
            + torch.cross(current_rot[:, :, 2], desired_rot[:, :, 2], dim=1)
        ) * 0.5
        err_norm = torch.linalg.norm(err, dim=1, keepdim=True).clamp_min(1.0e-6)
        return err * torch.clamp(float(self.cfg.max_delta_rot_rad) / err_norm, max=1.0)

    def _topdown_approach_error_b(self) -> torch.Tensor:
        current_x_w = self._gripper_base_rot_w()[:, :, 0]
        desired_x_w = torch.zeros_like(current_x_w)
        desired_x_w[:, 2] = -1.0
        err_w = torch.cross(current_x_w, desired_x_w, dim=1)
        err_norm = torch.linalg.norm(err_w, dim=1, keepdim=True).clamp_min(1.0e-6)
        err_w = err_w * torch.clamp(float(self.cfg.max_delta_rot_rad) / err_norm, max=1.0)
        root_rot_matrix = matrix_from_quat(quat_inv(self._robot.data.root_pose_w[:, 3:7]))
        return torch.bmm(root_rot_matrix, err_w.unsqueeze(-1)).squeeze(-1)

    def _object_pos_with_latch_w(self, tcp_pos: torch.Tensor) -> torch.Tensor:
        object_pos = self._object.data.root_pos_w
        if not bool(self.cfg.grasp_latch_enabled):
            return object_pos
        latched_pos = tcp_pos + self._grasp_latch_offset_w
        latched_pos[:, 2] = torch.maximum(latched_pos[:, 2], self._object_initial_z)
        return torch.where(self._grasp_latched.unsqueeze(-1), latched_pos, object_pos)

    def _update_grasp_latch(
        self,
        tcp_pos: torch.Tensor,
        target: torch.Tensor,
        close_cmd: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(self.cfg.grasp_latch_enabled):
            return self._object.data.root_pos_w

        release = self._grasp_latched & (close_cmd < float(self.cfg.grasp_latch_release_cmd))
        if torch.any(release):
            self._grasp_latched[release] = False
            self._grasp_latch_offset_w[release] = 0.0

        rel = tcp_pos - target
        xy_dist = torch.linalg.norm(rel[:, 0:2], dim=1)
        z_abs = torch.abs(rel[:, 2])
        object_pos = self._object.data.root_pos_w
        latch_offset = object_pos - tcp_pos
        latch_ready = (
            (~self._grasp_latched)
            & (close_cmd >= float(self.cfg.grasp_latch_min_close_cmd))
            & (xy_dist < float(self.cfg.grasp_latch_xy_tolerance_m))
            & (z_abs < float(self.cfg.grasp_latch_z_tolerance_m))
            & (torch.linalg.norm(latch_offset, dim=1) < float(self.cfg.grasp_latch_max_offset_m))
        )
        if torch.any(latch_ready):
            self._grasp_latched[latch_ready] = True
            self._grasp_latch_offset_w[latch_ready] = latch_offset[latch_ready]

        effective_object_pos = self._object_pos_with_latch_w(tcp_pos)
        latched_ids = torch.nonzero(self._grasp_latched, as_tuple=False).squeeze(-1)
        if latched_ids.numel() > 0:
            root_pose = torch.cat((effective_object_pos[latched_ids], self._object.data.root_quat_w[latched_ids]), dim=1)
            self._object.write_root_pose_to_sim(root_pose, env_ids=latched_ids)
            self._object.write_root_velocity_to_sim(torch.zeros((latched_ids.numel(), 6), device=self.device), env_ids=latched_ids)
        return effective_object_pos

    def _finger_open_fraction(self) -> torch.Tensor:
        if self._finger_joint_ids:
            finger_pos = self._robot.data.joint_pos[:, self._finger_joint_ids]
            finger_open = torch.mean((finger_pos - GRIPPER_CLOSED_M) / max(GRIPPER_OPEN_M - GRIPPER_CLOSED_M, 1.0e-6), dim=1)
            return torch.clamp(finger_open, 0.0, 1.0)
        return 1.0 - self._gripper_close_cmd

    def _finger_width_m(self) -> torch.Tensor:
        if self._finger_joint_ids:
            finger_pos = self._robot.data.joint_pos[:, self._finger_joint_ids]
            return torch.sum(finger_pos, dim=1)
        per_finger = GRIPPER_OPEN_M - self._gripper_close_cmd * (GRIPPER_OPEN_M - GRIPPER_CLOSED_M)
        return 2.0 * per_finger

    def _teacher_mix_probability(self) -> float:
        if not bool(self.cfg.teacher_enabled):
            return 0.0
        if bool(self.cfg.teacher_only):
            return 1.0
        step = int(self._global_policy_step)
        warmup = max(0, int(self.cfg.teacher_warmup_steps))
        if step < warmup:
            return 1.0
        decay = max(1, int(self.cfg.teacher_decay_steps))
        alpha = min(1.0, max(0.0, float(step - warmup) / float(decay)))
        start = float(self.cfg.teacher_mix_start)
        end = float(self.cfg.teacher_mix_end)
        return max(0.0, min(1.0, (1.0 - alpha) * start + alpha * end))

    def _scripted_teacher_action(
        self,
        tcp_pos: torch.Tensor | None = None,
        estimated_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tcp_pos is None:
            tcp_pos = self._tcp_pos_w()
        if estimated_target is None:
            estimated_target = self._estimated_grasp_target_w()

        rel = tcp_pos - estimated_target
        xy_dist = torch.linalg.norm(rel[:, 0:2], dim=1)
        z_above_target = rel[:, 2]

        close_ready = (
            (xy_dist < float(self.cfg.teacher_close_xy_tolerance_m))
            & (torch.abs(z_above_target) < float(self.cfg.teacher_close_z_tolerance_m))
        )
        self._teacher_close_steps = torch.where(close_ready, self._teacher_close_steps + 1.0, torch.zeros_like(self._teacher_close_steps))
        self._teacher_lift_latched |= close_ready & (self._teacher_close_steps >= float(self.cfg.teacher_close_hold_steps))

        desired = tcp_pos.clone()
        grasp_z = torch.clamp(
            estimated_target[:, 2],
            min=float(self.cfg.tcp_min_height_m),
            max=float(self.cfg.tcp_max_height_m),
        )

        lift_mask = self._teacher_lift_latched
        servo_mask = ~(close_ready | lift_mask)
        desired[servo_mask, 0:2] = estimated_target[servo_mask, 0:2]
        desired[servo_mask, 2] = grasp_z[servo_mask]

        close_mask = close_ready & ~lift_mask
        desired[close_mask] = tcp_pos[close_mask]

        desired[lift_mask, 0:2] = tcp_pos[lift_mask, 0:2]
        desired[lift_mask, 2] = torch.clamp(
            tcp_pos[lift_mask, 2] + float(self.cfg.max_delta_pos_m),
            min=float(self.cfg.tcp_min_height_m),
            max=float(self.cfg.tcp_max_height_m),
        )

        actions = torch.zeros_like(self._actions)
        actions[:, 0:3] = torch.clamp((desired - tcp_pos) / max(float(self.cfg.max_delta_pos_m), 1.0e-6), -1.0, 1.0)
        actions[:, 3] = torch.where(close_ready | lift_mask, torch.ones(self.num_envs, device=self.device), -torch.ones(self.num_envs, device=self.device))
        return actions

    def _stage_masks(
        self,
        tcp_pos: torch.Tensor,
        target: torch.Tensor,
        object_lift: torch.Tensor,
        close_cmd: torch.Tensor,
        object_pos: torch.Tensor | None = None,
        attached: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if object_pos is None:
            object_pos = self._object.data.root_pos_w
        rel = tcp_pos - target
        xy_dist = torch.linalg.norm(rel[:, 0:2], dim=1)
        z_abs = torch.abs(rel[:, 2])
        reached = xy_dist < float(self.cfg.stage_reach_xy_tolerance_m)
        descended = reached & (z_abs < float(self.cfg.stage_descend_z_tolerance_m))
        closed_cmd = close_cmd > float(self.cfg.success_close_cmd_threshold)
        closed = descended & closed_cmd
        tcp_object_dist = torch.linalg.norm(tcp_pos - object_pos, dim=1)
        if attached is None:
            attached = tcp_object_dist < float(self.cfg.success_tcp_object_dist_m)
        lifted = (
            closed_cmd
            & attached
            & (object_lift > float(self.cfg.stage_lift_height_m))
            & (tcp_object_dist < float(self.cfg.success_tcp_object_dist_m))
        )
        return {
            "xy_dist": xy_dist,
            "z_abs": z_abs,
            "reached": reached,
            "descended": descended,
            "closed": closed,
            "attached": attached,
            "lifted": lifted,
        }

    def _grasp_success_mask(
        self,
        tcp_pos: torch.Tensor | None = None,
        object_lift: torch.Tensor | None = None,
        object_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tcp_pos is None:
            tcp_pos = self._tcp_pos_w()
        if object_pos is None:
            object_pos = self._object_pos_with_latch_w(tcp_pos)
        if object_lift is None:
            object_lift = object_pos[:, 2] - self._object_initial_z

        tcp_object_dist = torch.linalg.norm(tcp_pos - object_pos, dim=1)
        closed = self._gripper_close_cmd > float(self.cfg.success_close_cmd_threshold)
        lifted = object_lift > float(self.cfg.lift_success_height_m)
        retained = tcp_object_dist < float(self.cfg.success_tcp_object_dist_m)
        if bool(self.cfg.grasp_latch_enabled):
            attached = self._grasp_latched
        elif bool(self.cfg.strict_physical_success):
            finger_blocked = self._finger_width_m() > float(self.cfg.success_min_actual_finger_width_m)
            attached = retained & finger_blocked
        else:
            attached = retained
        return closed & lifted & retained & attached, retained

    def _pre_physics_step(self, actions: torch.Tensor):
        policy_actions = actions.clone().clamp(-1.0, 1.0)
        tcp_pos = self._tcp_pos_w()
        estimated_target = self._estimated_grasp_target_w()
        teacher_actions = self._scripted_teacher_action(tcp_pos=tcp_pos, estimated_target=estimated_target)
        self._last_teacher_action = teacher_actions

        mix_probability = self._teacher_mix_probability()
        self._last_teacher_mix_probability = torch.tensor(float(mix_probability), device=self.device)
        if mix_probability > 0.0:
            teacher_mask = torch.rand(self.num_envs, device=self.device) < float(mix_probability)
            mixed_actions = torch.where(teacher_mask.unsqueeze(-1), teacher_actions, policy_actions)
            self._teacher_used_steps += teacher_mask.float()
            self._teacher_total_steps += 1.0
            try:
                actions.copy_(mixed_actions)
            except Exception:
                pass
            self._actions = mixed_actions.clamp(-1.0, 1.0)
        else:
            self._teacher_total_steps += 1.0
            self._actions = policy_actions
        self._global_policy_step += 1

        desired = tcp_pos + self._actions[:, 0:3] * self.cfg.max_delta_pos_m

        lower = estimated_target - self.cfg.workspace_radius_m
        upper = estimated_target + self.cfg.workspace_radius_m
        lower[:, 2] = self.cfg.tcp_min_height_m
        upper[:, 2] = self.cfg.tcp_max_height_m
        self._desired_tcp_pos_w = torch.minimum(torch.maximum(desired, lower), upper)
        self._gripper_close_cmd = torch.clamp((self._actions[:, 3] + 1.0) * 0.5, 0.0, 1.0)

    def _apply_action(self):
        tcp_pos_b, tcp_quat_b = self._tcp_pose_b()
        desired_tcp_pos_b, desired_tcp_quat_b = self._desired_tcp_pose_b()
        del tcp_quat_b, desired_tcp_quat_b
        position_error = desired_tcp_pos_b - tcp_pos_b
        if bool(self.cfg.enforce_topdown_grasp_orientation):
            orientation_error = self._topdown_approach_error_b()
        else:
            orientation_error = torch.zeros_like(position_error)
        pose_error = torch.cat((position_error, orientation_error), dim=1)
        jacobian = self._tcp_jacobian_b()
        orientation_weight = float(self.cfg.ik_orientation_axis_weight)
        pose_error[:, 3:6] *= orientation_weight
        jacobian[:, 3:6, :] *= orientation_weight

        joint_pos = self._robot.data.joint_pos
        jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (float(self.cfg.dls_damping) ** 2) * torch.eye(n=6, device=self.device)
        delta_q_raw = (
            jacobian_t
            @ torch.inverse(jacobian @ jacobian_t + lambda_matrix)
            @ pose_error.unsqueeze(-1)
        ).squeeze(-1)
        desired_joints_raw = joint_pos[:, self._right_joint_ids] + delta_q_raw
        delta_q = desired_joints_raw - joint_pos[:, self._right_joint_ids]
        delta_q = torch.clamp(delta_q, -self.cfg.max_joint_step_rad, self.cfg.max_joint_step_rad)

        desired_joints = joint_pos[:, self._right_joint_ids] + delta_q
        desired_joints = torch.minimum(
            torch.maximum(desired_joints, self._joint_lower[self._right_joint_ids]),
            self._joint_upper[self._right_joint_ids],
        )

        self._joint_pos_target[:] = self._robot.data.default_joint_pos
        self._joint_pos_target[:, self._right_joint_ids] = desired_joints
        if self._finger_joint_ids:
            finger_width = GRIPPER_OPEN_M - self._gripper_close_cmd * (GRIPPER_OPEN_M - GRIPPER_CLOSED_M)
            finger_target = finger_width.unsqueeze(-1).repeat(1, len(self._finger_joint_ids))
            self._joint_pos_target[:, self._finger_joint_ids] = finger_target

        self._robot.set_joint_position_target(self._joint_pos_target)

    def _get_observations(self) -> dict:
        tcp_pos = self._tcp_pos_w()
        estimated_target = self._estimated_grasp_target_w()
        rel_target = (estimated_target - tcp_pos) / 0.16
        tcp_vel = self._robot.data.body_lin_vel_w[:, self._ee_body_id] / 0.50

        joint_pos = self._robot.data.joint_pos[:, self._right_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._right_joint_ids]
        lower = self._joint_lower[self._right_joint_ids]
        upper = self._joint_upper[self._right_joint_ids]
        joint_pos_scaled = 2.0 * (joint_pos - lower) / torch.clamp(upper - lower, min=1.0e-6) - 1.0

        if self._finger_joint_ids:
            finger_pos = self._robot.data.joint_pos[:, self._finger_joint_ids]
            finger_open = torch.mean((finger_pos - GRIPPER_CLOSED_M) / max(GRIPPER_OPEN_M - GRIPPER_CLOSED_M, 1.0e-6), dim=1, keepdim=True)
            finger_open = torch.clamp(finger_open, 0.0, 1.0)
        else:
            finger_open = 1.0 - self._gripper_close_cmd.unsqueeze(-1)
        tcp_height = (tcp_pos[:, 2:3] - TABLE_SURFACE_Z) / 0.25
        target_height = (estimated_target[:, 2:3] - TABLE_SURFACE_Z) / 0.25
        close_cmd = self._gripper_close_cmd.unsqueeze(-1)

        obs = torch.cat(
            (
                rel_target,
                tcp_vel,
                joint_pos_scaled,
                joint_vel * 0.20,
                finger_open,
                tcp_height,
                target_height,
                close_cmd,
                self._last_actions,
            ),
            dim=-1,
        )
        self._last_actions[:] = self._actions
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        tcp_pos = self._tcp_pos_w()
        target = self._object_grasp_target_w()
        close_cmd = self._gripper_close_cmd
        object_pos = self._update_grasp_latch(tcp_pos, target, close_cmd)
        rel = tcp_pos - target
        dist = torch.linalg.norm(rel, dim=1)
        pushed = torch.linalg.norm(object_pos[:, 0:2] - self._object_initial_xy, dim=1)
        object_lift = torch.clamp(object_pos[:, 2] - self._object_initial_z, min=0.0)
        lift_progress = torch.clamp(object_lift / max(float(self.cfg.lift_success_height_m), 1.0e-6), 0.0, 1.0)
        success, grasp_retained = self._grasp_success_mask(tcp_pos, object_lift, object_pos)
        open_cmd = 1.0 - close_cmd
        grasp_attached = self._grasp_latched if bool(self.cfg.grasp_latch_enabled) else grasp_retained
        stage = self._stage_masks(tcp_pos, target, object_lift, close_cmd, object_pos, grasp_attached)
        xy_dist = stage["xy_dist"]
        z_abs = stage["z_abs"]
        reached = stage["reached"]
        descended = stage["descended"]
        closed = stage["closed"]
        attached = stage["attached"]
        lifted = stage["lifted"]
        close_ready = (xy_dist < 0.040) & (z_abs < 0.035)
        closed_cmd = close_cmd > float(self.cfg.success_close_cmd_threshold)
        grasp_lift_gate = closed_cmd.float() * attached.float()

        xy_progress = torch.clamp(self._prev_xy_dist - xy_dist, min=-0.015, max=0.015)
        z_progress = torch.clamp(self._prev_z_abs - z_abs, min=-0.015, max=0.015)
        lift_delta = torch.clamp(object_lift - self._prev_object_lift, min=-0.01, max=0.02)
        hover_error = torch.abs(rel[:, 2] - float(self.cfg.teacher_hover_height_m))

        xy_reward = (1.0 - torch.tanh(xy_dist / 0.045)) * self.cfg.reward_xy_align_scale
        xy_progress_reward = xy_progress * self.cfg.reward_xy_progress_scale
        hover_height_reward = (1.0 - torch.tanh(hover_error / 0.060)) * (~reached).float() * self.cfg.reward_hover_height_scale
        z_reward = reached.float() * (1.0 - torch.tanh(z_abs / 0.040)) * self.cfg.reward_z_align_scale
        z_progress_reward = reached.float() * z_progress * self.cfg.reward_z_progress_scale
        near_grasp = torch.exp(-torch.square(dist / 0.045))
        near_grasp_reward = near_grasp * self.cfg.reward_near_grasp_scale
        stage_reach_reward = reached.float() * self.cfg.reward_stage_reach_bonus
        stage_descend_reward = descended.float() * self.cfg.reward_stage_descend_bonus
        stage_close_reward = closed.float() * self.cfg.reward_stage_close_bonus
        stage_lift_reward = lifted.float() * self.cfg.reward_stage_lift_bonus
        grasp_latch_reward = attached.float() * self.cfg.reward_grasp_latch_scale
        close_near_reward = close_cmd * close_ready.float() * self.cfg.reward_close_near_scale
        lift_progress_reward = grasp_lift_gate * (lift_progress + lift_delta * 8.0) * self.cfg.reward_lift_progress_scale
        success_reward = success.float() * self.cfg.reward_success
        downward_open_reward = torch.clamp(-self._actions[:, 2], min=0.0) * open_cmd * (xy_dist < 0.045).float() * (rel[:, 2] > 0.025).float() * self.cfg.reward_downward_open_scale
        upward_closed_reward = torch.clamp(self._actions[:, 2], min=0.0) * close_cmd * close_ready.float() * self.cfg.reward_upward_closed_scale
        teacher_action_error = torch.linalg.norm(self._actions - self._last_teacher_action, dim=1)
        teacher_action_reward = torch.exp(-torch.square(teacher_action_error / 0.75)) * self.cfg.reward_teacher_action_scale
        action_penalty = torch.sum(torch.square(self._actions), dim=1) * self.cfg.action_penalty_scale
        premature_close_penalty = close_cmd * (~close_ready).float() * self.cfg.premature_close_penalty_scale
        push_penalty = torch.clamp(pushed - float(self.cfg.object_push_penalty_start_m), min=0.0) * self.cfg.object_push_penalty_scale
        table_penalty = (tcp_pos[:, 2] < TABLE_SURFACE_Z + 0.004).float() * self.cfg.table_penalty
        time_penalty = torch.full_like(action_penalty, float(self.cfg.time_penalty))

        reward = (
            xy_reward
            + xy_progress_reward
            + hover_height_reward
            + z_reward
            + z_progress_reward
            + near_grasp_reward
            + stage_reach_reward
            + stage_descend_reward
            + stage_close_reward
            + stage_lift_reward
            + grasp_latch_reward
            + close_near_reward
            + lift_progress_reward
            + success_reward
            + downward_open_reward
            + upward_closed_reward
            + teacher_action_reward
            - action_penalty
            - premature_close_penalty
            - push_penalty
            - table_penalty
            - time_penalty
        )

        self._episode_sums["xy_align"] += xy_reward + xy_progress_reward + hover_height_reward
        self._episode_sums["z_align"] += z_reward + z_progress_reward
        self._episode_sums["near_grasp"] += near_grasp_reward
        self._episode_sums["stage_reach"] += stage_reach_reward
        self._episode_sums["stage_descend"] += stage_descend_reward
        self._episode_sums["close_near"] += close_near_reward
        self._episode_sums["stage_close"] += stage_close_reward
        self._episode_sums["grasp_latch"] += grasp_latch_reward
        self._episode_sums["stage_lift"] += stage_lift_reward
        self._episode_sums["lift"] += lift_progress_reward
        self._episode_sums["teacher_action"] += teacher_action_reward
        self._episode_sums["success_bonus"] += success_reward
        self._episode_sums["penalty"] += action_penalty + premature_close_penalty + push_penalty + table_penalty + time_penalty
        self._min_tcp_target_dist = torch.minimum(self._min_tcp_target_dist, dist)
        self._max_object_lift = torch.maximum(self._max_object_lift, object_lift)
        self._max_close_cmd = torch.maximum(self._max_close_cmd, close_cmd)
        self._max_grasp_retained = torch.maximum(self._max_grasp_retained, grasp_retained.float())
        self._max_stage_reach = torch.maximum(self._max_stage_reach, reached.float())
        self._max_stage_descend = torch.maximum(self._max_stage_descend, descended.float())
        self._max_stage_close = torch.maximum(self._max_stage_close, closed.float())
        self._max_grasp_latched = torch.maximum(self._max_grasp_latched, attached.float())
        self._max_stage_lift = torch.maximum(self._max_stage_lift, lifted.float())
        self._prev_xy_dist = xy_dist.detach()
        self._prev_z_abs = z_abs.detach()
        self._prev_object_lift = object_lift.detach()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        tcp_pos = self._tcp_pos_w()
        below_table = tcp_pos[:, 2] < TABLE_SURFACE_Z - 0.01
        target = self._object_grasp_target_w()
        object_pos = self._update_grasp_latch(tcp_pos, target, self._gripper_close_cmd)
        object_lift = object_pos[:, 2] - self._object_initial_z
        self._success, _ = self._grasp_success_mask(tcp_pos, object_lift, object_pos)
        terminated = self._success | below_table
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self.extras["log"] = {}
        if hasattr(self, "_episode_sums"):
            for key, value in self._episode_sums.items():
                self.extras["log"][f"Episode_Reward/{key}"] = torch.mean(value[env_ids]) / self.max_episode_length_s
                value[env_ids] = 0.0
            self.extras["log"]["Metrics/success_rate"] = torch.mean(self._success[env_ids].float())
            min_dist = torch.nan_to_num(self._min_tcp_target_dist[env_ids], posinf=float(self.cfg.workspace_radius_m))
            self.extras["log"]["Metrics/min_tcp_target_distance_m"] = torch.mean(min_dist)
            self.extras["log"]["Metrics/max_object_lift_m"] = torch.mean(self._max_object_lift[env_ids])
            self.extras["log"]["Metrics/max_close_cmd"] = torch.mean(self._max_close_cmd[env_ids])
            self.extras["log"]["Metrics/finger_width_m"] = torch.mean(self._finger_width_m()[env_ids])
            self.extras["log"]["Metrics/grasp_retained_rate"] = torch.mean(self._max_grasp_retained[env_ids])
            self.extras["log"]["Metrics/stage_reach_rate"] = torch.mean(self._max_stage_reach[env_ids])
            self.extras["log"]["Metrics/stage_descend_rate"] = torch.mean(self._max_stage_descend[env_ids])
            self.extras["log"]["Metrics/stage_close_rate"] = torch.mean(self._max_stage_close[env_ids])
            self.extras["log"]["Metrics/grasp_latch_rate"] = torch.mean(self._max_grasp_latched[env_ids])
            self.extras["log"]["Metrics/stage_lift_rate"] = torch.mean(self._max_stage_lift[env_ids])
            teacher_fraction = self._teacher_used_steps[env_ids] / torch.clamp(self._teacher_total_steps[env_ids], min=1.0)
            self.extras["log"]["Metrics/teacher_action_fraction"] = torch.mean(teacher_fraction)
            self.extras["log"]["Metrics/teacher_mix_probability"] = self._last_teacher_mix_probability

        self._robot.reset(env_ids)
        self._table.reset(env_ids)
        self._object.reset(env_ids)
        super()._reset_idx(env_ids)

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] = ROBOT_ROOT_Z_ON_WHEELS
        root_state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        if hasattr(self, "_right_joint_ids"):
            right_home = torch.tensor(RIGHT_ARM_OBJECT_HOVER_CONFIG, device=self.device).repeat(len(env_ids), 1)
            joint_noise = torch.empty_like(right_home).uniform_(
                -float(self.cfg.initial_right_arm_joint_noise_rad),
                float(self.cfg.initial_right_arm_joint_noise_rad),
            )
            right_home = right_home + joint_noise
            right_lower = self._joint_lower[self._right_joint_ids].unsqueeze(0)
            right_upper = self._joint_upper[self._right_joint_ids].unsqueeze(0)
            joint_pos[:, self._right_joint_ids] = torch.minimum(torch.maximum(right_home, right_lower), right_upper)
        if hasattr(self, "_finger_joint_ids") and self._finger_joint_ids:
            joint_pos[:, self._finger_joint_ids] = GRIPPER_OPEN_M
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        table_pose = self._table.data.default_root_state[env_ids, 0:7].clone()
        table_pose[:, 0:3] += self.scene.env_origins[env_ids]
        self._table.write_root_pose_to_sim(table_pose, env_ids=env_ids)
        self._table.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)

        object_pose = self._object.data.default_root_state[env_ids, 0:7].clone()
        object_pose[:, 0:3] += self.scene.env_origins[env_ids]
        object_xy = torch.tensor(OBJECT_SPAWN_XY, device=self.device).repeat(len(env_ids), 1)
        object_xy += torch.empty((len(env_ids), 2), device=self.device).uniform_(
            -float(self.cfg.object_spawn_xy_range_m),
            float(self.cfg.object_spawn_xy_range_m),
        )
        object_pose[:, 0:2] = object_xy + self.scene.env_origins[env_ids, 0:2]
        object_pose[:, 2] = TABLE_SURFACE_Z + OBJECT_HALF_Z + self.scene.env_origins[env_ids, 2]
        object_pose[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)
        self._object.write_root_pose_to_sim(object_pose, env_ids=env_ids)
        self._object.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)

        if hasattr(self, "_success"):
            self._success[env_ids] = False
            self._actions[env_ids] = 0.0
            self._last_actions[env_ids] = 0.0
            self._gripper_close_cmd[env_ids] = 0.0
            self._min_tcp_target_dist[env_ids] = float("inf")
            self._max_object_lift[env_ids] = 0.0
            self._max_close_cmd[env_ids] = 0.0
            self._max_grasp_retained[env_ids] = 0.0
            self._max_stage_reach[env_ids] = 0.0
            self._max_stage_descend[env_ids] = 0.0
            self._max_stage_close[env_ids] = 0.0
            self._max_grasp_latched[env_ids] = 0.0
            self._max_stage_lift[env_ids] = 0.0
            self._teacher_used_steps[env_ids] = 0.0
            self._teacher_total_steps[env_ids] = 0.0
            self._teacher_close_steps[env_ids] = 0.0
            self._teacher_lift_latched[env_ids] = False
            self._grasp_latched[env_ids] = False
            self._grasp_latch_offset_w[env_ids] = 0.0
            self._prev_xy_dist[env_ids] = float(self.cfg.workspace_radius_m)
            self._prev_z_abs[env_ids] = float(self.cfg.teacher_hover_height_m)
            self._prev_object_lift[env_ids] = 0.0
            self._object_initial_xy[env_ids] = object_pose[:, 0:2]
            self._object_initial_z[env_ids] = object_pose[:, 2]
            self._target_estimate_offset_w[env_ids, 0:2] = torch.randn((len(env_ids), 2), device=self.device) * float(self.cfg.target_estimate_xy_noise_std_m)
            self._target_estimate_offset_w[env_ids, 2] = torch.randn(len(env_ids), device=self.device) * float(self.cfg.target_estimate_z_noise_std_m)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_target_marker"):
                target_cfg = CUBOID_MARKER_CFG.copy()
                target_cfg.prim_path = "/Visuals/Task319LocalGrasp/object_top"
                target_cfg.markers["cuboid"].size = (0.035, 0.035, 0.035)
                target_cfg.markers["cuboid"].visual_material = _surface((1.0, 0.05, 0.05))
                self._target_marker = VisualizationMarkers(target_cfg)
                frame_cfg = FRAME_MARKER_CFG.copy()
                frame_cfg.prim_path = "/Visuals/Task319LocalGrasp/tcp"
                frame_cfg.markers["frame"].scale = (0.07, 0.07, 0.07)
                self._tcp_marker = VisualizationMarkers(frame_cfg)
            self._target_marker.set_visibility(True)
            self._tcp_marker.set_visibility(True)
        else:
            if hasattr(self, "_target_marker"):
                self._target_marker.set_visibility(False)
                self._tcp_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        self._target_marker.visualize(self._object_grasp_target_w())
        self._tcp_marker.visualize(self._tcp_pos_w(), self._robot.data.body_quat_w[:, self._ee_body_id])
