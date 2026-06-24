"""DualArmVLAPipeline — full dual-arm VLA pipeline for Task 2.2."""

from pathlib import Path
import re
from typing import Optional, List, Tuple, Dict

import numpy as np
import mujoco

from control import MuJoCoEnv
from robot_common.infra.config import TaskConfig, load_config
from robot_common.infra.logging import logger
from robot_common.infra.metrics import MetricsTracker, EpisodeMetrics
from robot_common.infra.motion_trace import MotionTrace
from robot_common.infra.debug_artifacts import EpisodeDebugWriter
from robot_common.infra.task_lifecycle import (
    create_episode_debug_writer,
    load_raw_yaml,
    prewarm_perception_models,
    save_motion_trace,
)
from perception import PerceptionHub, DetectionResult
from robot_common.decision import (
    TaskRouter, LLMTaskParser, TaskPlan, SubTask,
    TaskPhase, ArmRole,
)
from robot_common.decision.state_machine import TaskStateMachine, TaskState
from planning import (
    MinkIKSolver, TrajectoryGenerator, CollisionMonitor, IKSolution,
)
from control import DUAL_PANDA_GRIPPERS, GraspManager, RobosuiteWBCController, DualArmWBCController
from control.wbc import WBC_SOURCE
from planning.backends import (
    MINK_BACKEND,
    MissingBackendError,
    check_backend,
)
from control.backends import ROBOSUITE_WBC_BACKEND

from .state import DualArmState
from .coordinator import DualArmCoordinator
from .dag_planner import DAGTaskPlanner
class DualArmVLAPipeline:
    """Full dual-arm VLA pipeline for Task 2.2."""

    def __init__(self, config_path: str = "configs/task_22_dual_arm.yaml"):
        self.config = load_config(config_path)
        self._raw_config = self._load_raw_yaml(config_path)
        self.env: Optional[MuJoCoEnv] = None
        self.coordinator: Optional[DualArmCoordinator] = None
        self.planner: Optional[DAGTaskPlanner] = None
        self.perception: Optional[PerceptionHub] = None
        self.wbc_controller: Optional[DualArmWBCController] = None
        self.metrics = MetricsTracker("dual_arm_vla_22")
        self.motion_trace: Optional[MotionTrace] = None
        self._setup()

    def _load_raw_yaml(self, config_path: str) -> dict:
        return load_raw_yaml(config_path)

    def _setup(self):
        """Initialize simulation and components."""
        xml_path = Path(self.config.robot.mjcf_path)
        if not xml_path.exists():
            alt_paths = [
                Path("simulation/menagerie/franka_fr3/scene.xml"),
            ]
            for alt in alt_paths:
                if alt.exists():
                    xml_path = alt
                    break
            else:
                # We'll use a placeholder - the scene needs to be created
                logger.warning(f"No dual-arm scene found: {xml_path}")
                xml_path = Path("simulation/menagerie/franka_fr3/scene.xml")
                if not xml_path.exists():
                    raise FileNotFoundError("No dual-arm Panda scene found")

        logger.info(f"Loading dual-arm scene: {xml_path}")
        self.env = MuJoCoEnv(
            str(xml_path),
            camera_name="scene_camera",
            width=640, height=480,
            control_freq=50.0,
        )

        # Find arm joints
        left_joints = self._find_arm_joints("left")
        right_joints = self._find_arm_joints("right")
        if not right_joints:
            raise RuntimeError("Dual-arm scene must expose right-arm joints")

        logger.info(f"Left arm: {len(left_joints)} joints, Right arm: {len(right_joints)} joints")

        self.coordinator = DualArmCoordinator(
            self.env, left_joints, right_joints,
            arm_base_distance=self._raw_config.get("dual_arm_params", {}).get("arm_base_distance", 0.8),
            max_sync_error=self._raw_config.get("dual_arm_params", {}).get("max_sync_error", 0.02),
            max_tilt_angle=self._raw_config.get("dual_arm_params", {}).get("max_tilt_angle", 5.0),
        )
        self.grasp = GraspManager(self.env.model, self.env.data)
        left_gripper = DUAL_PANDA_GRIPPERS["left"]
        right_gripper = DUAL_PANDA_GRIPPERS["right"]
        self.grasp.register_weld(left_gripper.weld_name,
            gripper_body=left_gripper.primary_gripper_body, default_obj_body="long_rod",
            extra_gripper_bodies=list(left_gripper.extra_gripper_bodies))
        self.grasp.register_weld(right_gripper.weld_name,
            gripper_body=right_gripper.primary_gripper_body, default_obj_body="long_rod",
            extra_gripper_bodies=list(right_gripper.extra_gripper_bodies))
        self.motion_trace = MotionTrace.from_task_config(
            "dual_arm_vla_22",
            self._raw_config,
            tracked_bodies=["long_rod", "box_obj"],
            tilt_bodies=[],
            controlled_joint_prefixes=["l_", "r_"],
        )
        self.env.set_motion_trace(self.motion_trace)
        self.planner = DAGTaskPlanner(self.coordinator)
        self.perception = PerceptionHub(self.config.perception)
        check_backend(MINK_BACKEND, required=True)
        wbc_cfg = self._raw_config.get("dual_arm_params", {}).get("wbc", {})
        self.wbc_controller = DualArmWBCController(
            self.env,
            left_joints,
            right_joints,
            left_body=left_gripper.pinch_body,
            right_body=right_gripper.pinch_body,
            kp_pos=float(wbc_cfg.get("kp_pos", 120.0)),
            kd_pos=float(wbc_cfg.get("kd_pos", 24.0)),
            kp_rel=float(wbc_cfg.get("kp_rel", 260.0)),
            kd_rel=float(wbc_cfg.get("kd_rel", 38.0)),
            kp_ori=float(wbc_cfg.get("kp_ori", 60.0)),
            kd_ori=float(wbc_cfg.get("kd_ori", 14.0)),
            damping=float(wbc_cfg.get("damping", 2e-4)),
            joint_damping=float(wbc_cfg.get("joint_damping", 2.0)),
            posture_gain=float(wbc_cfg.get("posture_gain", 1.5)),
            torque_limit=float(wbc_cfg.get("torque_limit", 80.0)),
            max_torque_delta=float(wbc_cfg.get("max_torque_delta", 3.0)),
        )
        if not self.wbc_controller.validate():
            raise RuntimeError("Dual-arm WBC requires 14 torque motor arm actuators")
        self.coordinator.set_mature_backends(wbc_controller=self.wbc_controller)
        self._apply_dual_arm_servo_tuning()
        logger.info("DualArmVLAPipeline initialized")

    def _apply_dual_arm_servo_tuning(self):
        """Tune passive damping; arm motors are torque-controlled by WBC."""
        kp = float(self._raw_config.get("dual_arm_params", {}).get("joint_kp", 80.0))
        damping = float(self._raw_config.get("dual_arm_params", {}).get("joint_damping", 4.0))
        for a_id in range(self.env.model.nu):
            jnt_id = int(self.env.model.actuator_trnid[a_id, 0])
            if jnt_id < 0:
                continue
            j_name = self.env.model.joint(jnt_id).name or ""
            bias_type = int(self.env.model.actuator_biastype[a_id])
            if j_name.startswith(("l_grip", "r_grip")) and bias_type != int(mujoco.mjtBias.mjBIAS_NONE):
                self.env.model.actuator_gainprm[a_id, 0] = kp
                self.env.model.actuator_biasprm[a_id, 1] = -kp
        for j_id in range(self.env.model.njnt):
            j_name = self.env.model.joint(j_id).name or ""
            if j_name.startswith(("l_", "r_")):
                dadr = int(self.env.model.jnt_dofadr[j_id])
                if dadr >= 0:
                    self.env.model.dof_damping[dadr] = max(self.env.model.dof_damping[dadr], damping)
        self.coordinator.max_ctrl_delta = float(
            self._raw_config.get("dual_arm_params", {}).get("max_ctrl_delta", 0.08)
        )

    def _reset_to_joint_home(self):
        """Reset arm joints to a high, collision-free ready pose."""
        positions = {}
        current_q = self.env.data.qpos.copy()
        ready_left = np.asarray(
            self._raw_config.get("dual_arm_params", {}).get("ready_left", [-0.12, 0.32, 0.95]),
            dtype=float,
        )
        ready_right = np.asarray(
            self._raw_config.get("dual_arm_params", {}).get("ready_right", [-0.12, -0.32, 0.95]),
            dtype=float,
        )
        sol = self.coordinator.ik.solve_dual_ik(
            targets=[
                {"pos": ready_left, "body": self.coordinator._ee_body("left")},
                {"pos": ready_right, "body": self.coordinator._ee_body("right")},
            ],
            current_q=current_q,
            orientation_cost=0.0,
            posture_cost=1e-4,
            max_steps=800,
            tolerance=0.002,
        )
        source_q = sol.joint_positions if sol.success and sol.joint_positions.size else current_q
        for j_name in self.coordinator.left_joints + self.coordinator.right_joints:
            try:
                jid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
                qadr = self.env.model.jnt_qposadr[jid]
                positions[j_name] = float(source_q[qadr])
            except Exception:
                pass
        self.env.set_joint_positions(positions)
        actuator_map = self.coordinator._get_actuator_map()
        for j_name, value in positions.items():
            if j_name in actuator_map:
                act_id = actuator_map[j_name]
                if int(self.env.model.actuator_biastype[act_id]) == int(mujoco.mjtBias.mjBIAS_NONE):
                    self.env.data.ctrl[act_id] = 0.0
                else:
                    self.env.data.ctrl[act_id] = value
        mujoco.mj_forward(self.env.model, self.env.data)
        if self.wbc_controller is not None:
            self.wbc_controller.reset()

    def _command_dual_grippers(self, value: float, steps: int = 40):
        """Command the simple two-finger grippers while preserving WBC arm torque."""
        for act_name in (spec.actuator_name for spec in DUAL_PANDA_GRIPPERS.values()):
            act_id = self.env.actuator_index(act_name)
            if act_id is not None:
                lo, hi = self.env.model.actuator_ctrlrange[act_id]
                self.env.data.ctrl[act_id] = float(np.clip(value, lo, hi))
        for _ in range(max(0, int(steps))):
            self.env.step()

    def _dual_finger_distances(self) -> dict:
        return {
            "left": float(np.linalg.norm(
                self.env.get_body_position(DUAL_PANDA_GRIPPERS["left"].finger_bodies[0])
                - self.env.get_body_position(DUAL_PANDA_GRIPPERS["left"].finger_bodies[1])
            )),
            "right": float(np.linalg.norm(
                self.env.get_body_position(DUAL_PANDA_GRIPPERS["right"].finger_bodies[0])
                - self.env.get_body_position(DUAL_PANDA_GRIPPERS["right"].finger_bodies[1])
            )),
        }

    def prewarm(self):
        """Load lazy perception models before opening the live viewer."""
        prewarm_perception_models(self.perception)

    def validate_mature_backends(self):
        """Fail before viewer launch if required external planners are missing."""
        check_backend(MINK_BACKEND, required=True)
        if self.wbc_controller is None or not self.wbc_controller.validate():
            raise MissingBackendError("Task 2.2 requires torque-level DualArmWBCController")

    def _visible_ready_pose(self):
        """Short visible motion at episode start before perception/plan logs."""
        for _ in range(20):
            state = self.coordinator.get_state()
            self.wbc_controller.step(
                left_target_world=state.left_ee_pos,
                right_target_world=state.right_ee_pos,
                relative_target_world=state.left_ee_pos - state.right_ee_pos,
                dt=self.env.dt,
            )
            self.env.step()

    def _find_arm_joints(self, prefix: str) -> List[str]:
        """Find all joint names with given prefix or pattern."""
        joints = []
        # Direct pattern: l_j1..l_j7 or r_j1..r_j7 (our dual_panda_scene)
        side_map = {"left": "l", "right": "r"}
        side_letter = side_map.get(prefix.lower(), prefix[0].lower())
        for i in range(1, 9):  # up to 8 joints
            name = f"{side_letter}_j{i}"
            try:
                self.env.model.joint(name)
                joints.append(name)
            except Exception:
                pass
        if joints:
            return joints
        # Generic fallback: search by prefix
        for i in range(self.env.model.njnt):
            name = self.env.model.joint(i).name
            if name and prefix.lower() in name.lower():
                joints.append(name)
        # Last resort for Franka FR3: fr3_joint*
        if not joints and prefix.lower() == "right":
            for i in range(self.env.model.njnt):
                name = self.env.model.joint(i).name
                if name and "fr3_joint" in name and int(name.split("fr3_joint")[1]) >= 1:
                    joints.append(name)
        return joints

    def run_single_episode(self, instruction: str = "") -> EpisodeMetrics:
        """Run one full dual-arm collaborative transport episode.

        Pipeline:
          S1: LLM instruction parsing + DAG plan generation
          S2: Scene perception (detection + 3D localization)
          S3: Arm role assignment (master/slave/sync)
          S4: Dual-arm trajectory generation (synchronized)
          S5: Execute with collision monitoring
          S6: Closed-loop VLA fine-tuning (optional)
        """
        self.env.reset()
        self._reset_to_joint_home()
        instruction = instruction or "用双手把长杆放到指定区域"

        ep = self.metrics.new_episode(
            instruction=instruction,
            task_type="dual_arm_vla",
        )
        debug = create_episode_debug_writer("22", ep.episode_id, self._raw_config)
        ep.custom["debug_dir"] = str(debug.root)
        debug.add("instruction", instruction)

        import time
        t_start = time.time()
        self._visible_ready_pose()

        # S1: Parse instruction and generate DAG plan
        logger.info(f"Instruction: {instruction}")
        objects = ["long_rod", "box_obj", "target_region"]  # From scene config
        plan = self.planner.plan(instruction, objects)
        logger.info(f"Plan: {plan.arm_role.name}, {len(plan.sub_tasks)} sub-tasks")
        debug.add("plan", {
            "arm_role": plan.arm_role.name,
            "subtasks": [
                {
                    "id": st.id,
                    "phase": st.phase.value,
                    "arm": st.arm,
                    "target_object": st.target_object,
                    "depends_on": st.depends_on,
                }
                for st in plan.sub_tasks
            ],
        })
        for st in plan.sub_tasks:
            deps = f" (depends: {st.depends_on})" if st.depends_on else ""
            logger.info(f"  [{st.arm}] {st.id}: {st.phase.value} → {st.target_object}{deps}")

        # S2: Scene perception
        rgb = self.env.render("scene_camera")
        depth = self.env.render_depth("scene_camera")
        detections = self.perception.detect(rgb, objects, depth)
        ep.object_detected = len(detections) > 0
        debug.save_image("scene_detection_overlay", rgb, detections)
        debug.add("scene_camera", debug.camera_snapshot(self.env, "scene_camera"))

        # Build object position map
        obj_positions = {}
        visual_positions = {}
        detection_sources = {}
        for det in detections:
            if det.position_3d is not None:
                obj_name = self._canonical_object_name(det.class_name)
                if obj_name:
                    visual_world = self.env.camera_point_to_world(
                        det.position_3d, "scene_camera"
                    )
                    candidate = {
                        "label": det.class_name,
                        "confidence": float(det.confidence),
                        "bbox": list(det.bbox or []),
                        "point_camera": np.asarray(det.position_3d, dtype=float),
                        "point_world_visual": visual_world,
                    }
                    if obj_name in self.env._body_names:
                        candidate["body_error_m"] = float(
                            np.linalg.norm(visual_world - self.env.get_body_position(obj_name))
                        )
                    prev = detection_sources.get(obj_name)
                    prev_key = (
                        float(prev.get("body_error_m", 1e9)) if prev else 1e9,
                        -float(prev.get("confidence", 0.0)) if prev else 0.0,
                    )
                    new_key = (
                        float(candidate.get("body_error_m", 1e9)),
                        -float(candidate["confidence"]),
                    )
                    if prev is None or new_key < prev_key:
                        visual_positions[obj_name] = visual_world
                        detection_sources[obj_name] = candidate

        carried_object = objects[0] if objects else ""
        if carried_object and carried_object not in visual_positions:
            logger.error(
                f"Detector did not identify required object '{carried_object}'. "
                "Refusing scene-truth recognition fallback."
            )
            ep.custom["failure"] = "required_object_not_detected"
            ep.success = False
            ep.total_time = time.time() - t_start
            debug.add("detections", self._debug_detections(detections, visual_positions))
            debug.add("object_positions", detection_sources)
            debug.add("final_metrics", ep.to_dict())
            debug.write()
            self._finalize_motion_trace(ep)
            return ep

        for obj_name, visual_pos in visual_positions.items():
            if obj_name in self.env._body_names:
                body_pos = self.env.get_body_position(obj_name)
                obj_positions[obj_name] = body_pos
                detection_sources[obj_name]["point_world_control"] = body_pos
                detection_sources[obj_name]["visual_control_error_m"] = float(np.linalg.norm(visual_pos - body_pos))
            else:
                obj_positions[obj_name] = visual_pos
        obj_positions.setdefault("target_region", self.env.get_body_position("target_region"))
        detection_sources["target_region"] = {
            "source": "scene_goal_marker",
            "point_world_control": obj_positions["target_region"],
        }
        debug.add("detections", self._debug_detections(detections, visual_positions))
        debug.add("object_positions", detection_sources)

        # S3: Set arm roles
        master_arm = self._raw_config.get("dual_arm_params", {}).get("master_arm", "left")
        slave_arm = "right" if master_arm == "left" else "left"

        # S4-S5: Execute subtasks in DAG order
        state = TaskStateMachine(plan)
        state.transition(TaskState.INITIALIZING)

        substep_counter = 0
        reference_quat = (
            self.env.get_body_quat(carried_object)
            if carried_object in self.env._body_names else np.array([1.0, 0.0, 0.0, 0.0])
        )
        object_attached = False
        carry_reached_target = False
        execution_failed = False
        ep.custom["execution_backend"] = WBC_SOURCE
        ep.custom["motion_source"] = WBC_SOURCE
        ep.custom["wbc_controller_ready"] = bool(
            self.wbc_controller and self.wbc_controller.validate()
        )

        if any(st.phase == TaskPhase.CARRY and st.arm == "both" for st in plan.sub_tasks):
            logger.info("  Executing dual-arm synchronized carry primitive")
            object_attached, carry_reached_target, substep_counter = self._execute_dual_carry_primitive(
                carried_object=carried_object,
                target_region=obj_positions["target_region"],
                reference_quat=reference_quat,
                ep=ep,
                debug=debug,
            )
            execution_failed = not (object_attached and carry_reached_target)
            for st in plan.sub_tasks:
                state.mark_subtask_done(st.id)
            ep.custom["failure"] = "" if not execution_failed else ep.custom.get("failure", "dual_carry_failed")
        else:
            logger.warning("No synchronized carry subtask found; using generic DAG execution")

        if any(st.phase == TaskPhase.CARRY and st.arm == "both" for st in plan.sub_tasks):
            self.grasp.release_all()
            ep.num_steps = substep_counter
            final_pos = self.env.get_body_position(carried_object) if carried_object in self.env._body_names else np.zeros(3)
            ep.position_error = float(np.linalg.norm(final_pos[:2] - obj_positions["target_region"][:2]))
            max_pos_error = float(
                self._raw_config.get("dual_arm_params", {}).get("max_position_error", 0.12)
            )
            max_sync_error = float(
                self._raw_config.get("dual_arm_params", {}).get("max_sync_error", 0.06)
            )
            max_tilt = np.deg2rad(float(
                self._raw_config.get("dual_arm_params", {}).get("max_tilt_angle", 5.0)
            ))
            trace_summary = self.motion_trace.summary() if self.motion_trace else {}
            trace_smooth = bool(trace_summary.get("smooth", True))
            quality_pass = (
                ep.position_error <= max_pos_error
                and ep.sync_error <= max_sync_error
                and ep.orientation_error <= max_tilt
                and trace_smooth
                and ep.custom.get("motion_source") == WBC_SOURCE
            )
            ep.custom["quality_thresholds"] = {
                "max_position_error_m": max_pos_error,
                "max_sync_error_m": max_sync_error,
                "max_tilt_rad": float(max_tilt),
            }
            ep.custom["control_quality"] = {
                "motion_source": ep.custom.get("motion_source"),
                "wbc_controller_ready": bool(ep.custom.get("wbc_controller_ready")),
                "position_error_m": float(ep.position_error),
                "sync_error_m": float(ep.sync_error),
                "tilt_rad": float(ep.orientation_error),
                "tilt_deg": float(np.rad2deg(ep.orientation_error)),
                "trace_smooth": bool(trace_smooth),
            }
            ep.custom["quality_pass"] = bool(quality_pass)
            ep.custom["trace_smooth_required"] = trace_smooth
            if object_attached and carry_reached_target and not quality_pass:
                ep.custom["failure"] = "dual_carry_quality_failed"
            ep.success = (
                (not execution_failed)
                and object_attached
                and carry_reached_target
                and quality_pass
            )
            ep.grasp_success = bool(object_attached)
            ep.grasp_attempts = 1
            ep.total_time = time.time() - t_start
            self._finalize_motion_trace(ep)
            debug.add("final_metrics", ep.to_dict())
            debug.write()
            logger.info(f"  Done: {ep.success=}, sync_error={ep.sync_error:.4f}m, "
                        f"pos_error={ep.position_error:.4f}m")
            return ep

        for subtask in plan.sub_tasks:
            # Check dependencies
            if subtask.depends_on:
                deps_met = all(d in state.completed_subtasks for d in subtask.depends_on)
                if not deps_met:
                    logger.warning(f"  Skipping {subtask.id}: dependencies not met")
                    execution_failed = True
                    break

            logger.info(f"  Executing: {subtask.id} ({subtask.phase.value})")

            if subtask.arm == "both":
                # Synchronized dual-arm movement
                target_obj = subtask.target_object
                target_pos = obj_positions.get(target_obj, np.array([0.5, 0.0, 0.8]))

                if subtask.phase == TaskPhase.GRASP:
                    target_obj = carried_object
                    target_pos = obj_positions.get(target_obj, np.array([0.5, 0.0, 0.8]))
                elif subtask.phase == TaskPhase.PLACE:
                    target_obj = "target_region"
                    target_pos = obj_positions[target_obj]
                elif subtask.phase == TaskPhase.LIFT:
                    target_obj = carried_object
                    target_pos = obj_positions.get(target_obj, np.array([0.5, 0.0, 0.8])) + np.array([0.0, 0.0, 0.18])

                # Compute side grasps for the y-oriented rod. The grippers are
                # mounted on opposite sides of the table, so the reachable
                # grasp points are the two lateral ends of the object.
                grasp_sep = float(self._raw_config.get("dual_arm_params", {}).get("rod_grasp_separation", 0.34))
                left_offset = np.array([0.0, grasp_sep / 2.0, 0.0])
                right_offset = np.array([0.0, -grasp_sep / 2.0, 0.0])
                self.coordinator.nominal_gripper_separation = float(np.linalg.norm(left_offset - right_offset))

                left_target = np.array([
                    target_pos[0] + left_offset[0],
                    target_pos[1] + left_offset[1],
                    target_pos[2] + left_offset[2],
                    0, 0, 0, 0,
                ])
                right_target = np.array([
                    target_pos[0] + right_offset[0],
                    target_pos[1] + right_offset[1],
                    target_pos[2] + right_offset[2],
                    0, 0, 0, 0,
                ])

                # Plan synchronized trajectory
                left_traj, right_traj = self.coordinator.plan_synchronized_trajectory(
                    left_target, right_target, num_steps=50
                )
                plan_debug = dict(getattr(self.coordinator, "last_plan_debug", {}) or {})
                plan_debug.update({
                    "subtask": subtask.id,
                    "phase": subtask.phase.value,
                    "target_object": target_obj,
                    "target_center_world": target_pos[:3].copy(),
                    "left_offset": left_offset.copy(),
                    "right_offset": right_offset.copy(),
                })
                debug.append("trajectory_plans", plan_debug)

                left_runtime_attached = False
                right_runtime_attached = False

                # Execute with collision monitoring. During grasp, attach on
                # first real contact so the rod is not swept away by the pads.
                for t in range(len(left_traj)):
                    self.coordinator._set_joints("left", left_traj[t])
                    self.coordinator._set_joints("right", right_traj[t])
                    self.env.step()

                    if subtask.phase == TaskPhase.GRASP and target_obj in self.env._body_names:
                        if (not left_runtime_attached
                                and self._verified_dual_finger_contact("grasp_left", target_obj)):
                            left_runtime_attached = self.grasp.attach("grasp_left", target_obj)
                            logger.info(f"Left grasp attached on contact at step {t}: {target_obj}")
                        if (not right_runtime_attached
                                and self._verified_dual_finger_contact("grasp_right", target_obj)):
                            right_runtime_attached = self.grasp.attach("grasp_right", target_obj)
                            logger.info(f"Right grasp attached on contact at step {t}: {target_obj}")

                    # Monitor sync error
                    state_data = self.coordinator.get_state()
                    ep.sync_error = max(ep.sync_error, state_data.sync_error)
                    if carried_object in self.env._body_names:
                        tilt = self.coordinator.object_tilt(carried_object, reference_quat)
                        ep.orientation_error = max(ep.orientation_error, tilt)

                    substep_counter += 1

                if subtask.phase == TaskPhase.GRASP and target_obj in self.env._body_names:
                    for _ in range(20):
                        self.coordinator._set_joints("left", left_traj[-1])
                        self.coordinator._set_joints("right", right_traj[-1])
                        self.env.step()
                    state_data = self.coordinator.get_state()
                    obj_pos = self.env.get_body_position(target_obj)
                    left_grasp_pos = obj_pos + left_offset
                    right_grasp_pos = obj_pos + right_offset
                    left_dist = float(np.linalg.norm(state_data.left_ee_pos - left_grasp_pos))
                    right_dist = float(np.linalg.norm(state_data.right_ee_pos - right_grasp_pos))
                    max_attach_distance = float(
                        self._raw_config.get("dual_arm_params", {}).get("max_attach_distance", 0.09)
                    )
                    if max(left_dist, right_dist) > max_attach_distance:
                        logger.error(
                            f"Dual grasp approach too far for {target_obj}: "
                            f"left={left_dist:.3f}m right={right_dist:.3f}m "
                            f"limit={max_attach_distance:.3f}m"
                        )
                        ep.custom["failure"] = "dual_grasp_distance_too_far"
                        debug.append("grasp_checks", {
                            "target": target_obj,
                            "left_dist_m": left_dist,
                            "right_dist_m": right_dist,
                            "limit_m": max_attach_distance,
                            "left_ee_world": state_data.left_ee_pos.copy(),
                            "right_ee_world": state_data.right_ee_pos.copy(),
                            "left_grasp_world": left_grasp_pos.copy(),
                            "right_grasp_world": right_grasp_pos.copy(),
                        })
                        execution_failed = True
                        break
                    left_ok = self._verified_dual_finger_contact("grasp_left", target_obj)
                    right_ok = self._verified_dual_finger_contact("grasp_right", target_obj)
                    left_ok = left_ok or self.grasp.is_attached("grasp_left")
                    right_ok = right_ok or self.grasp.is_attached("grasp_right")
                    if not (left_ok and right_ok):
                        logger.error(
                            f"Dual grasp contact incomplete for {target_obj}: "
                            f"left_contact={left_ok} right_contact={right_ok} "
                            f"left_dist={left_dist:.3f}m right_dist={right_dist:.3f}m"
                        )
                        ep.custom["failure"] = "dual_grasp_contact_incomplete"
                        debug.append("grasp_checks", {
                            "target": target_obj,
                            "left_contact": bool(left_ok),
                            "right_contact": bool(right_ok),
                            "left_dist_m": left_dist,
                            "right_dist_m": right_dist,
                        })
                        execution_failed = True
                        break
                    if not self.grasp.is_attached("grasp_left"):
                        left_runtime_attached = self.grasp.attach("grasp_left", target_obj)
                    if not self.grasp.is_attached("grasp_right"):
                        right_runtime_attached = self.grasp.attach("grasp_right", target_obj)
                    object_attached = (
                        self.grasp.is_attached("grasp_left")
                        and self.grasp.is_attached("grasp_right")
                    )
                    for _ in range(20):
                        self.coordinator._set_joints("left", left_traj[-1])
                        self.coordinator._set_joints("right", right_traj[-1])
                        self.env.step()
                elif subtask.phase == TaskPhase.PLACE and carried_object in self.env._body_names:
                    obj_pos = self.env.get_body_position(carried_object)
                    carry_reached_target = (
                        float(np.linalg.norm(obj_pos[:2] - obj_positions["target_region"][:2])) < 0.25
                    )
                    for _ in range(20):
                        self.coordinator._set_joints("left", left_traj[-1])
                        self.coordinator._set_joints("right", right_traj[-1])
                        self.env.step()

            elif subtask.arm in ("left", "right"):
                # Single-arm movement
                target_obj = subtask.target_object
                target_pos = obj_positions.get(target_obj, np.array([0.5, 0.0, 0.8]))
                side = subtask.arm

                if subtask.phase == TaskPhase.APPROACH and target_obj == carried_object:
                    ep.custom.setdefault("deferred_approach", []).append(subtask.id)
                    for _ in range(10):
                        self.coordinator._set_joints("left", self.coordinator._get_joint_positions("left"))
                        self.coordinator._set_joints("right", self.coordinator._get_joint_positions("right"))
                        self.env.step()
                        substep_counter += 1
                    state.mark_subtask_done(subtask.id)
                    continue

                target = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.1])
                ee_body = self.coordinator._ee_body(side)
                sol = self.coordinator.ik.solve_ik(
                    target_pos=target, ee_body_name=ee_body,
                )

                if sol.success:
                    traj = self._plan_single_arm_trajectory(side, sol.joint_positions, num_steps=60)
                    for joints in traj:
                        self.coordinator._set_joints(side, joints)
                        self.env.step()
                        substep_counter += 1

            state.mark_subtask_done(subtask.id)
            if execution_failed:
                break

        self.grasp.release_all()

        ep.num_steps = substep_counter

        # Check final state
        state_data = self.coordinator.get_state()
        ep.sync_error = state_data.sync_error
        trace_summary = self.motion_trace.summary() if self.motion_trace else {}
        trace_smooth = bool(trace_summary.get("smooth", True))
        max_sync_error = float(
            self._raw_config.get("dual_arm_params", {}).get("max_sync_error", 0.02)
        )
        max_tilt = np.deg2rad(float(
            self._raw_config.get("dual_arm_params", {}).get("max_tilt_angle", 5.0)
        ))
        control_quality_ok = (
            trace_smooth
            and ep.sync_error <= max_sync_error
            and ep.orientation_error <= max_tilt
            and ep.custom.get("motion_source") == WBC_SOURCE
        )
        ep.custom["control_quality"] = {
            "motion_source": ep.custom.get("motion_source"),
            "wbc_controller_ready": bool(ep.custom.get("wbc_controller_ready")),
            "sync_error_m": float(ep.sync_error),
            "tilt_rad": float(ep.orientation_error),
            "tilt_deg": float(np.rad2deg(ep.orientation_error)),
            "trace_smooth": bool(trace_smooth),
        }
        ep.custom["quality_thresholds"] = {
            "max_sync_error_m": max_sync_error,
            "max_tilt_rad": float(max_tilt),
        }
        ep.custom["quality_pass"] = bool(control_quality_ok)
        ep.custom["trace_smooth_required"] = trace_smooth

        # Verify object position
        if objects:
            try:
                target_obj = objects[0]
                final_pos = self.env.get_body_position(target_obj) if target_obj in self.env._body_names else np.zeros(3)
                target_region = obj_positions.get("target_region", np.array([0.8, 0.0, 0.8]))
                ep.position_error = float(np.linalg.norm(final_pos - target_region))
                ep.success = (
                    (not execution_failed)
                    and object_attached
                    and carry_reached_target
                    and control_quality_ok
                )
            except Exception:
                ep.success = (
                    (not execution_failed)
                    and bool(object_attached)
                    and control_quality_ok
                )

        ep.total_time = time.time() - t_start
        self._finalize_motion_trace(ep)
        debug.add("final_metrics", ep.to_dict())
        debug.write()
        logger.info(f"  Done: {ep.success=}, sync_error={ep.sync_error:.4f}m, "
                    f"pos_error={ep.position_error:.4f}m")
        return ep

    def _canonical_object_name(self, label: str) -> str:
        """Map perception labels back to MJCF body names."""
        raw = (label or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", " ", raw).strip()
        compact = text.replace(" ", "")
        aliases = {
            "long rod": "long_rod",
            "rod": "long_rod",
            "box object": "box_obj",
            "box": "box_obj",
            "target region": "target_region",
            "target": "target_region",
        }
        for key, value in aliases.items():
            if key in text or key.replace(" ", "") in compact:
                return value
        return label if label in self.env._body_names else ""

    def _debug_detections(self, detections, visual_positions: dict) -> list[dict]:
        rows = []
        for det in detections:
            body = self._canonical_object_name(det.class_name)
            row = {
                "label": det.class_name,
                "canonical_body": body,
                "confidence": float(det.confidence),
                "bbox": list(det.bbox or []),
            }
            if det.position_3d is not None:
                row["point_camera"] = np.asarray(det.position_3d, dtype=float).round(5).tolist()
            if body in visual_positions:
                row["point_world_visual"] = visual_positions[body].round(5).tolist()
                if body in self.env._body_names:
                    body_pos = self.env.get_body_position(body)
                    row["body_world"] = body_pos.round(5).tolist()
                    row["visual_body_error_m"] = float(np.linalg.norm(visual_positions[body] - body_pos))
            rows.append(row)
        return rows

    def _dual_targets_from_center(self, center: np.ndarray, separation: float) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(center[:3], dtype=float)
        left = center + np.array([0.0, separation / 2.0, 0.0])
        right = center + np.array([0.0, -separation / 2.0, 0.0])
        return left, right

    def _execute_dual_waypoint(self, center: np.ndarray, separation: float,
                               steps: int, hold: int, phase: str,
                               target_body: str, ep: EpisodeMetrics,
                               debug: EpisodeDebugWriter,
                               attach_on_near: bool = False,
                               offsets: Optional[tuple[np.ndarray, np.ndarray]] = None,
                               sync_reference: Optional[np.ndarray] = None) -> tuple[bool, bool, int]:
        """Track one synchronized dual-arm waypoint through torque WBC."""
        if self.wbc_controller is None or not self.wbc_controller.validate():
            raise RuntimeError("2.2 requires DualArmWBCController; no joint-trajectory fallback is allowed")
        if offsets is None:
            left_pos, right_pos = self._dual_targets_from_center(center, separation)
        else:
            left_pos = np.asarray(center[:3], dtype=float) + np.asarray(offsets[0], dtype=float)
            right_pos = np.asarray(center[:3], dtype=float) + np.asarray(offsets[1], dtype=float)

        state0 = self.coordinator.get_state()
        left_start = state0.left_ee_pos.copy()
        right_start = state0.right_ee_pos.copy()
        left_goal = np.asarray(left_pos[:3], dtype=float)
        right_goal = np.asarray(right_pos[:3], dtype=float)
        rel_goal = (
            np.asarray(sync_reference[:3], dtype=float)
            if sync_reference is not None else left_goal - right_goal
        )
        total = int(max(2, steps) + max(0, hold))
        debug.append("trajectory_plans", {
            "source": WBC_SOURCE,
            "phase": phase,
            "target_body": target_body,
            "center_world": np.asarray(center[:3], dtype=float),
            "separation_m": float(separation),
            "steps": int(total),
            "left_start_world": left_start.copy(),
            "right_start_world": right_start.copy(),
            "left_target_world": left_goal.copy(),
            "right_target_world": right_goal.copy(),
            "relative_target_world": rel_goal.copy(),
            "control_law": "MuJoCo mass-matrix/Jacobian torque WBC",
        })

        left_attached = self.grasp.is_attached("grasp_left")
        right_attached = self.grasp.is_attached("grasp_right")
        count = 0
        max_left_error = 0.0
        max_right_error = 0.0
        max_torque_norm = 0.0
        saturated_steps = 0
        wbc_cfg = self._raw_config.get("dual_arm_params", {}).get("wbc", {})
        object_feedback_gain = float(wbc_cfg.get("object_feedback_gain", 0.85))
        max_object_correction = float(wbc_cfg.get("max_object_correction", 0.10))
        grip_open = float(self._raw_config.get("dual_arm_params", {}).get("grip_open", 0.015))
        grip_closed = float(self._raw_config.get("dual_arm_params", {}).get("grip_closed", 0.0))
        if phase == "approach":
            self._command_dual_grippers(grip_open, steps=10)
        for i in range(total):
            raw = min(1.0, float(i) / float(max(1, steps - 1)))
            alpha = raw * raw * (3.0 - 2.0 * raw)
            if phase == "dual_grasp" and i == max(1, steps - 1):
                self._command_dual_grippers(grip_closed, steps=35)
            desired_left = left_start + alpha * (left_goal - left_start)
            desired_right = right_start + alpha * (right_goal - right_start)
            if (
                offsets is not None
                and target_body in self.env._body_names
                and self.grasp.is_attached("grasp_left")
                and self.grasp.is_attached("grasp_right")
            ):
                obj_pos_now = self.env.get_body_position(target_body)
                left_offset = np.asarray(offsets[0], dtype=float)
                right_offset = np.asarray(offsets[1], dtype=float)
                desired_center = 0.5 * (
                    (desired_left - left_offset) + (desired_right - right_offset)
                )
                object_error = desired_center - obj_pos_now
                correction = object_feedback_gain * np.clip(
                    object_error,
                    -max_object_correction,
                    max_object_correction,
                )
                desired_left = desired_left + correction
                desired_right = desired_right + correction
            desired_rel = rel_goal if sync_reference is not None else desired_left - desired_right
            info = self.wbc_controller.step(
                left_target_world=desired_left,
                right_target_world=desired_right,
                relative_target_world=desired_rel,
                object_quat_target=getattr(self, "_current_object_reference_quat", None),
                object_body_name=target_body if target_body in self.env._body_names else None,
                dt=self.env.dt,
            )
            if info.source != WBC_SOURCE:
                raise RuntimeError("WBC did not control this step")
            self.env.step()
            count += 1
            max_left_error = max(max_left_error, info.left_error)
            max_right_error = max(max_right_error, info.right_error)
            max_torque_norm = max(max_torque_norm, info.torque_norm)
            saturated_steps += int(info.saturated)

            state_data = self.coordinator.get_state()
            ep.sync_error = max(ep.sync_error, info.sync_error)
            ep.orientation_error = max(ep.orientation_error, info.tilt_error)

            if target_body in self.env._body_names:
                obj_pos = self.env.get_body_position(target_body)
                if offsets is None:
                    left_obj_goal, right_obj_goal = self._dual_targets_from_center(obj_pos, separation)
                else:
                    left_obj_goal = obj_pos + np.asarray(offsets[0], dtype=float)
                    right_obj_goal = obj_pos + np.asarray(offsets[1], dtype=float)
                left_dist = float(np.linalg.norm(state_data.left_ee_pos - left_obj_goal))
                right_dist = float(np.linalg.norm(state_data.right_ee_pos - right_obj_goal))
                left_pinch_dist = self._pinch_distance("left", left_obj_goal)
                right_pinch_dist = self._pinch_distance("right", right_obj_goal)
                near_limit = min(
                    0.025,
                    float(self._raw_config.get("dual_arm_params", {}).get("near_attach_distance", 0.025)),
                )
                left_contact = self._verified_dual_finger_contact("grasp_left", target_body)
                right_contact = self._verified_dual_finger_contact("grasp_right", target_body)
                left_narrow = self._dual_attach_allowed("grasp_left", target_body, left_obj_goal)
                right_narrow = self._dual_attach_allowed("grasp_right", target_body, right_obj_goal)
                if attach_on_near and not left_attached and left_narrow:
                    left_attached = self.grasp.attach("grasp_left", target_body)
                    logger.info(
                        f"Left grasp attached during {phase}: "
                        f"contact={left_contact} ee_dist={left_dist:.3f}m "
                        f"pinch_dist={left_pinch_dist:.3f}m fingers={self._dual_finger_distances()['left']:.3f}m"
                    )
                if attach_on_near and not right_attached and right_narrow:
                    right_attached = self.grasp.attach("grasp_right", target_body)
                    logger.info(
                        f"Right grasp attached during {phase}: "
                        f"contact={right_contact} ee_dist={right_dist:.3f}m "
                        f"pinch_dist={right_pinch_dist:.3f}m fingers={self._dual_finger_distances()['right']:.3f}m"
                    )
                if i in (0, total - 1):
                    debug.append("waypoint_checks", {
                        "phase": phase,
                        "step": int(i),
                        "left_dist_m": left_dist,
                        "right_dist_m": right_dist,
                        "left_pinch_dist_m": left_pinch_dist,
                        "right_pinch_dist_m": right_pinch_dist,
                        "left_contact": bool(left_contact),
                        "right_contact": bool(right_contact),
                        "left_contact_bodies": self.grasp.contacting_gripper_bodies("grasp_left", target_body),
                        "right_contact_bodies": self.grasp.contacting_gripper_bodies("grasp_right", target_body),
                        "finger_distances_m": self._dual_finger_distances(),
                        "left_attached": bool(left_attached),
                        "right_attached": bool(right_attached),
                        "object_world": obj_pos.copy(),
                        "left_ee_world": state_data.left_ee_pos.copy(),
                        "right_ee_world": state_data.right_ee_pos.copy(),
                        "wbc_left_error_m": float(info.left_error),
                        "wbc_right_error_m": float(info.right_error),
                        "wbc_sync_error_m": float(info.sync_error),
                        "wbc_torque_norm": float(info.torque_norm),
                        "wbc_saturated": bool(info.saturated),
                    })

        debug.append("control_quality", {
            "phase": phase,
            "source": WBC_SOURCE,
            "max_left_error_m": float(max_left_error),
            "max_right_error_m": float(max_right_error),
            "max_torque_norm": float(max_torque_norm),
            "saturated_steps": int(saturated_steps),
            "steps": int(count),
        })
        return left_attached, right_attached, count

    def _execute_dual_waypoint_closed_loop(self, left_pos: np.ndarray, right_pos: np.ndarray,
                                           center: np.ndarray, separation: float,
                                           steps: int, hold: int, phase: str,
                                           target_body: str, ep: EpisodeMetrics,
                                           debug: EpisodeDebugWriter,
                                           attach_on_near: bool = False,
                                           offsets: Optional[tuple[np.ndarray, np.ndarray]] = None,
                                           sync_reference: Optional[np.ndarray] = None) -> tuple[bool, bool, int]:
        """Track one waypoint with closed-loop dual-arm Mink Cartesian servo.

        This keeps the mature QP IK solver in the control loop instead of
        solving once and replaying an open-loop joint trajectory.
        """
        state0 = self.coordinator.get_state()
        left_start = state0.left_ee_pos.copy()
        right_start = state0.right_ee_pos.copy()
        left_goal = np.asarray(left_pos[:3], dtype=float)
        right_goal = np.asarray(right_pos[:3], dtype=float)
        total = int(max(2, steps) + max(0, hold))
        left_attached = self.grasp.is_attached("grasp_left")
        right_attached = self.grasp.is_attached("grasp_right")
        count = 0
        max_ik_error = 0.0
        failed_solves = 0

        debug.append("trajectory_plans", {
            "source": "mink_closed_loop_cartesian_servo",
            "phase": phase,
            "target_body": target_body,
            "center_world": np.asarray(center[:3], dtype=float),
            "separation_m": float(separation),
            "steps": int(total),
            "left_target_world": left_goal.copy(),
            "right_target_world": right_goal.copy(),
            "control_law": "per-step Mink dual QP IK from live qpos",
        })

        for i in range(total):
            raw = min(1.0, float(i) / float(max(1, steps - 1)))
            alpha = raw * raw * (3.0 - 2.0 * raw)
            desired_left = left_start + alpha * (left_goal - left_start)
            desired_right = right_start + alpha * (right_goal - right_start)

            sol = self.coordinator.ik.solve_dual_ik(
                targets=[
                    {"pos": desired_left, "body": self.coordinator._ee_body("left")},
                    {"pos": desired_right, "body": self.coordinator._ee_body("right")},
                ],
                current_q=self.env.data.qpos.copy(),
                orientation_cost=0.02,
                posture_cost=5e-4,
                max_steps=80,
                tolerance=0.003,
            )
            max_ik_error = max(max_ik_error, float(sol.position_error))
            if sol.success and sol.joint_positions.size:
                left_cmd = self.coordinator.ik.qpos_for_joints(sol.joint_positions, self.coordinator.left_joints)
                right_cmd = self.coordinator.ik.qpos_for_joints(sol.joint_positions, self.coordinator.right_joints)
                self.coordinator._set_joints("left", left_cmd)
                self.coordinator._set_joints("right", right_cmd)
            else:
                failed_solves += 1
                self.coordinator._set_joints("left", self.coordinator._get_joint_positions("left"))
                self.coordinator._set_joints("right", self.coordinator._get_joint_positions("right"))

            self.env.step()
            count += 1

            state_data = self.coordinator.get_state()
            if sync_reference is not None:
                sync_err = float(np.linalg.norm(
                    (state_data.left_ee_pos - state_data.right_ee_pos) - sync_reference
                ))
            else:
                sync_err = state_data.sync_error
            ep.sync_error = max(ep.sync_error, sync_err)

            if target_body in self.env._body_names:
                obj_pos = self.env.get_body_position(target_body)
                if offsets is None:
                    left_obj_goal, right_obj_goal = self._dual_targets_from_center(obj_pos, separation)
                else:
                    left_obj_goal = obj_pos + np.asarray(offsets[0], dtype=float)
                    right_obj_goal = obj_pos + np.asarray(offsets[1], dtype=float)
                left_dist = float(np.linalg.norm(state_data.left_ee_pos - left_obj_goal))
                right_dist = float(np.linalg.norm(state_data.right_ee_pos - right_obj_goal))
                left_pinch_dist = self._pinch_distance("left", left_obj_goal)
                right_pinch_dist = self._pinch_distance("right", right_obj_goal)
                near_limit = min(
                    0.025,
                    float(self._raw_config.get("dual_arm_params", {}).get("near_attach_distance", 0.025)),
                )
                left_contact = self._verified_dual_finger_contact("grasp_left", target_body)
                right_contact = self._verified_dual_finger_contact("grasp_right", target_body)
                left_narrow = left_contact and left_pinch_dist <= near_limit
                right_narrow = right_contact and right_pinch_dist <= near_limit
                if attach_on_near and not left_attached and left_narrow:
                    left_attached = self.grasp.attach("grasp_left", target_body)
                    logger.info(
                        f"Left grasp attached during {phase}: "
                        f"contact={left_contact} ee_dist={left_dist:.3f}m pinch_dist={left_pinch_dist:.3f}m"
                    )
                if attach_on_near and not right_attached and right_narrow:
                    right_attached = self.grasp.attach("grasp_right", target_body)
                    logger.info(
                        f"Right grasp attached during {phase}: "
                        f"contact={right_contact} ee_dist={right_dist:.3f}m pinch_dist={right_pinch_dist:.3f}m"
                    )
                if i in (0, total - 1):
                    debug.append("waypoint_checks", {
                        "phase": phase,
                        "step": int(i),
                        "left_dist_m": left_dist,
                        "right_dist_m": right_dist,
                        "left_pinch_dist_m": left_pinch_dist,
                        "right_pinch_dist_m": right_pinch_dist,
                        "left_contact": bool(left_contact),
                        "right_contact": bool(right_contact),
                        "left_attached": bool(left_attached),
                        "right_attached": bool(right_attached),
                        "object_world": obj_pos.copy(),
                        "left_ee_world": state_data.left_ee_pos.copy(),
                        "right_ee_world": state_data.right_ee_pos.copy(),
                        "max_ik_error_m": float(max_ik_error),
                        "failed_solves": int(failed_solves),
                    })

        debug.append("control_quality", {
            "phase": phase,
            "source": "mink_closed_loop_cartesian_servo",
            "max_ik_error_m": float(max_ik_error),
            "failed_solves": int(failed_solves),
        })
        return left_attached, right_attached, count

    def _pinch_distance(self, side: str, target: np.ndarray) -> float:
        body_name = DUAL_PANDA_GRIPPERS[side].pinch_body
        if body_name not in self.env._body_names:
            return float("inf")
        mujoco.mj_forward(self.env.model, self.env.data)
        return float(np.linalg.norm(
            self.env.get_body_position(body_name) - np.asarray(target[:3], dtype=float)
        ))

    def _dual_attach_allowed(self, weld: str, target_body: str, target: np.ndarray) -> bool:
        bodies = self.grasp.contacting_gripper_bodies(weld, target_body)
        if not bodies:
            return False
        side = "left" if weld.endswith("left") else "right"
        expected = set(DUAL_PANDA_GRIPPERS[side].finger_bodies)
        if not expected.intersection(bodies):
            return False
        narrow_limit = float(
            self._raw_config.get("dual_arm_params", {}).get("pinch_frame_attach_limit", 0.045)
        )
        return self._pinch_distance(side, target) <= narrow_limit

    def _verified_dual_finger_contact(self, weld: str, target_body: str) -> bool:
        bodies = self.grasp.contacting_gripper_bodies(weld, target_body)
        if len(bodies) >= 2:
            return True
        if not bodies:
            return False
        side = "left" if weld.endswith("left") else "right"
        obj_pos = self.env.get_body_position(target_body)
        separation = float(self._raw_config.get("dual_arm_params", {}).get("rod_grasp_separation", 0.34))
        offset = np.array([0.0, separation / 2.0, 0.0])
        target = obj_pos + (offset if side == "left" else -offset)
        return self._dual_attach_allowed(weld, target_body, target)

    def _execute_dual_carry_primitive(self, carried_object: str,
                                      target_region: np.ndarray,
                                      reference_quat: np.ndarray,
                                      ep: EpisodeMetrics,
                                      debug: EpisodeDebugWriter) -> tuple[bool, bool, int]:
        """Execute the task's synchronized long-object carry chain."""
        if carried_object not in self.env._body_names:
            ep.custom["failure"] = "missing_carried_object"
            return False, False, 0

        separation = float(self._raw_config.get("dual_arm_params", {}).get("rod_grasp_separation", 0.34))
        self.coordinator.nominal_gripper_separation = separation
        self._current_object_reference_quat = np.asarray(reference_quat, dtype=float)
        steps_total = 0

        obj_pos = self.env.get_body_position(carried_object)
        approach_z = float(self._raw_config.get("dual_arm_params", {}).get("approach_z", 0.88))
        grasp_z = float(self._raw_config.get("dual_arm_params", {}).get("grasp_z", 0.88))
        lift_z = float(self._raw_config.get("dual_arm_params", {}).get("lift_z", 0.95))
        place_z = float(self._raw_config.get("dual_arm_params", {}).get("place_z", 0.88))

        logger.info(f"  Dual carry: approach center={obj_pos.round(3)}")
        l_att, r_att, n = self._execute_dual_waypoint(
            np.array([obj_pos[0], obj_pos[1], approach_z]),
            separation, steps=160, hold=30, phase="approach",
            target_body=carried_object, ep=ep, debug=debug,
            attach_on_near=False,
        )
        steps_total += n

        obj_pos = self.env.get_body_position(carried_object)
        l_att, r_att, n = self._execute_dual_waypoint(
            np.array([obj_pos[0], obj_pos[1], grasp_z]),
            separation, steps=140, hold=220, phase="dual_grasp",
            target_body=carried_object, ep=ep, debug=debug,
            attach_on_near=True,
        )
        steps_total += n

        object_attached = self.grasp.is_attached("grasp_left") and self.grasp.is_attached("grasp_right")
        if not object_attached:
            ep.custom["failure"] = "dual_grasp_attach_failed"
            return False, False, steps_total
        self._configure_dual_object_welds(carried_object, separation, debug)

        state_after_attach = self.coordinator.get_state()
        attach_obj_pos = self.env.get_body_position(carried_object)
        left_offset_actual = state_after_attach.left_ee_pos - attach_obj_pos
        right_offset_actual = state_after_attach.right_ee_pos - attach_obj_pos
        sync_reference = state_after_attach.left_ee_pos - state_after_attach.right_ee_pos
        ep.sync_error = 0.0
        debug.add("attach_constraint", {
            "object_world": attach_obj_pos.copy(),
            "left_offset_actual": left_offset_actual.copy(),
            "right_offset_actual": right_offset_actual.copy(),
            "sync_reference": sync_reference.copy(),
        })

        obj_pos = self.env.get_body_position(carried_object)
        l_att, r_att, n = self._execute_dual_waypoint(
            np.array([obj_pos[0], obj_pos[1], lift_z]),
            separation, steps=120, hold=40, phase="lift",
            target_body=carried_object, ep=ep, debug=debug,
            attach_on_near=False,
            offsets=(left_offset_actual, right_offset_actual),
            sync_reference=sync_reference,
        )
        steps_total += n

        target = np.asarray(target_region[:3], dtype=float)
        carry_centers = [
            np.array([target[0], obj_pos[1], lift_z]),
            np.array([target[0], target[1], lift_z]),
            np.array([target[0], target[1], place_z]),
        ]
        carry_steps = [360, 360, 360]
        carry_holds = [160, 140, 160]
        for idx, center in enumerate(carry_centers):
            l_att, r_att, n = self._execute_dual_waypoint(
                center,
                separation, steps=carry_steps[idx], hold=carry_holds[idx],
                phase=f"carry_{idx + 1}",
                target_body=carried_object, ep=ep, debug=debug,
                attach_on_near=False,
                offsets=(left_offset_actual, right_offset_actual),
                sync_reference=sync_reference,
            )
            steps_total += n
            tilt = self.coordinator.object_tilt(carried_object, reference_quat)
            ep.orientation_error = max(ep.orientation_error, tilt)

        final_pos = self.env.get_body_position(carried_object)
        carry_reached_target = float(np.linalg.norm(final_pos[:2] - target[:2])) < 0.25
        debug.append("place_checks", {
            "object_world_before_release": final_pos.copy(),
            "target_region_world": target.copy(),
            "xy_error_m": float(np.linalg.norm(final_pos[:2] - target[:2])),
            "tilt_rad": float(ep.orientation_error),
        })
        return object_attached, carry_reached_target, steps_total

    def _configure_dual_object_welds(self, obj_body: str, separation: float,
                                     debug: EpisodeDebugWriter):
        """Anchor the two active welds at the object's left/right grasp points."""
        if obj_body not in self.env._body_names:
            return
        anchors = {
            "grasp_left": np.array([0.0, separation / 2.0, 0.0]),
            "grasp_right": np.array([0.0, -separation / 2.0, 0.0]),
        }
        obj_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, obj_body)
        mujoco.mj_forward(self.env.model, self.env.data)
        obj_pos = self.env.data.xpos[obj_id].copy()
        obj_rot = self.env.data.xmat[obj_id].reshape(3, 3).copy()
        rows = []
        for weld_name, anchor_obj in anchors.items():
            info = getattr(self.grasp, "_welds", {}).get(weld_name)
            if not info:
                continue
            eq_id = int(info["eq_id"])
            gripper_body = info["gripper_body"]
            gripper_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, gripper_body)
            if gripper_id < 0:
                continue
            grip_pos = self.env.data.xpos[gripper_id].copy()
            grip_rot = self.env.data.xmat[gripper_id].reshape(3, 3).copy()
            anchor_world = obj_pos + obj_rot @ anchor_obj
            anchor_gripper = grip_rot.T @ (anchor_world - grip_pos)
            rel_rot = grip_rot.T @ obj_rot
            rel_quat = np.zeros(4)
            mujoco.mju_mat2Quat(rel_quat, rel_rot.reshape(-1))
            self.env.model.eq_data[eq_id, 0:3] = anchor_obj
            self.env.model.eq_data[eq_id, 3:6] = anchor_gripper
            self.env.model.eq_data[eq_id, 6:10] = rel_quat
            self.env.model.eq_obj2id[eq_id] = obj_id
            if hasattr(self.env.data, "eq_active"):
                self.env.data.eq_active[eq_id] = 1
            rows.append({
                "weld": weld_name,
                "anchor_obj": anchor_obj.copy(),
                "anchor_world": anchor_world.copy(),
                "anchor_gripper": anchor_gripper.copy(),
            })
        mujoco.mj_forward(self.env.model, self.env.data)
        debug.add("dual_weld_anchors", rows)

    def _plan_single_arm_trajectory(self, side: str, target_qpos: np.ndarray,
                                    num_steps: int = 80) -> np.ndarray:
        joints = self.coordinator.left_joints if side == "left" else self.coordinator.right_joints
        current = self.coordinator._get_joint_positions(side)
        target = self.coordinator.ik.qpos_for_joints(target_qpos, joints)
        max_step = float(self._raw_config.get("task", {}).get("max_joint_step", 0.12))
        joint_delta = float(np.max(np.abs(target - current))) if len(target) else 0.0
        steps = max(num_steps, int(np.ceil(joint_delta / max(max_step, 1e-6))) + 1)
        return TrajectoryGenerator.cubic_spline(current, target, steps)

    def _finalize_motion_trace(self, ep: EpisodeMetrics):
        summary = save_motion_trace(self.motion_trace, "22", ep.episode_id)
        if summary:
            ep.custom["motion_trace"] = summary

    def run_evaluation(self, tasks: List[str] = None) -> dict:
        """Run evaluation over multiple task instructions."""
        if tasks is None:
            tasks = [
                "用双手把长杆放到指定区域",
                "一起抱起箱子搬运到目的地",
            ]

        logger.info(f"Running evaluation with {len(tasks)} tasks")
        for i, instr in enumerate(tasks):
            logger.info(f"\n{'='*50}")
            logger.info(f"Task {i+1}/{len(tasks)}: {instr}")
            logger.info(f"{'='*50}")
            ep = self.run_single_episode(instr)
            logger.info(f"  Success: {ep.success} | Time: {ep.total_time:.1f}s")

        self.metrics.print_summary()
        self.metrics.save(f"results_22_{len(self.metrics.episodes)}ep.json")
        return self.metrics.summary()

    def cleanup(self):
        """Clean up resources."""
        if self.wbc_controller and hasattr(self.wbc_controller, "close"):
            self.wbc_controller.close()
        if self.env:
            self.env.close()

