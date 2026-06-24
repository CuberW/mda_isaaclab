#!/usr/bin/env python
"""
System health checks for the active robot task pipelines.

Modes:
  --light  checks the existing repo skeleton: scenes, cameras, Mink scratch IK,
           controller motion, trace plumbing, and no runtime teleport writes.
  --full   additionally requires mature backends: GraspNet inference runtime
           and torque-level dual-arm WBC.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _result(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{': ' + detail if detail else ''}")
    return ok


def _status(name: str, status: str, detail: str = "") -> None:
    print(f"[{status}] {name}{': ' + detail if detail else ''}")


def _render_nonblank(env, camera: str) -> bool:
    img = env.render(camera)
    return bool(img.size and np.max(img) > np.min(img))


def _check_camera_frame(env, camera: str, bodies) -> list[bool]:
    checks = []
    if isinstance(bodies, str):
        bodies = [bodies]
    intr = env.get_camera_intrinsics(camera)
    pos, rot = env.get_camera_pose(camera)
    finite_intr = all(np.isfinite(float(intr[k])) for k in ("fx", "fy", "cx", "cy"))
    finite_pose = np.isfinite(pos).all() and np.isfinite(rot).all()
    checks.append(_result(f"{camera} intrinsics finite", finite_intr))
    checks.append(_result(f"{camera} extrinsics finite", finite_pose))
    existing = [body for body in bodies if body in getattr(env, "_body_names", [])]
    if not existing:
        checks.append(_result(f"{camera} projects candidate body", False, "body missing"))
        return checks
    body = existing[0]
    point_world = env.get_body_position(body)
    point_camera = env.world_point_to_camera(point_world, camera)
    round_trip = env.camera_point_to_world(point_camera, camera)
    round_trip_error = float(np.linalg.norm(round_trip - point_world))
    checks.append(_result(
        f"{camera} world/camera round-trip",
        np.isfinite(round_trip_error) and round_trip_error < 1e-8,
        f"err={round_trip_error:.2e}m",
    ))
    visible_body = ""
    visible_projection = None
    visible_camera = None
    for candidate in existing:
        candidate_world = env.get_body_position(candidate)
        candidate_camera = env.world_point_to_camera(candidate_world, camera)
        candidate_projection = env.project_world_point(candidate_world, camera)
        candidate_pixel = np.asarray(candidate_projection["pixel"], dtype=float)
        if np.isfinite(candidate_pixel).all() and candidate_projection["visible"]:
            visible_body = candidate
            visible_projection = candidate_projection
            visible_camera = candidate_camera
            break
    if visible_projection is None:
        projection = env.project_world_point(point_world, camera)
        pixel = np.asarray(projection["pixel"], dtype=float)
        checks.append(_result(
            f"{camera} projects candidate body",
            False,
            f"none visible; first={body} pixel=({pixel[0]:.1f},{pixel[1]:.1f})",
        ))
        return checks
    pixel = np.asarray(visible_projection["pixel"], dtype=float)
    checks.append(_result(
        f"{camera} projects {visible_body}",
        True,
        f"pixel=({pixel[0]:.1f},{pixel[1]:.1f}) cam_z={visible_camera[2]:.3f}",
    ))
    return checks


def _check_static_no_runtime_teleport() -> bool:
    allowed = {
        ("robot_common/env/__init__.py", "set_joint_positions"),
        ("robot_common/env/__init__.py", "reset"),
        ("robot_common/execution/mink_ik.py", "solve_ik"),
        ("robot_common/execution/mink_ik.py", "solve_dual_ik"),
    }
    targets = [
        PROJECT_ROOT / "task_319_garbage_sort" / "__init__.py",
        PROJECT_ROOT / "task_22_dual_arm" / "__init__.py",
        PROJECT_ROOT / "robot_common" / "execution" / "mink_ik.py",
    ]
    violations = []
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        parent = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AugAssign)):
                continue
            assign_targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = []
            for target in assign_targets:
                text = ast.unparse(target)
                if ".data.qpos" in text or (".qpos" in text and "data" in text):
                    names.append(text)
            if not names:
                continue
            func = ""
            cur = node
            while cur in parent:
                cur = parent[cur]
                if isinstance(cur, ast.FunctionDef):
                    func = cur.name
                    break
            if (rel, func) not in allowed:
                violations.append(f"{rel}:{node.lineno} in {func or '<module>'}: {', '.join(names)}")
    if violations:
        for item in violations:
            print(f"  teleport write: {item}")
    return _result("no runtime qpos teleport writes", not violations)


def _body_distance(env, body_a: str, body_b: str) -> float:
    return float(np.linalg.norm(env.get_body_position(body_a) - env.get_body_position(body_b)))


def check_backend_matrix(full: bool) -> bool:
    from robot_common.execution.mature_backends import (
        ANYGRASP_BACKEND,
        GRASPNET_BACKEND,
        GROUNDING_DINO_BACKEND,
        MINK_BACKEND,
        SAM_BACKEND,
        check_backend,
        describe_backend,
    )

    requirements = [MINK_BACKEND, GROUNDING_DINO_BACKEND, SAM_BACKEND]
    if full:
        requirements.extend([GRASPNET_BACKEND])
        # AnyGrasp is license-gated; report it but do not block GraspNet-based
        # assignment acceptance.
        any_ok = check_backend(ANYGRASP_BACKEND, required=False)
        _status(
            "backend AnyGrasp optional",
            "PASS" if any_ok else "SKIP",
            "" if any_ok else "license/sdk not configured",
        )
    def _check_graspnet_full_stack() -> tuple[bool, str]:
        try:
            from robot_common.execution.grasp_estimators import GraspNetEstimator

            est = GraspNetEstimator(required=False)
            ok = est.refresh_availability()
            return ok, est.availability_detail()
        except Exception as exc:
            return False, f"WSL GraspNet probe failed: {exc}"

    checks = []
    for req in requirements:
        if full and req.name == GRASPNET_BACKEND.name:
            ok, detail = _check_graspnet_full_stack()
            checks.append(_result(f"backend {req.name}", ok, detail))
            continue
        ok = check_backend(req, required=False)
        checks.append(_result(f"backend {req.name}", ok, "" if ok else describe_backend(req)))
    return all(checks)


def check_full_backend_runtime() -> bool:
    checks = []
    try:
        from robot_common.execution.grasp_estimators import GraspNetEstimator

        est = GraspNetEstimator(required=False)
        checks.append(_result("GraspNet estimator configured", est.available))
        if est.available:
            rgb = np.zeros((32, 32, 3), dtype=np.uint8)
            depth = np.ones((32, 32), dtype=np.float32) * 0.5
            mask = np.ones((32, 32), dtype=np.uint8)
            try:
                pose = est.estimate(rgb, depth, {"fx": 500, "fy": 500, "cx": 16, "cy": 16}, mask=mask)
                checks.append(_result("GraspNet one-frame inference", np.isfinite(pose.position).all()))
            except Exception as exc:
                checks.append(_result("GraspNet one-frame inference", False, str(exc)))
    except Exception as exc:
        checks.append(_result("GraspNet runtime import", False, str(exc)))

    return all(checks)


def check_task_319(full: bool) -> bool:
    from task_319_garbage_sort import GarbageSortingPipeline

    p = GarbageSortingPipeline()
    try:
        checks = []
        checks.append(_result("3.19 scene bodies", "base_link" in p.env._body_names and "trash_01" in p.env._body_names))
        checks.append(_result(
            "3.19 Stretch real gripper fingers present",
            "link_gripper_finger_left" in p.env._body_names
            and "link_gripper_finger_right" in p.env._body_names
            and p.env.actuator_index("grip") is not None,
        ))
        checks.append(_result("3.19 camera render", _render_nonblank(p.env, "camera_rgb")))
        checks.extend(_check_camera_frame(
            p.env, "camera_rgb",
            [name for name in p.env._body_names if name.startswith("trash_")],
        ))
        p.env.reset()
        start = p.controller.get_base_pose().copy()
        p.controller.move_base(0.24, 0.0)
        for _ in range(80):
            p.env.step()
        end = p.controller.get_base_pose().copy()
        p.controller.move_base(0.0, 0.0)
        moved = float(np.linalg.norm(end[:2] - start[:2]))
        checks.append(_result("3.19 base actuator moves visibly", moved > 0.025, f"delta={moved:.4f}m"))
        try:
            source = (PROJECT_ROOT / "task_319_garbage_sort" / "pipeline.py").read_text(encoding="utf-8")
            contract_ok = all(token in source for token in (
                "navigation_complete",
                "delivery_complete",
                "bin_delivery_results",
            ))
            checks.append(_result("3.19 success requires navigation/grasp/delivery contract", contract_ok))
        except Exception as exc:
            checks.append(_result("3.19 success requires navigation/grasp/delivery contract", False, str(exc)))
        if full:
            grasp_ready = (
                p.grasp_estimator is not None
                and p.grasp_estimator.refresh_availability()
            )
            detail = (
                p.grasp_estimator.availability_detail()
                if p.grasp_estimator is not None else "not initialized"
            )
            checks.append(_result("3.19 GraspNet configured", grasp_ready, detail))
        return all(checks)
    finally:
        p.cleanup()


def check_task_319_kuavo_wheel(full: bool = False) -> bool:
    import mujoco

    from control import MuJoCoEnv
    from robot_common.infra.task_lifecycle import load_raw_yaml
    from task_319_garbage_sort.kuavo_controller import KuavoWheelGarbageController

    raw_cfg = load_raw_yaml(PROJECT_ROOT / "configs" / "task_319_kuavo_wheel.yaml")
    execution_cfg = raw_cfg.get("execution", {})
    mobile_cfg = raw_cfg.get("mobile_manipulation", {})
    env = MuJoCoEnv(
        PROJECT_ROOT / "simulation" / "scenes" / "task_319_kuavo_wheel_s62.xml",
        camera_name="camera_rgb",
    )
    try:
        controller = KuavoWheelGarbageController(
            env,
            official_ik_config=execution_cfg.get("official_ik", {}),
            official_control_config=execution_cfg.get("official_control", {}),
        )
        checks = []
        for body in ("base_link", "r_pinch", "r_f_fingers", "r_b_fingers", "trash_01"):
            checks.append(_result(f"3.19 Kuavo body {body}", body in env._body_names))
        full_right_hand_mesh_present = any(
            env.model.mesh(i).name == "r_hand_pitch"
            for i in range(env.model.nmesh)
        )
        full_right_hand_geom_present = any(
            env.model.geom(i).name == "r_hand_pitch"
            or (
                int(env.model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH)
                and env.model.mesh(int(env.model.geom_dataid[i])).name == "r_hand_pitch"
            )
            for i in range(env.model.ngeom)
            if int(env.model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        )
        checks.append(_result(
            "3.19 Kuavo full right hand removed when official claw is mounted",
            not full_right_hand_mesh_present and not full_right_hand_geom_present,
            (
                f"full_hand_mesh_present={full_right_hand_mesh_present}, "
                f"full_hand_geom_present={full_right_hand_geom_present}; "
                "r_hand_pitch_noHand is allowed as wrist/claw mount"
            ),
        ))

        joints = {
            name: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in (
                "LF_wheel_pitch_joint",
                "RF_wheel_pitch_joint",
                "zarm_r1_joint",
                "zarm_r7_joint",
                "zhead_1_joint",
                "zhead_2_joint",
                "r_f_bar-1",
                "r_b_bar-1",
                "r_f_fingers",
                "r_b_fingers",
            )
        }
        actuator_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "r_fingers_actuator")
        checks.append(_result(
            "3.19 Kuavo wheel/arm/head/official claw joints present",
            all(jid >= 0 for jid in joints.values()) and actuator_id >= 0,
            ", ".join(f"{name}={jid}" for name, jid in joints.items()) + f", r_fingers_actuator={actuator_id}",
        ))
        checks.append(_result("3.19 Kuavo head camera render", _render_nonblank(env, "camera_rgb")))
        checks.extend(_check_camera_frame(env, "camera_rgb", ["trash_01", "trash_02", "trash_03", "trash_04"]))

        env.reset()
        base0 = env.get_body_position("base_link").copy()
        controller.move_planar_to(np.array([0.08, 0.0, 0.0], dtype=float), steps=900, tolerance=0.025)
        base1 = env.get_body_position("base_link").copy()
        controller.stop_base()
        base_delta = float(np.linalg.norm(base1[:2] - base0[:2]))
        checks.append(_result("3.19 Kuavo base actuator responds", base_delta > 0.04, f"delta={base_delta:.4f}m"))

        env.reset()
        ee0 = env.get_body_position("r_pinch").copy()
        controller.move_arm(lift=0.35, extend=0.15, preserve_base=True)
        for _ in range(120):
            env.step()
        ee1 = env.get_body_position("r_pinch").copy()
        ee_delta = float(np.linalg.norm(ee1 - ee0))
        checks.append(_result("3.19 Kuavo right arm actuator responds", ee_delta > 0.015, f"delta={ee_delta:.4f}m"))

        env.reset()
        controller.open_gripper()
        for _ in range(80):
            env.step()
        open_dist = _body_distance(env, "r_f_fingers", "r_b_fingers")
        controller.close_gripper()
        for _ in range(80):
            env.step()
        close_dist = _body_distance(env, "r_f_fingers", "r_b_fingers")
        checks.append(_result(
            "3.19 Kuavo official claw open/close changes finger distance",
            open_dist - close_dist > 0.010,
            f"open={open_dist:.4f}m close={close_dist:.4f}m",
        ))
        env.reset()
        mujoco.mj_forward(env.model, env.data)
        wrist_to_pinch = _body_distance(env, "zarm_r7_end_effector", "r_pinch")
        finger_center = 0.5 * (env.get_body_position("r_f_fingers") + env.get_body_position("r_b_fingers"))
        center_to_pinch = float(np.linalg.norm(finger_center - env.get_body_position("r_pinch")))
        checks.append(_result(
            "3.19 Kuavo claw TCP aligned with official pinch center",
            wrist_to_pinch < 0.09 and center_to_pinch < 0.005,
            f"wrist_to_pinch={wrist_to_pinch:.4f}m center_to_pinch={center_to_pinch:.4f}m",
        ))
        if full:
            if (
                bool(mobile_cfg.get("full_requires_ros2", False))
                or bool(mobile_cfg.get("full_requires_nav2", False))
                or bool(mobile_cfg.get("full_requires_moveit2", False))
            ):
                from task_319_garbage_sort.ros2_mobile_manipulation_pipeline import ROS2MobileManipulationPipeline

                ros2_pipeline = ROS2MobileManipulationPipeline(raw_cfg)
                ros2_status = ros2_pipeline.ready_status()
                details = ros2_status.details
                nav_status = details.get("nav2")
                moveit_status = details.get("moveit2")
                kuavo_ros2_status = details.get("kuavo")
                checks.append(_result(
                    "3.19 ROS2/Nav2 graph ready",
                    bool(nav_status and nav_status.ready),
                    ", ".join(getattr(nav_status, "missing", ())) if nav_status else "no status",
                ))
                checks.append(_result(
                    "3.19 ROS2/MoveIt2 graph ready",
                    bool(moveit_status and moveit_status.ready),
                    ", ".join(getattr(moveit_status, "missing", ())) if moveit_status else "no status",
                ))
                checks.append(_result(
                    "3.19 Kuavo ROS2 official control graph ready",
                    bool(kuavo_ros2_status and kuavo_ros2_status.ready),
                    ", ".join(getattr(kuavo_ros2_status, "missing", ())) if kuavo_ros2_status else "no status",
                ))
            else:
                checks.append(_result(
                    "3.19 ROS2/Nav2/MoveIt2 graph not required for current ROS1 official Kuavo path",
                    True,
                    "mobile_manipulation full_requires_ros2/nav2/moveit2 are false",
                ))
            controller.require_official_stack(True)
            status = controller.official_stack_status()
            checks.append(_result(
                "3.19 Kuavo full forbids local IK fallback",
                not status["allow_local_ik_fallback"] and status["require_official_control"],
            ))
            checks.append(_result(
                "3.19 Kuavo official IK service ready",
                status["ik_available"],
                status.get("ik_message", ""),
            ))
            checks.append(_result(
                "3.19 Kuavo official SDK control ready",
                status["control_ready"],
                (
                    f"sdk={status['sdk_import']} ros={status['ros_ready']} "
                    f"arm={status['arm_trajectory_ready']} base={status['base_control_ready']} "
                    f"claw={status['claw_ready']} detail={status['message']}"
                ),
            ))
        return all(checks)
    finally:
        env.close()


def check_task_22(full: bool) -> bool:
    from task_22_dual_arm import DualArmVLAPipeline

    p = DualArmVLAPipeline()
    try:
        checks = []
        checks.append(_result("2.2 scene bodies", "long_rod" in p.env._body_names and "target_region" in p.env._body_names))
        checks.append(_result("2.2 scene render", _render_nonblank(p.env, "scene_camera")))
        checks.extend(_check_camera_frame(p.env, "scene_camera", "long_rod"))
        checks.append(_result("2.2 arm joints", len(p.coordinator.left_joints) >= 7 and len(p.coordinator.right_joints) >= 7))
        try:
            pad_ok = True
            details = []
            for geom_name in ("l_grasp_pad", "r_grasp_pad"):
                gid = p.env.geom_index(geom_name)
                if gid is None:
                    pad_ok = False
                    details.append(f"{geom_name}=missing")
                    continue
                size = float(np.max(p.env.model.geom_size[gid]))
                contype = int(p.env.model.geom_contype[gid])
                conaff = int(p.env.model.geom_conaffinity[gid])
                ok = size <= 0.025 and contype == 0 and conaff == 0
                pad_ok = pad_ok and ok
                details.append(f"{geom_name}:size={size:.3f},contype={contype},conaff={conaff}")
            checks.append(_result("2.2 grasp pads are non-contact visual markers", pad_ok, "; ".join(details)))
        except Exception as exc:
            checks.append(_result("2.2 grasp pads are non-contact visual markers", False, str(exc)))
        try:
            weld_ok = (
                p.grasp._welds["grasp_left"]["gripper_body"] == "l_pinch"
                and p.grasp._welds["grasp_right"]["gripper_body"] == "r_pinch"
            )
            finger_bodies = set()
            for gid in p.grasp._welds["grasp_left"]["gripper_geom_ids"] + p.grasp._welds["grasp_right"]["gripper_geom_ids"]:
                finger_bodies.add(p.env.model.body(int(p.env.model.geom_bodyid[gid])).name)
            finger_ok = {"l_finger_l", "l_finger_r", "r_finger_l", "r_finger_r"}.issubset(finger_bodies)
            checks.append(_result(
                "2.2 grasp weld/contact uses pinch and real fingers",
                weld_ok and finger_ok,
                ",".join(sorted(finger_bodies)),
            ))
        except Exception as exc:
            checks.append(_result("2.2 grasp weld/contact uses pinch and real fingers", False, str(exc)))
        try:
            p.env.reset()
            act = p.env.actuator_index("ml_grip")
            p.env.data.ctrl[act] = 0.015
            for _ in range(80):
                p.env.step()
            open_dist = _body_distance(p.env, "l_finger_l", "l_finger_r")
            p.env.data.ctrl[act] = 0.0
            for _ in range(80):
                p.env.step()
            close_dist = _body_distance(p.env, "l_finger_l", "l_finger_r")
            checks.append(_result(
                "2.2 simple gripper open/close changes finger distance",
                open_dist - close_dist > 0.010,
                f"open={open_dist:.4f}m close={close_dist:.4f}m",
            ))
        except Exception as exc:
            checks.append(_result("2.2 simple gripper open/close changes finger distance", False, str(exc)))
        before = p.env.get_qpos()
        left = np.array([0.2, 0.0, 0.45, 0, 0, 0, 0])
        right = np.array([0.4, 0.0, 0.45, 0, 0, 0, 0])
        lt, rt = p.coordinator.plan_synchronized_trajectory(left, right, num_steps=5)
        after = p.env.get_qpos()
        checks.append(_result("2.2 IK planning does not mutate live qpos", np.allclose(before, after)))
        checks.append(_result("2.2 synchronized trajectory", len(lt) == len(rt) and len(lt) >= 5))
        if full:
            import mujoco
            arm_actuator_ids = []
            for joint_name in p.coordinator.left_joints + p.coordinator.right_joints:
                jid = mujoco.mj_name2id(p.env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                act_id = next(
                    (
                        i for i in range(p.env.model.nu)
                        if int(p.env.model.actuator_trnid[i, 0]) == jid
                    ),
                    -1,
                )
                if act_id >= 0:
                    arm_actuator_ids.append(act_id)
            torque_ok = (
                len(arm_actuator_ids) == 14
                and all(int(p.env.model.actuator_biastype[i]) == int(mujoco.mjtBias.mjBIAS_NONE) for i in arm_actuator_ids)
            )
            checks.append(_result("2.2 WBC arm actuators are torque motors", torque_ok, f"count={len(arm_actuator_ids)}"))
            state = p.coordinator.get_state()
            try:
                info = p.wbc_controller.step(
                    left_target_world=state.left_ee_pos,
                    right_target_world=state.right_ee_pos,
                    relative_target_world=state.left_ee_pos - state.right_ee_pos,
                    dt=p.env.dt,
                )
                p.env.step()
                checks.append(_result(
                    "2.2 WBC torque step",
                    info.source == "dual_arm_wbc_qp_torque" and np.isfinite(info.torque_norm),
                    f"source={info.source} torque_norm={info.torque_norm:.3f}",
                ))
            except Exception as exc:
                checks.append(_result("2.2 WBC torque step", False, str(exc)))
        return all(checks)
    finally:
        p.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--light", action="store_true", help="Check current conda-compatible skeleton")
    mode.add_argument("--full", action="store_true", help="Check complete mature backend stack")
    args = parser.parse_args()
    full = bool(args.full)

    print("System health mode:", "full" if full else "light")
    checks = [
        _check_static_no_runtime_teleport(),
        check_backend_matrix(full),
        check_task_319(full),
        check_task_319_kuavo_wheel(full),
        check_task_22(full),
    ]
    if full:
        checks.append(check_full_backend_runtime())
    ok = all(checks)
    print("\nSystem health:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

