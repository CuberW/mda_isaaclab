"""Head-camera visual grasp demo for task 319.

This entrypoint uses the same phase-one style scene as the ROS/Nav2 preview:
Kuavo S62, a table, four open bins, ten textured YCB trash objects, the attached
right gripper, and the robot head RGB-D camera. The current official perception
chain consumes only the head camera RGB-D frame through VISS v28:

head RGB-D -> Qwen global proposals -> YOLO11 ROI segmentation -> Qwen/VLM verification -> learned grasp candidates
-> optional right-arm grasp/lift/hold.

Legacy GLM/full-frame YOLO code is retained only for historical debugging; it is
not part of the main task path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import shlex
import socket
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

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
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,z.")
    return (parts[0], parts[1], parts[2])


def parse_float_csv(value: str) -> list[float]:
    if value.strip() == "":
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def cli_arg_provided(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def parse_urdf_xyz(value: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=np.float32)
    parts = [float(item) for item in value.split()]
    if len(parts) != 3:
        raise ValueError(f"Expected URDF xyz triplet, got {value!r}.")
    return np.asarray(parts, dtype=np.float32)


def rpy_matrix_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def derive_parallel_gripper_tcp_from_urdf(gripper_urdf: Path) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Compute the pinch-center TCP from the gripper URDF geometry.

    For this attached two-finger gripper, each finger's useful contact pad is
    the forward-most collision element on the finger link. The TCP is the
    midpoint between the left/right pad centers at zero jaw opening, expressed
    in the gripper_base frame.
    """

    fallback = (0.115, 0.0, 0.0)
    metadata: dict[str, Any] = {
        "source": "two_finger_gripper.urdf",
        "urdf_path": str(gripper_urdf),
        "fallback_m": list(fallback),
        "valid": False,
    }
    try:
        root = ET.parse(gripper_urdf).getroot()
        link_by_name = {elem.attrib.get("name", ""): elem for elem in root.findall("link")}
        pad_points: list[np.ndarray] = []
        pad_records: list[dict[str, Any]] = []
        joint_records: list[dict[str, Any]] = []
        for joint_name in ("left_finger_joint", "right_finger_joint"):
            joint = root.find(f"joint[@name='{joint_name}']")
            if joint is None:
                raise RuntimeError(f"Missing joint {joint_name!r}.")
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or parent.attrib.get("link") != "gripper_base" or child is None:
                raise RuntimeError(f"Unexpected parent/child for {joint_name!r}.")
            child_name = str(child.attrib.get("link", ""))
            link = link_by_name.get(child_name)
            if link is None:
                raise RuntimeError(f"Missing child link {child_name!r}.")
            joint_origin_elem = joint.find("origin")
            joint_origin = parse_urdf_xyz(joint_origin_elem.attrib.get("xyz") if joint_origin_elem is not None else None)
            axis_elem = joint.find("axis")
            axis = parse_urdf_xyz(axis_elem.attrib.get("xyz") if axis_elem is not None else None, default=(1.0, 0.0, 0.0))
            limit_elem = joint.find("limit")
            lower = float(limit_elem.attrib.get("lower", "0")) if limit_elem is not None else 0.0
            upper = float(limit_elem.attrib.get("upper", "0")) if limit_elem is not None else 0.0
            collision_records: list[tuple[np.ndarray, np.ndarray | None]] = []
            for collision in link.findall("collision"):
                origin = collision.find("origin")
                collision_origin = parse_urdf_xyz(origin.attrib.get("xyz") if origin is not None else None)
                box_size = None
                geometry = collision.find("geometry")
                box = geometry.find("box") if geometry is not None else None
                if box is not None and box.attrib.get("size"):
                    box_size = parse_urdf_xyz(box.attrib.get("size"))
                collision_records.append((collision_origin, box_size))
            if not collision_records:
                raise RuntimeError(f"Finger link {child_name!r} has no collision origins.")
            pad_origin, pad_box_size = max(collision_records, key=lambda item: float(item[0][0]))
            pad_in_base = joint_origin + pad_origin
            pad_points.append(pad_in_base)
            pad_records.append(
                {
                    "joint": joint_name,
                    "child_link": child_name,
                    "joint_origin_xyz_m": joint_origin.astype(float).tolist(),
                    "joint_axis_xyz": axis.astype(float).tolist(),
                    "joint_limit_lower_m": lower,
                    "joint_limit_upper_m": upper,
                    "selected_pad_origin_xyz_m": pad_origin.astype(float).tolist(),
                    "selected_pad_box_size_m": pad_box_size.astype(float).tolist() if pad_box_size is not None else None,
                    "pad_center_in_gripper_base_m": pad_in_base.astype(float).tolist(),
                }
            )
            joint_records.append(
                {
                    "joint": joint_name,
                    "child_link": child_name,
                    "axis_xyz": axis.astype(float).tolist(),
                    "lower_m": lower,
                    "upper_m": upper,
                    "motion_direction": "opens jaw outward from centerline" if upper > lower else "fixed_or_invalid",
                }
            )
        tcp = np.mean(np.stack(pad_points, axis=0), axis=0).astype(np.float32)
        left_y = float(pad_points[0][1])
        right_y = float(pad_points[1][1])
        center_gap_y = abs(left_y - right_y)
        left_thickness_y = float((pad_records[0].get("selected_pad_box_size_m") or [0.0, 0.0, 0.0])[1])
        right_thickness_y = float((pad_records[1].get("selected_pad_box_size_m") or [0.0, 0.0, 0.0])[1])
        closed_gap_m = max(0.0, center_gap_y - 0.5 * (left_thickness_y + right_thickness_y))
        max_opening_m = float(closed_gap_m + sum(max(0.0, item["joint_limit_upper_m"] - item["joint_limit_lower_m"]) for item in pad_records))
        metadata.update(
            {
                "valid": True,
                "joint_records": joint_records,
                "pad_records": pad_records,
                "tcp_offset_gripper_base_m": tcp.astype(float).tolist(),
                "closed_inner_gap_m": closed_gap_m,
                "max_opening_width_m": max_opening_m,
                "finger_pad_effective_grasp_center_m": tcp.astype(float).tolist(),
            }
        )
        return (float(tcp[0]), float(tcp[1]), float(tcp[2])), metadata
    except Exception as exc:
        metadata["error"] = repr(exc)
        return fallback, metadata


RIGHT_GRIPPER_LOCAL_TCP_OFFSET_M, RIGHT_GRIPPER_TCP_URDF_CALIBRATION = derive_parallel_gripper_tcp_from_urdf(
    WORKSPACE_ROOT / "task_319_garbage_sort/two_finger_gripper.urdf"
)
RIGHT_GRIPPER_INLINE_MOUNT_RPY = (0.0, math.pi / 2.0, 0.0)
RIGHT_GRIPPER_INLINE_MOUNT_ROT = rpy_matrix_xyz(*RIGHT_GRIPPER_INLINE_MOUNT_RPY)
RIGHT_WRIST_INLINE_TCP_OFFSET_M = tuple(
    float(v)
    for v in (
        RIGHT_GRIPPER_INLINE_MOUNT_ROT
        @ np.asarray(RIGHT_GRIPPER_LOCAL_TCP_OFFSET_M, dtype=np.float32)
    ).tolist()
)
RIGHT_GRIPPER_FINGER_PAD_CENTER_LOCAL_M = (0.075, 0.0, 0.0)


parser = argparse.ArgumentParser(description="Task 319 head-camera visual grasp demo.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_cycles", type=int, default=0, help="Number of perception cycles. 0 loops until Isaac closes.")
parser.add_argument("--warmup_steps", type=int, default=80)
parser.add_argument("--cycle_interval_steps", type=int, default=1)
parser.add_argument("--output_dir", default="task_319_garbage_sort/output/head_camera_grasp_records")
parser.add_argument("--robot_pos", type=parse_xyz, default=(0.18, 0.0, 0.0), help="Robot root position x,y,z. Default is a backed-off observation pose for full-table RGB-D capture.")
parser.add_argument("--robot_yaw", type=float, default=0.0, help="Robot yaw in radians; 0 faces the +X table direction.")
parser.add_argument("--head_yaw", type=float, default=0.0)
parser.add_argument("--head_pitch", type=float, default=0.0)
parser.add_argument("--head_camera_width", type=int, default=1280, help="Head RGB-D camera width for the grasp perception frame.")
parser.add_argument("--head_camera_height", type=int, default=960, help="Head RGB-D camera height for the grasp perception frame.")
parser.add_argument("--wrist_camera_width", type=int, default=640, help="Right wrist RGB-D camera width for local grasp refinement.")
parser.add_argument("--wrist_camera_height", type=int, default=480, help="Right wrist RGB-D camera height for local grasp refinement.")
parser.add_argument("--perception_source", choices=("v28_original",), default="v28_original", help="Official visual perception chain: VISS v28 Qwen-global proposals + YOLO11 ROI segmentation + Qwen/VLM verification on the current head RGB-D frame. Legacy GLM/full-frame YOLO sources are retired.")
parser.add_argument("--yolo_weights", default="models/yolo/yolov8m-seg.pt")
parser.add_argument("--yolo_conf", type=float, default=0.08)
parser.add_argument("--yolo_imgsz", type=int, default=640)
parser.add_argument("--skip_yolo", action="store_true")
parser.add_argument("--skip_vlm", action="store_true", help="Debug-only non-visual demos may set this internally. The official v28 perception path always uses Qwen/VLM verification.")
parser.add_argument("--viss_qwen_script", default="viss/scripts/perception/yolo11_qwen_perception_offline.py", help="VISS Qwen-first perception script used by the official --perception_source v28_original path.")
parser.add_argument("--viss_qwen_model", default="viss/models/yolo11s-seg-best.pt", help="YOLO segmentation model used inside the v28 Qwen-first ROI refinement chain.")
parser.add_argument("--viss_qwen_conf", type=float, default=0.15)
parser.add_argument("--viss_qwen_roi_expand", type=float, default=2.0)
parser.add_argument("--viss_qwen_verify_mode", choices=("all", "planner_only", "top1", "none"), default="all")
parser.add_argument("--viss_qwen_max_candidates", type=int, default=0, help="Maximum Qwen coarse candidates for v28 qwen_first. Use 0 for no limit.")
parser.add_argument("--viss_qwen_max_roi_refine", type=int, default=0, help="Maximum YOLO ROI refinements for v28 qwen_first. Use 0 for no limit.")
parser.add_argument("--viss_qwen_timeout_s", type=float, default=0.0, help="Timeout for the v28/VISS Qwen subprocess. Use 0 or a negative value to wait indefinitely.")
parser.add_argument("--viss_qwen_api_style", choices=("dashscope", "openai"), default="dashscope")
parser.add_argument("--vlm_model", default="glm-5v-turbo")
parser.add_argument("--vlm_endpoint", default="https://open.bigmodel.cn/api/paas/v4/chat/completions")
parser.add_argument("--vlm_timeout_s", type=float, default=30.0)
parser.add_argument("--vlm_retries", type=int, default=1)
parser.add_argument("--vlm_min_conf", type=float, default=0.35)
parser.add_argument("--max_vlm_instances", type=int, default=8)
parser.add_argument("--target_category", default="", help="Optional target category: 可回收物/厨余垃圾/其他垃圾/有害垃圾.")
parser.add_argument("--target_object_name", default="", help="Optional substring match against the VLM object name.")
parser.add_argument("--skip_graspnet", action="store_true", help="Debug only: skip the learned grasp backend and use optional centroid fallback.")
parser.add_argument("--grasp_backend", choices=("graspnet", "anygrasp"), default="graspnet", help="Learned grasp detector backend.")
parser.add_argument("--use_centroid_fallback", action="store_true", help="Use a local point-cloud top-down grasp if the learned backend has no valid candidate.")
parser.add_argument("--rgbd_center_grasp_base", action=argparse.BooleanOptionalAction, default=True, help="Use the verified current-frame RGB-D geometric center grasp as the default non-strict execution base instead of learned grasp poses.")
parser.add_argument("--disable_depth_component_fallback", action="store_true", help="Disable head-depth tabletop component fallback masks.")
parser.add_argument("--strict_model_chain", action="store_true", help="Require the official v28 visual chain and the selected learned grasp backend to all participate for strict success.")
parser.add_argument("--graspnet_repo", default="third_party/graspnet-baseline")
parser.add_argument("--graspnet_checkpoint", default="models/graspnet-rs/checkpoint-rs.tar")
parser.add_argument("--graspnet_num_point", type=int, default=8000)
parser.add_argument("--graspnet_max_draw", type=int, default=80)
parser.add_argument("--graspnet_collision_thresh", type=float, default=0.0, help="GraspNet model-free collision threshold; set <=0 to inspect raw decoded grasps.")
parser.add_argument("--graspnet_voxel_size", type=float, default=0.01, help="Voxel size for GraspNet model-free collision detection.")
parser.add_argument("--graspnet_score_thresh", type=float, default=0.0, help="Optional post-NMS GraspNet score floor; 0 disables score filtering for debugging.")
parser.add_argument("--graspnet_force_objectness_top_k", type=int, default=0, help="Diagnostic only: if GraspNet objectness has zero positives, force the top-K objectness-probability seeds through decoding.")
parser.add_argument("--graspnet_grasp_to_tcp", default="1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1", help="16 comma-separated row-major matrix from GraspNet/GraspGroup frame to Kuavo TCP frame.")
parser.add_argument("--anygrasp_repo", default="third_party/anygrasp_sdk")
parser.add_argument("--anygrasp_checkpoint", default="models/anygrasp/checkpoint_detection.tar")
parser.add_argument("--anygrasp_max_gripper_width", type=float, default=0.12)
parser.add_argument("--anygrasp_gripper_height", type=float, default=0.03)
parser.add_argument("--anygrasp_top_down_grasp", action="store_true")
parser.add_argument("--anygrasp_apply_object_mask", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--anygrasp_dense_grasp", action="store_true")
parser.add_argument("--anygrasp_collision_detection", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--anygrasp_grasp_to_tcp", default="1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1", help="16 comma-separated row-major matrix from AnyGrasp grasp frame to Kuavo TCP frame.")
parser.add_argument("--mask_distance_threshold_m", type=float, default=0.02, help="3D distance tolerance for relaxed target-mask filtering.")
parser.add_argument("--min_filtered_grasps", type=int, default=8, help="Minimum mask-filtered learned grasp candidates before enabling 3D distance relaxation.")
parser.add_argument("--ik_prescreen_top_k", type=int, default=20, help="Number of ranked learned grasp candidates to check before execution.")
parser.add_argument("--ik_prescreen_max_joint_delta_rad", type=float, default=5.3, help="Maximum single-probe IK joint delta before rejecting a ranked candidate.")
parser.add_argument("--grasp_ik_prescreen", action=argparse.BooleanOptionalAction, default=True, help="Run dry-run IK filtering before executing learned grasp candidates. Disable for GraspNet diagnosis when precheck removes all candidates.")
parser.add_argument("--target_reachability_ik_check", action=argparse.BooleanOptionalAction, default=True, help="Gate target selection with a dry-run IK check using only the current right-arm joints.")
parser.add_argument("--target_reachability_position_only", action=argparse.BooleanOptionalAction, default=True, help="For target selection, judge reachability by TCP position only; object/gripper axis alignment is diagnostic, not required.")
parser.add_argument("--target_reachability_center_only", action=argparse.BooleanOptionalAction, default=True, help="For centroid fallback, judge reachability only at the RGB-D geometric center that will actually be grasped.")
parser.add_argument("--target_reachability_probe_steps", type=int, default=120, help="Dry-run IK steps for target-center right-arm reachability checks.")
parser.add_argument("--target_self_occlusion_filter", action=argparse.BooleanOptionalAction, default=True, help="Reject visual grasp targets in the lower image band where the robot body can self-occlude the head camera.")
parser.add_argument("--target_self_occlusion_bottom_fraction", type=float, default=0.14, help="Fraction of image height reserved as the lower self-occlusion guard band.")
parser.add_argument("--target_self_occlusion_max_overlap_fraction", type=float, default=0.02, help="Reject a target when this fraction of its mask overlaps the self-occlusion guard band.")
parser.add_argument("--max_grasp_retries", type=int, default=3, help="Maximum learned grasp candidates to physically attempt before optional fallback.")
parser.add_argument("--pregrasp_error_threshold_m", type=float, default=0.07)
parser.add_argument("--grasp_error_threshold_m", type=float, default=0.015)
parser.add_argument("--lift_error_threshold_m", type=float, default=0.08)
parser.add_argument("--gripper_require_contact_before_lift", action=argparse.BooleanOptionalAction, default=True, help="Require actual gripper closure/contact evidence before starting LIFT.")
parser.add_argument("--gripper_contact_min_width_m", type=float, default=0.008, help="Minimum actual jaw opening after close that counts as contact for thin objects.")
parser.add_argument("--gripper_contact_width_fraction", type=float, default=0.20, help="Candidate-width fraction used with --gripper_contact_min_width_m to decide whether the closed gripper is holding an object.")
parser.add_argument("--gripper_tcp_feedback_correction", action=argparse.BooleanOptionalAction, default=False, help="After a near-miss TCP move, run one residual-compensated position-only correction so the actual finger midpoint/TCP matches the object center.")
parser.add_argument("--gripper_tcp_feedback_gain", type=float, default=1.0, help="Gain for --gripper_tcp_feedback_correction residual compensation.")
parser.add_argument("--top_grasp_shallow_target", action=argparse.BooleanOptionalAction, default=True, help="For RGB-D center grasps, command the TCP to the object top surface minus --top_grasp_depth_m instead of the 3D geometric-center Z.")
parser.add_argument("--top_grasp_depth_m", type=float, default=0.0, help="Depth below the RGB-D point-cloud Z_max used as the shallow top-grasp TCP target. Default 0 avoids open-loop rigid-body interpenetration before contact control is available.")
parser.add_argument("--top_grasp_hover_m", type=float, default=0.12, help="Vertical hover distance above RGB-D point-cloud Z_max before the final straight-down grasp descent.")
parser.add_argument("--rgbd_top_grasp_directionless_envelope", action=argparse.BooleanOptionalAction, default=True, help="Use a fixed top-down, max-open parallel-jaw envelope for RGB-D top grasps instead of aligning the jaw axis to the object's PCA axis.")
parser.add_argument("--rgbd_top_grasp_envelope_jaw_axis", type=parse_xyz, default=(0.0, 1.0, 0.0), help="World-frame jaw-opening axis used by --rgbd_top_grasp_directionless_envelope. Default keeps the max-open gripper aligned laterally while ignoring object yaw.")
parser.add_argument("--rgbd_top_grasp_object_aware_orientation", action=argparse.BooleanOptionalAction, default=True, help="Use the selected RGB-D point cloud PCA to choose the top-grasp jaw opening axis instead of a fixed wrist roll.")
parser.add_argument("--rgbd_top_grasp_enforce_orientation", action=argparse.BooleanOptionalAction, default=True, help="For RGB-D top grasps, keep the object-aware gripper posture through hover and final descent instead of allowing position-only posture drift.")
parser.add_argument("--rgbd_top_grasp_tcp_axis", type=parse_xyz, default=(0.0, 0.0, -1.0), help="Nominal world direction from wrist to TCP for RGB-D top grasps. Default is true top-down to avoid side-sweeping or clipping tabletop objects.")
parser.add_argument("--trash_contact_offset_m", type=float, default=0.004, help="Collision contact_offset applied to runtime trash-object colliders for shallow grasp contact tuning.")
parser.add_argument("--trash_rest_offset_m", type=float, default=0.0, help="Collision rest_offset applied to runtime trash-object colliders. Default 0 forbids deliberate overlap/penetration.")
parser.add_argument("--gripper_contact_offset_m", type=float, default=0.004, help="Collision contact_offset applied to right gripper collision prims when available.")
parser.add_argument("--gripper_rest_offset_m", type=float, default=0.0, help="Collision rest_offset applied to right gripper collision prims. Default 0 forbids deliberate overlap/penetration.")
parser.add_argument("--execute_grasp", action="store_true")
parser.add_argument("--record_debug", action=argparse.BooleanOptionalAction, default=True, help="Keep explicit FSM/debug metadata in each cycle directory.")
parser.add_argument("--debug_force_scene_grasp_object", default="", help="Debug only: bypass perception target selection and create a centroid grasp for this scene object key/name, e.g. trash_07 or trash_battery_0.")
parser.add_argument("--debug_force_scene_grasp_width", type=float, default=0.055, help="Debug only: gripper width used with --debug_force_scene_grasp_object.")
parser.add_argument("--debug_reposition_scene_object", default="", help="Debug only: move this scene object key/name to --debug_reposition_scene_object_pos before the task loop.")
parser.add_argument("--debug_reposition_scene_object_pos", type=parse_xyz, default=(0.70, -0.16, 0.5625), help="World x,y,z center used by --debug_reposition_scene_object.")
parser.add_argument("--debug_reposition_scene_object_yaw", type=float, default=0.0, help="World yaw in radians used by --debug_reposition_scene_object.")
parser.add_argument("--debug_cube_grasp_demo", action="store_true", help="Debug only: spawn a small reachable dynamic cube and force a direct grasp on it, bypassing visual perception.")
parser.add_argument("--debug_cube_rgbd_target", action=argparse.BooleanOptionalAction, default=False, help="With --debug_cube_grasp_demo, execute the grasp from the current head RGB-D scene-guided center instead of the simulator object root. The simulator root is used only for error audit.")
parser.add_argument("--debug_cube_isolated_scene", action=argparse.BooleanOptionalAction, default=False, help="With --debug_cube_grasp_demo, spawn only robot, ground, lights, cameras, and the debug cube; omit table, bins, and all trash objects.")
parser.add_argument("--debug_cube_static", action=argparse.BooleanOptionalAction, default=False, help="Make the debug cube kinematic/static. Useful for IK/TCP pose calibration without object motion.")
parser.add_argument("--debug_cube_pos", type=parse_xyz, default=(0.70, -0.16, 0.5625), help="World x,y,z for the debug cube center. Default is a conservative lock-waist right-arm calibration point.")
parser.add_argument("--debug_cube_size", type=float, default=0.045, help="Edge length in meters for --debug_cube_grasp_demo.")
parser.add_argument("--debug_cube_mass", type=float, default=0.04, help="Mass in kg for --debug_cube_grasp_demo dynamic cube.")
parser.add_argument("--save_legacy_yolo_debug", action="store_true", help="Historical debug only: save the retired full-frame YOLO overlay/json. Disabled by default; v28_original outputs are the official visual records.")
parser.add_argument("--save_legacy_grasp_debug", action="store_true", help="Save old GraspNet/fallback debug overlays and input PLY files. Disabled by default.")
parser.add_argument("--target_pose_debug", action=argparse.BooleanOptionalAction, default=True, help="Compare camera-derived target center against the simulator rigid-object root before grasping.")
parser.add_argument("--max_target_pose_xy_error_m", type=float, default=0.08, help="Maximum camera-derived target XY error against simulator truth before execution is blocked.")
parser.add_argument("--enable_sort_nav", action="store_true", help="After a verified grasp, drive to the hard-coded matching bin and drop.")
parser.add_argument("--dynamic_grasp_standpoint_nav", action=argparse.BooleanOptionalAction, default=True, help="Dynamically compute the pre-grasp table-side standpoint from the selected object's current RGB-D world position.")
parser.add_argument("--dynamic_observe_pose", type=parse_xyz, default=(0.18, 0.0, 0.0), help="Default backed-off observation pose used for full-table RGB-D capture and documentation.")
parser.add_argument("--dynamic_standpoint_fallback_to_preset", action=argparse.BooleanOptionalAction, default=True, help="If dynamic RGB-D standpoint planning has no reachable candidate, fall back to the preset standpoint planner.")
parser.add_argument("--grasp_standpoint_nav", action=argparse.BooleanOptionalAction, default=False, help="Before grasping, choose a preset table-side standpoint from the selected object's RGB-D center, navigate there with Nav2, then re-shoot for grasp planning.")
parser.add_argument("--grasp_standpoint_candidates", default="table_front_left,table_front,table_front_right,table_left_near,table_left_far,table_right_near,table_right_far", help="Comma-separated waypoint names considered by --grasp_standpoint_nav.")
parser.add_argument("--grasp_standpoint_reshoot", action=argparse.BooleanOptionalAction, default=True, help="After pre-grasp standpoint navigation, capture RGB-D again and re-run the same VISS/v28 perception chain before planning the grasp.")
parser.add_argument("--grasp_standpoint_require_geometric_reachable", action=argparse.BooleanOptionalAction, default=True, help="Block pre-grasp execution when no preset standpoint puts the selected object in the right-arm geometric reach window.")
parser.add_argument("--standpoint_nav_only", action="store_true", help="Run perception, compute the dynamic/preset grasp standpoint, navigate there with Nav2, reshoot, and stop without executing a grasp.")
parser.add_argument("--motion_only_sort_demo", action="store_true", help="Pause vision/grasp inference and run only the base navigation/drop state machine.")
parser.add_argument("--mind_sort_demo", action="store_true", help="Run visual multi-object sorting with v28/VISS perception, dynamic standpoints, Nav2, and physical right-gripper pickup by default.")
parser.add_argument("--mind_sort_max_objects", type=int, default=0, help="Maximum objects to process in --mind_sort_demo. 0 means continue until no valid visual target remains.")
parser.add_argument("--mind_sort_settle_steps", type=int, default=120, help="Stationary table-facing hold steps at each grasp standpoint before grasp/pick.")
parser.add_argument("--mind_sort_attach_height_m", type=float, default=0.92, help="World Z height used while the mind-picked object follows the robot.")
parser.add_argument("--mind_sort_attach_forward_m", type=float, default=0.34, help="Robot-local forward offset for the carried object in --mind_sort_demo.")
parser.add_argument("--mind_sort_attach_lateral_m", type=float, default=-0.18, help="Robot-local lateral offset for the carried object in --mind_sort_demo. Negative is right side.")
parser.add_argument("--mind_sort_snap_observe_pose", action=argparse.BooleanOptionalAction, default=False, help="After Nav2 reaches the backed-off observation pose in --mind_sort_demo, optionally snap the simulated base exactly to that pose before RGB-D capture. Default is false to keep the navigation loop continuous without pose jumps.")
parser.add_argument("--mind_sort_allow_depth_component_targets", action=argparse.BooleanOptionalAction, default=True, help="Allow head-depth tabletop components as fallback mind-sort targets after v28/VISS visual targets are exhausted. Visual perception targets still have strict priority.")
parser.add_argument("--mind_sort_simulated_pick", action=argparse.BooleanOptionalAction, default=False, help="Explicit demo-only mode: bypass real gripper grasp by attaching the selected object to the robot for Nav2/bin visualization.")
parser.add_argument("--mind_sort_physical_grasp", action=argparse.BooleanOptionalAction, default=True, help="In --mind_sort_demo, require real right-gripper physical grasp/lift verification before bin navigation. Disable only with --mind_sort_simulated_pick.")
parser.add_argument("--mind_sort_force_stable_grasp_profile", action=argparse.BooleanOptionalAction, default=False, help="In --mind_sort_demo physical grasp mode, force the validated debug-cube local position primitive instead of the cuRobo motion profile. Disabled by default because the 20260624_225603 cuRobo profile produced the smoothest no-collision approach.")
parser.add_argument("--mind_sort_grasp_proposal", choices=("rgbd_center", "graspnet_baseline"), default="rgbd_center", help="Physical grasp candidate source used after v28/VISS selects and verifies the target object.")
parser.add_argument("--mind_sort_graspnet_allow_centroid_fallback", action=argparse.BooleanOptionalAction, default=False, help="When --mind_sort_grasp_proposal graspnet_baseline has no candidate, allow the old RGB-D center fallback. Disabled by default so GraspNet failures stay visible.")
parser.add_argument("--mind_sort_graspnet_save_debug", action=argparse.BooleanOptionalAction, default=True, help="Save GraspNet Baseline input masks, PLY, and candidate overlays under physical_grasp/graspnet_baseline for mainline diagnosis.")
parser.add_argument("--mind_sort_gripper_proximity_assist", action=argparse.BooleanOptionalAction, default=False, help="Demo carry mode. In --mind_sort_physical_grasp mode, if the real gripper reaches the selected object but lift verification fails, count a close gripper as holding the object and continue the sorting loop. Enabled by default in --mind_sort_demo unless explicitly disabled.")
parser.add_argument("--mind_sort_gripper_proximity_assist_max_distance_m", type=float, default=0.02, help="Maximum final TCP-to-target/root distance allowed for gripper-proximity assisted pickup. Default is 0.02 m; set <=0 to disable this proximity gate.")
parser.add_argument("--mind_sort_gripper_proximity_assist_steps", type=int, default=0, help="Animation steps for gripper-proximity assisted pickup. 0 reuses --grasp_steps.")
parser.add_argument("--mind_sort_suction_assist", action=argparse.BooleanOptionalAction, default=False, help="Legacy alias for --mind_sort_gripper_proximity_assist. Kept only for old commands.")
parser.add_argument("--mind_sort_suction_assist_max_distance_m", type=float, default=0.75, help="Legacy suction-assist proximity gate. Ignored unless --mind_sort_suction_assist is explicitly used.")
parser.add_argument("--mind_sort_suction_assist_steps", type=int, default=0, help="Legacy suction-assist animation steps. Ignored unless --mind_sort_suction_assist is explicitly used.")
parser.add_argument("--wrist_refine_grasp", action=argparse.BooleanOptionalAction, default=True, help="Before physical mind-sort grasp, move the wrist above the selected object and use wrist RGB-D to refine the grasp center.")
parser.add_argument("--wrist_refine_roi_radius_px", type=int, default=120, help="Pixel radius around the projected target center used for wrist RGB-D local depth refinement.")
parser.add_argument("--wrist_refine_view_height_m", type=float, default=0.28, help="TCP height above the target center for wrist local-view RGB-D refinement.")
parser.add_argument("--wrist_refine_steps", type=int, default=180, help="IK steps used to move the right wrist/TCP to the local wrist-camera view pose.")
parser.add_argument("--mind_sort_physical_reselect_after_reshoot", action=argparse.BooleanOptionalAction, default=False, help="After table-standpoint reshoot in physical grasp mode, reselect among right-arm IK-reachable VISS targets instead of forcing the originally selected object.")
parser.add_argument("--allow_reshoot_target_fallback_reselect", action=argparse.BooleanOptionalAction, default=False, help="If the selected object is not found in the post-navigation reshoot, allow choosing a different target. Disabled by default to prevent stale/wrong-object grasp coordinates.")
parser.add_argument("--mind_sort_reachability_ik_top_k", type=int, default=3, help="After standpoint reshoot, run expensive right-arm IK reachability only on the top-K targets ranked by current-root geometric right-arm fit.")
parser.add_argument("--mind_sort_grasp_retries", type=int, default=3, help="Maximum physical grasp candidates/attempts per selected mind-sort object.")
parser.add_argument("--mind_sort_grasp_posture_reset_steps", type=int, default=140, help="Before each physical mind-sort grasp, smooth the right arm back to the calibrated natural-down IK seed posture.")
parser.add_argument("--mind_sort_physical_max_standpoint_error_m", type=float, default=0.08, help="Maximum Nav2 final XY error allowed before attempting physical grasp in --mind_sort_physical_grasp mode.")
parser.add_argument("--waypoint_nav_demo", action="store_true", help="Pause vision/grasp inference and navigate between named ground waypoints using wheel-driven Nav2.")
parser.add_argument("--waypoint_route", default="home,table_left", help="Comma-separated waypoint names for --waypoint_nav_demo.")
parser.add_argument("--dynamic_standpoint_nav_demo", action="store_true", help="Pause vision/grasp inference, compute a continuous table-side grasp standpoint for a target point, and navigate to it with Nav2.")
parser.add_argument("--dynamic_target_world_xyz", type=parse_xyz, default=(1.20, 0.0, 0.60), help="Target object center in world/map coordinates for --dynamic_standpoint_nav_demo.")
parser.add_argument("--dynamic_allowed_table_sides", default="front,left,right", help="Comma-separated table sides allowed for dynamic grasp standpoint generation: front,left,right.")
parser.add_argument("--dynamic_ik_diagnostics", action=argparse.BooleanOptionalAction, default=False, help="Run dry-run right-arm IK diagnostics for dynamic grasp standpoint candidates. Default is false because diagnostics temporarily reposition the robot and are not part of continuous navigation execution.")
parser.add_argument("--dynamic_require_ik_reachable", action=argparse.BooleanOptionalAction, default=False, help="Reject dynamic grasp standpoint candidates whose dry-run right-arm IK diagnostic fails.")
parser.add_argument("--dynamic_standpoint_clearance_m", type=float, default=0.42, help="Minimum robot root clearance from the detected table rectangle when generating dynamic standpoints.")
parser.add_argument("--ros_cmd_vel_demo", action="store_true", help="Pause vision/grasp/Nav2 goals and verify that a ROS2 /cmd_vel publisher physically moves the robot in Isaac.")
parser.add_argument("--ros_cmd_vel_demo_steps", type=int, default=240)
parser.add_argument("--ros_cmd_vel_demo_linear_x", type=float, default=0.20)
parser.add_argument("--ros_cmd_vel_demo_linear_y", type=float, default=0.0)
parser.add_argument("--ros_cmd_vel_demo_angular_z", type=float, default=0.0)
parser.add_argument("--ros_cmd_vel_demo_rate_hz", type=float, default=30.0)
parser.add_argument("--ros_cmd_vel_demo_min_translation_m", type=float, default=0.05)
parser.add_argument("--ros_cmd_vel_demo_min_yaw_rad", type=float, default=0.05)
parser.add_argument("--ros_cmd_vel_demo_auto_publish", action=argparse.BooleanOptionalAction, default=True, help="Start a ROS2 test publisher that writes /cmd_vel during --ros_cmd_vel_demo.")
parser.add_argument("--wheel_open_loop_demo", action="store_true", help="Pause vision/grasp inference and run an open-loop wheel-contact movement check.")
parser.add_argument("--wheel_open_loop_steps", type=int, default=360)
parser.add_argument("--wheel_open_loop_linear_speed", type=float, default=0.25)
parser.add_argument("--wheel_open_loop_angular_speed", type=float, default=0.0)
parser.add_argument("--wheel_open_loop_raw_velocity", default="", help="Optional comma-separated raw wheel velocity targets for LF,LB,RF,RB, bypassing cmd_vel kinematics.")
parser.add_argument("--wheel_open_loop_sweep", action="store_true", help="Sweep wheel sign patterns in one Isaac run to identify a usable physical wheel mapping.")
parser.add_argument("--wheel_open_loop_sweep_steps", type=int, default=180)
parser.add_argument("--wheel_open_loop_sweep_speed", type=float, default=2.0)
parser.add_argument("--wheel_open_loop_min_translation_m", type=float, default=0.05)
parser.add_argument("--wheel_open_loop_min_yaw_rad", type=float, default=0.08)
parser.add_argument("--nav_backend", choices=("nav2",), default="nav2", help="Navigation backend for motion-only and sort navigation. Only external ROS2/Nav2 is supported.")
parser.add_argument("--motion_test_category", default="其他垃圾", help="Waste category to route in --motion_only_sort_demo.")
parser.add_argument("--motion_test_all_categories", action="store_true", help="In --motion_only_sort_demo, route all four waste categories once.")
parser.add_argument("--ros2_setup", default=str(WORKSPACE_ROOT / "task_319_garbage_sort/scripts/setup_nav2_user_install.bash"), help="ROS 2 setup.bash used by external Nav2 helper processes.")
parser.add_argument("--ros2_python", default="/usr/bin/python3", help="Python interpreter for ROS 2 helper processes.")
parser.add_argument("--ros2_bridge_host", default="127.0.0.1")
parser.add_argument("--ros2_bridge_port", type=int, default=31970)
parser.add_argument("--ros2_bridge_connect_timeout_s", type=float, default=20.0)
parser.add_argument("--ros2_bridge_exchange_timeout_s", type=float, default=1.0, help="Socket read timeout for one Isaac<->ROS2 Nav2 bridge state/cmd exchange.")
parser.add_argument("--start_nav2_stack", action=argparse.BooleanOptionalAction, default=True, help="Start the bundled minimal Nav2 stack for motion-only/sort navigation.")
parser.add_argument("--nav2_stack_script", default=str(WORKSPACE_ROOT / "task_319_garbage_sort/scripts/ros2_nav2_stack_launcher.py"))
parser.add_argument("--nav2_stack_startup_s", type=float, default=2.0)
parser.add_argument("--nav2_action_name", default="/navigate_to_pose")
parser.add_argument("--nav2_goal_frame", default="map")
parser.add_argument("--nav2_goal_timeout_s", type=float, default=0.0, help="Per-goal Nav2 timeout in seconds. Use 0 or a negative value to wait indefinitely.")
parser.add_argument("--nav2_planner_tolerance", type=float, default=0.12, help="Bundled Nav2 NavFn planner tolerance in meters.")
parser.add_argument("--nav2_xy_goal_tolerance", type=float, default=0.035, help="Bundled Nav2 goal checker XY tolerance in meters.")
parser.add_argument("--nav2_yaw_goal_tolerance", type=float, default=0.07, help="Bundled Nav2 goal checker yaw tolerance in radians.")
parser.add_argument("--nav_publish_synthetic_obstacles", action=argparse.BooleanOptionalAction, default=False, help="Publish coarse synthetic scan obstacles. Disabled by default because the Nav2 static map contains the known table/bin geometry.")
parser.add_argument("--trajectory_steps", type=int, default=520)
parser.add_argument("--grasp_steps", type=int, default=220)
parser.add_argument("--lift_steps", type=int, default=300)
parser.add_argument("--hold_steps", type=int, default=180)
parser.add_argument("--gripper_post_close_hold_steps", type=int, default=80, help="Closed-gripper settle steps after contact before lifting, used to let PhysX contact/friction stabilize.")
parser.add_argument("--drop_steps", type=int, default=120)
parser.add_argument("--max_joint_step", type=float, default=0.010)
parser.add_argument("--pregrasp_offset", type=float, default=0.04)
parser.add_argument("--arm_motion_backend", choices=("local_position_primitive", "legacy_differential_ik", "kuavo_ik", "kuavo_analytic_ik", "curobo_right_arm", "auto"), default="local_position_primitive", help="Right-arm grasp execution backend. local_position_primitive is the stable mainline: staged RGB-D-center TCP position grasp with a fixed angled wrist. curobo_right_arm remains available for collision-aware diagnostics; legacy_differential_ik keeps the old 6D/position DLS path; kuavo_ik uses the external Kuavo IK sidecar; kuavo_analytic_ik uses the official Kuavo AnalyticArmIk solver; auto tries Kuavo IK then local primitive.")
parser.add_argument("--arm_motion_ready_pose", choices=("none", "carry"), default="carry", help="Joint-space pre-shape before Cartesian grasp motion.")
parser.add_argument("--arm_motion_ready_steps", type=int, default=120)
parser.add_argument("--arm_motion_pregrasp_clearance_m", type=float, default=0.14, help="Vertical TCP clearance above the grasp point for the staged position-only primitive.")
parser.add_argument("--arm_motion_min_table_clearance_m", type=float, default=0.10, help="Minimum staged pregrasp TCP height above the table surface.")
parser.add_argument("--arm_motion_converge_extra_steps", type=int, default=120, help="Extra closed-loop IK steps allowed per waypoint if TCP error remains above the threshold.")
parser.add_argument("--arm_motion_converge_chunk_steps", type=int, default=60, help="Chunk size for extra closed-loop IK convergence.")
parser.add_argument("--arm_motion_stop_on_regression", action=argparse.BooleanOptionalAction, default=True, help="Stop extra IK convergence chunks early when TCP error regresses. Disable for long reach motions that need the full convergence budget.")
parser.add_argument("--arm_motion_use_target_center_position", action=argparse.BooleanOptionalAction, default=True, help="For non-legacy position-only execution, command the RGB-D target geometric center instead of the learned grasp translation.")
parser.add_argument("--arm_motion_wrist_orientation", choices=("current", "side_pinch", "top_down", "angled_top_down"), default="angled_top_down", help="Wrist orientation used by the staged primitive. side_pinch keeps the inline gripper horizontal; angled_top_down tilts the inline gripper downward without requiring a strict vertical top grasp.")
parser.add_argument("--arm_motion_enforce_wrist_orientation", action=argparse.BooleanOptionalAction, default=False, help="For local_position_primitive, treat --arm_motion_wrist_orientation as a hard 6D pose constraint. Disabled by default because current trash objects are direction-agnostic and TCP position is the stable grasp objective.")
parser.add_argument("--angled_top_down_tcp_axis", type=parse_xyz, default=(0.65, 0.0, -0.76), help="World direction for the wrist-to-TCP axis when --arm_motion_wrist_orientation angled_top_down is used. This is the inline gripper local +X / wrist local -Z direction.")
parser.add_argument("--whole_body_ik_assist", choices=("off", "waist", "waist_leg"), default="off", help="Experimental IsaacLab DLS whole-body assist for grasping/reachability. off controls only the right arm; waist adds waist pitch/yaw; waist_leg also adds knee/leg joints. Navigation standpoints remain unchanged.")
parser.add_argument("--whole_body_ik_allow_with_auto", action=argparse.BooleanOptionalAction, default=True, help="Allow --whole_body_ik_assist when --arm_motion_backend auto falls back to the local IsaacLab IK primitive.")
parser.add_argument("--whole_body_stance_velocity_limit", type=float, default=0.6, help="Velocity limit for knee/leg actuators when --whole_body_ik_assist waist_leg is enabled.")
parser.add_argument("--whole_body_torso_velocity_limit", type=float, default=0.8, help="Velocity limit for waist actuators when --whole_body_ik_assist is enabled.")
parser.add_argument("--torso_preshape_assist", action=argparse.BooleanOptionalAction, default=False, help="Before right-arm grasp execution, dry-run a bounded waist/stance sample set, move to the best torso pre-shape, then lock it and use the right-arm TCP controller.")
parser.add_argument("--torso_preshape_yaw_samples_deg", type=parse_float_csv, default=parse_float_csv("-25,-15,0,15,25"), help="Comma-separated waist yaw samples in degrees for --torso_preshape_assist.")
parser.add_argument("--torso_preshape_pitch_samples_deg", type=parse_float_csv, default=parse_float_csv("-9,-5,0"), help="Comma-separated waist pitch samples in degrees for --torso_preshape_assist. The URDF lower pitch bound is about -9 degrees.")
parser.add_argument("--torso_preshape_knee_samples_rad", type=parse_float_csv, default=parse_float_csv("0"), help="Comma-separated knee samples in radians for --torso_preshape_assist.")
parser.add_argument("--torso_preshape_leg_samples_rad", type=parse_float_csv, default=parse_float_csv("0"), help="Comma-separated leg samples in radians for --torso_preshape_assist.")
parser.add_argument("--torso_preshape_probe_steps", type=int, default=120, help="Dry-run right-arm-only IK steps per torso sample.")
parser.add_argument("--torso_preshape_move_steps", type=int, default=120, help="Executed smoothing steps to move waist/stance to the selected pre-shape.")
parser.add_argument("--torso_preshape_score_posture_weight", type=float, default=0.01, help="Small penalty on torso/stance posture magnitude when ranking pre-shape samples.")
parser.add_argument("--torso_preshape_apply_error_threshold", type=float, default=0.08, help="Only apply a sampled torso pre-shape when its dry-run TCP error is at or below this threshold in meters; otherwise keep the original posture and run the right-arm TCP controller.")
parser.add_argument("--kuavo_ik_bridge_host", default="127.0.0.1")
parser.add_argument("--kuavo_ik_bridge_port", type=int, default=31975)
parser.add_argument("--kuavo_ik_connect_timeout_s", type=float, default=1.5)
parser.add_argument("--kuavo_ik_auto_start", action=argparse.BooleanOptionalAction, default=False, help="Start the Kuavo IK socket bridge in a sourced ROS1 shell before trying the kuavo_ik backend.")
parser.add_argument("--kuavo_ik_ros_setup", default=str(WORKSPACE_ROOT / "kuavo-ros-opensource/devel_mda319/setup.bash"), help="ROS1 setup.bash used when --kuavo_ik_auto_start is enabled.")
parser.add_argument("--kuavo_ik_python", default="python3")
parser.add_argument("--kuavo_ik_bridge_script", default=str(WORKSPACE_ROOT / "task_319_garbage_sort/scripts/kuavo_ik_socket_bridge.py"))
parser.add_argument("--kuavo_ik_service", default="/ik/two_arm_hand_pose_cmd_srv_muli_refer")
parser.add_argument("--kuavo_ik_frame", type=int, default=2, help="Kuavo IK frame field. 2 is local/base frame; 1 is world/odom frame.")
parser.add_argument("--kuavo_ik_constraint_mode", type=int, default=2, help="External Kuavo IK constraint mode. 2 means position hard + orientation soft.")
parser.add_argument("--kuavo_ik_pos_tol_m", type=float, default=0.006)
parser.add_argument("--kuavo_ik_ori_tol_rad", type=float, default=0.50)
parser.add_argument("--kuavo_ik_pos_cost_weight", type=float, default=80.0)
parser.add_argument("--kuavo_analytic_ik_source", default=str(WORKSPACE_ROOT / "task_319_garbage_sort/scripts/kuavo_analytic_ik_cli.cpp"))
parser.add_argument("--kuavo_analytic_ik_cli", default=str(WORKSPACE_ROOT / "task_319_garbage_sort/build/kuavo_analytic_ik_cli"))
parser.add_argument("--kuavo_analytic_ik_auto_build", action=argparse.BooleanOptionalAction, default=True, help="Build the official Kuavo AnalyticArmIk CLI when the binary is missing or stale.")
parser.add_argument("--kuavo_analytic_ik_timeout_s", type=float, default=1.0)
parser.add_argument("--kuavo_analytic_ik_joint_limit_margin_rad", type=float, default=0.02)
parser.add_argument("--kuavo_analytic_ik_max_fk_error_m", type=float, default=0.03, help="Reject official Kuavo analytic IK outputs whose own FK misses the requested link7 target by more than this distance. Set <=0 to disable.")
parser.add_argument("--kuavo_analytic_ik_tcp_refine", action=argparse.BooleanOptionalAction, default=True, help="Use the official Kuavo analytic IK result as a seed, then refine the gripper TCP position with IsaacLab position-only IK.")
parser.add_argument("--kuavo_analytic_ik_refine_steps", type=int, default=160, help="Initial TCP position-only refinement steps after each Kuavo analytic IK seed segment.")
parser.add_argument("--kuavo_analytic_ik_refine_error_threshold_m", type=float, default=0.03, help="TCP error target used by the IsaacLab position-only refinement after Kuavo analytic IK seed segments.")
parser.add_argument("--kuavo_analytic_ik_grasp_tcp_threshold_m", type=float, default=0.025, help="Stricter TCP error threshold for the final GRASP descent when using the Kuavo analytic IK backend.")
parser.add_argument("--kuavo_analytic_ik_execute_top_k", type=int, default=4, help="Maximum official Kuavo analytic IK pose samples to execute and rank by actual IsaacLab TCP error per stage.")
parser.add_argument("--kuavo_analytic_ik_local_grasp_descent_on_seed_failure", action=argparse.BooleanOptionalAction, default=True, help="If official Kuavo analytic IK has no legal GRASP seed from pregrasp, descend to the gripper TCP target with position-only IK.")
parser.add_argument("--kuavo_analytic_ik_approach_dirs", default="current,+y,diag_down_left,diag_down_right,+x,-x,-y,+z,-z", help="Comma-separated Kuavo analytic IK wrist TCP-offset-axis direction samples. The object is direction-agnostic; each sample still commands the same TCP position.")
parser.add_argument("--kuavo_analytic_ik_roll_samples_rad", default="0,1.0471975512,-1.0471975512,1.5707963268,-1.5707963268,2.0943951024,-2.0943951024,3.1415926536,-3.1415926536", help="Comma-separated wrist roll samples for Kuavo analytic IK around the commanded TCP-offset axis.")
parser.add_argument("--curobo_device", default="cuda:0")
parser.add_argument("--curobo_max_attempts", type=int, default=3)
parser.add_argument("--curobo_timeout_s", type=float, default=8.0)
parser.add_argument("--curobo_enable_graph", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--curobo_plan_table_obstacles", action=argparse.BooleanOptionalAction, default=True, help="Add the known table top and legs as cuRobo cuboid obstacles in the robot base frame.")
parser.add_argument("--curobo_collision_activation_distance_m", type=float, default=0.025)
parser.add_argument("--curobo_rotation_threshold_rad", type=float, default=0.75, help="Default cuRobo pose-goal orientation tolerance used for high-clearance motion where exact gripper axis alignment is not yet safety-critical.")
parser.add_argument("--curobo_final_grasp_rotation_threshold_rad", type=float, default=0.75, help="cuRobo pose-goal orientation tolerance used for the low final grasp approach. Default matches the high-clearance stage so small posture mismatch does not abort the demo; use 0.12 for strict posture diagnostics.")
parser.add_argument("--curobo_joint_limit_clip_rad", type=float, default=0.0, help="Optional cuRobo c-space limit buffer for every active right-arm joint. Default is 0.0 to avoid over-constraining reachable poses.")
parser.add_argument("--curobo_command_min_steps", type=int, default=120, help="Minimum Isaac control steps used to execute each cuRobo segment.")
parser.add_argument("--curobo_final_settle_steps", type=int, default=80, help="Extra Isaac control steps holding the last cuRobo joint waypoint so simulated joints can settle before TCP error is judged.")
parser.add_argument("--curobo_tcp_refine", action=argparse.BooleanOptionalAction, default=True, help="After a cuRobo segment, use a short right-arm position-only IsaacLab IK refinement if TCP tracking error remains high.")
parser.add_argument("--curobo_tcp_refine_steps", type=int, default=80)
parser.add_argument("--curobo_grasp_local_tcp_descent", action=argparse.BooleanOptionalAction, default=True, help="After cuRobo reaches pregrasp, descend vertically from the hover TCP instead of replanning a low table-near segment.")
parser.add_argument("--curobo_grasp_servo_descent", action=argparse.BooleanOptionalAction, default=False, help="Experimental final descent: use IsaacLab TCP position-only closed-loop servo after cuRobo reaches hover. Disabled by default because the free wrist orientation can drift.")
parser.add_argument("--curobo_grasp_servo_stop_distance_m", type=float, default=0.02, help="Stop the final TCP position servo once the gripper TCP is within this distance of the visual object grasp point. Default matches the 2 cm gripper-proximity carry gate.")
parser.add_argument("--curobo_lift_local_tcp_ascent", action=argparse.BooleanOptionalAction, default=True, help="After closing the gripper, lift vertically from the actual grasp TCP pose with a cuRobo Cartesian IK chain instead of free-space replanning.")
parser.add_argument("--curobo_cartesian_descent_waypoint_spacing_m", type=float, default=0.002, help="Maximum TCP Z spacing between cuRobo IK waypoints during the final Cartesian grasp descent.")
parser.add_argument("--curobo_cartesian_descent_min_waypoints", type=int, default=50, help="Minimum number of cuRobo IK target waypoints used for the final Cartesian grasp descent.")
parser.add_argument("--curobo_cartesian_descent_return_seeds", type=int, default=48, help="Number of cuRobo IK seeds returned per Cartesian descent waypoint.")
parser.add_argument("--curobo_cartesian_descent_steps_per_waypoint", type=int, default=24, help="Minimum Isaac control steps per Cartesian descent waypoint; higher values slow the last descent.")
parser.add_argument("--curobo_cartesian_descent_max_joint_step", type=float, default=0.0015, help="Joint target step clamp used only for the final slow Cartesian descent.")
parser.add_argument("--curobo_cartesian_descent_max_xy_error_m", type=float, default=0.015, help="Abort the final descent if actual TCP XY drifts farther than this from the locked hover XY.")
parser.add_argument("--curobo_cartesian_descent_max_waypoint_joint_l2_rad", type=float, default=0.8, help="Reject any Cartesian descent IK waypoint whose nearest valid solution jumps farther than this from the previous waypoint.")
parser.add_argument("--curobo_cartesian_descent_position_only", action=argparse.BooleanOptionalAction, default=True, help="Use cuRobo partial-pose IK for the final vertical descent: constrain TCP XYZ, preserve the hover posture by continuity scoring, and do not require exact wrist axis alignment.")
parser.add_argument("--curobo_grasp_fixed_wrist_pose_fallback", action=argparse.BooleanOptionalAction, default=True, help="If the final grasp descent misses the TCP target, restore pregrasp and descend with 6D pose IK using the pregrasp wrist orientation so the TCP offset stays fixed.")
parser.add_argument("--curobo_use_kuavo_analytic_seed", action=argparse.BooleanOptionalAction, default=True, help="Use the official Kuavo analytic IK solution as the default redundant-posture target, then let cuRobo plan a collision-aware joint-space trajectory to it.")
parser.add_argument("--curobo_position_only_tcp", action=argparse.BooleanOptionalAction, default=True, help="Use cuRobo partial-pose planning for TCP targets: constrain XYZ position, ignore end-effector axis alignment, and regularize to the current joint state to avoid redundant twisted postures.")
parser.add_argument("--curobo_prefer_kuavo_seed_for_position_only", action=argparse.BooleanOptionalAction, default=False, help="When cuRobo is in position-only mode, first try a joint-space plan to the official Kuavo analytic IK posture seed, then fall back to position-only planning if that seed is not executable.")
parser.add_argument("--safe_pregrasp_start", action=argparse.BooleanOptionalAction, default=True, help="Move the right TCP through high-clearance waypoints before PRE_GRASP to avoid sweeping tabletop objects.")
parser.add_argument("--safe_pregrasp_steps", type=int, default=180, help="Steps per high-clearance safety waypoint before PRE_GRASP.")
parser.add_argument("--safe_pregrasp_table_clearance_m", type=float, default=0.18, help="Minimum TCP height above the table surface during the safety waypoint.")
parser.add_argument("--safe_pregrasp_object_clearance_m", type=float, default=0.18, help="Minimum TCP height above the selected grasp point during the safety waypoint.")
parser.add_argument("--safe_pregrasp_error_threshold_m", type=float, default=0.10, help="Maximum TCP error allowed for the high-clearance safety waypoint.")
parser.add_argument("--grasp_abort_if_object_moves_during_approach", action=argparse.BooleanOptionalAction, default=False, help="In simulation, abort before gripper close if the intended rigid object moved during the low approach. Default is false: record the diagnostic but keep the demo tolerant to small contact shifts.")
parser.add_argument("--grasp_object_motion_abort_threshold_m", type=float, default=0.008, help="World-frame object-root displacement threshold for --grasp_abort_if_object_moves_during_approach.")
parser.add_argument("--fallback_position_only_ik", action=argparse.BooleanOptionalAction, default=True, help="Execute centroid fallback grasps by tracking TCP position only; target axes are not treated as constraints.")
parser.add_argument("--pregrasp_use_current_wrist_orientation", action=argparse.BooleanOptionalAction, default=True, help="Keep the current wrist orientation during PRE_GRASP; apply the learned grasp orientation in GRASP.")
parser.add_argument("--learned_grasp_position_only_ik", action=argparse.BooleanOptionalAction, default=False, help="Execute GraspNet/AnyGrasp grasps by tracking TCP position only; learned grasp axes are diagnostic and not treated as IK constraints.")
parser.add_argument("--learned_grasp_use_current_wrist_orientation", action=argparse.BooleanOptionalAction, default=True, help="Use GraspNet/AnyGrasp TCP position and width with the current reachable wrist orientation for GRASP/LIFT when side-pinch orientation is disabled.")
parser.add_argument("--learned_grasp_use_side_pinch_orientation", action=argparse.BooleanOptionalAction, default=True, help="Use a calibrated table-side pinch wrist orientation for learned grasps: TCP offset along world +X, jaw width along world Y.")
parser.add_argument("--gripper_tcp_offset", type=parse_xyz, default=RIGHT_WRIST_INLINE_TCP_OFFSET_M, help="Kuavo right wrist to gripper TCP offset in the right-wrist frame, x,y,z meters. Default matches the inline gripper mount: wrist local -Z.")
parser.add_argument("--gripper_local_tcp_offset", type=parse_xyz, default=RIGHT_GRIPPER_LOCAL_TCP_OFFSET_M, help="Gripper-base-frame offset to the TCP, used by cuRobo extra-link calibration.")
parser.add_argument("--lift_height", type=float, default=0.18)
parser.add_argument("--nav_max_steps_per_waypoint", type=int, default=1800)
parser.add_argument("--nav_position_tolerance", type=float, default=0.08)
parser.add_argument("--nav_yaw_tolerance", type=float, default=0.08)
parser.add_argument("--nav_linear_gain", type=float, default=0.9)
parser.add_argument("--nav_angular_gain", type=float, default=2.4)
parser.add_argument("--nav_max_linear_speed", type=float, default=0.45)
parser.add_argument("--nav_max_angular_speed", type=float, default=0.60)
parser.add_argument("--nav_final_dock", action=argparse.BooleanOptionalAction, default=True, help="After Nav2 succeeds for precision grasp standpoints, run a low-speed wheel correction to the exact target pose.")
parser.add_argument("--nav_final_dock_position_tolerance", type=float, default=0.012)
parser.add_argument("--nav_final_dock_yaw_tolerance", type=float, default=0.025)
parser.add_argument("--nav_final_dock_max_steps", type=int, default=720)
parser.add_argument("--nav_final_dock_max_linear_speed", type=float, default=0.10)
parser.add_argument("--nav_final_dock_max_angular_speed", type=float, default=0.22)
parser.add_argument("--nav_cmd_angular_scale", type=float, default=1.4, help="Scale external Nav2 cmd_vel.angular.z before Isaac execution, then clamp to --nav_max_angular_speed.")
parser.add_argument("--nav_cmd_smoothing", action=argparse.BooleanOptionalAction, default=True, help="Apply Isaac-side acceleration limiting to Nav2 cmd_vel before wheel/root execution.")
parser.add_argument("--nav_cmd_linear_accel_limit", type=float, default=0.45, help="Maximum Isaac-side linear cmd_vel change in m/s^2 when --nav_cmd_smoothing is enabled.")
parser.add_argument("--nav_cmd_angular_accel_limit", type=float, default=0.70, help="Maximum Isaac-side angular cmd_vel change in rad/s^2 when --nav_cmd_smoothing is enabled.")
parser.add_argument("--nav_cmd_filter_alpha", type=float, default=0.85, help="First-order filter target blend after acceleration limiting. 1 disables extra low-pass filtering.")
parser.add_argument("--wheel_radius", type=float, default=0.13035)
parser.add_argument("--track_width", type=float, default=0.52)
parser.add_argument("--nav_actuation_mode", choices=("root_velocity", "wheel"), default="wheel", help="How Isaac applies Nav2 cmd_vel. Nav2 remains the planner/controller in both modes.")
parser.add_argument("--wheel_drive_model", choices=("mecanum45", "differential", "urdf_diagonal"), default="urdf_diagonal", help="Isaac wheel inverse kinematics for /cmd_vel execution.")
parser.add_argument("--mecanum_wheel_sign", type=float, default=-1.0, help="Global sign from base-frame mecanum wheel velocity to URDF joint velocity.")
parser.add_argument("--wheel_velocity_scale", type=float, default=0.35, help="Scale wheel joint velocity targets without changing the chassis cmd_vel used by kinematic coupling.")
parser.add_argument("--wheel_ground_coupling", choices=("kinematic", "kinematic_stable", "contact"), default="kinematic_stable", help="In wheel actuation mode, either emulate the low-level wheel-ground coupling with planar root velocity, use stable integrated kinematic pose, or rely on pure wheel contact physics.")
parser.add_argument("--wheel_root_stabilization", action=argparse.BooleanOptionalAction, default=True, help="In wheel actuation mode, keep writing root roll/pitch/z stabilization each step. This is enabled by default for kinematic_stable to prevent visible chassis shake from wheel contact residuals.")
parser.add_argument("--gui_realtime_playback", action=argparse.BooleanOptionalAction, default=True, help="In GUI mode, pace simulation steps so arm motion is visible.")
parser.add_argument("--gui_playback_rate", type=float, default=1.0, help="GUI playback speed multiplier. 1.0 approximates real time; 2.0 is twice as fast.")
parser.add_argument("--async_model_inference", action=argparse.BooleanOptionalAction, default=True, help="Run YOLO/VLM/grasp inference in a background worker during GUI demos.")
parser.add_argument("--post_action_hold_steps", type=int, default=240, help="Extra GUI-visible simulation steps after a one-cycle grasp before closing.")
parser.add_argument("--record_video", action=argparse.BooleanOptionalAction, default=True, help="Record an external observer-camera MP4 showing the robot/table/grasp motion.")
parser.add_argument("--exit_after_video_saved", action=argparse.BooleanOptionalAction, default=True, help="After a normal run finishes and the observer MP4/manifest have been flushed, exit the Python process directly instead of waiting for slow Isaac/Kit shutdown.")
parser.add_argument("--video_width", type=int, default=1280)
parser.add_argument("--video_height", type=int, default=720)
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--video_sample_stride", type=int, default=4, help="Capture one observer video frame every N simulation steps.")
parser.add_argument("--save_video_frames", action="store_true", help="Also keep individual PNG frames next to the MP4.")
parser.add_argument("--observer_camera_pos", type=parse_xyz, default=(2.15, 4.65, 3.10), help="External video camera position x,y,z.")
parser.add_argument("--observer_camera_target", type=parse_xyz, default=(2.05, 0.0, 0.62), help="External video camera look-at target x,y,z.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if cli_arg_provided("--gripper_local_tcp_offset") and not cli_arg_provided("--gripper_tcp_offset"):
    inline_mount_rot = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    args_cli.gripper_tcp_offset = tuple(
        float(v) for v in (inline_mount_rot @ np.asarray(args_cli.gripper_local_tcp_offset, dtype=np.float32)).tolist()
    )
if args_cli.strict_model_chain:
    incompatible = []
    if args_cli.skip_yolo:
        incompatible.append("--skip_yolo")
    if args_cli.skip_vlm:
        incompatible.append("--skip_vlm")
    if args_cli.skip_graspnet:
        incompatible.append("--skip_graspnet")
    if incompatible:
        parser.error(f"--strict_model_chain cannot be combined with {', '.join(incompatible)}.")
if args_cli.debug_cube_grasp_demo:
    if not cli_arg_provided("--arm_motion_ready_pose"):
        args_cli.arm_motion_ready_pose = "none"
    if not cli_arg_provided("--safe_pregrasp_start") and not cli_arg_provided("--no-safe_pregrasp_start"):
        args_cli.safe_pregrasp_start = False
    if not cli_arg_provided("--arm_motion_converge_extra_steps"):
        args_cli.arm_motion_converge_extra_steps = 800
    if not cli_arg_provided("--arm_motion_converge_chunk_steps"):
        args_cli.arm_motion_converge_chunk_steps = 80
    if not cli_arg_provided("--gripper_tcp_feedback_correction") and not cli_arg_provided("--no-gripper_tcp_feedback_correction"):
        args_cli.gripper_tcp_feedback_correction = True
    if bool(args_cli.debug_cube_rgbd_target):
        if not args_cli.target_object_name:
            args_cli.target_object_name = "debug_cube"
    else:
        args_cli.debug_force_scene_grasp_object = "debug_cube"
    if not cli_arg_provided("--debug_force_scene_grasp_width"):
        args_cli.debug_force_scene_grasp_width = max(0.0, float(args_cli.debug_cube_size) + 0.004)
    else:
        args_cli.debug_force_scene_grasp_width = max(float(args_cli.debug_force_scene_grasp_width), float(args_cli.debug_cube_size))
    args_cli.perception_source = "v28_original"
    args_cli.skip_yolo = True
    args_cli.skip_vlm = True
    args_cli.skip_graspnet = True
    args_cli.use_centroid_fallback = bool(args_cli.debug_cube_rgbd_target)
    args_cli.target_reachability_ik_check = False
    args_cli.grasp_ik_prescreen = False
    args_cli.dynamic_grasp_standpoint_nav = False
if args_cli.standpoint_nav_only:
    args_cli.execute_grasp = False
    args_cli.skip_graspnet = True
    args_cli.use_centroid_fallback = False

if args_cli.strict_model_chain and args_cli.use_centroid_fallback:
    print("[WARN] --use_centroid_fallback is ignored in --strict_model_chain; using learned grasp backend only.", flush=True)
if bool(getattr(args_cli, "rgbd_center_grasp_base", False)) and not args_cli.strict_model_chain and not args_cli.standpoint_nav_only:
    args_cli.use_centroid_fallback = True
if (
    not cli_arg_provided("--target_reachability_probe_steps")
    and (args_cli.execute_grasp or args_cli.strict_model_chain)
):
    args_cli.target_reachability_probe_steps = max(
        int(args_cli.target_reachability_probe_steps),
        int(args_cli.trajectory_steps),
        240,
    )
if args_cli.motion_only_sort_demo:
    args_cli.enable_sort_nav = True
    args_cli.skip_yolo = True
    args_cli.skip_vlm = True
    args_cli.skip_graspnet = True
if args_cli.mind_sort_demo:
    if args_cli.mind_sort_simulated_pick:
        args_cli.mind_sort_physical_grasp = False
    elif not args_cli.mind_sort_physical_grasp:
        raise ValueError("--mind_sort_demo now requires real physical grasp by default. Use --mind_sort_simulated_pick only for explicit navigation-display demos.")
    args_cli.enable_sort_nav = True
    args_cli.execute_grasp = bool(args_cli.mind_sort_physical_grasp)
    if args_cli.mind_sort_physical_grasp:
        if not cli_arg_provided("--mind_sort_gripper_proximity_assist") and not cli_arg_provided("--no-mind_sort_gripper_proximity_assist"):
            args_cli.mind_sort_gripper_proximity_assist = True
        if not cli_arg_provided("--mind_sort_gripper_proximity_assist_max_distance_m"):
            args_cli.mind_sort_gripper_proximity_assist_max_distance_m = 0.02
        mind_sort_grasp_proposal = str(args_cli.mind_sort_grasp_proposal)
        if mind_sort_grasp_proposal == "graspnet_baseline":
            args_cli.grasp_backend = "graspnet"
            args_cli.skip_graspnet = False
            args_cli.rgbd_center_grasp_base = False
            args_cli.use_centroid_fallback = bool(args_cli.mind_sort_graspnet_allow_centroid_fallback)
            if not cli_arg_provided("--arm_motion_backend"):
                args_cli.arm_motion_backend = "curobo_right_arm"
            if not cli_arg_provided("--arm_motion_use_target_center_position") and not cli_arg_provided("--no-arm_motion_use_target_center_position"):
                args_cli.arm_motion_use_target_center_position = False
            if not cli_arg_provided("--safe_pregrasp_start") and not cli_arg_provided("--no-safe_pregrasp_start"):
                args_cli.safe_pregrasp_start = False
            if not cli_arg_provided("--mind_sort_grasp_posture_reset_steps"):
                args_cli.mind_sort_grasp_posture_reset_steps = 0
            if not cli_arg_provided("--graspnet_collision_thresh"):
                args_cli.graspnet_collision_thresh = 0.0
            if not cli_arg_provided("--graspnet_score_thresh"):
                args_cli.graspnet_score_thresh = 0.0
        elif bool(args_cli.mind_sort_force_stable_grasp_profile):
            args_cli.arm_motion_backend = "local_position_primitive"
            args_cli.arm_motion_use_target_center_position = True
            args_cli.arm_motion_wrist_orientation = "angled_top_down"
            args_cli.arm_motion_enforce_wrist_orientation = False
            args_cli.learned_grasp_position_only_ik = True
            args_cli.fallback_position_only_ik = True
            args_cli.pregrasp_use_current_wrist_orientation = True
            args_cli.learned_grasp_use_current_wrist_orientation = True
            args_cli.whole_body_ik_assist = "off"
            args_cli.torso_preshape_assist = False
            args_cli.skip_graspnet = True
            args_cli.use_centroid_fallback = True
        else:
            args_cli.skip_graspnet = True
            args_cli.use_centroid_fallback = True
            if not cli_arg_provided("--arm_motion_backend"):
                args_cli.arm_motion_backend = "curobo_right_arm"
            if not cli_arg_provided("--arm_motion_use_target_center_position") and not cli_arg_provided("--no-arm_motion_use_target_center_position"):
                args_cli.arm_motion_use_target_center_position = True
            if not cli_arg_provided("--safe_pregrasp_start") and not cli_arg_provided("--no-safe_pregrasp_start"):
                args_cli.safe_pregrasp_start = False
            if not cli_arg_provided("--mind_sort_grasp_posture_reset_steps"):
                args_cli.mind_sort_grasp_posture_reset_steps = 0
            if not cli_arg_provided("--curobo_grasp_local_tcp_descent") and not cli_arg_provided("--no-curobo_grasp_local_tcp_descent"):
                args_cli.curobo_grasp_local_tcp_descent = True
            if not cli_arg_provided("--curobo_tcp_refine") and not cli_arg_provided("--no-curobo_tcp_refine"):
                args_cli.curobo_tcp_refine = False
            if not cli_arg_provided("--curobo_grasp_fixed_wrist_pose_fallback") and not cli_arg_provided("--no-curobo_grasp_fixed_wrist_pose_fallback"):
                args_cli.curobo_grasp_fixed_wrist_pose_fallback = False
            if not cli_arg_provided("--curobo_prefer_kuavo_seed_for_position_only") and not cli_arg_provided("--no-curobo_prefer_kuavo_seed_for_position_only"):
                args_cli.curobo_prefer_kuavo_seed_for_position_only = False
            if not cli_arg_provided("--curobo_position_only_tcp") and not cli_arg_provided("--no-curobo_position_only_tcp"):
                args_cli.curobo_position_only_tcp = False if bool(args_cli.rgbd_top_grasp_directionless_envelope) else True
            if not cli_arg_provided("--rgbd_top_grasp_object_aware_orientation") and not cli_arg_provided("--no-rgbd_top_grasp_object_aware_orientation"):
                args_cli.rgbd_top_grasp_object_aware_orientation = False if bool(args_cli.rgbd_top_grasp_directionless_envelope) else True
            if bool(args_cli.rgbd_top_grasp_directionless_envelope):
                if not cli_arg_provided("--top_grasp_depth_m"):
                    args_cli.top_grasp_depth_m = 0.02
                if not cli_arg_provided("--curobo_rotation_threshold_rad"):
                    args_cli.curobo_rotation_threshold_rad = min(float(args_cli.curobo_rotation_threshold_rad), 0.35)
                if not cli_arg_provided("--curobo_final_grasp_rotation_threshold_rad"):
                    args_cli.curobo_final_grasp_rotation_threshold_rad = min(float(args_cli.curobo_final_grasp_rotation_threshold_rad), 0.35)
            if not cli_arg_provided("--kuavo_analytic_ik_approach_dirs"):
                args_cli.kuavo_analytic_ik_approach_dirs = "current"
            if not cli_arg_provided("--kuavo_analytic_ik_roll_samples_rad"):
                args_cli.kuavo_analytic_ik_roll_samples_rad = "0"
        args_cli.max_grasp_retries = max(1, int(args_cli.mind_sort_grasp_retries))
        args_cli.grasp_ik_prescreen = False
        if not cli_arg_provided("--arm_motion_ready_pose"):
            args_cli.arm_motion_ready_pose = "none"
        if not cli_arg_provided("--arm_motion_pregrasp_clearance_m"):
            args_cli.arm_motion_pregrasp_clearance_m = max(float(args_cli.arm_motion_pregrasp_clearance_m), 0.14)
        if not cli_arg_provided("--arm_motion_min_table_clearance_m"):
            args_cli.arm_motion_min_table_clearance_m = max(float(args_cli.arm_motion_min_table_clearance_m), 0.10)
        if (
            mind_sort_grasp_proposal != "graspnet_baseline"
            and not bool(args_cli.top_grasp_shallow_target)
            and not cli_arg_provided("--safe_pregrasp_start")
            and not cli_arg_provided("--no-safe_pregrasp_start")
        ):
            args_cli.safe_pregrasp_start = True
        if not cli_arg_provided("--safe_pregrasp_steps"):
            args_cli.safe_pregrasp_steps = max(int(args_cli.safe_pregrasp_steps), 220)
        if not cli_arg_provided("--trajectory_steps"):
            args_cli.trajectory_steps = max(int(args_cli.trajectory_steps), 560)
        if not cli_arg_provided("--grasp_steps"):
            args_cli.grasp_steps = max(int(args_cli.grasp_steps), 240)
        if not cli_arg_provided("--lift_steps"):
            args_cli.lift_steps = max(int(args_cli.lift_steps), 320)
        if not cli_arg_provided("--max_joint_step"):
            args_cli.max_joint_step = min(float(args_cli.max_joint_step), 0.010)
        if not cli_arg_provided("--wrist_refine_grasp") and not cli_arg_provided("--no-wrist_refine_grasp"):
            args_cli.wrist_refine_grasp = False
        if not cli_arg_provided("--target_reachability_ik_check") and not cli_arg_provided("--no-target_reachability_ik_check"):
            args_cli.target_reachability_ik_check = False
        if not cli_arg_provided("--arm_motion_converge_extra_steps"):
            args_cli.arm_motion_converge_extra_steps = max(int(args_cli.arm_motion_converge_extra_steps), 800)
        if not cli_arg_provided("--arm_motion_converge_chunk_steps"):
            args_cli.arm_motion_converge_chunk_steps = max(int(args_cli.arm_motion_converge_chunk_steps), 80)
        if not cli_arg_provided("--arm_motion_stop_on_regression") and not cli_arg_provided("--no-arm_motion_stop_on_regression"):
            args_cli.arm_motion_stop_on_regression = False
        if not cli_arg_provided("--target_reachability_probe_steps"):
            args_cli.target_reachability_probe_steps = max(
                int(args_cli.target_reachability_probe_steps),
                int(args_cli.trajectory_steps),
                420,
            )
        if not cli_arg_provided("--gripper_tcp_feedback_correction") and not cli_arg_provided("--no-gripper_tcp_feedback_correction"):
            args_cli.gripper_tcp_feedback_correction = True
    else:
        args_cli.skip_graspnet = True
        args_cli.use_centroid_fallback = False
    args_cli.dynamic_grasp_standpoint_nav = True
    args_cli.grasp_standpoint_nav = True
    args_cli.grasp_standpoint_reshoot = True
if args_cli.waypoint_nav_demo or args_cli.dynamic_standpoint_nav_demo or args_cli.wheel_open_loop_demo or args_cli.ros_cmd_vel_demo:
    args_cli.enable_sort_nav = True
    args_cli.skip_yolo = True
    args_cli.skip_vlm = True
    args_cli.skip_graspnet = True
    args_cli.nav_actuation_mode = "wheel"
if args_cli.dynamic_grasp_standpoint_nav and not (
    args_cli.motion_only_sort_demo or args_cli.waypoint_nav_demo or args_cli.dynamic_standpoint_nav_demo or args_cli.wheel_open_loop_demo or args_cli.ros_cmd_vel_demo
):
    args_cli.grasp_standpoint_nav = True
if args_cli.grasp_standpoint_nav:
    args_cli.enable_sort_nav = True
if args_cli.dynamic_standpoint_nav_demo:
    if not cli_arg_provided("--wheel_drive_model"):
        args_cli.wheel_drive_model = "urdf_diagonal"
    if not cli_arg_provided("--wheel_ground_coupling"):
        args_cli.wheel_ground_coupling = "kinematic_stable"
    if not cli_arg_provided("--wheel_velocity_scale"):
        args_cli.wheel_velocity_scale = 0.35
if args_cli.enable_sort_nav and not (args_cli.motion_only_sort_demo or args_cli.waypoint_nav_demo or args_cli.dynamic_standpoint_nav_demo or args_cli.wheel_open_loop_demo or args_cli.ros_cmd_vel_demo):
    if not cli_arg_provided("--nav_actuation_mode"):
        args_cli.nav_actuation_mode = "wheel"
    if not cli_arg_provided("--wheel_drive_model"):
        args_cli.wheel_drive_model = "urdf_diagonal"
    if not cli_arg_provided("--wheel_ground_coupling"):
        args_cli.wheel_ground_coupling = "kinematic_stable"
    if not cli_arg_provided("--wheel_velocity_scale"):
        args_cli.wheel_velocity_scale = 0.35
if args_cli.arm_motion_backend != "legacy_differential_ik" and not (
    cli_arg_provided("--learned_grasp_position_only_ik") or cli_arg_provided("--no-learned_grasp_position_only_ik")
):
    args_cli.learned_grasp_position_only_ik = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

try:
    import warp as wp

    wp.init()
except Exception:
    wp = None

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix, subtract_frame_transforms

from task_319_garbage_sort.grasp_pipeline.execution.gripper_control import AttachedParallelGripper
from task_319_garbage_sort.grasp_pipeline.grasping.anygrasp_wrapper import AnyGraspWrapper, calibration_metadata, parse_homogeneous_matrix
from task_319_garbage_sort.grasp_pipeline.grasping.grasp_selector import rank_grasps
from task_319_garbage_sort.grasp_pipeline.grasping.mask_filter import clamp_width, filter_grasps_by_mask_relaxed
from task_319_garbage_sort.grasp_pipeline.grasping.graspnet_wrapper import GraspNetWrapper
from task_319_garbage_sort.grasp_pipeline.perception.depth_utils import masked_depth_center_world
from task_319_garbage_sort.grasp_pipeline.perception.scene_observer import TASK319_OBJECT_BY_NAME
from task_319_garbage_sort.grasp_pipeline.types import GraspCandidates, PerceivedObject, SelectedGrasp
from task_319_garbage_sort.curobo_right_arm import CuroboPlanResult, RIGHT_ARM_JOINT_NAMES, KuavoRightArmCuroboPlanner, build_world_cuboids
from task_319_garbage_sort.gripper_robot_urdf import ensure_kuavo_with_gripper_urdf


_YOLO_MODEL: Any | None = None
_VISS_YOLO_ALL_MODEL: Any | None = None
_GRASPNET_WRAPPER: GraspNetWrapper | None = None
_ANYGRASP_WRAPPER: AnyGraspWrapper | None = None
WASTE_CATEGORIES = ("可回收物", "厨余垃圾", "其他垃圾", "有害垃圾")


@dataclass(slots=True)
class VlmClassification:
    object_name: str
    waste_category: str
    confidence: float
    reason: str
    raw_text: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.waste_category in WASTE_CATEGORIES and self.confidence > 0.0


_VLM_CLASSIFIER: None = None
_VIDEO_RECORDER: "ObserverVideoRecorder | None" = None
_CALIBRATION_DEBUG_MARKERS: dict[str, VisualizationMarkers] = {}
_RECENT_TARGET_FAILURES: dict[str, int] = {}


def gui_playback_tick(sim_dt: float) -> None:
    """Keep GUI runs visually paced while leaving headless validation fast."""

    if getattr(args_cli, "headless", False) or not args_cli.gui_realtime_playback:
        return
    try:
        simulation_app.update()
    except Exception:
        pass
    rate = max(float(args_cli.gui_playback_rate), 1e-3)
    time.sleep(max(0.0, float(sim_dt) / rate))


def wait_for_model_result(label: str, func, *args):
    """Run model inference off the Isaac UI thread during GUI demos."""

    if getattr(args_cli, "headless", False) or not args_cli.async_model_inference:
        return func(*args)
    print(f"[INFO] {label} is running in a background worker; keeping the Isaac viewport responsive.", flush=True)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="task319_model") as executor:
        future = executor.submit(func, *args)
        while not future.done():
            try:
                simulation_app.update()
            except Exception:
                pass
            time.sleep(1.0 / 30.0)
        return future.result()


def record_video_frame(scene: InteractiveScene) -> None:
    if _VIDEO_RECORDER is not None:
        _VIDEO_RECORDER.capture(scene)


ROOT_DIR = WORKSPACE_ROOT
GRIPPER_URDF = ROOT_DIR / "task_319_garbage_sort/two_finger_gripper.urdf"
KUAVO_BASE_URDF = ROOT_DIR / "kuavo-ros-opensource/src/kuavo_assets/models/biped_s62/urdf/biped_s62.urdf"
KUAVO_URDF = ensure_kuavo_with_gripper_urdf(KUAVO_BASE_URDF, GRIPPER_URDF, mount_rpy=RIGHT_GRIPPER_INLINE_MOUNT_RPY)
YCB_AXIS_ALIGNED_DIR = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned"
YCB_AXIS_ALIGNED_PHYSICS_DIR = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics"

TABLE_TOP_CENTER_Z = 0.51
TABLE_TOP_THICKNESS = 0.06
TABLE_SURFACE_Z = TABLE_TOP_CENTER_Z + TABLE_TOP_THICKNESS / 2.0
TRASH_VISUAL_CLEARANCE = 0.002
ROBOT_STABILIZED_BASE_Z = 0.0
LEFT_ARM_JOINT_EXPR = "zarm_l[1-7]_joint"
RIGHT_ARM_JOINT_EXPR = "zarm_r[1-7]_joint"
TORSO_JOINT_EXPR = "waist_.*_joint"
STANCE_JOINT_EXPRS = ["knee_joint", "leg_joint"]
LEFT_EE_BODY = "zarm_l7_end_effector"
RIGHT_EE_BODY = "zarm_r7_end_effector"
RIGHT_ARM_BASE_BODY = "zarm_r1_link"
RIGHT_ARM_LINK7_BODY = "zarm_r7_link"
WHEEL_JOINTS = [
    "wheel_left_front_joint",
    "wheel_left_behind_joint",
    "wheel_right_front_joint",
    "wheel_right_behind_joint",
]
WHEEL_HALF_SPAN_XY = 0.23248871
SQRT_HALF = math.sqrt(0.5)
MECANUM45_WHEEL_GEOMETRY = (
    ((WHEEL_HALF_SPAN_XY, WHEEL_HALF_SPAN_XY), (SQRT_HALF, SQRT_HALF)),
    ((-WHEEL_HALF_SPAN_XY, WHEEL_HALF_SPAN_XY), (-SQRT_HALF, SQRT_HALF)),
    ((WHEEL_HALF_SPAN_XY, -WHEEL_HALF_SPAN_XY), (SQRT_HALF, -SQRT_HALF)),
    ((-WHEEL_HALF_SPAN_XY, -WHEEL_HALF_SPAN_XY), (-SQRT_HALF, -SQRT_HALF)),
)
URDF_DIAGONAL_WHEEL_GEOMETRY = (
    ((WHEEL_HALF_SPAN_XY, WHEEL_HALF_SPAN_XY), 0.785398163),
    ((-WHEEL_HALF_SPAN_XY, WHEEL_HALF_SPAN_XY), 2.356194372),
    ((WHEEL_HALF_SPAN_XY, -WHEEL_HALF_SPAN_XY), -0.785398163),
    ((-WHEEL_HALF_SPAN_XY, -WHEEL_HALF_SPAN_XY), -2.356194372),
)
_KINEMATIC_STABLE_POSE_XYYAW: list[tuple[float, float, float]] | None = None
_NAV_SMOOTHED_CMD: tuple[float, float] = (0.0, 0.0)
STABLE_GRASP_MOTION_PROFILE = "debug_cube_local_position_primitive_v1"
ROT_X_90 = (0.70710678, 0.70710678, 0.0, 0.0)


def whole_body_ik_assist_enabled_for_current_backend() -> bool:
    mode = str(getattr(args_cli, "whole_body_ik_assist", "off"))
    if mode == "off":
        return False
    backend = str(getattr(args_cli, "arm_motion_backend", "local_position_primitive"))
    return backend in {"local_position_primitive", "legacy_differential_ik"} or (
        backend == "auto" and bool(getattr(args_cli, "whole_body_ik_allow_with_auto", True))
    )


def isaaclab_grasp_ik_joint_exprs() -> list[str]:
    """Joint set used by IsaacLab DLS IK for TCP reachability/grasp execution."""
    mode = str(getattr(args_cli, "whole_body_ik_assist", "off"))
    if not whole_body_ik_assist_enabled_for_current_backend():
        return [RIGHT_ARM_JOINT_EXPR]
    if mode == "waist":
        return [TORSO_JOINT_EXPR, RIGHT_ARM_JOINT_EXPR]
    if mode == "waist_leg":
        return [*STANCE_JOINT_EXPRS, TORSO_JOINT_EXPR, RIGHT_ARM_JOINT_EXPR]
    return [RIGHT_ARM_JOINT_EXPR]


def isaaclab_grasp_ik_metadata() -> dict[str, Any]:
    return {
        "mode": str(getattr(args_cli, "whole_body_ik_assist", "off")),
        "enabled_for_backend": bool(whole_body_ik_assist_enabled_for_current_backend()),
        "joint_exprs": isaaclab_grasp_ik_joint_exprs(),
        "solver": "IsaacLab DifferentialIKController DLS",
        "scope": "post-navigation grasp/reachability only; Nav2 standpoint calculation is unchanged",
    }


def quat_wxyz_from_matrix(rot: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rot[2, 1] - rot[1, 2]) / scale
        y = (rot[0, 2] - rot[2, 0]) / scale
        z = (rot[1, 0] - rot[0, 1]) / scale
    else:
        diag = np.diag(rot)
        idx = int(np.argmax(diag))
        if idx == 0:
            scale = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / scale
            x = 0.25 * scale
            y = (rot[0, 1] + rot[1, 0]) / scale
            z = (rot[0, 2] + rot[2, 0]) / scale
        elif idx == 1:
            scale = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / scale
            x = (rot[0, 1] + rot[1, 0]) / scale
            y = 0.25 * scale
            z = (rot[1, 2] + rot[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / scale
            x = (rot[0, 2] + rot[2, 0]) / scale
            y = (rot[1, 2] + rot[2, 1]) / scale
            z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1.0e-9)
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
YCB_VISUAL_RUNTIME_MASSES = {
    "trash_banana_0": 0.08,
    "trash_potted_meat_can_0": 0.09,
    "trash_bleach_cleanser_0": 0.18,
    "trash_foam_brick_0": 0.02,
    "trash_mug_0": 0.12,
}
TARGET_PRIORITY = {"有害垃圾": 0, "厨余垃圾": 1, "可回收物": 2, "其他垃圾": 3}
MAX_TARGET_MASK_FRACTION = 0.20
MAX_TARGET_BBOX_AREA_FRACTION = 0.45
MIN_TARGET_MASK_PIXELS = 120
MIN_TARGET_POINT_COUNT = 30
TARGET_SELECTION_SCORE_WEIGHTS = {
    "unreachable_penalty": 1000.0,
    "v28_group_penalty": 40.0,
    "ik_cost": 120.0,
    "grasp_distance_cost": 80.0,
    "graspability_penalty": 100.0,
    "clutter_penalty": 60.0,
    "depth_uncertainty": 40.0,
    "low_confidence_penalty": 20.0,
    "failed_recently_penalty": 10.0,
}
TARGET_CLUTTER_CLEARANCE_M = 0.16
TARGET_DEPTH_GOOD_POINT_COUNT = 450
BACKGROUND_YOLO_LABELS = {
    "bed",
    "chair",
    "couch",
    "dining table",
    "person",
    "table",
}
VISUAL_TARGET_SOURCES = {"v28_original"}
DEPTH_COMPONENT_TARGET_SOURCE = "depth_component"


def dynamic_target_sources() -> set[str]:
    sources = set(VISUAL_TARGET_SOURCES)
    if (
        bool(args_cli.mind_sort_demo)
        and bool(args_cli.mind_sort_allow_depth_component_targets)
        and not bool(args_cli.disable_depth_component_fallback)
    ):
        sources.add(DEPTH_COMPONENT_TARGET_SOURCE)
    return sources


def dynamic_source_priority(source: str) -> int:
    if source in VISUAL_TARGET_SOURCES:
        return 0
    if source == DEPTH_COMPONENT_TARGET_SOURCE:
        return 1
    return 9


QWEN_CATEGORY_TO_TASK319 = {
    "recyclable": "可回收物",
    "kitchen": "厨余垃圾",
    "hazardous": "有害垃圾",
    "other": "其他垃圾",
    "unknown": "其他垃圾",
}
TRASH_SCENE_OBJECTS = [
    ("trash_00", "trash_cracker_box_0"),
    ("trash_01", "trash_sugar_box_0"),
    ("trash_02", "trash_tomato_soup_can_0"),
    ("trash_03", "trash_mustard_bottle_0"),
    ("trash_04", "trash_banana_0"),
    ("trash_05", "trash_potted_meat_can_0"),
    ("trash_06", "trash_bleach_cleanser_0"),
    ("trash_07", "trash_battery_0"),
    ("trash_08", "trash_foam_brick_0"),
    ("trash_09", "trash_mug_0"),
    ("debug_cube", "debug_test_cube"),
]
BIN_Y_BY_CATEGORY = {
    "可回收物": -0.72,
    "可回收垃圾": -0.72,
    "厨余垃圾": -0.24,
    "有害垃圾": 0.24,
    "其他垃圾": 0.72,
}
BIN_NAME_BY_CATEGORY = {
    "可回收物": "recycle",
    "可回收垃圾": "recycle",
    "厨余垃圾": "kitchen",
    "有害垃圾": "hazard",
    "其他垃圾": "other",
}
TABLE_APPROACH_POSE = (0.18, 0.0, 0.0)
DEBUG_CUBE_TRASH_PARK_X = 8.0
TABLE_STANDPOINTS = {
    "front_center": (0.53, 0.0, 0.0),
    "front_left": (0.53, -0.30, 0.0),
    "front_right": (0.53, 0.30, 0.0),
}
TABLE_RETURN_APPROACH_BACKOFF_M = 0.45
SIDE_CORRIDOR_Y = 1.35
TABLE_SIDE_X = 1.80
BIN_DRIVE_X = 3.75
BIN_STAGING_POSE = (BIN_DRIVE_X, 0.0, math.pi)
BIN_DROP_X = 3.03
BIN_DROP_Z = 0.98
BIN_OPENING_X_BOUNDS = (2.80, 3.20)
BIN_OPENING_HALF_Y = 0.20
TABLE_X_LIMITS = (0.92, 2.68)
TABLE_Y_LIMITS = (-0.42, 0.42)
TABLE_CENTER_XY = ((TABLE_X_LIMITS[0] + TABLE_X_LIMITS[1]) * 0.5, (TABLE_Y_LIMITS[0] + TABLE_Y_LIMITS[1]) * 0.5)
TABLE_GRASP_SIDE_CLEARANCE_M = 0.42
TABLE_LEFT_GRASP_Y = TABLE_Y_LIMITS[0] - TABLE_GRASP_SIDE_CLEARANCE_M
TABLE_RIGHT_GRASP_Y = TABLE_Y_LIMITS[1] + TABLE_GRASP_SIDE_CLEARANCE_M
BIN_FACE_X = 3.00
MOTION_SORT_CATEGORIES = ("可回收物", "厨余垃圾", "有害垃圾", "其他垃圾")
WAYPOINT_REGISTRY = {
    "home": {"pose": TABLE_APPROACH_POSE, "role": "home", "description": "initial table-front home pose"},
    "table_front": {"pose": TABLE_STANDPOINTS["front_center"], "role": "table_standpoint", "description": "front table grasp candidate"},
    "table_front_left": {"pose": TABLE_STANDPOINTS["front_left"], "role": "table_standpoint", "description": "front-left table grasp candidate"},
    "table_front_right": {"pose": TABLE_STANDPOINTS["front_right"], "role": "table_standpoint", "description": "front-right table grasp candidate"},
    "table_left": {"pose": (TABLE_SIDE_X, -SIDE_CORRIDOR_Y, math.pi / 2.0), "role": "table_standpoint", "description": "left-side table grasp candidate"},
    "table_left_near": {"pose": (1.24, TABLE_LEFT_GRASP_Y, math.pi / 2.0), "role": "table_standpoint", "description": "left-side near-corner close table grasp candidate"},
    "table_left_far": {"pose": (2.36, TABLE_LEFT_GRASP_Y, math.pi / 2.0), "role": "table_standpoint", "description": "left-side far-corner close table grasp candidate"},
    "table_right": {"pose": (TABLE_SIDE_X, SIDE_CORRIDOR_Y, -math.pi / 2.0), "role": "table_standpoint", "description": "right-side table grasp candidate"},
    "table_right_near": {"pose": (1.24, TABLE_RIGHT_GRASP_Y, -math.pi / 2.0), "role": "table_standpoint", "description": "right-side near-corner close table grasp candidate"},
    "table_right_far": {"pose": (2.36, TABLE_RIGHT_GRASP_Y, -math.pi / 2.0), "role": "table_standpoint", "description": "right-side far-corner close table grasp candidate"},
    "bin_center": {"pose": BIN_STAGING_POSE, "role": "bin_standpoint", "description": "shared bin-side staging pose"},
    "bin_recycle": {"pose": (BIN_DRIVE_X, -0.72, math.pi), "role": "bin_standpoint", "description": "recycle bin standpoint"},
    "bin_kitchen": {"pose": (BIN_DRIVE_X, -0.24, math.pi), "role": "bin_standpoint", "description": "kitchen bin standpoint"},
    "bin_hazard": {"pose": (BIN_DRIVE_X, 0.24, math.pi), "role": "bin_standpoint", "description": "hazard bin standpoint"},
    "bin_other": {"pose": (BIN_DRIVE_X, 0.72, math.pi), "role": "bin_standpoint", "description": "other bin standpoint"},
}
ARM_NATURAL_DOWN_JOINT_POS = {
    "zarm_l1_joint": 0.0,
    "zarm_l2_joint": 0.0,
    "zarm_l3_joint": 0.0,
    "zarm_l4_joint": 0.0,
    "zarm_l5_joint": 0.0,
    "zarm_l6_joint": 0.0,
    "zarm_l7_joint": 0.0,
    "zarm_r1_joint": 0.0,
    "zarm_r2_joint": 0.0,
    "zarm_r3_joint": 0.0,
    # zarm_r4 upper limit is exactly 0.0 in the Kuavo URDF. Keep the rest pose
    # slightly inside the hard limit so controller settling does not trip
    # cuRobo's strict start-state joint-limit check.
    "zarm_r4_joint": -0.02,
    "zarm_r5_joint": 0.0,
    "zarm_r6_joint": 0.0,
    "zarm_r7_joint": 0.0,
}
RIGHT_ARM_CARRY_JOINT_POS = {
    "zarm_r1_joint": -0.95,
    "zarm_r2_joint": 0.0,
    "zarm_r3_joint": 0.0,
    "zarm_r4_joint": -1.15,
    "zarm_r5_joint": 0.0,
    "zarm_r6_joint": 0.0,
    "zarm_r7_joint": 0.15,
}
STATIC_NAV_SCAN_OBSTACLES = (
    (1.80, 0.00, 1.05),   # table top envelope plus safety margin
    (3.00, -0.72, 0.42),
    (3.00, -0.24, 0.42),
    (3.00, 0.24, 0.42),
    (3.00, 0.72, 0.42),
)
GRASP_APPROACH_AXIS_LOCAL = np.array([1.0, 0.0, 0.0], dtype=np.float32)
GRASP_FRAME_CONVENTION = {
    "grasp_group_x_axis": "gripper depth/approach axis",
    "grasp_group_y_axis": "gripper opening width axis",
    "grasp_group_z_axis": "finger height axis",
    "kuavo_wrist_tcp_offset_axis": "-Z from right wrist to fingertip center after inline gripper mount",
    "kuavo_gripper_local_tcp_offset_axis": "+X in the attached gripper_base frame",
    "graspnet_tcp_origin": "GraspNet translation plus grasp depth along local +X before grasp_to_tcp calibration",
    "side_pinch_wrist_frame": "right-wrist local -Z/TCP offset points world +X, local +Y jaw opening points world +Y",
}
RIGHT_ARM_TARGET_Y_BOUNDS = (-0.46, 0.08)
RIGHT_ARM_TARGET_X_BOUNDS = (0.35, 0.56)
RIGHT_ARM_PREFERRED_TARGET_X = 0.50
RIGHT_ARM_PREFERRED_TARGET_Y = -0.16
SCENE_OBJECT_WASTE_CATEGORY = {
    "trash_00": "可回收物",
    "trash_01": "可回收物",
    "trash_02": "可回收物",
    "trash_03": "可回收物",
    "trash_04": "厨余垃圾",
    "trash_05": "厨余垃圾",
    "trash_06": "有害垃圾",
    "trash_07": "有害垃圾",
    "trash_08": "其他垃圾",
    "trash_09": "其他垃圾",
    "debug_cube": "其他垃圾",
}
SCENE_GRASP_PRIORITY = {
    "trash_07": 0,  # hazardous battery block: thicker than the old marker and suitable for pinch tests.
    "trash_08": 1,
    "trash_05": 2,
    "trash_02": 3,
    "trash_09": 4,
    "trash_06": 5,
    "debug_cube": -10,
}
SCENE_TARGET_RADIUS_M = {
    "trash_00": 0.11,
    "trash_01": 0.10,
    "trash_02": 0.08,
    "trash_03": 0.10,
    "trash_04": 0.11,
    "trash_05": 0.08,
    "trash_06": 0.12,
    "trash_07": 0.08,
    "trash_08": 0.08,
    "trash_09": 0.09,
    "debug_cube": 0.035,
}


@dataclass(slots=True)
class YoloInstance:
    index: int
    yolo_label: str
    yolo_confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray
    vlm: VlmClassification | None = None
    center_3d: np.ndarray | None = None
    source: str = "yolo"
    scene_key: str | None = None
    scene_object_name: str | None = None
    component_score: float = 0.0
    right_arm_reachability: dict[str, Any] | None = None
    points_world: np.ndarray | None = None
    rgbd_center_metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class TargetIdentity:
    instance_id: int
    source: str
    yolo_label: str
    object_name: str | None
    waste_category: str | None
    scene_key: str | None
    scene_object_name: str | None
    bbox_xyxy: tuple[float, float, float, float]
    mask_pixels: int
    mask_hash: int
    center_3d: np.ndarray | None


TASK_STATE_NAMES = [
    "REST",
    "BASE_TO_TABLE",
    "STATIC_PERCEPT",
    "PLAN_GRASP",
    "PRE_GRASP",
    "GRASP",
    "LIFT",
    "VERIFY_HOLD",
    "OBSERVE_TABLE_BACKOFF",
    "PERCEIVE_TABLE_OBJECTS",
    "SELECT_NEXT_OBJECT",
    "PLAN_GRASP_STANDPOINT",
    "NAV_TO_TABLE_STANDPOINT",
    "FACE_TABLE_AND_SETTLE",
    "RESHOOT_AT_STANDPOINT",
    "MOVE_WRIST_TO_LOCAL_VIEW",
    "WRIST_RGBD_REFINE_TARGET",
    "PLAN_PHYSICAL_GRASP",
    "GRASP_CLOSE",
    "VERIFY_PHYSICAL_HOLD",
    "LIFT_TO_CARRY_POSE",
    "MIND_PICK_ATTACH",
    "SUCTION_ASSIST_ATTACH",
    "GRIPPER_PROXIMITY_ASSIST",
    "PICK_STUB_OR_HOLD_OBJECT",
    "NAV_TO_BIN_STAGING",
    "SELECT_BIN_BY_CATEGORY",
    "NAV_TO_BIN",
    "MIND_DROP_ALIGN",
    "MIND_DROP_RELEASE",
    "DROP",
    "RETURN_HOME",
    "NEXT_OBJECT",
    "DONE",
    "RECOVER",
    "SAFE_START",
    "TORSO_PRESHAPE",
    "RETURN_TO_BIN_STAGING",
    "RETURN_HOME_APPROACH",
    "RETURN_TO_SIDE_CORRIDOR",
    "RETURN_HOME_ALIGN",
    "RETURN_TO_OBSERVE_BACKOFF",
    "WHEEL_OPEN_LOOP",
    "WAYPOINT_NAV",
]
TASK_STATE_TO_ID = {name: idx for idx, name in enumerate(TASK_STATE_NAMES)}


if wp is not None:

    @wp.kernel
    def _task_fsm_kernel(
        dt: wp.array(dtype=float),
        state: wp.array(dtype=int),
        wait_time: wp.array(dtype=float),
        requested_state: wp.array(dtype=int),
        gripper_state: wp.array(dtype=float),
    ):
        tid = wp.tid()
        requested = requested_state[tid]
        if requested >= 0:
            state[tid] = requested
            wait_time[tid] = 0.0
            requested_state[tid] = -1
        current = state[tid]
        if current == 5 or current == 6 or current == 7 or current == 9 or current == 10 or current == 11 or current == 12:
            gripper_state[tid] = -1.0
        else:
            gripper_state[tid] = 1.0
        wait_time[tid] = wait_time[tid] + dt[tid]


class Task319StateTracker:
    """Small state tracker mirroring IsaacLab's Warp FSM pattern for traceable task execution."""

    def __init__(self, dt: float, device: str | torch.device) -> None:
        self.dt = float(dt)
        self.device = device
        self.trace: list[dict[str, Any]] = []
        self._warp_enabled = wp is not None
        if self._warp_enabled:
            try:
                self._dt = torch.full((1,), self.dt, device=device)
                self._state = torch.zeros((1,), dtype=torch.int32, device=device)
                self._wait = torch.zeros((1,), dtype=torch.float32, device=device)
                self._requested = torch.full((1,), -1, dtype=torch.int32, device=device)
                self._gripper = torch.ones((1,), dtype=torch.float32, device=device)
                self._dt_wp = wp.from_torch(self._dt, wp.float32)
                self._state_wp = wp.from_torch(self._state, wp.int32)
                self._wait_wp = wp.from_torch(self._wait, wp.float32)
                self._requested_wp = wp.from_torch(self._requested, wp.int32)
                self._gripper_wp = wp.from_torch(self._gripper, wp.float32)
            except Exception as exc:
                self._warp_enabled = False
                self.trace.append({"state": "REST", "event": "warp_disabled", "reason": repr(exc)})
        self.state_name = "REST"

    @property
    def warp_enabled(self) -> bool:
        return self._warp_enabled

    def enter(self, state_name: str, **metadata: Any) -> None:
        if state_name not in TASK_STATE_TO_ID:
            raise ValueError(f"Unknown task state: {state_name}")
        self.state_name = state_name
        if self._warp_enabled:
            self._requested[0] = TASK_STATE_TO_ID[state_name]
            self.tick()
        event = {"state": state_name, "event": "enter", **metadata}
        if self._warp_enabled:
            event["wait_time_s"] = float(self._wait[0].detach().cpu())
            event["gripper_command"] = float(self._gripper[0].detach().cpu())
        self.trace.append(event)

    def tick(self) -> None:
        if not self._warp_enabled:
            return
        wp.launch(
            kernel=_task_fsm_kernel,
            dim=1,
            inputs=[self._dt_wp, self._state_wp, self._wait_wp, self._requested_wp, self._gripper_wp],
            device=self.device,
        )

    def event(self, name: str, **metadata: Any) -> None:
        self.trace.append({"state": self.state_name, "event": name, **metadata})


def quat_wxyz_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def yaw_from_quat_wxyz(q: Any) -> float:
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def canonical_category(category: str | None) -> str:
    if not category:
        return "其他垃圾"
    if category in BIN_Y_BY_CATEGORY:
        return "可回收物" if category == "可回收垃圾" else category
    if "有害" in category:
        return "有害垃圾"
    if "厨余" in category or "湿垃圾" in category:
        return "厨余垃圾"
    if "回收" in category:
        return "可回收物"
    return "其他垃圾"


def category_for_scene_key(scene_key: str | None) -> str:
    return SCENE_OBJECT_WASTE_CATEGORY.get(scene_key or "", "其他垃圾")


def scene_object_name_for_key(scene_key: str | None) -> str | None:
    if not scene_key:
        return None
    for key, object_name in TRASH_SCENE_OBJECTS:
        if key == scene_key:
            return object_name
    return None


def scene_key_and_name_for_alias(alias: str | None) -> tuple[str | None, str | None]:
    if not alias:
        return None, None
    needle = alias.casefold()
    for scene_key, object_name in TRASH_SCENE_OBJECTS:
        if needle == scene_key.casefold() or needle == object_name.casefold():
            return scene_key, object_name
        if any(needle == item.casefold() for item in scene_object_aliases(scene_key, object_name)):
            return scene_key, object_name
    return None, None


def scene_target_radius(scene_key: str | None) -> float:
    return SCENE_TARGET_RADIUS_M.get(scene_key or "", 0.09)


def yolo_scene_match_radius(scene_key: str | None) -> float:
    return min(0.08, 0.65 * scene_target_radius(scene_key))


def scene_object_aliases(scene_key: str, object_name: str) -> list[str]:
    spec = TASK319_OBJECT_BY_NAME.get(object_name)
    aliases = [scene_key, object_name]
    if spec is not None:
        aliases.extend([spec.ycb_name, spec.class_name, spec.semantic_class, spec.waste_category])
    return aliases


def explicit_target_scene_match() -> tuple[str, str] | None:
    if not args_cli.target_object_name:
        return None
    needle = args_cli.target_object_name.casefold()
    for scene_key, object_name in TRASH_SCENE_OBJECTS:
        if any(needle in alias.casefold() for alias in scene_object_aliases(scene_key, object_name)):
            return scene_key, object_name
    return None


def scene_object_display_name(object_name: str) -> str:
    spec = TASK319_OBJECT_BY_NAME.get(object_name)
    return spec.class_name if spec is not None else object_name


def reachability_score(center_3d: np.ndarray | None) -> tuple[int, float]:
    if center_3d is None:
        return (1, 1.0)
    dx = float(center_3d[0] - args_cli.robot_pos[0])
    dy = float(center_3d[1] - args_cli.robot_pos[1])
    dz = float(center_3d[2])
    reachable = (
        RIGHT_ARM_TARGET_X_BOUNDS[0] <= dx <= RIGHT_ARM_TARGET_X_BOUNDS[1]
        and RIGHT_ARM_TARGET_Y_BOUNDS[0] <= dy <= RIGHT_ARM_TARGET_Y_BOUNDS[1]
        and TABLE_SURFACE_Z + 0.015 <= dz <= TABLE_SURFACE_Z + 0.35
    )
    preferred_dx = RIGHT_ARM_PREFERRED_TARGET_X
    right_arm_side_penalty = 1.5 * max(0.0, dy - RIGHT_ARM_TARGET_Y_BOUNDS[1])
    return (0 if reachable else 1, abs(dx - preferred_dx) + 0.45 * abs(dy - RIGHT_ARM_PREFERRED_TARGET_Y) + right_arm_side_penalty)


def target_reachability_metadata(center_3d: np.ndarray | None) -> dict[str, Any]:
    if center_3d is None:
        return {
            "method": "not_evaluated",
            "reachable": False,
            "reason": "missing_3d_center",
            "relative_to_robot_m": None,
        }
    center = np.asarray(center_3d, dtype=np.float32)
    dx = float(center[0] - args_cli.robot_pos[0])
    dy = float(center[1] - args_cli.robot_pos[1])
    dz = float(center[2])
    reach_penalty, reach_cost = reachability_score(center)
    return {
        "method": "legacy_geometry_hint",
        "note": "Used only when dry-run IK reachability is disabled or unavailable.",
        "reachable": bool(reach_penalty == 0),
        "reach_cost": float(reach_cost),
        "relative_to_robot_m": {"dx": dx, "dy": dy, "z_world": dz},
        "bounds": {
            "dx_m": list(RIGHT_ARM_TARGET_X_BOUNDS),
            "dy_m": list(RIGHT_ARM_TARGET_Y_BOUNDS),
            "z_world_m": [TABLE_SURFACE_Z + 0.015, TABLE_SURFACE_Z + 0.35],
        },
    }


def instance_reachability_metadata(inst: YoloInstance | None) -> dict[str, Any]:
    if inst is None:
        return target_reachability_metadata(None)
    if inst.right_arm_reachability is not None:
        return inst.right_arm_reachability
    return target_reachability_metadata(inst.center_3d)


def instance_reachability_score(inst: YoloInstance) -> tuple[int, float]:
    reachability = inst.right_arm_reachability
    if reachability is not None:
        reachable = bool(reachability.get("reachable", False))
        cost = float(reachability.get("reach_cost", reachability.get("best_tcp_error_m", 1.0) or 1.0))
        return (0 if reachable else 1, cost)
    return reachability_score(inst.center_3d)


def mask_identity_hash(mask: np.ndarray) -> int:
    mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))
    # A small deterministic hash is enough to catch accidental mask/instance swaps in debug metadata.
    return int(zlib.crc32(mask_u8.tobytes()) & 0xFFFFFFFF)


def freeze_target_identity(target: YoloInstance | None) -> TargetIdentity | None:
    if target is None:
        return None
    return TargetIdentity(
        instance_id=int(target.index),
        source=target.source,
        yolo_label=target.yolo_label,
        object_name=target.vlm.object_name if target.vlm else None,
        waste_category=target.vlm.waste_category if target.vlm else None,
        scene_key=target.scene_key,
        scene_object_name=target.scene_object_name,
        bbox_xyxy=tuple(float(v) for v in target.bbox_xyxy),
        mask_pixels=int(target.mask.sum()),
        mask_hash=mask_identity_hash(target.mask),
        center_3d=np.asarray(target.center_3d, dtype=np.float32).copy() if target.center_3d is not None else None,
    )


def target_identity_metadata(identity: TargetIdentity | None) -> dict[str, Any] | None:
    if identity is None:
        return None
    return {
        "target_instance_id": int(identity.instance_id),
        "target_source": identity.source,
        "target_yolo_label": identity.yolo_label,
        "target_object_name": identity.object_name,
        "target_waste_category": identity.waste_category,
        "target_scene_key": identity.scene_key,
        "target_scene_object_name": identity.scene_object_name,
        "target_bbox_xyxy": list(identity.bbox_xyxy),
        "target_mask_pixels": int(identity.mask_pixels),
        "target_mask_hash": int(identity.mask_hash),
        "target_center_3d_world": identity.center_3d.tolist() if identity.center_3d is not None else None,
    }


def target_matches_identity(target: YoloInstance | None, identity: TargetIdentity | None) -> tuple[bool, str]:
    if identity is None:
        return False, "missing_frozen_target_identity"
    if target is None:
        return False, "missing_target"
    if int(target.index) != int(identity.instance_id):
        return False, f"instance_id_mismatch:{target.index}!={identity.instance_id}"
    if target.source != identity.source:
        return False, f"source_mismatch:{target.source}!={identity.source}"
    if mask_identity_hash(target.mask) != int(identity.mask_hash):
        return False, "mask_hash_mismatch"
    if target.scene_key != identity.scene_key:
        return False, f"scene_key_mismatch:{target.scene_key}!={identity.scene_key}"
    return True, ""


def selected_grasp_matches_identity(selected: SelectedGrasp | None, identity: TargetIdentity | None) -> tuple[bool, str]:
    if identity is None:
        return False, "missing_frozen_target_identity"
    if selected is None or selected.target is None:
        return False, "missing_selected_grasp_target"
    meta = selected.target.metadata or {}
    instance_id = meta.get("target_instance_id", meta.get("instance_index"))
    if instance_id is None:
        return False, "grasp_target_missing_instance_id"
    if int(instance_id) != int(identity.instance_id):
        return False, f"grasp_target_instance_id_mismatch:{instance_id}!={identity.instance_id}"
    if meta.get("target_mask_hash", identity.mask_hash) != identity.mask_hash:
        return False, "grasp_target_mask_hash_mismatch"
    if meta.get("target_scene_key", meta.get("scene_key")) != identity.scene_key:
        return False, f"grasp_target_scene_key_mismatch:{meta.get('target_scene_key', meta.get('scene_key'))}!={identity.scene_key}"
    return True, ""


def is_background_yolo_label(label: str) -> bool:
    normalized = " ".join(str(label).casefold().strip().split())
    return normalized in BACKGROUND_YOLO_LABELS


def require_reachable_auto_target() -> bool:
    if bool(getattr(args_cli, "grasp_standpoint_nav", False)):
        return False
    return bool(args_cli.execute_grasp or args_cli.strict_model_chain)


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
    return [("class", semantic_class)]


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


def collision_props(contact_offset: float = 0.003, rest_offset: float = 0.0) -> sim_utils.CollisionPropertiesCfg:
    rest = max(0.0, float(rest_offset))
    contact = max(float(contact_offset), rest + 1.0e-4)
    return sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=contact, rest_offset=rest)


def grasp_object_contact_offset(default_offset: float, semantic_class: str, *, static: bool = False) -> float:
    if static or semantic_class in {"table", "bin"}:
        return float(default_offset)
    return float(max(float(default_offset), float(args_cli.trash_contact_offset_m)))


def grasp_object_rest_offset(default_offset: float, semantic_class: str, *, static: bool = False) -> float:
    if static or semantic_class in {"table", "bin"}:
        return float(default_offset)
    return float(args_cli.trash_rest_offset_m)


def debug_cube_scene_pos(pos: tuple[float, float, float], slot: int) -> tuple[float, float, float]:
    if not args_cli.debug_cube_grasp_demo:
        return pos
    return (DEBUG_CUBE_TRASH_PARK_X + 0.18 * float(slot), 1.8, pos[2])


def cuboid_rigid(
    name: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
    semantic_class: str,
    *,
    static: bool = False,
    material: sim_utils.RigidBodyMaterialCfg | None = None,
    contact_offset: float = 0.003,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=static_rigid_props() if static else rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=collision_props(
                grasp_object_contact_offset(contact_offset, semantic_class, static=static),
                grasp_object_rest_offset(0.0, semantic_class, static=static),
            ),
            physics_material=material or high_friction_material(),
            visual_material=colored_surface(color),
            semantic_tags=semantic_tags(semantic_class),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def usd_rigid(
    name: str,
    usd_path: Path | str,
    pos: tuple[float, float, float],
    scale: tuple[float, float, float],
    mass: float,
    material: sim_utils.RigidBodyMaterialCfg,
    semantic_class: str,
    *,
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
            collision_props=(
                collision_props(
                    grasp_object_contact_offset(contact_offset, semantic_class),
                    grasp_object_rest_offset(0.0, semantic_class),
                )
                if apply_runtime_physics
                else None
            ),
            semantic_tags=semantic_tags(semantic_class),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def wheel_actuator_cfg() -> ImplicitActuatorCfg:
    if args_cli.enable_sort_nav:
        return ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*_joint"],
            effort_limit_sim=35.0,
            velocity_limit_sim=25.0,
            stiffness=0.0,
            damping=18.0,
        )
    return ImplicitActuatorCfg(
        joint_names_expr=["wheel_.*_joint"],
        effort_limit_sim=350.0,
        velocity_limit_sim=0.05,
        stiffness=5000.0,
        damping=650.0,
    )


KUAVO_HEAD_GRASP_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Kuavo62",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(KUAVO_URDF),
        fix_base=not args_cli.enable_sort_nav,
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
            fix_root_link=not args_cli.enable_sort_nav,
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
            **ARM_NATURAL_DOWN_JOINT_POS,
            "left_finger_joint": 0.055,
            "right_finger_joint": 0.055,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wheels": wheel_actuator_cfg(),
        "locked_stance": ImplicitActuatorCfg(
            joint_names_expr=["knee_joint", "leg_joint"],
            effort_limit_sim=900.0,
            velocity_limit_sim=float(args_cli.whole_body_stance_velocity_limit) if args_cli.whole_body_ik_assist == "waist_leg" else 0.1,
            stiffness=1800.0 if args_cli.whole_body_ik_assist == "waist_leg" else 4500.0,
            damping=220.0 if args_cli.whole_body_ik_assist == "waist_leg" else 520.0,
        ),
        "locked_torso": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim=450.0,
            velocity_limit_sim=float(args_cli.whole_body_torso_velocity_limit) if args_cli.whole_body_ik_assist != "off" else 0.1,
            stiffness=1600.0 if args_cli.whole_body_ik_assist != "off" else 3200.0,
            damping=180.0 if args_cli.whole_body_ik_assist != "off" else 380.0,
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
            effort_limit_sim=220.0,
            velocity_limit_sim=3.0,
            stiffness=1100.0,
            damping=110.0,
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
            effort_limit_sim=70.0,
            velocity_limit_sim=0.18,
            stiffness=1200.0,
            damping=90.0,
        ),
    },
)


@configclass
class HeadCameraGraspSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.CuboidCfg(
            size=(12.0, 12.0, 0.02),
            collision_props=collision_props(0.002),
            physics_material=high_friction_material(),
            visual_material=colored_surface((0.35, 0.36, 0.34)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.82, 0.86, 0.9)))
    robot: ArticulationCfg = KUAVO_HEAD_GRASP_CFG

    table_top = cuboid_rigid("sorting_table_top", (1.80, 0.90, TABLE_TOP_THICKNESS), (1.80, 0.0, TABLE_TOP_CENTER_Z), 60.0, (0.48, 0.36, 0.24), "table", static=True, material=WOOD_MATERIAL)
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

    trash_00 = usd_rigid("trash_cracker_box_0", f"{YCB_AXIS_ALIGNED_PHYSICS_DIR}/003_cracker_box.usd", debug_cube_scene_pos(ycb_table_pos_from_min_y(1.30, -0.31, "003_cracker_box", scale=0.60), 0), uniform_scale(0.60), 0.04, PAPER_MATERIAL, "recyclable_paper", rot=ROT_X_90)
    trash_01 = usd_rigid("trash_sugar_box_0", f"{YCB_AXIS_ALIGNED_PHYSICS_DIR}/004_sugar_box.usd", debug_cube_scene_pos(ycb_table_pos_from_min_y(1.49, -0.19, "004_sugar_box"), 1), uniform_scale(1.0), 0.10, PAPER_MATERIAL, "recyclable_paper", rot=ROT_X_90)
    trash_02 = usd_rigid("trash_tomato_soup_can_0", f"{YCB_AXIS_ALIGNED_PHYSICS_DIR}/005_tomato_soup_can.usd", debug_cube_scene_pos(ycb_table_pos_from_min_y(1.68, -0.33, "005_tomato_soup_can"), 2), uniform_scale(1.0), 0.08, METAL_MATERIAL, "recyclable_metal", rot=ROT_X_90)
    trash_03 = usd_rigid("trash_mustard_bottle_0", f"{YCB_AXIS_ALIGNED_PHYSICS_DIR}/006_mustard_bottle.usd", debug_cube_scene_pos(ycb_table_pos_from_min_y(1.86, -0.20, "006_mustard_bottle"), 3), uniform_scale(1.0), 0.08, PLASTIC_MATERIAL, "recyclable_plastic", rot=ROT_X_90)
    trash_04 = usd_rigid("trash_banana_0", f"{YCB_AXIS_ALIGNED_DIR}/011_banana.usd", debug_cube_scene_pos(ycb_table_pos_from_min_y(2.04, -0.32, "011_banana"), 4), uniform_scale(1.0), 0.08, FOOD_SKIN_MATERIAL, "kitchen_food", rot=ROT_X_90, apply_runtime_physics=False)
    trash_05 = usd_rigid("trash_potted_meat_can_0", f"{YCB_AXIS_ALIGNED_DIR}/010_potted_meat_can.usd", debug_cube_scene_pos(ycb_table_pos_from_min_y(1.04, -0.16, "010_potted_meat_can"), 5), uniform_scale(1.0), 0.09, METAL_MATERIAL, "kitchen_food_residue", rot=ROT_X_90, apply_runtime_physics=False)
    trash_06 = usd_rigid("trash_bleach_cleanser_0", f"{YCB_AXIS_ALIGNED_DIR}/021_bleach_cleanser.usd", debug_cube_scene_pos(ycb_table_pos_from_min_y(1.48, 0.26, "021_bleach_cleanser", scale=0.85), 6), uniform_scale(0.85), 0.18, PLASTIC_MATERIAL, "hazard_chemical", rot=ROT_X_90, apply_runtime_physics=False)
    trash_07 = cuboid_rigid("trash_battery_0", (0.12, 0.045, 0.045), debug_cube_scene_pos((1.06, 0.06, TABLE_SURFACE_Z + TRASH_VISUAL_CLEARANCE + 0.0225), 7), 0.06, (0.08, 0.08, 0.075), "hazard_battery", material=contact_material(1.05, 0.82, 0.01), contact_offset=0.002)
    trash_08 = usd_rigid("trash_foam_brick_0", f"{YCB_AXIS_ALIGNED_DIR}/061_foam_brick.usd", debug_cube_scene_pos(ycb_table_pos_from_min_y(1.10, 0.31, "061_foam_brick"), 8), uniform_scale(1.0), 0.02, PLASTIC_MATERIAL, "other_waste", rot=ROT_X_90, apply_runtime_physics=False)
    trash_09 = usd_rigid("trash_mug_0", f"{YCB_AXIS_ALIGNED_DIR}/025_mug.usd", debug_cube_scene_pos(ycb_table_pos_from_min_y(1.76, 0.11, "025_mug"), 9), uniform_scale(1.0), 0.12, METAL_MATERIAL, "other_waste", rot=ROT_X_90, apply_runtime_physics=False)
    if bool(args_cli.debug_cube_grasp_demo):
        debug_cube = cuboid_rigid(
            "debug_test_cube",
            (float(args_cli.debug_cube_size), float(args_cli.debug_cube_size), float(args_cli.debug_cube_size)),
            tuple(float(v) for v in args_cli.debug_cube_pos),
            float(args_cli.debug_cube_mass),
            (0.95, 0.18, 0.10),
            "debug_grasp_cube",
            static=bool(args_cli.debug_cube_static),
            material=contact_material(1.8, 1.4, 0.0),
            contact_offset=0.002,
        )

    head_rgbd = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo62/head_camera_depth/head_rgbd",
        update_period=1.0 / 30.0,
        width=args_cli.head_camera_width,
        height=args_cli.head_camera_height,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, focus_distance=1.2, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.405309, -0.579417, 0.579417, -0.405309), convention="ros"),
        update_latest_camera_pose=True,
    )

    wrist_rgbd = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo62/zarm_r7_end_effector/right_wrist_rgbd",
        update_period=1.0 / 30.0,
        width=args_cli.wrist_camera_width,
        height=args_cli.wrist_camera_height,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, focus_distance=0.6, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=(0.06, 0.0, 0.02), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        update_latest_camera_pose=True,
    )

    observer_rgb = CameraCfg(
        prim_path="{ENV_REGEX_NS}/observer_rgb",
        update_period=0.0,
        width=args_cli.video_width,
        height=args_cli.video_height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=4.8,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 12.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=args_cli.observer_camera_pos,
            rot=look_at_quat_world(args_cli.observer_camera_pos, args_cli.observer_camera_target),
            convention="world",
        ),
    )


@configclass
class IsolatedDebugCubeGraspSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.CuboidCfg(
            size=(12.0, 12.0, 0.02),
            collision_props=collision_props(0.002),
            physics_material=high_friction_material(),
            visual_material=colored_surface((0.35, 0.36, 0.34)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.82, 0.86, 0.9)))
    robot: ArticulationCfg = KUAVO_HEAD_GRASP_CFG
    calibration_table_top = cuboid_rigid(
        "debug_calibration_table_top",
        (0.55, 0.45, TABLE_TOP_THICKNESS),
        (float(args_cli.debug_cube_pos[0]), float(args_cli.debug_cube_pos[1]), TABLE_TOP_CENTER_Z),
        40.0,
        (0.48, 0.36, 0.24),
        "debug_calibration_table",
        static=True,
        material=WOOD_MATERIAL,
    )
    debug_cube = cuboid_rigid(
        "debug_test_cube",
        (float(args_cli.debug_cube_size), float(args_cli.debug_cube_size), float(args_cli.debug_cube_size)),
        tuple(float(v) for v in args_cli.debug_cube_pos),
        float(args_cli.debug_cube_mass),
        (0.95, 0.18, 0.10),
        "debug_grasp_cube",
        static=bool(args_cli.debug_cube_static),
        material=contact_material(1.8, 1.4, 0.0),
        contact_offset=0.002,
    )
    head_rgbd = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo62/head_camera_depth/head_rgbd",
        update_period=1.0 / 30.0,
        width=args_cli.head_camera_width,
        height=args_cli.head_camera_height,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, focus_distance=1.2, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.405309, -0.579417, 0.579417, -0.405309), convention="ros"),
        update_latest_camera_pose=True,
    )
    wrist_rgbd = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo62/zarm_r7_end_effector/right_wrist_rgbd",
        update_period=1.0 / 30.0,
        width=args_cli.wrist_camera_width,
        height=args_cli.wrist_camera_height,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, focus_distance=0.6, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=(0.06, 0.0, 0.02), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        update_latest_camera_pose=True,
    )
    observer_rgb = CameraCfg(
        prim_path="{ENV_REGEX_NS}/observer_rgb",
        update_period=0.0,
        width=args_cli.video_width,
        height=args_cli.video_height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=4.8,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 12.0),
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
    """Records an external view without changing the head-camera perception path."""

    def __init__(self, run_dir: Path) -> None:
        self.enabled = bool(args_cli.record_video)
        self.run_dir = run_dir
        self.path = run_dir / "external_grasp_demo.mp4"
        self.frame_dir = run_dir / "external_video_frames"
        self.fps = max(float(args_cli.video_fps), 1.0)
        self.sample_stride = max(int(args_cli.video_sample_stride), 1)
        self._step_count = 0
        self.frame_count = 0
        self._writer: Any | None = None
        self._writer_failed = False
        self.error = ""
        if self.enabled and args_cli.save_video_frames:
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, scene: InteractiveScene) -> None:
        if not self.enabled:
            return
        self._step_count += 1
        if self._step_count % self.sample_stride != 0:
            return
        if "observer_rgb" not in scene.keys():
            self.error = "observer_rgb camera is not available in scene."
            return
        try:
            frame = sanitize_rgb(tensor_to_numpy(scene["observer_rgb"].data.output["rgb"])[0])
        except Exception as exc:
            self.error = repr(exc)
            return
        if frame.size == 0:
            self.error = "observer_rgb returned an empty frame."
            return
        self._write_frame(frame)

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
                    self.error = "OpenCV VideoWriter could not open the mp4 output."
            except Exception as exc:
                self._writer = None
                self._writer_failed = True
                self.error = repr(exc)
        if self._writer is not None:
            import cv2  # type: ignore

            self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if args_cli.save_video_frames or self._writer_failed:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(self.frame_dir / f"frame_{self.frame_count:06d}.png")
        self.frame_count += 1

    def finalize(self) -> None:
        if self._writer is not None:
            self._writer.release()
        if not self.enabled:
            return
        manifest = {
            "enabled": True,
            "path": str(self.path),
            "frame_dir": str(self.frame_dir) if (args_cli.save_video_frames or self._writer_failed) else "",
            "frame_count": self.frame_count,
            "fps": self.fps,
            "sample_stride": self.sample_stride,
            "error": self.error,
            "mp4_exists": self.path.exists(),
        }
        (self.run_dir / "video_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        if self.path.exists():
            print(f"[INFO] Observer video saved: {self.path} ({self.frame_count} frames)", flush=True)
        else:
            print(f"[WARN] Observer video was not written. Frames: {self.frame_dir}. Error: {self.error}", flush=True)

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": str(self.path),
            "frame_count": self.frame_count,
            "fps": self.fps,
            "sample_stride": self.sample_stride,
            "error": self.error,
        }


def sanitize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim == 4:
        depth = depth[0]
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def save_depth_images(depth_m: np.ndarray, out_dir: Path, prefix: str = "head_depth") -> None:
    finite = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    depth_mm[finite] = np.clip(depth_m[finite] * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm, mode="I;16").save(out_dir / f"{prefix}_mm.png")
    np.save(out_dir / f"{prefix}_m.npy", depth_m)
    vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if finite.any():
        lo, hi = np.percentile(depth_m[finite], [2.0, 98.0])
        if hi <= lo:
            hi = lo + 1e-3
        norm = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
        norm_u8 = (255.0 * (1.0 - norm)).astype(np.uint8)
        vis[..., 0] = norm_u8
        vis[..., 1] = np.clip(255 - np.abs(norm_u8.astype(np.int16) - 128) * 2, 0, 255).astype(np.uint8)
        vis[..., 2] = 255 - norm_u8
        vis[~finite] = 0
    Image.fromarray(vis).save(out_dir / f"{prefix}_vis.png")


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [(230, 57, 70), (29, 150, 65), (37, 99, 235), (245, 158, 11), (168, 85, 247), (20, 184, 166), (236, 72, 153), (132, 204, 22)]
    return palette[index % len(palette)]


def blend_mask(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.42) -> np.ndarray:
    out = base.copy().astype(np.float32)
    mask = mask.astype(bool)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0


def crop_instance(rgb: np.ndarray, mask: np.ndarray, bbox: tuple[float, float, float, float], margin: int = 16) -> Image.Image:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    x0_i = max(0, int(math.floor(x0)) - margin)
    y0_i = max(0, int(math.floor(y0)) - margin)
    x1_i = min(w, int(math.ceil(x1)) + margin)
    y1_i = min(h, int(math.ceil(y1)) + margin)
    crop = rgb[y0_i:y1_i, x0_i:x1_i].copy()
    crop_mask = mask[y0_i:y1_i, x0_i:x1_i]
    if crop.size == 0:
        return Image.fromarray(rgb)
    background = np.full_like(crop, 32)
    masked = np.where(crop_mask[..., None], crop, background)
    return Image.fromarray(masked)


def get_yolo_model() -> Any:
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO

        _YOLO_MODEL = YOLO(str((WORKSPACE_ROOT / args_cli.yolo_weights).resolve()))
    return _YOLO_MODEL


def get_viss_yolo_all_model() -> Any:
    global _VISS_YOLO_ALL_MODEL
    if _VISS_YOLO_ALL_MODEL is None:
        from ultralytics import YOLO

        _VISS_YOLO_ALL_MODEL = YOLO(str((WORKSPACE_ROOT / args_cli.viss_qwen_model).resolve()))
    return _VISS_YOLO_ALL_MODEL


def save_yolo_all_overlay(rgb: np.ndarray, out_dir: Path) -> dict[str, Any]:
    """Save a full-frame YOLO11-seg recognition overlay for visual debugging only."""
    overlay_path = out_dir / "yolo_all_overlay.png"
    json_path = out_dir / "yolo_all_detections.json"
    height, width = rgb.shape[:2]
    metadata: dict[str, Any] = {
        "enabled": not args_cli.skip_yolo,
        "model": str((WORKSPACE_ROOT / args_cli.viss_qwen_model).resolve()),
        "conf": float(args_cli.viss_qwen_conf),
        "imgsz": int(args_cli.yolo_imgsz),
        "overlay": str(overlay_path),
        "detections": [],
        "error": "",
    }
    if args_cli.skip_yolo:
        Image.fromarray(rgb.copy()).save(overlay_path)
        metadata["error"] = "skip_yolo_enabled"
        json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return metadata
    try:
        result = get_viss_yolo_all_model().predict(rgb, imgsz=args_cli.yolo_imgsz, conf=args_cli.viss_qwen_conf, verbose=False)[0]
        names = result.names or {}
        boxes = result.boxes
        masks = None if result.masks is None or result.masks.data is None else result.masks.data.detach().cpu().numpy()
        overlay = rgb.copy()
        detections: list[dict[str, Any]] = []
        if boxes is not None:
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(cls_id)
                xyxy_arr = box.xyxy[0].detach().cpu().numpy().astype(float)
                x0 = float(max(0.0, min(width - 1.0, xyxy_arr[0])))
                y0 = float(max(0.0, min(height - 1.0, xyxy_arr[1])))
                x1 = float(max(0.0, min(width - 1.0, xyxy_arr[2])))
                y1 = float(max(0.0, min(height - 1.0, xyxy_arr[3])))
                mask_pixels = 0
                if masks is not None and idx < len(masks):
                    mask = resize_mask(masks[idx] > 0.5, (height, width))
                    mask_pixels = int(mask.sum())
                    overlay = blend_mask(overlay, mask, color_for_index(idx), alpha=0.30)
                detections.append({
                    "index": int(idx),
                    "class_id": int(cls_id),
                    "class_name": label,
                    "confidence": conf,
                    "bbox_xyxy": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
                    "mask_pixels": mask_pixels,
                    "bbox_area_fraction": float(max(0.0, (x1 - x0) * (y1 - y0)) / max(1, width * height)),
                })
        pil = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil)
        for det in detections:
            color = color_for_index(int(det["index"]))
            x0, y0, x1, y1 = det["bbox_xyxy"]
            draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
            label = f"{det['index']}:{det['class_name']} {det['confidence']:.2f}"
            text_y = max(0.0, y0 - 16.0)
            draw.rectangle((x0, text_y, min(width - 1.0, x0 + max(80, len(label) * 7)), text_y + 15), fill=color)
            draw.text((x0 + 2, text_y + 1), label, fill=(255, 255, 255))
        pil.save(overlay_path)
        metadata["detections"] = detections
        metadata["count"] = len(detections)
    except Exception as exc:
        Image.fromarray(rgb.copy()).save(overlay_path)
        metadata["error"] = repr(exc)
        metadata["count"] = 0
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"[YOLO_ALL] detections={metadata.get('count', 0)} model={metadata['model']} -> {overlay_path}", flush=True)
    return metadata


def run_yolo(rgb: np.ndarray, out_dir: Path) -> list[YoloInstance]:
    height, width = rgb.shape[:2]
    instances: list[YoloInstance] = []
    overlay = rgb.copy()
    union = np.zeros((height, width), dtype=bool)
    if args_cli.skip_yolo:
        Image.fromarray(overlay).save(out_dir / "yolo_overlay.png")
        Image.fromarray(union.astype(np.uint8) * 255).save(out_dir / "yolo_union_mask.png")
        (out_dir / "yolo_instances.json").write_text("[]")
        return instances

    result = get_yolo_model().predict(rgb, imgsz=args_cli.yolo_imgsz, conf=args_cli.yolo_conf, verbose=False)[0]
    masks = None if result.masks is None or result.masks.data is None else result.masks.data.detach().cpu().numpy()
    boxes = result.boxes
    pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil)
    if boxes is not None and len(boxes) > 0:
        for idx in range(len(boxes)):
            xyxy = tuple(float(x) for x in boxes.xyxy[idx].detach().cpu().numpy().tolist())
            cls_id = int(boxes.cls[idx].item())
            conf = float(boxes.conf[idx].item())
            label = str(result.names.get(cls_id, str(cls_id)))
            mask = np.zeros((height, width), dtype=bool)
            if masks is not None and idx < masks.shape[0]:
                mask = resize_mask(masks[idx] > 0.5, (height, width))
                union |= mask
                overlay = blend_mask(np.array(pil), mask, color_for_index(idx))
                pil = Image.fromarray(overlay)
                draw = ImageDraw.Draw(pil)
            color = color_for_index(idx)
            draw.rectangle(xyxy, outline=color, width=3)
            draw.text((xyxy[0] + 3, max(0.0, xyxy[1] - 14)), f"{idx}:{label} {conf:.2f}", fill=color)
            if mask.any():
                instances.append(YoloInstance(idx, label, conf, xyxy, mask))
    instances.sort(key=lambda item: (item.yolo_confidence, int(item.mask.sum())), reverse=True)
    Image.fromarray(np.array(pil)).save(out_dir / "yolo_overlay.png")
    Image.fromarray(union.astype(np.uint8) * 255).save(out_dir / "yolo_union_mask.png")
    (out_dir / "yolo_instances.json").write_text(
        json.dumps(
            [
                {
                    "index": inst.index,
                    "yolo_label_weak": inst.yolo_label,
                    "yolo_confidence": inst.yolo_confidence,
                    "bbox_xyxy": inst.bbox_xyxy,
                    "mask_pixels": int(inst.mask.sum()),
                }
                for inst in instances
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return instances


def clamp_bbox_xyxy(bbox: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in bbox]
    except Exception:
        return None
    x0 = max(0.0, min(float(width - 1), x0))
    x1 = max(0.0, min(float(width - 1), x1))
    y0 = max(0.0, min(float(height - 1), y0))
    y1 = max(0.0, min(float(height - 1), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def mask_from_viss_detection(det: dict[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    height, width = shape
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    polygon = det.get("polygon")
    if isinstance(polygon, list) and len(polygon) >= 3:
        points: list[tuple[float, float]] = []
        for item in polygon:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    points.append((max(0.0, min(float(width - 1), float(item[0]))), max(0.0, min(float(height - 1), float(item[1])))))
                except Exception:
                    pass
        if len(points) >= 3:
            draw.polygon(points, fill=255)
            mask = np.array(mask_img, dtype=np.uint8) > 0
            if int(mask.sum()) > 0:
                return mask, "polygon"
    bbox = clamp_bbox_xyxy(det.get("bbox_xyxy"), width, height)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        draw.rectangle((x0, y0, x1, y1), fill=255)
        return np.array(mask_img, dtype=np.uint8) > 0, "bbox_fallback"
    return np.zeros((height, width), dtype=bool), "empty"


def qwen_category_to_task319(category: str | None) -> str:
    return QWEN_CATEGORY_TO_TASK319.get(str(category or "unknown").strip().lower(), "其他垃圾")


def v28_instance_group_priority(inst: YoloInstance) -> float:
    if inst.source != "v28_original":
        return 0.0
    return 0.0 if float(inst.component_score) >= 0.9 else 1.0


def v28_detection_group_name(raw_group: str) -> str:
    if raw_group == "detections":
        return "planner"
    if raw_group == "approach_candidates":
        return "approach"
    return str(raw_group or "unknown")


def v28_detection_to_instance(det: dict[str, Any], index: int, rgb_shape: tuple[int, int, int], *, group: str, source: str = "v28_original") -> tuple[YoloInstance | None, dict[str, Any]]:
    inst, record = viss_detection_to_instance(det, index, rgb_shape)
    group_name = v28_detection_group_name(group)
    source_label = "v28"
    record[f"{source_label}_group"] = group_name
    record[f"{source_label}_raw_group"] = group
    if inst is None:
        return None, record
    inst.source = source
    inst.component_score = 1.0 if group_name == "planner" else 0.5
    if inst.vlm is not None:
        inst.vlm.reason = f"{source_label} original {group_name}: {inst.vlm.reason}"
    return inst, record


def draw_v28_instances_overlay(rgb: np.ndarray, instances: list[YoloInstance], out_dir: Path, filename: str) -> None:
    height, width = rgb.shape[:2]
    pil = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(pil)
    for rank, inst in enumerate(instances):
        color = color_for_index(rank)
        overlay = blend_mask(np.array(pil), inst.mask, color, alpha=0.26)
        pil = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil)
        x0, y0, x1, y1 = inst.bbox_xyxy
        draw.rectangle(inst.bbox_xyxy, outline=color, width=3)
        group = "planner" if float(inst.component_score) >= 0.9 else "approach"
        label = f"{rank}:{group} {inst.vlm.object_name if inst.vlm else inst.yolo_label}/{inst.vlm.waste_category if inst.vlm else ''} {inst.yolo_confidence:.2f}"
        text_y = min(max(0.0, y0 + 4), height - 20)
        draw.rectangle((x0, text_y, min(width - 1.0, x0 + max(140, len(label) * 7)), text_y + 16), fill=color)
        draw.text((x0 + 2, text_y + 2), label, fill=(255, 255, 255))
    pil.save(out_dir / filename)


def run_viss_original(rgb: np.ndarray, out_dir: Path, *, source: str) -> list[YoloInstance]:
    image_path = out_dir / "head_rgb.png"
    if not image_path.exists():
        Image.fromarray(rgb).save(image_path)
    source_label = "v28" if source == "v28_original" else "v27"
    source_dir = out_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)
    script_path = (WORKSPACE_ROOT / args_cli.viss_qwen_script).resolve()
    model_path = (WORKSPACE_ROOT / args_cli.viss_qwen_model).resolve()
    result_path = source_dir / "yolo_seg_offline_result.json"
    overlay_path = source_dir / "yolo11_qwen_overlay_latest.jpg"
    stdout_path = source_dir / f"{source}_stdout.log"
    stderr_path = source_dir / f"{source}_stderr.log"
    metadata: dict[str, Any] = {
        "script": str(script_path),
        "model": str(model_path),
        "image": str(image_path),
        "api_style": str(args_cli.viss_qwen_api_style),
        "qwen_api_key_present": bool(os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")),
        "config_policy": f"{source}_defaults_plus_task319_output_paths",
        "source": source,
        "result_json": str(result_path),
        "overlay": str(overlay_path),
        "effective_config": {
            "pipeline": "qwen_first",
            "model": str(model_path),
            "conf": float(args_cli.viss_qwen_conf),
            "roi_expand": float(args_cli.viss_qwen_roi_expand),
            "verify_mode": str(args_cli.viss_qwen_verify_mode),
            "max_qwen_candidates": int(args_cli.viss_qwen_max_candidates),
            "max_roi_refine": int(args_cli.viss_qwen_max_roi_refine),
            "qwen_verify_workers": 4,
        },
        "included_groups_for_grasp": ["detections", "approach_candidates"],
        "excluded_groups_for_grasp": ["rejected_detections"],
    }
    if not script_path.exists() or not model_path.exists():
        metadata["error"] = f"missing_{source}_script_or_model"
        (source_dir / f"{source}_instances.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return []
    script_text = script_path.read_text(encoding="utf-8", errors="ignore")
    supports_qwen_api_style = "--qwen-api-style" in script_text
    supports_output_json = "--output-json" in script_text
    supports_overlay_output = "--overlay-output" in script_text
    supports_vis_mode = "--vis-mode" in script_text
    cmd = [
        sys.executable,
        str(script_path),
        "--pipeline", "qwen_first",
        "--image", str(image_path),
        "--model", str(model_path),
        "--conf", str(args_cli.viss_qwen_conf),
        "--roi-expand", str(args_cli.viss_qwen_roi_expand),
        "--verify-mode", str(args_cli.viss_qwen_verify_mode),
        "--max-qwen-candidates", str(args_cli.viss_qwen_max_candidates),
        "--max-roi-refine", str(args_cli.viss_qwen_max_roi_refine),
        "--qwen-verify-workers", "4",
    ]
    if supports_vis_mode:
        cmd.extend(["--vis-mode", "planner"])
    if supports_qwen_api_style:
        cmd.extend(["--qwen-api-style", str(args_cli.viss_qwen_api_style)])
    if supports_output_json:
        cmd.extend(["--output-json", str(result_path)])
    if supports_overlay_output:
        cmd.extend(["--overlay-output", str(overlay_path)])
    cmd.append("--save-vis")
    env = os.environ.copy()
    viss_home = source_dir / "viss_home"
    env["HOME"] = str(viss_home)
    env.setdefault("TRASHBOT_WS_ROOT", str(source_dir / "workspace"))
    env.setdefault("QWEN_API_STYLE", str(args_cli.viss_qwen_api_style))
    if not env.get("QWEN_API_KEY") and env.get("DASHSCOPE_API_KEY"):
        env["QWEN_API_KEY"] = env["DASHSCOPE_API_KEY"]
    env.setdefault("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    env.setdefault("QWEN_MODEL", "qwen3-vl-flash")
    default_result_path = viss_home / "trashbot_ws" / "data" / "logs" / "yolo_seg_offline_result.json"
    default_overlay_path = viss_home / "trashbot_ws" / "data" / "logs" / "yolo11_qwen_overlay_latest.jpg"
    metadata.update({
        "cli_capabilities": {
            "qwen_api_style": supports_qwen_api_style,
            "output_json": supports_output_json,
            "overlay_output": supports_overlay_output,
            "vis_mode": supports_vis_mode,
        },
        "viss_home": str(viss_home),
        "default_result_json": str(default_result_path),
        "default_overlay": str(default_overlay_path),
    })
    t0 = time.time()
    qwen_timeout = None if float(args_cli.viss_qwen_timeout_s) <= 0.0 else float(args_cli.viss_qwen_timeout_s)
    try:
        proc = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), env=env, capture_output=True, text=True, timeout=qwen_timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        metadata.update({"error": f"{source}_timeout", "elapsed_sec": time.time() - t0, "cmd": cmd})
        (source_dir / f"{source}_instances.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return []
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    metadata.update({"returncode": int(proc.returncode), "elapsed_sec": time.time() - t0, "cmd": cmd})
    actual_result_path = result_path
    actual_overlay_path = overlay_path
    if not result_path.exists() and default_result_path.exists():
        shutil.copy2(default_result_path, result_path)
        actual_result_path = default_result_path
    if not overlay_path.exists() and default_overlay_path.exists():
        shutil.copy2(default_overlay_path, overlay_path)
        actual_overlay_path = default_overlay_path
    metadata.update({
        "actual_result_json": str(actual_result_path),
        "actual_overlay": str(actual_overlay_path),
        "copied_default_result": actual_result_path != result_path,
        "copied_default_overlay": actual_overlay_path != overlay_path,
    })
    if proc.returncode != 0 or not result_path.exists():
        metadata["error"] = f"{source}_failed"
        (source_dir / f"{source}_instances.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return []
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        metadata.update({"error": f"{source}_json_parse_error:{exc!r}"})
        (source_dir / f"{source}_instances.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return []
    executed_pipeline = str(result.get("executed_pipeline") or result.get("pipeline") or "")
    fallback_used = bool(result.get("fallback_used", False))
    if executed_pipeline != "qwen_first" or fallback_used:
        metadata.update({
            "error": f"{source}_not_qwen_first",
            "executed_pipeline": executed_pipeline,
            "fallback_used": fallback_used,
            "fallback_reason": result.get("fallback_reason", ""),
        })
        (source_dir / f"{source}_instances.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return []
    instances: list[YoloInstance] = []
    records: list[dict[str, Any]] = []
    for group in ("detections", "approach_candidates"):
        for det in result.get(group, []) or []:
            if not isinstance(det, dict):
                continue
            inst, record = v28_detection_to_instance(det, len(instances), rgb.shape, group=group, source=source)
            records.append(record)
            if inst is not None:
                instances.append(inst)
                Image.fromarray(inst.mask.astype(np.uint8) * 255).save(source_dir / f"{source_label}_mask_{len(instances)-1:02d}.png")
    rejected_records = []
    for det in result.get("rejected_detections", []) or []:
        if isinstance(det, dict):
            rejected_records.append({
                "candidate_id": det.get("candidate_id"),
                "object_id": det.get("object_id"),
                "raw_class_name": det.get("raw_class_name"),
                "reason": det.get("reason") or det.get("reject_reason"),
                "qwen_bbox_xyxy": det.get("qwen_bbox_xyxy"),
                "bbox_xyxy": det.get("bbox_xyxy"),
                "duplicate_of": det.get("duplicate_of"),
            })
    metadata.update({
        "backend": result.get("backend"),
        "pipeline": result.get("pipeline"),
        "executed_pipeline": executed_pipeline,
        "fallback_used": fallback_used,
        "planner_ready": bool(result.get("planner_ready", False)),
        "num_qwen_candidates": int(result.get("num_qwen_candidates", 0) or 0),
        "num_planner_detections": len(result.get("detections", []) or []),
        "num_approach_candidates": len(result.get("approach_candidates", []) or []),
        "num_rejected_detections": len(result.get("rejected_detections", []) or []),
        "accepted_instance_count": len(instances),
        "detections": records,
        "rejected_detections_debug_only": rejected_records,
    })
    (source_dir / f"{source}_instances.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    draw_v28_instances_overlay(rgb, instances, source_dir, f"{source}_instances_overlay.png")
    print(
        f"[{source.upper()}] pipeline={executed_pipeline} planner={metadata['num_planner_detections']} "
        f"approach={metadata['num_approach_candidates']} instances={len(instances)} -> {result_path}",
        flush=True,
    )
    return instances

def run_v28_original(rgb: np.ndarray, out_dir: Path) -> list[YoloInstance]:
    return run_viss_original(rgb, out_dir, source="v28_original")


def viss_detection_to_instance(det: dict[str, Any], index: int, rgb_shape: tuple[int, int, int]) -> tuple[YoloInstance | None, dict[str, Any]]:
    height, width = rgb_shape[:2]
    bbox = clamp_bbox_xyxy(det.get("bbox_xyxy"), width, height)
    mask, mask_source = mask_from_viss_detection(det, (height, width))
    qwen_object_name = str(det.get("verify_object_name") or det.get("coarse_object_name") or det.get("object_name") or "trash_object")
    meta = {
        "index": int(index),
        "object_id": det.get("object_id"),
        "vlm_object_name": qwen_object_name,
        "yolo_raw_class_name": det.get("raw_class_name"),
        "bbox_xyxy": det.get("bbox_xyxy"),
        "mask_source": mask_source,
        "mask_pixels": int(mask.sum()),
        "vlm_garbage_category": det.get("garbage_category"),
        "target_bin": det.get("target_bin"),
        "confidence": float(det.get("confidence", 0.0) or 0.0),
    }
    if bbox is None or int(mask.sum()) <= 0:
        meta["reject_reason"] = "missing_bbox_or_mask"
        return None, meta
    qwen_category = str(det.get("garbage_category") or "unknown").strip().lower()
    waste_category = qwen_category_to_task319(qwen_category)
    object_name = qwen_object_name
    try:
        confidence = float(det.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    vlm = VlmClassification(
        object_name=object_name,
        waste_category=waste_category,
        confidence=max(0.0, min(1.0, confidence)),
        reason=f"viss qwen_first category={qwen_category} target_bin={det.get('target_bin', 'unknown')}",
        raw_text=json.dumps(det, ensure_ascii=False),
        error="" if qwen_category in {"recyclable", "kitchen", "hazardous", "other"} else "Qwen category unknown.",
    )
    inst = YoloInstance(
        index=int(index),
        yolo_label=object_name,
        yolo_confidence=confidence,
        bbox_xyxy=bbox,
        mask=mask,
        vlm=vlm,
        source="retired_legacy_qwen_adapter",
    )
    meta.update({
        "accepted": True,
        "object_name": object_name,
        "waste_category": waste_category,
        "category_source": "viss_qwen",
        "label_policy": "vlm_output_over_yolo_raw_class",
        "mask_hash": mask_identity_hash(mask),
    })
    return inst, meta


def run_retired_legacy_qwen_adapter(rgb: np.ndarray, out_dir: Path) -> list[YoloInstance]:
    height, width = rgb.shape[:2]
    image_path = out_dir / "head_rgb.png"
    if not image_path.exists():
        Image.fromarray(rgb).save(image_path)
    script_path = (WORKSPACE_ROOT / args_cli.viss_qwen_script).resolve()
    model_path = (WORKSPACE_ROOT / args_cli.viss_qwen_model).resolve()
    result_path = out_dir / "viss_qwen_result.json"
    overlay_path = out_dir / "viss_qwen_overlay.png"
    stdout_path = out_dir / "viss_qwen_stdout.log"
    stderr_path = out_dir / "viss_qwen_stderr.log"
    metadata: dict[str, Any] = {
        "script": str(script_path),
        "model": str(model_path),
        "image": str(image_path),
        "api_style": str(args_cli.viss_qwen_api_style),
        "qwen_api_key_present": bool(os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")),
        "v27_config_policy": "v27_defaults_except_model_output_paths_and_api_style",
        "v27_effective_config": {
            "pipeline": "qwen_first",
            "model": str(model_path),
            "conf": float(args_cli.viss_qwen_conf),
            "roi_expand": float(args_cli.viss_qwen_roi_expand),
            "verify_mode": str(args_cli.viss_qwen_verify_mode),
            "max_qwen_candidates": int(args_cli.viss_qwen_max_candidates),
            "max_roi_refine": int(args_cli.viss_qwen_max_roi_refine),
            "qwen_verify_workers": 4,
        },
        "result_json": str(result_path),
        "overlay": str(overlay_path),
    }
    if not script_path.exists() or not model_path.exists():
        metadata["error"] = "missing_viss_script_or_model"
        (out_dir / "viss_qwen_instances.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return []
    cmd = [
        sys.executable,
        str(script_path),
        "--pipeline", "qwen_first",
        "--image", str(image_path),
        "--model", str(model_path),
        "--conf", str(args_cli.viss_qwen_conf),
        "--roi-expand", str(args_cli.viss_qwen_roi_expand),
        "--verify-mode", str(args_cli.viss_qwen_verify_mode),
        "--max-qwen-candidates", str(args_cli.viss_qwen_max_candidates),
        "--max-roi-refine", str(args_cli.viss_qwen_max_roi_refine),
        "--vis-mode", "planner",
        "--qwen-verify-workers", "4",
        "--qwen-api-style", str(args_cli.viss_qwen_api_style),
        "--output-json", str(result_path),
        "--overlay-output", str(overlay_path),
        "--save-vis",
    ]
    env = os.environ.copy()
    env.setdefault("TRASHBOT_WS_ROOT", str(out_dir / "viss_workspace"))
    env.setdefault("QWEN_API_STYLE", str(args_cli.viss_qwen_api_style))
    t0 = time.time()
    qwen_timeout = None if float(args_cli.viss_qwen_timeout_s) <= 0.0 else float(args_cli.viss_qwen_timeout_s)
    try:
        proc = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), env=env, capture_output=True, text=True, timeout=qwen_timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        metadata.update({"error": "viss_qwen_timeout", "elapsed_sec": time.time() - t0, "cmd": cmd})
        (out_dir / "viss_qwen_instances.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return []
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    metadata.update({"returncode": int(proc.returncode), "elapsed_sec": time.time() - t0, "cmd": cmd})
    if proc.returncode != 0 or not result_path.exists():
        metadata["error"] = "viss_qwen_failed"
        (out_dir / "viss_qwen_instances.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return []
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        metadata.update({"error": f"viss_qwen_json_parse_error:{exc!r}"})
        (out_dir / "viss_qwen_instances.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return []
    detections = result.get("detections", []) if isinstance(result, dict) else []
    instances: list[YoloInstance] = []
    records: list[dict[str, Any]] = []
    for idx, det in enumerate(detections):
        if not isinstance(det, dict):
            continue
        inst, record = viss_detection_to_instance(det, idx, rgb.shape)
        records.append(record)
        if inst is not None:
            instances.append(inst)
            Image.fromarray(inst.mask.astype(np.uint8) * 255).save(out_dir / f"viss_mask_{idx:02d}.png")
    metadata.update({
        "backend": result.get("backend"),
        "pipeline": result.get("pipeline"),
        "planner_ready": bool(result.get("planner_ready", False)),
        "num_detections": int(result.get("num_detections", len(detections)) or 0),
        "accepted_instance_count": len(instances),
        "detections": records,
    })
    (out_dir / "viss_qwen_instances.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    pil = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(pil)
    for idx, inst in enumerate(instances):
        color = color_for_index(idx)
        x0, y0, x1, y1 = inst.bbox_xyxy
        draw.rectangle(inst.bbox_xyxy, outline=color, width=3)
        draw.text((x0 + 3, min(max(0.0, y0 + 4), height - 20)), f"{idx}:{inst.vlm.object_name}/{inst.vlm.waste_category} {inst.vlm.confidence:.2f}", fill=color)
    pil.save(out_dir / "viss_instances_overlay.png")
    print(f"[VISS_QWEN] detections={len(detections)} instances={len(instances)} api_key_present={metadata['qwen_api_key_present']} -> {result_path}", flush=True)
    return instances


def get_vlm_classifier() -> None:
    raise RuntimeError("Legacy GLM VLM classifier is retired. Use VISS v28 Qwen/VLM output only.")


def classify_instances(rgb: np.ndarray, instances: list[YoloInstance], out_dir: Path) -> None:
    pil = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(pil)
    for rank, inst in enumerate(instances):
        crop = crop_instance(rgb, inst.mask, inst.bbox_xyxy)
        crop.save(out_dir / f"vlm_crop_{rank:02d}_yolo_{inst.index:02d}.jpg")
        if args_cli.skip_vlm:
            if inst.vlm is None:
                inst.vlm = VlmClassification(inst.yolo_label, "其他垃圾", inst.yolo_confidence, "Debug fallback from YOLO weak label.")
        else:
            try:
                raise RuntimeError("Legacy GLM VLM classifier is retired. Use VISS v28 Qwen/VLM output only.")
            except Exception as exc:
                inst.vlm = VlmClassification("unknown", "", 0.0, "", error=f"retired_legacy_vlm_unavailable:{exc!r}")
        color = color_for_index(rank)
        x0, y0, x1, y1 = inst.bbox_xyxy
        label = inst.vlm.object_name if inst.vlm else "unknown"
        category = inst.vlm.waste_category if inst.vlm else ""
        conf = inst.vlm.confidence if inst.vlm else 0.0
        if inst.vlm and inst.vlm.error:
            label = "VLM_ERROR"
        draw.rectangle(inst.bbox_xyxy, outline=color, width=3)
        draw.text((x0 + 3, min(max(0.0, y0 + 4), rgb.shape[0] - 20)), f"{rank}:{label}/{category} {conf:.2f}", fill=color)
    pil.save(out_dir / "vlm_overlay.png")
    (out_dir / "vlm_results.json").write_text(
        json.dumps(
            [
                {
                    "rank": rank,
                    "yolo_index": inst.index,
                    "yolo_label_weak": inst.yolo_label,
                    "yolo_confidence": inst.yolo_confidence,
                    "bbox_xyxy": inst.bbox_xyxy,
                    "mask_pixels": int(inst.mask.sum()),
                    "source": inst.source,
                    "scene_key": inst.scene_key,
                    "scene_object_name": inst.scene_object_name,
                    "center_3d": inst.center_3d.tolist() if inst.center_3d is not None else None,
                    "object_name": inst.vlm.object_name if inst.vlm else "unknown",
                    "waste_category": inst.vlm.waste_category if inst.vlm else "",
                    "vlm_confidence": inst.vlm.confidence if inst.vlm else 0.0,
                    "vlm_reason": inst.vlm.reason if inst.vlm else "",
                    "vlm_error": inst.vlm.error if inst.vlm else "not_run",
                }
                for rank, inst in enumerate(instances)
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


def select_vlm_instances(instances: list[YoloInstance], out_dir: Path) -> list[YoloInstance]:
    candidates: list[tuple[tuple[float, ...], YoloInstance, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for inst in instances:
        if inst.source != "yolo":
            continue
        mask_pixels = int(inst.mask.sum())
        image_pixels = int(inst.mask.shape[0] * inst.mask.shape[1])
        x0, y0, x1, y1 = inst.bbox_xyxy
        bbox_area_fraction = max(0.0, (x1 - x0) * (y1 - y0)) / max(1, image_pixels)
        mask_area_fraction = mask_pixels / max(1, image_pixels)
        reach_penalty, reach_cost = instance_reachability_score(inst)
        item = {
            "yolo_index": int(inst.index),
            "yolo_label": inst.yolo_label,
            "yolo_confidence": float(inst.yolo_confidence),
            "mask_pixels": mask_pixels,
            "center_3d_world": inst.center_3d.tolist() if inst.center_3d is not None else None,
            "reachability": instance_reachability_metadata(inst),
            "bbox_area_fraction": float(bbox_area_fraction),
            "mask_area_fraction": float(mask_area_fraction),
        }
        if mask_pixels < MIN_TARGET_MASK_PIXELS:
            item["reject_reason"] = "mask_too_small"
            rejected.append(item)
            continue
        if mask_area_fraction > MAX_TARGET_MASK_FRACTION or bbox_area_fraction > MAX_TARGET_BBOX_AREA_FRACTION:
            item["reject_reason"] = "mask_or_bbox_too_large"
            rejected.append(item)
            continue
        sort_key = (
            float(reach_penalty),
            float(reach_cost),
            -float(inst.yolo_confidence),
            -float(mask_pixels),
        )
        candidates.append((sort_key, inst, item))
    candidates.sort(key=lambda item: item[0])
    selected = [inst for _, inst, _ in candidates[: max(1, int(args_cli.max_vlm_instances))]]
    metadata = {
        "requires_reachable_target": require_reachable_auto_target(),
        "source_policy": "yolo_only",
        "reachability_policy": "rgbd_mask_geometry_plus_right_arm_ik_not_yolo_or_glm",
        "note": "Reachability is used only to prioritize limited GLM calls; GLM still provides category only.",
        "max_vlm_instances": int(args_cli.max_vlm_instances),
        "selected": [item for _, _, item in candidates[: max(1, int(args_cli.max_vlm_instances))]],
        "rejected": rejected,
    }
    (out_dir / "vlm_selection.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    if selected:
        print(
            "[VLM_SELECT] "
            + ", ".join(
                f"yolo={inst.index}:{inst.yolo_label} reach={instance_reachability_metadata(inst).get('reachable')} err={float(instance_reachability_metadata(inst).get('best_tcp_error_m', float('inf'))):.3f}"
                for inst in selected
            ),
            flush=True,
        )
    else:
        print("[VLM_SELECT] no valid retired legacy YOLO instance selected", flush=True)
    return selected


def target_name_matches(inst: YoloInstance) -> bool:
    if not args_cli.target_object_name:
        return True
    needle = args_cli.target_object_name.casefold()
    candidates = [
        inst.yolo_label,
        inst.vlm.object_name if inst.vlm else "",
    ]
    return any(needle in str(value).casefold() for value in candidates)


def target_failure_key(inst: YoloInstance | None) -> str | None:
    if inst is None:
        return None
    if inst.scene_key:
        return f"scene:{inst.scene_key}"
    if inst.scene_object_name:
        return f"scene_object:{inst.scene_object_name}"
    if inst.center_3d is not None:
        center = np.asarray(inst.center_3d, dtype=np.float32)
        rounded = ",".join(f"{float(v):.2f}" for v in center[:3])
        label = inst.vlm.object_name if inst.vlm else inst.yolo_label
        return f"center:{inst.source}:{label}:{rounded}"
    return f"instance:{inst.source}:{inst.index}:{inst.yolo_label}"


def target_point_cloud_metrics(inst: YoloInstance) -> dict[str, Any]:
    points = inst.points_world
    if points is None or int(points.shape[0]) <= 0:
        return {
            "point_count": 0,
            "valid": False,
            "z_range_m": None,
            "z_std_m": None,
            "xy_extent_m": None,
            "reason": "missing_rgbd_points",
        }
    pts = np.asarray(points, dtype=np.float32)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] <= 0:
        return {
            "point_count": 0,
            "valid": False,
            "z_range_m": None,
            "z_std_m": None,
            "xy_extent_m": None,
            "reason": "no_finite_rgbd_points",
        }
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    xy_extent = maxs[:2] - mins[:2]
    return {
        "point_count": int(pts.shape[0]),
        "valid": True,
        "z_range_m": float(maxs[2] - mins[2]),
        "z_std_m": float(np.std(pts[:, 2])),
        "xy_extent_m": [float(v) for v in xy_extent],
        "centroid_world": [float(v) for v in np.mean(pts, axis=0)],
        "median_world": [float(v) for v in np.median(pts, axis=0)],
    }


def target_depth_uncertainty_cost(metrics: dict[str, Any]) -> float:
    point_count = int(metrics.get("point_count", 0) or 0)
    if point_count <= 0:
        return 1.0
    count_cost = max(0.0, (float(TARGET_DEPTH_GOOD_POINT_COUNT) - float(point_count)) / float(TARGET_DEPTH_GOOD_POINT_COUNT))
    z_std = metrics.get("z_std_m")
    z_noise_cost = 0.0 if z_std is None else min(1.0, max(0.0, float(z_std) / 0.04))
    return float(min(1.0, max(count_cost, 0.25 * z_noise_cost)))


def target_graspability_cost(metrics: dict[str, Any]) -> dict[str, Any]:
    point_count = int(metrics.get("point_count", 0) or 0)
    xy_extent = metrics.get("xy_extent_m") or [0.0, 0.0]
    xy_values = [float(v) for v in xy_extent] if isinstance(xy_extent, (list, tuple)) else [0.0, 0.0]
    min_xy = min(xy_values) if xy_values else 0.0
    max_xy = max(xy_values) if xy_values else 0.0
    z_range = metrics.get("z_range_m")
    z_range_f = 0.0 if z_range is None else float(z_range)
    count_cost = max(0.0, (520.0 - float(point_count)) / 520.0)
    flat_cost = max(0.0, (0.020 - z_range_f) / 0.020)
    narrow_cost = max(0.0, (0.035 - min_xy) / 0.035)
    oversized_cost = max(0.0, (max_xy - 0.22) / 0.10)
    cost = min(1.0, max(count_cost, flat_cost, narrow_cost, oversized_cost))
    return {
        "cost": float(cost),
        "point_count": point_count,
        "z_range_m": float(z_range_f),
        "min_xy_extent_m": float(min_xy),
        "max_xy_extent_m": float(max_xy),
        "count_cost": float(min(1.0, count_cost)),
        "flat_cost": float(min(1.0, flat_cost)),
        "narrow_cost": float(min(1.0, narrow_cost)),
        "oversized_cost": float(min(1.0, oversized_cost)),
        "policy": "prefer_stable_3d_rgbd_components_that_fit_parallel_gripper",
    }


def bbox_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def target_clutter_metrics(inst: YoloInstance, instances: list[YoloInstance]) -> dict[str, Any]:
    center = inst.center_3d
    nearest_xy = float("inf")
    nearest_index: int | None = None
    max_bbox_iou = 0.0
    for other in instances:
        if other is inst or other.source not in VISUAL_TARGET_SOURCES:
            continue
        max_bbox_iou = max(max_bbox_iou, bbox_iou_xyxy(inst.bbox_xyxy, other.bbox_xyxy))
        if center is None or other.center_3d is None:
            continue
        dist = float(np.linalg.norm(np.asarray(center[:2], dtype=np.float32) - np.asarray(other.center_3d[:2], dtype=np.float32)))
        if dist < nearest_xy:
            nearest_xy = dist
            nearest_index = int(other.index)
    close_cost = 0.0 if not np.isfinite(nearest_xy) else max(0.0, (TARGET_CLUTTER_CLEARANCE_M - nearest_xy) / TARGET_CLUTTER_CLEARANCE_M)
    bbox_cost = min(1.0, max_bbox_iou * 2.0)
    return {
        "nearest_xy_distance_m": None if not np.isfinite(nearest_xy) else float(nearest_xy),
        "nearest_instance_index": nearest_index,
        "max_bbox_iou": float(max_bbox_iou),
        "cost": float(min(1.0, max(close_cost, bbox_cost))),
        "clearance_reference_m": float(TARGET_CLUTTER_CLEARANCE_M),
    }


def target_distance_cost(inst: YoloInstance) -> tuple[float, float | None]:
    if inst.center_3d is None:
        return 2.0, None
    center = np.asarray(inst.center_3d, dtype=np.float32)
    robot_xy = np.asarray(args_cli.robot_pos[:2], dtype=np.float32)
    dist = float(np.linalg.norm(center[:2] - robot_xy))
    return float(min(2.0, dist / 1.2)), dist


def target_image_guard_metrics(inst: YoloInstance) -> dict[str, Any]:
    height, width = inst.mask.shape[:2]
    bottom_fraction = min(0.45, max(0.0, float(args_cli.target_self_occlusion_bottom_fraction)))
    guard_top = int(round(float(height) * (1.0 - bottom_fraction)))
    guard_top = max(0, min(height, guard_top))
    mask = inst.mask.astype(bool)
    mask_pixels = int(np.count_nonzero(mask))
    protected_mask = np.zeros((height, width), dtype=bool)
    if guard_top < height:
        protected_mask[guard_top:, :] = True
    protected_pixels = int(np.count_nonzero(mask & protected_mask))
    overlap_fraction = float(protected_pixels / max(1, mask_pixels))
    x0, y0, x1, y1 = [float(v) for v in inst.bbox_xyxy]
    bbox_center_y = 0.5 * (y0 + y1)
    max_overlap = max(0.0, float(args_cli.target_self_occlusion_max_overlap_fraction))
    enabled = bool(args_cli.target_self_occlusion_filter)
    table_object_exempt = False
    table_object_reason = ""
    if inst.center_3d is not None:
        center = np.asarray(inst.center_3d, dtype=np.float32)
        table_margin_xy = 0.08
        in_table_xy = (
            float(TABLE_X_LIMITS[0] - table_margin_xy) <= float(center[0]) <= float(TABLE_X_LIMITS[1] + table_margin_xy)
            and float(TABLE_Y_LIMITS[0] - table_margin_xy) <= float(center[1]) <= float(TABLE_Y_LIMITS[1] + table_margin_xy)
        )
        above_table = float(center[2]) >= float(TABLE_SURFACE_Z + 0.012)
        not_wall_high = float(center[2]) <= float(TABLE_SURFACE_Z + 0.45)
        table_object_exempt = bool(in_table_xy and above_table and not_wall_high)
        table_object_reason = "rgbd_center_on_tabletop_object" if table_object_exempt else ""
    raw_rejected = bool(enabled and (overlap_fraction > max_overlap or bbox_center_y >= float(guard_top)))
    rejected = False
    return {
        "enabled": enabled,
        "policy": "lower_image_self_occlusion_guard_risk_only_no_hard_reject",
        "image_shape_hw": [int(height), int(width)],
        "bottom_fraction": bottom_fraction,
        "guard_top_px": int(guard_top),
        "mask_pixels": mask_pixels,
        "protected_overlap_pixels": protected_pixels,
        "protected_overlap_fraction": overlap_fraction,
        "max_overlap_fraction": max_overlap,
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_center_y_px": float(bbox_center_y),
        "raw_rejected_by_image_band": raw_rejected,
        "tabletop_object_exempt": table_object_exempt,
        "tabletop_object_exempt_reason": table_object_reason,
        "risk_only": True,
        "rejected": rejected,
    }


def target_ik_cost(inst: YoloInstance) -> float:
    reachability = instance_reachability_metadata(inst)
    if "best_tcp_error_m" in reachability:
        threshold = max(float(args_cli.pregrasp_error_threshold_m), 1.0e-3)
        return float(min(5.0, max(0.0, float(reachability.get("best_tcp_error_m", threshold * 5.0)) / threshold)))
    return float(min(5.0, max(0.0, float(reachability.get("reach_cost", 1.0) or 1.0) / 0.25)))


def instance_best_preset_standpoint_summary(inst: YoloInstance) -> dict[str, Any]:
    if bool(getattr(args_cli, "dynamic_grasp_standpoint_nav", False)):
        return {"enabled": False, "reason": "dynamic_grasp_standpoint_nav_takes_precedence"}
    if not bool(getattr(args_cli, "grasp_standpoint_nav", False)):
        return {"enabled": False}
    if inst.center_3d is None:
        return {"enabled": True, "selected": None, "penalty": 1, "score": float("inf"), "reason": "missing_rgbd_center"}
    try:
        names = parse_grasp_standpoint_candidates(args_cli.grasp_standpoint_candidates)
        robot_pose = [float(args_cli.robot_pos[0]), float(args_cli.robot_pos[1]), float(args_cli.robot_yaw)]
        target_w = np.asarray(inst.center_3d, dtype=np.float32).reshape(3)
        candidates = [grasp_standpoint_candidate_record(name, target_w, robot_pose) for name in names]
    except Exception as exc:
        return {"enabled": True, "selected": None, "penalty": 1, "score": float("inf"), "reason": repr(exc)}
    candidates.sort(key=lambda item: float(item["score"]))
    reachable = [item for item in candidates if item.get("geometric_reachable")]
    selected = reachable[0] if reachable else (candidates[0] if candidates else None)
    return {
        "enabled": True,
        "selected": selected,
        "penalty": 0 if reachable else 1,
        "score": float(selected.get("score", float("inf"))) if selected else float("inf"),
        "candidate_count": len(candidates),
        "reachable_count": len(reachable),
    }


def instance_best_preset_standpoint_score(inst: YoloInstance) -> tuple[int, float]:
    summary = instance_best_preset_standpoint_summary(inst)
    if not summary.get("enabled", False):
        return (0, 0.0)
    return (int(summary.get("penalty", 1)), float(summary.get("score", float("inf"))))


def target_selection_score(inst: YoloInstance, instances: list[YoloInstance]) -> dict[str, Any]:
    reach_penalty, reach_cost = instance_reachability_score(inst)
    standpoint_penalty, standpoint_cost = instance_best_preset_standpoint_score(inst)
    v28_group = float(v28_instance_group_priority(inst))
    ik_cost = target_ik_cost(inst)
    distance_cost, distance_m = target_distance_cost(inst)
    clutter = target_clutter_metrics(inst, instances)
    point_metrics = target_point_cloud_metrics(inst)
    depth_cost = target_depth_uncertainty_cost(point_metrics)
    graspability = target_graspability_cost(point_metrics)
    confidence = float(inst.vlm.confidence if inst.vlm else inst.yolo_confidence)
    low_confidence_cost = max(0.0, min(1.0, 1.0 - confidence))
    failure_key = target_failure_key(inst)
    recent_failures = int(_RECENT_TARGET_FAILURES.get(failure_key or "", 0))
    failed_recently_cost = 1.0 if recent_failures > 0 else 0.0
    weights = TARGET_SELECTION_SCORE_WEIGHTS
    components = {
        "unreachable_penalty": float(reach_penalty),
        "preset_standpoint_penalty": float(standpoint_penalty),
        "preset_standpoint_cost": float(standpoint_cost if np.isfinite(standpoint_cost) else 1.0e3),
        "v28_group_penalty": v28_group,
        "ik_cost": float(ik_cost),
        "grasp_distance_cost": float(distance_cost),
        "graspability_penalty": float(graspability["cost"]),
        "clutter_penalty": float(clutter["cost"]),
        "depth_uncertainty": float(depth_cost),
        "low_confidence_penalty": float(low_confidence_cost),
        "failed_recently_penalty": float(failed_recently_cost),
    }
    weighted = {name: float(weights[name] * components[name]) for name in weights}
    return {
        "policy": "reachable_first_then_ik_distance_graspability_v28_group_clutter_depth_confidence_failure",
        "total": float(sum(weighted.values())),
        "weights": {name: float(value) for name, value in weights.items()},
        "components": components,
        "weighted_components": weighted,
        "reach_cost_raw": float(reach_cost),
        "preset_standpoint": instance_best_preset_standpoint_summary(inst),
        "distance_to_robot_xy_m": distance_m,
        "clutter": clutter,
        "point_cloud": point_metrics,
        "graspability": graspability,
        "failure_key": failure_key,
        "recent_failure_count": recent_failures,
    }


def update_target_failure_memory(target: YoloInstance | None, execution_meta: dict[str, Any]) -> None:
    if target is None or not execution_meta.get("enabled", False):
        return
    key = target_failure_key(target)
    if not key:
        return
    if bool(execution_meta.get("grasp_success", False)):
        _RECENT_TARGET_FAILURES.pop(key, None)
        return
    _RECENT_TARGET_FAILURES[key] = min(3, int(_RECENT_TARGET_FAILURES.get(key, 0)) + 1)


def target_candidate_record(
    inst: YoloInstance,
    *,
    reject_reason: str | None = None,
    sort_key: tuple[float, ...] | None = None,
    selection_score: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reachability = instance_reachability_metadata(inst)
    point_metrics = target_point_cloud_metrics(inst)
    return {
        "instance_index": int(inst.index),
        "source": inst.source,
        "yolo_label": inst.yolo_label,
        "yolo_confidence": float(inst.yolo_confidence),
        "scene_key": inst.scene_key,
        "scene_object_name": inst.scene_object_name,
        "object_name": inst.vlm.object_name if inst.vlm else None,
        "waste_category": inst.vlm.waste_category if inst.vlm else None,
        "vlm_confidence": inst.vlm.confidence if inst.vlm else None,
        "mask_pixels": int(inst.mask.sum()),
        "center_3d_world": inst.center_3d.tolist() if inst.center_3d is not None else None,
        "rgbd_center_metadata": inst.rgbd_center_metadata,
        "component_score": float(inst.component_score),
        "v28_group_priority": float(v28_instance_group_priority(inst)),
        "point_cloud": point_metrics,
        "graspability": target_graspability_cost(point_metrics),
        "image_guard": target_image_guard_metrics(inst),
        "selection_score": selection_score,
        "reachability": reachability,
        "best_preset_standpoint": instance_best_preset_standpoint_summary(inst),
        "reject_reason": reject_reason,
        "sort_key": [float(v) for v in sort_key] if sort_key is not None else None,
    }


def choose_target(instances: list[YoloInstance], out_dir: Path | None = None) -> tuple[YoloInstance | None, str]:
    candidates: list[tuple[tuple[float, ...], YoloInstance, dict[str, Any]]] = []
    reachable_candidates: list[tuple[tuple[float, ...], YoloInstance, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    filtered_count = 0
    for inst in instances:
        selection_score = target_selection_score(inst, instances)
        if inst.source not in VISUAL_TARGET_SOURCES:
            rejected.append(target_candidate_record(inst, reject_reason="target_source_must_be_visual_perception", selection_score=selection_score))
            continue
        mask_pixels = int(inst.mask.sum())
        image_pixels = int(inst.mask.shape[0] * inst.mask.shape[1])
        x0, y0, x1, y1 = inst.bbox_xyxy
        bbox_area_fraction = max(0.0, (x1 - x0) * (y1 - y0)) / max(1, image_pixels)
        mask_area_fraction = mask_pixels / max(1, image_pixels)
        edge_margin = 4.0
        touches_edge = x0 <= edge_margin or y0 <= edge_margin or x1 >= inst.mask.shape[1] - edge_margin or y1 >= inst.mask.shape[0] - edge_margin
        point_metrics = target_point_cloud_metrics(inst)
        image_guard = target_image_guard_metrics(inst)
        if image_guard.get("rejected", False):
            rejected.append(target_candidate_record(inst, reject_reason="robot_self_occlusion_or_bottom_image_guard", selection_score=selection_score))
            continue
        if touches_edge and inst.center_3d is None:
            rejected.append(target_candidate_record(inst, reject_reason="touches_image_edge_without_3d_center", selection_score=selection_score))
            continue
        if mask_pixels < MIN_TARGET_MASK_PIXELS:
            rejected.append(target_candidate_record(inst, reject_reason="mask_too_small", selection_score=selection_score))
            continue
        if mask_area_fraction > MAX_TARGET_MASK_FRACTION or bbox_area_fraction > MAX_TARGET_BBOX_AREA_FRACTION:
            rejected.append(target_candidate_record(inst, reject_reason="mask_or_bbox_too_large", selection_score=selection_score))
            continue
        if inst.center_3d is None:
            rejected.append(target_candidate_record(inst, reject_reason="missing_rgbd_3d_center", selection_score=selection_score))
            continue
        if int(point_metrics.get("point_count", 0) or 0) < MIN_TARGET_POINT_COUNT:
            rejected.append(target_candidate_record(inst, reject_reason="insufficient_rgbd_points", selection_score=selection_score))
            continue
        if inst.vlm is None:
            rejected.append(target_candidate_record(inst, reject_reason="missing_v28_vlm_classification", selection_score=selection_score))
            continue
        if args_cli.skip_vlm:
            rejected.append(target_candidate_record(inst, reject_reason="v28_vlm_required_but_skip_vlm_enabled", selection_score=selection_score))
            continue
        if inst.vlm.error:
            rejected.append(target_candidate_record(inst, reject_reason=f"v28_vlm_error:{inst.vlm.error}", selection_score=selection_score))
            continue
        if not args_cli.skip_vlm and inst.vlm.confidence < args_cli.vlm_min_conf:
            rejected.append(target_candidate_record(inst, reject_reason="v28_vlm_confidence_below_threshold", selection_score=selection_score))
            continue
        if args_cli.target_category and inst.vlm.waste_category != args_cli.target_category:
            rejected.append(target_candidate_record(inst, reject_reason="target_category_mismatch", selection_score=selection_score))
            continue
        if not target_name_matches(inst):
            rejected.append(target_candidate_record(inst, reject_reason="target_name_mismatch", selection_score=selection_score))
            continue
        reach_penalty, reach_cost = instance_reachability_score(inst)
        standpoint_penalty, standpoint_cost = instance_best_preset_standpoint_score(inst)
        sort_key = (
            float(selection_score["total"]),
            float(standpoint_penalty),
            float(standpoint_cost if np.isfinite(standpoint_cost) else 1.0e3),
            float(reach_penalty),
            float(v28_instance_group_priority(inst)),
            float(target_ik_cost(inst)),
            float(selection_score["components"]["grasp_distance_cost"]),
            float(selection_score["components"]["graspability_penalty"]),
            float(selection_score["components"]["clutter_penalty"]),
            float(selection_score["components"]["depth_uncertainty"]),
            -float(inst.vlm.confidence),
            -float(mask_pixels),
        )
        record = target_candidate_record(inst, sort_key=sort_key, selection_score=selection_score)
        candidates.append((sort_key, inst, record))
        filtered_count += 1
        if reach_penalty == 0:
            reachable_candidates.append((sort_key, inst, record))
    if not candidates:
        if out_dir is not None:
            (out_dir / "target_candidates.json").write_text(
                json.dumps(
                    {
                        "requires_reachable_target": require_reachable_auto_target(),
                        "selected": None,
                        "valid": [],
                        "rejected": rejected,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        return None, "No visual perception instance satisfied the target filters."
    if require_reachable_auto_target():
        if not reachable_candidates:
            if out_dir is not None:
                (out_dir / "target_candidates.json").write_text(
                    json.dumps(
                        {
                    "requires_reachable_target": True,
                    "selection_policy": "one_object_per_cycle_score_lowest_after_hard_filters",
                    "score_weights": TARGET_SELECTION_SCORE_WEIGHTS,
                    "selected": None,
                    "valid": [item[2] for item in sorted(candidates, key=lambda row: row[0])],
                    "rejected": rejected,
                },
                indent=2,
                        ensure_ascii=False,
                    )
                )
            return None, f"No reachable visual-perception target satisfied the filters ({filtered_count} valid visual candidates were all outside arm reach)."
        candidates = reachable_candidates
    selection_pool = "reachable_visual_perception" if require_reachable_auto_target() else "visual_perception"
    candidates.sort(key=lambda item: item[0])
    if out_dir is not None:
        (out_dir / "target_candidates.json").write_text(
            json.dumps(
                {
                    "requires_reachable_target": require_reachable_auto_target(),
                    "selection_pool": selection_pool,
                    "selection_policy": "one_object_per_cycle_score_lowest_after_hard_filters",
                    "score_weights": TARGET_SELECTION_SCORE_WEIGHTS,
                    "hard_filters": [
                        "visual_perception_source_only",
                        "mask_size_bounds",
                        "rgbd_3d_center_required",
                        "minimum_rgbd_point_count",
                        "vlm_category_required",
                        "self_occlusion_lower_image_guard_with_rgbd_tabletop_exemption",
                        "optional_user_target_category_or_name",
                        "right_arm_reachable_required_when_enabled",
                        "preset_standpoint_geometric_reachable_preferred_when_enabled",
                    ],
                    "sort_order": [
                        "total_weighted_score",
                        "preset_standpoint_penalty",
                        "preset_standpoint_score",
                        "reach_penalty",
                        "v28_group_priority_planner_before_approach",
                        "right_arm_ik_cost",
                        "distance_to_robot_or_gripper",
                        "rgbd_graspability_penalty",
                        "clutter_penalty",
                        "depth_uncertainty",
                        "negative_v28_vlm_confidence",
                        "negative_mask_pixels",
                    ],
                    "selected": candidates[0][2],
                    "valid": [item[2] for item in candidates],
                    "rejected": rejected,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    reason = f"auto_score_lowest_one_object_{candidates[0][1].source}"
    return candidates[0][1], reason


def get_graspnet_wrapper() -> GraspNetWrapper:
    global _GRASPNET_WRAPPER
    if _GRASPNET_WRAPPER is None:
        grasp_to_tcp = parse_homogeneous_matrix(args_cli.graspnet_grasp_to_tcp, name="graspnet_grasp_to_tcp")
        _GRASPNET_WRAPPER = GraspNetWrapper(
            WORKSPACE_ROOT / args_cli.graspnet_repo,
            WORKSPACE_ROOT / args_cli.graspnet_checkpoint,
            num_point=args_cli.graspnet_num_point,
            collision_thresh=args_cli.graspnet_collision_thresh,
            voxel_size=args_cli.graspnet_voxel_size,
            score_thresh=args_cli.graspnet_score_thresh,
            force_objectness_top_k=args_cli.graspnet_force_objectness_top_k,
            grasp_to_tcp=grasp_to_tcp,
        )
    return _GRASPNET_WRAPPER


def get_anygrasp_wrapper() -> AnyGraspWrapper:
    global _ANYGRASP_WRAPPER
    if _ANYGRASP_WRAPPER is None:
        grasp_to_tcp = parse_homogeneous_matrix(args_cli.anygrasp_grasp_to_tcp, name="anygrasp_grasp_to_tcp")
        _ANYGRASP_WRAPPER = AnyGraspWrapper(
            WORKSPACE_ROOT / args_cli.anygrasp_repo,
            WORKSPACE_ROOT / args_cli.anygrasp_checkpoint,
            max_gripper_width=args_cli.anygrasp_max_gripper_width,
            gripper_height=args_cli.anygrasp_gripper_height,
            top_down_grasp=args_cli.anygrasp_top_down_grasp,
            apply_object_mask=args_cli.anygrasp_apply_object_mask,
            dense_grasp=args_cli.anygrasp_dense_grasp,
            collision_detection=args_cli.anygrasp_collision_detection,
            grasp_to_tcp=grasp_to_tcp,
        )
    return _ANYGRASP_WRAPPER


def get_grasp_backend_wrapper() -> Any:
    if args_cli.grasp_backend == "anygrasp":
        return get_anygrasp_wrapper()
    return get_graspnet_wrapper()


def active_grasp_to_tcp_metadata() -> dict[str, Any]:
    if args_cli.grasp_backend == "anygrasp":
        matrix = parse_homogeneous_matrix(args_cli.anygrasp_grasp_to_tcp, name="anygrasp_grasp_to_tcp")
    else:
        matrix = parse_homogeneous_matrix(args_cli.graspnet_grasp_to_tcp, name="graspnet_grasp_to_tcp")
    meta = calibration_metadata(matrix)
    meta["frame_convention"] = dict(GRASP_FRAME_CONVENTION)
    return meta


def centroid_fallback_enabled() -> bool:
    return bool(args_cli.use_centroid_fallback and not args_cli.strict_model_chain)


def backend_display_name() -> str:
    return "GraspNet" if args_cli.grasp_backend == "graspnet" else "AnyGrasp"


def target_pose_calibration_label() -> str:
    if bool(getattr(args_cli, "mind_sort_demo", False)) and str(getattr(args_cli, "mind_sort_grasp_proposal", "")) == "rgbd_center":
        return "Target_RGBD_TopGrasp_Pose_World" if bool(args_cli.top_grasp_shallow_target) else "Target_RGBD_Center_Pose_World"
    return f"Top1_{backend_display_name()}_Pose_World"


def rpy_from_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    rot = np.asarray(rotation, dtype=np.float64)
    sy = math.sqrt(float(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
    singular = sy < 1.0e-6
    if not singular:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = 0.0
    return (roll, pitch, yaw)


def pose_diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    mat = np.asarray(matrix, dtype=np.float32)
    rpy_rad = rpy_from_matrix(mat[:3, :3])
    rpy_deg = tuple(math.degrees(v) for v in rpy_rad)
    return {
        "translation_m": [float(v) for v in mat[:3, 3]],
        "rpy_rad": [float(v) for v in rpy_rad],
        "rpy_deg": [float(v) for v in rpy_deg],
        "matrix": mat.astype(float).tolist(),
    }


def pose_delta_diagnostics(target_pose_w: np.ndarray, tcp_pose_w: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target_pose_w, dtype=np.float32)
    tcp = np.asarray(tcp_pose_w, dtype=np.float32)
    delta_translation = tcp[:3, 3] - target[:3, 3]
    delta_rotation = target[:3, :3].T @ tcp[:3, :3]
    delta_rpy_rad = rpy_from_matrix(delta_rotation)
    return {
        "tcp_minus_target_translation_m": [float(v) for v in delta_translation],
        "tcp_minus_target_distance_m": float(np.linalg.norm(delta_translation)),
        "tcp_relative_to_target_rpy_rad": [float(v) for v in delta_rpy_rad],
        "tcp_relative_to_target_rpy_deg": [float(math.degrees(v)) for v in delta_rpy_rad],
    }


def print_pose_diagnostics(label: str, matrix: np.ndarray, *, phase: str) -> None:
    diag = pose_diagnostics(matrix)
    print(
        f"[CALIB][{phase}] {label}: translation_m={diag['translation_m']} "
        f"rpy_rad={diag['rpy_rad']} rpy_deg={diag['rpy_deg']}",
        flush=True,
    )
    print(f"[CALIB][{phase}] {label}_matrix={diag['matrix']}", flush=True)


def print_calibration_triplet(
    *,
    phase: str,
    camera_to_world: np.ndarray,
    target_pose_world: np.ndarray,
    tcp_pose_world: np.ndarray,
    gripper_pose_world: np.ndarray | None = None,
) -> None:
    print_pose_diagnostics("Camera_to_World", camera_to_world, phase=phase)
    print_pose_diagnostics(target_pose_calibration_label(), target_pose_world, phase=phase)
    print_pose_diagnostics("Right_Arm_TCP_Pose_World", tcp_pose_world, phase=phase)
    if gripper_pose_world is not None:
        print_pose_diagnostics("Right_Gripper_Base_Pose_World", gripper_pose_world, phase=phase)
    delta = pose_delta_diagnostics(target_pose_world, tcp_pose_world)
    print(
        f"[CALIB][{phase}] TCP_minus_Target: translation_m={delta['tcp_minus_target_translation_m']} "
        f"distance_m={delta['tcp_minus_target_distance_m']:.6f} "
        f"relative_rpy_deg={delta['tcp_relative_to_target_rpy_deg']}",
        flush=True,
    )
    if gripper_pose_world is not None:
        gripper_delta = pose_delta_diagnostics(target_pose_world, gripper_pose_world)
        print(
            f"[CALIB][{phase}] GripperBase_minus_Target: translation_m={gripper_delta['tcp_minus_target_translation_m']} "
            f"distance_m={gripper_delta['tcp_minus_target_distance_m']:.6f} "
            f"relative_rpy_deg={gripper_delta['tcp_relative_to_target_rpy_deg']}",
            flush=True,
        )


def graspnet_failure_hint(exc: Exception) -> str:
    text = repr(exc)
    if "checkpoint" in text.lower() or "not found" in text.lower():
        return (
            "Run: /home/zhxm/miniconda3/envs/my_task319_safe/bin/python "
            "task_319_garbage_sort/scripts/prepare_grasp_deps.py --download_missing_models"
        )
    if "pointnet2" in text or "knn" in text or "_ext" in text or "grasp_nms" in text:
        return (
            "Build GraspNet CUDA deps in my_task319_safe: "
            "prepare_grasp_deps.py --install_python_deps --build_pointnet2 --build_knn --download_missing_models"
        )
    return "Check GraspNet repo, CUDA extensions, and checkpoint paths."


def grasp_backend_failure_hint(exc: Exception) -> str:
    if args_cli.grasp_backend == "anygrasp":
        text = repr(exc).lower()
        if "checkpoint" in text or "not found" in text:
            return "Place AnyGrasp weights at models/anygrasp/checkpoint_detection.tar or pass --anygrasp_checkpoint."
        if "gsnet" in text or "license" in text or "minkowski" in text or "pointnet2" in text:
            return "Install AnyGrasp SDK dependencies, copy the matching gsnet/lib_cxx binaries, provide OpenSSL 1.1, and place the SDK license folder under grasp_detection/license."
        return "Check AnyGrasp SDK path, license, MinkowskiEngine, OpenSSL 1.1, pointnet2, and checkpoint."
    return graspnet_failure_hint(exc)


def camera_to_world_matrix(camera: CameraCfg) -> np.ndarray:
    data = camera.data
    pos_w = tensor_to_numpy(data.pos_w)[0].astype(np.float32)
    quat_w_ros = data.quat_w_ros
    rot_w = tensor_to_numpy(matrix_from_quat(quat_w_ros))[0].astype(np.float32)
    t_camera_to_world = np.eye(4, dtype=np.float32)
    t_camera_to_world[:3, :3] = rot_w
    t_camera_to_world[:3, 3] = pos_w
    return t_camera_to_world


def capture_rgbd_camera(scene: InteractiveScene, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera = scene[name]
    rgb = sanitize_rgb(tensor_to_numpy(camera.data.output["rgb"])[0])
    depth = sanitize_depth(tensor_to_numpy(camera.data.output["distance_to_image_plane"])[0])
    intrinsics = tensor_to_numpy(camera.data.intrinsic_matrices)[0].astype(np.float32)
    t_camera_to_world = camera_to_world_matrix(camera)
    return rgb, depth, intrinsics, t_camera_to_world


def capture_head_camera(scene: InteractiveScene) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return capture_rgbd_camera(scene, "head_rgbd")


def capture_wrist_camera(scene: InteractiveScene) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return capture_rgbd_camera(scene, "wrist_rgbd")


def mask_points_world(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    *,
    min_z: float = TABLE_SURFACE_Z + 0.006,
    max_z: float = TABLE_SURFACE_Z + 0.45,
) -> np.ndarray:
    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    depths = depth_m[ys, xs]
    valid = np.isfinite(depths) & (depths > 0.05) & (depths < 5.0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    depths = depths[valid].astype(np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if fx == 0.0 or fy == 0.0:
        return np.zeros((0, 3), dtype=np.float32)
    points_camera = np.stack(((xs - cx) * depths / fx, (ys - cy) * depths / fy, depths), axis=1)
    points_h = np.concatenate([points_camera, np.ones((points_camera.shape[0], 1), dtype=np.float32)], axis=1)
    points_world = (t_camera_to_world @ points_h.T).T[:, :3].astype(np.float32)
    object_like = (points_world[:, 2] > min_z) & (points_world[:, 2] < max_z)
    if np.count_nonzero(object_like) < 25:
        return np.zeros((0, 3), dtype=np.float32)
    return points_world[object_like]


def depth_image_to_world_map(depth_m: np.ndarray, intrinsics: np.ndarray, t_camera_to_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_m.shape
    ys, xs = np.indices((height, width), dtype=np.float32)
    depths = depth_m.astype(np.float32)
    valid = np.isfinite(depths) & (depths > 0.05) & (depths < 5.0)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    points_world = np.zeros((height, width, 3), dtype=np.float32)
    if fx == 0.0 or fy == 0.0 or not np.any(valid):
        return points_world, np.zeros((height, width), dtype=bool)
    x_cam = (xs[valid] - cx) * depths[valid] / fx
    y_cam = (ys[valid] - cy) * depths[valid] / fy
    z_cam = depths[valid]
    points_camera = np.stack((x_cam, y_cam, z_cam), axis=1)
    points_h = np.concatenate([points_camera, np.ones((points_camera.shape[0], 1), dtype=np.float32)], axis=1)
    points_world[valid] = (t_camera_to_world @ points_h.T).T[:, :3].astype(np.float32)
    return points_world, valid


def draw_depth_component_overlay(rgb: np.ndarray, components: list[dict[str, Any]], out_dir: Path) -> None:
    pil = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(pil)
    for rank, item in enumerate(components):
        color = color_for_index(rank + 3)
        x0, y0, x1, y1 = item["bbox_xyxy"]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.text((x0 + 3, min(max(0.0, y0 + 4), rgb.shape[0] - 20)), f"D{rank}:{item['scene_object_name']} {item['category']}", fill=color)
    pil.save(out_dir / "depth_component_overlay.png")


def tabletop_workspace_mask(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    *,
    min_z: float = TABLE_SURFACE_Z + 0.006,
    max_z: float = TABLE_SURFACE_Z + 0.45,
) -> np.ndarray:
    points_world, valid = depth_image_to_world_map(depth_m, intrinsics, t_camera_to_world)
    z = points_world[..., 2]
    return (mask.astype(bool) & valid & np.isfinite(z) & (z >= min_z) & (z <= max_z))


def detect_tabletop_depth_components(
    scene: InteractiveScene,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    out_dir: Path,
) -> list[YoloInstance]:
    instances: list[YoloInstance] = []
    metadata: list[dict[str, Any]] = []
    if args_cli.disable_depth_component_fallback:
        draw_depth_component_overlay(rgb, metadata, out_dir)
        (out_dir / "depth_components.json").write_text("[]\n")
        return instances
    try:
        import cv2  # type: ignore
    except Exception as exc:
        draw_depth_component_overlay(rgb, metadata, out_dir)
        (out_dir / "depth_components.json").write_text(json.dumps({"error": f"OpenCV unavailable: {exc!r}"}, ensure_ascii=False, indent=2))
        return instances

    points_world, valid = depth_image_to_world_map(depth_m, intrinsics, t_camera_to_world)
    candidate = (
        valid
        & (points_world[:, :, 2] > TABLE_SURFACE_Z + 0.006)
        & (points_world[:, :, 2] < TABLE_SURFACE_Z + 0.36)
        & (points_world[:, :, 0] > TABLE_X_LIMITS[0])
        & (points_world[:, :, 0] < TABLE_X_LIMITS[1])
        & (points_world[:, :, 1] > TABLE_Y_LIMITS[0])
        & (points_world[:, :, 1] < TABLE_Y_LIMITS[1])
    )
    mask_u8 = candidate.astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    height, width = depth_m.shape
    for label_id in range(1, num_labels):
        x, y, w, h, area = (int(v) for v in stats[label_id])
        if area < 80 or area > 24000:
            continue
        if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
            continue
        component_mask = labels == label_id
        points = points_world[component_mask]
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] < 50:
            continue
        center, center_meta = rgbd_geometric_grasp_center(points)
        xy_extent = np.percentile(points[:, :2], 95, axis=0) - np.percentile(points[:, :2], 5, axis=0)
        z_extent = float(np.percentile(points[:, 2], 95) - np.percentile(points[:, 2], 5))
        if float(np.max(xy_extent)) > 0.30 or z_extent > 0.34:
            continue
        scene_key, scene_object_name, match_distance = locate_target_rigid_object(scene, center)
        if match_distance > max(0.10, scene_target_radius(scene_key) + 0.04):
            scene_key = None
            scene_object_name = "tabletop_object"
        category = category_for_scene_key(scene_key)
        reach_penalty, reach_cost = reachability_score(center)
        grasp_priority = SCENE_GRASP_PRIORITY.get(scene_key or "", 20)
        score = 1.0 / (1.0 + grasp_priority + 2.0 * reach_penalty + reach_cost + float(np.max(xy_extent)))
        vlm = VlmClassification(
            scene_object_name or "tabletop_object",
            category,
            0.55,
            "Head-depth tabletop component fallback matched to nearest YCB rigid object.",
        )
        inst = YoloInstance(
            1000 + len(instances),
            f"depth_{scene_object_name or 'object'}",
            float(score),
            (float(x), float(y), float(x + w), float(y + h)),
            component_mask,
            vlm=vlm,
            center_3d=center,
            source="depth_component",
            scene_key=scene_key,
            scene_object_name=scene_object_name,
            component_score=float(score),
            points_world=points.astype(np.float32, copy=False),
            rgbd_center_metadata=center_meta,
        )
        instances.append(inst)
        metadata.append(
            {
                "index": inst.index,
                "bbox_xyxy": inst.bbox_xyxy,
                "mask_pixels": int(component_mask.sum()),
                "center_3d": center.tolist(),
                "xy_extent_m": xy_extent.astype(float).tolist(),
                "z_extent_m": z_extent,
                "scene_key": scene_key,
                "scene_object_name": scene_object_name,
                "category": category,
                "match_distance_m": float(match_distance),
                "reach_penalty": int(reach_penalty),
                "reach_cost": float(reach_cost),
                "score": float(score),
            }
        )
    if len(instances) > 0:
        instances.sort(
            key=lambda inst: (
                reachability_score(inst.center_3d)[0],
                SCENE_GRASP_PRIORITY.get(inst.scene_key or "", 20),
                reachability_score(inst.center_3d)[1],
                -inst.component_score,
            )
        )
        instances = instances[:6]
        ordered_indices = {inst.index for inst in instances}
        metadata = [item for item in metadata if item["index"] in ordered_indices]
    draw_depth_component_overlay(rgb, metadata, out_dir)
    (out_dir / "depth_components.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return instances


def scene_guided_depth_instance(
    scene: InteractiveScene,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    scene_key: str,
    scene_object_name: str,
) -> YoloInstance | None:
    if scene_key not in scene.keys():
        return None
    points_world, valid = depth_image_to_world_map(depth_m, intrinsics, t_camera_to_world)
    root = tensor_to_numpy(scene[scene_key].data.root_state_w[0, :3]).astype(np.float32)
    radius = scene_target_radius(scene_key)
    z = points_world[:, :, 2]
    xy_dist = np.linalg.norm(points_world[:, :, :2] - root[:2], axis=2)
    mask = (
        valid
        & np.isfinite(z)
        & (z >= TABLE_SURFACE_Z + 0.006)
        & (z <= TABLE_SURFACE_Z + 0.28)
        & (xy_dist <= radius)
    )
    if np.count_nonzero(mask) < 25:
        mask = (
            valid
            & np.isfinite(z)
            & (z >= TABLE_SURFACE_Z + 0.006)
            & (z <= TABLE_SURFACE_Z + 0.32)
            & (xy_dist <= radius * 1.35)
        )
    if np.count_nonzero(mask) < 25:
        return None
    try:
        import cv2  # type: ignore

        mask_u8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num_labels > 1:
            root_xy = root[:2]
            best_label = None
            best_cost = float("inf")
            for label_id in range(1, num_labels):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                if area < 20:
                    continue
                pts = points_world[labels == label_id]
                center = np.median(pts[:, :2], axis=0)
                cost = float(np.linalg.norm(center - root_xy)) - 0.0005 * area
                if cost < best_cost:
                    best_cost = cost
                    best_label = label_id
            if best_label is not None:
                mask = labels == best_label
    except Exception:
        pass
    ys, xs = np.nonzero(mask)
    if len(xs) < 25:
        return None
    pts = points_world[mask]
    center, center_meta = rgbd_geometric_grasp_center(pts)
    spec = TASK319_OBJECT_BY_NAME.get(scene_object_name)
    category = spec.waste_category if spec is not None else category_for_scene_key(scene_key)
    object_label = scene_object_display_name(scene_object_name)
    vlm = VlmClassification(
        object_label,
        category,
        0.95,
        "Scene-guided depth target generated from the requested task-319 object name.",
    )
    try:
        guided_index = 2000 + int(scene_key.split("_")[-1])
    except Exception:
        guided_index = 2999
    return YoloInstance(
        guided_index,
        f"scene_{object_label}",
        0.95,
        (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
        mask,
        vlm=vlm,
        center_3d=center,
        source="scene_depth",
        scene_key=scene_key,
        scene_object_name=scene_object_name,
        component_score=0.95,
        points_world=pts.astype(np.float32, copy=False),
        rgbd_center_metadata=center_meta,
    )


def perceived_from_target(target: YoloInstance, center_3d: np.ndarray | None) -> PerceivedObject:
    identity = freeze_target_identity(target)
    identity_meta = target_identity_metadata(identity) or {}
    return PerceivedObject(
        object_id=target.index,
        object_name=target.vlm.object_name if target.vlm else "unknown",
        class_name=target.vlm.object_name if target.vlm else target.yolo_label,
        confidence=target.vlm.confidence if target.vlm else target.yolo_confidence,
        bbox_xyxy=target.bbox_xyxy,
        mask=target.mask,
        center_2d=(0.5 * (target.bbox_xyxy[0] + target.bbox_xyxy[2]), 0.5 * (target.bbox_xyxy[1] + target.bbox_xyxy[3])),
        center_3d=center_3d,
        waste_category=target.vlm.waste_category if target.vlm else "",
        metadata={
            **identity_meta,
            "instance_index": int(target.index),
            "source": target.source,
            "yolo_label": target.yolo_label,
            "scene_key": target.scene_key,
            "scene_object_name": target.scene_object_name,
            "right_arm_reachability": target.right_arm_reachability,
            "rgbd_center_metadata": target.rgbd_center_metadata,
        },
    )


def estimate_parallel_jaw_grasp_width(
    points_world: np.ndarray,
    *,
    min_width_m: float = 0.024,
    calibration_clearance_m: float = 0.004,
) -> tuple[float, dict[str, Any]]:
    pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)] if pts.size else pts
    meta: dict[str, Any] = {
        "point_count": int(pts.shape[0]),
        "min_width_m": float(min_width_m),
        "calibration_clearance_m": float(calibration_clearance_m),
    }
    if pts.shape[0] < 8:
        width = clamp_width(float(max(min_width_m, args_cli.debug_force_scene_grasp_width)))
        meta.update(
            {
                "method": "fallback_debug_width",
                "estimated_width_m": float(width),
            }
        )
        return width, meta

    xy_extent = np.percentile(pts[:, :2], 95, axis=0) - np.percentile(pts[:, :2], 5, axis=0)
    short_axis = float(np.min(xy_extent))
    long_axis = float(np.max(xy_extent))
    width = clamp_width(float(max(min_width_m, short_axis + calibration_clearance_m)))
    meta.update(
        {
            "method": "rgbd_xy_short_axis_plus_calibrated_clearance",
            "xy_extent_p05_p95_m": [float(v) for v in xy_extent],
            "short_axis_m": short_axis,
            "long_axis_m": long_axis,
            "estimated_width_m": float(width),
        }
    )
    return width, meta


def estimate_rgbd_top_grasp_wrist_rotation(points_world: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)] if pts.size else pts
    tcp_axis = np.asarray(args_cli.rgbd_top_grasp_tcp_axis, dtype=np.float32).reshape(3)
    tcp_axis_norm = float(np.linalg.norm(tcp_axis))
    if tcp_axis_norm < 1.0e-6:
        tcp_axis = np.array([0.35, 0.0, -0.94], dtype=np.float32)
    else:
        tcp_axis = tcp_axis / tcp_axis_norm
    fixed_jaw_axis = np.asarray(args_cli.rgbd_top_grasp_envelope_jaw_axis, dtype=np.float32).reshape(3)
    fixed_jaw_axis_norm = float(np.linalg.norm(fixed_jaw_axis))
    if fixed_jaw_axis_norm < 1.0e-6:
        fixed_jaw_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        fixed_jaw_axis = fixed_jaw_axis / fixed_jaw_axis_norm
    jaw_axis = fixed_jaw_axis.copy()
    directionless_envelope = bool(args_cli.rgbd_top_grasp_directionless_envelope)
    meta: dict[str, Any] = {
        "enabled": bool(args_cli.rgbd_top_grasp_object_aware_orientation) or directionless_envelope,
        "method": "rgbd_directionless_envelope_fixed_jaw_axis" if directionless_envelope else "rgbd_xy_pca_short_axis_jaw_top_grasp",
        "directionless_envelope": directionless_envelope,
        "point_count": int(pts.shape[0]),
        "tcp_offset_axis_world": tcp_axis.astype(float).tolist(),
        "fixed_envelope_jaw_axis_world": fixed_jaw_axis.astype(float).tolist(),
        "fallback_jaw_axis_world": jaw_axis.astype(float).tolist(),
    }
    if directionless_envelope:
        meta.update(
            {
                "success": True,
                "jaw_axis_world": jaw_axis.astype(float).tolist(),
                "reason": "directionless_envelope_uses_fixed_world_jaw_axis_and_max_open_gripper",
            }
        )
    elif bool(args_cli.rgbd_top_grasp_object_aware_orientation) and pts.shape[0] >= 12:
        xy = pts[:, :2].astype(np.float32)
        xy_center = np.median(xy, axis=0)
        xy_centered = xy - xy_center
        try:
            cov = np.cov(xy_centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)
            short_xy = eigvecs[:, int(order[0])].astype(np.float32)
            long_xy = eigvecs[:, int(order[-1])].astype(np.float32)
            if float(np.dot(short_xy, np.array([0.0, 1.0], dtype=np.float32))) < 0.0:
                short_xy = -short_xy
            if float(np.dot(long_xy, np.array([1.0, 0.0], dtype=np.float32))) < 0.0:
                long_xy = -long_xy
            jaw_axis = np.array([float(short_xy[0]), float(short_xy[1]), 0.0], dtype=np.float32)
            jaw_axis_norm = float(np.linalg.norm(jaw_axis))
            if jaw_axis_norm < 1.0e-6:
                jaw_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                jaw_axis = jaw_axis / jaw_axis_norm
            meta.update(
                {
                    "success": True,
                    "xy_center_m": [float(v) for v in xy_center.tolist()],
                    "eigenvalues": [float(v) for v in eigvals.tolist()],
                    "long_axis_world": [float(long_xy[0]), float(long_xy[1]), 0.0],
                    "jaw_axis_world": jaw_axis.astype(float).tolist(),
                    "reason": "jaw_axis_uses_short_xy_pca_axis",
                }
            )
        except Exception as exc:
            meta.update({"success": False, "reason": f"pca_failed:{exc!r}"})
    else:
        reason = "disabled" if not bool(args_cli.rgbd_top_grasp_object_aware_orientation) else f"insufficient_point_cloud:{pts.shape[0]}"
        meta.update({"success": False, "reason": reason, "jaw_axis_world": jaw_axis.astype(float).tolist()})
    wrist_rot = wrist_rotation_from_tcp_offset_axis(tcp_axis, preferred_jaw_axis_w=jaw_axis)
    gripper_base_rot = (wrist_rot @ RIGHT_GRIPPER_INLINE_MOUNT_ROT).astype(np.float32)
    meta["wrist_rotation_world"] = wrist_rot.astype(float).tolist()
    meta["gripper_base_rotation_world"] = gripper_base_rot.astype(float).tolist()
    meta["gripper_forward_axis_world"] = gripper_base_rot[:, 0].astype(float).tolist()
    meta["gripper_jaw_axis_world"] = gripper_base_rot[:, 1].astype(float).tolist()
    meta["gripper_jaw_axis_semantics"] = "gripper_base/right_gripper_tcp local +Y is the parallel-jaw opening axis; local +X points from gripper base toward the finger-pad TCP"
    return wrist_rot, meta


def top_grasp_target_from_points(
    center_world: np.ndarray,
    points_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    center = np.asarray(center_world, dtype=np.float32).reshape(3)
    pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)] if pts.size else pts
    depth_m = max(0.0, float(args_cli.top_grasp_depth_m))
    hover_m = max(0.0, float(args_cli.top_grasp_hover_m))
    meta: dict[str, Any] = {
        "enabled": bool(args_cli.top_grasp_shallow_target),
        "policy": "xy_rgbd_geometric_center_zmax_minus_shallow_depth",
        "point_count": int(pts.shape[0]),
        "grasp_depth_m": depth_m,
        "hover_above_zmax_m": hover_m,
        "center_world_m": center.astype(float).tolist(),
    }
    if not bool(args_cli.top_grasp_shallow_target):
        target = center.copy()
        hover = target.copy()
        hover[2] = max(float(target[2] + hover_m), float(TABLE_SURFACE_Z + args_cli.arm_motion_min_table_clearance_m))
        meta.update(
            {
                "success": False,
                "reason": "disabled",
                "target_world_m": target.astype(float).tolist(),
                "hover_world_m": hover.astype(float).tolist(),
            }
        )
        return target, hover, meta
    if pts.shape[0] < 8:
        target = center.copy()
        target[2] = max(float(target[2]), float(TABLE_SURFACE_Z + 0.035))
        hover = target.copy()
        hover[2] = max(float(target[2] + hover_m), float(TABLE_SURFACE_Z + args_cli.arm_motion_min_table_clearance_m))
        meta.update(
            {
                "success": False,
                "reason": f"insufficient_point_cloud:{pts.shape[0]}",
                "target_world_m": target.astype(float).tolist(),
                "hover_world_m": hover.astype(float).tolist(),
            }
        )
        return target, hover, meta

    z_max = float(np.max(pts[:, 2]))
    z_min = float(np.min(pts[:, 2]))
    target = center.copy()
    unclamped_target_z = float(z_max - depth_m)
    table_clearance_floor = float(TABLE_SURFACE_Z + 0.018)
    target[2] = max(table_clearance_floor, unclamped_target_z)
    hover = target.copy()
    hover[2] = max(float(z_max + hover_m), float(target[2] + hover_m), float(TABLE_SURFACE_Z + args_cli.arm_motion_min_table_clearance_m))
    meta.update(
        {
            "success": True,
            "z_min_m": z_min,
            "z_max_m": z_max,
            "z_range_m": z_max - z_min,
            "target_world_m": target.astype(float).tolist(),
            "hover_world_m": hover.astype(float).tolist(),
            "unclamped_target_z_m": unclamped_target_z,
            "table_clearance_floor_z_m": table_clearance_floor,
            "target_z_was_table_clamped": bool(target[2] > unclamped_target_z + 1.0e-6),
            "target_z_minus_center_z_m": float(target[2] - center[2]),
            "hover_z_minus_target_z_m": float(hover[2] - target[2]),
        }
    )
    return target, hover, meta


def centroid_fallback_grasp(
    target: YoloInstance,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
) -> tuple[SelectedGrasp | None, dict[str, Any]]:
    points_world = mask_points_world(target.mask, depth_m, intrinsics, t_camera_to_world)
    if points_world.shape[0] < 25:
        return None, {"source": "fallback", "error": f"Only {points_world.shape[0]} valid mask point-cloud samples."}
    center, center_meta = rgbd_geometric_grasp_center(points_world)
    target.center_3d = center
    target.rgbd_center_metadata = center_meta
    grasp_point = center.astype(np.float32)
    grasp_point_source = "visual_mask_rgbd_geometric_center" if target.source in VISUAL_TARGET_SOURCES else "target_3d_geometric_center"
    width, width_meta = estimate_parallel_jaw_grasp_width(points_world, min_width_m=0.020)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    pose[:3, 3] = grasp_point
    pose[2, 3] = max(float(grasp_point[2]), TABLE_SURFACE_Z + 0.035)
    selected = SelectedGrasp(
        pose_world=pose,
        width=width,
        score=0.25,
        center_2d=(0.5 * (target.bbox_xyxy[0] + target.bbox_xyxy[2]), 0.5 * (target.bbox_xyxy[1] + target.bbox_xyxy[3])),
        target=perceived_from_target(target, center),
        source="fallback",
    )
    return selected, {
        "source": "fallback",
        "point_count": int(points_world.shape[0]),
        "center_world": center.tolist(),
        "grasp_point_world": pose[:3, 3].astype(float).tolist(),
        "grasp_point_source": grasp_point_source,
        "rgbd_center_metadata": center_meta,
        "target_identity": target_identity_metadata(freeze_target_identity(target)),
        "center_to_grasp_point_delta_m": (pose[:3, 3] - center).astype(float).tolist(),
        "estimated_width_m": width,
        "grasp_width_estimation": width_meta,
    }


def selected_grasp_from_rgbd_center(
    target: YoloInstance,
    center_world: np.ndarray,
    points_world: np.ndarray | None,
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> SelectedGrasp:
    pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3) if points_world is not None else np.zeros((0, 3), dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)] if pts.size else pts
    requested_center = np.asarray(center_world, dtype=np.float32).reshape(3)
    center_meta: dict[str, Any] = {}
    if pts.shape[0] > 0:
        center, center_meta = rgbd_geometric_grasp_center(pts)
        center_meta["requested_center_world_m"] = requested_center.astype(float).tolist()
        center_meta["delta_corrected_minus_requested_m"] = (center - requested_center).astype(float).tolist()
    else:
        center = requested_center
    width, width_meta = estimate_parallel_jaw_grasp_width(pts)
    grasp_target, hover_target, top_grasp_meta = top_grasp_target_from_points(center, pts)
    wrist_rot, wrist_orientation_meta = estimate_rgbd_top_grasp_wrist_rotation(pts)
    gripper_base_rot = np.asarray(
        wrist_orientation_meta.get("gripper_base_rotation_world", (wrist_rot @ RIGHT_GRIPPER_INLINE_MOUNT_ROT).tolist()),
        dtype=np.float32,
    ).reshape(3, 3)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = gripper_base_rot
    pose[:3, 3] = grasp_target
    target.center_3d = center.copy()
    target.rgbd_center_metadata = center_meta or target.rgbd_center_metadata
    if pts.shape[0] > 0:
        target.points_world = pts.copy()
    return SelectedGrasp(
        pose_world=pose,
        width=width,
        score=0.55,
        center_2d=(0.5 * (target.bbox_xyxy[0] + target.bbox_xyxy[2]), 0.5 * (target.bbox_xyxy[1] + target.bbox_xyxy[3])),
        target=perceived_from_target(target, center),
        source=source,
        metadata={
            "grasp_point_source": source,
            "target_scene_key": target.scene_key,
            "scene_key": target.scene_key,
            "scene_object_name": target.scene_object_name,
            "object_name": target.vlm.object_name if target.vlm else target.yolo_label,
            "waste_category": target.vlm.waste_category if target.vlm else "",
            "point_count": int(pts.shape[0]),
            "rgbd_center_metadata": center_meta or target.rgbd_center_metadata,
            "estimated_width_m": float(width),
            "grasp_width_estimation": width_meta,
            "top_grasp": top_grasp_meta,
            "top_grasp_target_world_m": grasp_target.astype(float).tolist(),
            "top_grasp_hover_world_m": hover_target.astype(float).tolist(),
            "rgbd_top_grasp_orientation": wrist_orientation_meta,
            "rgbd_wrist_rotation_world": wrist_rot.astype(float).tolist(),
            "rgbd_gripper_base_rotation_world": gripper_base_rot.astype(float).tolist(),
            "grasp_motion_profile": STABLE_GRASP_MOTION_PROFILE,
            **(metadata or {}),
        },
    )


def wrist_refine_target_center(
    scene: InteractiveScene,
    target: YoloInstance,
    out_dir: Path,
) -> tuple[np.ndarray | None, np.ndarray, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb, depth_m, intrinsics, t_camera_to_world = capture_wrist_camera(scene)
    Image.fromarray(rgb).save(out_dir / "wrist_refine_rgb.png")
    save_depth_images(depth_m, out_dir, prefix="wrist_refine")
    meta: dict[str, Any] = {
        "enabled": bool(args_cli.wrist_refine_grasp),
        "camera": "wrist_rgbd",
        "rgb": str(out_dir / "wrist_refine_rgb.png"),
        "intrinsics": intrinsics.tolist(),
        "camera_to_world": t_camera_to_world.tolist(),
        "initial_target_center_world_m": target.center_3d.tolist() if target.center_3d is not None else None,
        "roi_radius_px": int(args_cli.wrist_refine_roi_radius_px),
        "success": False,
    }
    if target.center_3d is None:
        meta["reason"] = "target_has_no_initial_center"
        (out_dir / "wrist_refine_debug.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return None, np.zeros((0, 3), dtype=np.float32), meta
    projection = project_world_point_to_image(np.asarray(target.center_3d, dtype=np.float32), intrinsics, t_camera_to_world)
    meta["initial_center_projection"] = projection
    height, width = depth_m.shape[:2]
    if not projection.get("finite") or "uv" not in projection:
        meta["reason"] = "initial_center_not_projectable_to_wrist_camera"
        (out_dir / "wrist_refine_debug.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return None, np.zeros((0, 3), dtype=np.float32), meta
    u, v = float(projection["uv"][0]), float(projection["uv"][1])
    if not (0.0 <= u < width and 0.0 <= v < height):
        meta["reason"] = "initial_center_projection_outside_wrist_image"
        (out_dir / "wrist_refine_debug.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return None, np.zeros((0, 3), dtype=np.float32), meta
    radius = max(8, int(args_cli.wrist_refine_roi_radius_px))
    yy, xx = np.ogrid[:height, :width]
    roi = ((xx - u) ** 2 + (yy - v) ** 2) <= radius**2
    points_world = mask_points_world(
        roi,
        depth_m,
        intrinsics,
        t_camera_to_world,
        min_z=TABLE_SURFACE_Z + 0.006,
        max_z=TABLE_SURFACE_Z + 0.40,
    )
    if points_world.shape[0] < 25:
        meta["reason"] = f"insufficient_wrist_roi_depth_points:{points_world.shape[0]}"
        meta["point_count"] = int(points_world.shape[0])
        (out_dir / "wrist_refine_debug.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return None, points_world, meta
    center = np.median(points_world, axis=0).astype(np.float32)
    meta.update(
        {
            "success": True,
            "reason": "wrist_rgbd_roi_median_center",
            "point_count": int(points_world.shape[0]),
            "refined_center_world_m": center.astype(float).tolist(),
            "delta_from_initial_center_m": (center - np.asarray(target.center_3d, dtype=np.float32)).astype(float).tolist(),
            "point_cloud": point_cloud_debug_stats(points_world),
        }
    )
    pil = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(pil)
    draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline=(255, 0, 0), width=3)
    draw.line((u - 16, v, u + 16, v), fill=(255, 0, 0), width=3)
    draw.line((u, v - 16, u, v + 16), fill=(255, 0, 0), width=3)
    pil.save(out_dir / "wrist_refine_overlay.png")
    meta["overlay"] = str(out_dir / "wrist_refine_overlay.png")
    (out_dir / "wrist_refine_debug.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return center, points_world, meta


def point_cloud_debug_stats(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    stats: dict[str, Any] = {
        "point_count": int(pts.shape[0]),
        "shape": list(pts.shape),
        "frame": "world",
        "units": "meters",
    }
    if pts.shape[0] == 0:
        return stats
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    extent = maxs - mins
    stats.update(
        {
            "min_xyz_m": mins.astype(float).tolist(),
            "max_xyz_m": maxs.astype(float).tolist(),
            "extent_xyz_m": extent.astype(float).tolist(),
            "z_range_m": float(extent[2]),
            "z_is_flat_under_2cm": bool(float(extent[2]) < 0.02),
        }
    )
    return stats


def rgbd_geometric_grasp_center(
    points_world: np.ndarray,
    *,
    min_extent_z_for_midpoint_m: float = 0.012,
) -> tuple[np.ndarray, dict[str, Any]]:
    pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)] if pts.size else pts
    if pts.shape[0] == 0:
        center = np.zeros(3, dtype=np.float32)
        return center, {
            "method": "empty_point_cloud_zero_center",
            "point_count": 0,
        }

    raw_median = np.median(pts, axis=0).astype(np.float32)
    center = raw_median.copy()
    z_min = float(np.min(pts[:, 2]))
    z_max = float(np.max(pts[:, 2]))
    z_lo = z_min
    z_hi = z_max
    z_extent = max(0.0, z_hi - z_lo)
    use_extent_midpoint_z = bool(z_extent >= float(min_extent_z_for_midpoint_m))
    if use_extent_midpoint_z:
        center[2] = np.float32(0.5 * (z_lo + z_hi))
    center[2] = np.float32(max(float(center[2]), float(TABLE_SURFACE_Z + 0.025)))
    metadata = {
        "method": "rgbd_point_cloud_xy_median_z_extent_midpoint",
        "point_count": int(pts.shape[0]),
        "raw_median_center_world_m": raw_median.astype(float).tolist(),
        "corrected_center_world_m": center.astype(float).tolist(),
        "z_min_m": z_min,
        "z_max_m": z_max,
        "z_percentile_02_m": float(np.percentile(pts[:, 2], 2.0)),
        "z_percentile_98_m": float(np.percentile(pts[:, 2], 98.0)),
        "z_extent_min_max_m": float(z_extent),
        "min_extent_z_for_midpoint_m": float(min_extent_z_for_midpoint_m),
        "used_extent_midpoint_z": use_extent_midpoint_z,
        "z_correction_m": float(center[2] - raw_median[2]),
    }
    return center.astype(np.float32), metadata


def write_point_cloud_ply(path: Path, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\n")
        stream.write("format ascii 1.0\n")
        stream.write(f"element vertex {len(pts)}\n")
        stream.write("property float x\n")
        stream.write("property float y\n")
        stream.write("property float z\n")
        stream.write("end_header\n")
        for x, y, z in pts:
            stream.write(f"{float(x):.8f} {float(y):.8f} {float(z):.8f}\n")


def draw_selected_grasp_overlay(rgb: np.ndarray, selected: SelectedGrasp | None, out_dir: Path, *, label: str) -> None:
    pil = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(pil)
    if selected is not None:
        u, v = selected.center_2d
        draw.ellipse((u - 14, v - 14, u + 14, v + 14), outline=(255, 0, 0), width=5)
        draw.text((u + 16, v + 8), f"{label} {selected.score:.2f}", fill=(255, 0, 0))
    pil.save(out_dir / "selected_grasp_overlay.png")


def save_camera_self_occlusion_guard(rgb: np.ndarray, out_dir: Path) -> dict[str, Any]:
    height, width = rgb.shape[:2]
    bottom_fraction = min(0.45, max(0.0, float(args_cli.target_self_occlusion_bottom_fraction)))
    guard_top = int(round(float(height) * (1.0 - bottom_fraction)))
    guard_top = max(0, min(height, guard_top))
    pil = Image.fromarray(rgb.copy()).convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if bool(args_cli.target_self_occlusion_filter) and guard_top < height:
        draw.rectangle((0, guard_top, width - 1, height - 1), fill=(255, 0, 0, 58), outline=(255, 0, 0, 220), width=3)
        draw.text((8, max(0, guard_top + 8)), "self-occlusion guard: grasp targets rejected here", fill=(255, 0, 0, 255))
    composed = Image.alpha_composite(pil, overlay).convert("RGB")
    path = out_dir / "camera_self_occlusion_guard.png"
    composed.save(path)
    return {
        "enabled": bool(args_cli.target_self_occlusion_filter),
        "image": str(path),
        "bottom_fraction": bottom_fraction,
        "guard_top_px": int(guard_top),
        "max_overlap_fraction": float(args_cli.target_self_occlusion_max_overlap_fraction),
    }


def selected_grasp_metadata(selected: SelectedGrasp | None) -> dict[str, Any] | None:
    if selected is None:
        return None
    return {
        "source": selected.source,
        "score": selected.score,
        "width_m": selected.width,
        "center_2d": selected.center_2d,
        "pose_world": selected.pose_world.tolist(),
        "metadata": selected.metadata or {},
    }


def run_graspnet_for_target(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    target: YoloInstance | None,
    out_dir: Path,
    target_identity: TargetIdentity | None = None,
) -> tuple[list[SelectedGrasp], dict[str, Any]]:
    backend_name = args_cli.grasp_backend
    target_identity_meta = target_identity_metadata(target_identity)
    metadata: dict[str, Any] = {
        "enabled": not args_cli.skip_graspnet,
        "backend": backend_name,
        "candidate_count": 0,
        "filtered_count": 0,
        "selected": None,
        "selected_candidates": [],
        "candidate_pool_count": 0,
        "source": "",
        "fallback": None,
        "centroid_fallback_enabled": centroid_fallback_enabled(),
        "error": "",
        "mask_filter": {},
        "tcp_alignment_mode": f"{backend_name}_grasp_to_kuavo_tcp_to_wrist",
        "grasp_to_tcp_calibration": active_grasp_to_tcp_metadata(),
        "gripper_tcp_urdf_calibration": RIGHT_GRIPPER_TCP_URDF_CALIBRATION,
        "gripper_tcp_offset_m": gripper_tcp_offset_m().tolist(),
        "gripper_local_tcp_offset_m": gripper_local_tcp_offset_m().tolist(),
        "contact_tuning": {
            "trash_contact_offset_m": float(args_cli.trash_contact_offset_m),
            "trash_rest_offset_m": float(args_cli.trash_rest_offset_m),
            "gripper_contact_offset_m": float(args_cli.gripper_contact_offset_m),
            "gripper_rest_offset_m": float(args_cli.gripper_rest_offset_m),
            "grasp_abort_if_object_moves_during_approach": bool(args_cli.grasp_abort_if_object_moves_during_approach),
            "grasp_object_motion_abort_threshold_m": float(args_cli.grasp_object_motion_abort_threshold_m),
        },
        "right_gripper_mount_rpy_rad": list(RIGHT_GRIPPER_INLINE_MOUNT_RPY),
        "grasp_approach_axis_local": GRASP_APPROACH_AXIS_LOCAL.tolist(),
        "grasp_frame_convention": dict(GRASP_FRAME_CONVENTION),
        "target_identity": target_identity_meta,
        "target_identity_consistency": {
            "grasp_input": False,
            "selected_grasp": False,
            "consistent": False,
            "block_reason": "",
        },
    }
    blank = np.zeros(depth_m.shape, dtype=np.uint8)
    if target is None:
        if args_cli.save_legacy_grasp_debug:
            Image.fromarray(blank).save(out_dir / "target_mask.png")
            Image.fromarray(blank).save(out_dir / "graspnet_workspace_mask.png")
            Image.fromarray(rgb).save(out_dir / "graspnet_overlay.png")
            Image.fromarray(rgb).save(out_dir / "selected_grasp_overlay.png")
        metadata["error"] = "No target selected."
        return [], metadata
    target_ok, target_reason = target_matches_identity(target, target_identity) if target_identity is not None else (True, "")
    metadata["target_identity_consistency"]["grasp_input"] = target_ok
    if not target_ok:
        if args_cli.save_legacy_grasp_debug:
            Image.fromarray(target.mask.astype(np.uint8) * 255).save(out_dir / "target_mask.png")
            Image.fromarray(blank).save(out_dir / "graspnet_workspace_mask.png")
            blocked_overlay = Image.fromarray(rgb.copy())
            ImageDraw.Draw(blocked_overlay).text((12, 12), f"Target identity mismatch: {target_reason}", fill=(255, 0, 0))
            blocked_overlay.save(out_dir / "graspnet_overlay.png")
            blocked_overlay.save(out_dir / "selected_grasp_overlay.png")
        metadata["error"] = "Target identity mismatch before grasp planning."
        metadata["target_identity_consistency"]["block_reason"] = target_reason
        return [], metadata
    target_mask = target.mask.astype(bool)
    workspace_mask = tabletop_workspace_mask(target_mask, depth_m, intrinsics, t_camera_to_world)
    if args_cli.save_legacy_grasp_debug:
        Image.fromarray(target_mask.astype(np.uint8) * 255).save(out_dir / "target_mask.png")
        Image.fromarray(workspace_mask.astype(np.uint8) * 255).save(out_dir / "graspnet_workspace_mask.png")
    points_world = mask_points_world(workspace_mask, depth_m, intrinsics, t_camera_to_world)
    world_cloud_stats = point_cloud_debug_stats(points_world)
    metadata["graspnet_input_world_point_cloud"] = world_cloud_stats
    if args_cli.save_legacy_grasp_debug:
        write_point_cloud_ply(out_dir / "debug_graspnet_input_world.ply", points_world)
        (out_dir / "debug_graspnet_input_world.json").write_text(json.dumps(world_cloud_stats, indent=2, ensure_ascii=False))
    print(
        "[DEBUG] GraspNet Input Point Cloud World "
        f"shape={points_world.shape} z_range_m={world_cloud_stats.get('z_range_m')}",
        flush=True,
    )
    if points_world.shape[0]:
        target.center_3d, target.rgbd_center_metadata = rgbd_geometric_grasp_center(points_world)
        target.points_world = points_world.astype(np.float32, copy=False)
    else:
        target.center_3d = masked_depth_center_world(target_mask, depth_m, intrinsics, t_camera_to_world)
    perceived = perceived_from_target(target, target.center_3d)
    metadata["target_mask_pixels"] = int(np.count_nonzero(target_mask))
    metadata["workspace_mask_pixels"] = int(np.count_nonzero(workspace_mask))

    fallback_candidates: list[SelectedGrasp] = []
    if centroid_fallback_enabled():
        fallback_selected, fallback_meta = centroid_fallback_grasp(target, depth_m, intrinsics, t_camera_to_world)
        metadata["fallback"] = fallback_meta
        if fallback_selected is not None:
            fallback_candidates = [fallback_selected]

    if bool(getattr(args_cli, "rgbd_center_grasp_base", False)) and not args_cli.strict_model_chain:
        selected_candidates = fallback_candidates
        metadata["source"] = "fallback" if selected_candidates else ""
        metadata["selected"] = selected_grasp_metadata(selected_candidates[0] if selected_candidates else None)
        metadata["selected_candidates"] = [selected_grasp_metadata(item) for item in selected_candidates]
        metadata["candidate_pool_count"] = len(selected_candidates)
        metadata["rgbd_center_grasp_base"] = {
            "enabled": True,
            "policy": "current_frame_v28_mask_rgbd_geometric_center_xy_median_z_minmax_midpoint",
            "learned_grasp_backend_skipped": True,
            "requires_current_camera_to_world": True,
        }
        selected_ok, selected_reason = selected_grasp_matches_identity(selected_candidates[0], target_identity) if selected_candidates else (False, "missing_rgbd_center_grasp")
        metadata["target_identity_consistency"]["selected_grasp"] = selected_ok
        metadata["target_identity_consistency"]["consistent"] = bool(target_ok and selected_ok)
        metadata["target_identity_consistency"]["block_reason"] = selected_reason if not selected_ok else ""
        if not selected_candidates:
            metadata["error"] = metadata["fallback"].get("error", "RGB-D center grasp base failed.") if metadata["fallback"] else "RGB-D center grasp base failed."
        elif not selected_ok:
            metadata["error"] = "Selected RGB-D center grasp target identity mismatch."
            return [], metadata
        return selected_candidates, metadata

    if args_cli.skip_graspnet:
        if args_cli.save_legacy_grasp_debug:
            Image.fromarray(rgb).save(out_dir / "graspnet_overlay.png")
        selected_candidates = fallback_candidates
        if args_cli.save_legacy_grasp_debug:
            draw_selected_grasp_overlay(rgb, selected_candidates[0] if selected_candidates else None, out_dir, label="fallback")
        metadata["source"] = "fallback" if selected_candidates else ""
        metadata["selected"] = selected_grasp_metadata(selected_candidates[0] if selected_candidates else None)
        metadata["selected_candidates"] = [selected_grasp_metadata(item) for item in selected_candidates]
        metadata["candidate_pool_count"] = len(selected_candidates)
        selected_ok, selected_reason = selected_grasp_matches_identity(selected_candidates[0], target_identity) if selected_candidates else (False, "missing_selected_grasp")
        metadata["target_identity_consistency"]["selected_grasp"] = selected_ok
        metadata["target_identity_consistency"]["consistent"] = bool(target_ok and selected_ok)
        metadata["target_identity_consistency"]["block_reason"] = selected_reason if not selected_ok else ""
        if not selected_candidates and centroid_fallback_enabled():
            metadata["error"] = metadata["fallback"].get("error", "Centroid fallback failed.") if metadata["fallback"] else "Centroid fallback failed."
        elif not selected_candidates:
            metadata["error"] = "Learned backend skipped and centroid fallback is disabled."
        elif not selected_ok:
            metadata["error"] = "Selected fallback grasp target identity mismatch."
            return [], metadata
        return selected_candidates, metadata
    try:
        wrapper = get_grasp_backend_wrapper()
        if backend_name == "graspnet" and isinstance(wrapper, GraspNetWrapper):
            debug_dir = out_dir if args_cli.save_legacy_grasp_debug else None
            candidates = wrapper.detect(rgb, depth_m, intrinsics, workspace_mask, debug_dir=debug_dir)
            metadata["graspnet_input_camera_point_cloud"] = dict(wrapper.last_debug_info)
        else:
            candidates = wrapper.detect(rgb, depth_m, intrinsics, workspace_mask)
        metadata["candidate_count"] = int(len(candidates.scores))
        candidate_points_world = None
        if len(candidates.scores) > 0:
            candidate_points_cam = candidates.poses[:, :3, 3]
            candidate_points_world_h = np.concatenate([candidate_points_cam, np.ones((candidate_points_cam.shape[0], 1), dtype=np.float32)], axis=1)
            candidate_points_world = (t_camera_to_world @ candidate_points_world_h.T).T[:, :3].astype(np.float32)
        filter_result = filter_grasps_by_mask_relaxed(
            candidates,
            workspace_mask,
            margin=5,
            candidate_points_world=candidate_points_world,
            object_points_world=points_world,
            min_filtered_grasps=args_cli.min_filtered_grasps,
            distance_threshold_m=args_cli.mask_distance_threshold_m,
        )
        filtered = filter_result.candidates
        metadata["mask_filter"] = filter_result.metadata
        metadata["filtered_count"] = int(len(filtered.scores))
        selected_candidates = rank_grasps(
            filtered,
            perceived,
            t_camera_to_world=t_camera_to_world,
            table_surface_z=TABLE_SURFACE_Z,
            top_k=max(args_cli.ik_prescreen_top_k, args_cli.max_grasp_retries),
            source=backend_name,
        )
        if fallback_candidates:
            selected_candidates.extend(fallback_candidates)
        if args_cli.save_legacy_grasp_debug:
            draw_grasp_overlays(rgb, candidates, selected_candidates[0] if selected_candidates else None, out_dir)
        if selected_candidates:
            metadata["source"] = selected_candidates[0].source
            metadata["selected"] = selected_grasp_metadata(selected_candidates[0])
            metadata["selected_candidates"] = [selected_grasp_metadata(item) for item in selected_candidates]
            metadata["candidate_pool_count"] = len(selected_candidates)
            selected_ok, selected_reason = selected_grasp_matches_identity(selected_candidates[0], target_identity)
            metadata["target_identity_consistency"]["selected_grasp"] = selected_ok
            metadata["target_identity_consistency"]["consistent"] = bool(target_ok and selected_ok)
            metadata["target_identity_consistency"]["block_reason"] = selected_reason if not selected_ok else ""
            if not selected_ok:
                metadata["error"] = "Selected learned grasp target identity mismatch."
                return [], metadata
        else:
            metadata["error"] = "No physically valid grasp remained after filtering."
        return selected_candidates, metadata
    except Exception as exc:
        metadata["error"] = repr(exc)
        metadata["hint"] = grasp_backend_failure_hint(exc)
        if args_cli.save_legacy_grasp_debug:
            pil = Image.fromarray(rgb.copy())
            ImageDraw.Draw(pil).text((12, 12), f"{backend_name} error: {exc!r}", fill=(255, 0, 0))
            pil.save(out_dir / "graspnet_overlay.png")
            draw_selected_grasp_overlay(rgb, fallback_candidates[0] if fallback_candidates else None, out_dir, label="fallback" if fallback_candidates else "none")
        if fallback_candidates:
            metadata["source"] = "fallback"
            metadata["selected"] = selected_grasp_metadata(fallback_candidates[0])
            metadata["selected_candidates"] = [selected_grasp_metadata(item) for item in fallback_candidates]
            metadata["candidate_pool_count"] = len(fallback_candidates)
            selected_ok, selected_reason = selected_grasp_matches_identity(fallback_candidates[0], target_identity)
            metadata["target_identity_consistency"]["selected_grasp"] = selected_ok
            metadata["target_identity_consistency"]["consistent"] = bool(target_ok and selected_ok)
            metadata["target_identity_consistency"]["block_reason"] = selected_reason if not selected_ok else ""
            if not selected_ok:
                return [], metadata
        return fallback_candidates, metadata


def draw_grasp_overlays(rgb: np.ndarray, candidates: GraspCandidates, selected: SelectedGrasp | None, out_dir: Path) -> None:
    pil = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(pil)
    order = np.argsort(candidates.scores)[::-1][: min(args_cli.graspnet_max_draw, len(candidates.scores))]
    for rank, idx in enumerate(order):
        u, v = candidates.centers_2d[idx]
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        u_f = float(u)
        v_f = float(v)
        color = color_for_index(rank)
        r = 5 if rank else 9
        draw.ellipse((u_f - r, v_f - r, u_f + r, v_f + r), outline=color, width=3)
        draw.text((u_f + 7, v_f + 4), f"{rank}:{float(candidates.scores[idx]):.2f}", fill=color)
    pil.save(out_dir / "graspnet_overlay.png")
    selected_pil = pil.copy()
    selected_draw = ImageDraw.Draw(selected_pil)
    if selected is not None:
        u, v = selected.center_2d
        selected_draw.ellipse((u - 14, v - 14, u + 14, v + 14), outline=(255, 0, 0), width=5)
        selected_draw.text((u + 16, v + 8), f"selected {selected.score:.2f}", fill=(255, 0, 0))
    selected_pil.save(out_dir / "selected_grasp_overlay.png")


def run_yolo_vlm_stage(rgb: np.ndarray, out_dir: Path) -> list[YoloInstance]:
    instances = run_yolo(rgb, out_dir)
    classify_instances(rgb, instances, out_dir)
    return instances


def run_graspnet_stage(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    target: YoloInstance | None,
    out_dir: Path,
    target_identity: TargetIdentity | None = None,
) -> tuple[list[SelectedGrasp], dict[str, Any]]:
    return run_graspnet_for_target(rgb, depth_m, intrinsics, t_camera_to_world, target, out_dir, target_identity)


def reset_scene(scene: InteractiveScene) -> None:
    global _KINEMATIC_STABLE_POSE_XYYAW, _NAV_SMOOTHED_CMD
    _KINEMATIC_STABLE_POSE_XYYAW = None
    _NAV_SMOOTHED_CMD = (0.0, 0.0)
    robot: Articulation = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    scene.reset()


def stabilize_robot_base(robot: Articulation, locked_root_pose_w: torch.Tensor | None = None) -> None:
    root_state = robot.data.root_state_w.clone()
    for env_id in range(root_state.shape[0]):
        if locked_root_pose_w is not None:
            root_state[env_id, 0:2] = locked_root_pose_w[env_id, 0:2]
            yaw = yaw_from_quat_wxyz(locked_root_pose_w[env_id, 3:7])
        else:
            yaw = yaw_from_quat_wxyz(root_state[env_id, 3:7])
        root_state[env_id, 2] = ROBOT_STABILIZED_BASE_Z
        root_state[env_id, 3:7] = torch.tensor(quat_wxyz_from_yaw(yaw), device=robot.device)
        root_state[env_id, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])


def stabilize_robot_base_for_nav(robot: Articulation) -> None:
    if args_cli.nav_actuation_mode != "root_velocity" and not args_cli.wheel_root_stabilization:
        return
    root_state = robot.data.root_state_w.clone()
    for env_id in range(root_state.shape[0]):
        yaw = yaw_from_quat_wxyz(root_state[env_id, 3:7])
        root_state[env_id, 2] = ROBOT_STABILIZED_BASE_Z
        root_state[env_id, 3:7] = torch.tensor(quat_wxyz_from_yaw(yaw), device=robot.device)
        root_state[env_id, 9] = 0.0
        root_state[env_id, 10] = 0.0
        root_state[env_id, 11] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    if args_cli.nav_actuation_mode != "root_velocity":
        robot.write_root_velocity_to_sim(root_state[:, 7:])


def min_jerk(alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def gripper_command_width(gripper: AttachedParallelGripper, requested_width_m: float, *, clearance_m: float | None = None) -> float:
    return gripper.open_width_for_grasp(float(requested_width_m), clearance_m=clearance_m)


def gripper_fully_open_width(gripper: AttachedParallelGripper) -> float:
    return gripper.clamp_width(float(gripper.limits.max_width_m))


def run_slow_gripper_close(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    gripper: AttachedParallelGripper,
    tracker: Task319StateTracker,
    hold_joint_target: torch.Tensor,
    locked_root_pose_w: torch.Tensor,
    *,
    start_width_m: float,
    steps: int,
) -> dict[str, Any]:
    steps = max(1, int(steps))
    start_width = gripper.clamp_width(start_width_m)
    end_width = gripper.limits.min_width_m
    actual_start_width = gripper.current_width()
    for step in range(steps):
        alpha = min_jerk((step + 1) / steps)
        width = (1.0 - alpha) * start_width + alpha * end_width
        robot.set_joint_position_target(hold_joint_target)
        gripper.set_width(width)
        stabilize_robot_base(robot, locked_root_pose_w)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim.get_physics_dt())
        record_video_frame(scene)
        gui_playback_tick(sim.get_physics_dt())
        tracker.tick()
    actual_end_width = gripper.current_width()
    return {
        "close_mode": "smooth_width_ramp",
        "close_steps": int(steps),
        "close_start_width_m": float(start_width),
        "close_end_width_m": float(end_width),
        "close_start_actual_width_m": float(actual_start_width),
        "close_final_actual_width_m": float(actual_end_width),
    }


def run_closed_gripper_settle(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    gripper: AttachedParallelGripper,
    tracker: Task319StateTracker,
    hold_joint_target: torch.Tensor,
    locked_root_pose_w: torch.Tensor,
    *,
    steps: int,
) -> dict[str, Any]:
    steps = max(0, int(steps))
    start_width = gripper.current_width()
    for _ in range(steps):
        robot.set_joint_position_target(hold_joint_target)
        gripper.close()
        stabilize_robot_base(robot, locked_root_pose_w)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim.get_physics_dt())
        record_video_frame(scene)
        gui_playback_tick(sim.get_physics_dt())
        tracker.tick()
    end_width = gripper.current_width()
    return {
        "settle_mode": "closed_gripper_contact_stabilization",
        "settle_steps": int(steps),
        "settle_start_actual_width_m": float(start_width),
        "settle_final_actual_width_m": float(end_width),
    }


def gripper_contact_threshold_m(candidate: SelectedGrasp) -> float:
    return float(max(float(args_cli.gripper_contact_min_width_m), float(args_cli.gripper_contact_width_fraction) * float(candidate.width)))


def evaluate_gripper_contact_before_lift(close_segment: dict[str, Any], candidate: SelectedGrasp) -> tuple[bool, dict[str, Any], str]:
    threshold = gripper_contact_threshold_m(candidate)
    final_width = float(close_segment.get("close_final_actual_width_m", float("nan")))
    metadata = {
        "required": bool(args_cli.gripper_require_contact_before_lift),
        "candidate_width_m": float(candidate.width),
        "actual_final_width_m": final_width,
        "contact_threshold_m": threshold,
        "min_width_m": float(args_cli.gripper_contact_min_width_m),
        "width_fraction": float(args_cli.gripper_contact_width_fraction),
    }
    if not bool(args_cli.gripper_require_contact_before_lift):
        metadata["passed"] = True
        metadata["reason"] = "disabled"
        return True, metadata, ""
    if not math.isfinite(final_width):
        metadata["passed"] = False
        metadata["reason"] = "actual_width_not_finite"
        return False, metadata, "GRASP gripper contact check failed: actual jaw width is not finite."
    passed = final_width >= threshold
    metadata["passed"] = bool(passed)
    metadata["reason"] = "actual_jaw_width_retained" if passed else "gripper_closed_to_empty_width"
    if passed:
        return True, metadata, ""
    return False, metadata, (
        "GRASP gripper did not retain object contact after close; refusing to lift "
        f"(actual_width={final_width:.4f}m, required>={threshold:.4f}m)."
    )


def grasp_close_error_threshold_m(candidate: SelectedGrasp) -> float:
    base_threshold = max(float(args_cli.grasp_error_threshold_m), 1.0e-3)
    width = float(getattr(candidate, "width", float("nan")))
    if not bool(args_cli.gripper_require_contact_before_lift) or not math.isfinite(width) or width <= 0.0:
        return base_threshold
    coverage_threshold = min(0.045, max(0.0, 0.65 * width))
    return max(base_threshold, coverage_threshold)


def clamp_joint_step(desired: torch.Tensor, previous: torch.Tensor, max_step: float) -> torch.Tensor:
    return previous + torch.clamp(desired - previous, min=-max_step, max=max_step)


def interpolate_quat_shortest_path(q0: torch.Tensor, q1: torch.Tensor, alpha: float) -> torch.Tensor:
    q1 = torch.where(torch.sum(q0 * q1, dim=1, keepdim=True) < 0.0, -q1, q1)
    q = (1.0 - alpha) * q0 + alpha * q1
    return torch.nn.functional.normalize(q, dim=1)


def pose_to_torch_command(target_pose_w: np.ndarray, robot: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    target_pos_w = torch.tensor(target_pose_w[:3, 3], dtype=torch.float32, device=robot.device).repeat(robot.num_instances, 1)
    target_rot_w = torch.tensor(target_pose_w[:3, :3], dtype=torch.float32, device=robot.device).repeat(robot.num_instances, 1, 1)
    target_quat_w = quat_from_matrix(target_rot_w)
    root_pose_w = robot.data.root_pose_w
    return subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_pos_w, target_quat_w)


def gripper_tcp_offset_m() -> np.ndarray:
    return np.asarray(args_cli.gripper_tcp_offset, dtype=np.float32)


def gripper_local_tcp_offset_m() -> np.ndarray:
    return np.asarray(args_cli.gripper_local_tcp_offset, dtype=np.float32)


def tcp_pose_to_wrist_pose(tcp_pose_w: np.ndarray, wrist_rot_w: np.ndarray | None = None) -> np.ndarray:
    wrist_pose_w = np.array(tcp_pose_w, dtype=np.float32).copy()
    if wrist_rot_w is not None:
        wrist_pose_w[:3, :3] = np.asarray(wrist_rot_w, dtype=np.float32)
    else:
        wrist_pose_w[:3, :3] = np.asarray(tcp_pose_w[:3, :3], dtype=np.float32) @ RIGHT_GRIPPER_INLINE_MOUNT_ROT.T
    wrist_pose_w[:3, 3] = wrist_pose_w[:3, 3] - wrist_pose_w[:3, :3] @ gripper_tcp_offset_m()
    return wrist_pose_w


def run_ik_segment(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ik_controller: DifferentialIKController,
    robot_entity_cfg: SceneEntityCfg,
    ee_body_id: int,
    ee_jacobi_idx: int,
    target_pose_w: np.ndarray,
    steps: int,
    *,
    gripper: AttachedParallelGripper | None = None,
    gripper_width: float | None = None,
    locked_root_pose_w: torch.Tensor | None = None,
    debug_target_pose_w: np.ndarray | None = None,
    target_tcp_pose_w: np.ndarray | None = None,
    control_tcp_position: bool = False,
) -> dict[str, Any]:
    sim_dt = sim.get_physics_dt()
    target_pos_b, target_quat_b = pose_to_torch_command(target_pose_w, robot)
    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
    start_pos_b, start_quat_b = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    tcp_position_control = bool(control_tcp_position and ik_controller.action_dim == 3 and target_tcp_pose_w is not None)
    start_tcp_pos_w = None
    target_tcp_pos_w = None
    target_tcp_np = None
    if target_tcp_pose_w is not None:
        target_tcp_np = np.asarray(target_tcp_pose_w, dtype=np.float32).reshape(4, 4)[:3, 3].copy()
        start_tcp_pos_w = torch.tensor(tcp_pose_matrix(robot, ee_body_id)[:3, 3], dtype=torch.float32, device=robot.device).repeat(robot.num_instances, 1)
        target_tcp_pos_w = torch.tensor(target_tcp_np, dtype=torch.float32, device=robot.device).repeat(robot.num_instances, 1)
    tcp_offset = torch.tensor(gripper_tcp_offset_m(), dtype=torch.float32, device=robot.device).reshape(1, 3, 1).repeat(robot.num_instances, 1, 1)
    previous_joint_target = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
    locked_joint_target = robot.data.joint_pos.clone()
    max_joint_step = 0.0
    final_joint_error = float("nan")
    final_pos_error = float("nan")
    final_wrist_pos_error = float("nan")
    final_tcp_pos_error = float("nan")
    final_pos_b = start_pos_b.clone()
    ik_controller.reset()
    command = torch.zeros(robot.num_instances, ik_controller.action_dim, device=robot.device)
    for step in range(max(1, steps)):
        alpha = min_jerk(step / max(1, steps - 1))
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        if tcp_position_control and start_tcp_pos_w is not None and target_tcp_pos_w is not None:
            desired_tcp_pos_w = (1.0 - alpha) * start_tcp_pos_w + alpha * target_tcp_pos_w
            wrist_rot_w = matrix_from_quat(ee_pose_w[:, 3:7])
            desired_wrist_pos_w = desired_tcp_pos_w - torch.bmm(wrist_rot_w, tcp_offset).squeeze(-1)
            desired_pos_b, _ = subtract_frame_transforms(
                robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], desired_wrist_pos_w, ee_pose_w[:, 3:7]
            )
            desired_quat_b = ee_quat_b
        else:
            desired_pos_b = (1.0 - alpha) * start_pos_b + alpha * target_pos_b
            desired_quat_b = interpolate_quat_shortest_path(start_quat_b, target_quat_b, alpha)
        command[:, 0:3] = desired_pos_b
        if ik_controller.action_dim == 3:
            ik_controller.set_command(command, ee_quat=ee_quat_b)
        else:
            command[:, 3:7] = desired_quat_b
            ik_controller.set_command(command)
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, robot.data.joint_pos[:, robot_entity_cfg.joint_ids])
        joint_pos_des = clamp_joint_step(joint_pos_des, previous_joint_target, args_cli.max_joint_step)
        max_joint_step = max(max_joint_step, float(torch.max(torch.abs(joint_pos_des - previous_joint_target)).item()))
        previous_joint_target = joint_pos_des.clone()

        robot.set_joint_position_target(locked_joint_target)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        if gripper is not None and gripper_width is not None:
            gripper.set_width(gripper_width)
        stabilize_robot_base(robot, locked_root_pose_w)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        if debug_target_pose_w is not None and (step % 4 == 0 or step == max(1, steps) - 1):
            render_calibration_debug_markers(scene, debug_target_pose_w, tcp_pose_matrix(robot, ee_body_id))
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        final_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        final_wrist_pos_error = float(torch.linalg.norm(target_pos_b - final_pos_b, dim=1).max().item())
        if target_tcp_np is not None:
            final_tcp_pos_error = float(np.linalg.norm(tcp_pose_matrix(robot, ee_body_id)[:3, 3] - target_tcp_np))
        final_pos_error = final_tcp_pos_error if tcp_position_control else final_wrist_pos_error
    return {
        "max_joint_step_rad": max_joint_step,
        "final_pos_error_m": final_pos_error,
        "final_wrist_pos_error_m": final_wrist_pos_error,
        "final_tcp_pos_error_m": final_tcp_pos_error,
        "position_error_reference": "tcp" if tcp_position_control else "wrist",
        "ik_command_type": str(ik_controller.cfg.command_type),
        "tcp_position_control": tcp_position_control,
        "start_pos_b": start_pos_b[0].detach().cpu().tolist(),
        "target_pos_b": target_pos_b[0].detach().cpu().tolist(),
        "final_pos_b": final_pos_b[0].detach().cpu().tolist(),
        "target_tcp_world_m": target_tcp_np.astype(float).tolist() if target_tcp_np is not None else None,
        "final_tcp_world_m": tcp_pose_matrix(robot, ee_body_id)[:3, 3].astype(float).tolist() if target_tcp_np is not None else None,
    }


def run_tcp_position_servo_descent(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ik_controller: DifferentialIKController,
    robot_entity_cfg: SceneEntityCfg,
    ee_body_id: int,
    ee_jacobi_idx: int,
    target_tcp_pose_w: np.ndarray,
    object_target_tcp_w: np.ndarray,
    steps: int,
    *,
    gripper: AttachedParallelGripper | None = None,
    gripper_width: float | None = None,
    locked_root_pose_w: torch.Tensor | None = None,
    debug_target_pose_w: np.ndarray | None = None,
    proximity_stop_distance_m: float | None = None,
    max_joint_step: float | None = None,
) -> dict[str, Any]:
    sim_dt = sim.get_physics_dt()
    if ik_controller.action_dim != 3:
        return {
            "executed_steps": 0,
            "final_pos_error_m": float("inf"),
            "final_tcp_pos_error_m": float("inf"),
            "position_error_reference": "tcp",
            "aborted": True,
            "abort_reason": f"position-only servo requires a 3D position IK controller, got action_dim={ik_controller.action_dim}",
        }

    target_tcp_np = np.asarray(target_tcp_pose_w, dtype=np.float32).reshape(4, 4)[:3, 3].copy()
    object_target_np = np.asarray(object_target_tcp_w, dtype=np.float32).reshape(3).copy()
    start_tcp_pose = tcp_pose_matrix(robot, ee_body_id).astype(np.float32)
    start_tcp_np = start_tcp_pose[:3, 3].copy()
    lock_xy_np = start_tcp_np[:2].copy()
    servo_target_np = target_tcp_np.copy()
    servo_target_np[:2] = lock_xy_np
    stop_distance = float(proximity_stop_distance_m if proximity_stop_distance_m is not None else args_cli.mind_sort_gripper_proximity_assist_max_distance_m)
    if not math.isfinite(stop_distance) or stop_distance <= 0.0:
        stop_distance = float("inf")
    joint_step_limit = float(args_cli.max_joint_step)
    if max_joint_step is not None and math.isfinite(float(max_joint_step)) and float(max_joint_step) > 0.0:
        joint_step_limit = min(joint_step_limit, float(max_joint_step))

    tcp_offset = torch.tensor(gripper_tcp_offset_m(), dtype=torch.float32, device=robot.device).reshape(1, 3, 1).repeat(robot.num_instances, 1, 1)
    previous_joint_target = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
    locked_joint_target = robot.data.joint_pos.clone()
    command = torch.zeros(robot.num_instances, ik_controller.action_dim, device=robot.device)
    ik_controller.reset()

    executed_steps = 0
    max_joint_step_seen = 0.0
    max_xy_error = 0.0
    closest_object_distance = float("inf")
    closest_vertical_distance = float("inf")
    closest_tcp_world = start_tcp_np.copy()
    reached_proximity = False
    reached_precise_target = False
    aborted = False
    abort_reason = ""
    steps = max(1, int(steps))
    for step in range(steps):
        alpha = min_jerk(float(step + 1) / float(steps))
        desired_tcp_np = start_tcp_np.copy()
        desired_tcp_np[:2] = lock_xy_np
        desired_tcp_np[2] = float((1.0 - alpha) * start_tcp_np[2] + alpha * servo_target_np[2])

        desired_tcp_w = torch.tensor(desired_tcp_np, dtype=torch.float32, device=robot.device).repeat(robot.num_instances, 1)
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        wrist_rot_w = matrix_from_quat(ee_pose_w[:, 3:7])
        desired_wrist_pos_w = desired_tcp_w - torch.bmm(wrist_rot_w, tcp_offset).squeeze(-1)
        desired_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], desired_wrist_pos_w, ee_pose_w[:, 3:7]
        )
        command[:, 0:3] = desired_pos_b
        ik_controller.set_command(command, ee_quat=ee_quat_b)
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, robot.data.joint_pos[:, robot_entity_cfg.joint_ids])
        if not torch.isfinite(joint_pos_des).all():
            aborted = True
            abort_reason = "position-only TCP servo produced non-finite joint targets"
            break
        joint_pos_des = clamp_joint_step(joint_pos_des, previous_joint_target, joint_step_limit)
        max_joint_step_seen = max(max_joint_step_seen, float(torch.max(torch.abs(joint_pos_des - previous_joint_target)).item()))
        previous_joint_target = joint_pos_des.clone()

        robot.set_joint_position_target(locked_joint_target)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        if gripper is not None and gripper_width is not None:
            gripper.set_width(gripper_width)
        stabilize_robot_base(robot, locked_root_pose_w)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        executed_steps += 1
        if debug_target_pose_w is not None and (step % 4 == 0 or step == steps - 1):
            render_calibration_debug_markers(scene, debug_target_pose_w, tcp_pose_matrix(robot, ee_body_id))

        tcp_pos = tcp_pose_matrix(robot, ee_body_id)[:3, 3].astype(np.float32)
        xy_error = float(np.linalg.norm(tcp_pos[:2] - lock_xy_np))
        vertical_distance = float(np.linalg.norm(tcp_pos - servo_target_np))
        object_distance = float(np.linalg.norm(tcp_pos - object_target_np))
        max_xy_error = max(max_xy_error, xy_error)
        if object_distance < closest_object_distance:
            closest_object_distance = object_distance
            closest_tcp_world = tcp_pos.copy()
        closest_vertical_distance = min(closest_vertical_distance, vertical_distance)
        reached_precise_target = bool(vertical_distance <= float(args_cli.grasp_error_threshold_m))
        reached_proximity = bool(object_distance <= stop_distance)
        if reached_precise_target or reached_proximity:
            break

    final_tcp_pose = tcp_pose_matrix(robot, ee_body_id).astype(np.float32)
    final_tcp_np = final_tcp_pose[:3, 3].copy()
    final_vertical_error = float(np.linalg.norm(final_tcp_np - servo_target_np))
    final_object_distance = float(np.linalg.norm(final_tcp_np - object_target_np))
    if final_object_distance < closest_object_distance:
        closest_object_distance = final_object_distance
        closest_tcp_world = final_tcp_np.copy()
    reached_proximity = bool(reached_proximity or closest_object_distance <= stop_distance)
    return {
        "max_joint_step_rad": max_joint_step_seen,
        "executed_steps": int(executed_steps),
        "final_pos_error_m": final_vertical_error,
        "final_tcp_pos_error_m": final_vertical_error,
        "final_tcp_to_object_target_m": final_object_distance,
        "closest_tcp_to_object_target_m": closest_object_distance,
        "closest_tcp_to_vertical_target_m": closest_vertical_distance,
        "closest_tcp_world_m": closest_tcp_world.astype(float).tolist(),
        "position_error_reference": "tcp_position_servo_vertical_target",
        "ik_command_type": str(ik_controller.cfg.command_type),
        "tcp_position_control": True,
        "start_tcp_world_m": start_tcp_np.astype(float).tolist(),
        "target_tcp_world_m": servo_target_np.astype(float).tolist(),
        "object_grasp_target_tcp_world_m": object_target_np.astype(float).tolist(),
        "final_tcp_world_m": final_tcp_np.astype(float).tolist(),
        "locked_xy_world_m": lock_xy_np.astype(float).tolist(),
        "locked_xy_minus_object_target_xy_m": (lock_xy_np - object_target_np[:2]).astype(float).tolist(),
        "locked_xy_object_target_distance_m": float(np.linalg.norm(lock_xy_np - object_target_np[:2])),
        "max_cartesian_xy_error_m": max_xy_error,
        "proximity_stop_distance_m": stop_distance,
        "reached_proximity_stop": reached_proximity,
        "reached_precise_vertical_target": bool(reached_precise_target),
        "aborted": aborted,
        "abort_reason": abort_reason,
    }


def run_ik_segment_until_converged(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ik_controller: DifferentialIKController,
    robot_entity_cfg: SceneEntityCfg,
    ee_body_id: int,
    ee_jacobi_idx: int,
    target_pose_w: np.ndarray,
    steps: int,
    error_threshold_m: float,
    *,
    gripper: AttachedParallelGripper | None = None,
    gripper_width: float | None = None,
    locked_root_pose_w: torch.Tensor | None = None,
    debug_target_pose_w: np.ndarray | None = None,
    target_tcp_pose_w: np.ndarray | None = None,
    control_tcp_position: bool = False,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    first = run_ik_segment(
        sim,
        scene,
        robot,
        ik_controller,
        robot_entity_cfg,
        ee_body_id,
        ee_jacobi_idx,
        target_pose_w,
        steps,
        gripper=gripper,
        gripper_width=gripper_width,
        locked_root_pose_w=locked_root_pose_w,
        debug_target_pose_w=debug_target_pose_w,
        target_tcp_pose_w=target_tcp_pose_w,
        control_tcp_position=control_tcp_position,
    )
    records.append(first)
    best_index = 0
    best_error = float(first.get("final_pos_error_m", float("inf")))
    best_state = capture_dry_run_state(scene, robot)
    remaining = max(0, int(args_cli.arm_motion_converge_extra_steps))
    chunk_steps = max(1, int(args_cli.arm_motion_converge_chunk_steps))
    regression_tolerance = max(0.002, 0.25 * max(float(error_threshold_m), 1.0e-3))
    while remaining > 0:
        final_error = float(records[-1].get("final_pos_error_m", float("inf")))
        if math.isfinite(final_error) and final_error <= float(error_threshold_m):
            break
        current_steps = min(chunk_steps, remaining)
        current = run_ik_segment(
            sim,
            scene,
            robot,
            ik_controller,
            robot_entity_cfg,
            ee_body_id,
            ee_jacobi_idx,
            target_pose_w,
            current_steps,
            gripper=gripper,
            gripper_width=gripper_width,
            locked_root_pose_w=locked_root_pose_w,
            debug_target_pose_w=debug_target_pose_w,
            target_tcp_pose_w=target_tcp_pose_w,
            control_tcp_position=control_tcp_position,
        )
        records.append(current)
        current_error = float(current.get("final_pos_error_m", float("inf")))
        if math.isfinite(current_error) and (not math.isfinite(best_error) or current_error < best_error):
            best_error = current_error
            best_index = len(records) - 1
            best_state = capture_dry_run_state(scene, robot)
        if (
            bool(args_cli.arm_motion_stop_on_regression)
            and
            math.isfinite(current_error)
            and math.isfinite(best_error)
            and current_error > best_error + regression_tolerance
        ):
            break
        remaining -= current_steps

    restored_best_state = best_index != len(records) - 1
    if restored_best_state:
        restore_dry_run_state(scene, robot, best_state)
    merged = dict(records[best_index])
    merged["subsegments"] = records
    merged["subsegment_count"] = len(records)
    merged["total_steps"] = int(steps) + sum(min(chunk_steps, max(0, int(args_cli.arm_motion_converge_extra_steps)) - i * chunk_steps) for i in range(len(records) - 1))
    merged["requested_steps"] = int(steps)
    merged["extra_steps_used"] = max(0, int(merged["total_steps"]) - int(steps))
    merged["error_threshold_m"] = float(error_threshold_m)
    merged["converged"] = bool(math.isfinite(float(merged.get("final_pos_error_m", float("inf")))) and float(merged.get("final_pos_error_m", float("inf"))) <= float(error_threshold_m))
    merged["max_joint_step_rad"] = max(float(item.get("max_joint_step_rad", 0.0) or 0.0) for item in records)
    merged["best_subsegment_index"] = int(best_index)
    merged["restored_best_state"] = bool(restored_best_state)
    merged["convergence_regression_tolerance_m"] = float(regression_tolerance)
    return merged


def run_joint_segment(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    robot_entity_cfg: SceneEntityCfg,
    target_joint_pos: torch.Tensor,
    steps: int,
    *,
    gripper: AttachedParallelGripper | None = None,
    gripper_width: float | None = None,
    locked_root_pose_w: torch.Tensor | None = None,
    target_pose_w: np.ndarray | None = None,
) -> dict[str, Any]:
    sim_dt = sim.get_physics_dt()
    joint_ids = robot_entity_cfg.joint_ids
    start_joint_pos = robot.data.joint_pos[:, joint_ids].clone()
    target_joint_pos = target_joint_pos.to(device=robot.device).clone()
    if target_joint_pos.ndim == 1:
        target_joint_pos = target_joint_pos.unsqueeze(0).repeat(robot.num_instances, 1)
    locked_joint_target = robot.data.joint_pos.clone()
    previous_joint_target = start_joint_pos.clone()
    max_joint_step = 0.0
    final_pos_error = float("nan")
    target_pos_b = None
    if target_pose_w is not None:
        target_pos_b, _ = pose_to_torch_command(target_pose_w, robot)
    final_pos_b = None
    for step in range(max(1, steps)):
        alpha = min_jerk(step / max(1, steps - 1))
        joint_pos_des = (1.0 - alpha) * start_joint_pos + alpha * target_joint_pos
        max_joint_step = max(max_joint_step, float(torch.max(torch.abs(joint_pos_des - previous_joint_target)).item()))
        previous_joint_target = joint_pos_des.clone()
        robot.set_joint_position_target(locked_joint_target)
        robot.set_joint_position_target(joint_pos_des, joint_ids=joint_ids)
        if gripper is not None and gripper_width is not None:
            gripper.set_width(gripper_width)
        stabilize_robot_base(robot, locked_root_pose_w)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        if target_pos_b is not None:
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            final_pos_b, _ = subtract_frame_transforms(
                robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            final_pos_error = float(torch.linalg.norm(target_pos_b - final_pos_b, dim=1).max().item())
    final_joint_error = float(torch.max(torch.abs(robot.data.joint_pos[:, joint_ids] - target_joint_pos)).item())
    return {
        "max_joint_step_rad": max_joint_step,
        "final_joint_error_rad": final_joint_error,
        "target_joint_pos": target_joint_pos[0].detach().cpu().tolist(),
        "final_joint_pos": robot.data.joint_pos[0, joint_ids].detach().cpu().tolist(),
        "final_pos_error_m": final_pos_error,
        "target_pos_b": target_pos_b[0].detach().cpu().tolist() if target_pos_b is not None else None,
        "final_pos_b": final_pos_b[0].detach().cpu().tolist() if final_pos_b is not None else None,
    }


def robot_root_pose_matrix(robot: Articulation) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = robot.data.root_pose_w[0, :3].detach().cpu().numpy().astype(np.float32)
    pose[:3, :3] = matrix_from_quat(robot.data.root_pose_w[:, 3:7])[0].detach().cpu().numpy().astype(np.float32)
    return pose


def world_pose_to_robot_base_pose(robot: Articulation, pose_w: np.ndarray) -> np.ndarray:
    return (np.linalg.inv(robot_root_pose_matrix(robot)) @ np.asarray(pose_w, dtype=np.float32).reshape(4, 4)).astype(np.float32)


def curobo_cuboid_pose_in_robot_base(robot: Articulation, center_w: tuple[float, float, float], rot_w: np.ndarray | None = None) -> list[float]:
    pose_w = np.eye(4, dtype=np.float32)
    pose_w[:3, 3] = np.asarray(center_w, dtype=np.float32)
    if rot_w is not None:
        pose_w[:3, :3] = np.asarray(rot_w, dtype=np.float32).reshape(3, 3)
    pose_b = world_pose_to_robot_base_pose(robot, pose_w)
    quat = quat_wxyz_from_matrix(pose_b[:3, :3])
    return [float(pose_b[0, 3]), float(pose_b[1, 3]), float(pose_b[2, 3]), float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


def curobo_world_for_current_robot(robot: Articulation):
    if not bool(args_cli.curobo_plan_table_obstacles):
        return None
    cuboids = [
        {
            "name": "sorting_table_top",
            "pose": curobo_cuboid_pose_in_robot_base(robot, (1.80, 0.0, TABLE_TOP_CENTER_Z)),
            "dims": [1.86, 0.96, TABLE_TOP_THICKNESS + 0.04],
        },
        {
            "name": "sorting_table_leg_fl",
            "pose": curobo_cuboid_pose_in_robot_base(robot, (0.96, -0.38, 0.24)),
            "dims": [0.09, 0.09, 0.50],
        },
        {
            "name": "sorting_table_leg_fr",
            "pose": curobo_cuboid_pose_in_robot_base(robot, (2.64, -0.38, 0.24)),
            "dims": [0.09, 0.09, 0.50],
        },
        {
            "name": "sorting_table_leg_bl",
            "pose": curobo_cuboid_pose_in_robot_base(robot, (0.96, 0.38, 0.24)),
            "dims": [0.09, 0.09, 0.50],
        },
        {
            "name": "sorting_table_leg_br",
            "pose": curobo_cuboid_pose_in_robot_base(robot, (2.64, 0.38, 0.24)),
            "dims": [0.09, 0.09, 0.50],
        },
    ]
    return build_world_cuboids(cuboids)


def run_curobo_joint_trajectory(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    robot_entity_cfg: SceneEntityCfg,
    joint_positions: np.ndarray,
    *,
    min_steps: int,
    gripper: AttachedParallelGripper | None = None,
    gripper_width: float | None = None,
    locked_root_pose_w: torch.Tensor | None = None,
    target_tcp_pose_w: np.ndarray | None = None,
    max_joint_step_override: float | None = None,
    cartesian_lock_xy_w: np.ndarray | None = None,
    max_cartesian_xy_error_m: float | None = None,
    abort_on_cartesian_xy_error: bool = False,
    object_target_tcp_w: np.ndarray | None = None,
    proximity_stop_distance_m: float | None = None,
) -> dict[str, Any]:
    sim_dt = sim.get_physics_dt()
    joint_ids = robot_entity_cfg.joint_ids
    plan = np.asarray(joint_positions, dtype=np.float32)
    if plan.ndim != 2 or plan.shape[1] != len(joint_ids) or plan.shape[0] == 0:
        return {
            "max_joint_step_rad": 0.0,
            "executed_steps": 0,
            "final_pos_error_m": float("inf"),
            "final_tcp_pos_error_m": float("inf"),
            "position_error_reference": "tcp",
            "reason": f"Invalid cuRobo joint plan shape {plan.shape}, expected [N,{len(joint_ids)}].",
        }

    previous_joint_target = robot.data.joint_pos[:, joint_ids].clone()
    locked_joint_target = robot.data.joint_pos.clone()
    target_tcp_np = np.asarray(target_tcp_pose_w, dtype=np.float32).reshape(4, 4)[:3, 3].copy() if target_tcp_pose_w is not None else None
    object_target_np = np.asarray(object_target_tcp_w, dtype=np.float32).reshape(3).copy() if object_target_tcp_w is not None else None
    min_substeps_per_edge = max(1, int(math.ceil(max(1, int(min_steps)) / max(1, int(plan.shape[0] - 1)))))
    joint_step_limit = float(args_cli.max_joint_step)
    if max_joint_step_override is not None and math.isfinite(float(max_joint_step_override)) and float(max_joint_step_override) > 0.0:
        joint_step_limit = min(joint_step_limit, float(max_joint_step_override))
    lock_xy_np = None
    if cartesian_lock_xy_w is not None:
        lock_xy_np = np.asarray(cartesian_lock_xy_w, dtype=np.float32).reshape(2)
    xy_error_limit = float("inf")
    if max_cartesian_xy_error_m is not None and math.isfinite(float(max_cartesian_xy_error_m)) and float(max_cartesian_xy_error_m) > 0.0:
        xy_error_limit = float(max_cartesian_xy_error_m)
    stop_distance = float("inf")
    if proximity_stop_distance_m is not None and math.isfinite(float(proximity_stop_distance_m)) and float(proximity_stop_distance_m) > 0.0:
        stop_distance = float(proximity_stop_distance_m)
    max_joint_step = 0.0
    max_cartesian_xy_error = 0.0
    closest_object_distance = float("inf")
    closest_tcp_world = None
    reached_proximity_stop = False
    executed_steps = 0
    settle_steps_executed = 0
    final_tcp_error = float("nan")
    aborted = False
    abort_reason = ""
    stop_requested = False

    def update_tcp_runtime_metrics() -> None:
        nonlocal max_cartesian_xy_error, closest_object_distance, closest_tcp_world, reached_proximity_stop, stop_requested
        tcp_pos = tcp_pose_matrix(robot, robot_entity_cfg.body_ids[0])[:3, 3].astype(np.float32)
        if lock_xy_np is not None:
            xy_error = float(np.linalg.norm(tcp_pos[:2] - lock_xy_np))
            max_cartesian_xy_error = max(max_cartesian_xy_error, xy_error)
            if bool(abort_on_cartesian_xy_error) and xy_error > xy_error_limit:
                stop_requested = True
        if object_target_np is not None:
            object_distance = float(np.linalg.norm(tcp_pos - object_target_np))
            if object_distance < closest_object_distance:
                closest_object_distance = object_distance
                closest_tcp_world = tcp_pos.copy()
            if object_distance <= stop_distance:
                reached_proximity_stop = True
                stop_requested = True

    for waypoint in plan:
        if aborted or stop_requested:
            break
        target_joint_pos = torch.tensor(waypoint, dtype=torch.float32, device=robot.device).reshape(1, -1).repeat(robot.num_instances, 1)
        max_delta = float(torch.max(torch.abs(target_joint_pos - previous_joint_target)).item())
        delta_substeps = max(1, int(math.ceil(max_delta / max(joint_step_limit, 1e-6))))
        substeps = max(min_substeps_per_edge, delta_substeps)
        segment_start = previous_joint_target.clone()
        for sub_idx in range(substeps):
            alpha = min_jerk((sub_idx + 1) / max(1, substeps))
            joint_pos_des = (1.0 - alpha) * segment_start + alpha * target_joint_pos
            max_joint_step = max(max_joint_step, float(torch.max(torch.abs(joint_pos_des - previous_joint_target)).item()))
            previous_joint_target = joint_pos_des.clone()
            robot.set_joint_position_target(locked_joint_target)
            robot.set_joint_position_target(joint_pos_des, joint_ids=joint_ids)
            if gripper is not None and gripper_width is not None:
                gripper.set_width(gripper_width)
            stabilize_robot_base(robot, locked_root_pose_w)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim_dt)
            record_video_frame(scene)
            gui_playback_tick(sim_dt)
            executed_steps += 1
            update_tcp_runtime_metrics()
            if bool(abort_on_cartesian_xy_error) and stop_requested and not reached_proximity_stop:
                aborted = True
                abort_reason = f"Cartesian descent TCP XY drift exceeded threshold ({max_cartesian_xy_error:.4f}m > {xy_error_limit:.4f}m)."
                break
            if reached_proximity_stop:
                break

    if not aborted and not reached_proximity_stop and plan.shape[0] > 0 and int(args_cli.curobo_final_settle_steps) > 0:
        final_joint_target = torch.tensor(plan[-1], dtype=torch.float32, device=robot.device).reshape(1, -1).repeat(robot.num_instances, 1)
        for _ in range(int(args_cli.curobo_final_settle_steps)):
            robot.set_joint_position_target(locked_joint_target)
            robot.set_joint_position_target(final_joint_target, joint_ids=joint_ids)
            if gripper is not None and gripper_width is not None:
                gripper.set_width(gripper_width)
            stabilize_robot_base(robot, locked_root_pose_w)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim_dt)
            record_video_frame(scene)
            gui_playback_tick(sim_dt)
            executed_steps += 1
            settle_steps_executed += 1
            update_tcp_runtime_metrics()
            if bool(abort_on_cartesian_xy_error) and stop_requested and not reached_proximity_stop:
                aborted = True
                abort_reason = f"Cartesian descent TCP XY drift exceeded threshold during settle ({max_cartesian_xy_error:.4f}m > {xy_error_limit:.4f}m)."
                break
            if reached_proximity_stop:
                break

    final_tcp_pose = tcp_pose_matrix(robot, robot_entity_cfg.body_ids[0])
    if object_target_np is not None:
        final_object_distance = float(np.linalg.norm(final_tcp_pose[:3, 3].astype(np.float32) - object_target_np))
        if final_object_distance < closest_object_distance:
            closest_object_distance = final_object_distance
            closest_tcp_world = final_tcp_pose[:3, 3].astype(np.float32).copy()
        reached_proximity_stop = bool(reached_proximity_stop or final_object_distance <= stop_distance)
    else:
        final_object_distance = float("nan")
    final_orientation_delta: dict[str, Any] | None = None
    if target_tcp_np is not None:
        final_tcp_error = float(np.linalg.norm(final_tcp_pose[:3, 3] - target_tcp_np))
        try:
            final_orientation_delta = pose_delta_diagnostics(np.asarray(target_tcp_pose_w, dtype=np.float32).reshape(4, 4), final_tcp_pose)
        except Exception:
            final_orientation_delta = None
    final_joint_pos = robot.data.joint_pos[:, joint_ids].detach().cpu().numpy()
    final_target_joint_pos = np.asarray(plan[-1], dtype=np.float32).reshape(1, -1)
    final_joint_error = float(np.max(np.abs(final_joint_pos - final_target_joint_pos)))
    return {
        "max_joint_step_rad": max_joint_step,
        "executed_steps": int(executed_steps),
        "final_settle_steps_executed": int(settle_steps_executed),
        "curobo_plan_steps": int(plan.shape[0]),
        "final_joint_error_rad": final_joint_error,
        "target_joint_pos": final_target_joint_pos[0].astype(float).tolist(),
        "final_joint_pos": final_joint_pos[0].astype(float).tolist(),
        "final_pos_error_m": final_tcp_error,
        "final_tcp_pos_error_m": final_tcp_error,
        "final_tcp_orientation_delta": final_orientation_delta,
        "position_error_reference": "tcp",
        "target_tcp_world_m": target_tcp_np.astype(float).tolist() if target_tcp_np is not None else None,
        "final_tcp_world_m": final_tcp_pose[:3, 3].astype(float).tolist() if target_tcp_np is not None else None,
        "joint_step_limit_rad": float(joint_step_limit),
        "cartesian_lock_xy_world_m": lock_xy_np.astype(float).tolist() if lock_xy_np is not None else None,
        "max_cartesian_xy_error_m": float(max_cartesian_xy_error),
        "cartesian_xy_error_threshold_m": None if not math.isfinite(xy_error_limit) else float(xy_error_limit),
        "object_grasp_target_tcp_world_m": object_target_np.astype(float).tolist() if object_target_np is not None else None,
        "final_tcp_to_object_target_m": final_object_distance,
        "closest_tcp_to_object_target_m": float(closest_object_distance),
        "closest_tcp_world_m": closest_tcp_world.astype(float).tolist() if closest_tcp_world is not None else None,
        "proximity_stop_distance_m": None if not math.isfinite(stop_distance) else float(stop_distance),
        "reached_proximity_stop": bool(reached_proximity_stop),
        "aborted": bool(aborted),
        "abort_reason": abort_reason,
    }


def capture_dry_run_state(scene: InteractiveScene, robot: Articulation) -> dict[str, Any]:
    rigid_states: dict[str, torch.Tensor] = {}
    for scene_key, _ in TRASH_SCENE_OBJECTS:
        if scene_key in scene.keys():
            rigid_states[scene_key] = scene[scene_key].data.root_state_w.clone()
    return {
        "robot_root_state": robot.data.root_state_w.clone(),
        "robot_joint_pos": robot.data.joint_pos.clone(),
        "robot_joint_vel": robot.data.joint_vel.clone(),
        "rigid_states": rigid_states,
    }


def restore_dry_run_state(scene: InteractiveScene, robot: Articulation, state: dict[str, Any]) -> None:
    robot_root_state = state["robot_root_state"]
    robot_joint_pos = state["robot_joint_pos"]
    robot_joint_vel = state["robot_joint_vel"]
    robot.write_root_pose_to_sim(robot_root_state[:, :7])
    robot.write_root_velocity_to_sim(robot_root_state[:, 7:])
    robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel)
    robot.set_joint_position_target(robot_joint_pos)
    for scene_key, root_state in state.get("rigid_states", {}).items():
        if scene_key not in scene.keys():
            continue
        obj = scene[scene_key]
        obj.write_root_pose_to_sim(root_state[:, :7])
        obj.write_root_velocity_to_sim(root_state[:, 7:])


def dry_run_ik_to_wrist_pose(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ik_controller: DifferentialIKController,
    robot_entity_cfg: SceneEntityCfg,
    ee_body_id: int,
    ee_jacobi_idx: int,
    wrist_target_pose_w: np.ndarray,
    tcp_target_pose_w: np.ndarray,
    *,
    steps: int,
    locked_root_pose_w: torch.Tensor,
    control_tcp_position: bool = False,
) -> dict[str, Any]:
    sim_dt = sim.get_physics_dt()
    target_pos_b, target_quat_b = pose_to_torch_command(wrist_target_pose_w, robot)
    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
    start_pos_b, start_quat_b = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    tcp_position_control = bool(control_tcp_position and ik_controller.action_dim == 3)
    start_tcp_pos_w = torch.tensor(tcp_pose_matrix(robot, ee_body_id)[:3, 3], dtype=torch.float32, device=robot.device).repeat(robot.num_instances, 1)
    target_tcp_pos_w = torch.tensor(tcp_target_pose_w[:3, 3], dtype=torch.float32, device=robot.device).repeat(robot.num_instances, 1)
    tcp_offset = torch.tensor(gripper_tcp_offset_m(), dtype=torch.float32, device=robot.device).reshape(1, 3, 1).repeat(robot.num_instances, 1, 1)
    previous_joint_target = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
    locked_joint_target = robot.data.joint_pos.clone()
    max_joint_step = 0.0
    final_wrist_error = float("inf")
    final_tcp_error = float("inf")
    final_tcp_pose_w = tcp_pose_matrix(robot, ee_body_id)
    ik_controller.reset()
    command = torch.zeros(robot.num_instances, ik_controller.action_dim, device=robot.device)
    for step in range(max(1, int(steps))):
        alpha = min_jerk(step / max(1, int(steps) - 1))
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        if tcp_position_control:
            desired_tcp_pos_w = (1.0 - alpha) * start_tcp_pos_w + alpha * target_tcp_pos_w
            wrist_rot_w = matrix_from_quat(ee_pose_w[:, 3:7])
            desired_wrist_pos_w = desired_tcp_pos_w - torch.bmm(wrist_rot_w, tcp_offset).squeeze(-1)
            desired_pos_b, _ = subtract_frame_transforms(
                robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], desired_wrist_pos_w, ee_pose_w[:, 3:7]
            )
            desired_quat_b = ee_quat_b
        else:
            desired_pos_b = (1.0 - alpha) * start_pos_b + alpha * target_pos_b
            desired_quat_b = interpolate_quat_shortest_path(start_quat_b, target_quat_b, alpha)
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        command[:, 0:3] = desired_pos_b
        if ik_controller.action_dim == 3:
            ik_controller.set_command(command, ee_quat=ee_quat_b)
        else:
            command[:, 3:7] = desired_quat_b
            ik_controller.set_command(command)
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, robot.data.joint_pos[:, robot_entity_cfg.joint_ids])
        if not torch.isfinite(joint_pos_des).all():
            break
        joint_pos_des = clamp_joint_step(joint_pos_des, previous_joint_target, args_cli.max_joint_step)
        max_joint_step = max(max_joint_step, float(torch.max(torch.abs(joint_pos_des - previous_joint_target)).item()))
        previous_joint_target = joint_pos_des.clone()
        robot.set_joint_position_target(locked_joint_target)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        stabilize_robot_base(robot, locked_root_pose_w)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim_dt)
    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
    final_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    final_wrist_error = float(torch.linalg.norm(target_pos_b - final_pos_b, dim=1).max().item())
    final_tcp_pose_w = tcp_pose_matrix(robot, ee_body_id)
    final_tcp_error = float(np.linalg.norm(final_tcp_pose_w[:3, 3] - tcp_target_pose_w[:3, 3]))
    return {
        "final_wrist_error_m": final_wrist_error,
        "final_tcp_error_m": final_tcp_error,
        "max_joint_step_rad": max_joint_step,
        "ik_command_type": str(ik_controller.cfg.command_type),
        "tcp_position_control": tcp_position_control,
        "orientation_constraint_mode": "position_only_tcp" if tcp_position_control else "pose",
        "target_wrist_pos_b": target_pos_b[0].detach().cpu().tolist(),
        "final_wrist_pos_b": final_pos_b[0].detach().cpu().tolist(),
        "target_tcp_world_m": tcp_target_pose_w[:3, 3].astype(float).tolist(),
        "final_tcp_world_m": final_tcp_pose_w[:3, 3].astype(float).tolist(),
    }


def add_distinct_probe_point(samples: list[dict[str, Any]], label: str, point_w: np.ndarray) -> None:
    point = np.asarray(point_w, dtype=np.float32).reshape(3)
    if not np.isfinite(point).all():
        return
    for sample in samples:
        if np.linalg.norm(point - sample["point_world"]) < 0.015:
            return
    samples.append({"label": label, "point_world": point})


def target_reachability_probe_points(inst: YoloInstance, robot: Articulation, ee_body_id: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if inst.center_3d is not None:
        add_distinct_probe_point(samples, "mask_median_center", inst.center_3d)
    if bool(args_cli.target_reachability_center_only):
        return samples[:1]
    points = inst.points_world
    if points is None or points.shape[0] == 0:
        return samples
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        return samples

    current_tcp_pos_w = tcp_pose_matrix(robot, ee_body_id)[:3, 3].astype(np.float32)
    closest_idx = int(np.argmin(np.linalg.norm(points - current_tcp_pos_w.reshape(1, 3), axis=1)))
    add_distinct_probe_point(samples, "closest_visible_to_current_tcp", points[closest_idx])

    root_pos_w = robot.data.root_pose_w[0, :3].detach().cpu().numpy().astype(np.float32)
    if inst.center_3d is not None:
        approach_vec = np.asarray(inst.center_3d, dtype=np.float32) - root_pos_w
        approach_vec[2] = 0.0
        norm = float(np.linalg.norm(approach_vec))
        if norm > 1.0e-6:
            approach_vec /= norm
            projections = (points - root_pos_w.reshape(1, 3)) @ approach_vec
            near_cutoff = float(np.quantile(projections, 0.08))
            near_points = points[projections <= near_cutoff]
            if near_points.shape[0] > 0:
                add_distinct_probe_point(samples, "near_side_visible_quantile", np.median(near_points, axis=0))

    z_cutoff = float(np.quantile(points[:, 2], 0.90))
    top_points = points[points[:, 2] >= z_cutoff]
    if top_points.shape[0] > 0:
        add_distinct_probe_point(samples, "top_visible_quantile", np.median(top_points, axis=0))
    return samples[:4]


def target_reachability_tcp_poses(target_w: np.ndarray, robot: Articulation, ee_body_id: int, point_label: str) -> list[dict[str, Any]]:
    center = np.asarray(target_w, dtype=np.float32).reshape(3)
    current_wrist_rot_w = matrix_from_quat(robot.data.body_pose_w[:, ee_body_id, 3:7])[0].detach().cpu().numpy().astype(np.float32)
    pose_specs = [
        ("current_wrist_orientation", current_wrist_rot_w),
        ("side_pinch_orientation", side_pinch_wrist_rotation()),
    ]
    poses: list[dict[str, Any]] = []
    for name, wrist_rot_w in pose_specs:
        tcp_pose_w = np.eye(4, dtype=np.float32)
        tcp_pose_w[:3, :3] = wrist_rot_w
        tcp_pose_w[:3, 3] = center
        poses.append({
            "mode": name,
            "target_point_mode": point_label,
            "tcp_pose_world": tcp_pose_w,
            "wrist_pose_world": tcp_pose_to_wrist_pose(tcp_pose_w, wrist_rot_w),
        })
    return poses


def evaluate_right_arm_target_reachability(
    sim: SimulationContext,
    scene: InteractiveScene,
    instances: list[YoloInstance],
    out_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not args_cli.target_reachability_ik_check:
        for inst in instances:
            inst.right_arm_reachability = target_reachability_metadata(inst.center_3d)
            records.append({"instance_index": int(inst.index), "source": inst.source, "enabled": False, "reachability": inst.right_arm_reachability})
        (out_dir / "right_arm_reachability.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
        return records
    robot: Articulation = scene["robot"]
    reachability_command_type = "position" if args_cli.target_reachability_position_only else "pose"
    ik_cfg = DifferentialIKControllerCfg(command_type=reachability_command_type, use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=sim.device)
    entity_cfg = SceneEntityCfg("robot", joint_names=isaaclab_grasp_ik_joint_exprs(), body_names=[RIGHT_EE_BODY])
    entity_cfg.resolve(scene)
    ee_body_id = entity_cfg.body_ids[0]
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
    locked_root_pose_w = robot.data.root_pose_w.clone()
    threshold = float(args_cli.pregrasp_error_threshold_m)
    current_root_pose_w = robot_root_pose_matrix(robot)
    for inst in instances:
        legacy_meta = target_reachability_metadata(inst.center_3d)
        relative_to_robot = (
            None
            if inst.center_3d is None
            else {
                "dx": float(point_world_in_frame(inst.center_3d, current_root_pose_w)[0]),
                "dy": float(point_world_in_frame(inst.center_3d, current_root_pose_w)[1]),
                "z_world": float(np.asarray(inst.center_3d, dtype=np.float32).reshape(3)[2]),
                "source": "current_robot_root_after_navigation",
            }
        )
        if inst.center_3d is None:
            inst.right_arm_reachability = {
                "method": "right_arm_dry_run_ik",
                "reachable": False,
                "reason": "missing_3d_center",
                "relative_to_robot_m": None,
                "ik_command_type": str(ik_controller.cfg.command_type),
                "orientation_constraint_mode": "position_only_tcp" if args_cli.target_reachability_position_only else "pose",
            }
            records.append({"instance_index": int(inst.index), "source": inst.source, "reachability": inst.right_arm_reachability})
            continue
        probe_points = target_reachability_probe_points(inst, robot, ee_body_id)
        state = capture_dry_run_state(scene, robot)
        probes: list[dict[str, Any]] = []
        try:
            for point_item in probe_points:
                for pose_item in target_reachability_tcp_poses(point_item["point_world"], robot, ee_body_id, point_item["label"]):
                    restore_dry_run_state(scene, robot, state)
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(sim.get_physics_dt())
                    probe = dry_run_ik_to_wrist_pose(
                        sim,
                        scene,
                        robot,
                        ik_controller,
                        entity_cfg,
                        ee_body_id,
                        ee_jacobi_idx,
                        pose_item["wrist_pose_world"],
                        pose_item["tcp_pose_world"],
                        steps=args_cli.target_reachability_probe_steps,
                        locked_root_pose_w=locked_root_pose_w,
                        control_tcp_position=args_cli.target_reachability_position_only,
                    )
                    probe["mode"] = pose_item["mode"]
                    probe["target_point_mode"] = pose_item["target_point_mode"]
                    probes.append(probe)
        finally:
            restore_dry_run_state(scene, robot, state)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
        best = min(probes, key=lambda item: float(item.get("final_tcp_error_m", float("inf")))) if probes else {}
        best_error = float(best.get("final_tcp_error_m", float("inf")))
        reachable = bool(np.isfinite(best_error) and best_error <= threshold)
        reachability = {
            "method": "right_arm_dry_run_ik",
            "reachable": reachable,
            "reason": "ik_tcp_error_within_threshold" if reachable else "ik_tcp_error_exceeds_threshold",
            "relative_to_robot_m": relative_to_robot,
            "best_tcp_error_m": best_error,
            "reach_cost": best_error,
            "threshold_m": threshold,
            "probe_steps": int(args_cli.target_reachability_probe_steps),
            "probe_point_policy": "rgbd_geometric_center_only" if args_cli.target_reachability_center_only else "center_plus_visible_surface_diagnostics",
            "ik_command_type": str(ik_controller.cfg.command_type),
            "isaaclab_grasp_ik": isaaclab_grasp_ik_metadata(),
            "orientation_constraint_mode": "position_only_tcp" if args_cli.target_reachability_position_only else "pose",
            "frame_diagnostics": point_world_frame_diagnostics(robot, inst.center_3d, ee_body_id),
            "legacy_static_robot_pos_relative_to_robot_m": legacy_meta.get("relative_to_robot_m"),
            "best_mode": best.get("mode", ""),
            "best_target_point_mode": best.get("target_point_mode", ""),
            "probe_points_world": [
                {"label": item["label"], "point_world": item["point_world"].astype(float).tolist()}
                for item in probe_points
            ],
            "probes": probes,
            "right_arm_joint_expr": RIGHT_ARM_JOINT_EXPR,
            "isaaclab_grasp_ik": isaaclab_grasp_ik_metadata(),
            "ee_body": RIGHT_EE_BODY,
            "waist_locked": True,
            "base_locked": True,
        }
        inst.right_arm_reachability = reachability
        records.append({
            "instance_index": int(inst.index),
            "source": inst.source,
            "yolo_label": inst.yolo_label,
            "scene_key": inst.scene_key,
            "scene_object_name": inst.scene_object_name,
            "center_3d_world": inst.center_3d.tolist(),
            "reachability": reachability,
        })
    (out_dir / "right_arm_reachability.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(
        "[REACH_IK] "
        + ", ".join(
            f"{item.get('source', '')}:{item.get('instance_index')} reachable={item['reachability'].get('reachable')} err={float(item['reachability'].get('best_tcp_error_m', float('inf'))):.3f}"
            for item in records
            if item.get("reachability", {}).get("method") == "right_arm_dry_run_ik"
        ),
        flush=True,
    )
    return records


def set_robot_planar_pose_for_dry_run(robot: Articulation, pose_xyyaw: tuple[float, float, float] | list[float]) -> None:
    root_state = robot.data.root_state_w.clone()
    root_state[:, 0] = float(pose_xyyaw[0])
    root_state[:, 1] = float(pose_xyyaw[1])
    root_state[:, 2] = ROBOT_STABILIZED_BASE_Z
    yaw_quat = torch.tensor(quat_wxyz_from_yaw(float(pose_xyyaw[2])), dtype=root_state.dtype, device=robot.device)
    root_state[:, 3:7] = yaw_quat.reshape(1, 4).repeat(root_state.shape[0], 1)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])


def validate_dynamic_standpoint_candidates_with_ik(
    sim: SimulationContext,
    scene: InteractiveScene,
    candidate_meta: dict[str, Any],
) -> dict[str, Any]:
    if not bool(args_cli.dynamic_ik_diagnostics):
        pool = [item for item in candidate_meta["candidates"] if item.get("geometric_reachable")]
        candidate_meta["selected"] = min(pool, key=lambda item: float(item["score"])) if pool else None
        candidate_meta["ik_validation"] = {"enabled": False, "required": False, "reason": "diagnostics_disabled_by_cli"}
        return candidate_meta

    robot: Articulation = scene["robot"]
    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=sim.device)
    entity_cfg = SceneEntityCfg("robot", joint_names=isaaclab_grasp_ik_joint_exprs(), body_names=[RIGHT_EE_BODY])
    entity_cfg.resolve(scene)
    ee_body_id = entity_cfg.body_ids[0]
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
    target_w = np.asarray(candidate_meta["target_world_xyz"], dtype=np.float32).reshape(3)
    original_state = capture_dry_run_state(scene, robot)
    threshold = float(args_cli.pregrasp_error_threshold_m)
    try:
        for candidate in candidate_meta["candidates"]:
            if not bool(candidate.get("geometric_reachable", False)):
                candidate["ik_reachable"] = False
                candidate["ik"] = {"enabled": False, "reason": "skipped_geometric_reject"}
                continue
            restore_dry_run_state(scene, robot, original_state)
            set_robot_planar_pose_for_dry_run(robot, candidate["pose"])
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            candidate_state = capture_dry_run_state(scene, robot)
            locked_root_pose_w = robot.data.root_pose_w.clone()
            probes: list[dict[str, Any]] = []
            try:
                for pose_item in target_reachability_tcp_poses(target_w, robot, ee_body_id, "dynamic_target_center"):
                    restore_dry_run_state(scene, robot, candidate_state)
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(sim.get_physics_dt())
                    probe = dry_run_ik_to_wrist_pose(
                        sim,
                        scene,
                        robot,
                        ik_controller,
                        entity_cfg,
                        ee_body_id,
                        ee_jacobi_idx,
                        pose_item["wrist_pose_world"],
                        pose_item["tcp_pose_world"],
                        steps=args_cli.target_reachability_probe_steps,
                        locked_root_pose_w=locked_root_pose_w,
                        control_tcp_position=True,
                    )
                    probe["mode"] = pose_item["mode"]
                    probe["target_point_mode"] = pose_item["target_point_mode"]
                    probes.append(probe)
            finally:
                restore_dry_run_state(scene, robot, candidate_state)
            best = min(probes, key=lambda item: float(item.get("final_tcp_error_m", float("inf")))) if probes else {}
            best_error = float(best.get("final_tcp_error_m", float("inf")))
            reachable = bool(np.isfinite(best_error) and best_error <= threshold)
            candidate["ik_reachable"] = reachable
            candidate["score"] = float(candidate["geometric_score"]) + (5.0 * best_error if np.isfinite(best_error) else 999.0)
            candidate["ik"] = {
                "enabled": True,
                "method": "right_arm_dry_run_ik_at_candidate_root",
                "reachable": reachable,
                "reason": "ik_tcp_error_within_threshold" if reachable else "ik_tcp_error_exceeds_threshold",
                "best_tcp_error_m": best_error,
                "threshold_m": threshold,
                "probe_steps": int(args_cli.target_reachability_probe_steps),
                "orientation_constraint_mode": "position_only_tcp",
                "isaaclab_grasp_ik": isaaclab_grasp_ik_metadata(),
                "best_mode": best.get("mode", ""),
                "best_target_point_mode": best.get("target_point_mode", ""),
                "probes": probes,
            }
            if not reachable:
                suffix = "ik_tcp_error_exceeds_threshold"
                candidate["reject_reason"] = f"{candidate['reject_reason']};{suffix}" if candidate["reject_reason"] else suffix
    finally:
        restore_dry_run_state(scene, robot, original_state)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())

    ik_pool = [
        item
        for item in candidate_meta["candidates"]
        if bool(item.get("geometric_reachable", False)) and bool(item.get("ik_reachable", False))
    ]
    geometric_pool = [item for item in candidate_meta["candidates"] if bool(item.get("geometric_reachable", False))]
    if ik_pool:
        selected = min(ik_pool, key=lambda item: float(item["score"]))
        selection_policy = "geometric_and_ik_reachable"
    elif bool(args_cli.dynamic_require_ik_reachable):
        selected = None
        selection_policy = "ik_required_no_reachable_candidate"
    else:
        selected = min(geometric_pool, key=lambda item: float(item["geometric_score"])) if geometric_pool else None
        selection_policy = "geometric_reachable_with_ik_diagnostic"
    candidate_meta["selected"] = selected
    candidate_meta["ik_validation"] = {
        "enabled": True,
        "required": bool(args_cli.dynamic_require_ik_reachable),
        "selection_policy": selection_policy,
        "threshold_m": threshold,
        "candidate_count": len(candidate_meta["candidates"]),
        "reachable_count": len(ik_pool),
        "geometric_reachable_count": len(geometric_pool),
    }
    return candidate_meta


def trash_root_positions(scene: InteractiveScene) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for scene_key, object_name in TRASH_SCENE_OBJECTS:
        if scene_key not in scene.keys():
            continue
        positions[object_name] = tensor_to_numpy(scene[scene_key].data.root_state_w[0, :3]).astype(np.float32)
    return positions


def locate_target_rigid_object(scene: InteractiveScene, target_center_w: np.ndarray | None) -> tuple[str | None, str | None, float]:
    positions = trash_root_positions(scene)
    if not positions:
        return None, None, float("inf")
    if target_center_w is None:
        best_name, best_pos = min(positions.items(), key=lambda item: abs(float(item[1][2]) - TABLE_SURFACE_Z))
        scene_key = next((key for key, name in TRASH_SCENE_OBJECTS if name == best_name), None)
        return scene_key, best_name, float(abs(best_pos[2] - TABLE_SURFACE_Z))
    center = np.asarray(target_center_w, dtype=np.float32)
    best_name, best_pos = min(positions.items(), key=lambda item: float(np.linalg.norm(item[1][:2] - center[:2])))
    scene_key = next((key for key, name in TRASH_SCENE_OBJECTS if name == best_name), None)
    return scene_key, best_name, float(np.linalg.norm(best_pos[:2] - center[:2]))


def object_root_z(scene: InteractiveScene, scene_key: str | None) -> float | None:
    if scene_key is None or scene_key not in scene.keys():
        return None
    return float(scene[scene_key].data.root_state_w[0, 2].detach().cpu())


def lifted_object_delta(scene: InteractiveScene, initial_z_by_key: dict[str, float]) -> tuple[str | None, float]:
    best_key = None
    best_delta = -float("inf")
    for scene_key, _ in TRASH_SCENE_OBJECTS:
        if scene_key not in scene.keys() or scene_key not in initial_z_by_key:
            continue
        current_z = float(scene[scene_key].data.root_state_w[0, 2].detach().cpu())
        delta = current_z - initial_z_by_key[scene_key]
        if delta > best_delta:
            best_key = scene_key
            best_delta = delta
    return best_key, best_delta


def non_wheel_joint_ids(robot: Articulation, wheel_ids: list[int]) -> list[int]:
    wheel_id_set = {int(item) for item in wheel_ids}
    return [idx for idx in range(robot.num_joints) if idx not in wheel_id_set]


def hold_non_wheel_joints(robot: Articulation, locked_joint_target: torch.Tensor, wheel_ids: list[int]) -> None:
    joint_ids = non_wheel_joint_ids(robot, wheel_ids)
    if joint_ids:
        robot.set_joint_position_target(locked_joint_target[:, joint_ids], joint_ids=joint_ids)


def apply_named_joint_positions_to_target(robot: Articulation, joint_target: torch.Tensor, joint_positions: dict[str, float]) -> None:
    joint_ids, joint_names = robot.find_joints(list(joint_positions.keys()), preserve_order=True)
    if len(joint_ids) != len(joint_positions):
        missing = sorted(set(joint_positions.keys()) - set(joint_names))
        raise RuntimeError(f"Could not resolve expected joints for pose target: {missing}")
    for joint_id, joint_name in zip(joint_ids, joint_names):
        joint_target[:, int(joint_id)] = float(joint_positions[joint_name])


def arm_pose_target(robot: Articulation, base_target: torch.Tensor, joint_positions: dict[str, float]) -> torch.Tensor:
    target = base_target.clone()
    apply_named_joint_positions_to_target(robot, target, joint_positions)
    return target


def _clamp_delta(previous: float, target: float, max_delta: float) -> float:
    if max_delta <= 0.0:
        return float(target)
    delta = float(target) - float(previous)
    return float(previous) + max(-max_delta, min(max_delta, delta))


def reset_nav_command_filter() -> None:
    global _NAV_SMOOTHED_CMD
    _NAV_SMOOTHED_CMD = (0.0, 0.0)


def smooth_nav_cmd(vx: float, wz: float, dt: float | None) -> tuple[float, float]:
    global _NAV_SMOOTHED_CMD
    if not bool(args_cli.nav_cmd_smoothing) or dt is None:
        _NAV_SMOOTHED_CMD = (float(vx), float(wz))
        return _NAV_SMOOTHED_CMD
    step_dt = max(float(dt), 0.0)
    prev_vx, prev_wz = _NAV_SMOOTHED_CMD
    target_vx = _clamp_delta(prev_vx, float(vx), max(0.0, float(args_cli.nav_cmd_linear_accel_limit)) * step_dt)
    target_wz = _clamp_delta(prev_wz, float(wz), max(0.0, float(args_cli.nav_cmd_angular_accel_limit)) * step_dt)
    alpha = max(0.0, min(1.0, float(args_cli.nav_cmd_filter_alpha)))
    filtered_vx = prev_vx + alpha * (target_vx - prev_vx)
    filtered_wz = prev_wz + alpha * (target_wz - prev_wz)
    if abs(filtered_vx) < 1.0e-4 and abs(vx) < 1.0e-4:
        filtered_vx = 0.0
    if abs(filtered_wz) < 1.0e-4 and abs(wz) < 1.0e-4:
        filtered_wz = 0.0
    _NAV_SMOOTHED_CMD = (float(filtered_vx), float(filtered_wz))
    return _NAV_SMOOTHED_CMD


def limit_external_nav_cmd(vx: float, wz: float, dt: float | None = None) -> tuple[float, float]:
    linear = max(-float(args_cli.nav_max_linear_speed), min(float(args_cli.nav_max_linear_speed), float(vx)))
    angular = float(wz) * max(float(args_cli.nav_cmd_angular_scale), 0.0)
    angular = max(-float(args_cli.nav_max_angular_speed), min(float(args_cli.nav_max_angular_speed), angular))
    return smooth_nav_cmd(linear, angular, dt)


def wheel_velocity_targets(vx: float, wz: float) -> list[float]:
    if args_cli.wheel_drive_model == "differential":
        left = (float(vx) - 0.5 * args_cli.track_width * float(wz)) / args_cli.wheel_radius
        right = (float(vx) + 0.5 * args_cli.track_width * float(wz)) / args_cli.wheel_radius
        targets = [left, left, right, right]
        return [float(args_cli.wheel_velocity_scale) * value for value in targets]
    sign = float(args_cli.mecanum_wheel_sign)
    targets: list[float] = []
    if args_cli.wheel_drive_model == "urdf_diagonal":
        wheel_geometry = tuple(
            ((wheel_xy[0], wheel_xy[1]), (math.cos(yaw), math.sin(yaw)))
            for wheel_xy, yaw in URDF_DIAGONAL_WHEEL_GEOMETRY
        )
    else:
        wheel_geometry = MECANUM45_WHEEL_GEOMETRY
    for (wheel_x, wheel_y), (axis_x, axis_y) in wheel_geometry:
        tangent_x = -axis_y
        tangent_y = axis_x
        wheel_vx = float(vx) - float(wz) * wheel_y
        wheel_vy = 0.0 + float(wz) * wheel_x
        targets.append(float(args_cli.wheel_velocity_scale) * sign * (tangent_x * wheel_vx + tangent_y * wheel_vy) / args_cli.wheel_radius)
    return targets


def parse_raw_wheel_velocity(value: str) -> list[float] | None:
    if not str(value).strip():
        return None
    parts = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(parts) != 4:
        raise ValueError("--wheel_open_loop_raw_velocity must contain four comma-separated wheel speeds.")
    return parts


def apply_raw_wheel_velocity(robot: Articulation, wheel_ids: list[int], wheel_targets: list[float]) -> None:
    if len(wheel_ids) != 4:
        return
    target = torch.zeros((robot.num_instances, len(wheel_ids)), device=robot.device)
    for idx, velocity in enumerate(wheel_targets):
        target[:, idx] = float(velocity)
    robot.set_joint_velocity_target(target, joint_ids=wheel_ids)


def write_planar_root_velocity(robot: Articulation, vx: float, wz: float) -> None:
    root_state = robot.data.root_state_w.clone()
    for env_id in range(root_state.shape[0]):
        yaw = yaw_from_quat_wxyz(root_state[env_id, 3:7])
        root_state[env_id, 7] = float(vx) * math.cos(yaw)
        root_state[env_id, 8] = float(vx) * math.sin(yaw)
        root_state[env_id, 9] = 0.0
        root_state[env_id, 10] = 0.0
        root_state[env_id, 11] = 0.0
        root_state[env_id, 12] = float(wz)
    robot.write_root_velocity_to_sim(root_state[:, 7:])


def write_planar_root_motion(robot: Articulation, vx: float, wz: float, dt: float) -> None:
    root_state = robot.data.root_state_w.clone()
    step_dt = max(float(dt), 0.0)
    for env_id in range(root_state.shape[0]):
        yaw = yaw_from_quat_wxyz(root_state[env_id, 3:7])
        mid_yaw = yaw + 0.5 * float(wz) * step_dt
        next_yaw = wrap_to_pi(yaw + float(wz) * step_dt)
        root_state[env_id, 0] += float(vx) * math.cos(mid_yaw) * step_dt
        root_state[env_id, 1] += float(vx) * math.sin(mid_yaw) * step_dt
        root_state[env_id, 2] = ROBOT_STABILIZED_BASE_Z
        root_state[env_id, 3:7] = torch.tensor(quat_wxyz_from_yaw(next_yaw), device=robot.device)
        root_state[env_id, 7] = float(vx) * math.cos(next_yaw)
        root_state[env_id, 8] = float(vx) * math.sin(next_yaw)
        root_state[env_id, 9] = 0.0
        root_state[env_id, 10] = 0.0
        root_state[env_id, 11] = 0.0
        root_state[env_id, 12] = float(wz)
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])


def write_planar_root_motion_stable(robot: Articulation, vx: float, wz: float, dt: float) -> None:
    global _KINEMATIC_STABLE_POSE_XYYAW
    root_state = robot.data.root_state_w.clone()
    env_count = int(root_state.shape[0])
    if _KINEMATIC_STABLE_POSE_XYYAW is None or len(_KINEMATIC_STABLE_POSE_XYYAW) != env_count:
        _KINEMATIC_STABLE_POSE_XYYAW = [
            (float(root_state[env_id, 0]), float(root_state[env_id, 1]), yaw_from_quat_wxyz(root_state[env_id, 3:7]))
            for env_id in range(env_count)
        ]
    step_dt = max(float(dt), 0.0)
    next_state: list[tuple[float, float, float]] = []
    for env_id in range(env_count):
        observed_x = float(root_state[env_id, 0])
        observed_y = float(root_state[env_id, 1])
        observed_yaw = yaw_from_quat_wxyz(root_state[env_id, 3:7])
        state_x, state_y, state_yaw = _KINEMATIC_STABLE_POSE_XYYAW[env_id]
        if math.hypot(observed_x - state_x, observed_y - state_y) > 0.75 or abs(wrap_to_pi(observed_yaw - state_yaw)) > 0.75:
            state_x, state_y, state_yaw = observed_x, observed_y, observed_yaw
        mid_yaw = state_yaw + 0.5 * float(wz) * step_dt
        next_yaw = wrap_to_pi(state_yaw + float(wz) * step_dt)
        next_x = state_x + float(vx) * math.cos(mid_yaw) * step_dt
        next_y = state_y + float(vx) * math.sin(mid_yaw) * step_dt
        next_state.append((next_x, next_y, next_yaw))
        root_state[env_id, 0] = next_x
        root_state[env_id, 1] = next_y
        root_state[env_id, 2] = ROBOT_STABILIZED_BASE_Z
        root_state[env_id, 3:7] = torch.tensor(quat_wxyz_from_yaw(next_yaw), device=robot.device)
        root_state[env_id, 7] = float(vx) * math.cos(next_yaw)
        root_state[env_id, 8] = float(vx) * math.sin(next_yaw)
        root_state[env_id, 9] = 0.0
        root_state[env_id, 10] = 0.0
        root_state[env_id, 11] = 0.0
        root_state[env_id, 12] = float(wz)
    _KINEMATIC_STABLE_POSE_XYYAW = next_state
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])


def apply_wheel_velocity(robot: Articulation, wheel_ids: list[int], vx: float, wz: float, dt: float | None = None) -> None:
    if len(wheel_ids) != 4:
        return
    apply_raw_wheel_velocity(robot, wheel_ids, wheel_velocity_targets(vx, wz))
    if args_cli.nav_actuation_mode == "root_velocity":
        write_planar_root_velocity(robot, vx, wz)
    elif args_cli.nav_actuation_mode == "wheel" and args_cli.wheel_ground_coupling == "kinematic":
        if dt is None:
            write_planar_root_velocity(robot, vx, wz)
        else:
            write_planar_root_motion(robot, vx, wz, dt)
    elif args_cli.nav_actuation_mode == "wheel" and args_cli.wheel_ground_coupling == "kinematic_stable":
        if dt is None:
            write_planar_root_velocity(robot, vx, wz)
        else:
            write_planar_root_motion_stable(robot, vx, wz, dt)


def wheel_state_snapshot(robot: Articulation, wheel_ids: list[int], wheel_names: list[str]) -> dict[str, Any]:
    joint_pos = robot.data.joint_pos[0, wheel_ids].detach().cpu().numpy().tolist()
    joint_vel = robot.data.joint_vel[0, wheel_ids].detach().cpu().numpy().tolist()
    return {
        "names": [str(name) for name in wheel_names],
        "position_rad": [float(value) for value in joint_pos],
        "velocity_radps": [float(value) for value in joint_vel],
    }


def synthetic_nav_scan(base_x: float, base_y: float, base_yaw: float, count: int = 361) -> list[float]:
    ranges = [float("inf")] * count
    for obs_x, obs_y, radius in STATIC_NAV_SCAN_OBSTACLES:
        dx = float(obs_x) - base_x
        dy = float(obs_y) - base_y
        distance = math.hypot(dx, dy) - float(radius)
        if distance <= 0.05 or distance >= 8.0:
            continue
        bearing = wrap_to_pi(math.atan2(dy, dx) - base_yaw)
        beam = int(round(math.degrees(bearing))) + 180
        spread = max(1, int(math.degrees(math.atan2(float(radius), max(distance, 0.05)))))
        for idx in range(max(0, beam - spread), min(count, beam + spread + 1)):
            ranges[idx] = min(ranges[idx], distance)
    return ranges


def yaw_toward_xy(from_x: float, from_y: float, target_x: float, target_y: float) -> float:
    return math.atan2(float(target_y) - float(from_y), float(target_x) - float(from_x))


def yaw_facing_table(from_x: float, from_y: float) -> float:
    return yaw_toward_xy(from_x, from_y, TABLE_CENTER_XY[0], TABLE_CENTER_XY[1])


def yaw_facing_bin(from_x: float, from_y: float, bin_y: float | None = None) -> float:
    return yaw_toward_xy(from_x, from_y, BIN_FACE_X, float(0.0 if bin_y is None else bin_y))


def pose_facing_role(pose: tuple[float, float, float], role: str, waypoint_name: str = "") -> tuple[float, float, float]:
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    if role == "table_standpoint" or waypoint_name.startswith("table_"):
        yaw = yaw_facing_table(x, y)
    elif role == "bin_standpoint" or waypoint_name.startswith("bin_"):
        bin_y = bin_y_for_category(None)
        if waypoint_name == "bin_recycle":
            bin_y = bin_y_for_category("可回收物")
        elif waypoint_name == "bin_kitchen":
            bin_y = bin_y_for_category("厨余垃圾")
        elif waypoint_name == "bin_hazard":
            bin_y = bin_y_for_category("有害垃圾")
        elif waypoint_name == "bin_other":
            bin_y = bin_y_for_category("其他垃圾")
        elif waypoint_name == "bin_center":
            bin_y = 0.0
        yaw = yaw_facing_bin(x, y, bin_y)
    return (x, y, yaw)


def nav2_bin_pose_for_category(category: str | None) -> tuple[float, float, float]:
    bin_y = bin_y_for_category(category)
    return (BIN_DRIVE_X, bin_y, yaw_facing_bin(BIN_DRIVE_X, bin_y, bin_y))


def sort_bin_layout_metadata() -> dict[str, dict[str, Any]]:
    layout: dict[str, dict[str, Any]] = {}
    for category in MOTION_SORT_CATEGORIES:
        layout[category] = {
            "bin_name": BIN_NAME_BY_CATEGORY.get(category, "other"),
            "bin_y_m": float(bin_y_for_category(category)),
            "nav2_standpoint_pose": [float(v) for v in nav2_bin_pose_for_category(category)],
            "drop_pose_world": bin_drop_position_for_category(category).astype(float).tolist(),
            "opening_bounds": bin_opening_bounds_for_category(category),
        }
    return layout


def waypoint_names() -> list[str]:
    return list(WAYPOINT_REGISTRY.keys())


def waypoint_pose(name: str) -> tuple[float, float, float]:
    if name not in WAYPOINT_REGISTRY:
        raise ValueError(f"Unknown waypoint '{name}'. Available waypoints: {', '.join(waypoint_names())}")
    data = WAYPOINT_REGISTRY[name]
    pose = data["pose"]
    return pose_facing_role((float(pose[0]), float(pose[1]), float(pose[2])), str(data.get("role", "")), name)


def waypoint_registry_metadata() -> dict[str, Any]:
    return {
        name: {
            "pose": [float(value) for value in data["pose"]],
            "role": str(data["role"]),
            "description": str(data["description"]),
        }
        for name, data in WAYPOINT_REGISTRY.items()
    }


def parse_waypoint_route(value: str) -> list[str]:
    route = [item.strip() for item in str(value).split(",") if item.strip()]
    if not route:
        raise ValueError("--waypoint_route must contain at least one waypoint name.")
    unknown = [name for name in route if name not in WAYPOINT_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown waypoint(s) in --waypoint_route: {unknown}. Available: {waypoint_names()}")
    return route


def parse_dynamic_table_sides(value: str) -> list[str]:
    sides = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    allowed = {"front", "left", "right"}
    if not sides:
        raise ValueError("--dynamic_allowed_table_sides must contain at least one side.")
    unknown = [side for side in sides if side not in allowed]
    if unknown:
        raise ValueError(f"Unknown dynamic table side(s): {unknown}. Allowed: {sorted(allowed)}")
    return list(dict.fromkeys(sides))


def local_xy_to_world_delta(local_xy: tuple[float, float] | np.ndarray, yaw: float) -> np.ndarray:
    x, y = float(local_xy[0]), float(local_xy[1])
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)


def world_delta_to_local_xy(delta_xy: tuple[float, float] | np.ndarray, yaw: float) -> np.ndarray:
    x, y = float(delta_xy[0]), float(delta_xy[1])
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.array([c * x + s * y, -s * x + c * y], dtype=np.float32)


def table_side_yaw(side: str) -> float:
    if side == "front":
        return 0.0
    if side == "left":
        return math.pi / 2.0
    if side == "right":
        return -math.pi / 2.0
    raise ValueError(f"Unsupported table side: {side}")


def candidate_reject_reasons(target_w: np.ndarray, local_xy: np.ndarray, target_on_table: bool) -> list[str]:
    reasons: list[str] = []
    if not target_on_table:
        reasons.append("target_outside_table_bounds")
    if not (TABLE_SURFACE_Z + 0.015 <= float(target_w[2]) <= TABLE_SURFACE_Z + 0.35):
        reasons.append("target_z_outside_tabletop_grasp_window")
    return reasons


def right_arm_window_risk(local_xy: np.ndarray) -> list[str]:
    risks: list[str] = []
    if not (RIGHT_ARM_TARGET_X_BOUNDS[0] <= float(local_xy[0]) <= RIGHT_ARM_TARGET_X_BOUNDS[1]):
        risks.append("target_local_x_outside_preferred_right_arm_window")
    if not (RIGHT_ARM_TARGET_Y_BOUNDS[0] <= float(local_xy[1]) <= RIGHT_ARM_TARGET_Y_BOUNDS[1]):
        risks.append("target_local_y_outside_preferred_right_arm_window")
    return risks


def compute_grasp_standpoint_candidates(
    target_world_xyz: tuple[float, float, float] | np.ndarray,
    robot_pose_xyyaw: tuple[float, float, float] | list[float],
    allowed_sides: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    target_w = np.asarray(target_world_xyz, dtype=np.float32).reshape(3)
    robot_pose = np.asarray(robot_pose_xyyaw, dtype=np.float32).reshape(3)
    sides = list(allowed_sides or ("front", "left", "right"))
    clearance = max(float(args_cli.dynamic_standpoint_clearance_m), 0.0)
    preferred_local = np.array([RIGHT_ARM_PREFERRED_TARGET_X, RIGHT_ARM_PREFERRED_TARGET_Y], dtype=np.float32)
    target_on_table = bool(
        TABLE_X_LIMITS[0] <= float(target_w[0]) <= TABLE_X_LIMITS[1]
        and TABLE_Y_LIMITS[0] <= float(target_w[1]) <= TABLE_Y_LIMITS[1]
    )
    candidates: list[dict[str, Any]] = []
    for side in sides:
        side_yaw = table_side_yaw(side)
        preferred_delta = local_xy_to_world_delta(preferred_local, side_yaw)
        stand_xy = target_w[:2].astype(np.float32) - preferred_delta
        if side == "front":
            stand_xy[0] = min(float(stand_xy[0]), TABLE_X_LIMITS[0] - clearance)
            stand_xy[1] = float(np.clip(float(stand_xy[1]), -SIDE_CORRIDOR_Y, SIDE_CORRIDOR_Y))
        elif side == "left":
            stand_xy[0] = float(np.clip(float(stand_xy[0]), TABLE_X_LIMITS[0], TABLE_X_LIMITS[1]))
            stand_xy[1] = min(float(stand_xy[1]), TABLE_Y_LIMITS[0] - clearance)
        elif side == "right":
            stand_xy[0] = float(np.clip(float(stand_xy[0]), TABLE_X_LIMITS[0], TABLE_X_LIMITS[1]))
            stand_xy[1] = max(float(stand_xy[1]), TABLE_Y_LIMITS[1] + clearance)
        yaw = float(side_yaw)
        local_xy = world_delta_to_local_xy(target_w[:2] - stand_xy, yaw)
        reasons = candidate_reject_reasons(target_w, local_xy, target_on_table)
        window_risks = right_arm_window_risk(local_xy)
        travel_distance = float(np.linalg.norm(stand_xy - robot_pose[:2]))
        local_error = abs(float(local_xy[0]) - float(preferred_local[0])) + 0.45 * abs(float(local_xy[1]) - float(preferred_local[1]))
        score = local_error + 0.20 * travel_distance + 0.75 * len(window_risks)
        candidates.append({
            "id": f"table_{side}_dynamic_0",
            "side": side,
            "waypoint": f"dynamic_{side}",
            "pose": [float(stand_xy[0]), float(stand_xy[1]), float(yaw)],
            "yaw_policy": "face_table_side_normal_preserve_right_arm_local_offset",
            "side_normal_yaw": float(side_yaw),
            "target_local_xy": [float(local_xy[0]), float(local_xy[1])],
            "preferred_target_local_xy": [float(preferred_local[0]), float(preferred_local[1])],
            "geometric_reachable": len(reasons) == 0,
            "right_arm_window_risk_only": window_risks,
            "ik_reachable": None,
            "score": float(score),
            "geometric_score": float(score),
            "travel_distance_m": travel_distance,
            "reject_reason": ";".join(reasons),
            "ik": None,
        })
    geometric_candidates = [item for item in candidates if item["geometric_reachable"]]
    selected = min(geometric_candidates, key=lambda item: float(item["score"])) if geometric_candidates else None
    return {
        "target_world_xyz": [float(v) for v in target_w],
        "target_on_table": target_on_table,
        "table_bounds": {
            "x": [float(TABLE_X_LIMITS[0]), float(TABLE_X_LIMITS[1])],
            "y": [float(TABLE_Y_LIMITS[0]), float(TABLE_Y_LIMITS[1])],
        },
        "allowed_sides": sides,
        "standpoint_clearance_m": float(clearance),
        "right_arm_window": {
            "target_local_x_m": [float(RIGHT_ARM_TARGET_X_BOUNDS[0]), float(RIGHT_ARM_TARGET_X_BOUNDS[1])],
            "target_local_y_m": [float(RIGHT_ARM_TARGET_Y_BOUNDS[0]), float(RIGHT_ARM_TARGET_Y_BOUNDS[1])],
            "preferred_target_local_xy": [float(preferred_local[0]), float(preferred_local[1])],
        },
        "selected": selected,
        "candidates": candidates,
    }


def nav2_pose_with_travel_yaw(robot: Articulation, target_xy: tuple[float, float]) -> tuple[float, float, float]:
    root = robot.data.root_state_w[0].detach().cpu()
    x = float(root[0])
    y = float(root[1])
    target_x = float(target_xy[0])
    target_y = float(target_xy[1])
    return (target_x, target_y, math.atan2(target_y - y, target_x - x))


def robot_planar_pose(robot: Articulation) -> list[float]:
    root = robot.data.root_state_w[0].detach().cpu()
    return [float(root[0]), float(root[1]), yaw_from_quat_wxyz(root[3:7])]


def nav2_failure_hints(run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "nav2_stack/logs/controller_server.log"
    if not log_path.exists():
        return {"controller_log": str(log_path), "exists": False}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    interesting = [
        line
        for line in text.splitlines()
        if "detected collision ahead" in line or "Failed to make progress" in line or "Controller patience exceeded" in line
    ]
    return {
        "controller_log": str(log_path),
        "exists": True,
        "collision_ahead": "detected collision ahead" in text,
        "failed_to_make_progress": "Failed to make progress" in text,
        "controller_patience_exceeded": "Controller patience exceeded" in text,
        "tail": interesting[-12:],
    }


def ros2_shell_command(argv: list[str]) -> list[str]:
    setup = str(args_cli.ros2_setup)
    command = " ".join(shlex.quote(part) for part in argv)
    if setup:
        command = f"source {shlex.quote(setup)} >/dev/null 2>&1; exec {command}"
    return ["bash", "-lc", command]


def ros1_shell_command(argv: list[str]) -> list[str]:
    setup = str(args_cli.kuavo_ik_ros_setup)
    command = " ".join(shlex.quote(part) for part in argv)
    if setup:
        command = f"source {shlex.quote(setup)} >/dev/null 2>&1; exec {command}"
    return ["bash", "-lc", command]


def start_nav2_stack(run_dir: Path) -> tuple[subprocess.Popen[Any] | None, Path | None]:
    if not bool(args_cli.start_nav2_stack):
        return None, None
    script = Path(args_cli.nav2_stack_script)
    if not script.is_absolute():
        script = WORKSPACE_ROOT / script
    if not script.exists():
        raise RuntimeError(f"Nav2 stack launcher script is missing: {script}")
    stack_dir = run_dir / "nav2_stack"
    stack_dir.mkdir(parents=True, exist_ok=True)
    log_path = stack_dir / "nav2_stack_launcher.log"
    argv = [
        args_cli.ros2_python,
        str(script),
        "--output-dir",
        str(stack_dir),
        "--scan-topic",
        "/scan",
        "--cmd-vel-topic",
        "/cmd_vel",
        "--planner-tolerance",
        f"{float(args_cli.nav2_planner_tolerance):.6f}",
        "--xy-goal-tolerance",
        f"{float(args_cli.nav2_xy_goal_tolerance):.6f}",
        "--yaw-goal-tolerance",
        f"{float(args_cli.nav2_yaw_goal_tolerance):.6f}",
    ]
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(ros2_shell_command(argv), cwd=str(WORKSPACE_ROOT), stdout=log_file, stderr=subprocess.STDOUT, text=True)
    log_file.close()
    time.sleep(max(float(args_cli.nav2_stack_startup_s), 0.0))
    if process.poll() is not None:
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8")[-4000:]
        except Exception:
            pass
        raise RuntimeError(f"Nav2 stack exited early with code {process.returncode}. Log: {log_path}\n{tail}")
    print(f"[NAV2] Started bundled Nav2 stack. Log: {log_path}", flush=True)
    return process, log_path


def start_ros2_cmd_vel_test_publisher(run_dir: Path, duration_s: float) -> tuple[subprocess.Popen[Any] | None, Path | None]:
    if not bool(args_cli.ros_cmd_vel_demo_auto_publish):
        return None, None
    script = WORKSPACE_ROOT / "task_319_garbage_sort/scripts/ros2_cmd_vel_test_publisher.py"
    if not script.exists():
        raise RuntimeError(f"ROS2 cmd_vel test publisher script is missing: {script}")
    log_path = run_dir / "ros_cmd_vel_test_publisher.log"
    argv = [
        args_cli.ros2_python,
        str(script),
        "--topic",
        "/cmd_vel",
        "--linear-x",
        f"{float(args_cli.ros_cmd_vel_demo_linear_x):.6f}",
        "--linear-y",
        f"{float(args_cli.ros_cmd_vel_demo_linear_y):.6f}",
        "--angular-z",
        f"{float(args_cli.ros_cmd_vel_demo_angular_z):.6f}",
        "--duration-s",
        f"{max(float(duration_s), 0.1):.6f}",
        "--rate-hz",
        f"{max(float(args_cli.ros_cmd_vel_demo_rate_hz), 1.0):.6f}",
        "--startup-delay-s",
        "0.5",
        "--stop-count",
        "10",
    ]
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(ros2_shell_command(argv), cwd=str(WORKSPACE_ROOT), stdout=log_file, stderr=subprocess.STDOUT, text=True)
    log_file.close()
    print(f"[ROS_CMD] Started ROS2 /cmd_vel test publisher. Log: {log_path}", flush=True)
    return process, log_path


def stop_external_process(process: subprocess.Popen[Any] | None, name: str) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        print(f"[NAV2] {name} did not terminate promptly; killing.", flush=True)
        process.kill()


class ExternalRos2NavBridgeClient:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.process: subprocess.Popen[Any] | None = None
        self.sock: socket.socket | None = None
        self.file: Any | None = None
        self.last_cmd: tuple[float, float] = (0.0, 0.0)
        self.recv_buffer = b""

    def start(self) -> None:
        script = WORKSPACE_ROOT / "task_319_garbage_sort/scripts/ros2_nav_socket_bridge.py"
        if not script.exists():
            raise RuntimeError(f"ROS2 socket bridge script is missing: {script}")
        argv = [
            args_cli.ros2_python,
            str(script),
            "--host",
            args_cli.ros2_bridge_host,
            "--port",
            str(args_cli.ros2_bridge_port),
        ]
        self.process = subprocess.Popen(ros2_shell_command(argv), cwd=str(WORKSPACE_ROOT), text=True)
        deadline = time.time() + float(args_cli.ros2_bridge_connect_timeout_s)
        last_error = ""
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"ROS2 bridge exited early with code {self.process.returncode}.")
            try:
                self.sock = socket.create_connection((args_cli.ros2_bridge_host, int(args_cli.ros2_bridge_port)), timeout=0.5)
                self.sock.settimeout(max(0.05, float(args_cli.ros2_bridge_exchange_timeout_s)))
                self.file = None
                self.recv_buffer = b""
                print(f"[NAV2] Connected ROS2 bridge at {args_cli.ros2_bridge_host}:{args_cli.ros2_bridge_port}", flush=True)
                return
            except OSError as exc:
                last_error = repr(exc)
                time.sleep(0.1)
        raise RuntimeError(f"Could not connect to ROS2 bridge before timeout: {last_error}")

    def exchange(self, robot: Articulation) -> tuple[float, float]:
        if self.sock is None:
            return (0.0, 0.0)
        root = robot.data.root_state_w[0].detach().cpu()
        yaw = yaw_from_quat_wxyz(root[3:7])
        payload = {
            "type": "state",
            "stamp": time.time(),
            "pose": [float(root[0]), float(root[1]), float(root[2]), float(root[3]), float(root[4]), float(root[5]), float(root[6])],
            "twist": [float(v) for v in root[7:13]],
            "scan": synthetic_nav_scan(float(root[0]), float(root[1]), yaw) if args_cli.nav_publish_synthetic_obstacles else [float("inf")] * 361,
            "scan_angle_min": -math.pi,
            "scan_angle_max": math.pi,
            "scan_angle_increment": math.radians(1.0),
        }
        try:
            self.sock.sendall((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
            while b"\n" not in self.recv_buffer:
                chunk = self.sock.recv(65536)
                if not chunk:
                    self.last_cmd = (0.0, 0.0)
                    return self.last_cmd
                self.recv_buffer += chunk
            raw_line, self.recv_buffer = self.recv_buffer.split(b"\n", 1)
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                return self.last_cmd
            response = json.loads(line)
            if response.get("type") == "cmd_vel":
                self.last_cmd = (float(response.get("linear_x", 0.0)), float(response.get("angular_z", 0.0)))
        except Exception as exc:
            self.last_cmd = (0.0, 0.0)
            print(f"[NAV2] Bridge exchange failed; zeroing cmd_vel: {exc!r}", flush=True)
        return self.last_cmd

    def close(self) -> None:
        try:
            if self.sock is not None:
                self.sock.sendall((json.dumps({"type": "shutdown"}) + "\n").encode("utf-8"))
        except Exception:
            pass
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()


def quat_xyzw_from_wxyz(q: Any) -> list[float]:
    values = [float(item) for item in q]
    return [values[1], values[2], values[3], values[0]]


def kuavo_pose_payload_from_world_pose(target_pose_w: np.ndarray, robot: Articulation) -> dict[str, list[float]]:
    pose = np.asarray(target_pose_w, dtype=np.float32).reshape(4, 4)
    if int(args_cli.kuavo_ik_frame) == 2:
        pos_b, quat_b = pose_to_torch_command(pose, robot)
        return {
            "pos_xyz": [float(item) for item in pos_b[0].detach().cpu().tolist()],
            "quat_xyzw": quat_xyzw_from_wxyz(quat_b[0].detach().cpu().tolist()),
        }
    quat_wxyz = quat_wxyz_from_matrix(pose[:3, :3])
    return {
        "pos_xyz": [float(item) for item in pose[:3, 3].tolist()],
        "quat_xyzw": quat_xyzw_from_wxyz(quat_wxyz),
    }


def current_body_pose_matrix(robot: Articulation, body_id: int) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = robot.data.body_pose_w[0, body_id, 0:3].detach().cpu().numpy().astype(np.float32)
    pose[:3, :3] = matrix_from_quat(robot.data.body_pose_w[:, body_id, 3:7])[0].detach().cpu().numpy().astype(np.float32)
    return pose


def current_bilateral_arm_q(robot: Articulation) -> list[float] | None:
    left_ids, left_names = robot.find_joints([LEFT_ARM_JOINT_EXPR], preserve_order=True)
    right_ids, right_names = robot.find_joints([RIGHT_ARM_JOINT_EXPR], preserve_order=True)
    if len(left_ids) != 7 or len(right_ids) != 7:
        print(f"[KUAVO_IK] Could not resolve 14 arm q0 joints: left={left_names}, right={right_names}", flush=True)
        return None
    q = robot.data.joint_pos[0].detach().cpu()
    return [float(q[int(idx)]) for idx in list(left_ids) + list(right_ids)]


class KuavoIkSocketClient:
    def __init__(self, run_dir: Path | None = None) -> None:
        self.run_dir = run_dir
        self.process: subprocess.Popen[Any] | None = None
        self.sock: socket.socket | None = None
        self.file: Any | None = None
        self.last_error = ""

    def start(self) -> bool:
        if bool(args_cli.kuavo_ik_auto_start):
            script = Path(args_cli.kuavo_ik_bridge_script)
            if not script.is_absolute():
                script = WORKSPACE_ROOT / script
            if not script.exists():
                self.last_error = f"Kuavo IK bridge script is missing: {script}"
                return False
            log_file = None
            if self.run_dir is not None:
                log_dir = self.run_dir / "kuavo_ik_bridge"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = (log_dir / "kuavo_ik_socket_bridge.log").open("w", encoding="utf-8")
            argv = [
                args_cli.kuavo_ik_python,
                str(script),
                "--host",
                args_cli.kuavo_ik_bridge_host,
                "--port",
                str(args_cli.kuavo_ik_bridge_port),
                "--service",
                args_cli.kuavo_ik_service,
            ]
            self.process = subprocess.Popen(
                ros1_shell_command(argv),
                cwd=str(WORKSPACE_ROOT),
                stdout=log_file or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                text=True,
            )
        deadline = time.time() + float(args_cli.kuavo_ik_connect_timeout_s)
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                self.last_error = f"Kuavo IK bridge exited early with code {self.process.returncode}."
                return False
            try:
                self.sock = socket.create_connection((args_cli.kuavo_ik_bridge_host, int(args_cli.kuavo_ik_bridge_port)), timeout=0.5)
                self.sock.settimeout(max(0.5, float(args_cli.kuavo_ik_connect_timeout_s)))
                self.file = self.sock.makefile("rw", encoding="utf-8", newline="\n")
                health = self.exchange({"type": "health"})
                if health.get("ok"):
                    print(f"[KUAVO_IK] Connected bridge at {args_cli.kuavo_ik_bridge_host}:{args_cli.kuavo_ik_bridge_port}", flush=True)
                    return True
                self.last_error = json.dumps(health, ensure_ascii=False)
            except OSError as exc:
                self.last_error = repr(exc)
                time.sleep(0.1)
            except Exception as exc:
                self.last_error = repr(exc)
                return False
        return False

    def exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.file is None:
            raise RuntimeError("Kuavo IK bridge is not connected.")
        self.file.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")
        self.file.flush()
        line = self.file.readline()
        if not line:
            raise RuntimeError("Kuavo IK bridge closed the socket.")
        return json.loads(line)

    def solve(
        self,
        label: str,
        robot: Articulation,
        left_pose_w: np.ndarray,
        right_pose_w: np.ndarray,
        q_arm: list[float] | None,
    ) -> dict[str, Any]:
        payload = {
            "type": "solve",
            "label": label,
            "service": args_cli.kuavo_ik_service,
            "frame": int(args_cli.kuavo_ik_frame),
            "left_pose": kuavo_pose_payload_from_world_pose(left_pose_w, robot),
            "right_pose": kuavo_pose_payload_from_world_pose(right_pose_w, robot),
            "q_arm": q_arm,
            "params": {
                "major_optimality_tol": 1e-3,
                "major_feasibility_tol": 1e-3,
                "minor_feasibility_tol": 1e-3,
                "major_iterations_limit": 100,
                "oritation_constraint_tol": float(args_cli.kuavo_ik_ori_tol_rad),
                "pos_constraint_tol": float(args_cli.kuavo_ik_pos_tol_m),
                "pos_cost_weight": float(args_cli.kuavo_ik_pos_cost_weight),
                "constraint_mode": int(args_cli.kuavo_ik_constraint_mode),
                "elbow_cost_scale": 0.1,
            },
        }
        return self.exchange(payload)

    def close(self) -> None:
        try:
            if self.file is not None:
                self.file.write(json.dumps({"type": "shutdown"}) + "\n")
                self.file.flush()
        except Exception:
            pass
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()


class KuavoAnalyticIkClient:
    def __init__(self, run_dir: Path | None = None) -> None:
        self.run_dir = run_dir
        self.last_error = ""

    def start(self) -> bool:
        cli = Path(args_cli.kuavo_analytic_ik_cli)
        source = Path(args_cli.kuavo_analytic_ik_source)
        header = WORKSPACE_ROOT / "kuavo-ros-opensource/src/manipulation_nodes/motion_capture_ik/include/motion_capture_ik/AnalyticArmIk.hpp"
        if not cli.is_absolute():
            cli = WORKSPACE_ROOT / cli
        if not source.is_absolute():
            source = WORKSPACE_ROOT / source
        try:
            needs_build = not cli.exists()
            if cli.exists() and source.exists() and cli.stat().st_mtime < source.stat().st_mtime:
                needs_build = True
            if cli.exists() and header.exists() and cli.stat().st_mtime < header.stat().st_mtime:
                needs_build = True
            if needs_build:
                if not bool(args_cli.kuavo_analytic_ik_auto_build):
                    self.last_error = f"Kuavo analytic IK CLI is missing or stale: {cli}"
                    return False
                cli.parent.mkdir(parents=True, exist_ok=True)
                include_dir = WORKSPACE_ROOT / "kuavo-ros-opensource/src/manipulation_nodes/motion_capture_ik/include"
                cmd = [
                    "g++",
                    "-std=c++17",
                    "-O2",
                    "-I/usr/include/eigen3",
                    f"-I{include_dir}",
                    str(source),
                    "-o",
                    str(cli),
                ]
                log_path = None
                if self.run_dir is not None:
                    log_dir = self.run_dir / "kuavo_analytic_ik"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_path = log_dir / "build.log"
                result = subprocess.run(
                    cmd,
                    cwd=str(WORKSPACE_ROOT),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=30.0,
                    check=False,
                )
                if log_path is not None:
                    log_path.write_text(result.stdout, encoding="utf-8")
                if result.returncode != 0:
                    self.last_error = f"Failed to build Kuavo analytic IK CLI: {result.stdout[-1000:]}"
                    return False
            if not os.access(cli, os.X_OK):
                cli.chmod(cli.stat().st_mode | 0o111)
            print(f"[KUAVO_ANALYTIC_IK] Ready: {cli}", flush=True)
            return True
        except Exception as exc:
            self.last_error = repr(exc)
            return False

    def solve_right(
        self,
        label: str,
        robot: Articulation,
        right_base_body_id: int,
        right_link7_body_id: int,
        right_ee_body_id: int,
        right_ee_pose_w: np.ndarray,
    ) -> dict[str, Any]:
        cli = Path(args_cli.kuavo_analytic_ik_cli)
        if not cli.is_absolute():
            cli = WORKSPACE_ROOT / cli
        base_pose_w = current_body_pose_matrix(robot, right_base_body_id)
        link7_pose_w = current_body_pose_matrix(robot, right_link7_body_id)
        ee_pose_w = current_body_pose_matrix(robot, right_ee_body_id)
        link7_to_ee = np.linalg.inv(link7_pose_w) @ ee_pose_w
        target_link7_pose_w = np.asarray(right_ee_pose_w, dtype=np.float64).reshape(4, 4) @ np.linalg.inv(link7_to_ee)
        p_rel = target_link7_pose_w[:3, 3] - base_pose_w[:3, 3]
        payload = {
            "arm": "right",
            "label": label,
            "rotation": target_link7_pose_w[:3, :3].reshape(-1).astype(float).tolist(),
            "position": p_rel.astype(float).tolist(),
            "joint_limit_margin_rad": float(args_cli.kuavo_analytic_ik_joint_limit_margin_rad),
            "max_fk_position_error_m": float(args_cli.kuavo_analytic_ik_max_fk_error_m),
        }
        try:
            result = subprocess.run(
                [str(cli)],
                input=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
                cwd=str(WORKSPACE_ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=float(args_cli.kuavo_analytic_ik_timeout_s),
                check=False,
            )
            line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "{}"
            response = json.loads(line)
            response.update(
                {
                    "label": label,
                    "returncode": int(result.returncode),
                    "stderr": result.stderr[-1200:],
                    "right_base_body": RIGHT_ARM_BASE_BODY,
                    "right_link7_body": RIGHT_ARM_LINK7_BODY,
                    "right_ee_body": RIGHT_EE_BODY,
                    "target_link7_world": target_link7_pose_w.astype(float).tolist(),
                    "target_link7_position_relative_to_base_m": p_rel.astype(float).tolist(),
                    "current_link7_to_ee": link7_to_ee.astype(float).tolist(),
                }
            )
            return response
        except Exception as exc:
            return {
                "success": False,
                "solver": "kuavo_official_analytic_arm_ik",
                "label": label,
                "error_reason": repr(exc),
            }

    def close(self) -> None:
        return None


def nav2_goal_process(target_pose: tuple[float, float, float], label: str) -> subprocess.Popen[str]:
    script = WORKSPACE_ROOT / "task_319_garbage_sort/scripts/ros2_nav_goal_client.py"
    if not script.exists():
        raise RuntimeError(f"Nav2 goal client script is missing: {script}")
    x, y, yaw = target_pose
    argv = [
        args_cli.ros2_python,
        str(script),
        "--x",
        f"{float(x):.6f}",
        "--y",
        f"{float(y):.6f}",
        "--yaw",
        f"{float(yaw):.6f}",
        "--frame-id",
        args_cli.nav2_goal_frame,
        "--action-name",
        args_cli.nav2_action_name,
        "--timeout-s",
        f"{float(args_cli.nav2_goal_timeout_s):.3f}",
        "--label",
        label,
    ]
    return subprocess.Popen(ros2_shell_command(argv), cwd=str(WORKSPACE_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def step_nav2_until_goal_done(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    bridge: ExternalRos2NavBridgeClient,
    process: subprocess.Popen[str],
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
    target_pose: tuple[float, float, float],
    step_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    sim_dt = sim.get_physics_dt()
    nav_timeout_s = float(args_cli.nav2_goal_timeout_s)
    deadline = None if nav_timeout_s <= 0.0 else time.time() + nav_timeout_s + 10.0
    step_count = 0
    reset_nav_command_filter()
    while process.poll() is None and simulation_app.is_running():
        if deadline is not None and time.time() > deadline:
            process.kill()
            output, _ = process.communicate(timeout=2.0)
            apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0)
            reset_nav_command_filter()
            return {"success": False, "status": "TIMEOUT", "target_pose": target_pose, "output": output, "steps": step_count}
        vx, wz = bridge.exchange(robot)
        vx, wz = limit_external_nav_cmd(vx, wz, sim_dt)
        hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
        apply_wheel_velocity(robot, wheel_ids, vx, wz, sim_dt)
        stabilize_robot_base_for_nav(robot)
        if step_hook is not None:
            step_hook()
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        if step_count % 120 == 0:
            root = robot.data.root_state_w[0].detach().cpu()
            tracker.event("nav2_progress", target_pose=target_pose, current_pose=[float(root[0]), float(root[1]), yaw_from_quat_wxyz(root[3:7])], cmd_vel=[vx, wz])
        tracker.tick()
        step_count += 1
    output, _ = process.communicate(timeout=2.0)
    apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0)
    reset_nav_command_filter()
    success = process.returncode == 0
    status = "SUCCEEDED" if success else "FAILED"
    parsed: dict[str, Any] | None = None
    for line in reversed([item.strip() for item in output.splitlines() if item.strip()]):
        try:
            parsed = json.loads(line)
            break
        except Exception:
            continue
    if parsed is not None:
        success = bool(parsed.get("success", success))
        status = str(parsed.get("status", status))
    root = robot.data.root_state_w[0].detach().cpu()
    final_pose = [float(root[0]), float(root[1]), yaw_from_quat_wxyz(root[3:7])]
    position_error_m = float(math.hypot(float(target_pose[0]) - final_pose[0], float(target_pose[1]) - final_pose[1]))
    yaw_error_rad = float(wrap_to_pi(float(target_pose[2]) - final_pose[2]))
    return {
        "success": success,
        "status": status,
        "target_pose": target_pose,
        "final_pose": final_pose,
        "position_error_m": position_error_m,
        "yaw_error_rad": yaw_error_rad,
        "steps": step_count,
        "returncode": process.returncode,
        "nav2_result": parsed,
        "output": output[-2000:],
    }


def final_dock_to_pose(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
    target_pose: tuple[float, float, float],
    step_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Low-speed wheel correction after Nav2's coarse goal checker succeeds."""
    sim_dt = sim.get_physics_dt()
    pos_tol = max(0.001, float(args_cli.nav_final_dock_position_tolerance))
    yaw_tol = max(0.001, float(args_cli.nav_final_dock_yaw_tolerance))
    max_steps = max(1, int(args_cli.nav_final_dock_max_steps))
    max_v = max(0.01, float(args_cli.nav_final_dock_max_linear_speed))
    max_w = max(0.01, float(args_cli.nav_final_dock_max_angular_speed))
    success = False
    final_pose = robot_planar_pose(robot)
    final_dist = float("inf")
    final_yaw_error = float("inf")
    steps = 0
    reset_nav_command_filter()
    for step in range(max_steps):
        final_pose = robot_planar_pose(robot)
        dx = float(target_pose[0]) - float(final_pose[0])
        dy = float(target_pose[1]) - float(final_pose[1])
        yaw = float(final_pose[2])
        final_dist = float(math.hypot(dx, dy))
        vx = 0.0
        wz = 0.0
        final_yaw_error = wrap_to_pi(float(target_pose[2]) - yaw)
        if final_dist > pos_tol:
            desired_heading = math.atan2(dy, dx)
            heading_error = wrap_to_pi(desired_heading - yaw)
            if abs(heading_error) > max(0.025, yaw_tol * 0.5):
                wz = max(-max_w, min(max_w, 1.8 * heading_error))
            else:
                vx = max(-max_v, min(max_v, 0.9 * final_dist))
                if 0.0 < abs(vx) < 0.02:
                    vx = 0.02
                if abs(heading_error) > 0.01:
                    wz = max(-max_w, min(max_w, 1.2 * heading_error))
        elif abs(final_yaw_error) > yaw_tol:
            wz = max(-max_w, min(max_w, 1.8 * final_yaw_error))
        else:
            success = True
            break
        vx, wz = smooth_nav_cmd(vx, wz, sim_dt)
        hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
        apply_wheel_velocity(robot, wheel_ids, vx, wz, sim_dt)
        stabilize_robot_base_for_nav(robot)
        if step_hook is not None:
            step_hook()
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        if step % 60 == 0:
            tracker.event(
                "nav_final_dock_progress",
                target_pose=target_pose,
                current_pose=final_pose,
                distance_m=final_dist,
                yaw_error_rad=final_yaw_error,
                cmd_vel=[vx, wz],
            )
        tracker.tick()
        steps = step + 1
    apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0, sim_dt)
    reset_nav_command_filter()
    final_pose = robot_planar_pose(robot)
    final_dist = float(math.hypot(float(target_pose[0]) - float(final_pose[0]), float(target_pose[1]) - float(final_pose[1])))
    final_yaw_error = float(wrap_to_pi(float(target_pose[2]) - float(final_pose[2])))
    success = bool(success or (final_dist <= pos_tol and abs(final_yaw_error) <= yaw_tol))
    result = {
        "enabled": True,
        "success": success,
        "target_pose": [float(v) for v in target_pose],
        "final_pose": final_pose,
        "position_error_m": final_dist,
        "yaw_error_rad": final_yaw_error,
        "steps": steps,
        "position_tolerance_m": pos_tol,
        "yaw_tolerance_rad": yaw_tol,
    }
    tracker.event("nav_final_dock_result", **result)
    return result


def drive_to_pose(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    target_pose: tuple[float, float, float],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
) -> dict[str, Any]:
    sim_dt = sim.get_physics_dt()
    success = False
    final_dist = float("inf")
    final_yaw_error = float("inf")
    for step in range(args_cli.nav_max_steps_per_waypoint):
        root = robot.data.root_state_w[0].detach().cpu()
        x = float(root[0])
        y = float(root[1])
        yaw = yaw_from_quat_wxyz(root[3:7])
        dx = target_pose[0] - x
        dy = target_pose[1] - y
        final_dist = math.hypot(dx, dy)
        if final_dist > args_cli.nav_position_tolerance:
            desired_heading = math.atan2(dy, dx)
            heading_error = wrap_to_pi(desired_heading - yaw)
            if abs(heading_error) > 0.18:
                vx = 0.0
            else:
                vx = min(args_cli.nav_max_linear_speed, args_cli.nav_linear_gain * final_dist)
            wz = max(-args_cli.nav_max_angular_speed, min(args_cli.nav_max_angular_speed, args_cli.nav_angular_gain * heading_error))
        else:
            final_yaw_error = wrap_to_pi(target_pose[2] - yaw)
            vx = 0.0
            wz = max(-args_cli.nav_max_angular_speed, min(args_cli.nav_max_angular_speed, args_cli.nav_angular_gain * final_yaw_error))
            if abs(final_yaw_error) <= args_cli.nav_yaw_tolerance:
                success = True
                break
        hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
        apply_wheel_velocity(robot, wheel_ids, vx, wz, sim_dt)
        stabilize_robot_base_for_nav(robot)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        if step % 120 == 0:
            tracker.event("nav_progress", target_pose=target_pose, distance_m=final_dist, yaw_error_rad=final_yaw_error)
        tracker.tick()
    apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0)
    return {"success": success, "target_pose": target_pose, "final_distance_m": final_dist, "final_yaw_error_rad": final_yaw_error}


def bin_y_for_category(category: str | None) -> float:
    return BIN_Y_BY_CATEGORY.get(canonical_category(category), BIN_Y_BY_CATEGORY["其他垃圾"])


def bin_drop_position_for_category(category: str | None) -> np.ndarray:
    return np.array([BIN_DROP_X, bin_y_for_category(category), BIN_DROP_Z], dtype=np.float32)


def bin_opening_bounds_for_category(category: str | None) -> dict[str, list[float]]:
    bin_y = bin_y_for_category(category)
    return {
        "x_m": [float(BIN_OPENING_X_BOUNDS[0]), float(BIN_OPENING_X_BOUNDS[1])],
        "y_m": [float(bin_y - BIN_OPENING_HALF_Y), float(bin_y + BIN_OPENING_HALF_Y)],
    }


def point_inside_bin_opening_xy(point_w: np.ndarray | Sequence[float], category: str | None) -> bool:
    point = np.asarray(point_w, dtype=np.float32).reshape(-1)
    if point.size < 2 or not np.isfinite(point[:2]).all():
        return False
    bounds = bin_opening_bounds_for_category(category)
    return (
        bounds["x_m"][0] <= float(point[0]) <= bounds["x_m"][1]
        and bounds["y_m"][0] <= float(point[1]) <= bounds["y_m"][1]
    )


def navigation_waypoints_to_bin(category: str | None) -> list[tuple[float, float, float]]:
    bin_y = bin_y_for_category(category)
    return [
        (TABLE_APPROACH_POSE[0], SIDE_CORRIDOR_Y, math.pi / 2.0),
        (BIN_DRIVE_X, SIDE_CORRIDOR_Y, 0.0),
        (BIN_DRIVE_X, bin_y, math.pi),
    ]


def navigation_waypoints_home() -> list[tuple[float, float, float]]:
    return [
        (BIN_DRIVE_X, SIDE_CORRIDOR_Y, math.pi / 2.0),
        (TABLE_APPROACH_POSE[0], SIDE_CORRIDOR_Y, math.pi),
        (TABLE_APPROACH_POSE[0], TABLE_APPROACH_POSE[1], TABLE_APPROACH_POSE[2]),
    ]


def drop_pose_for_category(selected: SelectedGrasp, category: str | None) -> np.ndarray:
    pose = np.array(selected.pose_world, dtype=np.float32).copy()
    pose[:3, 3] = bin_drop_position_for_category(category)
    return pose


def wrist_pose_matrix(robot: Articulation, ee_body_id: int) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = robot.data.body_pose_w[0, ee_body_id, 0:3].detach().cpu().numpy().astype(np.float32)
    pose[:3, :3] = matrix_from_quat(robot.data.body_pose_w[:, ee_body_id, 3:7])[0].detach().cpu().numpy().astype(np.float32)
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


def resolve_right_ee_body_id(scene: InteractiveScene) -> int:
    robot: Articulation = scene["robot"]
    body_names = getattr(robot, "body_names", None)
    if body_names is not None and RIGHT_EE_BODY in body_names:
        return int(body_names.index(RIGHT_EE_BODY))
    entity_cfg = SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY])
    entity_cfg.resolve(scene)
    return int(entity_cfg.body_ids[0])


def tcp_pose_matrix(robot: Articulation, ee_body_id: int) -> np.ndarray:
    gripper_base_pose_w = body_pose_matrix_by_name(robot, "gripper_base")
    if gripper_base_pose_w is not None:
        tcp_pose_w = gripper_base_pose_w.copy()
        tcp_pose_w[:3, 3] = gripper_base_pose_w[:3, 3] + gripper_base_pose_w[:3, :3] @ gripper_local_tcp_offset_m()
        return tcp_pose_w
    wrist_pose_w = wrist_pose_matrix(robot, ee_body_id)
    tcp_pose_w = wrist_pose_w.copy()
    tcp_pose_w[:3, :3] = wrist_pose_w[:3, :3] @ RIGHT_GRIPPER_INLINE_MOUNT_ROT
    tcp_pose_w[:3, 3] = wrist_pose_w[:3, 3] + wrist_pose_w[:3, :3] @ gripper_tcp_offset_m()
    return tcp_pose_w


def point_from_pose_local(pose_w: np.ndarray, local_point: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w, dtype=np.float32).reshape(4, 4)
    local = np.asarray(local_point, dtype=np.float32).reshape(3)
    return (pose[:3, 3] + pose[:3, :3] @ local).astype(np.float32)


def point_world_in_frame(point_w: np.ndarray | None, frame_pose_w: np.ndarray | None) -> list[float] | None:
    if point_w is None or frame_pose_w is None:
        return None
    point_h = np.ones(4, dtype=np.float32)
    point_h[:3] = np.asarray(point_w, dtype=np.float32).reshape(3)
    return (np.linalg.inv(np.asarray(frame_pose_w, dtype=np.float32).reshape(4, 4)) @ point_h)[:3].astype(float).tolist()


def pose_translation_quat_dict(pose_w: np.ndarray | None) -> dict[str, Any] | None:
    if pose_w is None:
        return None
    pose = np.asarray(pose_w, dtype=np.float32).reshape(4, 4)
    return {
        "translation_world_m": pose[:3, 3].astype(float).tolist(),
        "quat_wxyz": [float(v) for v in quat_wxyz_from_matrix(pose[:3, :3])],
        "rpy_deg": [float(math.degrees(v)) for v in rpy_from_matrix(pose[:3, :3])],
    }


def point_world_frame_diagnostics(
    robot: Articulation,
    point_w: np.ndarray | None,
    ee_body_id: int | None = None,
) -> dict[str, Any]:
    root_pose = robot_root_pose_matrix(robot)
    right_arm_base_pose = body_pose_matrix_by_name(robot, RIGHT_ARM_BASE_BODY)
    tcp_pose = tcp_pose_matrix(robot, ee_body_id) if ee_body_id is not None else None
    root_state = robot.data.root_state_w[0].detach().cpu().numpy().astype(np.float32)
    out: dict[str, Any] = {
        "robot_root_world": pose_translation_quat_dict(root_pose),
        "robot_planar_pose": robot_planar_pose(robot),
        "robot_root_state_xyz_quat": root_state[:7].astype(float).tolist(),
        "right_arm_base_world": pose_translation_quat_dict(right_arm_base_pose),
        "right_tcp_world": pose_translation_quat_dict(tcp_pose),
        "point_world_m": None,
        "point_in_robot_root_m": None,
        "point_in_right_arm_base_m": None,
        "point_minus_tcp_world_m": None,
        "point_tcp_distance_m": None,
    }
    if point_w is not None:
        point = np.asarray(point_w, dtype=np.float32).reshape(3)
        out["point_world_m"] = point.astype(float).tolist()
        out["point_in_robot_root_m"] = point_world_in_frame(point, root_pose)
        out["point_in_right_arm_base_m"] = point_world_in_frame(point, right_arm_base_pose)
        if tcp_pose is not None:
            delta = point - tcp_pose[:3, 3]
            out["point_minus_tcp_world_m"] = delta.astype(float).tolist()
            out["point_tcp_distance_m"] = float(np.linalg.norm(delta))
    return out


def joint_positions_by_name(robot: Articulation, joint_names: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    robot_joint_names = getattr(robot, "joint_names", None)
    for name in joint_names:
        if robot_joint_names is None or name not in robot_joint_names:
            out[name] = None
            continue
        idx = int(robot_joint_names.index(name))
        out[name] = float(robot.data.joint_pos[0, idx].detach().cpu())
    return out


def gripper_alignment_diagnostics(robot: Articulation, ee_body_id: int, target_pose_w: np.ndarray | None = None) -> dict[str, Any]:
    target_pos = None
    if target_pose_w is not None:
        target_pos = np.asarray(target_pose_w, dtype=np.float32).reshape(4, 4)[:3, 3]
    tcp_pose = tcp_pose_matrix(robot, ee_body_id)
    tcp_pos = tcp_pose[:3, 3].astype(np.float32)
    root_pose = robot_root_pose_matrix(robot)
    right_arm_base_pose = body_pose_matrix_by_name(robot, RIGHT_ARM_BASE_BODY)
    right_link7_pose = body_pose_matrix_by_name(robot, RIGHT_ARM_LINK7_BODY)
    gripper_base_pose = body_pose_matrix_by_name(robot, "gripper_base")
    left_finger_pose = body_pose_matrix_by_name(robot, "left_finger")
    right_finger_pose = body_pose_matrix_by_name(robot, "right_finger")

    base_tcp_pos = None
    left_pad_pos = None
    right_pad_pos = None
    pad_center_pos = None
    if gripper_base_pose is not None:
        base_tcp_pos = point_from_pose_local(gripper_base_pose, gripper_local_tcp_offset_m())
    if left_finger_pose is not None:
        left_pad_pos = point_from_pose_local(left_finger_pose, RIGHT_GRIPPER_FINGER_PAD_CENTER_LOCAL_M)
    if right_finger_pose is not None:
        right_pad_pos = point_from_pose_local(right_finger_pose, RIGHT_GRIPPER_FINGER_PAD_CENTER_LOCAL_M)
    if left_pad_pos is not None and right_pad_pos is not None:
        pad_center_pos = (left_pad_pos + right_pad_pos) * 0.5

    def point_meta(point: np.ndarray | None) -> dict[str, Any] | None:
        if point is None:
            return None
        out = {"world_m": point.astype(float).tolist()}
        if target_pos is not None:
            delta = point - target_pos
            out["minus_target_m"] = delta.astype(float).tolist()
            out["distance_to_target_m"] = float(np.linalg.norm(delta))
        return out

    out: dict[str, Any] = {
        "target_world_m": target_pos.astype(float).tolist() if target_pos is not None else None,
        "tcp_calibration_from_urdf": RIGHT_GRIPPER_TCP_URDF_CALIBRATION,
        "wrist_tcp_offset_m": gripper_tcp_offset_m().astype(float).tolist(),
        "gripper_local_tcp_offset_m": gripper_local_tcp_offset_m().astype(float).tolist(),
        "finger_pad_center_local_m": [float(v) for v in RIGHT_GRIPPER_FINGER_PAD_CENTER_LOCAL_M],
        "tcp_from_wrist": point_meta(tcp_pos),
        "tcp_from_gripper_base": point_meta(base_tcp_pos),
        "left_finger_pad_center": point_meta(left_pad_pos),
        "right_finger_pad_center": point_meta(right_pad_pos),
        "finger_pad_midpoint": point_meta(pad_center_pos),
        "coordinate_frames": {
            "root_world": pose_translation_quat_dict(root_pose),
            "right_arm_base_world": pose_translation_quat_dict(right_arm_base_pose),
            "right_link7_world": pose_translation_quat_dict(right_link7_pose),
            "right_ee_world": pose_translation_quat_dict(wrist_pose_matrix(robot, ee_body_id)),
            "target_in_root_m": point_world_in_frame(target_pos, root_pose),
            "tcp_in_root_m": point_world_in_frame(tcp_pos, root_pose),
            "target_in_right_arm_base_m": point_world_in_frame(target_pos, right_arm_base_pose),
            "tcp_in_right_arm_base_m": point_world_in_frame(tcp_pos, right_arm_base_pose),
            "target_in_right_ee_m": point_world_in_frame(target_pos, wrist_pose_matrix(robot, ee_body_id)),
            "tcp_in_right_ee_m": point_world_in_frame(tcp_pos, wrist_pose_matrix(robot, ee_body_id)),
        },
        "whole_body_joint_positions_rad": joint_positions_by_name(
            robot,
            ["knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint"],
        ),
        "body_names_present": {
            RIGHT_ARM_BASE_BODY: right_arm_base_pose is not None,
            RIGHT_ARM_LINK7_BODY: right_link7_pose is not None,
            "gripper_base": gripper_base_pose is not None,
            "left_finger": left_finger_pose is not None,
            "right_finger": right_finger_pose is not None,
        },
    }
    if base_tcp_pos is not None:
        delta = base_tcp_pos - tcp_pos
        out["gripper_base_tcp_minus_wrist_tcp_m"] = delta.astype(float).tolist()
        out["gripper_base_tcp_minus_wrist_tcp_distance_m"] = float(np.linalg.norm(delta))
    if pad_center_pos is not None:
        delta = pad_center_pos - tcp_pos
        out["finger_midpoint_minus_wrist_tcp_m"] = delta.astype(float).tolist()
        out["finger_midpoint_minus_wrist_tcp_distance_m"] = float(np.linalg.norm(delta))
    if pad_center_pos is not None and base_tcp_pos is not None:
        delta = pad_center_pos - base_tcp_pos
        out["finger_midpoint_minus_gripper_base_tcp_m"] = delta.astype(float).tolist()
        out["finger_midpoint_minus_gripper_base_tcp_distance_m"] = float(np.linalg.norm(delta))
    return out


def project_world_point_to_image(point_world: np.ndarray, intrinsics: np.ndarray, t_camera_to_world: np.ndarray) -> dict[str, Any]:
    point_w = np.asarray(point_world, dtype=np.float32).reshape(3)
    try:
        t_world_to_camera = np.linalg.inv(np.asarray(t_camera_to_world, dtype=np.float32))
        point_cam_h = t_world_to_camera @ np.array([point_w[0], point_w[1], point_w[2], 1.0], dtype=np.float32)
        z = float(point_cam_h[2])
        if not np.isfinite(z) or z <= 1.0e-6:
            return {"finite": False, "reason": "Projected point is behind the camera or too close.", "point_camera": point_cam_h[:3].astype(float).tolist()}
        u = float(intrinsics[0, 0] * point_cam_h[0] / z + intrinsics[0, 2])
        v = float(intrinsics[1, 1] * point_cam_h[1] / z + intrinsics[1, 2])
        return {"finite": bool(np.isfinite(u) and np.isfinite(v)), "uv": [u, v], "depth_camera_m": z, "point_camera": point_cam_h[:3].astype(float).tolist()}
    except Exception as exc:
        return {"finite": False, "reason": repr(exc)}



def write_target_pose_debug(
    scene: InteractiveScene,
    rgb: np.ndarray,
    target: YoloInstance | None,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    out_dir: Path,
) -> dict[str, Any]:
    path = out_dir / "target_pose_debug.jpg"
    json_path = out_dir / "target_pose_debug.json"
    image = rgb.copy()
    if target is not None and target.mask.shape == image.shape[:2]:
        image = blend_mask(image, target.mask, (0, 255, 255), alpha=0.18)
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    metadata: dict[str, Any] = {
        "enabled": bool(args_cli.target_pose_debug),
        "image": str(path),
        "json": str(json_path),
        "allow_execution": True,
        "max_xy_error_m": float(args_cli.max_target_pose_xy_error_m),
    }
    if not args_cli.target_pose_debug:
        draw.text((12, 12), "target pose debug disabled", fill=(255, 220, 0))
        pil.save(path, quality=95)
        json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return metadata
    if target is None:
        metadata["error"] = "No target selected."
        draw.text((12, 12), "No target selected", fill=(255, 0, 0))
        pil.save(path, quality=95)
        json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return metadata
    metadata["target"] = {
        "yolo_index": int(target.index),
        "source": target.source,
        "yolo_label": target.yolo_label,
        "vlm_object_name": target.vlm.object_name if target.vlm else "",
        "vlm_category": target.vlm.waste_category if target.vlm else "",
        "scene_key": target.scene_key,
        "scene_object_name": target.scene_object_name,
        "bbox_xyxy": [float(v) for v in target.bbox_xyxy],
        "mask_pixels": int(np.count_nonzero(target.mask)),
    }
    if target.center_3d is None:
        metadata["allow_execution"] = False
        metadata["error"] = "Selected target has no camera-derived 3D center."
        draw.text((12, 12), "No camera-derived target center", fill=(255, 0, 0))
        pil.save(path, quality=95)
        json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return metadata

    perceived_w = np.asarray(target.center_3d, dtype=np.float32).reshape(3)
    scene_key = target.scene_key
    scene_object_name = target.scene_object_name
    nearest_scene_key, nearest_name, nearest_xy = locate_target_rigid_object(scene, perceived_w)
    match_source = "target_scene_key"
    if scene_key is None or scene_key not in scene.keys():
        scene_key = nearest_scene_key
        scene_object_name = nearest_name
        match_source = "nearest_rigid_object"
    if scene_key is None or scene_key not in scene.keys():
        metadata["allow_execution"] = False
        metadata["error"] = "No simulator rigid-object pose is available for comparison."
        metadata["camera_derived_center_world_m"] = perceived_w.astype(float).tolist()
        draw.text((12, 12), "No simulator truth pose", fill=(255, 0, 0))
        pil.save(path, quality=95)
        json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return metadata

    sim_root_w = tensor_to_numpy(scene[scene_key].data.root_state_w[0, :3]).astype(np.float32)
    delta = perceived_w - sim_root_w
    xy_error = float(np.linalg.norm(delta[:2]))
    xyz_error = float(np.linalg.norm(delta))
    pass_xy = bool(xy_error <= float(args_cli.max_target_pose_xy_error_m))
    metadata.update(
        {
            "match_source": match_source,
            "matched_scene_key": scene_key,
            "matched_scene_object_name": scene_object_name,
            "nearest_scene_key": nearest_scene_key,
            "nearest_scene_object_name": nearest_name,
            "nearest_xy_distance_m": float(nearest_xy),
            "camera_derived_center_world_m": perceived_w.astype(float).tolist(),
            "simulator_object_root_world_m": sim_root_w.astype(float).tolist(),
            "delta_camera_minus_sim_m": delta.astype(float).tolist(),
            "xy_error_m": xy_error,
            "xyz_error_m": xyz_error,
            "z_error_m": float(delta[2]),
            "pass_xy_gate": pass_xy,
            "allow_execution": pass_xy,
        }
    )
    perceived_projection = project_world_point_to_image(perceived_w, intrinsics, t_camera_to_world)
    sim_projection = project_world_point_to_image(sim_root_w, intrinsics, t_camera_to_world)
    metadata["camera_derived_projection"] = perceived_projection
    metadata["simulator_truth_projection"] = sim_projection

    def draw_cross(projection: dict[str, Any], color: tuple[int, int, int], label: str) -> tuple[int, int] | None:
        if not projection.get("finite") or "uv" not in projection:
            return None
        u, v = float(projection["uv"][0]), float(projection["uv"][1])
        width, height = pil.size
        inside = 0.0 <= u < width and 0.0 <= v < height
        draw_u = int(max(0, min(width - 1, round(u))))
        draw_v = int(max(0, min(height - 1, round(v))))
        radius = 15 if inside else 10
        draw.line((draw_u - radius, draw_v, draw_u + radius, draw_v), fill=color, width=4)
        draw.line((draw_u, draw_v - radius, draw_u, draw_v + radius), fill=color, width=4)
        draw.ellipse((draw_u - radius, draw_v - radius, draw_u + radius, draw_v + radius), outline=color, width=3)
        text_x = max(4, min(draw_u + 12, width - 230))
        text_y = max(4, min(draw_v - 18, height - 18))
        draw.text((text_x, text_y), label, fill=color)
        return draw_u, draw_v

    perceived_uv = draw_cross(perceived_projection, (0, 255, 255), "camera->world center")
    sim_uv = draw_cross(sim_projection, (255, 220, 0), "sim object root")
    if perceived_uv is not None and sim_uv is not None:
        draw.line((perceived_uv[0], perceived_uv[1], sim_uv[0], sim_uv[1]), fill=(255, 80, 80), width=3)
    x0, y0, x1, y1 = [float(v) for v in target.bbox_xyxy]
    draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 255), width=3)
    status = "PASS" if pass_xy else "BLOCK"
    status_color = (0, 220, 0) if pass_xy else (255, 0, 0)
    draw.rectangle((8, 8, 360, 82), fill=(0, 0, 0))
    draw.text((14, 14), f"target pose gate: {status}", fill=status_color)
    draw.text((14, 36), f"xy error: {xy_error:.3f} m <= {args_cli.max_target_pose_xy_error_m:.3f} m", fill=(255, 255, 255))
    draw.text((14, 58), f"match: {scene_key} / {scene_object_name}", fill=(255, 255, 255))
    pil.save(path, quality=95)
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    if pass_xy:
        print(f"[TARGET_POSE] PASS xy_error={xy_error:.3f}m target={target.vlm.object_name if target.vlm else target.yolo_label} match={scene_key}/{scene_object_name}", flush=True)
    else:
        print(f"[TARGET_POSE] BLOCK xy_error={xy_error:.3f}m target={target.vlm.object_name if target.vlm else target.yolo_label} match={scene_key}/{scene_object_name}", flush=True)
    return metadata


def write_physical_grasp_target_alignment(
    scene: InteractiveScene,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    candidate: SelectedGrasp,
    scene_key: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Overlay the commanded TCP grasp target and the selected object's current pose."""
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "physical_grasp_target_alignment.png"
    json_path = out_dir / "physical_grasp_target_alignment.json"
    image = rgb.copy()
    target = candidate.target
    target_mask = getattr(target, "mask", None)
    if target_mask is not None and target_mask.shape == image.shape[:2]:
        image = blend_mask(image, target_mask, (0, 255, 0), alpha=0.16)
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    width, height = pil.size

    commanded_tcp_w = np.asarray(candidate.pose_world[:3, 3], dtype=np.float32).reshape(3)
    visual_center_w = (
        np.asarray(target.center_3d, dtype=np.float32).reshape(3)
        if target is not None and target.center_3d is not None
        else commanded_tcp_w.copy()
    )
    sim_root_w = None
    if scene_key in scene.keys():
        sim_root_w = tensor_to_numpy(scene[scene_key].data.root_state_w[0, :3]).astype(np.float32)
    else:
        nearest_scene_key, _, _ = locate_target_rigid_object(scene, visual_center_w)
        if nearest_scene_key is not None and nearest_scene_key in scene.keys():
            sim_root_w = tensor_to_numpy(scene[nearest_scene_key].data.root_state_w[0, :3]).astype(np.float32)
            scene_key = nearest_scene_key

    points: list[dict[str, Any]] = [
        {
            "key": "commanded_tcp_grasp_target",
            "label": "RED cmd TCP target",
            "point_world": commanded_tcp_w,
            "color": (255, 0, 0),
            "radius": 19,
        },
        {
            "key": "visual_rgbd_center",
            "label": "GREEN RGB-D center",
            "point_world": visual_center_w,
            "color": (0, 230, 0),
            "radius": 15,
        },
    ]
    if sim_root_w is not None:
        points.append(
            {
                "key": "sim_object_root",
                "label": "BLUE sim object root",
                "point_world": sim_root_w,
                "color": (0, 120, 255),
                "radius": 13,
            }
        )

    meta: dict[str, Any] = {
        "image": str(image_path),
        "json": str(json_path),
        "scene_key": scene_key,
        "scene_object_name": scene_object_name_for_key(scene_key),
        "object_name": getattr(target, "object_name", None) if target is not None else None,
        "waste_category": getattr(target, "waste_category", None) if target is not None else None,
        "target_metadata": getattr(target, "metadata", {}) if target is not None else {},
        "candidate_source": candidate.source,
        "commanded_tcp_grasp_target_world_m": commanded_tcp_w.astype(float).tolist(),
        "visual_rgbd_center_world_m": visual_center_w.astype(float).tolist(),
        "sim_object_root_world_m": None if sim_root_w is None else sim_root_w.astype(float).tolist(),
        "command_minus_visual_center_m": (commanded_tcp_w - visual_center_w).astype(float).tolist(),
        "command_minus_visual_center_distance_m": float(np.linalg.norm(commanded_tcp_w - visual_center_w)),
        "command_minus_sim_root_m": None if sim_root_w is None else (commanded_tcp_w - sim_root_w).astype(float).tolist(),
        "command_minus_sim_root_distance_m": None if sim_root_w is None else float(np.linalg.norm(commanded_tcp_w - sim_root_w)),
        "visual_center_minus_sim_root_m": None if sim_root_w is None else (visual_center_w - sim_root_w).astype(float).tolist(),
        "visual_center_minus_sim_root_distance_m": None if sim_root_w is None else float(np.linalg.norm(visual_center_w - sim_root_w)),
        "camera_to_world": np.asarray(t_camera_to_world, dtype=np.float32).astype(float).tolist(),
        "intrinsics": np.asarray(intrinsics, dtype=np.float32).astype(float).tolist(),
        "points": {},
    }

    drawn: dict[str, tuple[int, int]] = {}
    for item in points:
        point_w = np.asarray(item["point_world"], dtype=np.float32).reshape(3)
        projection = project_world_point_to_image(point_w, intrinsics, t_camera_to_world)
        key = str(item["key"])
        meta["points"][key] = {
            "label": item["label"],
            "world_m": point_w.astype(float).tolist(),
            "projection": projection,
        }
        color = item["color"]
        label = item["label"]
        if not projection.get("finite") or "uv" not in projection:
            continue
        u, v = float(projection["uv"][0]), float(projection["uv"][1])
        clamped_u = int(max(0, min(width - 1, round(u))))
        clamped_v = int(max(0, min(height - 1, round(v))))
        r = int(item["radius"])
        draw.line((clamped_u - r, clamped_v, clamped_u + r, clamped_v), fill=color, width=4)
        draw.line((clamped_u, clamped_v - r, clamped_u, clamped_v + r), fill=color, width=4)
        draw.ellipse((clamped_u - r, clamped_v - r, clamped_u + r, clamped_v + r), outline=color, width=3)
        text_x = max(4, min(clamped_u + 14, width - 260))
        text_y = max(4, min(clamped_v - 18, height - 22))
        draw.text((text_x, text_y), label, fill=color)
        drawn[key] = (clamped_u, clamped_v)

    line_pairs = [
        ("commanded_tcp_grasp_target", "visual_rgbd_center", (255, 255, 0)),
        ("commanded_tcp_grasp_target", "sim_object_root", (255, 120, 0)),
        ("visual_rgbd_center", "sim_object_root", (0, 255, 255)),
    ]
    for a, b, color in line_pairs:
        if a in drawn and b in drawn:
            draw.line((drawn[a][0], drawn[a][1], drawn[b][0], drawn[b][1]), fill=color, width=3)

    if target is not None:
        x0, y0, x1, y1 = [float(v) for v in target.bbox_xyxy]
        draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0), width=3)
    draw.rectangle((8, 8, 620, 116), fill=(0, 0, 0))
    draw.text((14, 14), "Physical grasp target alignment", fill=(255, 255, 255))
    draw.text((14, 38), "RED: commanded gripper/TCP target", fill=(255, 80, 80))
    draw.text((14, 60), "GREEN: current RGB-D object center", fill=(80, 255, 80))
    draw.text((14, 82), "BLUE: simulator rigid-object root", fill=(80, 170, 255))
    pil.save(image_path, quality=95)
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(
        "[GRASP_ALIGN] saved "
        f"{image_path} cmd-vs-visual={meta['command_minus_visual_center_distance_m']:.3f}m "
        f"cmd-vs-sim={meta['command_minus_sim_root_distance_m']}",
        flush=True,
    )
    return meta


def write_scene_guided_rgbd_alignment(
    scene: InteractiveScene,
    instances: list[YoloInstance],
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    out_dir: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for inst in instances:
        if inst.center_3d is None or not inst.scene_key or inst.scene_key not in scene.keys():
            continue
        visual_w = np.asarray(inst.center_3d, dtype=np.float32).reshape(3)
        sim_w = tensor_to_numpy(scene[inst.scene_key].data.root_state_w[0, :3]).astype(np.float32)
        delta = visual_w - sim_w
        records.append(
            {
                "instance_index": int(inst.index),
                "source": inst.source,
                "scene_key": inst.scene_key,
                "scene_object_name": inst.scene_object_name,
                "object_name": inst.vlm.object_name if inst.vlm else inst.yolo_label,
                "rgbd_center_world_m": visual_w.astype(float).tolist(),
                "simulator_root_world_m": sim_w.astype(float).tolist(),
                "delta_rgbd_minus_sim_m": delta.astype(float).tolist(),
                "xy_error_m": float(np.linalg.norm(delta[:2])),
                "xyz_error_m": float(np.linalg.norm(delta)),
                "rgbd_projection": project_world_point_to_image(visual_w, intrinsics, t_camera_to_world),
                "sim_projection": project_world_point_to_image(sim_w, intrinsics, t_camera_to_world),
                "point_count": int(inst.points_world.shape[0]) if inst.points_world is not None else None,
                "mask_pixels": int(np.count_nonzero(inst.mask)),
                "rgbd_center_metadata": inst.rgbd_center_metadata,
            }
        )
    meta = {
        "enabled": bool(records),
        "policy": "compare_scene_guided_rgbd_center_to_simulator_truth_for_coordinate_audit",
        "records": records,
    }
    (out_dir / "scene_guided_rgbd_alignment.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def scene_root_position_world(scene: InteractiveScene, scene_key: str | None) -> np.ndarray | None:
    if scene_key is None or scene_key not in scene.keys():
        return None
    return tensor_to_numpy(scene[scene_key].data.root_state_w[0, :3]).astype(np.float32)


def target_mask_detection_center_px(target: YoloInstance | None) -> list[float] | None:
    if target is None:
        return None
    mask = getattr(target, "mask", None)
    if mask is not None and mask.size > 0 and np.count_nonzero(mask) > 0:
        ys, xs = np.nonzero(mask)
        return [float(np.mean(xs)), float(np.mean(ys))]
    x0, y0, x1, y1 = [float(v) for v in target.bbox_xyxy]
    return [0.5 * (x0 + x1), 0.5 * (y0 + y1)]


def draw_projection_cross(
    draw: ImageDraw.ImageDraw,
    projection: dict[str, Any],
    image_size: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
    *,
    radius: int = 14,
) -> tuple[int, int] | None:
    if not projection.get("finite") or "uv" not in projection:
        return None
    width, height = image_size
    u, v = float(projection["uv"][0]), float(projection["uv"][1])
    px = int(max(0, min(width - 1, round(u))))
    py = int(max(0, min(height - 1, round(v))))
    draw.line((px - radius, py, px + radius, py), fill=color, width=4)
    draw.line((px, py - radius, px, py + radius), fill=color, width=4)
    draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=color, width=3)
    text_x = max(4, min(px + 12, width - 260))
    text_y = max(4, min(py - 18, height - 20))
    draw.text((text_x, text_y), label, fill=color)
    return px, py


def write_rgbd_vs_truth_debug(
    scene: InteractiveScene,
    rgb: np.ndarray,
    target: YoloInstance | None,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    out_dir: Path,
    *,
    label: str = "target",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "rgbd_vs_truth_debug.png"
    json_path = out_dir / "rgbd_vs_truth_debug.json"
    image = rgb.copy()
    if target is not None and target.mask.shape == image.shape[:2]:
        image = blend_mask(image, target.mask, (0, 255, 0), alpha=0.16)
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    width, height = pil.size
    meta: dict[str, Any] = {
        "enabled": target is not None,
        "label": label,
        "image": str(image_path),
        "json": str(json_path),
        "camera_frame": "head_rgbd optical camera frame used by current capture",
        "world_frame": "Isaac world frame",
        "intrinsics": np.asarray(intrinsics, dtype=np.float32).astype(float).tolist(),
        "camera_to_world": pose_diagnostics(t_camera_to_world),
        "target": target_candidate_record(target) if target is not None else None,
        "detection_center_px": target_mask_detection_center_px(target),
        "allow_execution": True,
    }
    if target is None:
        meta["reason"] = "no_target"
        draw.text((12, 12), "rgbd_vs_truth: no target", fill=(255, 0, 0))
        pil.save(image_path, quality=95)
        json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta
    rgbd_center = np.asarray(target.center_3d, dtype=np.float32).reshape(3) if target.center_3d is not None else None
    scene_key = target.scene_key
    scene_object_name = target.scene_object_name
    nearest_key = None
    nearest_name = None
    nearest_xy = float("inf")
    if rgbd_center is not None:
        nearest_key, nearest_name, nearest_xy = locate_target_rigid_object(scene, rgbd_center)
    if scene_key is None or scene_key not in scene.keys():
        scene_key = nearest_key
        scene_object_name = nearest_name
    truth_center = scene_root_position_world(scene, scene_key)
    meta.update(
        {
            "scene_key_used_for_truth": scene_key,
            "scene_object_name_used_for_truth": scene_object_name,
            "nearest_scene_key": nearest_key,
            "nearest_scene_object_name": nearest_name,
            "nearest_xy_distance_m": float(nearest_xy),
            "rgbd_center_world_m": None if rgbd_center is None else rgbd_center.astype(float).tolist(),
            "truth_object_root_world_m": None if truth_center is None else truth_center.astype(float).tolist(),
            "rgbd_center_metadata": target.rgbd_center_metadata,
            "point_cloud": point_cloud_debug_stats(target.points_world) if target.points_world is not None else None,
        }
    )
    drawn: dict[str, tuple[int, int]] = {}
    if meta["detection_center_px"] is not None:
        u, v = meta["detection_center_px"]
        px = int(max(0, min(width - 1, round(float(u)))))
        py = int(max(0, min(height - 1, round(float(v)))))
        r = 12
        draw.rectangle((px - r, py - r, px + r, py + r), outline=(255, 0, 255), width=3)
        draw.line((px - r, py, px + r, py), fill=(255, 0, 255), width=3)
        draw.line((px, py - r, px, py + r), fill=(255, 0, 255), width=3)
        draw.text((max(4, min(px + 12, width - 260)), max(4, min(py + 10, height - 20))), "mask/bbox center", fill=(255, 0, 255))
        drawn["detection_center_px"] = (px, py)
    if rgbd_center is not None:
        projection = project_world_point_to_image(rgbd_center, intrinsics, t_camera_to_world)
        meta["rgbd_center_projection"] = projection
        uv = draw_projection_cross(draw, projection, pil.size, (0, 255, 255), "RGB-D world center")
        if uv is not None:
            drawn["rgbd_center_projection"] = uv
    if truth_center is not None:
        projection = project_world_point_to_image(truth_center, intrinsics, t_camera_to_world)
        meta["truth_object_root_projection"] = projection
        uv = draw_projection_cross(draw, projection, pil.size, (255, 220, 0), "sim truth root")
        if uv is not None:
            drawn["truth_object_root_projection"] = uv
    if rgbd_center is not None and truth_center is not None:
        delta = rgbd_center - truth_center
        meta.update(
            {
                "delta_rgbd_minus_truth_m": delta.astype(float).tolist(),
                "xy_error_m": float(np.linalg.norm(delta[:2])),
                "z_error_m": float(delta[2]),
                "xyz_error_m": float(np.linalg.norm(delta)),
            }
        )
        if "rgbd_center_projection" in drawn and "truth_object_root_projection" in drawn:
            a = drawn["rgbd_center_projection"]
            b = drawn["truth_object_root_projection"]
            draw.line((a[0], a[1], b[0], b[1]), fill=(255, 80, 80), width=3)
    if target is not None:
        x0, y0, x1, y1 = [float(v) for v in target.bbox_xyxy]
        draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0), width=3)
    draw.rectangle((8, 8, 620, 116), fill=(0, 0, 0))
    draw.text((14, 14), "RGB-D vs simulator truth", fill=(255, 255, 255))
    draw.text((14, 38), "MAGENTA: 2D detection center", fill=(255, 120, 255))
    draw.text((14, 60), "CYAN: RGB-D backprojected world center", fill=(120, 255, 255))
    draw.text((14, 82), "YELLOW: simulator object root", fill=(255, 230, 120))
    pil.save(image_path, quality=95)
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def compact_segment_debug(segment: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "name",
        "motion_backend",
        "fallback_motion_backend",
        "fallback_reason",
        "curobo_success",
        "curobo_reason",
        "curobo_planning_mode",
        "curobo_kuavo_seed_joint_plan_fallback_reason",
        "curobo_tcp_refine_used",
        "converged",
        "final_pos_error_m",
        "final_tcp_pos_error_m",
        "error_threshold_m",
        "candidate_width_m",
        "target_tcp_world_m",
        "final_tcp_world_m",
        "actual_final_width_m",
        "target_final_width_m",
        "steps",
    ]
    out = {key: segment.get(key) for key in keys if key in segment}
    seed = segment.get("curobo_kuavo_analytic_seed")
    if isinstance(seed, dict):
        out["kuavo_analytic_seed"] = {
            "enabled": bool(seed.get("enabled", False)),
            "available": bool(seed.get("available", False)),
            "selected": bool(seed.get("selected", False)),
            "reason": seed.get("reason", ""),
            "selection_key": seed.get("selection_key"),
            "roll_attempt_count": len(seed.get("roll_attempts", []) or []),
            "selected_q": seed.get("selected_q"),
        }
    metadata = segment.get("curobo_metadata")
    if isinstance(metadata, dict):
        out["curobo_metadata_summary"] = {
            key: metadata.get(key)
            for key in (
                "status",
                "solve_time_s",
                "interpolated_steps",
                "start_limit_audit",
                "goal_limit_audit",
                "planned_final_target_pos_error_m",
            )
            if key in metadata
        }
    return out


def classify_execution_failure(metrics: dict[str, Any]) -> dict[str, Any]:
    reason = str(metrics.get("reason", ""))
    attempts = metrics.get("attempts", []) if isinstance(metrics.get("attempts"), list) else []
    last_attempt = attempts[-1] if attempts else {}
    segments = last_attempt.get("segments", []) if isinstance(last_attempt, dict) else []
    all_text = " ".join(
        [
            reason,
            " ".join(str(seg.get("curobo_reason", "")) for seg in segments if isinstance(seg, dict)),
            " ".join(str(seg.get("fallback_reason", "")) for seg in segments if isinstance(seg, dict)),
        ]
    )
    bucket = "unknown"
    if "No object was verified above the table" in reason:
        bucket = "object_not_held_after_close_or_lift"
    elif "contact" in reason or "gripper" in reason or "width" in reason:
        bucket = "gripper_width_or_contact_failure"
    elif "RGB-D" in all_text or "target center" in all_text:
        bucket = "rgbd_localization_or_target_missing"
    elif "TCP" in all_text and "error exceeded" in all_text:
        bucket = "execution_tracking_or_tcp_target_error"
    elif "Kuavo analytic" in all_text or "analytic IK" in all_text or "IK_FAIL" in all_text or "IK error" in all_text:
        bucket = "ik_unreachable_or_bad_target_pose"
    elif "cuRobo" in all_text or "MotionGen" in all_text or "collision" in all_text:
        bucket = "trajectory_planning_or_collision"
    elif "contact" in all_text or "gripper" in all_text or "width" in all_text:
        bucket = "gripper_width_or_contact_failure"
    elif "No object was verified above the table" in all_text:
        bucket = "object_not_held_after_close_or_lift"
    return {
        "bucket": bucket,
        "reason": reason,
        "grasp_success": bool(metrics.get("grasp_success", False)),
        "success": bool(metrics.get("success", False)),
    }


def write_arm_trajectory_debug(metrics: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    path = out_dir / "arm_trajectory_debug.json"
    attempts = metrics.get("attempts", []) if isinstance(metrics.get("attempts"), list) else []
    compact_attempts: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        segments = attempt.get("segments", []) if isinstance(attempt.get("segments"), list) else []
        compact_attempts.append(
            {
                "attempt_index": attempt.get("attempt_index"),
                "motion_backend": attempt.get("motion_backend", attempt.get("motion_backend_requested")),
                "candidate_source": attempt.get("candidate_source"),
                "target_scene_key": attempt.get("target_scene_key"),
                "target_object_name": attempt.get("target_object_name"),
                "candidate_width_m": attempt.get("candidate_width_m"),
                "tcp_grasp_pose_world": attempt.get("tcp_grasp_pose_world"),
                "wrist_grasp_pose_world": attempt.get("wrist_grasp_pose_world"),
                "success": bool(attempt.get("success", False)),
                "grasp_success": bool(attempt.get("grasp_success", False)),
                "reason": attempt.get("reason", ""),
                "gripper_contact_check": attempt.get("gripper_contact_check"),
                "segments": [compact_segment_debug(seg) for seg in segments if isinstance(seg, dict)],
            }
        )
    backend_used = str(metrics.get("motion_backend") or metrics.get("arm_motion_backend") or "")
    if backend_used == "local_position_primitive":
        policy = (
            "Stable debug-cube grasp profile: command the current RGB-D geometric center as a TCP "
            "position-only target, use a nominal angled-top-down wrist, move through high-clearance "
            "safe/pregrasp waypoints, descend vertically, close the gripper slowly, then lift only "
            "after contact verification."
        )
    elif backend_used == "curobo_right_arm":
        policy = "cuRobo plans the right-arm trajectory; Kuavo analytic IK may provide a seed when configured."
    elif backend_used in {"kuavo_ik", "kuavo_analytic_ik"}:
        policy = "Kuavo IK backend generates the right-arm joint posture; IsaacLab may run local residual correction."
    else:
        policy = "Legacy/diagnostic grasp backend; inspect attempt segments for exact command type."
    meta = {
        "file": str(path),
        "policy": policy,
        "grasp_motion_profile": metrics.get("grasp_motion_profile"),
        "backend_requested": metrics.get("arm_motion_backend"),
        "backend_used": metrics.get("motion_backend"),
        "ik_command_type": metrics.get("ik_command_type"),
        "tcp_position_control": metrics.get("tcp_position_control"),
        "orientation_constraint_mode": metrics.get("orientation_constraint_mode"),
        "axis_alignment_required": metrics.get("axis_alignment_required"),
        "success": bool(metrics.get("success", False)),
        "grasp_success": bool(metrics.get("grasp_success", False)),
        "failure_classification": classify_execution_failure(metrics),
        "attempt_count": len(compact_attempts),
        "attempts": compact_attempts,
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def write_gripper_alignment_debug(
    scene: InteractiveScene,
    robot: Articulation,
    ee_body_id: int,
    metrics: dict[str, Any],
    out_dir: Path,
    *,
    camera_to_world: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
) -> dict[str, Any]:
    path = out_dir / "gripper_alignment_debug.json"
    target_pose = None
    if metrics.get("tcp_grasp_pose_world") is not None:
        target_pose = np.asarray(metrics["tcp_grasp_pose_world"], dtype=np.float32).reshape(4, 4)
    target_scene_key = metrics.get("target_scene_key")
    target_truth = scene_root_position_world(scene, target_scene_key if isinstance(target_scene_key, str) else None)
    tcp_pose = tcp_pose_matrix(robot, ee_body_id)
    root_pose = robot_root_pose_matrix(robot)
    right_arm_base_pose = body_pose_matrix_by_name(robot, RIGHT_ARM_BASE_BODY)
    right_link7_pose = body_pose_matrix_by_name(robot, RIGHT_ARM_LINK7_BODY)
    wrist_pose = wrist_pose_matrix(robot, ee_body_id)
    gripper_base_pose = body_pose_matrix_by_name(robot, "gripper_base")
    attempts = metrics.get("attempts", []) if isinstance(metrics.get("attempts"), list) else []
    checkpoints: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        for key, value in attempt.items():
            if key.startswith("gripper_alignment_"):
                checkpoints.append({"attempt_index": attempt.get("attempt_index"), "checkpoint": key, "diagnostics": value})
    meta = {
        "file": str(path),
        "frames": {
            "camera_frame": "head_rgbd optical frame; points are backprojected by intrinsics/depth then transformed by camera_to_world",
            "world_frame": "Isaac world frame",
            "robot_base_frame": "robot base_link/root articulation frame",
            "right_arm_base_frame": RIGHT_ARM_BASE_BODY,
            "right_wrist_or_link7_frame": RIGHT_ARM_LINK7_BODY,
            "gripper_tcp": "URDF-derived midpoint between the two effective finger pads, expressed through gripper_base/right wrist",
            "finger_pad_grasp_center": "midpoint of left_finger and right_finger pad centers",
        },
        "transforms": {
            "camera_to_world": None if camera_to_world is None else pose_diagnostics(camera_to_world),
            "world_to_robot_base": pose_diagnostics(np.linalg.inv(root_pose)),
            "robot_base_to_world": pose_diagnostics(root_pose),
            "right_arm_base_to_world": None if right_arm_base_pose is None else pose_diagnostics(right_arm_base_pose),
            "right_link7_to_world": None if right_link7_pose is None else pose_diagnostics(right_link7_pose),
            "right_wrist_to_world": pose_diagnostics(wrist_pose),
            "gripper_base_to_world": None if gripper_base_pose is None else pose_diagnostics(gripper_base_pose),
            "gripper_tcp_to_world": pose_diagnostics(tcp_pose),
        },
        "intrinsics": None if intrinsics is None else np.asarray(intrinsics, dtype=np.float32).astype(float).tolist(),
        "target_scene_key": target_scene_key,
        "target_object_name": metrics.get("target_object_name"),
        "target_tcp_pose_world": None if target_pose is None else pose_diagnostics(target_pose),
        "target_truth_object_root_world_m": None if target_truth is None else target_truth.astype(float).tolist(),
        "current_alignment": gripper_alignment_diagnostics(robot, ee_body_id, target_pose),
        "execution_checkpoints": checkpoints,
        "gripper_geometry": RIGHT_GRIPPER_TCP_URDF_CALIBRATION,
        "joint_positions": joint_positions_by_name(robot, RIGHT_ARM_JOINT_NAMES + ["left_finger_joint", "right_finger_joint"]),
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def write_grasp_execution_debug_files(
    scene: InteractiveScene,
    robot: Articulation,
    ee_body_id: int,
    metrics: dict[str, Any],
    out_dir: Path,
    calibration_debug_context: dict[str, Any] | None,
) -> None:
    camera_to_world = None
    intrinsics = None
    if calibration_debug_context is not None:
        if calibration_debug_context.get("camera_to_world") is not None:
            camera_to_world = np.asarray(calibration_debug_context["camera_to_world"], dtype=np.float32)
        if calibration_debug_context.get("intrinsics") is not None:
            intrinsics = np.asarray(calibration_debug_context["intrinsics"], dtype=np.float32)
    try:
        metrics["arm_trajectory_debug"] = write_arm_trajectory_debug(metrics, out_dir)
    except Exception as exc:
        metrics["arm_trajectory_debug_error"] = repr(exc)
    try:
        metrics["gripper_alignment_debug"] = write_gripper_alignment_debug(
            scene,
            robot,
            ee_body_id,
            metrics,
            out_dir,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
        )
    except Exception as exc:
        metrics["gripper_alignment_debug_error"] = repr(exc)


def debug_forced_scene_grasp(
    scene: InteractiveScene,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    requested_object: str,
) -> tuple[YoloInstance | None, list[SelectedGrasp], dict[str, Any]]:
    scene_key, scene_object_name = scene_key_and_name_for_alias(requested_object)
    metadata: dict[str, Any] = {
        "enabled": bool(requested_object),
        "requested_object": requested_object,
        "scene_key": scene_key,
        "scene_object_name": scene_object_name,
        "selected_candidates": [],
    }
    if not requested_object:
        metadata["reason"] = "not_requested"
        return None, [], metadata
    if scene_key is None or scene_object_name is None:
        metadata["error"] = f"Unknown debug scene grasp object: {requested_object}"
        return None, [], metadata
    if scene_key not in scene.keys():
        metadata["error"] = f"Scene object key is not present in this scene: {scene_key}"
        return None, [], metadata

    center = tensor_to_numpy(scene[scene_key].data.root_state_w[0, :3]).astype(np.float32)
    height, width = rgb.shape[:2]
    projection = project_world_point_to_image(center, intrinsics, t_camera_to_world)
    if projection.get("finite") and "uv" in projection:
        u = int(round(float(projection["uv"][0])))
        v = int(round(float(projection["uv"][1])))
    else:
        u = width // 2
        v = height // 2
    radius_px = 18
    yy, xx = np.ogrid[:height, :width]
    mask = ((xx - u) ** 2 + (yy - v) ** 2) <= radius_px**2
    x0 = float(max(0, u - radius_px))
    y0 = float(max(0, v - radius_px))
    x1 = float(min(width - 1, u + radius_px))
    y1 = float(min(height - 1, v + radius_px))
    object_name = scene_object_display_name(scene_object_name)
    category = category_for_scene_key(scene_key)
    vlm = VlmClassification(
        object_name=object_name,
        waste_category=category,
        confidence=1.0,
        reason="debug_force_scene_grasp_object uses simulator scene identity for motion-backend validation.",
        raw_text="debug_force_scene_grasp_object",
        error="",
    )
    target = YoloInstance(
        index=9000,
        yolo_label=object_name,
        yolo_confidence=1.0,
        bbox_xyxy=(x0, y0, x1, y1),
        mask=mask,
        vlm=vlm,
        center_3d=center.copy(),
        source="debug_scene_grasp",
        scene_key=scene_key,
        scene_object_name=scene_object_name,
        component_score=1.0,
        points_world=center.reshape(1, 3),
    )
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    pose[:3, 3] = center
    if scene_key == "debug_cube":
        pose[2, 3] = float(center[2])
    else:
        pose[2, 3] = max(float(center[2]), float(TABLE_SURFACE_Z + 0.045))
    selected = SelectedGrasp(
        pose_world=pose,
        width=clamp_width(float(args_cli.debug_force_scene_grasp_width)),
        score=1.0,
        center_2d=(float(u), float(v)),
        target=perceived_from_target(target, center),
        source="debug_scene_grasp",
        metadata={
            "debug_force_scene_grasp_object": requested_object,
            "target_scene_key": scene_key,
            "scene_key": scene_key,
            "scene_object_name": scene_object_name,
            "object_name": object_name,
            "waste_category": category,
            "projection": projection,
            "grasp_point_source": "simulator_scene_object_root_for_curobo_backend_validation",
            "debug_cube_dynamic_physics": bool(scene_key == "debug_cube"),
            "debug_cube_physics": {
                "static": bool(args_cli.debug_cube_static),
                "kinematic": bool(args_cli.debug_cube_static),
                "mass_kg": float(args_cli.debug_cube_mass),
                "size_m": float(args_cli.debug_cube_size),
                "movable_by_contacts": bool(scene_key == "debug_cube" and not args_cli.debug_cube_static),
            } if scene_key == "debug_cube" else None,
        },
    )
    grasp_meta = {
        "enabled": True,
        "backend": "debug_force_scene_grasp",
        "candidate_count": 1,
        "filtered_count": 1,
        "selected": selected_grasp_metadata(selected),
        "selected_candidates": [selected_grasp_metadata(selected)],
        "candidate_pool_count": 1,
        "source": "debug_scene_grasp",
        "fallback": None,
        "centroid_fallback_enabled": centroid_fallback_enabled(),
        "error": "",
        "hint": "Debug forced scene grasp bypasses visual target selection to validate the motion backend.",
        "mask_filter": {},
        "tcp_alignment_mode": "debug_scene_root_to_kuavo_tcp",
        "gripper_tcp_urdf_calibration": RIGHT_GRIPPER_TCP_URDF_CALIBRATION,
        "gripper_tcp_offset_m": gripper_tcp_offset_m().tolist(),
        "gripper_local_tcp_offset_m": gripper_local_tcp_offset_m().tolist(),
        "right_gripper_mount_rpy_rad": list(RIGHT_GRIPPER_INLINE_MOUNT_RPY),
        "target_identity": target_identity_metadata(freeze_target_identity(target)),
        "target_identity_consistency": {
            "selected": True,
            "grasp_input": True,
            "selected_grasp": True,
            "execution": False,
            "consistent": True,
            "block_reason": "",
        },
    }
    metadata.update(
        {
            "success": True,
            "projection": projection,
            "center_world_m": center.astype(float).tolist(),
            "bbox_xyxy": [x0, y0, x1, y1],
            "mask_pixels": int(mask.sum()),
            "grasp": grasp_meta,
            "selected_candidates": grasp_meta["selected_candidates"],
        }
    )
    return target, [selected], metadata


def save_debug_projection(
    rgb: np.ndarray,
    selected: SelectedGrasp | None,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    out_dir: Path,
) -> dict[str, Any]:
    path = out_dir / "debug_projection.jpg"
    image = rgb.copy()
    height, width = image.shape[:2]
    metadata: dict[str, Any] = {"path": str(path), "backend": args_cli.grasp_backend, "selected_source": selected.source if selected else ""}
    if selected is None:
        pil = Image.fromarray(image)
        ImageDraw.Draw(pil).text((12, 12), "No learned grasp selected", fill=(255, 0, 0))
        pil.save(path, quality=95)
        metadata["error"] = "No selected grasp to project."
        return metadata
    projection = project_world_point_to_image(selected.pose_world[:3, 3], intrinsics, t_camera_to_world)
    metadata["target_pose_world"] = selected.pose_world.tolist()
    metadata["grasp_point_world_m"] = [float(v) for v in selected.pose_world[:3, 3]]
    metadata["backend_center_2d"] = [float(v) for v in selected.center_2d]
    metadata["projection"] = projection
    if projection.get("finite") and "uv" in projection:
        u, v = (float(projection["uv"][0]), float(projection["uv"][1]))
        inside = bool(0.0 <= u < width and 0.0 <= v < height)
        metadata["inside_image"] = inside
        metadata["uv_delta_from_backend_center_px"] = [float(u - selected.center_2d[0]), float(v - selected.center_2d[1])]
        draw_u = int(max(0, min(width - 1, round(u))))
        draw_v = int(max(0, min(height - 1, round(v))))
        try:
            import cv2  # type: ignore

            cv2.drawMarker(image, (draw_u, draw_v), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=42, thickness=4, line_type=cv2.LINE_AA)
            cv2.circle(image, (draw_u, draw_v), 18, (255, 0, 0), 3, lineType=cv2.LINE_AA)
            label = "Top1 projected" if inside else "Top1 projected outside image"
            cv2.putText(image, label, (max(4, min(draw_u + 12, width - 220)), max(24, draw_v - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        except Exception:
            pil = Image.fromarray(image)
            draw = ImageDraw.Draw(pil)
            draw.line((draw_u - 24, draw_v, draw_u + 24, draw_v), fill=(255, 0, 0), width=5)
            draw.line((draw_u, draw_v - 24, draw_u, draw_v + 24), fill=(255, 0, 0), width=5)
            draw.ellipse((draw_u - 18, draw_v - 18, draw_u + 18, draw_v + 18), outline=(255, 0, 0), width=4)
            draw.text((max(4, min(draw_u + 12, width - 190)), max(4, draw_v - 24)), "Top1 projected", fill=(255, 0, 0))
            pil.save(path, quality=95)
    else:
        pil = Image.fromarray(image)
        ImageDraw.Draw(pil).text((12, 12), f"Projection invalid: {projection.get('reason', 'unknown')}", fill=(255, 0, 0))
        pil.save(path, quality=95)
        metadata["inside_image"] = False
    return metadata


def calibration_marker(name: str, prim_path: str, scale: float) -> VisualizationMarkers:
    marker = _CALIBRATION_DEBUG_MARKERS.get(name)
    if marker is None:
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (scale, scale, scale)
        marker = VisualizationMarkers(marker_cfg.replace(prim_path=prim_path))
        _CALIBRATION_DEBUG_MARKERS[name] = marker
    return marker


def render_calibration_debug_markers(
    scene: InteractiveScene,
    target_pose_world: np.ndarray,
    tcp_pose_world: np.ndarray,
    *,
    rgbd_center_world: np.ndarray | None = None,
    object_truth_world: np.ndarray | None = None,
) -> None:
    robot: Articulation = scene["robot"]
    target_pose = np.asarray(target_pose_world, dtype=np.float32)
    tcp_pose = np.asarray(tcp_pose_world, dtype=np.float32)
    target_marker = calibration_marker("target_grasp_pose", "/Visuals/task319_target_grasp_pose", 0.10)
    tcp_marker = calibration_marker("current_tcp_pose", "/Visuals/task319_current_tcp_pose", 0.075)
    finger_marker = calibration_marker("finger_pad_grasp_center", "/Visuals/task319_finger_pad_grasp_center", 0.055)
    rgbd_marker = calibration_marker("rgbd_target_center", "/Visuals/task319_rgbd_target_center", 0.065)
    truth_marker = calibration_marker("object_truth_center", "/Visuals/task319_object_truth_center", 0.065)
    target_marker.visualize(
        torch.tensor(target_pose[:3, 3], dtype=torch.float32, device=robot.device).reshape(1, 3),
        torch.tensor(quat_wxyz_from_matrix(target_pose[:3, :3]), dtype=torch.float32, device=robot.device).reshape(1, 4),
    )
    tcp_marker.visualize(
        torch.tensor(tcp_pose[:3, 3], dtype=torch.float32, device=robot.device).reshape(1, 3),
        torch.tensor(quat_wxyz_from_matrix(tcp_pose[:3, :3]), dtype=torch.float32, device=robot.device).reshape(1, 4),
    )
    left_finger_pose = body_pose_matrix_by_name(robot, "left_finger")
    right_finger_pose = body_pose_matrix_by_name(robot, "right_finger")
    if left_finger_pose is not None and right_finger_pose is not None:
        left_pad_pos = point_from_pose_local(left_finger_pose, RIGHT_GRIPPER_FINGER_PAD_CENTER_LOCAL_M)
        right_pad_pos = point_from_pose_local(right_finger_pose, RIGHT_GRIPPER_FINGER_PAD_CENTER_LOCAL_M)
        pad_center = ((left_pad_pos + right_pad_pos) * 0.5).astype(np.float32)
        finger_marker.visualize(
            torch.tensor(pad_center, dtype=torch.float32, device=robot.device).reshape(1, 3),
            torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=robot.device).reshape(1, 4),
        )
    if rgbd_center_world is not None:
        rgbd_center = np.asarray(rgbd_center_world, dtype=np.float32).reshape(3)
        rgbd_marker.visualize(
            torch.tensor(rgbd_center, dtype=torch.float32, device=robot.device).reshape(1, 3),
            torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=robot.device).reshape(1, 4),
        )
    if object_truth_world is not None:
        truth_center = np.asarray(object_truth_world, dtype=np.float32).reshape(3)
        truth_marker.visualize(
            torch.tensor(truth_center, dtype=torch.float32, device=robot.device).reshape(1, 3),
            torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=robot.device).reshape(1, 4),
        )


def write_calibration_debug_outputs(
    scene: InteractiveScene,
    selected: SelectedGrasp | None,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    out_dir: Path,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "enabled": selected is not None,
        "backend": args_cli.grasp_backend,
        "projection_image": str(out_dir / "debug_projection.jpg"),
        "marker_target_prim_path": "/Visuals/task319_target_grasp_pose",
        "marker_tcp_prim_path": "/Visuals/task319_current_tcp_pose",
        "marker_object_truth_prim_path": "/Visuals/task319_object_truth_center",
        "marker_rgbd_center_prim_path": "/Visuals/task319_rgbd_target_center",
        "marker_finger_pad_prim_path": "/Visuals/task319_finger_pad_grasp_center",
    }
    projection_meta = save_debug_projection(rgb, selected, intrinsics, t_camera_to_world, out_dir)
    metadata["projection"] = projection_meta
    if selected is None:
        metadata["error"] = "No selected grasp; calibration markers were not updated."
        (out_dir / "calibration_debug.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return metadata
    try:
        robot: Articulation = scene["robot"]
        ee_body_id = resolve_right_ee_body_id(scene)
        tcp_pose_w = tcp_pose_matrix(robot, ee_body_id)
        render_calibration_debug_markers(scene, selected.pose_world, tcp_pose_w)
        metadata["camera_to_world"] = pose_diagnostics(t_camera_to_world)
        metadata[f"top1_{args_cli.grasp_backend}_pose_world"] = pose_diagnostics(selected.pose_world)
        metadata["right_arm_tcp_pose_world"] = pose_diagnostics(tcp_pose_w)
        metadata["tcp_minus_target"] = pose_delta_diagnostics(selected.pose_world, tcp_pose_w)
        metadata["ee_body_id"] = int(ee_body_id)
        print_calibration_triplet(
            phase="planned_before_execution",
            camera_to_world=t_camera_to_world,
            target_pose_world=selected.pose_world,
            tcp_pose_world=tcp_pose_w,
        )
    except Exception as exc:
        metadata["error"] = repr(exc)
        print(f"[WARN] Calibration debug marker update failed: {exc!r}", flush=True)
    (out_dir / "calibration_debug.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def learned_grasp_orientation_mode() -> str:
    if args_cli.arm_motion_backend != "legacy_differential_ik":
        return "position_only_tcp"
    if args_cli.learned_grasp_position_only_ik:
        return "position_only_tcp"
    if args_cli.learned_grasp_use_side_pinch_orientation:
        return "side_pinch"
    if args_cli.learned_grasp_use_current_wrist_orientation:
        return "current_wrist"
    return "grasp_orientation"


def side_pinch_wrist_rotation() -> np.ndarray:
    # Columns are right-wrist local X/Y/Z axes expressed in world.
    # The inline gripper TCP offset is wrist local -Z, so local -Z -> world +X
    # gives a horizontal side pinch instead of a table-pressing vertical poke.
    return np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


def rotation_x_np(angle_rad: float) -> np.ndarray:
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def rotation_z_np(angle_rad: float) -> np.ndarray:
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def top_down_wrist_rotation() -> np.ndarray:
    # With the inline mount, the TCP offset is wrist local -Z.  Identity maps
    # wrist local -Z to world -Z, keeping the wrist above a top-down target.
    return np.eye(3, dtype=np.float32)


def angled_top_down_wrist_rotation() -> np.ndarray:
    axis = np.asarray(args_cli.angled_top_down_tcp_axis, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-6:
        axis = np.array([0.65, 0.0, -0.76], dtype=np.float32)
    else:
        axis = axis / norm
    return wrist_rotation_from_tcp_offset_axis(axis, preferred_jaw_axis_w=np.array([0.0, 1.0, 0.0], dtype=np.float32))


def kuavo_analytic_ik_roll_samples() -> list[float]:
    values: list[float] = []
    for item in str(args_cli.kuavo_analytic_ik_roll_samples_rad).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError:
            continue
        if not any(abs(value - existing) < 1e-6 for existing in values):
            values.append(value)
    if not any(abs(value) < 1e-6 for value in values):
        values.insert(0, 0.0)
    return values


def orthonormal_rotation_from_x_axis(x_axis_w: np.ndarray, preferred_up_w: np.ndarray | None = None) -> np.ndarray:
    x_axis = np.asarray(x_axis_w, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(x_axis))
    if norm < 1e-6:
        return np.eye(3, dtype=np.float32)
    x_axis = x_axis / norm
    up = np.asarray(preferred_up_w if preferred_up_w is not None else [0.0, 0.0, 1.0], dtype=np.float32).reshape(3)
    if abs(float(np.dot(x_axis, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    y_axis = np.cross(up, x_axis)
    y_axis_norm = float(np.linalg.norm(y_axis))
    if y_axis_norm < 1e-6:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        y_axis = y_axis / y_axis_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / max(float(np.linalg.norm(z_axis)), 1e-6)
    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)


def wrist_rotation_from_tcp_offset_axis(tcp_offset_axis_w: np.ndarray, preferred_jaw_axis_w: np.ndarray | None = None) -> np.ndarray:
    # The inline gripper TCP offset is wrist local -Z.  Build a wrist rotation
    # whose local -Z axis points along the requested TCP-offset direction.
    neg_z_axis = np.asarray(tcp_offset_axis_w, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(neg_z_axis))
    if norm < 1e-6:
        return np.eye(3, dtype=np.float32)
    neg_z_axis = neg_z_axis / norm
    z_axis = -neg_z_axis
    preferred_y = np.asarray(preferred_jaw_axis_w if preferred_jaw_axis_w is not None else [0.0, 1.0, 0.0], dtype=np.float32).reshape(3)
    if abs(float(np.dot(z_axis, preferred_y))) > 0.95:
        preferred_y = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_axis = np.cross(preferred_y, z_axis)
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-6)
    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)


def kuavo_analytic_approach_rotation_samples(nominal_rot: np.ndarray) -> list[dict[str, Any]]:
    direction_map = {
        "+x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "+y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "-y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "+z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "-z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "diag_down_left": np.array([0.0, 1.0, -1.0], dtype=np.float32),
        "diag_down_right": np.array([0.0, -1.0, -1.0], dtype=np.float32),
        "diag_down_forward": np.array([1.0, 0.0, -1.0], dtype=np.float32),
        "diag_forward_right": np.array([1.0, -1.0, 0.0], dtype=np.float32),
        "diag_forward_left": np.array([1.0, 1.0, 0.0], dtype=np.float32),
    }
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_token in str(args_cli.kuavo_analytic_ik_approach_dirs).split(","):
        token = raw_token.strip().lower()
        if not token:
            continue
        if token == "current":
            rot = np.asarray(nominal_rot, dtype=np.float32).reshape(3, 3)
            key = "current"
        else:
            if token not in direction_map:
                continue
            rot = wrist_rotation_from_tcp_offset_axis(direction_map[token])
            key = token
        tcp_axis = (-rot[:, 2]).astype(np.float32)
        rounded_key = ",".join(f"{v:.3f}" for v in tcp_axis.tolist()) + f":{key}"
        if rounded_key in seen:
            continue
        seen.add(rounded_key)
        samples.append({"label": key, "rotation": rot})
    if not samples:
        samples.append({"label": "current", "rotation": np.asarray(nominal_rot, dtype=np.float32).reshape(3, 3)})
    return samples


def kuavo_analytic_wrist_pose_samples(tcp_pose_w: np.ndarray, nominal_wrist_pose_w: np.ndarray) -> list[dict[str, Any]]:
    nominal_rot = np.asarray(nominal_wrist_pose_w, dtype=np.float32).reshape(4, 4)[:3, :3]
    samples: list[dict[str, Any]] = []
    for approach in kuavo_analytic_approach_rotation_samples(nominal_rot):
        approach_rot = np.asarray(approach["rotation"], dtype=np.float32).reshape(3, 3)
        for roll in kuavo_analytic_ik_roll_samples():
            wrist_rot = approach_rot @ rotation_z_np(roll)
            wrist_pose = tcp_pose_to_wrist_pose(tcp_pose_w, wrist_rot)
            samples.append(
                {
                    "approach_label": str(approach["label"]),
                    "approach_tcp_offset_axis_world": (-approach_rot[:, 2]).astype(float).tolist(),
                    "roll_rad": float(roll),
                    "wrist_pose": wrist_pose,
                    "wrist_rotation_world": wrist_rot.astype(float).tolist(),
                }
            )
    return samples


def grasp_execution_poses(selected: SelectedGrasp, robot: Articulation, ee_body_id: int, fallback_wrist_rot_w: np.ndarray | None = None) -> dict[str, np.ndarray]:
    tcp_grasp_pose = np.array(selected.pose_world, dtype=np.float32)
    tcp_pregrasp_pose = tcp_grasp_pose.copy()
    tcp_pregrasp_pose[:3, 3] -= tcp_grasp_pose[:3, :3] @ (GRASP_APPROACH_AXIS_LOCAL * np.float32(args_cli.pregrasp_offset))
    tcp_lift_pose = tcp_grasp_pose.copy()
    tcp_lift_pose[2, 3] = max(
        float(tcp_grasp_pose[2, 3] + args_cli.lift_height),
        float(TABLE_SURFACE_Z + args_cli.safe_pregrasp_table_clearance_m),
    )
    current_wrist_rot_w = matrix_from_quat(robot.data.body_pose_w[:, ee_body_id, 3:7])[0].detach().cpu().numpy().astype(np.float32)
    current_or_initial_wrist_rot_w = fallback_wrist_rot_w if fallback_wrist_rot_w is not None else current_wrist_rot_w
    wrist_rot_w = None
    if selected.source == "fallback":
        wrist_rot_w = current_or_initial_wrist_rot_w
    elif args_cli.learned_grasp_use_side_pinch_orientation:
        wrist_rot_w = side_pinch_wrist_rotation()
    elif args_cli.learned_grasp_use_current_wrist_orientation:
        wrist_rot_w = current_or_initial_wrist_rot_w
    pregrasp_wrist_rot_w = wrist_rot_w
    if (
        selected.source != "fallback"
        and args_cli.pregrasp_use_current_wrist_orientation
        and not args_cli.learned_grasp_use_side_pinch_orientation
    ):
        pregrasp_wrist_rot_w = current_or_initial_wrist_rot_w
    return {
        "tcp_grasp": tcp_grasp_pose,
        "tcp_pregrasp": tcp_pregrasp_pose,
        "tcp_lift": tcp_lift_pose,
        "wrist_grasp": tcp_pose_to_wrist_pose(tcp_grasp_pose, wrist_rot_w),
        "wrist_pregrasp": tcp_pose_to_wrist_pose(tcp_pregrasp_pose, pregrasp_wrist_rot_w),
        "wrist_lift": tcp_pose_to_wrist_pose(tcp_lift_pose, wrist_rot_w),
    }


def safe_pregrasp_waypoints(poses: dict[str, np.ndarray], robot: Articulation, ee_body_id: int) -> dict[str, np.ndarray]:
    current_tcp_pose = tcp_pose_matrix(robot, ee_body_id).astype(np.float32)
    current_wrist_rot_w = matrix_from_quat(robot.data.body_pose_w[:, ee_body_id, 3:7])[0].detach().cpu().numpy().astype(np.float32)
    safe_z = max(
        float(TABLE_SURFACE_Z + args_cli.safe_pregrasp_table_clearance_m),
        float(poses["tcp_grasp"][2, 3] + args_cli.safe_pregrasp_object_clearance_m),
        float(current_tcp_pose[2, 3] + 0.015),
    )

    tcp_lift_current = current_tcp_pose.copy()
    tcp_lift_current[2, 3] = safe_z

    tcp_standoff = poses["tcp_pregrasp"].copy()
    tcp_standoff[2, 3] = safe_z

    pregrasp_wrist_rot = poses["wrist_pregrasp"][:3, :3]
    lift_current_wrist_rot = (
        pregrasp_wrist_rot
        if args_cli.arm_motion_wrist_orientation in {"side_pinch", "top_down", "angled_top_down"}
        else current_wrist_rot_w
    )
    return {
        "tcp_safe_lift_current": tcp_lift_current,
        "tcp_safe_standoff": tcp_standoff,
        "wrist_safe_lift_current": tcp_pose_to_wrist_pose(tcp_lift_current, lift_current_wrist_rot),
        "wrist_safe_standoff": tcp_pose_to_wrist_pose(tcp_standoff, pregrasp_wrist_rot),
        "safe_z_m": float(safe_z),
        "lift_current_uses_pregrasp_wrist_rotation": bool(args_cli.arm_motion_wrist_orientation in {"side_pinch", "top_down", "angled_top_down"}),
    }


def position_only_grasp_poses(
    poses: dict[str, np.ndarray],
    robot: Articulation,
    ee_body_id: int,
    candidate: SelectedGrasp | None = None,
) -> dict[str, Any]:
    current_wrist_rot_w = matrix_from_quat(robot.data.body_pose_w[:, ee_body_id, 3:7])[0].detach().cpu().numpy().astype(np.float32)
    candidate_wrist_rot_w = None
    candidate_gripper_base_rot_w = None
    candidate_orientation_meta = None
    if candidate is not None and isinstance(candidate.metadata, dict):
        candidate_orientation_meta = candidate.metadata.get("rgbd_top_grasp_orientation")
        raw_candidate_rot = candidate.metadata.get("rgbd_wrist_rotation_world")
        if raw_candidate_rot is not None:
            try:
                candidate_rot = np.asarray(raw_candidate_rot, dtype=np.float32).reshape(3, 3)
                if np.isfinite(candidate_rot).all():
                    candidate_wrist_rot_w = candidate_rot
            except Exception:
                candidate_wrist_rot_w = None
        raw_gripper_rot = candidate.metadata.get("rgbd_gripper_base_rotation_world")
        if raw_gripper_rot is not None:
            try:
                gripper_rot = np.asarray(raw_gripper_rot, dtype=np.float32).reshape(3, 3)
                if np.isfinite(gripper_rot).all():
                    candidate_gripper_base_rot_w = gripper_rot
            except Exception:
                candidate_gripper_base_rot_w = None
    if candidate_wrist_rot_w is not None:
        wrist_rot_w = candidate_wrist_rot_w
        if isinstance(candidate_orientation_meta, dict) and bool(candidate_orientation_meta.get("directionless_envelope")):
            wrist_orientation_mode = "rgbd_directionless_envelope_top_grasp"
        else:
            wrist_orientation_mode = "rgbd_pca_top_grasp"
    elif args_cli.arm_motion_wrist_orientation == "top_down":
        wrist_rot_w = top_down_wrist_rotation()
        wrist_orientation_mode = args_cli.arm_motion_wrist_orientation
    elif args_cli.arm_motion_wrist_orientation == "angled_top_down":
        wrist_rot_w = angled_top_down_wrist_rotation()
        wrist_orientation_mode = args_cli.arm_motion_wrist_orientation
    elif args_cli.arm_motion_wrist_orientation == "side_pinch":
        wrist_rot_w = side_pinch_wrist_rotation()
        wrist_orientation_mode = args_cli.arm_motion_wrist_orientation
    else:
        wrist_rot_w = current_wrist_rot_w
        wrist_orientation_mode = args_cli.arm_motion_wrist_orientation
    tcp_grasp_pose = np.array(poses["tcp_grasp"], dtype=np.float32).copy()
    if candidate_gripper_base_rot_w is not None:
        tcp_grasp_pose[:3, :3] = candidate_gripper_base_rot_w
    commanded_position_source = "learned_grasp_translation"
    top_grasp_target = None
    top_grasp_hover = None
    top_grasp_meta = None
    if candidate is not None and isinstance(candidate.metadata, dict):
        top_grasp_meta = candidate.metadata.get("top_grasp")
        top_grasp_target = candidate.metadata.get("top_grasp_target_world_m")
        top_grasp_hover = candidate.metadata.get("top_grasp_hover_world_m")
    if top_grasp_target is not None:
        target = np.asarray(top_grasp_target, dtype=np.float32).reshape(3)
        if np.isfinite(target).all():
            tcp_grasp_pose[:3, 3] = target
            commanded_position_source = "rgbd_top_grasp_zmax_minus_depth"
    elif (
        bool(args_cli.arm_motion_use_target_center_position)
        and candidate is not None
        and candidate.target is not None
        and candidate.target.center_3d is not None
    ):
        center = np.asarray(candidate.target.center_3d, dtype=np.float32).reshape(3)
        if np.isfinite(center).all():
            tcp_grasp_pose[:3, 3] = center
            commanded_position_source = "target_rgbd_geometric_center"
    tcp_pregrasp_pose = tcp_grasp_pose.copy()
    pregrasp_source = "grasp_target_plus_clearance"
    if top_grasp_hover is not None:
        hover = np.asarray(top_grasp_hover, dtype=np.float32).reshape(3)
        if np.isfinite(hover).all():
            tcp_pregrasp_pose[:3, 3] = hover
            pregrasp_source = "rgbd_zmax_hover_same_xy"
    if pregrasp_source != "rgbd_zmax_hover_same_xy":
        tcp_pregrasp_pose[2, 3] = max(
            float(TABLE_SURFACE_Z + args_cli.arm_motion_min_table_clearance_m),
            float(tcp_grasp_pose[2, 3] + args_cli.arm_motion_pregrasp_clearance_m),
        )
    tcp_lift_pose = tcp_grasp_pose.copy()
    tcp_lift_pose[2, 3] = max(
        float(tcp_grasp_pose[2, 3] + args_cli.lift_height),
        float(TABLE_SURFACE_Z + args_cli.safe_pregrasp_table_clearance_m),
    )
    return {
        "tcp_grasp": tcp_grasp_pose,
        "tcp_pregrasp": tcp_pregrasp_pose,
        "tcp_lift": tcp_lift_pose,
        "wrist_grasp": tcp_pose_to_wrist_pose(tcp_grasp_pose, wrist_rot_w),
        "wrist_pregrasp": tcp_pose_to_wrist_pose(tcp_pregrasp_pose, wrist_rot_w),
        "wrist_lift": tcp_pose_to_wrist_pose(tcp_lift_pose, wrist_rot_w),
        "wrist_rotation_world": wrist_rot_w.astype(float).tolist(),
        "gripper_base_rotation_world": (
            candidate_gripper_base_rot_w.astype(float).tolist()
            if candidate_gripper_base_rot_w is not None
            else tcp_grasp_pose[:3, :3].astype(float).tolist()
        ),
        "wrist_orientation_mode": wrist_orientation_mode,
        "rgbd_top_grasp_orientation": candidate_orientation_meta if isinstance(candidate_orientation_meta, dict) else None,
        "angled_top_down_tcp_axis_world": [float(v) for v in args_cli.angled_top_down_tcp_axis],
        "pregrasp_clearance_m": float(tcp_pregrasp_pose[2, 3] - tcp_grasp_pose[2, 3]),
        "commanded_position_source": commanded_position_source,
        "pregrasp_position_source": pregrasp_source,
        "vertical_descent_only": bool(
            abs(float(tcp_pregrasp_pose[0, 3] - tcp_grasp_pose[0, 3])) < 1.0e-6
            and abs(float(tcp_pregrasp_pose[1, 3] - tcp_grasp_pose[1, 3])) < 1.0e-6
        ),
        "top_grasp": top_grasp_meta if isinstance(top_grasp_meta, dict) else None,
    }


def right_arm_joint_target_from_named_pose(robot: Articulation, entity_cfg: SceneEntityCfg, joint_positions: dict[str, float]) -> torch.Tensor:
    full_target = arm_pose_target(robot, robot.data.joint_pos.clone(), joint_positions)
    return full_target[:, entity_cfg.joint_ids]


def right_arm_curobo_joint_ids(robot: Articulation) -> tuple[list[int], list[str]]:
    joint_names = list(getattr(robot, "joint_names", []) or [])
    ids: list[int] = []
    missing: list[str] = []
    for name in RIGHT_ARM_JOINT_NAMES:
        if name not in joint_names:
            missing.append(name)
            continue
        ids.append(int(joint_names.index(name)))
    if missing:
        raise RuntimeError(f"Cannot resolve cuRobo right-arm joints in Isaac articulation: missing={missing}")
    resolved = [joint_names[i] for i in ids]
    if resolved != list(RIGHT_ARM_JOINT_NAMES):
        raise RuntimeError(f"cuRobo joint-order adapter failed: resolved={resolved}, expected={RIGHT_ARM_JOINT_NAMES}")
    return ids, resolved


def right_arm_curobo_entity_cfg(robot: Articulation, base_entity_cfg: SceneEntityCfg) -> Any:
    joint_ids, resolved = right_arm_curobo_joint_ids(robot)
    return SimpleNamespace(joint_ids=joint_ids, body_ids=list(base_entity_cfg.body_ids), joint_names=resolved)


def curobo_stage_frame_diagnostics(
    robot: Articulation,
    ee_body_id: int,
    tcp_pose_w: np.ndarray,
    start_q: np.ndarray,
    joint_ids: list[int],
) -> dict[str, Any]:
    target_pose = np.asarray(tcp_pose_w, dtype=np.float32).reshape(4, 4)
    current_tcp = tcp_pose_matrix(robot, ee_body_id)
    right_arm_base = body_pose_matrix_by_name(robot, RIGHT_ARM_BASE_BODY)
    root_pose = robot_root_pose_matrix(robot)
    target_world = target_pose[:3, 3].astype(np.float32)
    current_tcp_world = current_tcp[:3, 3].astype(np.float32)
    diagnostics: dict[str, Any] = {
        "adapter": "explicit_isaac_joint_names_to_curobo_right_arm",
        "curobo_joint_names": list(RIGHT_ARM_JOINT_NAMES),
        "isaac_joint_ids": [int(i) for i in joint_ids],
        "isaac_joint_names": [str(getattr(robot, "joint_names", [])[int(i)]) for i in joint_ids],
        "start_q_by_curobo_name": {name: float(value) for name, value in zip(RIGHT_ARM_JOINT_NAMES, np.asarray(start_q, dtype=np.float32).reshape(-1))},
        "robot_root_world": pose_translation_quat_dict(root_pose),
        "right_arm_base_world": pose_translation_quat_dict(right_arm_base),
        "current_tcp_world": pose_translation_quat_dict(current_tcp),
        "target_tcp_world": pose_translation_quat_dict(target_pose),
        "target_in_robot_root_m": point_world_in_frame(target_world, root_pose),
        "target_in_right_arm_base_m": point_world_in_frame(target_world, right_arm_base),
        "current_tcp_in_robot_root_m": point_world_in_frame(current_tcp_world, root_pose),
        "target_minus_current_tcp_world_m": (target_world - current_tcp_world).astype(float).tolist(),
        "target_current_tcp_distance_m": float(np.linalg.norm(target_world - current_tcp_world)),
    }
    return diagnostics


def joint_index_by_name(robot: Articulation, joint_name: str) -> int | None:
    joint_names = getattr(robot, "joint_names", None)
    if joint_names is None or joint_name not in joint_names:
        return None
    return int(joint_names.index(joint_name))


def clamp_joint_position_to_limits(robot: Articulation, joint_name: str, value: float) -> float:
    idx = joint_index_by_name(robot, joint_name)
    if idx is None:
        return float(value)
    limits = getattr(robot.data, "soft_joint_pos_limits", None)
    if limits is None:
        return float(value)
    try:
        lower = float(limits[0, idx, 0].detach().cpu())
        upper = float(limits[0, idx, 1].detach().cpu())
    except Exception:
        return float(value)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        return float(value)
    return max(lower, min(upper, float(value)))


def set_named_joints_in_full_target(robot: Articulation, full_target: torch.Tensor, joint_positions: dict[str, float]) -> None:
    for joint_name, joint_value in joint_positions.items():
        idx = joint_index_by_name(robot, joint_name)
        if idx is None:
            continue
        full_target[:, idx] = clamp_joint_position_to_limits(robot, joint_name, float(joint_value))


def apply_named_joint_state_for_dry_run(robot: Articulation, joint_positions: dict[str, float]) -> None:
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    set_named_joints_in_full_target(robot, joint_pos, joint_positions)
    for joint_name in joint_positions:
        idx = joint_index_by_name(robot, joint_name)
        if idx is not None:
            joint_vel[:, idx] = 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)


def torso_preshape_samples(robot: Articulation) -> list[dict[str, float]]:
    yaw_samples = [math.radians(v) for v in args_cli.torso_preshape_yaw_samples_deg]
    pitch_samples = [math.radians(v) for v in args_cli.torso_preshape_pitch_samples_deg]
    knee_samples = list(args_cli.torso_preshape_knee_samples_rad)
    leg_samples = list(args_cli.torso_preshape_leg_samples_rad)
    if not yaw_samples:
        yaw_samples = [0.0]
    if not pitch_samples:
        pitch_samples = [0.0]
    if not knee_samples:
        knee_samples = [0.0]
    if not leg_samples:
        leg_samples = [0.0]
    samples: list[dict[str, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for knee in knee_samples:
        for leg in leg_samples:
            for pitch in pitch_samples:
                for yaw in yaw_samples:
                    sample = {
                        "knee_joint": clamp_joint_position_to_limits(robot, "knee_joint", float(knee)),
                        "leg_joint": clamp_joint_position_to_limits(robot, "leg_joint", float(leg)),
                        "waist_pitch_joint": clamp_joint_position_to_limits(robot, "waist_pitch_joint", float(pitch)),
                        "waist_yaw_joint": clamp_joint_position_to_limits(robot, "waist_yaw_joint", float(yaw)),
                    }
                    key = tuple(round(sample[name], 6) for name in ("knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint"))
                    if key in seen:
                        continue
                    seen.add(key)
                    samples.append(sample)
    return samples


def torso_preshape_posture_cost(sample: dict[str, float]) -> float:
    return float(
        abs(sample.get("waist_yaw_joint", 0.0))
        + 0.65 * abs(sample.get("waist_pitch_joint", 0.0))
        + 0.30 * abs(sample.get("knee_joint", 0.0))
        + 0.30 * abs(sample.get("leg_joint", 0.0))
    )


def run_torso_preshape_assist(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    gripper: AttachedParallelGripper,
    candidate: SelectedGrasp,
    tcp_grasp_pose_w: np.ndarray,
    tracker: Task319StateTracker,
    attempt_idx: int,
    locked_root_pose_w: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {
        "enabled": bool(args_cli.torso_preshape_assist),
        "mode": "bounded_torso_sample_then_right_arm_only_tcp_ik",
        "success": False,
        "reason": "",
        "samples": [],
        "selected": None,
    }
    if not bool(args_cli.torso_preshape_assist):
        meta["reason"] = "disabled"
        return meta, []

    target_tcp = np.asarray(tcp_grasp_pose_w, dtype=np.float32).reshape(4, 4)
    original_state = capture_dry_run_state(scene, robot)
    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=sim.device)
    right_arm_entity = SceneEntityCfg("robot", joint_names=[RIGHT_ARM_JOINT_EXPR], body_names=[RIGHT_EE_BODY])
    right_arm_entity.resolve(scene)
    ee_body_id = int(right_arm_entity.body_ids[0])
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
    samples = torso_preshape_samples(robot)
    best_record: dict[str, Any] | None = None
    try:
        for sample_index, sample in enumerate(samples):
            restore_dry_run_state(scene, robot, original_state)
            apply_named_joint_state_for_dry_run(robot, sample)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            pose_items = target_reachability_tcp_poses(target_tcp[:3, 3], robot, ee_body_id, "torso_preshape_target_center")
            probe_records: list[dict[str, Any]] = []
            for pose_item in pose_items:
                restore_dry_run_state(scene, robot, original_state)
                apply_named_joint_state_for_dry_run(robot, sample)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim.get_physics_dt())
                probe = dry_run_ik_to_wrist_pose(
                    sim,
                    scene,
                    robot,
                    ik_controller,
                    right_arm_entity,
                    ee_body_id,
                    ee_jacobi_idx,
                    pose_item["wrist_pose_world"],
                    pose_item["tcp_pose_world"],
                    steps=max(1, int(args_cli.torso_preshape_probe_steps)),
                    locked_root_pose_w=locked_root_pose_w,
                    control_tcp_position=True,
                )
                probe["mode"] = pose_item["mode"]
                probe_records.append(probe)
            best_probe = min(probe_records, key=lambda item: float(item.get("final_tcp_error_m", float("inf")))) if probe_records else {}
            best_error = float(best_probe.get("final_tcp_error_m", float("inf")))
            score = best_error + float(args_cli.torso_preshape_score_posture_weight) * torso_preshape_posture_cost(sample)
            record = {
                "sample_index": int(sample_index),
                "joint_positions_rad": {name: float(value) for name, value in sample.items()},
                "joint_positions_deg": {
                    name: float(math.degrees(value)) if "waist" in name else float(value)
                    for name, value in sample.items()
                },
                "best_tcp_error_m": best_error,
                "score": float(score),
                "best_probe": best_probe,
                "probe_count": len(probe_records),
            }
            meta["samples"].append(record)
            if best_record is None or float(record["score"]) < float(best_record["score"]):
                best_record = record
    finally:
        restore_dry_run_state(scene, robot, original_state)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())

    if best_record is None:
        meta["reason"] = "no_valid_samples"
        return meta, []

    selected_joints = {name: float(value) for name, value in best_record["joint_positions_rad"].items()}
    meta["selected"] = best_record
    selected_error = float(best_record.get("best_tcp_error_m", float("inf")))
    apply_threshold = float(args_cli.torso_preshape_apply_error_threshold)
    meta["apply_error_threshold_m"] = apply_threshold
    if not math.isfinite(selected_error):
        meta["success"] = False
        meta["reason"] = "selected_sample_has_nonfinite_ik_error"
        return meta, []
    if selected_error > apply_threshold:
        meta["success"] = False
        meta["reason"] = "best_sample_error_above_apply_threshold"
        return meta, []
    meta["success"] = True
    meta["reason"] = "selected_lowest_score_sample"

    torso_entity = SceneEntityCfg(
        "robot",
        joint_names=["knee_joint", "leg_joint", TORSO_JOINT_EXPR],
        body_names=[RIGHT_EE_BODY],
    )
    torso_entity.resolve(scene)
    target_full = robot.data.joint_pos.clone()
    set_named_joints_in_full_target(robot, target_full, selected_joints)
    tracker.enter("TORSO_PRESHAPE", attempt_index=attempt_idx, backend="torso_preshape_assist", selected=best_record)
    segment = run_joint_segment(
        sim,
        scene,
        robot,
        torso_entity,
        target_full[:, torso_entity.joint_ids],
        max(1, int(args_cli.torso_preshape_move_steps)),
        gripper=gripper,
        gripper_width=gripper_command_width(gripper, candidate.width),
        locked_root_pose_w=locked_root_pose_w,
    )
    segment["target_named_joint_positions_rad"] = selected_joints
    segment["final_named_joint_positions_rad"] = joint_positions_by_name(robot, list(selected_joints.keys()))
    return meta, [{"name": "torso_preshape", **segment}]


def run_collision_safe_pregrasp_waypoints(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    gripper: AttachedParallelGripper,
    ik_controller: DifferentialIKController,
    entity_cfg: SceneEntityCfg,
    ee_body_id: int,
    ee_jacobi_idx: int,
    poses: dict[str, np.ndarray],
    candidate: SelectedGrasp,
    tracker: Task319StateTracker,
    attempt_idx: int,
    locked_root_pose_w: torch.Tensor,
    *,
    backend: str,
    control_tcp_position: bool = True,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any], str]:
    safe_waypoints = safe_pregrasp_waypoints(poses, robot, ee_body_id)
    safe_meta: dict[str, Any] = {
        "enabled": bool(args_cli.safe_pregrasp_start),
        "mode": "lift_current_tcp_then_high_standoff_then_vertical_descent",
        "table_surface_z_m": float(TABLE_SURFACE_Z),
        "safe_z_m": float(safe_waypoints["safe_z_m"]),
        "table_clearance_m": float(safe_waypoints["safe_z_m"] - TABLE_SURFACE_Z),
        "object_clearance_m": float(safe_waypoints["safe_z_m"] - poses["tcp_grasp"][2, 3]),
        "tcp_safe_lift_current_world": safe_waypoints["tcp_safe_lift_current"].tolist(),
        "tcp_safe_standoff_world": safe_waypoints["tcp_safe_standoff"].tolist(),
        "lift_current_uses_pregrasp_wrist_rotation": bool(safe_waypoints.get("lift_current_uses_pregrasp_wrist_rotation", False)),
        "error_threshold_m": float(args_cli.safe_pregrasp_error_threshold_m),
    }
    if not bool(args_cli.safe_pregrasp_start):
        return True, [], safe_meta, ""

    segments: list[dict[str, Any]] = []
    open_width = gripper_fully_open_width(gripper)

    tracker.enter("SAFE_START", attempt_index=attempt_idx, stage="lift_current_tcp", backend=backend)
    safe_lift_segment = run_ik_segment_until_converged(
        sim,
        scene,
        robot,
        ik_controller,
        entity_cfg,
        ee_body_id,
        ee_jacobi_idx,
        safe_waypoints["wrist_safe_lift_current"],
        int(args_cli.safe_pregrasp_steps),
        float(args_cli.safe_pregrasp_error_threshold_m),
        gripper=gripper,
        gripper_width=open_width,
        locked_root_pose_w=locked_root_pose_w,
        debug_target_pose_w=poses["tcp_grasp"],
        target_tcp_pose_w=safe_waypoints["tcp_safe_lift_current"],
        control_tcp_position=control_tcp_position,
    )
    segments.append({"name": "safe_lift_current", **safe_lift_segment})
    if not bool(safe_lift_segment.get("converged")):
        safe_meta["failed_stage"] = "lift_current_tcp"
        return False, segments, safe_meta, "SAFE_START lift-current TCP error exceeded threshold."

    tracker.enter("SAFE_START", attempt_index=attempt_idx, stage="high_standoff", backend=backend)
    safe_standoff_segment = run_ik_segment_until_converged(
        sim,
        scene,
        robot,
        ik_controller,
        entity_cfg,
        ee_body_id,
        ee_jacobi_idx,
        safe_waypoints["wrist_safe_standoff"],
        int(args_cli.safe_pregrasp_steps),
        float(args_cli.safe_pregrasp_error_threshold_m),
        gripper=gripper,
        gripper_width=open_width,
        locked_root_pose_w=locked_root_pose_w,
        debug_target_pose_w=poses["tcp_grasp"],
        target_tcp_pose_w=safe_waypoints["tcp_safe_standoff"],
        control_tcp_position=control_tcp_position,
    )
    segments.append({"name": "safe_standoff", **safe_standoff_segment})
    if not bool(safe_standoff_segment.get("converged")):
        safe_meta["failed_stage"] = "high_standoff"
        return False, segments, safe_meta, "SAFE_START high-standoff TCP error exceeded threshold."

    return True, segments, safe_meta, ""


def execute_local_position_primitive_attempt(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    gripper: AttachedParallelGripper,
    pose_ik_controller: DifferentialIKController,
    position_ik_controller: DifferentialIKController,
    entity_cfg: SceneEntityCfg,
    ee_body_id: int,
    ee_jacobi_idx: int,
    poses: dict[str, np.ndarray],
    candidate: SelectedGrasp,
    tracker: Task319StateTracker,
    attempt_idx: int,
    locked_root_pose_w: torch.Tensor,
    calibration_camera_to_world: np.ndarray | None = None,
) -> dict[str, Any]:
    primitive = position_only_grasp_poses(poses, robot, ee_body_id, candidate)
    segments: list[dict[str, Any]] = []
    torso_preshape_meta, torso_preshape_segments = run_torso_preshape_assist(
        sim,
        scene,
        robot,
        gripper,
        candidate,
        primitive["tcp_grasp"],
        tracker,
        attempt_idx,
        locked_root_pose_w,
    )
    segments.extend(torso_preshape_segments)
    if bool(torso_preshape_meta.get("success", False)):
        primitive = position_only_grasp_poses(poses, robot, ee_body_id, candidate)
    use_pose_tracking = bool(args_cli.arm_motion_enforce_wrist_orientation) and args_cli.arm_motion_wrist_orientation in {"side_pinch", "top_down", "angled_top_down"}
    stage_ik_controller = pose_ik_controller if use_pose_tracking else position_ik_controller
    stage_tcp_position_control = not use_pose_tracking
    updates: dict[str, Any] = {
        "motion_backend": "local_position_primitive",
        "grasp_motion_profile": STABLE_GRASP_MOTION_PROFILE,
        "action_generation_logic": "rgbd_center_tcp_position_only_safe_pregrasp_vertical_descent_slow_close_lift",
        "ik_command_type": str(stage_ik_controller.cfg.command_type),
        "tcp_position_control": bool(stage_tcp_position_control),
        "orientation_constraint_mode": f"fixed_{args_cli.arm_motion_wrist_orientation}_wrist" if use_pose_tracking else "position_only_tcp_nominal_wrist_only",
        "axis_alignment_required": False,
        "wrist_orientation_hard_constraint": bool(use_pose_tracking),
        "isaaclab_grasp_ik": isaaclab_grasp_ik_metadata(),
        "safe_pregrasp": {
            "enabled": bool(args_cli.safe_pregrasp_start),
            "mode": "position_only_high_standoff_vertical_descent",
            "tcp_pregrasp_world": primitive["tcp_pregrasp"].tolist(),
            "pregrasp_clearance_m": primitive["pregrasp_clearance_m"],
            "minimum_table_clearance_m": float(args_cli.arm_motion_min_table_clearance_m),
        },
        "tcp_pregrasp_pose_world": primitive["tcp_pregrasp"].tolist(),
        "wrist_pregrasp_pose_world": primitive["wrist_pregrasp"].tolist(),
        "tcp_grasp_pose_world": primitive["tcp_grasp"].tolist(),
        "wrist_grasp_pose_world": primitive["wrist_grasp"].tolist(),
        "tcp_lift_pose_world": primitive["tcp_lift"].tolist(),
        "wrist_lift_pose_world": primitive["wrist_lift"].tolist(),
        "position_only_wrist_rotation_world": primitive["wrist_rotation_world"],
        "wrist_orientation_mode": primitive["wrist_orientation_mode"],
        "low_stage_control_mode": "tcp_position_only_after_pregrasp",
        "commanded_position_source": primitive["commanded_position_source"],
        "pregrasp_position_source": primitive["pregrasp_position_source"],
        "vertical_descent_only": bool(primitive["vertical_descent_only"]),
        "top_grasp": primitive["top_grasp"],
        "torso_preshape_assist": torso_preshape_meta,
    }

    open_width = gripper_fully_open_width(gripper)
    if args_cli.arm_motion_ready_pose == "carry" and int(args_cli.arm_motion_ready_steps) > 0:
        tracker.enter("SAFE_START", attempt_index=attempt_idx, stage="carry_ready_pose")
        ready_target = right_arm_joint_target_from_named_pose(robot, entity_cfg, RIGHT_ARM_CARRY_JOINT_POS)
        ready_segment = run_joint_segment(
            sim,
            scene,
            robot,
            entity_cfg,
            ready_target,
            int(args_cli.arm_motion_ready_steps),
            gripper=gripper,
            gripper_width=open_width,
            locked_root_pose_w=locked_root_pose_w,
        )
        segments.append({"name": "carry_ready_pose", **ready_segment})

    gripper.set_width(open_width)
    safe_ok, safe_segments, safe_meta, safe_reason = run_collision_safe_pregrasp_waypoints(
        sim,
        scene,
        robot,
        gripper,
        stage_ik_controller,
        entity_cfg,
        ee_body_id,
        ee_jacobi_idx,
        primitive,
        candidate,
        tracker,
        attempt_idx,
        locked_root_pose_w,
        backend="local_position_primitive",
        control_tcp_position=stage_tcp_position_control,
    )
    updates["safe_pregrasp"].update(safe_meta)
    segments.extend(safe_segments)
    if not safe_ok:
        updates["segments"] = segments
        return {
            "success": False,
            "reason": safe_reason,
            "segments": segments,
            "updates": updates,
            "lift_segment": None,
        }

    tracker.enter("PRE_GRASP", attempt_index=attempt_idx, backend="local_position_primitive")
    pre_segment = run_ik_segment_until_converged(
        sim,
        scene,
        robot,
        stage_ik_controller,
        entity_cfg,
        ee_body_id,
        ee_jacobi_idx,
        primitive["wrist_pregrasp"],
        int(args_cli.trajectory_steps),
        float(args_cli.pregrasp_error_threshold_m),
        gripper=gripper,
        gripper_width=open_width,
        locked_root_pose_w=locked_root_pose_w,
        debug_target_pose_w=primitive["tcp_grasp"],
        target_tcp_pose_w=primitive["tcp_pregrasp"],
        control_tcp_position=stage_tcp_position_control,
    )
    segments.append({"name": "pregrasp_position_only", **pre_segment})
    current_tcp_pose_w = tcp_pose_matrix(robot, ee_body_id)
    render_calibration_debug_markers(scene, primitive["tcp_grasp"], current_tcp_pose_w)
    updates["tcp_pose_after_pregrasp_world"] = current_tcp_pose_w.tolist()
    updates["gripper_alignment_after_pregrasp"] = gripper_alignment_diagnostics(robot, ee_body_id, primitive["tcp_grasp"])
    if calibration_camera_to_world is not None:
        print_calibration_triplet(
            phase=f"attempt_{attempt_idx}_after_pregrasp_position_only",
            camera_to_world=calibration_camera_to_world,
            target_pose_world=primitive["tcp_grasp"],
            tcp_pose_world=current_tcp_pose_w,
            gripper_pose_world=body_pose_matrix_by_name(robot, "gripper_base"),
        )
    if not pre_segment["converged"]:
        updates["segments"] = segments
        return {
            "success": False,
            "reason": "PRE_GRASP position-only IK error exceeded threshold.",
            "segments": segments,
            "updates": updates,
            "lift_segment": None,
        }

    tracker.enter("GRASP", attempt_index=attempt_idx, stage="approach_center", backend="local_position_primitive")
    low_stage_wrist_rot_w = tcp_pose_matrix(robot, ee_body_id)[:3, :3]
    low_stage_wrist_grasp = tcp_pose_to_wrist_pose(primitive["tcp_grasp"], low_stage_wrist_rot_w)
    grasp_segment = run_ik_segment_until_converged(
        sim,
        scene,
        robot,
        position_ik_controller,
        entity_cfg,
        ee_body_id,
        ee_jacobi_idx,
        low_stage_wrist_grasp,
        int(args_cli.trajectory_steps),
        float(args_cli.grasp_error_threshold_m),
        gripper=gripper,
        gripper_width=open_width,
        locked_root_pose_w=locked_root_pose_w,
        debug_target_pose_w=primitive["tcp_grasp"],
        target_tcp_pose_w=primitive["tcp_grasp"],
        control_tcp_position=True,
    )
    segments.append({"name": "grasp_tcp_position_only", **grasp_segment})
    current_tcp_pose_w = tcp_pose_matrix(robot, ee_body_id)
    render_calibration_debug_markers(scene, primitive["tcp_grasp"], current_tcp_pose_w)
    updates["tcp_pose_after_grasp_world"] = current_tcp_pose_w.tolist()
    updates["gripper_alignment_after_grasp"] = gripper_alignment_diagnostics(robot, ee_body_id, primitive["tcp_grasp"])
    if calibration_camera_to_world is not None:
        print_calibration_triplet(
            phase=f"attempt_{attempt_idx}_after_grasp_position_only",
            camera_to_world=calibration_camera_to_world,
            target_pose_world=primitive["tcp_grasp"],
            tcp_pose_world=current_tcp_pose_w,
        )
    if (
        not grasp_segment["converged"]
        and bool(args_cli.gripper_tcp_feedback_correction)
        and np.isfinite(current_tcp_pose_w[:3, 3]).all()
    ):
        original_tcp_grasp = primitive["tcp_grasp"].copy()
        residual_w = original_tcp_grasp[:3, 3] - current_tcp_pose_w[:3, 3]
        corrected_tcp_grasp = original_tcp_grasp.copy()
        corrected_tcp_grasp[:3, 3] = original_tcp_grasp[:3, 3] + float(args_cli.gripper_tcp_feedback_gain) * residual_w
        tracker.enter("GRASP", attempt_index=attempt_idx, stage="tcp_feedback_correction", backend="local_position_primitive")
        correction_wrist_rot_w = tcp_pose_matrix(robot, ee_body_id)[:3, :3]
        corrected_wrist_grasp = tcp_pose_to_wrist_pose(corrected_tcp_grasp, correction_wrist_rot_w)
        correction_segment = run_ik_segment_until_converged(
            sim,
            scene,
            robot,
            position_ik_controller,
            entity_cfg,
            ee_body_id,
            ee_jacobi_idx,
            corrected_wrist_grasp,
            int(args_cli.trajectory_steps),
            float(args_cli.grasp_error_threshold_m),
            gripper=gripper,
            gripper_width=open_width,
            locked_root_pose_w=locked_root_pose_w,
            debug_target_pose_w=original_tcp_grasp,
            target_tcp_pose_w=corrected_tcp_grasp,
            control_tcp_position=True,
        )
        current_tcp_pose_w = tcp_pose_matrix(robot, ee_body_id)
        original_error = float(np.linalg.norm(current_tcp_pose_w[:3, 3] - original_tcp_grasp[:3, 3]))
        correction_segment.update(
            {
                "feedback_gain": float(args_cli.gripper_tcp_feedback_gain),
                "pre_correction_residual_world_m": residual_w.astype(float).tolist(),
                "corrected_target_tcp_world_m": corrected_tcp_grasp[:3, 3].astype(float).tolist(),
                "original_target_tcp_world_m": original_tcp_grasp[:3, 3].astype(float).tolist(),
                "final_tcp_error_to_original_target_m": original_error,
                "converged_to_original_target": bool(original_error <= float(args_cli.grasp_error_threshold_m)),
            }
        )
        segments.append({"name": "grasp_tcp_feedback_correction", **correction_segment})
        render_calibration_debug_markers(scene, original_tcp_grasp, current_tcp_pose_w)
        updates["tcp_pose_after_grasp_feedback_world"] = current_tcp_pose_w.tolist()
        updates["gripper_alignment_after_grasp_feedback"] = gripper_alignment_diagnostics(robot, ee_body_id, original_tcp_grasp)
        if calibration_camera_to_world is not None:
            print_calibration_triplet(
                phase=f"attempt_{attempt_idx}_after_grasp_tcp_feedback",
                camera_to_world=calibration_camera_to_world,
                target_pose_world=original_tcp_grasp,
                tcp_pose_world=current_tcp_pose_w,
            )
        if original_error <= float(args_cli.grasp_error_threshold_m):
            correction_segment["converged"] = True
            correction_segment["final_pos_error_m"] = original_error
            correction_segment["final_tcp_pos_error_m"] = original_error
            correction_segment["position_error_reference"] = "original_target_tcp_after_feedback"
            segments[-1].update(
                {
                    "converged": True,
                    "final_pos_error_m": original_error,
                    "final_tcp_pos_error_m": original_error,
                    "position_error_reference": "original_target_tcp_after_feedback",
                }
            )
            grasp_segment = correction_segment
    if not grasp_segment["converged"]:
        updates["segments"] = segments
        return {
            "success": False,
            "reason": "GRASP position-only IK error exceeded threshold.",
            "segments": segments,
            "updates": updates,
            "lift_segment": None,
        }

    tracker.enter("GRASP", attempt_index=attempt_idx, stage="close_gripper", backend="local_position_primitive")
    grasp_hold_joint_target = robot.data.joint_pos.clone()
    close_segment = run_slow_gripper_close(
        sim,
        scene,
        robot,
        gripper,
        tracker,
        grasp_hold_joint_target,
        locked_root_pose_w,
        start_width_m=open_width,
        steps=int(args_cli.grasp_steps),
    )
    segments.append({"name": "close_gripper", **close_segment})
    updates["gripper_alignment_after_close"] = gripper_alignment_diagnostics(robot, ee_body_id, primitive["tcp_grasp"])
    contact_ok, contact_meta, contact_reason = evaluate_gripper_contact_before_lift(close_segment, candidate)
    updates["gripper_contact_check"] = contact_meta
    if not contact_ok:
        updates["segments"] = segments
        return {
            "success": False,
            "reason": contact_reason,
            "segments": segments,
            "updates": updates,
            "lift_segment": None,
        }

    tracker.enter("LIFT", attempt_index=attempt_idx, backend="local_position_primitive")
    lift_wrist_rot_w = tcp_pose_matrix(robot, ee_body_id)[:3, :3]
    low_stage_wrist_lift = tcp_pose_to_wrist_pose(primitive["tcp_lift"], lift_wrist_rot_w)
    lift_segment = run_ik_segment_until_converged(
        sim,
        scene,
        robot,
        position_ik_controller,
        entity_cfg,
        ee_body_id,
        ee_jacobi_idx,
        low_stage_wrist_lift,
        int(args_cli.lift_steps),
        float(args_cli.lift_error_threshold_m),
        gripper=gripper,
        gripper_width=0.0,
        locked_root_pose_w=locked_root_pose_w,
        debug_target_pose_w=primitive["tcp_grasp"],
        target_tcp_pose_w=primitive["tcp_lift"],
        control_tcp_position=True,
    )
    segments.append({"name": "lift_tcp_position_only", **lift_segment})
    updates["tcp_pose_after_lift_world"] = tcp_pose_matrix(robot, ee_body_id).tolist()
    updates["gripper_alignment_after_lift"] = gripper_alignment_diagnostics(robot, ee_body_id, primitive["tcp_lift"])

    tracker.enter("VERIFY_HOLD", attempt_index=attempt_idx, backend="local_position_primitive")
    lift_hold_joint_target = robot.data.joint_pos.clone()
    for _ in range(int(args_cli.hold_steps)):
        robot.set_joint_position_target(lift_hold_joint_target)
        gripper.close()
        stabilize_robot_base(robot, locked_root_pose_w)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim.get_physics_dt())
        record_video_frame(scene)
        gui_playback_tick(sim.get_physics_dt())
        tracker.tick()

    updates["segments"] = segments
    return {
        "success": bool(lift_segment["converged"]),
        "reason": "" if lift_segment["converged"] else "LIFT position-only IK error exceeded threshold.",
        "segments": segments,
        "updates": updates,
        "lift_segment": lift_segment,
    }


def execute_kuavo_ik_primitive_attempt(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    gripper: AttachedParallelGripper,
    kuavo_client: KuavoIkSocketClient,
    position_ik_controller: DifferentialIKController,
    entity_cfg: SceneEntityCfg,
    ee_body_id: int,
    ee_jacobi_idx: int,
    poses: dict[str, np.ndarray],
    candidate: SelectedGrasp,
    tracker: Task319StateTracker,
    attempt_idx: int,
    locked_root_pose_w: torch.Tensor,
    calibration_camera_to_world: np.ndarray | None = None,
) -> dict[str, Any]:
    primitive = position_only_grasp_poses(poses, robot, ee_body_id, candidate)
    left_entity_cfg = SceneEntityCfg("robot", body_names=[LEFT_EE_BODY])
    left_entity_cfg.resolve(scene)
    left_body_id = int(left_entity_cfg.body_ids[0])
    q_arm = current_bilateral_arm_q(robot)
    if q_arm is None:
        return {
            "success": False,
            "reason": "Could not resolve bilateral arm q0 for Kuavo IK.",
            "segments": [],
            "updates": {"motion_backend": "kuavo_ik", "kuavo_ik_available": False},
            "lift_segment": None,
        }

    segments: list[dict[str, Any]] = []
    updates: dict[str, Any] = {
        "motion_backend": "kuavo_ik",
        "kuavo_ik_available": True,
        "kuavo_ik_service": args_cli.kuavo_ik_service,
        "kuavo_ik_frame": int(args_cli.kuavo_ik_frame),
        "kuavo_ik_constraint_mode": int(args_cli.kuavo_ik_constraint_mode),
        "ik_command_type": "official_kuavo_ik_joint_segment",
        "tcp_position_control": True,
        "orientation_constraint_mode": "kuavo_position_hard_orientation_soft",
        "axis_alignment_required": False,
        "safe_pregrasp": {
            "enabled": bool(args_cli.safe_pregrasp_start),
            "mode": "kuavo_ik_high_standoff_vertical_descent",
            "tcp_pregrasp_world": primitive["tcp_pregrasp"].tolist(),
            "pregrasp_clearance_m": primitive["pregrasp_clearance_m"],
            "minimum_table_clearance_m": float(args_cli.arm_motion_min_table_clearance_m),
        },
        "tcp_pregrasp_pose_world": primitive["tcp_pregrasp"].tolist(),
        "wrist_pregrasp_pose_world": primitive["wrist_pregrasp"].tolist(),
        "tcp_grasp_pose_world": primitive["tcp_grasp"].tolist(),
        "wrist_grasp_pose_world": primitive["wrist_grasp"].tolist(),
        "tcp_lift_pose_world": primitive["tcp_lift"].tolist(),
        "wrist_lift_pose_world": primitive["wrist_lift"].tolist(),
        "position_only_wrist_rotation_world": primitive["wrist_rotation_world"],
        "commanded_position_source": primitive["commanded_position_source"],
        "pregrasp_position_source": primitive["pregrasp_position_source"],
        "vertical_descent_only": bool(primitive["vertical_descent_only"]),
        "top_grasp": primitive["top_grasp"],
    }
    open_width = gripper_fully_open_width(gripper)

    if args_cli.arm_motion_ready_pose == "carry" and int(args_cli.arm_motion_ready_steps) > 0:
        tracker.enter("SAFE_START", attempt_index=attempt_idx, stage="carry_ready_pose", backend="kuavo_ik")
        ready_target = right_arm_joint_target_from_named_pose(robot, entity_cfg, RIGHT_ARM_CARRY_JOINT_POS)
        ready_segment = run_joint_segment(
            sim,
            scene,
            robot,
            entity_cfg,
            ready_target,
            int(args_cli.arm_motion_ready_steps),
            gripper=gripper,
            gripper_width=open_width,
            locked_root_pose_w=locked_root_pose_w,
        )
        segments.append({"name": "carry_ready_pose", **ready_segment})
        q_arm = current_bilateral_arm_q(robot) or q_arm

    def solve_and_run_stage(name: str, state_name: str, wrist_pose_w: np.ndarray, tcp_pose_w: np.ndarray, steps: int, threshold_m: float, width: float) -> tuple[bool, dict[str, Any], str]:
        tracker.enter(state_name, attempt_index=attempt_idx, stage=name, backend="kuavo_ik")
        left_pose_w = current_body_pose_matrix(robot, left_body_id)
        response = kuavo_client.solve(
            label=f"attempt_{attempt_idx}_{name}",
            robot=robot,
            left_pose_w=left_pose_w,
            right_pose_w=wrist_pose_w,
            q_arm=q_arm,
        )
        if not bool(response.get("success")):
            return False, {"name": name, "kuavo_ik_response": response}, str(response.get("error_reason") or "Kuavo IK solve failed.")
        solved_q_arm = response.get("q_arm") or []
        if len(solved_q_arm) < 14:
            return False, {"name": name, "kuavo_ik_response": response}, f"Kuavo IK returned q_arm length {len(solved_q_arm)}, expected 14."
        right_q = torch.tensor(solved_q_arm[7:14], dtype=torch.float32, device=robot.device).reshape(1, 7)
        segment = run_joint_segment(
            sim,
            scene,
            robot,
            entity_cfg,
            right_q,
            int(steps),
            gripper=gripper,
            gripper_width=width,
            locked_root_pose_w=locked_root_pose_w,
            target_pose_w=wrist_pose_w,
        )
        final_tcp = tcp_pose_matrix(robot, ee_body_id)
        final_tcp_error = float(np.linalg.norm(final_tcp[:3, 3] - np.asarray(tcp_pose_w, dtype=np.float32)[:3, 3]))
        segment.update(
            {
                "name": name,
                "kuavo_ik_response": response,
                "position_error_reference": "tcp",
                "target_tcp_world_m": np.asarray(tcp_pose_w, dtype=np.float32)[:3, 3].astype(float).tolist(),
                "final_tcp_world_m": final_tcp[:3, 3].astype(float).tolist(),
                "final_tcp_pos_error_m": final_tcp_error,
                "final_pos_error_m": final_tcp_error,
                "error_threshold_m": float(threshold_m),
                "converged": bool(final_tcp_error <= float(threshold_m)),
            }
        )
        return bool(segment["converged"]), segment, ""

    gripper.set_width(open_width)
    safe_ok, safe_segments, safe_meta, safe_reason = run_collision_safe_pregrasp_waypoints(
        sim,
        scene,
        robot,
        gripper,
        position_ik_controller,
        entity_cfg,
        ee_body_id,
        ee_jacobi_idx,
        primitive,
        candidate,
        tracker,
        attempt_idx,
        locked_root_pose_w,
        backend="kuavo_ik",
        control_tcp_position=True,
    )
    updates["safe_pregrasp"].update(safe_meta)
    segments.extend(safe_segments)
    if not safe_ok:
        updates["segments"] = segments
        return {
            "success": False,
            "reason": safe_reason,
            "segments": segments,
            "updates": updates,
            "lift_segment": None,
        }
    q_arm = current_bilateral_arm_q(robot) or q_arm

    ok, pre_segment, reason = solve_and_run_stage(
        "pregrasp_kuavo_ik",
        "PRE_GRASP",
        primitive["wrist_pregrasp"],
        primitive["tcp_pregrasp"],
        int(args_cli.trajectory_steps),
        float(args_cli.pregrasp_error_threshold_m),
        open_width,
    )
    segments.append(pre_segment)
    updates["tcp_pose_after_pregrasp_world"] = tcp_pose_matrix(robot, ee_body_id).tolist()
    if calibration_camera_to_world is not None:
        print_calibration_triplet(
            phase=f"attempt_{attempt_idx}_after_pregrasp_kuavo_ik",
            camera_to_world=calibration_camera_to_world,
            target_pose_world=primitive["tcp_grasp"],
            tcp_pose_world=tcp_pose_matrix(robot, ee_body_id),
        )
    if not ok:
        return {"success": False, "reason": f"PRE_GRASP Kuavo IK failed: {reason}", "segments": segments, "updates": updates, "lift_segment": None}
    q_arm = current_bilateral_arm_q(robot) or q_arm

    ok, grasp_segment, reason = solve_and_run_stage(
        "grasp_kuavo_ik",
        "GRASP",
        primitive["wrist_grasp"],
        primitive["tcp_grasp"],
        int(args_cli.trajectory_steps),
        float(args_cli.grasp_error_threshold_m),
        open_width,
    )
    segments.append(grasp_segment)
    updates["tcp_pose_after_grasp_world"] = tcp_pose_matrix(robot, ee_body_id).tolist()
    if calibration_camera_to_world is not None:
        print_calibration_triplet(
            phase=f"attempt_{attempt_idx}_after_grasp_kuavo_ik",
            camera_to_world=calibration_camera_to_world,
            target_pose_world=primitive["tcp_grasp"],
            tcp_pose_world=tcp_pose_matrix(robot, ee_body_id),
        )
    if not ok:
        return {"success": False, "reason": f"GRASP Kuavo IK failed: {reason}", "segments": segments, "updates": updates, "lift_segment": None}
    q_arm = current_bilateral_arm_q(robot) or q_arm

    tracker.enter("GRASP", attempt_index=attempt_idx, stage="close_gripper", backend="kuavo_ik")
    grasp_hold_joint_target = robot.data.joint_pos.clone()
    close_segment = run_slow_gripper_close(
        sim,
        scene,
        robot,
        gripper,
        tracker,
        grasp_hold_joint_target,
        locked_root_pose_w,
        start_width_m=open_width,
        steps=int(args_cli.grasp_steps),
    )
    segments.append({"name": "close_gripper", **close_segment})
    contact_ok, contact_meta, contact_reason = evaluate_gripper_contact_before_lift(close_segment, candidate)
    updates["gripper_contact_check"] = contact_meta
    if not contact_ok:
        updates["segments"] = segments
        return {
            "success": False,
            "reason": contact_reason,
            "segments": segments,
            "updates": updates,
            "lift_segment": None,
        }

    ok, lift_segment, reason = solve_and_run_stage(
        "lift_kuavo_ik",
        "LIFT",
        primitive["wrist_lift"],
        primitive["tcp_lift"],
        int(args_cli.lift_steps),
        float(args_cli.lift_error_threshold_m),
        0.0,
    )
    segments.append(lift_segment)
    updates["tcp_pose_after_lift_world"] = tcp_pose_matrix(robot, ee_body_id).tolist()

    tracker.enter("VERIFY_HOLD", attempt_index=attempt_idx, backend="kuavo_ik")
    lift_hold_joint_target = robot.data.joint_pos.clone()
    for _ in range(int(args_cli.hold_steps)):
        robot.set_joint_position_target(lift_hold_joint_target)
        gripper.close()
        stabilize_robot_base(robot, locked_root_pose_w)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim.get_physics_dt())
        record_video_frame(scene)
        gui_playback_tick(sim.get_physics_dt())
        tracker.tick()

    return {
        "success": bool(ok),
        "reason": "" if ok else f"LIFT Kuavo IK failed: {reason}",
        "segments": segments,
        "updates": updates,
        "lift_segment": lift_segment,
    }


def execute_kuavo_analytic_ik_primitive_attempt(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    gripper: AttachedParallelGripper,
    analytic_client: KuavoAnalyticIkClient,
    position_ik_controller: DifferentialIKController,
    entity_cfg: SceneEntityCfg,
    ee_body_id: int,
    ee_jacobi_idx: int,
    poses: dict[str, np.ndarray],
    candidate: SelectedGrasp,
    tracker: Task319StateTracker,
    attempt_idx: int,
    locked_root_pose_w: torch.Tensor,
    calibration_camera_to_world: np.ndarray | None = None,
) -> dict[str, Any]:
    primitive = position_only_grasp_poses(poses, robot, ee_body_id, candidate)
    base_entity_cfg = SceneEntityCfg("robot", body_names=[RIGHT_ARM_BASE_BODY])
    base_entity_cfg.resolve(scene)
    link7_entity_cfg = SceneEntityCfg("robot", body_names=[RIGHT_ARM_LINK7_BODY])
    link7_entity_cfg.resolve(scene)
    right_base_body_id = int(base_entity_cfg.body_ids[0])
    right_link7_body_id = int(link7_entity_cfg.body_ids[0])

    segments: list[dict[str, Any]] = []
    updates: dict[str, Any] = {
        "motion_backend": "kuavo_analytic_ik",
        "kuavo_analytic_ik_available": True,
        "ik_command_type": "official_kuavo_analytic_ik_seed_plus_tcp_position_refine",
        "tcp_position_control": True,
        "orientation_constraint_mode": "kuavo_official_analytic_seed_then_position_only_tcp",
        "axis_alignment_required": False,
        "kuavo_analytic_ik_tcp_refine": bool(args_cli.kuavo_analytic_ik_tcp_refine),
        "kuavo_analytic_ik_refine_steps": int(args_cli.kuavo_analytic_ik_refine_steps),
        "kuavo_analytic_ik_refine_error_threshold_m": float(args_cli.kuavo_analytic_ik_refine_error_threshold_m),
        "kuavo_analytic_ik_grasp_tcp_threshold_m": float(args_cli.kuavo_analytic_ik_grasp_tcp_threshold_m),
        "kuavo_analytic_ik_execute_top_k": int(args_cli.kuavo_analytic_ik_execute_top_k),
        "kuavo_analytic_ik_local_grasp_descent_on_seed_failure": bool(args_cli.kuavo_analytic_ik_local_grasp_descent_on_seed_failure),
        "kuavo_analytic_ik_max_fk_error_m": float(args_cli.kuavo_analytic_ik_max_fk_error_m),
        "safe_pregrasp": {
            "enabled": bool(args_cli.safe_pregrasp_start),
            "mode": "kuavo_analytic_ik_high_standoff_vertical_descent",
            "tcp_pregrasp_world": primitive["tcp_pregrasp"].tolist(),
            "pregrasp_clearance_m": primitive["pregrasp_clearance_m"],
            "minimum_table_clearance_m": float(args_cli.arm_motion_min_table_clearance_m),
        },
        "tcp_pregrasp_pose_world": primitive["tcp_pregrasp"].tolist(),
        "wrist_pregrasp_pose_world": primitive["wrist_pregrasp"].tolist(),
        "tcp_grasp_pose_world": primitive["tcp_grasp"].tolist(),
        "wrist_grasp_pose_world": primitive["wrist_grasp"].tolist(),
        "tcp_lift_pose_world": primitive["tcp_lift"].tolist(),
        "wrist_lift_pose_world": primitive["wrist_lift"].tolist(),
        "position_only_wrist_rotation_world": primitive["wrist_rotation_world"],
        "wrist_orientation_mode": primitive["wrist_orientation_mode"],
        "commanded_position_source": primitive["commanded_position_source"],
        "pregrasp_position_source": primitive["pregrasp_position_source"],
        "vertical_descent_only": bool(primitive["vertical_descent_only"]),
        "top_grasp": primitive["top_grasp"],
        "right_arm_base_body": RIGHT_ARM_BASE_BODY,
        "right_arm_link7_body": RIGHT_ARM_LINK7_BODY,
    }
    open_width = gripper_fully_open_width(gripper)

    if args_cli.arm_motion_ready_pose == "carry" and int(args_cli.arm_motion_ready_steps) > 0:
        tracker.enter("SAFE_START", attempt_index=attempt_idx, stage="carry_ready_pose", backend="kuavo_analytic_ik")
        ready_target = right_arm_joint_target_from_named_pose(robot, entity_cfg, RIGHT_ARM_CARRY_JOINT_POS)
        ready_segment = run_joint_segment(
            sim,
            scene,
            robot,
            entity_cfg,
            ready_target,
            int(args_cli.arm_motion_ready_steps),
            gripper=gripper,
            gripper_width=open_width,
            locked_root_pose_w=locked_root_pose_w,
        )
        segments.append({"name": "carry_ready_pose", **ready_segment})

    def solve_and_run_stage(name: str, state_name: str, wrist_pose_w: np.ndarray, tcp_pose_w: np.ndarray, steps: int, threshold_m: float, width: float) -> tuple[bool, dict[str, Any], str]:
        tracker.enter(state_name, attempt_index=attempt_idx, stage=name, backend="kuavo_analytic_ik")
        stage_threshold_m = float(threshold_m)
        refine_threshold_m = min(stage_threshold_m, float(args_cli.kuavo_analytic_ik_refine_error_threshold_m))
        if state_name == "GRASP" or "grasp" in name:
            stage_threshold_m = min(stage_threshold_m, float(args_cli.kuavo_analytic_ik_grasp_tcp_threshold_m))
            refine_threshold_m = min(refine_threshold_m, stage_threshold_m)
        roll_attempts: list[dict[str, Any]] = []
        executable_candidates: list[tuple[dict[str, Any], np.ndarray]] = []
        for sample_idx, sample in enumerate(kuavo_analytic_wrist_pose_samples(tcp_pose_w, wrist_pose_w)):
            sample_wrist_pose_w = np.asarray(sample["wrist_pose"], dtype=np.float32)
            sample_response = analytic_client.solve_right(
                label=f"attempt_{attempt_idx}_{name}_roll_{sample_idx}",
                robot=robot,
                right_base_body_id=right_base_body_id,
                right_link7_body_id=right_link7_body_id,
                right_ee_body_id=ee_body_id,
                right_ee_pose_w=sample_wrist_pose_w,
            )
            sample_response["roll_sample_index"] = int(sample_idx)
            sample_response["roll_sample_rad"] = float(sample["roll_rad"])
            sample_response["approach_label"] = str(sample["approach_label"])
            sample_response["approach_tcp_offset_axis_world"] = sample["approach_tcp_offset_axis_world"]
            roll_attempts.append(sample_response)
            if bool(sample_response.get("success")):
                executable_candidates.append((sample_response, sample_wrist_pose_w))
        if not executable_candidates:
            if (state_name == "GRASP" or "grasp" in name) and bool(args_cli.kuavo_analytic_ik_local_grasp_descent_on_seed_failure):
                local_segment = run_ik_segment_until_converged(
                    sim,
                    scene,
                    robot,
                    position_ik_controller,
                    entity_cfg,
                    ee_body_id,
                    ee_jacobi_idx,
                    np.asarray(wrist_pose_w, dtype=np.float32),
                    max(1, int(args_cli.kuavo_analytic_ik_refine_steps)),
                    float(stage_threshold_m),
                    gripper=gripper,
                    gripper_width=width,
                    locked_root_pose_w=locked_root_pose_w,
                    debug_target_pose_w=tcp_pose_w,
                    target_tcp_pose_w=tcp_pose_w,
                    control_tcp_position=True,
                )
                final_tcp = tcp_pose_matrix(robot, ee_body_id)
                final_tcp_error = float(np.linalg.norm(final_tcp[:3, 3] - np.asarray(tcp_pose_w, dtype=np.float32)[:3, 3]))
                response = roll_attempts[-1] if roll_attempts else {"success": False, "error_reason": "no_roll_samples"}
                local_segment.update(
                    {
                        "name": name,
                        "kuavo_analytic_local_grasp_descent_on_seed_failure": True,
                        "kuavo_analytic_ik_response": response,
                        "kuavo_analytic_ik_roll_attempts": roll_attempts,
                        "kuavo_analytic_executable_candidate_count": 0,
                        "position_error_reference": "tcp",
                        "target_tcp_world_m": np.asarray(tcp_pose_w, dtype=np.float32)[:3, 3].astype(float).tolist(),
                        "final_tcp_world_m": final_tcp[:3, 3].astype(float).tolist(),
                        "final_tcp_pos_error_m": final_tcp_error,
                        "final_pos_error_m": final_tcp_error,
                        "error_threshold_m": float(stage_threshold_m),
                        "nominal_error_threshold_m": float(threshold_m),
                        "tcp_refine_error_threshold_m": float(refine_threshold_m),
                        "converged": bool(final_tcp_error <= float(stage_threshold_m)),
                    }
                )
                return bool(local_segment["converged"]), local_segment, str(response.get("error_reason", ""))
            response = roll_attempts[0] if roll_attempts else {"success": False, "error_reason": "no_roll_samples"}
            last_reason = str(roll_attempts[-1].get("error_reason") if roll_attempts else response.get("error_reason"))
            return False, {"name": name, "kuavo_analytic_ik_response": response, "kuavo_analytic_ik_roll_attempts": roll_attempts}, last_reason or "Kuavo analytic IK solve failed."

        executable_candidates.sort(
            key=lambda item: (
                float(item[0].get("fk_position_error_m", float("inf"))),
                int(item[0].get("roll_sample_index", 0)),
            )
        )
        execute_top_k = max(1, int(args_cli.kuavo_analytic_ik_execute_top_k))
        stage_start_state = capture_dry_run_state(scene, robot)
        executed_candidate_summaries: list[dict[str, Any]] = []
        best_segment: dict[str, Any] | None = None
        best_error = float("inf")
        best_reason = ""

        def execute_one_candidate(exec_idx: int, response: dict[str, Any], chosen_wrist_pose_w: np.ndarray) -> tuple[bool, dict[str, Any], str]:
            solved_q = response.get("q") or []
            if len(solved_q) != 7:
                return False, {"name": name, "kuavo_analytic_ik_response": response, "kuavo_analytic_ik_roll_attempts": roll_attempts}, f"Kuavo analytic IK returned q length {len(solved_q)}, expected 7."
            right_q = torch.tensor(solved_q, dtype=torch.float32, device=robot.device).reshape(1, 7)
            seed_segment = run_joint_segment(
                sim,
                scene,
                robot,
                entity_cfg,
                right_q,
                int(steps),
                gripper=gripper,
                gripper_width=width,
                locked_root_pose_w=locked_root_pose_w,
                target_pose_w=chosen_wrist_pose_w,
            )
            final_tcp = tcp_pose_matrix(robot, ee_body_id)
            final_tcp_error = float(np.linalg.norm(final_tcp[:3, 3] - np.asarray(tcp_pose_w, dtype=np.float32)[:3, 3]))
            refine_segment: dict[str, Any] | None = None
            refine_used = False
            if bool(args_cli.kuavo_analytic_ik_tcp_refine) and final_tcp_error > float(refine_threshold_m):
                refine_used = True
                refine_segment = run_ik_segment_until_converged(
                    sim,
                    scene,
                    robot,
                    position_ik_controller,
                    entity_cfg,
                    ee_body_id,
                    ee_jacobi_idx,
                    chosen_wrist_pose_w,
                    max(1, int(args_cli.kuavo_analytic_ik_refine_steps)),
                    float(refine_threshold_m),
                    gripper=gripper,
                    gripper_width=width,
                    locked_root_pose_w=locked_root_pose_w,
                    debug_target_pose_w=tcp_pose_w,
                    target_tcp_pose_w=tcp_pose_w,
                    control_tcp_position=True,
                )
                final_tcp = tcp_pose_matrix(robot, ee_body_id)
                final_tcp_error = float(np.linalg.norm(final_tcp[:3, 3] - np.asarray(tcp_pose_w, dtype=np.float32)[:3, 3]))
            segment = dict(refine_segment) if refine_segment is not None else dict(seed_segment)
            segment.update(
                {
                    "name": name,
                    "kuavo_analytic_execution_index": int(exec_idx),
                    "kuavo_analytic_seed_segment": seed_segment,
                    "kuavo_analytic_tcp_refine_segment": refine_segment,
                    "kuavo_analytic_tcp_refine_enabled": bool(args_cli.kuavo_analytic_ik_tcp_refine),
                    "kuavo_analytic_tcp_refine_used": bool(refine_used),
                    "kuavo_analytic_ik_response": response,
                    "kuavo_analytic_ik_roll_attempts": roll_attempts,
                    "selected_roll_sample_rad": float(response.get("roll_sample_rad", 0.0)),
                    "selected_approach_label": str(response.get("approach_label", "")),
                    "selected_approach_tcp_offset_axis_world": response.get("approach_tcp_offset_axis_world"),
                    "selected_wrist_pose_world": chosen_wrist_pose_w.astype(float).tolist(),
                    "position_error_reference": "tcp",
                    "target_tcp_world_m": np.asarray(tcp_pose_w, dtype=np.float32)[:3, 3].astype(float).tolist(),
                    "final_tcp_world_m": final_tcp[:3, 3].astype(float).tolist(),
                    "final_tcp_pos_error_m": final_tcp_error,
                    "final_pos_error_m": final_tcp_error,
                    "error_threshold_m": float(stage_threshold_m),
                    "nominal_error_threshold_m": float(threshold_m),
                    "tcp_refine_error_threshold_m": float(refine_threshold_m),
                    "converged": bool(final_tcp_error <= float(stage_threshold_m)),
                }
            )
            return bool(segment["converged"]), segment, ""

        for exec_idx, (response, chosen_wrist_pose_w) in enumerate(executable_candidates[:execute_top_k]):
            if exec_idx > 0:
                restore_dry_run_state(scene, robot, stage_start_state)
            ok, segment, reason = execute_one_candidate(exec_idx, response, chosen_wrist_pose_w)
            final_error = float(segment.get("final_tcp_pos_error_m", float("inf")))
            executed_candidate_summaries.append(
                {
                    "execution_index": int(exec_idx),
                    "approach_label": segment.get("selected_approach_label"),
                    "roll_sample_rad": segment.get("selected_roll_sample_rad"),
                    "fk_position_error_m": response.get("fk_position_error_m"),
                    "final_tcp_pos_error_m": final_error,
                    "converged": bool(ok),
                    "reason": reason,
                }
            )
            if final_error < best_error:
                best_error = final_error
                best_segment = segment
                best_reason = reason
            if ok:
                segment["kuavo_analytic_executed_candidate_summaries"] = executed_candidate_summaries
                segment["kuavo_analytic_executable_candidate_count"] = len(executable_candidates)
                return True, segment, ""

        restore_dry_run_state(scene, robot, stage_start_state)
        if best_segment is None:
            response = executable_candidates[0][0]
            return False, {"name": name, "kuavo_analytic_ik_response": response, "kuavo_analytic_ik_roll_attempts": roll_attempts}, "No executable Kuavo analytic IK candidate was run."
        best_segment["kuavo_analytic_executed_candidate_summaries"] = executed_candidate_summaries
        best_segment["kuavo_analytic_executable_candidate_count"] = len(executable_candidates)
        best_segment["stage_restored_after_failed_candidates"] = True
        return False, best_segment, best_reason

    gripper.set_width(open_width)
    safe_ok, safe_segments, safe_meta, safe_reason = run_collision_safe_pregrasp_waypoints(
        sim,
        scene,
        robot,
        gripper,
        position_ik_controller,
        entity_cfg,
        ee_body_id,
        ee_jacobi_idx,
        primitive,
        candidate,
        tracker,
        attempt_idx,
        locked_root_pose_w,
        backend="kuavo_analytic_ik",
        control_tcp_position=True,
    )
    updates["safe_pregrasp"].update(safe_meta)
    segments.extend(safe_segments)
    if not safe_ok:
        updates["segments"] = segments
        return {
            "success": False,
            "reason": safe_reason,
            "segments": segments,
            "updates": updates,
            "lift_segment": None,
        }

    ok, pre_segment, reason = solve_and_run_stage(
        "pregrasp_kuavo_analytic_ik",
        "PRE_GRASP",
        primitive["wrist_pregrasp"],
        primitive["tcp_pregrasp"],
        int(args_cli.trajectory_steps),
        float(args_cli.pregrasp_error_threshold_m),
        open_width,
    )
    segments.append(pre_segment)
    updates["tcp_pose_after_pregrasp_world"] = tcp_pose_matrix(robot, ee_body_id).tolist()
    if calibration_camera_to_world is not None:
        print_calibration_triplet(
            phase=f"attempt_{attempt_idx}_after_pregrasp_kuavo_analytic_ik",
            camera_to_world=calibration_camera_to_world,
            target_pose_world=primitive["tcp_grasp"],
            tcp_pose_world=tcp_pose_matrix(robot, ee_body_id),
        )
    if not ok:
        return {"success": False, "reason": f"PRE_GRASP Kuavo analytic IK failed: {reason}", "segments": segments, "updates": updates, "lift_segment": None}

    ok, grasp_segment, reason = solve_and_run_stage(
        "grasp_kuavo_analytic_ik",
        "GRASP",
        primitive["wrist_grasp"],
        primitive["tcp_grasp"],
        int(args_cli.trajectory_steps),
        float(args_cli.grasp_error_threshold_m),
        gripper_command_width(gripper, candidate.width, clearance_m=0.010),
    )
    segments.append(grasp_segment)
    updates["tcp_pose_after_grasp_world"] = tcp_pose_matrix(robot, ee_body_id).tolist()
    if calibration_camera_to_world is not None:
        print_calibration_triplet(
            phase=f"attempt_{attempt_idx}_after_grasp_kuavo_analytic_ik",
            camera_to_world=calibration_camera_to_world,
            target_pose_world=primitive["tcp_grasp"],
            tcp_pose_world=tcp_pose_matrix(robot, ee_body_id),
        )
    if not ok:
        return {"success": False, "reason": f"GRASP Kuavo analytic IK failed: {reason}", "segments": segments, "updates": updates, "lift_segment": None}

    tracker.enter("GRASP", attempt_index=attempt_idx, stage="close_gripper", backend="kuavo_analytic_ik")
    grasp_hold_joint_target = robot.data.joint_pos.clone()
    close_segment = run_slow_gripper_close(
        sim,
        scene,
        robot,
        gripper,
        tracker,
        grasp_hold_joint_target,
        locked_root_pose_w,
        start_width_m=open_width,
        steps=int(args_cli.grasp_steps),
    )
    segments.append({"name": "close_gripper", **close_segment})
    contact_ok, contact_meta, contact_reason = evaluate_gripper_contact_before_lift(close_segment, candidate)
    updates["gripper_contact_check"] = contact_meta
    if not contact_ok:
        updates["segments"] = segments
        return {
            "success": False,
            "reason": contact_reason,
            "segments": segments,
            "updates": updates,
            "lift_segment": None,
        }

    ok, lift_segment, reason = solve_and_run_stage(
        "lift_kuavo_analytic_ik",
        "LIFT",
        primitive["wrist_lift"],
        primitive["tcp_lift"],
        int(args_cli.lift_steps),
        float(args_cli.lift_error_threshold_m),
        0.0,
    )
    segments.append(lift_segment)
    updates["tcp_pose_after_lift_world"] = tcp_pose_matrix(robot, ee_body_id).tolist()

    tracker.enter("VERIFY_HOLD", attempt_index=attempt_idx, backend="kuavo_analytic_ik")
    lift_hold_joint_target = robot.data.joint_pos.clone()
    for _ in range(int(args_cli.hold_steps)):
        robot.set_joint_position_target(lift_hold_joint_target)
        gripper.close()
        stabilize_robot_base(robot, locked_root_pose_w)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim.get_physics_dt())
        record_video_frame(scene)
        gui_playback_tick(sim.get_physics_dt())
        tracker.tick()

    updates["segments"] = segments
    return {
        "success": bool(ok),
        "reason": "" if ok else f"LIFT Kuavo analytic IK failed: {reason}",
        "segments": segments,
        "updates": updates,
        "lift_segment": lift_segment,
    }


def execute_curobo_right_arm_attempt(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    gripper: AttachedParallelGripper,
    curobo_planner: KuavoRightArmCuroboPlanner | None,
    analytic_client: KuavoAnalyticIkClient | None,
    pose_ik_controller: DifferentialIKController,
    position_ik_controller: DifferentialIKController,
    entity_cfg: SceneEntityCfg,
    ee_body_id: int,
    ee_jacobi_idx: int,
    poses: dict[str, np.ndarray],
    candidate: SelectedGrasp,
    tracker: Task319StateTracker,
    attempt_idx: int,
    locked_root_pose_w: torch.Tensor,
    calibration_camera_to_world: np.ndarray | None = None,
) -> dict[str, Any]:
    primitive = position_only_grasp_poses(poses, robot, ee_body_id, candidate)
    if curobo_planner is None:
        return {
            "success": False,
            "reason": "cuRobo right-arm planner is unavailable.",
            "segments": [],
            "updates": {"motion_backend": "curobo_right_arm", "curobo_available": False},
            "lift_segment": None,
        }

    segments: list[dict[str, Any]] = []
    updates: dict[str, Any] = {
        "motion_backend": "curobo_right_arm",
        "curobo_available": True,
        "ik_command_type": "curobo_motion_gen_right_arm_joint_trajectory",
        "tcp_position_control": True,
        "orientation_constraint_mode": "curobo_position_only_tcp_no_axis_alignment" if bool(args_cli.curobo_position_only_tcp) else "current_tcp_orientation_soft_threshold",
        "axis_alignment_required": False,
        "curobo_right_arm_only": True,
        "curobo_joint_names": list(curobo_planner.joint_names),
        "curobo_max_attempts": int(args_cli.curobo_max_attempts),
        "curobo_enable_graph": bool(args_cli.curobo_enable_graph),
        "curobo_plan_table_obstacles": bool(args_cli.curobo_plan_table_obstacles),
        "curobo_joint_limit_clip_rad": float(args_cli.curobo_joint_limit_clip_rad),
        "curobo_rotation_threshold_rad": float(args_cli.curobo_rotation_threshold_rad),
        "curobo_final_grasp_rotation_threshold_rad": float(args_cli.curobo_final_grasp_rotation_threshold_rad),
        "curobo_final_settle_steps": int(args_cli.curobo_final_settle_steps),
        "curobo_tcp_refine": bool(args_cli.curobo_tcp_refine),
        "curobo_use_kuavo_analytic_seed": bool(args_cli.curobo_use_kuavo_analytic_seed),
        "curobo_position_only_tcp": bool(args_cli.curobo_position_only_tcp),
        "curobo_prefer_kuavo_seed_for_position_only": bool(args_cli.curobo_prefer_kuavo_seed_for_position_only),
        "rgbd_top_grasp_directionless_envelope": bool(args_cli.rgbd_top_grasp_directionless_envelope),
        "rgbd_top_grasp_envelope_jaw_axis": [float(v) for v in args_cli.rgbd_top_grasp_envelope_jaw_axis],
        "rgbd_top_grasp_enforce_orientation": bool(args_cli.rgbd_top_grasp_enforce_orientation),
        "kuavo_analytic_seed_available": bool(analytic_client is not None),
        "safe_pregrasp": {
            "enabled": bool(args_cli.safe_pregrasp_start),
            "mode": "curobo_collision_aware_high_standoff_vertical_descent",
            "tcp_pregrasp_world": primitive["tcp_pregrasp"].tolist(),
            "pregrasp_clearance_m": primitive["pregrasp_clearance_m"],
            "minimum_table_clearance_m": float(args_cli.arm_motion_min_table_clearance_m),
        },
        "tcp_pregrasp_pose_world": primitive["tcp_pregrasp"].tolist(),
        "wrist_pregrasp_pose_world": primitive["wrist_pregrasp"].tolist(),
        "tcp_grasp_pose_world": primitive["tcp_grasp"].tolist(),
        "wrist_grasp_pose_world": primitive["wrist_grasp"].tolist(),
        "tcp_lift_pose_world": primitive["tcp_lift"].tolist(),
        "wrist_lift_pose_world": primitive["wrist_lift"].tolist(),
        "position_only_wrist_rotation_world": primitive["wrist_rotation_world"],
        "wrist_orientation_mode": primitive["wrist_orientation_mode"],
        "commanded_position_source": primitive["commanded_position_source"],
        "pregrasp_position_source": primitive["pregrasp_position_source"],
        "vertical_descent_only": bool(primitive["vertical_descent_only"]),
        "top_grasp": primitive["top_grasp"],
        "gripper_alignment_before_curobo": gripper_alignment_diagnostics(robot, ee_body_id, primitive["tcp_grasp"]),
    }
    open_width = gripper_fully_open_width(gripper)
    try:
        curobo_entity_cfg = right_arm_curobo_entity_cfg(robot, entity_cfg)
    except Exception as exc:
        return {
            "success": False,
            "reason": f"cuRobo right-arm joint adapter failed: {exc}",
            "segments": [],
            "updates": {**updates, "curobo_joint_adapter_error": repr(exc)},
            "lift_segment": None,
        }
    updates["curobo_joint_adapter"] = {
        "joint_names": list(curobo_entity_cfg.joint_names),
        "joint_ids": [int(i) for i in curobo_entity_cfg.joint_ids],
        "source": "explicit RIGHT_ARM_JOINT_NAMES, not regex order",
    }
    right_base_body_id: int | None = None
    right_link7_body_id: int | None = None
    if analytic_client is not None and bool(args_cli.curobo_use_kuavo_analytic_seed):
        try:
            base_entity_cfg = SceneEntityCfg("robot", body_names=[RIGHT_ARM_BASE_BODY])
            base_entity_cfg.resolve(scene)
            link7_entity_cfg = SceneEntityCfg("robot", body_names=[RIGHT_ARM_LINK7_BODY])
            link7_entity_cfg.resolve(scene)
            right_base_body_id = int(base_entity_cfg.body_ids[0])
            right_link7_body_id = int(link7_entity_cfg.body_ids[0])
            updates["kuavo_analytic_seed_frames"] = {
                "right_arm_base_body": RIGHT_ARM_BASE_BODY,
                "right_arm_link7_body": RIGHT_ARM_LINK7_BODY,
                "right_base_body_id": right_base_body_id,
                "right_link7_body_id": right_link7_body_id,
            }
        except Exception as exc:
            updates["kuavo_analytic_seed_frame_error"] = repr(exc)
            right_base_body_id = None
            right_link7_body_id = None

    def refresh_curobo_world() -> None:
        if not bool(args_cli.curobo_plan_table_obstacles):
            return
        try:
            curobo_planner.update_world(curobo_world_for_current_robot(robot))
        except Exception as exc:
            updates["curobo_world_update_error"] = repr(exc)

    def kuavo_analytic_seed_for_tcp_stage(name: str, tcp_pose_w: np.ndarray, wrist_pose_w: np.ndarray | None, start_q: np.ndarray) -> dict[str, Any]:
        seed_meta: dict[str, Any] = {
            "enabled": bool(args_cli.curobo_use_kuavo_analytic_seed),
            "available": bool(analytic_client is not None and right_base_body_id is not None and right_link7_body_id is not None),
            "selected": False,
            "roll_attempts": [],
            "reason": "",
        }
        if not bool(args_cli.curobo_use_kuavo_analytic_seed):
            seed_meta["reason"] = "disabled_by_cli"
            return seed_meta
        if analytic_client is None or right_base_body_id is None or right_link7_body_id is None:
            seed_meta["reason"] = "official_kuavo_analytic_ik_unavailable"
            return seed_meta
        nominal_wrist_pose_w = np.asarray(wrist_pose_w, dtype=np.float32) if wrist_pose_w is not None else tcp_pose_to_wrist_pose(tcp_pose_w, tcp_pose_matrix(robot, ee_body_id)[:3, :3])
        executable: list[tuple[tuple[float, float, int], dict[str, Any]]] = []
        for sample_idx, sample in enumerate(kuavo_analytic_wrist_pose_samples(tcp_pose_w, nominal_wrist_pose_w)):
            sample_wrist_pose_w = np.asarray(sample["wrist_pose"], dtype=np.float32)
            response = analytic_client.solve_right(
                label=f"attempt_{attempt_idx}_{name}_curobo_seed_roll_{sample_idx}",
                robot=robot,
                right_base_body_id=int(right_base_body_id),
                right_link7_body_id=int(right_link7_body_id),
                right_ee_body_id=ee_body_id,
                right_ee_pose_w=sample_wrist_pose_w,
            )
            response["roll_sample_index"] = int(sample_idx)
            response["roll_sample_rad"] = float(sample["roll_rad"])
            response["approach_label"] = str(sample["approach_label"])
            response["approach_tcp_offset_axis_world"] = sample["approach_tcp_offset_axis_world"]
            response["sample_wrist_pose_world"] = sample_wrist_pose_w.astype(float).tolist()
            seed_meta["roll_attempts"].append(response)
            q = np.asarray(response.get("q") or [], dtype=np.float32)
            if bool(response.get("success")) and q.shape == (7,):
                fk_err = float(response.get("fk_position_error_m", 0.0) or 0.0)
                joint_dist = float(np.linalg.norm(q - np.asarray(start_q, dtype=np.float32).reshape(7)))
                executable.append(((fk_err, joint_dist, int(sample_idx)), {**response, "q": q.astype(float).tolist()}))
        if not executable:
            seed_meta["reason"] = "no_successful_official_kuavo_analytic_ik_roll_sample"
            return seed_meta
        executable.sort(key=lambda item: item[0])
        selected = executable[0][1]
        seed_meta.update(
            {
                "selected": True,
                "selected_response": selected,
                "selected_q": selected["q"],
                "executable_candidate_count": len(executable),
                "selection_key": {
                    "fk_position_error_m": float(executable[0][0][0]),
                    "joint_distance_from_start_l2_rad": float(executable[0][0][1]),
                    "roll_sample_index": int(executable[0][0][2]),
                },
                "reason": "selected_official_kuavo_analytic_ik_seed",
            }
        )
        return seed_meta

    def plan_and_run_stage(
        name: str,
        state_name: str,
        tcp_pose_w: np.ndarray,
        steps: int,
        threshold_m: float,
        width: float,
        *,
        wrist_pose_w: np.ndarray | None = None,
        allow_invalid_start_retry: bool = False,
        allow_local_ik_fallback: bool = False,
        rotation_threshold_rad: float | None = None,
    ) -> tuple[bool, dict[str, Any], str]:
        tracker.enter(state_name, attempt_index=attempt_idx, stage=name, backend="curobo_right_arm")
        refresh_curobo_world()
        start_q = robot.data.joint_pos[0, curobo_entity_cfg.joint_ids].detach().cpu().numpy().astype(np.float32)
        target_tcp_b = world_pose_to_robot_base_pose(robot, tcp_pose_w)
        stage_diagnostics = curobo_stage_frame_diagnostics(robot, ee_body_id, tcp_pose_w, start_q, curobo_entity_cfg.joint_ids)
        seed_meta: dict[str, Any] = {
            "enabled": bool(args_cli.curobo_use_kuavo_analytic_seed),
            "available": bool(analytic_client is not None and right_base_body_id is not None and right_link7_body_id is not None),
            "selected": False,
            "roll_attempts": [],
            "reason": "not_attempted",
        }
        planning_mode = "position_only_pose_goal" if bool(args_cli.curobo_position_only_tcp) else "pose_goal"
        fallback_pose_plan_reason = ""
        position_only_plan_reason = ""
        if bool(args_cli.curobo_position_only_tcp):
            plan = SimpleNamespace(success=False, reason="kuavo_seed_preferred_before_position_only", metadata={}, joint_positions=np.zeros((0, len(start_q)), dtype=np.float32))
            if bool(args_cli.curobo_prefer_kuavo_seed_for_position_only):
                seed_meta = kuavo_analytic_seed_for_tcp_stage(name, tcp_pose_w, wrist_pose_w, start_q)
                if bool(seed_meta.get("selected")):
                    planning_mode = "kuavo_analytic_seed_joint_space_before_position_only"
                    plan = curobo_planner.plan_to_joint_positions(
                        start_q,
                        np.asarray(seed_meta.get("selected_q"), dtype=np.float32),
                        max_attempts=int(args_cli.curobo_max_attempts),
                        enable_graph=bool(args_cli.curobo_enable_graph),
                        timeout_s=float(args_cli.curobo_timeout_s),
                    )
                    if not bool(plan.success):
                        fallback_pose_plan_reason = plan.reason or "cuRobo joint-space plan to preferred official Kuavo IK seed failed."
                else:
                    fallback_pose_plan_reason = seed_meta.get("reason", "")
            if not bool(plan.success):
                planning_mode = "position_only_pose_goal_after_seed_failed" if bool(args_cli.curobo_prefer_kuavo_seed_for_position_only) else "position_only_pose_goal"
                plan = curobo_planner.plan_to_pose(
                    start_q,
                    target_tcp_b,
                    max_attempts=int(args_cli.curobo_max_attempts),
                    enable_graph=bool(args_cli.curobo_enable_graph),
                    timeout_s=float(args_cli.curobo_timeout_s),
                    position_only=True,
                    rotation_threshold_rad=rotation_threshold_rad,
                )
            if not bool(plan.success):
                position_only_plan_reason = plan.reason or "cuRobo position-only TCP planning failed."
                if not bool(args_cli.curobo_prefer_kuavo_seed_for_position_only):
                    seed_meta = kuavo_analytic_seed_for_tcp_stage(name, tcp_pose_w, wrist_pose_w, start_q)
                if bool(seed_meta.get("selected")):
                    planning_mode = "kuavo_analytic_seed_joint_space_after_position_only_failed"
                    plan = curobo_planner.plan_to_joint_positions(
                        start_q,
                        np.asarray(seed_meta.get("selected_q"), dtype=np.float32),
                        max_attempts=int(args_cli.curobo_max_attempts),
                        enable_graph=bool(args_cli.curobo_enable_graph),
                        timeout_s=float(args_cli.curobo_timeout_s),
                    )
                    if not bool(plan.success):
                        fallback_pose_plan_reason = plan.reason or "cuRobo joint-space plan to official Kuavo IK seed failed."
                else:
                    fallback_pose_plan_reason = seed_meta.get("reason", "")
        else:
            seed_meta = kuavo_analytic_seed_for_tcp_stage(name, tcp_pose_w, wrist_pose_w, start_q)
            if bool(seed_meta.get("selected")):
                planning_mode = "kuavo_analytic_seed_joint_space"
                plan = curobo_planner.plan_to_joint_positions(
                    start_q,
                    np.asarray(seed_meta.get("selected_q"), dtype=np.float32),
                    max_attempts=int(args_cli.curobo_max_attempts),
                    enable_graph=bool(args_cli.curobo_enable_graph),
                    timeout_s=float(args_cli.curobo_timeout_s),
                )
                if not bool(plan.success):
                    fallback_pose_plan_reason = plan.reason or "cuRobo joint-space plan to official Kuavo IK seed failed."
                    planning_mode = "pose_goal_after_kuavo_seed_plan_failed"
                    plan = curobo_planner.plan_to_pose(
                        start_q,
                        target_tcp_b,
                        max_attempts=int(args_cli.curobo_max_attempts),
                        enable_graph=bool(args_cli.curobo_enable_graph),
                        timeout_s=float(args_cli.curobo_timeout_s),
                        position_only=False,
                        rotation_threshold_rad=rotation_threshold_rad,
                    )
            else:
                plan = curobo_planner.plan_to_pose(
                    start_q,
                    target_tcp_b,
                    max_attempts=int(args_cli.curobo_max_attempts),
                    enable_graph=bool(args_cli.curobo_enable_graph),
                    timeout_s=float(args_cli.curobo_timeout_s),
                    position_only=False,
                    rotation_threshold_rad=rotation_threshold_rad,
                )
        initial_plan_reason = str(plan.reason)
        initial_plan_metadata = dict(plan.metadata)
        retry_plan = None
        invalid_start_retry_reason = ""
        if "INVALID_START_STATE_WORLD_COLLISION" in initial_plan_reason:
            invalid_start_retry_reason = "INVALID_START_STATE_WORLD_COLLISION"
        elif "INVALID_START_STATE_JOINT_LIMITS" in initial_plan_reason:
            invalid_start_retry_reason = "INVALID_START_STATE_JOINT_LIMITS"
        if (
            (not bool(plan.success))
            and allow_invalid_start_retry
            and invalid_start_retry_reason
        ):
            retry_plan = curobo_planner.plan_to_pose(
                start_q,
                target_tcp_b,
                max_attempts=max(int(args_cli.curobo_max_attempts), 4),
                enable_graph=bool(args_cli.curobo_enable_graph),
                timeout_s=float(args_cli.curobo_timeout_s),
                check_start_validity=False,
                position_only=bool(args_cli.curobo_position_only_tcp),
                rotation_threshold_rad=rotation_threshold_rad,
            )
            if bool(retry_plan.success):
                plan = retry_plan
        segment: dict[str, Any] = {
            "name": name,
            "curobo_success": bool(plan.success),
            "curobo_reason": plan.reason,
            "curobo_metadata": plan.metadata,
            "curobo_planning_mode": planning_mode,
            "curobo_position_only_plan_reason": position_only_plan_reason,
            "curobo_kuavo_analytic_seed": seed_meta,
            "curobo_kuavo_seed_joint_plan_fallback_reason": fallback_pose_plan_reason,
            "curobo_target_tcp_pose_base": target_tcp_b.astype(float).tolist(),
            "curobo_start_q": start_q.astype(float).tolist(),
            "curobo_stage_frame_diagnostics": stage_diagnostics,
            "target_tcp_world_m": np.asarray(tcp_pose_w, dtype=np.float32)[:3, 3].astype(float).tolist(),
            "requested_rotation_threshold_rad": None if rotation_threshold_rad is None else float(rotation_threshold_rad),
        }
        if retry_plan is not None:
            segment["curobo_invalid_start_retry"] = {
                "used": True,
                "initial_reason": initial_plan_reason,
                "initial_metadata": initial_plan_metadata,
                "invalid_start_type": invalid_start_retry_reason,
                "check_start_validity": False,
                "success": bool(retry_plan.success),
                "reason": retry_plan.reason,
                "metadata": retry_plan.metadata,
            }
        if not plan.success:
            if allow_invalid_start_retry and "INVALID_START_STATE_WORLD_COLLISION" in str(plan.reason):
                escape_wrist_pose_w = tcp_pose_to_wrist_pose(tcp_pose_w, tcp_pose_matrix(robot, ee_body_id)[:3, :3])
                escape_segment = run_ik_segment_until_converged(
                    sim,
                    scene,
                    robot,
                    position_ik_controller,
                    curobo_entity_cfg,
                    ee_body_id,
                    ee_jacobi_idx,
                    escape_wrist_pose_w,
                    max(1, int(steps)),
                    float(threshold_m),
                    gripper=gripper,
                    gripper_width=width,
                    locked_root_pose_w=locked_root_pose_w,
                    debug_target_pose_w=primitive["tcp_grasp"],
                    target_tcp_pose_w=tcp_pose_w,
                    control_tcp_position=True,
                )
                escape_error = float(escape_segment.get("final_tcp_pos_error_m", escape_segment.get("final_pos_error_m", float("inf"))))
                segment.update(
                    {
                        "fallback_motion_backend": "local_position_ik_escape_from_invalid_start",
                        "fallback_reason": "cuRobo reported INVALID_START_STATE_WORLD_COLLISION for the low starting pose.",
                        "fallback_segment": escape_segment,
                        "final_pos_error_m": escape_error,
                        "final_tcp_pos_error_m": escape_error,
                        "converged": bool(math.isfinite(escape_error) and escape_error <= float(threshold_m)),
                    }
                )
                if bool(segment["converged"]):
                    return True, segment, ""
                return False, segment, "cuRobo start-state collision and local IK escape failed."
            if allow_local_ik_fallback:
                fallback_wrist_pose_w = tcp_pose_to_wrist_pose(tcp_pose_w, tcp_pose_matrix(robot, ee_body_id)[:3, :3])
                fallback_segment = run_ik_segment_until_converged(
                    sim,
                    scene,
                    robot,
                    position_ik_controller,
                    curobo_entity_cfg,
                    ee_body_id,
                    ee_jacobi_idx,
                    fallback_wrist_pose_w,
                    max(1, int(steps)),
                    float(threshold_m),
                    gripper=gripper,
                    gripper_width=width,
                    locked_root_pose_w=locked_root_pose_w,
                    debug_target_pose_w=primitive["tcp_grasp"],
                    target_tcp_pose_w=tcp_pose_w,
                    control_tcp_position=True,
                )
                fallback_error = float(fallback_segment.get("final_tcp_pos_error_m", fallback_segment.get("final_pos_error_m", float("inf"))))
                segment.update(
                    {
                        "fallback_motion_backend": "local_position_ik_after_curobo_plan_fail",
                        "fallback_reason": f"cuRobo plan failed for short TCP stage: {plan.reason}",
                        "fallback_segment": fallback_segment,
                        "final_pos_error_m": fallback_error,
                        "final_tcp_pos_error_m": fallback_error,
                        "final_tcp_world_m": tcp_pose_matrix(robot, ee_body_id)[:3, 3].astype(float).tolist(),
                        "converged": bool(math.isfinite(fallback_error) and fallback_error <= float(threshold_m)),
                    }
                )
                if bool(segment["converged"]):
                    return True, segment, ""
                return False, segment, "cuRobo plan failed and local TCP fallback failed."
            segment["converged"] = False
            segment["final_pos_error_m"] = float("inf")
            return False, segment, plan.reason or "cuRobo planning failed."

        exec_segment = run_curobo_joint_trajectory(
            sim,
            scene,
            robot,
            curobo_entity_cfg,
            plan.joint_positions,
            min_steps=max(int(args_cli.curobo_command_min_steps), int(steps)),
            gripper=gripper,
            gripper_width=width,
            locked_root_pose_w=locked_root_pose_w,
            target_tcp_pose_w=tcp_pose_w,
        )
        segment.update(exec_segment)
        segment["curobo_planned_joint_positions_first"] = plan.joint_positions[0].astype(float).tolist() if len(plan.joint_positions) else []
        segment["curobo_planned_joint_positions_last"] = plan.joint_positions[-1].astype(float).tolist() if len(plan.joint_positions) else []

        final_error = float(segment.get("final_tcp_pos_error_m", float("inf")))
        refine_segment: dict[str, Any] | None = None
        if bool(args_cli.curobo_tcp_refine) and (not math.isfinite(final_error) or final_error > float(threshold_m)):
            refine_segment = run_ik_segment_until_converged(
                sim,
                scene,
                robot,
                position_ik_controller,
                curobo_entity_cfg,
                ee_body_id,
                ee_jacobi_idx,
                tcp_pose_to_wrist_pose(tcp_pose_w, tcp_pose_matrix(robot, ee_body_id)[:3, :3]),
                max(1, int(args_cli.curobo_tcp_refine_steps)),
                float(threshold_m),
                gripper=gripper,
                gripper_width=width,
                locked_root_pose_w=locked_root_pose_w,
                debug_target_pose_w=tcp_pose_w,
                target_tcp_pose_w=tcp_pose_w,
                control_tcp_position=True,
            )
            final_error = float(refine_segment.get("final_tcp_pos_error_m", refine_segment.get("final_pos_error_m", final_error)))
            segment.update(
                {
                    "curobo_tcp_refine_segment": refine_segment,
                    "curobo_tcp_refine_used": True,
                    "final_pos_error_m": final_error,
                    "final_tcp_pos_error_m": final_error,
                    "final_tcp_world_m": tcp_pose_matrix(robot, ee_body_id)[:3, 3].astype(float).tolist(),
                }
            )
        else:
            segment["curobo_tcp_refine_used"] = False
            segment["curobo_tcp_refine_segment"] = refine_segment
        segment["error_threshold_m"] = float(threshold_m)
        segment["converged"] = bool(math.isfinite(final_error) and final_error <= float(threshold_m))
        return bool(segment["converged"]), segment, "" if segment["converged"] else "cuRobo execution TCP error exceeded threshold."

    if args_cli.arm_motion_ready_pose == "carry" and int(args_cli.arm_motion_ready_steps) > 0:
        tracker.enter("SAFE_START", attempt_index=attempt_idx, stage="carry_ready_pose", backend="curobo_right_arm")
        ready_target = right_arm_joint_target_from_named_pose(robot, curobo_entity_cfg, RIGHT_ARM_CARRY_JOINT_POS)
        ready_segment = run_joint_segment(
            sim,
            scene,
            robot,
            curobo_entity_cfg,
            ready_target,
            int(args_cli.arm_motion_ready_steps),
            gripper=gripper,
            gripper_width=open_width,
            locked_root_pose_w=locked_root_pose_w,
        )
        segments.append({"name": "carry_ready_pose", **ready_segment})

    gripper.set_width(open_width)
    if bool(args_cli.safe_pregrasp_start):
        safe_waypoints = safe_pregrasp_waypoints(primitive, robot, ee_body_id)
        safe_meta = {
            "enabled": True,
            "mode": "curobo_collision_aware_lift_current_tcp_then_high_standoff",
            "table_surface_z_m": float(TABLE_SURFACE_Z),
            "safe_z_m": float(safe_waypoints["safe_z_m"]),
            "table_clearance_m": float(safe_waypoints["safe_z_m"] - TABLE_SURFACE_Z),
            "object_clearance_m": float(safe_waypoints["safe_z_m"] - primitive["tcp_grasp"][2, 3]),
            "tcp_safe_lift_current_world": safe_waypoints["tcp_safe_lift_current"].tolist(),
            "tcp_safe_standoff_world": safe_waypoints["tcp_safe_standoff"].tolist(),
        }
        updates["safe_pregrasp"].update(safe_meta)
        ok, safe_lift_segment, reason = plan_and_run_stage(
            "safe_lift_current_curobo",
            "SAFE_START",
            safe_waypoints["tcp_safe_lift_current"],
            int(args_cli.safe_pregrasp_steps),
            float(args_cli.safe_pregrasp_error_threshold_m),
            open_width,
            allow_invalid_start_retry=True,
        )
        segments.append(safe_lift_segment)
        if not ok:
            updates["segments"] = segments
            updates["safe_pregrasp"]["failed_stage"] = "lift_current_tcp"
            return {"success": False, "reason": f"SAFE_START cuRobo lift-current failed: {reason}", "segments": segments, "updates": updates, "lift_segment": None}

        ok, safe_standoff_segment, reason = plan_and_run_stage(
            "safe_standoff_curobo",
            "SAFE_START",
            safe_waypoints["tcp_safe_standoff"],
            int(args_cli.safe_pregrasp_steps),
            float(args_cli.safe_pregrasp_error_threshold_m),
            open_width,
        )
        segments.append(safe_standoff_segment)
        if not ok:
            updates["segments"] = segments
            updates["safe_pregrasp"]["failed_stage"] = "high_standoff"
            return {"success": False, "reason": f"SAFE_START cuRobo high-standoff failed: {reason}", "segments": segments, "updates": updates, "lift_segment": None}

        ok, pre_segment, reason = plan_and_run_stage(
            "pregrasp_curobo",
            "PRE_GRASP",
            primitive["tcp_pregrasp"],
            int(args_cli.trajectory_steps),
            float(args_cli.pregrasp_error_threshold_m),
            open_width,
            wrist_pose_w=primitive["wrist_pregrasp"],
            allow_invalid_start_retry=True,
        )
    else:
        updates["safe_pregrasp"].update({"enabled": False, "mode": "direct_current_to_pregrasp_curobo"})
        ok, pre_segment, reason = plan_and_run_stage(
            "pregrasp_curobo",
            "PRE_GRASP",
            primitive["tcp_pregrasp"],
            int(args_cli.trajectory_steps),
            float(args_cli.pregrasp_error_threshold_m),
            open_width,
            wrist_pose_w=primitive["wrist_pregrasp"],
            allow_invalid_start_retry=True,
        )
    segments.append(pre_segment)
    current_tcp_pose_w = tcp_pose_matrix(robot, ee_body_id)
    render_calibration_debug_markers(scene, primitive["tcp_grasp"], current_tcp_pose_w)
    updates["tcp_pose_after_pregrasp_world"] = current_tcp_pose_w.tolist()
    updates["gripper_alignment_after_pregrasp"] = gripper_alignment_diagnostics(robot, ee_body_id, primitive["tcp_grasp"])
    if calibration_camera_to_world is not None:
        print_calibration_triplet(
            phase=f"attempt_{attempt_idx}_after_pregrasp_curobo",
            camera_to_world=calibration_camera_to_world,
            target_pose_world=primitive["tcp_grasp"],
            tcp_pose_world=current_tcp_pose_w,
        )
    if not ok:
        updates["segments"] = segments
        return {"success": False, "reason": f"PRE_GRASP cuRobo failed: {reason}", "segments": segments, "updates": updates, "lift_segment": None}
    pregrasp_state = capture_dry_run_state(scene, robot)
    pregrasp_wrist_rot_w = current_body_pose_matrix(robot, ee_body_id)[:3, :3].copy()
    target_scene_key_for_motion = None
    if isinstance(candidate.metadata, dict):
        raw_scene_key = candidate.metadata.get("scene_key") or candidate.metadata.get("target_scene_key")
        if isinstance(raw_scene_key, str) and raw_scene_key in scene.keys():
            target_scene_key_for_motion = raw_scene_key
    target_root_before_grasp = scene_root_position_world(scene, target_scene_key_for_motion)
    approach_motion_guard: dict[str, Any] = {
        "enabled": bool(args_cli.grasp_abort_if_object_moves_during_approach),
        "guard_source": "sim_target_root_diagnostic",
        "scene_key": target_scene_key_for_motion,
        "threshold_m": float(args_cli.grasp_object_motion_abort_threshold_m),
        "root_before_grasp_world_m": target_root_before_grasp.astype(float).tolist() if target_root_before_grasp is not None else None,
        "moved": False,
        "motion_m": 0.0,
    }
    final_grasp_threshold_m = grasp_close_error_threshold_m(candidate)
    updates["final_grasp_close_error_threshold_m"] = float(final_grasp_threshold_m)
    updates["final_grasp_base_error_threshold_m"] = float(args_cli.grasp_error_threshold_m)
    updates["final_grasp_candidate_width_m"] = float(candidate.width)

    if bool(args_cli.curobo_grasp_local_tcp_descent):
        tracker.enter(
            "GRASP",
            attempt_index=attempt_idx,
            stage="tcp_position_servo_descent" if bool(args_cli.curobo_grasp_servo_descent) else "cartesian_linear_curobo_ik_descent",
            backend="isaaclab_position_ik" if bool(args_cli.curobo_grasp_servo_descent) else "curobo_right_arm",
        )
        primitive_grasp_tcp_pose_w = np.asarray(primitive["tcp_grasp"], dtype=np.float32).reshape(4, 4)
        descent_start_tcp_pose_w = tcp_pose_matrix(robot, ee_body_id).astype(np.float32)
        vertical_grasp_tcp_pose_w = descent_start_tcp_pose_w.copy()
        vertical_grasp_tcp_pose_w[:3, :3] = descent_start_tcp_pose_w[:3, :3]
        vertical_grasp_tcp_pose_w[2, 3] = float(primitive_grasp_tcp_pose_w[2, 3])
        descent_delta_w = vertical_grasp_tcp_pose_w[:3, 3] - descent_start_tcp_pose_w[:3, 3]
        descent_distance_m = float(abs(descent_delta_w[2]))
        spacing_m = max(0.002, float(args_cli.curobo_cartesian_descent_waypoint_spacing_m))
        min_waypoints = max(2, int(args_cli.curobo_cartesian_descent_min_waypoints))
        waypoint_count = max(min_waypoints, int(math.ceil(descent_distance_m / spacing_m)))
        waypoint_count = max(1, waypoint_count)
        descent_tcp_waypoints_w: list[np.ndarray] = []
        for waypoint_idx in range(1, waypoint_count + 1):
            alpha = float(waypoint_idx) / float(waypoint_count)
            pose_w = descent_start_tcp_pose_w.copy()
            pose_w[:3, :3] = descent_start_tcp_pose_w[:3, :3]
            pose_w[0, 3] = float(descent_start_tcp_pose_w[0, 3])
            pose_w[1, 3] = float(descent_start_tcp_pose_w[1, 3])
            pose_w[2, 3] = float((1.0 - alpha) * descent_start_tcp_pose_w[2, 3] + alpha * vertical_grasp_tcp_pose_w[2, 3])
            descent_tcp_waypoints_w.append(pose_w.astype(np.float32))
        descent_min_steps = max(int(args_cli.trajectory_steps), waypoint_count * max(1, int(args_cli.curobo_cartesian_descent_steps_per_waypoint)))
        proximity_stop_distance = float(args_cli.curobo_grasp_servo_stop_distance_m)
        if not math.isfinite(proximity_stop_distance) or proximity_stop_distance <= 0.0:
            proximity_stop_distance = float(args_cli.mind_sort_gripper_proximity_assist_max_distance_m)
        if bool(args_cli.curobo_grasp_servo_descent):
            grasp_segment = run_tcp_position_servo_descent(
                sim,
                scene,
                robot,
                position_ik_controller,
                curobo_entity_cfg,
                ee_body_id,
                ee_jacobi_idx,
                vertical_grasp_tcp_pose_w,
                primitive_grasp_tcp_pose_w[:3, 3],
                descent_min_steps,
                gripper=gripper,
                gripper_width=open_width,
                locked_root_pose_w=locked_root_pose_w,
                debug_target_pose_w=primitive["tcp_grasp"],
                proximity_stop_distance_m=proximity_stop_distance,
                max_joint_step=float(args_cli.curobo_cartesian_descent_max_joint_step),
            )
            local_error = float(grasp_segment.get("final_tcp_pos_error_m", grasp_segment.get("final_pos_error_m", float("inf"))))
            ik_chain_plan = SimpleNamespace(success=None, reason="skipped_for_position_only_servo_descent", metadata={})
            executable_descent_waypoints = 0
            executed_partial_chain = False
        else:
            descent_tcp_waypoints_b = [world_pose_to_robot_base_pose(robot, pose_w) for pose_w in descent_tcp_waypoints_w]
            refresh_curobo_world()
            start_q = robot.data.joint_pos[0, curobo_entity_cfg.joint_ids].detach().cpu().numpy().astype(np.float32)
            ik_chain_plan = curobo_planner.solve_ik_chain_for_poses_sequential(
                start_q,
                descent_tcp_waypoints_b,
                return_seeds=int(args_cli.curobo_cartesian_descent_return_seeds),
                max_waypoint_joint_l2_rad=float(args_cli.curobo_cartesian_descent_max_waypoint_joint_l2_rad),
                position_only=bool(args_cli.curobo_cartesian_descent_position_only),
            )
            executable_joint_positions = np.asarray(ik_chain_plan.joint_positions, dtype=np.float32).reshape(-1, len(curobo_entity_cfg.joint_ids))
            executable_descent_waypoints = int(executable_joint_positions.shape[0])
            executed_partial_chain = bool((not bool(ik_chain_plan.success)) and executable_descent_waypoints > 0)
            descent_min_steps = max(
                int(args_cli.trajectory_steps),
                max(1, executable_descent_waypoints) * max(1, int(args_cli.curobo_cartesian_descent_steps_per_waypoint)),
            )
            if bool(ik_chain_plan.success) or executed_partial_chain:
                grasp_segment = run_curobo_joint_trajectory(
                    sim,
                    scene,
                    robot,
                    curobo_entity_cfg,
                    executable_joint_positions,
                    min_steps=descent_min_steps,
                    gripper=gripper,
                    gripper_width=open_width,
                    locked_root_pose_w=locked_root_pose_w,
                    target_tcp_pose_w=vertical_grasp_tcp_pose_w,
                    max_joint_step_override=float(args_cli.curobo_cartesian_descent_max_joint_step),
                    cartesian_lock_xy_w=descent_start_tcp_pose_w[:2, 3],
                    max_cartesian_xy_error_m=float(args_cli.curobo_cartesian_descent_max_xy_error_m),
                    abort_on_cartesian_xy_error=True,
                    object_target_tcp_w=primitive_grasp_tcp_pose_w[:3, 3],
                    proximity_stop_distance_m=proximity_stop_distance,
                )
                local_error = float(grasp_segment.get("final_tcp_pos_error_m", grasp_segment.get("final_pos_error_m", float("inf"))))
            else:
                grasp_segment = {
                    "final_pos_error_m": float("inf"),
                    "final_tcp_pos_error_m": float("inf"),
                    "position_error_reference": "tcp",
                    "executed_steps": 0,
                    "aborted": True,
                    "abort_reason": ik_chain_plan.reason,
                }
                local_error = float("inf")
        vertical_xy = vertical_grasp_tcp_pose_w[:2, 3].astype(np.float32)
        object_xy = primitive_grasp_tcp_pose_w[:2, 3].astype(np.float32)
        descent_reached_proximity = bool(grasp_segment.get("reached_proximity_stop", False))
        precise_reached = bool(math.isfinite(local_error) and local_error <= float(final_grasp_threshold_m))
        if bool(args_cli.curobo_grasp_servo_descent):
            descent_motion_backend = "isaaclab_position_only_tcp_servo_after_curobo_pregrasp"
            descent_orientation_mode = "position_only_tcp_servo_current_wrist_orientation"
            descent_orientation_policy = "Experimental servo descent ignores exact object axes; it commands TCP position only and keeps the current wrist orientation from hover."
        elif bool(args_cli.curobo_cartesian_descent_position_only):
            descent_motion_backend = "curobo_position_only_cartesian_ik_chain_after_curobo_pregrasp"
            descent_orientation_mode = "position_only_curobo_cartesian_straight_line_hold_hover_preference"
            descent_orientation_policy = "Final descent solves a cuRobo position-only batch IK chain in the gripper TCP frame: waypoint X/Y are locked to the actual hover gripper TCP, only world Z changes, and solution scoring preserves the hover wrist posture without requiring exact axis alignment."
        else:
            descent_motion_backend = "curobo_dense_cartesian_ik_chain_after_curobo_pregrasp"
            descent_orientation_mode = "full_pose_curobo_cartesian_straight_line_hold_hover_gripper_tcp_orientation"
            descent_orientation_policy = "Final descent solves a cuRobo full-pose batch IK chain in the gripper TCP frame: waypoint X/Y/quaternion are locked to the actual hover gripper TCP, and only world Z changes."
        grasp_segment.update(
            {
                "name": "grasp_cartesian_linear_curobo_ik_descent",
                "motion_backend": descent_motion_backend,
                "curobo_final_grasp_planning_skipped": True,
                "curobo_final_grasp_planning_skip_reason": "The low grasp segment is constrained to a vertical descent; no low table-near free-space cuRobo replanning is allowed.",
                "curobo_cartesian_ik_chain_success": None if bool(args_cli.curobo_grasp_servo_descent) else bool(ik_chain_plan.success),
                "curobo_cartesian_ik_chain_reason": ik_chain_plan.reason,
                "curobo_cartesian_ik_chain_metadata": ik_chain_plan.metadata,
                "curobo_cartesian_descent_position_only": bool(args_cli.curobo_cartesian_descent_position_only),
                "curobo_cartesian_ik_chain_partial_execution": bool(executed_partial_chain),
                "curobo_cartesian_executable_waypoints": int(executable_descent_waypoints),
                "servo_descent_enabled": bool(args_cli.curobo_grasp_servo_descent),
                "servo_descent_policy": "position-only closed-loop TCP servo; stop when target reached or TCP enters gripper proximity range",
                "servo_proximity_stop_distance_m": float(proximity_stop_distance),
                "servo_reached_proximity_stop": descent_reached_proximity if bool(args_cli.curobo_grasp_servo_descent) else False,
                "descent_reached_proximity_stop": descent_reached_proximity,
                "servo_reached_precise_vertical_target": precise_reached,
                "extra_convergence_disabled": True,
                "orientation_constraint_mode": descent_orientation_mode,
                "axis_alignment_required": False,
                "orientation_policy": descent_orientation_policy,
                "fixed_tcp_rotation_world": descent_start_tcp_pose_w[:3, :3].astype(float).tolist(),
                "descent_start_tcp_world_m": descent_start_tcp_pose_w[:3, 3].astype(float).tolist(),
                "object_grasp_target_tcp_world_m": primitive_grasp_tcp_pose_w[:3, 3].astype(float).tolist(),
                "target_tcp_world_m": vertical_grasp_tcp_pose_w[:3, 3].astype(float).tolist(),
                "vertical_descent_only": True,
                "cartesian_straight_line_descent": True,
                "locked_xy_source": "actual_tcp_after_pregrasp",
                "locked_xy_minus_object_target_xy_m": (vertical_xy - object_xy).astype(float).tolist(),
                "locked_xy_object_target_distance_m": float(np.linalg.norm(vertical_xy - object_xy)),
                "descent_delta_world_m": descent_delta_w.astype(float).tolist(),
                "cartesian_descent_waypoint_count": int(len(descent_tcp_waypoints_w)),
                "cartesian_descent_waypoint_spacing_m": float(spacing_m),
                "cartesian_descent_min_steps": int(descent_min_steps),
                "cartesian_descent_world_waypoints_m": [pose_w[:3, 3].astype(float).tolist() for pose_w in descent_tcp_waypoints_w],
                "no_penetration_contact_policy": {
                    "trash_rest_offset_m": float(args_cli.trash_rest_offset_m),
                    "gripper_rest_offset_m": float(args_cli.gripper_rest_offset_m),
                    "top_grasp_depth_m": float(args_cli.top_grasp_depth_m),
                },
                "final_tcp_world_m": tcp_pose_matrix(robot, ee_body_id)[:3, 3].astype(float).tolist(),
                "final_tcp_pos_error_m": local_error,
                "final_pos_error_m": local_error,
                "base_error_threshold_m": float(args_cli.grasp_error_threshold_m),
                "error_threshold_m": float(final_grasp_threshold_m),
                "candidate_width_m": float(candidate.width),
                "converged": bool((precise_reached or descent_reached_proximity) and not bool(grasp_segment.get("aborted", False))),
            }
        )
        ok = bool(grasp_segment["converged"])
        reason = "" if ok else (grasp_segment.get("abort_reason") or "Cartesian final descent did not reach the target/proximity threshold; stopping before gripper close.")
    else:
        ok, grasp_segment, reason = plan_and_run_stage(
            "grasp_curobo",
            "GRASP",
            primitive["tcp_grasp"],
            int(args_cli.trajectory_steps),
            float(final_grasp_threshold_m),
            open_width,
            wrist_pose_w=primitive["wrist_grasp"],
            allow_local_ik_fallback=False,
            rotation_threshold_rad=float(args_cli.curobo_final_grasp_rotation_threshold_rad),
        )
    segments.append(grasp_segment)
    current_tcp_pose_w = tcp_pose_matrix(robot, ee_body_id)
    render_calibration_debug_markers(scene, primitive["tcp_grasp"], current_tcp_pose_w)
    updates["tcp_pose_after_grasp_world"] = current_tcp_pose_w.tolist()
    updates["gripper_alignment_after_grasp"] = gripper_alignment_diagnostics(robot, ee_body_id, primitive["tcp_grasp"])
    if calibration_camera_to_world is not None:
        print_calibration_triplet(
            phase=f"attempt_{attempt_idx}_after_grasp_curobo",
            camera_to_world=calibration_camera_to_world,
            target_pose_world=primitive["tcp_grasp"],
            tcp_pose_world=current_tcp_pose_w,
            gripper_pose_world=body_pose_matrix_by_name(robot, "gripper_base"),
        )
    target_root_after_grasp = scene_root_position_world(scene, target_scene_key_for_motion)
    if target_root_before_grasp is not None and target_root_after_grasp is not None:
        motion_vec = np.asarray(target_root_after_grasp, dtype=np.float32) - np.asarray(target_root_before_grasp, dtype=np.float32)
        motion_m = float(np.linalg.norm(motion_vec))
        approach_motion_guard.update(
            {
                "root_after_grasp_world_m": target_root_after_grasp.astype(float).tolist(),
                "motion_vector_m": motion_vec.astype(float).tolist(),
                "motion_m": motion_m,
                "moved": bool(motion_m > float(args_cli.grasp_object_motion_abort_threshold_m)),
            }
        )
    elif target_scene_key_for_motion is None:
        approach_motion_guard["reason"] = "no_scene_key_for_target"
    else:
        approach_motion_guard["reason"] = "target_root_unavailable"
    grasp_segment["object_motion_guard"] = approach_motion_guard
    updates["approach_object_motion_guard"] = approach_motion_guard
    if bool(args_cli.grasp_abort_if_object_moves_during_approach) and bool(approach_motion_guard.get("moved")):
        ok = False
        reason = (
            "object_moved_during_approach; target pose is stale "
            f"(motion={float(approach_motion_guard.get('motion_m', 0.0)):.4f}m)"
        )
    if not ok:
        if bool(args_cli.curobo_grasp_fixed_wrist_pose_fallback) and not bool(args_cli.curobo_grasp_local_tcp_descent):
            restore_dry_run_state(scene, robot, pregrasp_state)
            fixed_tcp_pose_w = np.asarray(primitive["tcp_grasp"], dtype=np.float32).copy()
            fixed_tcp_pose_w[:3, :3] = pregrasp_wrist_rot_w
            fixed_wrist_pose_w = tcp_pose_to_wrist_pose(fixed_tcp_pose_w, pregrasp_wrist_rot_w)
            tracker.enter("GRASP", attempt_index=attempt_idx, stage="fixed_wrist_pose_ik_descent", backend="curobo_right_arm")
            fixed_segment = run_ik_segment_until_converged(
                sim,
                scene,
                robot,
                pose_ik_controller,
                entity_cfg,
                ee_body_id,
                ee_jacobi_idx,
                fixed_wrist_pose_w,
                int(args_cli.trajectory_steps),
                float(final_grasp_threshold_m),
                gripper=gripper,
                gripper_width=open_width,
                locked_root_pose_w=locked_root_pose_w,
                debug_target_pose_w=primitive["tcp_grasp"],
                target_tcp_pose_w=primitive["tcp_grasp"],
                control_tcp_position=False,
            )
            final_tcp_error = float(fixed_segment.get("final_tcp_pos_error_m", float("inf")))
            fixed_segment.update(
                {
                    "name": "grasp_fixed_wrist_pose_ik",
                    "fallback_motion_backend": "fixed_wrist_pose_ik_after_curobo_grasp_fail",
                    "fallback_reason": f"cuRobo final grasp descent failed: {reason}",
                    "fixed_wrist_rotation_world": pregrasp_wrist_rot_w.astype(float).tolist(),
                    "target_tcp_world_m": np.asarray(primitive["tcp_grasp"], dtype=np.float32)[:3, 3].astype(float).tolist(),
                    "final_tcp_world_m": tcp_pose_matrix(robot, ee_body_id)[:3, 3].astype(float).tolist(),
                    "final_tcp_pos_error_m": final_tcp_error,
                    "final_pos_error_m": final_tcp_error,
                    "position_error_reference": "tcp_after_fixed_wrist_pose_ik",
                    "base_error_threshold_m": float(args_cli.grasp_error_threshold_m),
                    "error_threshold_m": float(final_grasp_threshold_m),
                    "candidate_width_m": float(candidate.width),
                    "converged": bool(math.isfinite(final_tcp_error) and final_tcp_error <= float(final_grasp_threshold_m)),
                }
            )
            segments.append(fixed_segment)
            ok = bool(fixed_segment["converged"])
            reason = "" if ok else "fixed wrist pose IK descent TCP error exceeded threshold."
            current_tcp_pose_w = tcp_pose_matrix(robot, ee_body_id)
            render_calibration_debug_markers(scene, primitive["tcp_grasp"], current_tcp_pose_w)
            updates["tcp_pose_after_grasp_world"] = current_tcp_pose_w.tolist()
        if not ok:
            updates["segments"] = segments
            return {"success": False, "reason": f"GRASP cuRobo failed: {reason}", "segments": segments, "updates": updates, "lift_segment": None}

    tracker.enter("GRASP", attempt_index=attempt_idx, stage="close_gripper", backend="curobo_right_arm")
    grasp_hold_joint_target = robot.data.joint_pos.clone()
    close_segment = run_slow_gripper_close(
        sim,
        scene,
        robot,
        gripper,
        tracker,
        grasp_hold_joint_target,
        locked_root_pose_w,
        start_width_m=open_width,
        steps=int(args_cli.grasp_steps),
    )
    segments.append({"name": "close_gripper", **close_segment})
    updates["gripper_alignment_after_close"] = gripper_alignment_diagnostics(robot, ee_body_id, primitive["tcp_grasp"])
    contact_ok, contact_meta, contact_reason = evaluate_gripper_contact_before_lift(close_segment, candidate)
    updates["gripper_contact_check"] = contact_meta
    if not contact_ok:
        updates["segments"] = segments
        return {
            "success": False,
            "reason": contact_reason,
            "segments": segments,
            "updates": updates,
            "lift_segment": None,
        }

    post_close_hold_joint_target = robot.data.joint_pos.clone()
    if int(args_cli.gripper_post_close_hold_steps) > 0:
        tracker.enter("GRASP", attempt_index=attempt_idx, stage="post_close_contact_settle", backend="curobo_right_arm")
        settle_segment = run_closed_gripper_settle(
            sim,
            scene,
            robot,
            gripper,
            tracker,
            post_close_hold_joint_target,
            locked_root_pose_w,
            steps=int(args_cli.gripper_post_close_hold_steps),
        )
        segments.append({"name": "post_close_contact_settle", **settle_segment})

    if bool(args_cli.curobo_lift_local_tcp_ascent):
        tracker.enter("LIFT", attempt_index=attempt_idx, stage="cartesian_linear_curobo_ik_ascent", backend="curobo_right_arm")
        lift_start_tcp_pose_w = tcp_pose_matrix(robot, ee_body_id).astype(np.float32)
        vertical_lift_tcp_pose_w = lift_start_tcp_pose_w.copy()
        vertical_lift_tcp_pose_w[:3, :3] = lift_start_tcp_pose_w[:3, :3]
        vertical_lift_tcp_pose_w[2, 3] = max(
            float(lift_start_tcp_pose_w[2, 3] + float(args_cli.lift_height)),
            float(TABLE_SURFACE_Z + args_cli.safe_pregrasp_table_clearance_m),
        )
        lift_delta_w = vertical_lift_tcp_pose_w[:3, 3] - lift_start_tcp_pose_w[:3, 3]
        lift_distance_m = float(abs(lift_delta_w[2]))
        lift_spacing_m = max(0.002, float(args_cli.curobo_cartesian_descent_waypoint_spacing_m))
        lift_waypoint_count = max(
            max(2, int(args_cli.curobo_cartesian_descent_min_waypoints)),
            int(math.ceil(lift_distance_m / lift_spacing_m)),
        )
        lift_tcp_waypoints_w: list[np.ndarray] = []
        for waypoint_idx in range(1, lift_waypoint_count + 1):
            alpha = float(waypoint_idx) / float(lift_waypoint_count)
            pose_w = lift_start_tcp_pose_w.copy()
            pose_w[:3, :3] = lift_start_tcp_pose_w[:3, :3]
            pose_w[0, 3] = float(lift_start_tcp_pose_w[0, 3])
            pose_w[1, 3] = float(lift_start_tcp_pose_w[1, 3])
            pose_w[2, 3] = float((1.0 - alpha) * lift_start_tcp_pose_w[2, 3] + alpha * vertical_lift_tcp_pose_w[2, 3])
            lift_tcp_waypoints_w.append(pose_w.astype(np.float32))
        lift_tcp_waypoints_b = [world_pose_to_robot_base_pose(robot, pose_w) for pose_w in lift_tcp_waypoints_w]
        refresh_curobo_world()
        lift_start_q = robot.data.joint_pos[0, curobo_entity_cfg.joint_ids].detach().cpu().numpy().astype(np.float32)
        lift_chain_plan: CuroboPlanResult = curobo_planner.solve_ik_chain_for_poses(
            lift_start_q,
            lift_tcp_waypoints_b,
            return_seeds=int(args_cli.curobo_cartesian_descent_return_seeds),
            max_waypoint_joint_l2_rad=float(args_cli.curobo_cartesian_descent_max_waypoint_joint_l2_rad),
            position_only=False,
        )
        lift_min_steps = max(
            int(args_cli.lift_steps),
            max(1, int(lift_chain_plan.joint_positions.shape[0])) * max(1, int(args_cli.curobo_cartesian_descent_steps_per_waypoint)),
        )
        if bool(lift_chain_plan.success):
            lift_segment = run_curobo_joint_trajectory(
                sim,
                scene,
                robot,
                curobo_entity_cfg,
                lift_chain_plan.joint_positions,
                min_steps=lift_min_steps,
                gripper=gripper,
                gripper_width=0.0,
                locked_root_pose_w=locked_root_pose_w,
                target_tcp_pose_w=vertical_lift_tcp_pose_w,
                max_joint_step_override=float(args_cli.curobo_cartesian_descent_max_joint_step),
                cartesian_lock_xy_w=lift_start_tcp_pose_w[:2, 3],
                max_cartesian_xy_error_m=float(args_cli.curobo_cartesian_descent_max_xy_error_m),
                abort_on_cartesian_xy_error=True,
            )
            lift_error = float(lift_segment.get("final_tcp_pos_error_m", lift_segment.get("final_pos_error_m", float("inf"))))
        else:
            lift_segment = {
                "final_pos_error_m": float("inf"),
                "final_tcp_pos_error_m": float("inf"),
                "position_error_reference": "tcp",
                "executed_steps": 0,
                "aborted": True,
                "abort_reason": lift_chain_plan.reason,
            }
            lift_error = float("inf")
        lift_segment.update(
            {
                "name": "lift_cartesian_linear_curobo_ik_ascent",
                "motion_backend": "curobo_dense_cartesian_ik_chain_after_close",
                "curobo_lift_planning_skipped": True,
                "curobo_lift_planning_skip_reason": "After physical contact, lift locks the actual grasp TCP X/Y/quaternion and only raises world Z.",
                "curobo_cartesian_ik_chain_success": bool(lift_chain_plan.success),
                "curobo_cartesian_ik_chain_reason": lift_chain_plan.reason,
                "curobo_cartesian_ik_chain_metadata": lift_chain_plan.metadata,
                "orientation_constraint_mode": "full_pose_curobo_cartesian_straight_line_hold_closed_gripper_tcp_orientation",
                "vertical_lift_only": True,
                "cartesian_straight_line_ascent": True,
                "locked_xy_source": "actual_tcp_after_gripper_close",
                "lift_start_tcp_world_m": lift_start_tcp_pose_w[:3, 3].astype(float).tolist(),
                "target_tcp_world_m": vertical_lift_tcp_pose_w[:3, 3].astype(float).tolist(),
                "final_tcp_world_m": tcp_pose_matrix(robot, ee_body_id)[:3, 3].astype(float).tolist(),
                "lift_delta_world_m": lift_delta_w.astype(float).tolist(),
                "cartesian_lift_waypoint_count": int(len(lift_tcp_waypoints_w)),
                "cartesian_lift_min_steps": int(lift_min_steps),
                "final_tcp_pos_error_m": lift_error,
                "final_pos_error_m": lift_error,
                "error_threshold_m": float(args_cli.lift_error_threshold_m),
                "converged": bool(
                    bool(lift_chain_plan.success)
                    and not bool(lift_segment.get("aborted", False))
                    and math.isfinite(lift_error)
                    and lift_error <= float(args_cli.lift_error_threshold_m)
                ),
            }
        )
        ok = bool(lift_segment["converged"])
        reason = "" if ok else (lift_segment.get("abort_reason") or "cuRobo Cartesian straight-line lift error exceeded threshold.")
        if not ok:
            fallback_ok, fallback_lift_segment, fallback_reason = plan_and_run_stage(
                "lift_curobo",
                "LIFT",
                primitive["tcp_lift"],
                int(args_cli.lift_steps),
                float(args_cli.lift_error_threshold_m),
                0.0,
                wrist_pose_w=primitive["wrist_lift"],
                allow_local_ik_fallback=True,
            )
            fallback_lift_segment["fallback_after_cartesian_lift_failure"] = True
            fallback_lift_segment["fallback_reason"] = reason
            segments.append(lift_segment)
            lift_segment = fallback_lift_segment
            ok = fallback_ok
            reason = fallback_reason
    else:
        ok, lift_segment, reason = plan_and_run_stage(
            "lift_curobo",
            "LIFT",
            primitive["tcp_lift"],
            int(args_cli.lift_steps),
            float(args_cli.lift_error_threshold_m),
            0.0,
            wrist_pose_w=primitive["wrist_lift"],
            allow_local_ik_fallback=True,
        )
    segments.append(lift_segment)
    updates["tcp_pose_after_lift_world"] = tcp_pose_matrix(robot, ee_body_id).tolist()
    updates["gripper_alignment_after_lift"] = gripper_alignment_diagnostics(robot, ee_body_id, primitive["tcp_lift"])

    tracker.enter("VERIFY_HOLD", attempt_index=attempt_idx, backend="curobo_right_arm")
    lift_hold_joint_target = robot.data.joint_pos.clone()
    for _ in range(int(args_cli.hold_steps)):
        robot.set_joint_position_target(lift_hold_joint_target)
        gripper.close()
        stabilize_robot_base(robot, locked_root_pose_w)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim.get_physics_dt())
        record_video_frame(scene)
        gui_playback_tick(sim.get_physics_dt())
        tracker.tick()

    updates["segments"] = segments
    return {
        "success": bool(ok),
        "reason": "" if ok else f"LIFT cuRobo failed: {reason}",
        "segments": segments,
        "updates": updates,
        "lift_segment": lift_segment,
    }


def intended_target_identity(scene: InteractiveScene, candidate: SelectedGrasp, target_center_w: np.ndarray | None) -> tuple[str | None, str | None, float | None, dict[str, Any]]:
    metadata = candidate.target.metadata or {}
    intended_scene_key = metadata.get("scene_key")
    intended_scene_object_name = metadata.get("scene_object_name") or scene_object_name_for_key(intended_scene_key)
    identity_meta: dict[str, Any] = {
        "metadata_scene_key": intended_scene_key,
        "metadata_scene_object_name": intended_scene_object_name,
        "locked_to_visual_target": False,
    }
    if intended_scene_key in scene.keys():
        root = tensor_to_numpy(scene[intended_scene_key].data.root_state_w[0, :3]).astype(np.float32)
        if target_center_w is not None and np.isfinite(target_center_w).all():
            match_distance = float(np.linalg.norm(root - np.asarray(target_center_w, dtype=np.float32).reshape(3)))
        else:
            match_distance = None
        identity_meta["locked_to_visual_target"] = True
        return intended_scene_key, intended_scene_object_name, match_distance, identity_meta

    scene_key, scene_object_name, match_distance = locate_target_rigid_object(scene, target_center_w)
    identity_meta.update(
        {
            "nearest_scene_key": scene_key,
            "nearest_scene_object_name": scene_object_name,
            "nearest_match_distance_m": match_distance,
        }
    )
    return scene_key, scene_object_name, match_distance, identity_meta


def probe_ik_pose(
    robot: Articulation,
    ik_controller: DifferentialIKController,
    robot_entity_cfg: SceneEntityCfg,
    ee_body_id: int,
    ee_jacobi_idx: int,
    target_pose_w: np.ndarray,
    *,
    target_tcp_pose_w: np.ndarray | None = None,
    control_tcp_position: bool = False,
) -> dict[str, Any]:
    target_pos_b, target_quat_b = pose_to_torch_command(target_pose_w, robot)
    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    tcp_position_control = bool(control_tcp_position and ik_controller.action_dim == 3 and target_tcp_pose_w is not None)
    target_tcp_np = None
    initial_tcp_error = float("nan")
    if target_tcp_pose_w is not None:
        target_tcp_np = np.asarray(target_tcp_pose_w, dtype=np.float32).reshape(4, 4)[:3, 3].copy()
        initial_tcp_error = float(np.linalg.norm(tcp_pose_matrix(robot, ee_body_id)[:3, 3] - target_tcp_np))
    if tcp_position_control and target_tcp_np is not None:
        target_tcp_pos_w = torch.tensor(target_tcp_np, dtype=torch.float32, device=robot.device).repeat(robot.num_instances, 1)
        tcp_offset = torch.tensor(gripper_tcp_offset_m(), dtype=torch.float32, device=robot.device).reshape(1, 3, 1).repeat(robot.num_instances, 1, 1)
        wrist_rot_w = matrix_from_quat(ee_pose_w[:, 3:7])
        target_wrist_pos_w = target_tcp_pos_w - torch.bmm(wrist_rot_w, tcp_offset).squeeze(-1)
        target_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], target_wrist_pos_w, ee_pose_w[:, 3:7]
        )
    command = torch.zeros(robot.num_instances, ik_controller.action_dim, device=robot.device)
    command[:, 0:3] = target_pos_b
    ik_controller.reset()
    if ik_controller.action_dim == 3:
        ik_controller.set_command(command, ee_quat=ee_quat_b)
    else:
        command[:, 3:7] = target_quat_b
        ik_controller.set_command(command)
    jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
    joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, robot.data.joint_pos[:, robot_entity_cfg.joint_ids])
    target_pos = target_pos_b[0].detach().cpu().numpy()
    max_joint_delta = float(torch.max(torch.abs(joint_pos_des - robot.data.joint_pos[:, robot_entity_cfg.joint_ids])).item())
    wrist_position_error = float(torch.linalg.norm(target_pos_b - ee_pos_b, dim=1).max().item())
    position_error = initial_tcp_error if tcp_position_control else wrist_position_error
    in_workspace = bool(0.20 <= target_pos[0] <= 0.90 and -0.70 <= target_pos[1] <= 0.20 and 0.34 <= target_pos[2] <= 1.05)
    finite = bool(torch.isfinite(joint_pos_des).all().item() and np.all(np.isfinite(target_pos)))
    max_joint_delta_threshold = float(args_cli.ik_prescreen_max_joint_delta_rad)
    return {
        "finite": finite,
        "in_workspace": in_workspace,
        "initial_pos_error_m": position_error,
        "initial_wrist_pos_error_m": wrist_position_error,
        "initial_tcp_pos_error_m": initial_tcp_error,
        "max_joint_delta_rad": max_joint_delta,
        "max_joint_delta_threshold_rad": max_joint_delta_threshold,
        "ik_command_type": str(ik_controller.cfg.command_type),
        "tcp_position_control": tcp_position_control,
        "orientation_constraint_mode": "position_only_tcp" if tcp_position_control else "pose",
        "target_pos_b": target_pos.tolist(),
        "target_tcp_world_m": target_tcp_np.astype(float).tolist() if target_tcp_np is not None else None,
        "reachable": bool(finite and in_workspace and max_joint_delta <= max_joint_delta_threshold),
    }


def candidate_uses_position_only_tcp(candidate: SelectedGrasp) -> bool:
    if args_cli.arm_motion_backend != "legacy_differential_ik":
        return True
    if candidate.source == "fallback":
        return bool(args_cli.fallback_position_only_ik)
    if str(candidate.source).startswith("mind_sort_") or "rgbd_center" in str(candidate.source):
        return True
    if candidate.source in {"graspnet", "anygrasp"}:
        return bool(args_cli.learned_grasp_position_only_ik)
    return False


def ik_orientation_mode(ik_controller: DifferentialIKController, tcp_position_control: bool) -> str:
    if tcp_position_control and ik_controller.action_dim == 3:
        return "position_only_tcp"
    return "pose"


def prescreen_grasp_candidates(
    candidates: list[SelectedGrasp],
    robot: Articulation,
    pose_ik_controller: DifferentialIKController,
    position_ik_controller: DifferentialIKController,
    entity_cfg: SceneEntityCfg,
    ee_body_id: int,
    ee_jacobi_idx: int,
) -> tuple[list[SelectedGrasp], list[dict[str, Any]]]:
    kept: list[SelectedGrasp] = []
    records: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates[: max(args_cli.ik_prescreen_top_k, args_cli.max_grasp_retries)]):
        poses = grasp_execution_poses(candidate, robot, ee_body_id)
        tcp_position_control = candidate_uses_position_only_tcp(candidate)
        ik_controller = position_ik_controller if tcp_position_control else pose_ik_controller
        probes = {
            "pregrasp": probe_ik_pose(
                robot,
                ik_controller,
                entity_cfg,
                ee_body_id,
                ee_jacobi_idx,
                poses["wrist_pregrasp"],
                target_tcp_pose_w=poses["tcp_pregrasp"],
                control_tcp_position=tcp_position_control,
            ),
            "grasp": probe_ik_pose(
                robot,
                ik_controller,
                entity_cfg,
                ee_body_id,
                ee_jacobi_idx,
                poses["wrist_grasp"],
                target_tcp_pose_w=poses["tcp_grasp"],
                control_tcp_position=tcp_position_control,
            ),
            "lift": probe_ik_pose(
                robot,
                ik_controller,
                entity_cfg,
                ee_body_id,
                ee_jacobi_idx,
                poses["wrist_lift"],
                target_tcp_pose_w=poses["tcp_lift"],
                control_tcp_position=tcp_position_control,
            ),
        }
        reachable = all(item["reachable"] for item in probes.values())
        initial_errors = [float(item["initial_pos_error_m"]) for item in probes.values()]
        joint_deltas = [float(item["max_joint_delta_rad"]) for item in probes.values()]
        pregrasp_initial_error = float(probes["pregrasp"]["initial_pos_error_m"])
        max_initial_error = max(initial_errors) if initial_errors else float("inf")
        max_joint_delta = max(joint_deltas) if joint_deltas else float("inf")
        ik_cost = pregrasp_initial_error + 0.35 * float(probes["grasp"]["initial_pos_error_m"]) + 0.10 * max_initial_error + 0.02 * max_joint_delta
        candidate.metadata = dict(candidate.metadata or {})
        candidate.metadata.update({
            "ik_prescreen_rank": idx,
            "ik_prescreen": probes,
            "ik_prescreen_reachable": reachable,
            "ik_prescreen_pregrasp_initial_error_m": pregrasp_initial_error,
            "ik_prescreen_max_initial_error_m": max_initial_error,
            "ik_prescreen_max_joint_delta_rad": max_joint_delta,
            "ik_prescreen_cost": float(ik_cost),
            "ik_command_type": str(ik_controller.cfg.command_type),
            "tcp_position_control": tcp_position_control,
            "orientation_constraint_mode": ik_orientation_mode(ik_controller, tcp_position_control),
        })
        record = {
            "rank": idx,
            "source": candidate.source,
            "score": candidate.score,
            "reachable": reachable,
            "ik_command_type": str(ik_controller.cfg.command_type),
            "tcp_position_control": tcp_position_control,
            "orientation_constraint_mode": ik_orientation_mode(ik_controller, tcp_position_control),
            "pregrasp_initial_error_m": pregrasp_initial_error,
            "max_initial_error_m": max_initial_error,
            "max_joint_delta_rad": max_joint_delta,
            "ik_cost": float(ik_cost),
            "probes": probes,
            "candidate_metadata": candidate.metadata,
        }
        records.append(record)
        if reachable:
            kept.append(candidate)
    return kept, records


def ordered_execution_candidates(candidates: list[SelectedGrasp], prescreened: list[SelectedGrasp]) -> list[SelectedGrasp]:
    if not args_cli.grasp_ik_prescreen:
        def no_prescreen_cost(item: SelectedGrasp, idx: int) -> float:
            metadata = item.metadata or {}
            center_distance = float(metadata.get("target_center_distance_m", 0.15) or 0.15)
            center_distance = max(0.0, min(center_distance, 0.15))
            score = float(item.score or 0.0)
            return center_distance - 0.015 * score + 0.002 * idx

        ranked_backend_candidates = sorted(
            [(idx, item) for idx, item in enumerate([candidate for candidate in candidates if candidate.source != "fallback"])],
            key=lambda pair: no_prescreen_cost(pair[1], pair[0]),
        )
        backend_candidates: list[SelectedGrasp] = []
        for order_idx, (original_idx, item) in enumerate(ranked_backend_candidates):
            metadata = dict(item.metadata or {})
            order_cost = no_prescreen_cost(item, original_idx)
            metadata.update({
                "ik_prescreen_rank": original_idx,
                "ik_prescreen_reachable": None,
                "ik_prescreen_skipped": True,
                "ik_prescreen_cost": 0.0,
                "ik_execution_order_rank": int(order_idx),
                "ik_execution_order_cost": float(order_cost),
                "ik_execution_order_policy": "no_prescreen_center_distance_then_score",
            })
            item.metadata = metadata
            backend_candidates.append(item)
        fallback_candidates = [item for item in candidates if item.source == "fallback"][:1] if centroid_fallback_enabled() else []
        return backend_candidates[: max(args_cli.max_grasp_retries, 0)] + fallback_candidates

    def ik_execution_cost(item: SelectedGrasp) -> float:
        metadata = item.metadata or {}
        center_distance = float(metadata.get("target_center_distance_m", 0.15) or 0.15)
        center_distance = max(0.0, min(center_distance, 0.15))
        top_down_bonus = float(metadata.get("top_down_bonus", 0.0) or 0.0)
        ik_cost = float(metadata.get("ik_prescreen_cost", float("inf")))
        return ik_cost + 1.40 * center_distance - 0.03 * top_down_bonus

    def ik_order_key(item: SelectedGrasp) -> tuple[float, float, float, float, int]:
        metadata = item.metadata or {}
        order_cost = ik_execution_cost(item)
        metadata["ik_execution_order_cost"] = float(order_cost)
        item.metadata = metadata
        return (
            order_cost,
            float(metadata.get("target_center_distance_m", 0.15) or 0.15),
            float(metadata.get("ik_prescreen_pregrasp_initial_error_m", float("inf"))),
            float(metadata.get("ik_prescreen_max_joint_delta_rad", float("inf"))),
            int(metadata.get("rank_after_scoring", metadata.get("ik_prescreen_rank", 9999))),
        )

    backend_candidates = sorted([item for item in prescreened if item.source != "fallback"], key=ik_order_key)[: max(args_cli.max_grasp_retries, 0)]
    fallback_candidates = [item for item in candidates if item.source == "fallback"][:1] if centroid_fallback_enabled() else []
    return backend_candidates + fallback_candidates


def verify_grasp_after_lift(
    scene: InteractiveScene,
    robot: Articulation,
    initial_root_xy: torch.Tensor,
    initial_z_by_key: dict[str, float],
    target_scene_key: str | None,
    target_initial_z: float | None,
    lift_segment: dict[str, Any],
) -> dict[str, Any]:
    base_drift = float(torch.linalg.norm(robot.data.root_pos_w[:, 0:2] - initial_root_xy, dim=1).max().item())
    target_final_z = object_root_z(scene, target_scene_key)
    target_delta = float(target_final_z - target_initial_z) if target_final_z is not None and target_initial_z is not None else 0.0
    best_lifted_key, best_lift_delta = lifted_object_delta(scene, initial_z_by_key)
    lift_delta = max(target_delta, best_lift_delta)
    lift_ik_ok = float(lift_segment.get("final_pos_error_m", float("inf"))) <= float(args_cli.lift_error_threshold_m)
    grasp_success = bool(lift_delta >= 0.05)
    if grasp_success:
        reason = "Object lifted and held."
    elif not lift_ik_ok:
        reason = "LIFT IK error exceeded threshold and no object was verified above the table."
    else:
        reason = "No object was verified above the table after lift."
    return {
        "success": grasp_success,
        "grasp_success": grasp_success,
        "lift_delta_m": lift_delta,
        "target_delta_m": target_delta,
        "best_lift_delta_m": best_lift_delta,
        "best_lifted_scene_key": best_lifted_key,
        "base_drift_m": base_drift,
        "lift_ik_ok": lift_ik_ok,
        "reason": reason,
    }


def execute_grasp(
    sim: SimulationContext,
    scene: InteractiveScene,
    selected: SelectedGrasp | list[SelectedGrasp] | None,
    cycle_dir: Path | None = None,
    calibration_debug_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tracker = Task319StateTracker(sim.get_physics_dt(), sim.device)
    tracker.enter("REST", warp_enabled=tracker.warp_enabled)
    candidates = [] if selected is None else (selected if isinstance(selected, list) else [selected])
    if not candidates:
        tracker.enter("RECOVER", reason="No selected grasp.")
        if cycle_dir is not None:
            (cycle_dir / "fsm_trace.json").write_text(json.dumps(tracker.trace, indent=2, ensure_ascii=False))
        return {"enabled": True, "success": False, "reason": "No selected grasp.", "attempts": [], "fsm_trace": tracker.trace}
    robot: Articulation = scene["robot"]
    gripper = AttachedParallelGripper(robot)
    pose_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    pose_ik_controller = DifferentialIKController(pose_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    position_ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    position_ik_controller = DifferentialIKController(position_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    entity_cfg = SceneEntityCfg("robot", joint_names=isaaclab_grasp_ik_joint_exprs(), body_names=[RIGHT_EE_BODY])
    entity_cfg.resolve(scene)
    ee_body_id = entity_cfg.body_ids[0]
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
    initial_root_xy = robot.data.root_pos_w[:, 0:2].clone()
    locked_root_pose_w = robot.data.root_pose_w.clone()
    initial_wrist_pose_w = wrist_pose_matrix(robot, ee_body_id)
    initial_arm_joint_target = robot.data.joint_pos[:, entity_cfg.joint_ids].clone()
    fallback_wrist_rot_w = initial_wrist_pose_w[:3, :3].copy()
    calibration_camera_to_world = None
    if calibration_debug_context is not None and calibration_debug_context.get("camera_to_world") is not None:
        calibration_camera_to_world = np.asarray(calibration_debug_context["camera_to_world"], dtype=np.float32)
    initial_z_by_key = {
        scene_key: float(scene[scene_key].data.root_state_w[0, 2].detach().cpu())
        for scene_key, _ in TRASH_SCENE_OBJECTS
        if scene_key in scene.keys()
    }
    if args_cli.grasp_ik_prescreen:
        prescreened, prescreen_records = prescreen_grasp_candidates(
            candidates,
            robot,
            pose_ik_controller,
            position_ik_controller,
            entity_cfg,
            ee_body_id,
            ee_jacobi_idx,
        )
    else:
        prescreened = [candidate for candidate in candidates if candidate.source != "fallback"]
        prescreen_records = [{
            "enabled": False,
            "reason": "grasp IK prescreen disabled by --no-grasp_ik_prescreen",
            "candidate_count": len(candidates),
            "backend_candidate_count": len(prescreened),
            "max_grasp_retries": int(args_cli.max_grasp_retries),
        }]
    attempt_candidates = ordered_execution_candidates(candidates, prescreened)
    if not attempt_candidates:
        no_candidate_reason = "No IK-reachable grasp candidate." if args_cli.grasp_ik_prescreen else "No learned grasp candidate to execute after disabled IK prescreen."
        tracker.enter("RECOVER", reason=no_candidate_reason)
        if cycle_dir is not None:
            (cycle_dir / "fsm_trace.json").write_text(json.dumps(tracker.trace, indent=2, ensure_ascii=False))
        return {
            "enabled": True,
            "success": False,
            "grasp_success": False,
            "reason": no_candidate_reason,
            "grasp_ik_prescreen_enabled": bool(args_cli.grasp_ik_prescreen),
            "ik_prescreen": prescreen_records,
            "attempts": [],
            "fsm_trace": tracker.trace,
        }

    metrics: dict[str, Any] = {
        "enabled": True,
        "success": False,
        "grasp_success": False,
        "drop_success": False,
        "reason": "No grasp attempt succeeded.",
        "segments": [],
        "attempts": [],
        "grasp_ik_prescreen_enabled": bool(args_cli.grasp_ik_prescreen),
        "ik_prescreen": prescreen_records,
        "ik_reachable_count": len(prescreened),
        "candidate_count": len(candidates),
        "attempt_candidate_count": len(attempt_candidates),
        "base_drift_m": 0.0,
        "lift_delta_m": 0.0,
        "best_lifted_scene_key": None,
        "navigation": [],
        "drop": None,
        "gripper_tcp_urdf_calibration": RIGHT_GRIPPER_TCP_URDF_CALIBRATION,
        "gripper_tcp_offset_m": gripper_tcp_offset_m().tolist(),
        "gripper_local_tcp_offset_m": gripper_local_tcp_offset_m().tolist(),
        "right_gripper_mount_rpy_rad": list(RIGHT_GRIPPER_INLINE_MOUNT_RPY),
        "grasp_approach_axis_local": GRASP_APPROACH_AXIS_LOCAL.tolist(),
        "grasp_frame_convention": dict(GRASP_FRAME_CONVENTION),
        "tcp_alignment_mode": f"{args_cli.grasp_backend}_grasp_to_kuavo_tcp_to_wrist",
        "arm_motion_backend": args_cli.arm_motion_backend,
        "grasp_motion_profile": STABLE_GRASP_MOTION_PROFILE if args_cli.arm_motion_backend == "local_position_primitive" else str(args_cli.arm_motion_backend),
        "mind_sort_force_stable_grasp_profile": bool(getattr(args_cli, "mind_sort_force_stable_grasp_profile", False)),
        "pregrasp_orientation_mode": "current_wrist" if args_cli.pregrasp_use_current_wrist_orientation else "grasp_orientation",
        "learned_grasp_orientation_mode": learned_grasp_orientation_mode(),
        "fallback_position_only_ik": bool(args_cli.fallback_position_only_ik),
        "calibration_debug_context": {
            "camera_to_world_available": calibration_camera_to_world is not None,
            "planned_projection_image": str(calibration_debug_context.get("projection_image", "")) if calibration_debug_context else "",
        },
        "robot_fixed_base": bool(robot.is_fixed_base),
        "ee_body_id": int(ee_body_id),
        "ee_jacobi_idx": int(ee_jacobi_idx),
    }
    kuavo_ik_client: KuavoIkSocketClient | None = None
    kuavo_ik_available = False
    kuavo_ik_error = ""
    kuavo_analytic_ik_client: KuavoAnalyticIkClient | None = None
    kuavo_analytic_ik_available = False
    kuavo_analytic_ik_error = ""
    curobo_planner: KuavoRightArmCuroboPlanner | None = None
    curobo_available = False
    curobo_error = ""
    if args_cli.arm_motion_backend in {"kuavo_ik", "auto"}:
        kuavo_ik_client = KuavoIkSocketClient(cycle_dir)
        kuavo_ik_available = kuavo_ik_client.start()
        kuavo_ik_error = "" if kuavo_ik_available else kuavo_ik_client.last_error
        metrics["kuavo_ik_bridge"] = {
            "requested": True,
            "available": bool(kuavo_ik_available),
            "host": args_cli.kuavo_ik_bridge_host,
            "port": int(args_cli.kuavo_ik_bridge_port),
            "auto_start": bool(args_cli.kuavo_ik_auto_start),
            "error": kuavo_ik_error,
        }
    if args_cli.arm_motion_backend == "kuavo_analytic_ik" or (
        args_cli.arm_motion_backend == "curobo_right_arm" and bool(args_cli.curobo_use_kuavo_analytic_seed)
    ):
        kuavo_analytic_ik_client = KuavoAnalyticIkClient(cycle_dir)
        kuavo_analytic_ik_available = kuavo_analytic_ik_client.start()
        kuavo_analytic_ik_error = "" if kuavo_analytic_ik_available else kuavo_analytic_ik_client.last_error
        metrics["kuavo_analytic_ik"] = {
            "requested": True,
            "used_by_curobo_seed": bool(args_cli.arm_motion_backend == "curobo_right_arm"),
            "available": bool(kuavo_analytic_ik_available),
            "cli": str(args_cli.kuavo_analytic_ik_cli),
            "source": str(args_cli.kuavo_analytic_ik_source),
            "auto_build": bool(args_cli.kuavo_analytic_ik_auto_build),
            "joint_limit_margin_rad": float(args_cli.kuavo_analytic_ik_joint_limit_margin_rad),
            "max_fk_error_m": float(args_cli.kuavo_analytic_ik_max_fk_error_m),
            "tcp_refine": bool(args_cli.kuavo_analytic_ik_tcp_refine),
            "refine_steps": int(args_cli.kuavo_analytic_ik_refine_steps),
            "refine_error_threshold_m": float(args_cli.kuavo_analytic_ik_refine_error_threshold_m),
            "grasp_tcp_threshold_m": float(args_cli.kuavo_analytic_ik_grasp_tcp_threshold_m),
            "execute_top_k": int(args_cli.kuavo_analytic_ik_execute_top_k),
            "local_grasp_descent_on_seed_failure": bool(args_cli.kuavo_analytic_ik_local_grasp_descent_on_seed_failure),
            "error": kuavo_analytic_ik_error,
        }
    if args_cli.arm_motion_backend == "curobo_right_arm":
        try:
            curobo_planner = KuavoRightArmCuroboPlanner(
                KUAVO_BASE_URDF,
                GRIPPER_URDF,
                tcp_offset_m=tuple(float(v) for v in args_cli.gripper_local_tcp_offset),
                world_model=curobo_world_for_current_robot(robot),
                device=str(args_cli.curobo_device),
                warmup=True,
                use_cuda_graph=False,
                collision_activation_distance_m=float(args_cli.curobo_collision_activation_distance_m),
                joint_limit_clip_rad=float(args_cli.curobo_joint_limit_clip_rad),
                rotation_threshold_rad=float(args_cli.curobo_rotation_threshold_rad),
                strict_rotation_threshold_rad=float(args_cli.curobo_final_grasp_rotation_threshold_rad),
            )
            curobo_available = True
        except Exception as exc:
            curobo_error = repr(exc)
            curobo_planner = None
        metrics["curobo_right_arm"] = {
            "requested": True,
            "available": bool(curobo_available),
            "device": str(args_cli.curobo_device),
            "right_arm_only": True,
            "max_attempts": int(args_cli.curobo_max_attempts),
            "timeout_s": float(args_cli.curobo_timeout_s),
            "enable_graph": bool(args_cli.curobo_enable_graph),
            "plan_table_obstacles": bool(args_cli.curobo_plan_table_obstacles),
            "collision_activation_distance_m": float(args_cli.curobo_collision_activation_distance_m),
            "rotation_threshold_rad": float(args_cli.curobo_rotation_threshold_rad),
            "final_grasp_rotation_threshold_rad": float(args_cli.curobo_final_grasp_rotation_threshold_rad),
            "joint_limit_clip_rad": float(args_cli.curobo_joint_limit_clip_rad),
            "final_settle_steps": int(args_cli.curobo_final_settle_steps),
            "position_only_tcp": bool(args_cli.curobo_position_only_tcp),
            "use_kuavo_analytic_seed": bool(args_cli.curobo_use_kuavo_analytic_seed),
            "prefer_kuavo_seed_for_position_only": bool(args_cli.curobo_prefer_kuavo_seed_for_position_only),
            "gripper_local_tcp_offset_m": gripper_local_tcp_offset_m().tolist(),
            "error": curobo_error,
        }

    try:
        tracker.enter("BASE_TO_TABLE", mode="already_at_table")
        tracker.enter("STATIC_PERCEPT", camera_source="head_rgbd")
        for attempt_idx, candidate in enumerate(attempt_candidates):
            poses = grasp_execution_poses(candidate, robot, ee_body_id, fallback_wrist_rot_w=fallback_wrist_rot_w)
            open_width = gripper_fully_open_width(gripper)
            tcp_position_control = candidate_uses_position_only_tcp(candidate)
            ik_controller = position_ik_controller if tcp_position_control else pose_ik_controller
            orientation_constraint_mode = ik_orientation_mode(ik_controller, tcp_position_control)
            has_next_attempt = attempt_idx < len(attempt_candidates) - 1
            target_center_w = candidate.target.center_3d if candidate.target is not None else candidate.pose_world[:3, 3]
            target_scene_key, target_object_name, target_match_distance, target_identity_meta = intended_target_identity(scene, candidate, target_center_w)
            nearest_scene_key, nearest_object_name, nearest_match_distance = locate_target_rigid_object(scene, target_center_w)
            target_initial_z = initial_z_by_key.get(target_scene_key or "", object_root_z(scene, target_scene_key))
            target_category = canonical_category(candidate.target.waste_category if candidate.target else None)
            attempt: dict[str, Any] = {
                "attempt_index": attempt_idx,
                "candidate_source": candidate.source,
                "candidate_score": candidate.score,
                "candidate_width_m": candidate.width,
                "candidate_metadata": candidate.metadata or {},
                "target_scene_key": target_scene_key,
                "target_object_name": target_object_name,
                "target_match_distance_m": target_match_distance,
                "target_identity": target_identity_meta,
                "nearest_scene_key_from_target_center": nearest_scene_key,
                "nearest_object_name_from_target_center": nearest_object_name,
                "nearest_match_distance_from_target_center_m": nearest_match_distance,
                "target_category": target_category,
                "tcp_grasp_pose_world": poses["tcp_grasp"].tolist(),
                "wrist_grasp_pose_world": poses["wrist_grasp"].tolist(),
                "motion_backend_requested": args_cli.arm_motion_backend,
                "ik_command_type": str(ik_controller.cfg.command_type),
                "tcp_position_control": tcp_position_control,
                "orientation_constraint_mode": orientation_constraint_mode,
                "axis_alignment_required": not tcp_position_control,
                "segments": [],
                "success": False,
                "reason": "",
            }
            metrics["attempts"].append(attempt)
            metrics.update({
                "target_scene_key": target_scene_key,
                "target_object_name": target_object_name,
                "target_match_distance_m": target_match_distance,
                "target_identity": target_identity_meta,
                "target_category": target_category,
                "grasp_source": candidate.source,
                "tcp_grasp_pose_world": poses["tcp_grasp"].tolist(),
                "wrist_grasp_pose_world": poses["wrist_grasp"].tolist(),
                "ik_command_type": str(ik_controller.cfg.command_type),
                "tcp_position_control": tcp_position_control,
                "orientation_constraint_mode": orientation_constraint_mode,
                "axis_alignment_required": not tcp_position_control,
            })
            current_tcp_pose_w = tcp_pose_matrix(robot, ee_body_id)
            marker_rgbd_center = np.asarray(target_center_w, dtype=np.float32).reshape(3) if target_center_w is not None else None
            marker_truth_center = scene_root_position_world(scene, target_scene_key)
            render_calibration_debug_markers(
                scene,
                poses["tcp_grasp"],
                current_tcp_pose_w,
                rgbd_center_world=marker_rgbd_center,
                object_truth_world=marker_truth_center,
            )
            attempt["tcp_pose_before_pregrasp_world"] = current_tcp_pose_w.tolist()
            if calibration_camera_to_world is not None:
                print_calibration_triplet(
                    phase=f"attempt_{attempt_idx}_pregrasp_start",
                    camera_to_world=calibration_camera_to_world,
                    target_pose_world=poses["tcp_grasp"],
                    tcp_pose_world=current_tcp_pose_w,
                    gripper_pose_world=body_pose_matrix_by_name(robot, "gripper_base"),
                )
            tracker.enter("PLAN_GRASP", attempt_index=attempt_idx, grasp_width_m=candidate.width, grasp_score=candidate.score, target_scene_key=target_scene_key, source=candidate.source)
            if args_cli.arm_motion_backend != "legacy_differential_ik":
                primitive_result: dict[str, Any]
                if args_cli.arm_motion_backend in {"kuavo_ik", "auto"} and kuavo_ik_available and kuavo_ik_client is not None:
                    primitive_result = execute_kuavo_ik_primitive_attempt(
                        sim,
                        scene,
                        robot,
                        gripper,
                        kuavo_ik_client,
                        position_ik_controller,
                        entity_cfg,
                        ee_body_id,
                        ee_jacobi_idx,
                        poses,
                        candidate,
                        tracker,
                        attempt_idx,
                        locked_root_pose_w,
                        calibration_camera_to_world=calibration_camera_to_world,
                    )
                    if args_cli.arm_motion_backend == "auto" and primitive_result.get("lift_segment") is None:
                        attempt["kuavo_ik_auto_result_before_fallback"] = {
                            "reason": primitive_result.get("reason"),
                            "segments": primitive_result.get("segments", []),
                        }
                        primitive_result = execute_local_position_primitive_attempt(
                            sim,
                            scene,
                            robot,
                            gripper,
                            pose_ik_controller,
                            position_ik_controller,
                            entity_cfg,
                            ee_body_id,
                            ee_jacobi_idx,
                            poses,
                            candidate,
                            tracker,
                            attempt_idx,
                            locked_root_pose_w,
                            calibration_camera_to_world=calibration_camera_to_world,
                        )
                        primitive_result.setdefault("updates", {})["kuavo_ik_auto_fallback_reason"] = attempt["kuavo_ik_auto_result_before_fallback"]["reason"]
                elif args_cli.arm_motion_backend == "kuavo_analytic_ik" and kuavo_analytic_ik_available and kuavo_analytic_ik_client is not None:
                    primitive_result = execute_kuavo_analytic_ik_primitive_attempt(
                        sim,
                        scene,
                        robot,
                        gripper,
                        kuavo_analytic_ik_client,
                        position_ik_controller,
                        entity_cfg,
                        ee_body_id,
                        ee_jacobi_idx,
                        poses,
                        candidate,
                        tracker,
                        attempt_idx,
                        locked_root_pose_w,
                        calibration_camera_to_world=calibration_camera_to_world,
                    )
                elif args_cli.arm_motion_backend == "curobo_right_arm" and curobo_available and curobo_planner is not None:
                    primitive_result = execute_curobo_right_arm_attempt(
                        sim,
                        scene,
                        robot,
                        gripper,
                        curobo_planner,
                        kuavo_analytic_ik_client if kuavo_analytic_ik_available else None,
                        pose_ik_controller,
                        position_ik_controller,
                        entity_cfg,
                        ee_body_id,
                        ee_jacobi_idx,
                        poses,
                        candidate,
                        tracker,
                        attempt_idx,
                        locked_root_pose_w,
                        calibration_camera_to_world=calibration_camera_to_world,
                    )
                elif args_cli.arm_motion_backend == "kuavo_ik":
                    primitive_result = {
                        "success": False,
                        "reason": f"Kuavo IK bridge unavailable: {kuavo_ik_error}",
                        "segments": [],
                        "updates": {
                            "motion_backend": "kuavo_ik",
                            "kuavo_ik_available": False,
                            "kuavo_ik_error": kuavo_ik_error,
                        },
                        "lift_segment": None,
                    }
                elif args_cli.arm_motion_backend == "kuavo_analytic_ik":
                    primitive_result = {
                        "success": False,
                        "reason": f"Kuavo analytic IK unavailable: {kuavo_analytic_ik_error}",
                        "segments": [],
                        "updates": {
                            "motion_backend": "kuavo_analytic_ik",
                            "kuavo_analytic_ik_available": False,
                            "kuavo_analytic_ik_error": kuavo_analytic_ik_error,
                        },
                        "lift_segment": None,
                    }
                elif args_cli.arm_motion_backend == "curobo_right_arm":
                    primitive_result = {
                        "success": False,
                        "reason": f"cuRobo right-arm planner unavailable: {curobo_error}",
                        "segments": [],
                        "updates": {
                            "motion_backend": "curobo_right_arm",
                            "curobo_available": False,
                            "curobo_error": curobo_error,
                            "curobo_right_arm_only": True,
                        },
                        "lift_segment": None,
                    }
                else:
                    primitive_result = execute_local_position_primitive_attempt(
                        sim,
                        scene,
                        robot,
                        gripper,
                        pose_ik_controller,
                        position_ik_controller,
                        entity_cfg,
                        ee_body_id,
                        ee_jacobi_idx,
                        poses,
                        candidate,
                        tracker,
                        attempt_idx,
                        locked_root_pose_w,
                        calibration_camera_to_world=calibration_camera_to_world,
                    )
                for key, value in primitive_result.get("updates", {}).items():
                    if key != "segments":
                        attempt[key] = value
                        metrics[key] = value
                attempt["segments"].extend(primitive_result.get("segments", []))
                attempt["motion_backend_requested"] = args_cli.arm_motion_backend
                if args_cli.arm_motion_backend == "auto" and attempt.get("motion_backend") == "local_position_primitive":
                    attempt["motion_backend_fallback"] = "local_position_primitive"
                    attempt["motion_backend_fallback_reason"] = str(attempt.get("kuavo_ik_auto_fallback_reason") or kuavo_ik_error or "Kuavo IK bridge unavailable.")
                metrics["segments"] = attempt["segments"]
                metrics["motion_backend"] = attempt.get("motion_backend", "local_position_primitive")
                lift_segment = primitive_result.get("lift_segment")
                if lift_segment is None:
                    attempt["success"] = False
                    attempt["reason"] = str(primitive_result.get("reason") or "Position-only primitive failed before lift.")
                    tracker.event("attempt_rejected", attempt_index=attempt_idx, reason=attempt["reason"])
                    metrics["reason"] = attempt["reason"]
                    if has_next_attempt:
                        gripper.set_width(open_width)
                        reset_steps = min(max(args_cli.trajectory_steps // 2, 90), 240)
                        reset_segment = run_joint_segment(
                            sim,
                            scene,
                            robot,
                            entity_cfg,
                            initial_arm_joint_target,
                            reset_steps,
                            gripper=gripper,
                            gripper_width=open_width,
                            locked_root_pose_w=locked_root_pose_w,
                            target_pose_w=initial_wrist_pose_w,
                        )
                        attempt["segments"].append({"name": "retry_reset", **reset_segment})
                        metrics["segments"] = attempt["segments"]
                        continue
                    break

                verification = verify_grasp_after_lift(
                    scene,
                    robot,
                    initial_root_xy,
                    initial_z_by_key,
                    target_scene_key,
                    target_initial_z,
                    lift_segment,
                )
                attempt.update(verification)
                tracker.event(
                    "verify_lift",
                    attempt_index=attempt_idx,
                    grasp_success=verification["grasp_success"],
                    target_delta_m=verification["target_delta_m"],
                    best_lift_delta_m=verification["best_lift_delta_m"],
                    best_lifted_scene_key=verification["best_lifted_scene_key"],
                    base_drift_m=verification["base_drift_m"],
                )
                metrics["segments"] = attempt["segments"]
                metrics["base_drift_m"] = verification["base_drift_m"]
                metrics["lift_delta_m"] = verification["lift_delta_m"]
                metrics["best_lifted_scene_key"] = verification["best_lifted_scene_key"]
                metrics["grasp_success"] = verification["grasp_success"]
                metrics["success"] = verification["success"]
                metrics["reason"] = verification["reason"]
                if verification["success"]:
                    break
                gripper.set_width(open_width)
                retreat_steps = min(max(args_cli.trajectory_steps // 3, 30), 120)
                retreat_segment = run_ik_segment(
                    sim,
                    scene,
                    robot,
                    position_ik_controller,
                    entity_cfg,
                    ee_body_id,
                    ee_jacobi_idx,
                    np.asarray(attempt.get("wrist_pregrasp_pose_world", poses["wrist_pregrasp"]), dtype=np.float32),
                    retreat_steps,
                    gripper=gripper,
                    gripper_width=open_width,
                    locked_root_pose_w=locked_root_pose_w,
                    target_tcp_pose_w=np.asarray(attempt.get("tcp_pregrasp_pose_world", poses["tcp_pregrasp"]), dtype=np.float32),
                    control_tcp_position=True,
                )
                attempt["segments"].append({"name": "retreat", **retreat_segment})
                metrics["segments"] = attempt["segments"]
                continue
            if args_cli.safe_pregrasp_start:
                safe_waypoints = safe_pregrasp_waypoints(poses, robot, ee_body_id)
                attempt["safe_pregrasp"] = {
                    "enabled": True,
                    "safe_z_m": safe_waypoints["safe_z_m"],
                    "tcp_safe_lift_current_world": safe_waypoints["tcp_safe_lift_current"].tolist(),
                    "tcp_safe_standoff_world": safe_waypoints["tcp_safe_standoff"].tolist(),
                    "error_threshold_m": float(args_cli.safe_pregrasp_error_threshold_m),
                }
                tracker.enter("SAFE_START", attempt_index=attempt_idx, stage="lift_current_tcp")
                safe_lift_segment = run_ik_segment(
                    sim,
                    scene,
                    robot,
                    ik_controller,
                    entity_cfg,
                    ee_body_id,
                    ee_jacobi_idx,
                    safe_waypoints["wrist_safe_lift_current"],
                    args_cli.safe_pregrasp_steps,
                    locked_root_pose_w=locked_root_pose_w,
                    debug_target_pose_w=poses["tcp_grasp"],
                    target_tcp_pose_w=safe_waypoints["tcp_safe_lift_current"],
                    control_tcp_position=tcp_position_control,
                )
                attempt["segments"].append({"name": "safe_lift_current", **safe_lift_segment})
                tracker.enter("SAFE_START", attempt_index=attempt_idx, stage="high_standoff")
                safe_standoff_segment = run_ik_segment(
                    sim,
                    scene,
                    robot,
                    ik_controller,
                    entity_cfg,
                    ee_body_id,
                    ee_jacobi_idx,
                    safe_waypoints["wrist_safe_standoff"],
                    args_cli.safe_pregrasp_steps,
                    gripper=gripper,
                            gripper_width=open_width,
                    locked_root_pose_w=locked_root_pose_w,
                    debug_target_pose_w=poses["tcp_grasp"],
                    target_tcp_pose_w=safe_waypoints["tcp_safe_standoff"],
                    control_tcp_position=tcp_position_control,
                )
                attempt["segments"].append({"name": "safe_standoff", **safe_standoff_segment})
                if safe_standoff_segment["final_pos_error_m"] > args_cli.safe_pregrasp_error_threshold_m:
                    attempt["reason"] = "SAFE_START IK error exceeded threshold."
                    tracker.event("attempt_rejected", attempt_index=attempt_idx, reason=attempt["reason"], final_pos_error_m=safe_standoff_segment["final_pos_error_m"])
                    metrics["segments"] = attempt["segments"]
                    metrics["reason"] = attempt["reason"]
                    if has_next_attempt:
                        continue
                    break
            else:
                attempt["safe_pregrasp"] = {"enabled": False}
            gripper.set_width(open_width)
            tracker.enter("PRE_GRASP", attempt_index=attempt_idx)
            pre_segment = run_ik_segment(
                sim,
                scene,
                robot,
                ik_controller,
                entity_cfg,
                ee_body_id,
                ee_jacobi_idx,
                poses["wrist_pregrasp"],
                args_cli.trajectory_steps,
                gripper=gripper,
                gripper_width=open_width,
                locked_root_pose_w=locked_root_pose_w,
                debug_target_pose_w=poses["tcp_grasp"],
                target_tcp_pose_w=poses["tcp_pregrasp"],
                control_tcp_position=tcp_position_control,
            )
            attempt["segments"].append({"name": "pregrasp", **pre_segment})
            current_tcp_pose_w = tcp_pose_matrix(robot, ee_body_id)
            render_calibration_debug_markers(scene, poses["tcp_grasp"], current_tcp_pose_w)
            attempt["tcp_pose_after_pregrasp_world"] = current_tcp_pose_w.tolist()
            if calibration_camera_to_world is not None:
                print_calibration_triplet(
                    phase=f"attempt_{attempt_idx}_after_pregrasp",
                    camera_to_world=calibration_camera_to_world,
                    target_pose_world=poses["tcp_grasp"],
                    tcp_pose_world=current_tcp_pose_w,
                )
            if pre_segment["final_pos_error_m"] > args_cli.pregrasp_error_threshold_m:
                attempt["reason"] = "PRE_GRASP IK error exceeded threshold."
                tracker.event("attempt_rejected", attempt_index=attempt_idx, reason=attempt["reason"], final_pos_error_m=pre_segment["final_pos_error_m"])
                metrics["segments"] = attempt["segments"]
                if has_next_attempt:
                    gripper.set_width(open_width)
                    reset_steps = min(max(args_cli.trajectory_steps // 2, 90), 240)
                    reset_segment = run_joint_segment(
                        sim,
                        scene,
                        robot,
                        entity_cfg,
                        initial_arm_joint_target,
                        reset_steps,
                        gripper=gripper,
                        gripper_width=open_width,
                        locked_root_pose_w=locked_root_pose_w,
                        target_pose_w=initial_wrist_pose_w,
                    )
                    attempt["segments"].append({"name": "retry_reset", **reset_segment})
                continue
            grasp_segment = run_ik_segment(
                sim,
                scene,
                robot,
                ik_controller,
                entity_cfg,
                ee_body_id,
                ee_jacobi_idx,
                poses["wrist_grasp"],
                args_cli.trajectory_steps,
                gripper=gripper,
                gripper_width=open_width,
                locked_root_pose_w=locked_root_pose_w,
                debug_target_pose_w=poses["tcp_grasp"],
                target_tcp_pose_w=poses["tcp_grasp"],
                control_tcp_position=tcp_position_control,
            )
            attempt["segments"].append({"name": "grasp_pose", **grasp_segment})
            current_tcp_pose_w = tcp_pose_matrix(robot, ee_body_id)
            render_calibration_debug_markers(scene, poses["tcp_grasp"], current_tcp_pose_w)
            attempt["tcp_pose_after_grasp_world"] = current_tcp_pose_w.tolist()
            if calibration_camera_to_world is not None:
                print_calibration_triplet(
                    phase=f"attempt_{attempt_idx}_after_grasp_pose",
                    camera_to_world=calibration_camera_to_world,
                    target_pose_world=poses["tcp_grasp"],
                    tcp_pose_world=current_tcp_pose_w,
                )
            if grasp_segment["final_pos_error_m"] > args_cli.grasp_error_threshold_m:
                attempt["reason"] = "GRASP IK error exceeded threshold."
                tracker.event("attempt_rejected", attempt_index=attempt_idx, reason=attempt["reason"], final_pos_error_m=grasp_segment["final_pos_error_m"])
                metrics["segments"] = attempt["segments"]
                if has_next_attempt:
                    gripper.set_width(open_width)
                    reset_steps = min(max(args_cli.trajectory_steps // 2, 90), 240)
                    reset_segment = run_joint_segment(
                        sim,
                        scene,
                        robot,
                        entity_cfg,
                        initial_arm_joint_target,
                        reset_steps,
                        gripper=gripper,
                            gripper_width=open_width,
                        locked_root_pose_w=locked_root_pose_w,
                        target_pose_w=initial_wrist_pose_w,
                    )
                    attempt["segments"].append({"name": "retry_reset", **reset_segment})
                continue
            tracker.enter("GRASP", attempt_index=attempt_idx)
            grasp_hold_joint_target = robot.data.joint_pos.clone()
            close_segment = run_slow_gripper_close(
                sim,
                scene,
                robot,
                gripper,
                tracker,
                grasp_hold_joint_target,
                locked_root_pose_w,
                start_width_m=open_width,
                steps=int(args_cli.grasp_steps),
            )
            attempt["segments"].append({"name": "close_gripper", **close_segment})
            contact_ok, contact_meta, contact_reason = evaluate_gripper_contact_before_lift(close_segment, candidate)
            attempt["gripper_contact_check"] = contact_meta
            metrics["gripper_contact_check"] = contact_meta
            if not contact_ok:
                attempt["reason"] = contact_reason
                tracker.event("attempt_rejected", attempt_index=attempt_idx, reason=attempt["reason"], gripper_contact_check=contact_meta)
                metrics["segments"] = attempt["segments"]
                metrics["reason"] = attempt["reason"]
                if has_next_attempt:
                    continue
                break
            tracker.enter("LIFT", attempt_index=attempt_idx)
            lift_segment = run_ik_segment(
                sim,
                scene,
                robot,
                ik_controller,
                entity_cfg,
                ee_body_id,
                ee_jacobi_idx,
                poses["wrist_lift"],
                args_cli.lift_steps,
                gripper=gripper,
                gripper_width=0.0,
                locked_root_pose_w=locked_root_pose_w,
                target_tcp_pose_w=poses["tcp_lift"],
                control_tcp_position=tcp_position_control,
            )
            attempt["segments"].append({"name": "lift", **lift_segment})
            tracker.enter("VERIFY_HOLD", attempt_index=attempt_idx)
            lift_hold_joint_target = robot.data.joint_pos.clone()
            for _ in range(args_cli.hold_steps):
                robot.set_joint_position_target(lift_hold_joint_target)
                gripper.close()
                stabilize_robot_base(robot, locked_root_pose_w)
                scene.write_data_to_sim()
                sim.step(render=True)
                scene.update(sim.get_physics_dt())
                record_video_frame(scene)
                gui_playback_tick(sim.get_physics_dt())
                tracker.tick()
            base_drift = float(torch.linalg.norm(robot.data.root_pos_w[:, 0:2] - initial_root_xy, dim=1).max().item())
            target_final_z = object_root_z(scene, target_scene_key)
            target_delta = float(target_final_z - target_initial_z) if target_final_z is not None and target_initial_z is not None else 0.0
            best_lifted_key, best_lift_delta = lifted_object_delta(scene, initial_z_by_key)
            lift_delta = max(target_delta, best_lift_delta)
            lift_ik_ok = lift_segment["final_pos_error_m"] <= args_cli.lift_error_threshold_m
            grasp_success = lift_delta >= 0.05
            if grasp_success:
                attempt_reason = "Object lifted and held."
            elif not lift_ik_ok:
                attempt_reason = "LIFT IK error exceeded threshold and no object was verified above the table."
            else:
                attempt_reason = "No object was verified above the table after lift."
            attempt.update({
                "success": grasp_success,
                "grasp_success": grasp_success,
                "lift_delta_m": lift_delta,
                "target_delta_m": target_delta,
                "best_lift_delta_m": best_lift_delta,
                "best_lifted_scene_key": best_lifted_key,
                "base_drift_m": base_drift,
                "lift_ik_ok": lift_ik_ok,
                "reason": attempt_reason,
            })
            tracker.event("verify_lift", attempt_index=attempt_idx, grasp_success=grasp_success, target_delta_m=target_delta, best_lift_delta_m=best_lift_delta, best_lifted_scene_key=best_lifted_key, base_drift_m=base_drift)
            metrics["segments"] = attempt["segments"]
            metrics["base_drift_m"] = base_drift
            metrics["lift_delta_m"] = lift_delta
            metrics["best_lifted_scene_key"] = best_lifted_key
            metrics["grasp_success"] = grasp_success
            metrics["success"] = grasp_success
            metrics["reason"] = attempt["reason"]
            if grasp_success:
                break
            gripper.set_width(open_width)
            retreat_steps = min(max(args_cli.trajectory_steps // 3, 30), 120)
            retreat_segment = run_ik_segment(
                sim,
                scene,
                robot,
                ik_controller,
                entity_cfg,
                ee_body_id,
                ee_jacobi_idx,
                poses["wrist_pregrasp"],
                retreat_steps,
                gripper=gripper,
                gripper_width=gripper_command_width(gripper, candidate.width),
                locked_root_pose_w=locked_root_pose_w,
                target_tcp_pose_w=poses["tcp_pregrasp"],
                control_tcp_position=tcp_position_control,
            )
            attempt["segments"].append({"name": "retreat", **retreat_segment})
        tracker.enter("DONE" if metrics["success"] else "RECOVER", reason=metrics["reason"])
    except Exception as exc:
        metrics["success"] = False
        metrics["grasp_success"] = False
        metrics["reason"] = repr(exc)
        tracker.enter("RECOVER", reason=repr(exc))
    finally:
        if kuavo_ik_client is not None:
            kuavo_ik_client.close()
        if kuavo_analytic_ik_client is not None:
            kuavo_analytic_ik_client.close()
    metrics["fsm_trace"] = tracker.trace
    if cycle_dir is not None:
        write_grasp_execution_debug_files(scene, robot, ee_body_id, metrics, cycle_dir, calibration_debug_context)
        (cycle_dir / "fsm_trace.json").write_text(json.dumps(tracker.trace, indent=2, ensure_ascii=False))
    return metrics

def apply_ycb_visual_runtime_physics() -> None:
    from pxr import Usd

    stage = sim_utils.get_current_stage()
    convex_cfg = sim_utils.ConvexHullPropertiesCfg(hull_vertex_limit=64)
    collider_cfg = collision_props(
        contact_offset=float(args_cli.trash_contact_offset_m),
        rest_offset=float(args_cli.trash_rest_offset_m),
    )
    effective_trash_rest_offset = max(0.0, float(args_cli.trash_rest_offset_m))
    applied_count = 0
    for env_id in range(args_cli.num_envs):
        for name, mass in YCB_VISUAL_RUNTIME_MASSES.items():
            root_path = f"/World/envs/env_{env_id}/{name}"
            root_prim = stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                continue
            sim_utils.define_rigid_body_properties(root_path, rigid_props(), stage=stage)
            sim_utils.define_mass_properties(root_path, sim_utils.MassPropertiesCfg(mass=mass), stage=stage)
            for prim in Usd.PrimRange(root_prim):
                if prim.GetTypeName() != "Mesh":
                    continue
                mesh_path = str(prim.GetPath())
                sim_utils.define_collision_properties(mesh_path, collider_cfg, stage=stage)
                sim_utils.modify_mesh_collision_properties(mesh_path, convex_cfg, stage=stage)
                applied_count += 1
    if applied_count > 0:
        print(
            "[TRASH_CONTACT] applied "
            f"contact_offset={float(args_cli.trash_contact_offset_m):.4f}m "
            f"rest_offset={effective_trash_rest_offset:.4f}m "
            f"to {applied_count} runtime trash collision meshes.",
            flush=True,
        )


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
    name_tokens = ("gripper_base", "left_finger", "right_finger")
    effective_gripper_rest_offset = max(0.0, float(args_cli.gripper_rest_offset_m))
    applied_count = 0
    for env_id in range(args_cli.num_envs):
        root_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Kuavo62")
        if not root_prim.IsValid():
            continue
        for prim in Usd.PrimRange(root_prim):
            prim_path = str(prim.GetPath())
            if any(token in prim_path for token in name_tokens) and not prim.IsInstanceProxy() and prim.HasAPI(UsdPhysics.CollisionAPI):
                sim_utils.define_collision_properties(
                    prim_path,
                    collision_props(
                        float(args_cli.gripper_contact_offset_m),
                        float(args_cli.gripper_rest_offset_m),
                    ),
                    stage=stage,
                )
                sim_utils.bind_physics_material(prim_path, material_path, stage=stage)
                applied_count += 1
    if applied_count > 0:
        print(
            "[GRIPPER_CONTACT] applied "
            f"contact_offset={float(args_cli.gripper_contact_offset_m):.4f}m "
            f"rest_offset={effective_gripper_rest_offset:.4f}m "
            f"to {applied_count} gripper collision prims.",
            flush=True,
        )


def parse_grasp_standpoint_candidates(value: str) -> list[str]:
    names = [item.strip() for item in str(value).split(",") if item.strip()]
    if not names:
        raise ValueError("--grasp_standpoint_candidates must contain at least one waypoint name.")
    unknown = [name for name in names if name not in WAYPOINT_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown grasp standpoint waypoint(s): {unknown}. Available: {waypoint_names()}")
    invalid = [
        name
        for name in names
        if str(WAYPOINT_REGISTRY[name].get("role", "")) != "table_standpoint" and not name.startswith("table_")
    ]
    if invalid:
        raise ValueError(f"Grasp standpoint candidates must be table waypoints, got: {invalid}")
    return list(dict.fromkeys(names))


def run_active_perception_model(rgb: np.ndarray, out_dir: Path) -> list[YoloInstance]:
    if args_cli.perception_source == "v28_original":
        return wait_for_model_result("V28-ORIGINAL", run_v28_original, rgb, out_dir)
    raise RuntimeError(f"Unsupported retired perception source: {args_cli.perception_source}")


def attach_rgbd_geometry_to_instances(
    scene: InteractiveScene,
    instances: list[YoloInstance],
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
) -> None:
    for inst in instances:
        points_world = mask_points_world(inst.mask, depth_m, intrinsics, t_camera_to_world)
        inst.points_world = points_world.astype(np.float32, copy=False)
        if points_world.shape[0] > 0:
            inst.center_3d, inst.rgbd_center_metadata = rgbd_geometric_grasp_center(points_world)
            if inst.source in VISUAL_TARGET_SOURCES:
                scene_key, scene_object_name, match_distance = locate_target_rigid_object(scene, inst.center_3d)
                if match_distance <= yolo_scene_match_radius(scene_key):
                    inst.scene_key = scene_key
                    inst.scene_object_name = scene_object_name


def grasp_standpoint_candidate_record(name: str, target_world_xyz: np.ndarray, robot_pose_xyyaw: list[float]) -> dict[str, Any]:
    pose = waypoint_pose(name)
    stand_xy = np.array([float(pose[0]), float(pose[1])], dtype=np.float32)
    local_xy = world_delta_to_local_xy(target_world_xyz[:2] - stand_xy, float(pose[2]))
    target_on_table = bool(
        TABLE_X_LIMITS[0] <= float(target_world_xyz[0]) <= TABLE_X_LIMITS[1]
        and TABLE_Y_LIMITS[0] <= float(target_world_xyz[1]) <= TABLE_Y_LIMITS[1]
    )
    reject_reasons = candidate_reject_reasons(target_world_xyz, local_xy, target_on_table)
    preferred_local = np.array([RIGHT_ARM_PREFERRED_TARGET_X, RIGHT_ARM_PREFERRED_TARGET_Y], dtype=np.float32)
    local_error = abs(float(local_xy[0]) - float(preferred_local[0])) + 0.45 * abs(float(local_xy[1]) - float(preferred_local[1]))
    travel_distance = float(np.linalg.norm(stand_xy - np.asarray(robot_pose_xyyaw[:2], dtype=np.float32)))
    same_side_bias = 0.0
    if name.endswith("_near") and float(target_world_xyz[0]) > TABLE_CENTER_XY[0]:
        same_side_bias = 0.12
    if name.endswith("_far") and float(target_world_xyz[0]) < TABLE_CENTER_XY[0]:
        same_side_bias = 0.12
    score = local_error + 0.20 * travel_distance + same_side_bias + 10.0 * len(reject_reasons)
    return {
        "id": name,
        "waypoint": name,
        "pose": [float(pose[0]), float(pose[1]), float(pose[2])],
        "description": str(WAYPOINT_REGISTRY[name].get("description", "")),
        "target_local_xy": [float(local_xy[0]), float(local_xy[1])],
        "preferred_target_local_xy": [float(preferred_local[0]), float(preferred_local[1])],
        "geometric_reachable": len(reject_reasons) == 0,
        "reject_reason": ";".join(reject_reasons),
        "score": float(score),
        "geometric_score": float(local_error),
        "travel_distance_m": travel_distance,
        "same_side_bias": float(same_side_bias),
    }


def draw_grasp_standpoint_plan_rgb(
    rgb: np.ndarray,
    target: YoloInstance | None,
    plan_meta: dict[str, Any],
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    out_dir: Path,
) -> str:
    pil = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(pil)
    if target is not None:
        x0, y0, x1, y1 = target.bbox_xyxy
        draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=4)
        ys, xs = np.nonzero(target.mask)
        if xs.size:
            step = max(1, xs.size // 1200)
            for x, y in zip(xs[::step], ys[::step]):
                pil.putpixel((int(x), int(y)), (255, 32, 32))
        if target.center_3d is not None:
            proj = project_world_point_to_image(target.center_3d, intrinsics, t_camera_to_world)
            plan_meta["target_projection"] = proj
            if proj.get("finite") and "uv" in proj:
                u, v = proj["uv"]
                if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]:
                    draw.line((u - 14, v, u + 14, v), fill=(255, 0, 0), width=4)
                    draw.line((u, v - 14, u, v + 14), fill=(255, 0, 0), width=4)
        label = f"target {target.index}: {target.vlm.object_name if target.vlm else target.yolo_label}"
        draw.text((max(0.0, x0), max(0.0, y0 - 18)), label, fill=(255, 0, 0))
    selected = plan_meta.get("selected") or {}
    text = f"standpoint: {selected.get('waypoint', 'none')} score={float(selected.get('score', 0.0) or 0.0):.2f}"
    draw.rectangle((8, 8, 610, 40), fill=(0, 0, 0))
    draw.text((14, 16), text, fill=(255, 255, 255))
    path = out_dir / "planned_standpoint_rgb.png"
    pil.save(path)
    return str(path)


def draw_grasp_standpoint_plan_topdown(plan_meta: dict[str, Any], out_dir: Path) -> str:
    width, height = 900, 620
    x_min, x_max = 0.0, 4.2
    y_min, y_max = -1.75, 1.75

    def px(x: float, y: float) -> tuple[int, int]:
        u = int(round((float(x) - x_min) / (x_max - x_min) * (width - 1)))
        v = int(round((y_max - float(y)) / (y_max - y_min) * (height - 1)))
        return u, v

    pil = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(pil)
    tx0, ty0 = px(TABLE_X_LIMITS[0], TABLE_Y_LIMITS[1])
    tx1, ty1 = px(TABLE_X_LIMITS[1], TABLE_Y_LIMITS[0])
    draw.rectangle((tx0, ty0, tx1, ty1), outline=(80, 80, 80), fill=(218, 218, 218), width=3)
    draw.text((tx0 + 6, ty0 + 6), "table", fill=(40, 40, 40))
    target = plan_meta.get("target_world_xyz")
    if target:
        u, v = px(float(target[0]), float(target[1]))
        draw.ellipse((u - 8, v - 8, u + 8, v + 8), fill=(255, 0, 0), outline=(120, 0, 0))
        draw.text((u + 10, v - 10), "target", fill=(180, 0, 0))
    robot_pose = plan_meta.get("initial_robot_pose")
    if robot_pose:
        u, v = px(float(robot_pose[0]), float(robot_pose[1]))
        draw.rectangle((u - 8, v - 8, u + 8, v + 8), fill=(40, 40, 40))
        draw.text((u + 10, v - 10), "current", fill=(20, 20, 20))
    selected_id = (plan_meta.get("selected") or {}).get("waypoint")
    for item in plan_meta.get("candidates", []):
        pose = item.get("pose", [0.0, 0.0, 0.0])
        u, v = px(float(pose[0]), float(pose[1]))
        reachable = bool(item.get("geometric_reachable", False))
        color = (0, 150, 70) if reachable else (170, 170, 170)
        if item.get("waypoint") == selected_id:
            color = (0, 80, 255)
            draw.ellipse((u - 11, v - 11, u + 11, v + 11), outline=color, width=4)
        else:
            draw.ellipse((u - 7, v - 7, u + 7, v + 7), fill=color)
        yaw = float(pose[2])
        tip = (u + int(28 * math.cos(yaw)), v - int(28 * math.sin(yaw)))
        draw.line((u, v, tip[0], tip[1]), fill=color, width=3)
        draw.text((u + 10, v + 4), str(item.get("waypoint", "")), fill=color)
    draw.text((14, 14), "Preset grasp standpoint plan (world/map top-down)", fill=(0, 0, 0))
    path = out_dir / "planned_standpoint_topdown.png"
    pil.save(path)
    return str(path)


def plan_preset_grasp_standpoint(
    scene: InteractiveScene,
    target: YoloInstance | None,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    out_dir: Path,
) -> dict[str, Any]:
    robot: Articulation = scene["robot"]
    candidate_names = parse_grasp_standpoint_candidates(args_cli.grasp_standpoint_candidates)
    meta: dict[str, Any] = {
        "enabled": bool(args_cli.grasp_standpoint_nav),
        "mode": "preset_table_side_right_arm_window",
        "candidate_waypoints": candidate_names,
        "initial_robot_pose": robot_planar_pose(robot),
        "target_world_xyz": target.center_3d.astype(float).tolist() if target is not None and target.center_3d is not None else None,
        "right_arm_window": {
            "target_local_x_m": [float(RIGHT_ARM_TARGET_X_BOUNDS[0]), float(RIGHT_ARM_TARGET_X_BOUNDS[1])],
            "target_local_y_m": [float(RIGHT_ARM_TARGET_Y_BOUNDS[0]), float(RIGHT_ARM_TARGET_Y_BOUNDS[1])],
            "target_z_world_m": [float(TABLE_SURFACE_Z + 0.015), float(TABLE_SURFACE_Z + 0.35)],
            "preferred_target_local_xy": [float(RIGHT_ARM_PREFERRED_TARGET_X), float(RIGHT_ARM_PREFERRED_TARGET_Y)],
        },
        "selected": None,
        "candidates": [],
        "debug_images": {},
        "reason": "",
    }
    if target is None or target.center_3d is None:
        meta["reason"] = "No target with RGB-D center is available for standpoint planning."
    else:
        target_w = np.asarray(target.center_3d, dtype=np.float32).reshape(3)
        candidates = [grasp_standpoint_candidate_record(name, target_w, meta["initial_robot_pose"]) for name in candidate_names]
        meta["candidates"] = sorted(candidates, key=lambda item: float(item["score"]))
        reachable = [item for item in meta["candidates"] if item.get("geometric_reachable")]
        if reachable:
            meta["selected"] = reachable[0]
            meta["reason"] = "Selected lowest-score preset waypoint that puts the target in the right-arm local reach window."
        elif meta["candidates"]:
            meta["selected"] = meta["candidates"][0]
            meta["reason"] = "No preset waypoint passed the right-arm local reach window; selected lowest-score candidate for debug only."
        else:
            meta["reason"] = "No preset waypoint candidates."
    meta["debug_images"]["rgb"] = draw_grasp_standpoint_plan_rgb(rgb, target, meta, intrinsics, t_camera_to_world, out_dir)
    meta["debug_images"]["topdown"] = draw_grasp_standpoint_plan_topdown(meta, out_dir)
    (out_dir / "planned_grasp_standpoint.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def dynamic_standpoint_target_filter(inst: YoloInstance, instances: list[YoloInstance]) -> tuple[bool, str, dict[str, Any]]:
    selection_score = target_selection_score(inst, instances)
    if inst.source not in dynamic_target_sources():
        return False, "target_source_not_enabled_for_dynamic_standpoint", selection_score
    if target_image_guard_metrics(inst).get("rejected", False):
        return False, "robot_self_occlusion_or_bottom_image_guard", selection_score
    if int(inst.mask.sum()) < MIN_TARGET_MASK_PIXELS:
        return False, "mask_too_small", selection_score
    if inst.center_3d is None:
        return False, "missing_rgbd_3d_center", selection_score
    if int(target_point_cloud_metrics(inst).get("point_count", 0) or 0) < MIN_TARGET_POINT_COUNT:
        return False, "insufficient_rgbd_points", selection_score
    if inst.vlm is None:
        return False, "missing_vlm_classification", selection_score
    if args_cli.skip_vlm:
        return False, "vlm_required_but_skip_vlm_enabled", selection_score
    if inst.vlm.error:
        return False, f"vlm_error:{inst.vlm.error}", selection_score
    if inst.vlm.confidence < args_cli.vlm_min_conf:
        return False, "vlm_confidence_below_threshold", selection_score
    if args_cli.target_category and inst.vlm.waste_category != args_cli.target_category:
        return False, "target_category_mismatch", selection_score
    if not target_name_matches(inst):
        return False, "target_name_mismatch", selection_score
    return True, "", selection_score


def dynamic_standpoint_target_record(
    inst: YoloInstance,
    candidate_meta: dict[str, Any] | None,
    selection_score: dict[str, Any],
    reject_reason: str = "",
) -> dict[str, Any]:
    selected = (candidate_meta or {}).get("selected") if candidate_meta is not None else None
    reachable_count = 0
    if candidate_meta is not None:
        reachable_count = sum(1 for item in candidate_meta.get("candidates", []) if item.get("geometric_reachable"))
    candidate_score = float(selected.get("score", float("inf"))) if selected else float("inf")
    penalty = 0 if selected is not None else 1
    total_score = float(selection_score.get("total", 0.0)) + float(candidate_score if np.isfinite(candidate_score) else 1.0e3) + 10.0 * penalty
    record = target_candidate_record(inst, selection_score=selection_score, reject_reason=reject_reason)
    record.update({
        "dynamic_candidate_count": len(candidate_meta.get("candidates", [])) if candidate_meta is not None else 0,
        "dynamic_reachable_count": int(reachable_count),
        "dynamic_selected": selected,
        "dynamic_penalty": int(penalty),
        "dynamic_score": float(candidate_score if np.isfinite(candidate_score) else 1.0e3),
        "dynamic_total_score": float(total_score),
    })
    return record


def draw_dynamic_standpoint_rgb(
    rgb: np.ndarray,
    instances: list[YoloInstance],
    selected_target: YoloInstance | None,
    plan_meta: dict[str, Any],
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    out_dir: Path,
) -> str:
    pil = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(pil)
    selected_idx = selected_target.index if selected_target is not None else None
    draw_sources = dynamic_target_sources()
    for rank, inst in enumerate(instances):
        if inst.source not in draw_sources or inst.center_3d is None:
            continue
        if inst.index == selected_idx:
            color = (255, 0, 0)
        elif inst.source == DEPTH_COMPONENT_TARGET_SOURCE:
            color = (40, 130, 255)
        else:
            color = (0, 150, 70)
        x0, y0, x1, y1 = inst.bbox_xyxy
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4 if inst.index == selected_idx else 2)
        proj = project_world_point_to_image(inst.center_3d, intrinsics, t_camera_to_world)
        if proj.get("finite") and "uv" in proj:
            u, v = proj["uv"]
            if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]:
                draw.line((u - 9, v, u + 9, v), fill=color, width=3)
                draw.line((u, v - 9, u, v + 9), fill=color, width=3)
        label = f"{rank}:{inst.vlm.object_name if inst.vlm else inst.yolo_label}"
        if inst.source == DEPTH_COMPONENT_TARGET_SOURCE:
            label = f"{label} [depth]"
        draw.text((max(0.0, x0), max(0.0, y0 - 18)), label, fill=color)
    selected = plan_meta.get("selected") or {}
    text = f"dynamic: {selected.get('waypoint', 'none')} score={float(selected.get('score', 0.0) or 0.0):.2f}"
    draw.rectangle((8, 8, 660, 40), fill=(0, 0, 0))
    draw.text((14, 16), text, fill=(255, 255, 255))
    path = out_dir / "dynamic_standpoint_rgb.png"
    pil.save(path)
    return str(path)


def draw_dynamic_standpoint_topdown(plan_meta: dict[str, Any], out_dir: Path) -> str:
    width, height = 900, 620
    x_min, x_max = 0.0, 4.2
    y_min, y_max = -1.75, 1.75

    def px(x: float, y: float) -> tuple[int, int]:
        u = int(round((float(x) - x_min) / (x_max - x_min) * (width - 1)))
        v = int(round((y_max - float(y)) / (y_max - y_min) * (height - 1)))
        return u, v

    pil = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(pil)
    tx0, ty0 = px(TABLE_X_LIMITS[0], TABLE_Y_LIMITS[1])
    tx1, ty1 = px(TABLE_X_LIMITS[1], TABLE_Y_LIMITS[0])
    draw.rectangle((tx0, ty0, tx1, ty1), outline=(80, 80, 80), fill=(218, 218, 218), width=3)
    draw.text((tx0 + 6, ty0 + 6), "table", fill=(40, 40, 40))
    robot_pose = plan_meta.get("initial_robot_pose")
    if robot_pose:
        u, v = px(float(robot_pose[0]), float(robot_pose[1]))
        draw.rectangle((u - 8, v - 8, u + 8, v + 8), fill=(40, 40, 40))
        draw.text((u + 10, v - 10), "observe", fill=(20, 20, 20))
    selected_instance = (plan_meta.get("selected_target") or {}).get("instance_index")
    for item in plan_meta.get("valid_targets", []):
        center = item.get("center_3d_world")
        if not center:
            continue
        is_selected = item.get("instance_index") == selected_instance
        color = (255, 0, 0) if is_selected else (0, 150, 70)
        u, v = px(float(center[0]), float(center[1]))
        draw.ellipse((u - 7, v - 7, u + 7, v + 7), fill=color)
        draw.text((u + 9, v - 8), str(item.get("instance_index")), fill=color)
    selected = plan_meta.get("selected") or {}
    for item in plan_meta.get("candidates", []):
        pose = item.get("pose", [0.0, 0.0, 0.0])
        u, v = px(float(pose[0]), float(pose[1]))
        color = (0, 80, 255) if item.get("id") == selected.get("id") else (150, 150, 255)
        draw.ellipse((u - 9, v - 9, u + 9, v + 9), outline=color, width=3)
        yaw = float(pose[2])
        tip = (u + int(28 * math.cos(yaw)), v - int(28 * math.sin(yaw)))
        draw.line((u, v, tip[0], tip[1]), fill=color, width=3)
        draw.text((u + 10, v + 4), str(item.get("side", "dyn")), fill=color)
    draw.text((14, 14), "Dynamic RGB-D grasp standpoint plan", fill=(0, 0, 0))
    path = out_dir / "dynamic_standpoint_topdown.png"
    pil.save(path)
    return str(path)


def plan_dynamic_grasp_standpoint(
    scene: InteractiveScene,
    instances: list[YoloInstance],
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
    out_dir: Path,
) -> tuple[YoloInstance | None, str, dict[str, Any]]:
    robot: Articulation = scene["robot"]
    allowed_sides = parse_dynamic_table_sides(args_cli.dynamic_allowed_table_sides)
    robot_pose = robot_planar_pose(robot)
    meta: dict[str, Any] = {
        "enabled": bool(args_cli.dynamic_grasp_standpoint_nav),
        "mode": "dynamic_rgbd_table_side_right_arm_window",
        "initial_robot_pose": robot_pose,
        "observe_pose": [float(v) for v in args_cli.dynamic_observe_pose],
        "allowed_sides": allowed_sides,
        "table_bounds": {
            "x": [float(TABLE_X_LIMITS[0]), float(TABLE_X_LIMITS[1])],
            "y": [float(TABLE_Y_LIMITS[0]), float(TABLE_Y_LIMITS[1])],
        },
        "right_arm_window": {
            "target_local_x_m": [float(RIGHT_ARM_TARGET_X_BOUNDS[0]), float(RIGHT_ARM_TARGET_X_BOUNDS[1])],
            "target_local_y_m": [float(RIGHT_ARM_TARGET_Y_BOUNDS[0]), float(RIGHT_ARM_TARGET_Y_BOUNDS[1])],
            "target_z_world_m": [float(TABLE_SURFACE_Z + 0.015), float(TABLE_SURFACE_Z + 0.35)],
            "preferred_target_local_xy": [float(RIGHT_ARM_PREFERRED_TARGET_X), float(RIGHT_ARM_PREFERRED_TARGET_Y)],
        },
        "selected": None,
        "selected_target": None,
        "candidates": [],
        "valid_targets": [],
        "rejected_targets": [],
        "target_source_policy": {
            "visual_sources": sorted(VISUAL_TARGET_SOURCES),
            "enabled_sources": sorted(dynamic_target_sources()),
            "visual_priority": True,
            "depth_component_fallback": bool(DEPTH_COMPONENT_TARGET_SOURCE in dynamic_target_sources()),
        },
        "debug_images": {},
        "reason": "",
    }
    ranked: list[tuple[tuple[float, ...], YoloInstance, dict[str, Any], dict[str, Any]]] = []
    for inst in instances:
        accepted, reject_reason, selection_score = dynamic_standpoint_target_filter(inst, instances)
        if not accepted:
            meta["rejected_targets"].append(target_candidate_record(inst, selection_score=selection_score, reject_reason=reject_reason))
            continue
        candidate_meta = compute_grasp_standpoint_candidates(inst.center_3d, robot_pose, allowed_sides)
        record = dynamic_standpoint_target_record(inst, candidate_meta, selection_score)
        meta["valid_targets"].append(record)
        selected = candidate_meta.get("selected")
        penalty = 0 if selected is not None else 1
        candidate_score = float(selected.get("score", 1.0e3)) if selected is not None else 1.0e3
        sort_key = (
            float(dynamic_source_priority(inst.source)),
            float(penalty),
            float(candidate_score),
            float(selection_score.get("total", 0.0)),
            -float(inst.vlm.confidence if inst.vlm else inst.yolo_confidence),
            -float(inst.mask.sum()),
        )
        ranked.append((sort_key, inst, candidate_meta, record))
    if ranked:
        ranked.sort(key=lambda item: item[0])
        _, target, candidate_meta, target_record = ranked[0]
        meta["selected"] = candidate_meta.get("selected")
        meta["selected_target"] = target_record
        meta["candidates"] = candidate_meta.get("candidates", [])
        if meta["selected"] is not None:
            meta["reason"] = "Selected target and continuous RGB-D standpoint with lowest dynamic reachable score."
        else:
            meta["reason"] = "No dynamic candidate passed the right-arm local reach window for any target."
    else:
        target = None
        meta["reason"] = "No visual perception instance satisfied dynamic standpoint target filters."
    meta["debug_images"]["rgb"] = draw_dynamic_standpoint_rgb(rgb, instances, target, meta, intrinsics, t_camera_to_world, out_dir)
    meta["debug_images"]["topdown"] = draw_dynamic_standpoint_topdown(meta, out_dir)
    (out_dir / "dynamic_grasp_standpoint.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if target is None:
        return None, meta["reason"], meta
    return target, f"dynamic_rgbd_standpoint:{meta['reason']}", meta


def choose_reshoot_target(
    reshoot_instances: list[YoloInstance],
    original_identity: TargetIdentity | None,
    original_target: YoloInstance | None,
    out_dir: Path,
) -> tuple[YoloInstance | None, str, dict[str, Any]]:
    meta: dict[str, Any] = {
        "original_identity": target_identity_metadata(original_identity),
        "original_center_3d_world": original_target.center_3d.tolist() if original_target is not None and original_target.center_3d is not None else None,
        "policy": "prefer_same_scene_key_else_same_category_name_nearest_center; no fallback reselect unless explicitly allowed",
        "allow_fallback_reselect": bool(args_cli.allow_reshoot_target_fallback_reselect),
        "selected": None,
        "reason": "",
    }
    if not reshoot_instances:
        meta["reason"] = "reshoot_no_instances"
        (out_dir / "reshoot_target_match.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return None, "No visual perception instance was found after standpoint navigation.", meta
    pool: list[YoloInstance] = []
    if original_identity is not None and original_identity.scene_key:
        pool = [inst for inst in reshoot_instances if inst.scene_key == original_identity.scene_key]
    if not pool and original_target is not None and original_target.vlm is not None:
        pool = [
            inst
            for inst in reshoot_instances
            if inst.vlm is not None
            and inst.vlm.waste_category == original_target.vlm.waste_category
            and (inst.vlm.object_name == original_target.vlm.object_name or not original_target.vlm.object_name)
        ]
    if pool and original_target is not None and original_target.center_3d is not None:
        original_center = np.asarray(original_target.center_3d, dtype=np.float32).reshape(3)
        pool = [inst for inst in pool if inst.center_3d is not None]
        if pool:
            selected = min(pool, key=lambda inst: float(np.linalg.norm(np.asarray(inst.center_3d, dtype=np.float32) - original_center)))
            distance = float(np.linalg.norm(np.asarray(selected.center_3d, dtype=np.float32) - original_center))
            meta["selected"] = target_candidate_record(selected)
            meta["reason"] = f"matched_reshoot_target_distance_m:{distance:.4f}"
            (out_dir / "reshoot_target_match.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            return selected, f"standpoint_reshoot_same_target:{meta['reason']}", meta
    if not bool(args_cli.allow_reshoot_target_fallback_reselect):
        meta["reason"] = "reshoot_same_target_not_found_and_fallback_reselect_disabled"
        (out_dir / "reshoot_target_match.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return None, "Post-navigation reshoot did not recover the same selected target; refusing stale/fallback grasp coordinates.", meta
    selected, reason = choose_target(reshoot_instances, out_dir)
    meta["selected"] = target_candidate_record(selected) if selected is not None else None
    meta["reason"] = f"fallback_choose_target_after_reshoot:{reason}"
    (out_dir / "reshoot_target_match.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return selected, meta["reason"], meta


def current_root_geometric_reachability_record(inst: YoloInstance, robot: Articulation) -> dict[str, Any]:
    center = np.asarray(inst.center_3d, dtype=np.float32).reshape(3) if inst.center_3d is not None else None
    robot_pose = robot_planar_pose(robot)
    if center is None:
        return {
            "instance_index": int(inst.index),
            "source": inst.source,
            "scene_key": inst.scene_key,
            "scene_object_name": inst.scene_object_name,
            "reachable": False,
            "reject_reason": "missing_3d_center",
            "score": 1.0e6,
            "target_local_xy": None,
            "robot_planar_pose": [float(v) for v in robot_pose],
            "target": target_candidate_record(inst),
        }
    local_xy = world_delta_to_local_xy(center[:2] - np.asarray(robot_pose[:2], dtype=np.float32), float(robot_pose[2]))
    target_on_table = bool(
        TABLE_X_LIMITS[0] <= float(center[0]) <= TABLE_X_LIMITS[1]
        and TABLE_Y_LIMITS[0] <= float(center[1]) <= TABLE_Y_LIMITS[1]
    )
    reasons = candidate_reject_reasons(center, local_xy, target_on_table)
    window_risks = right_arm_window_risk(local_xy)
    preferred_local = np.array([RIGHT_ARM_PREFERRED_TARGET_X, RIGHT_ARM_PREFERRED_TARGET_Y], dtype=np.float32)
    local_error = abs(float(local_xy[0]) - float(preferred_local[0])) + 0.45 * abs(float(local_xy[1]) - float(preferred_local[1]))
    z_penalty = 0.0
    if center[2] < TABLE_SURFACE_Z + 0.015:
        z_penalty = float(TABLE_SURFACE_Z + 0.015 - center[2])
    elif center[2] > TABLE_SURFACE_Z + 0.35:
        z_penalty = float(center[2] - (TABLE_SURFACE_Z + 0.35))
    score = local_error + 4.0 * z_penalty + 10.0 * len(reasons)
    return {
        "instance_index": int(inst.index),
        "source": inst.source,
        "scene_key": inst.scene_key,
        "scene_object_name": inst.scene_object_name,
        "object_name": inst.vlm.object_name if inst.vlm else inst.yolo_label,
        "waste_category": inst.vlm.waste_category if inst.vlm else None,
        "reachable": len(reasons) == 0,
        "reject_reason": ";".join(reasons),
        "right_arm_window_risk_only": window_risks,
        "score": float(score),
        "geometric_score": float(local_error),
        "target_world_m": center.astype(float).tolist(),
        "target_local_xy": [float(local_xy[0]), float(local_xy[1])],
        "target_local_z_world_m": float(center[2]),
        "preferred_target_local_xy": [float(preferred_local[0]), float(preferred_local[1])],
        "robot_planar_pose": [float(v) for v in robot_pose],
        "right_arm_window": {
            "target_local_x_m": [float(RIGHT_ARM_TARGET_X_BOUNDS[0]), float(RIGHT_ARM_TARGET_X_BOUNDS[1])],
            "target_local_y_m": [float(RIGHT_ARM_TARGET_Y_BOUNDS[0]), float(RIGHT_ARM_TARGET_Y_BOUNDS[1])],
            "target_z_world_m": [float(TABLE_SURFACE_Z + 0.015), float(TABLE_SURFACE_Z + 0.35)],
        },
        "target": target_candidate_record(inst),
    }


def select_reshoot_ik_pool(
    instances: list[YoloInstance],
    robot: Articulation,
    out_dir: Path,
) -> tuple[list[YoloInstance], dict[str, Any]]:
    top_k = max(1, int(args_cli.mind_sort_reachability_ik_top_k))
    ranked: list[tuple[tuple[float, float, float, float], YoloInstance, dict[str, Any]]] = []
    for inst in instances:
        record = current_root_geometric_reachability_record(inst, robot)
        reachable_penalty = 0.0 if bool(record.get("reachable", False)) else 1.0
        source_priority = float(dynamic_source_priority(inst.source))
        confidence = float(inst.vlm.confidence if inst.vlm else inst.yolo_confidence)
        mask_area = float(inst.mask.sum()) if inst.mask is not None else 0.0
        sort_key = (
            reachable_penalty,
            source_priority,
            float(record.get("score", 1.0e6)),
            -confidence - 1.0e-7 * mask_area,
        )
        ranked.append((sort_key, inst, record))
    ranked.sort(key=lambda item: item[0])
    pool = [item[1] for item in ranked[:top_k]]
    meta = {
        "policy": "current_root_geometry_top_k_before_expensive_right_arm_ik",
        "top_k": int(top_k),
        "robot_planar_pose": robot_planar_pose(robot),
        "selected_for_ik": [item[2] for item in ranked[:top_k]],
        "all_ranked": [item[2] for item in ranked],
    }
    (out_dir / "reshoot_ik_candidate_pool.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return pool, meta


def run_pregrasp_standpoint_nav(
    sim: SimulationContext,
    scene: InteractiveScene,
    run_dir: Path,
    cycle_dir: Path,
    standpoint_plan_meta: dict[str, Any],
) -> dict[str, Any]:
    robot: Articulation = scene["robot"]
    tracker = Task319StateTracker(sim.get_physics_dt(), sim.device)
    selected = standpoint_plan_meta.get("selected") or {}
    metadata: dict[str, Any] = {
        "enabled": bool(args_cli.grasp_standpoint_nav),
        "mode": "pre_grasp_table_standpoint_nav",
        "nav_backend": args_cli.nav_backend,
        "selected_waypoint": selected.get("waypoint"),
        "target_pose": selected.get("pose"),
        "initial_pose": robot_planar_pose(robot),
        "final_pose": None,
        "navigation": None,
        "success": False,
        "reason": "",
        "wheel_drive": wheel_drive_metadata(),
        "nav2_stack": {
            "auto_start": bool(args_cli.start_nav2_stack),
            "script": args_cli.nav2_stack_script,
            "startup_s": float(args_cli.nav2_stack_startup_s),
            "log": "",
        },
    }
    if not bool(args_cli.grasp_standpoint_nav):
        metadata["reason"] = "not_requested"
        return metadata
    if args_cli.nav_backend != "nav2":
        metadata["reason"] = f"Unsupported nav backend: {args_cli.nav_backend}"
        return metadata
    if not selected or selected.get("pose") is None:
        metadata["reason"] = "No selected grasp standpoint."
        return metadata
    if bool(args_cli.grasp_standpoint_require_geometric_reachable) and not bool(selected.get("geometric_reachable", False)):
        metadata["reason"] = "Selected grasp standpoint is not geometrically reachable for the right arm."
        (cycle_dir / "pre_grasp_standpoint_nav.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return metadata
    wheel_ids, wheel_names = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
    metadata["wheel_joints"] = wheel_names
    if len(wheel_ids) != 4:
        metadata["reason"] = f"Pre-grasp standpoint navigation requires 4 wheel joints, found {wheel_names}."
        (cycle_dir / "pre_grasp_standpoint_nav.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return metadata
    locked_joint_target = robot.data.joint_pos.clone()
    bridge = ExternalRos2NavBridgeClient(run_dir)
    nav2_stack_process: subprocess.Popen[Any] | None = None
    tracker.enter("REST", mode="pre_grasp_table_standpoint_nav", nav_backend="nav2")
    try:
        bridge.start()
        nav2_stack_process, nav2_stack_log = start_nav2_stack(run_dir)
        if nav2_stack_log is not None:
            metadata["nav2_stack"]["log"] = str(nav2_stack_log)
        sim_dt = sim.get_physics_dt()
        for _ in range(args_cli.warmup_steps):
            vx, wz = bridge.exchange(robot)
            vx, wz = limit_external_nav_cmd(vx, wz, sim_dt)
            hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
            apply_wheel_velocity(robot, wheel_ids, vx, wz, sim_dt)
            stabilize_robot_base_for_nav(robot)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim_dt)
            record_video_frame(scene)
            gui_playback_tick(sim_dt)
            tracker.tick()
        target_pose = tuple(float(v) for v in selected["pose"])
        nav_result = run_nav2_goal(
            sim,
            scene,
            robot,
            bridge,
            wheel_ids,
            locked_joint_target,
            tracker,
            "NAV_TO_TABLE_STANDPOINT",
            f"pre_grasp_{selected.get('waypoint', 'standpoint')}",
            target_pose,
        )
        metadata["navigation"] = nav_result
        metadata["success"] = bool(nav_result.get("success", False))
        metadata["reason"] = "Pre-grasp standpoint Nav2 goal completed." if metadata["success"] else f"Nav2 failed: {nav_result.get('status')}"
        tracker.enter("DONE" if metadata["success"] else "RECOVER", reason=metadata["reason"])
    except Exception as exc:
        metadata["success"] = False
        metadata["reason"] = repr(exc)
        tracker.enter("RECOVER", reason=repr(exc))
    finally:
        bridge.close()
        stop_external_process(nav2_stack_process, "Nav2 stack")
        metadata["final_pose"] = robot_planar_pose(robot)
        metadata["nav2_failure_hints"] = nav2_failure_hints(run_dir)
        metadata["fsm_trace"] = tracker.trace
        (cycle_dir / "pre_grasp_standpoint_nav.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def write_cycle(sim: SimulationContext, scene: InteractiveScene, run_dir: Path, cycle_index: int) -> bool:
    cycle_dir = run_dir / f"cycle_{cycle_index:04d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    rgb, depth_m, intrinsics, t_camera_to_world = capture_head_camera(scene)
    Image.fromarray(rgb).save(cycle_dir / "head_rgb.png")
    camera_self_occlusion_guard_meta = save_camera_self_occlusion_guard(rgb, cycle_dir)
    save_depth_images(depth_m, cycle_dir)
    if args_cli.save_legacy_yolo_debug:
        yolo_all_debug_meta = save_yolo_all_overlay(rgb, cycle_dir)
    else:
        yolo_all_debug_meta = {
            "enabled": False,
            "reason": "legacy_full_frame_yolo_debug_disabled_by_default",
            "official_visual_records": "v28_original/",
        }
    if not getattr(args_cli, "headless", False):
        print("[INFO] Running visual model planning. The Isaac viewport remains responsive while model workers run.", flush=True)
        gui_playback_tick(0.0)
    yolo_instances = run_active_perception_model(rgb, cycle_dir)
    scene_guided_instances: list[YoloInstance] = []
    explicit_scene_target = explicit_target_scene_match()
    if explicit_scene_target is None and args_cli.debug_force_scene_grasp_object:
        explicit_scene_target = scene_key_and_name_for_alias(args_cli.debug_force_scene_grasp_object)
    if explicit_scene_target is not None:
        guided = scene_guided_depth_instance(scene, depth_m, intrinsics, t_camera_to_world, *explicit_scene_target)
        if guided is not None:
            scene_guided_instances.append(guided)
    instances = [*yolo_instances]
    depth_instances = detect_tabletop_depth_components(scene, rgb, depth_m, intrinsics, t_camera_to_world, cycle_dir) if centroid_fallback_enabled() else []
    all_debug_instances = [*scene_guided_instances, *yolo_instances, *depth_instances]
    attach_rgbd_geometry_to_instances(scene, instances, depth_m, intrinsics, t_camera_to_world)
    right_arm_reachability_records = evaluate_right_arm_target_reachability(sim, scene, yolo_instances, cycle_dir)
    if args_cli.perception_source == "v28_original":
        vlm_instances = []
    else:
        raise RuntimeError(f"Unsupported retired perception source: {args_cli.perception_source}")
    target, target_reason = choose_target(instances, cycle_dir)
    if bool(args_cli.debug_cube_grasp_demo) and bool(args_cli.debug_cube_rgbd_target) and scene_guided_instances:
        target = scene_guided_instances[0]
        target_reason = "debug_cube_rgbd_target:scene_guided_head_rgbd_center"
        instances = [target, *instances]
        all_debug_instances = [target, *all_debug_instances]
    debug_force_target: YoloInstance | None = None
    debug_force_selected_grasps: list[SelectedGrasp] = []
    debug_force_meta: dict[str, Any] = {"enabled": False}
    if args_cli.debug_force_scene_grasp_object:
        debug_force_target, debug_force_selected_grasps, debug_force_meta = debug_forced_scene_grasp(
            scene,
            rgb,
            intrinsics,
            t_camera_to_world,
            args_cli.debug_force_scene_grasp_object,
        )
        if debug_force_target is not None and debug_force_selected_grasps:
            target = debug_force_target
            target_reason = f"debug_force_scene_grasp_object:{args_cli.debug_force_scene_grasp_object}"
            instances = [debug_force_target, *instances]
            all_debug_instances = [debug_force_target, *all_debug_instances]
    dynamic_grasp_standpoint_meta: dict[str, Any] = {"enabled": False, "reason": "not_requested"}
    use_preset_standpoint_plan = True
    if (
        args_cli.dynamic_grasp_standpoint_nav
        and not args_cli.debug_force_scene_grasp_object
        and not args_cli.debug_cube_grasp_demo
    ):
        dynamic_target, dynamic_target_reason, dynamic_grasp_standpoint_meta = plan_dynamic_grasp_standpoint(
            scene,
            instances,
            rgb,
            intrinsics,
            t_camera_to_world,
            cycle_dir,
        )
        if dynamic_target is not None and dynamic_grasp_standpoint_meta.get("selected") is not None:
            target = dynamic_target
            target_reason = dynamic_target_reason
            use_preset_standpoint_plan = False
        elif not bool(args_cli.dynamic_standpoint_fallback_to_preset):
            target = dynamic_target
            target_reason = dynamic_target_reason
            use_preset_standpoint_plan = False
    frozen_target_identity = freeze_target_identity(target)
    if use_preset_standpoint_plan:
        grasp_standpoint_plan_meta = plan_preset_grasp_standpoint(scene, target, rgb, intrinsics, t_camera_to_world, cycle_dir)
        if dynamic_grasp_standpoint_meta.get("enabled", False):
            grasp_standpoint_plan_meta["fallback_from_dynamic"] = True
            grasp_standpoint_plan_meta["dynamic_failure_reason"] = dynamic_grasp_standpoint_meta.get("reason", "")
    else:
        grasp_standpoint_plan_meta = dynamic_grasp_standpoint_meta
    pre_grasp_standpoint_nav_meta: dict[str, Any] = {
        "enabled": bool(args_cli.grasp_standpoint_nav),
        "success": False,
        "reason": "not_requested" if not args_cli.grasp_standpoint_nav else "not_run",
    }
    standpoint_reshoot_meta: dict[str, Any] = {"enabled": False}
    standpoint_reshoot_required_failed = False
    if (
        args_cli.grasp_standpoint_nav
        and (args_cli.execute_grasp or args_cli.standpoint_nav_only)
        and target is not None
        and not args_cli.debug_force_scene_grasp_object
        and not args_cli.debug_cube_grasp_demo
    ):
        pre_grasp_standpoint_nav_meta = run_pregrasp_standpoint_nav(sim, scene, run_dir, cycle_dir, grasp_standpoint_plan_meta)
        if bool(pre_grasp_standpoint_nav_meta.get("success", False)) and bool(args_cli.grasp_standpoint_reshoot):
            reshoot_dir = cycle_dir / "standpoint_reshoot"
            reshoot_dir.mkdir(parents=True, exist_ok=True)
            rgb2, depth2_m, intrinsics2, t_camera_to_world2 = capture_head_camera(scene)
            Image.fromarray(rgb2).save(reshoot_dir / "head_rgb.png")
            save_depth_images(depth2_m, reshoot_dir)
            reshoot_instances = run_active_perception_model(rgb2, reshoot_dir)
            attach_rgbd_geometry_to_instances(scene, reshoot_instances, depth2_m, intrinsics2, t_camera_to_world2)
            if args_cli.perception_source != "v28_original":
                raise RuntimeError(f"Unsupported retired perception source: {args_cli.perception_source}")
            target2, target_reason2, reshoot_match_meta = choose_reshoot_target(reshoot_instances, frozen_target_identity, target, reshoot_dir)
            standpoint_reshoot_meta = {
                "enabled": True,
                "instance_count": len(reshoot_instances),
                "selected": target_candidate_record(target2) if target2 is not None else None,
                "target_reason": target_reason2,
                "match": reshoot_match_meta,
                "rgb": str(reshoot_dir / "head_rgb.png"),
                "directory": str(reshoot_dir),
            }
            if target2 is not None:
                rgb = rgb2
                depth_m = depth2_m
                intrinsics = intrinsics2
                t_camera_to_world = t_camera_to_world2
                yolo_instances = reshoot_instances
                instances = [*reshoot_instances]
                all_debug_instances = [*reshoot_instances]
                right_arm_reachability_records = evaluate_right_arm_target_reachability(sim, scene, yolo_instances, reshoot_dir)
                target = target2
                target_reason = target_reason2
                frozen_target_identity = freeze_target_identity(target)
                grasp_standpoint_plan_meta["reshoot_replaced_grasp_input"] = True
                grasp_standpoint_plan_meta["reshoot_dir"] = str(reshoot_dir)
            else:
                standpoint_reshoot_required_failed = True
                grasp_standpoint_plan_meta["reshoot_replaced_grasp_input"] = False
                grasp_standpoint_plan_meta["reshoot_required_failed"] = True
                grasp_standpoint_plan_meta["reshoot_failure_reason"] = target_reason2
        grasp_standpoint_plan_meta["navigation"] = pre_grasp_standpoint_nav_meta
        grasp_standpoint_plan_meta["reshoot"] = standpoint_reshoot_meta
    target_identity_ok, target_identity_reason = target_matches_identity(target, frozen_target_identity) if frozen_target_identity is not None else (False, "no_target_selected")
    target_pose_debug_meta = write_target_pose_debug(scene, rgb, target, intrinsics, t_camera_to_world, cycle_dir)
    scene_guided_alignment_meta = write_scene_guided_rgbd_alignment(scene, scene_guided_instances, intrinsics, t_camera_to_world, cycle_dir)
    rgbd_vs_truth_meta = write_rgbd_vs_truth_debug(
        scene,
        rgb,
        target,
        intrinsics,
        t_camera_to_world,
        cycle_dir,
        label="cycle_selected_target",
    )
    target_pose_blocked = bool(target is not None and not target_pose_debug_meta.get("allow_execution", True))
    standpoint_nav_blocked = bool(
        args_cli.grasp_standpoint_nav
        and args_cli.execute_grasp
        and not args_cli.debug_force_scene_grasp_object
        and not args_cli.debug_cube_grasp_demo
        and not bool(pre_grasp_standpoint_nav_meta.get("success", False))
    )
    reshoot_blocked = bool(args_cli.execute_grasp and standpoint_reshoot_required_failed)
    pre_execution_blocked = bool(target_pose_blocked or standpoint_nav_blocked or reshoot_blocked)
    if debug_force_selected_grasps and target_pose_blocked:
        target_pose_debug_meta["debug_force_scene_grasp_override"] = True
        target_pose_debug_meta["allow_execution"] = True
        target_pose_blocked = False
        pre_execution_blocked = bool(standpoint_nav_blocked)
        (cycle_dir / "target_pose_debug.json").write_text(json.dumps(target_pose_debug_meta, indent=2, ensure_ascii=False))
    if pre_execution_blocked:
        blank = np.zeros(depth_m.shape, dtype=np.uint8)
        if args_cli.save_legacy_grasp_debug:
            if target is not None:
                Image.fromarray(target.mask.astype(np.uint8) * 255).save(cycle_dir / "target_mask.png")
            else:
                Image.fromarray(blank).save(cycle_dir / "target_mask.png")
            Image.fromarray(blank).save(cycle_dir / "graspnet_workspace_mask.png")
            blocked_overlay = Image.fromarray(rgb.copy())
            ImageDraw.Draw(blocked_overlay).text((12, 12), "Pre-execution debug blocked GraspNet/execution", fill=(255, 0, 0))
            blocked_overlay.save(cycle_dir / "graspnet_overlay.png")
            blocked_overlay.save(cycle_dir / "selected_grasp_overlay.png")
        block_hint = "Check target_pose_debug.jpg/json: camera-derived target center disagrees with simulator object truth."
        block_error = "Target pose debug blocked learned grasp backend before execution."
        if standpoint_nav_blocked:
            block_hint = "Check planned_grasp_standpoint.json and pre_grasp_standpoint_nav.json: robot did not reach a valid pre-grasp standpoint."
            block_error = f"Pre-grasp standpoint navigation blocked learned grasp backend before execution: {pre_grasp_standpoint_nav_meta.get('reason', '')}"
        if reshoot_blocked:
            block_hint = "Check cycle_*/standpoint_reshoot/reshoot_target_match.json: post-navigation current-frame perception did not recover the same target."
            block_error = f"Post-navigation reshoot blocked execution to avoid stale coordinates: {standpoint_reshoot_meta.get('target_reason', '')}"
        selected_grasps = []
        graspnet_meta = {
            "enabled": not args_cli.skip_graspnet,
            "backend": args_cli.grasp_backend,
            "candidate_count": 0,
            "filtered_count": 0,
            "selected": None,
            "selected_candidates": [],
            "candidate_pool_count": 0,
            "source": "",
            "fallback": None,
            "centroid_fallback_enabled": centroid_fallback_enabled(),
            "error": block_error,
            "hint": block_hint,
            "mask_filter": {},
            "tcp_alignment_mode": f"{args_cli.grasp_backend}_grasp_to_kuavo_tcp_to_wrist",
            "grasp_to_tcp_calibration": active_grasp_to_tcp_metadata(),
            "gripper_tcp_offset_m": gripper_tcp_offset_m().tolist(),
            "gripper_local_tcp_offset_m": gripper_local_tcp_offset_m().tolist(),
            "right_gripper_mount_rpy_rad": list(RIGHT_GRIPPER_INLINE_MOUNT_RPY),
            "grasp_approach_axis_local": GRASP_APPROACH_AXIS_LOCAL.tolist(),
            "grasp_frame_convention": dict(GRASP_FRAME_CONVENTION),
            "target_pose_debug": target_pose_debug_meta,
            "grasp_standpoint_plan": grasp_standpoint_plan_meta,
            "target_identity": target_identity_metadata(frozen_target_identity),
            "target_identity_consistency": {
                "selected": target_identity_ok,
                "grasp_input": False,
                "selected_grasp": False,
                "execution": False,
                "consistent": False,
                "block_reason": "standpoint_nav_blocked" if standpoint_nav_blocked else "target_pose_debug_blocked",
            },
        }
        if reshoot_blocked:
            graspnet_meta["target_identity_consistency"]["block_reason"] = "post_navigation_reshoot_target_missing"
    else:
        if debug_force_selected_grasps:
            selected_grasps = debug_force_selected_grasps
            graspnet_meta = dict(debug_force_meta.get("grasp", {}))
        else:
            selected_grasps, graspnet_meta = wait_for_model_result(args_cli.grasp_backend, run_graspnet_stage, rgb, depth_m, intrinsics, t_camera_to_world, target, cycle_dir, frozen_target_identity)
    selected = selected_grasps[0] if selected_grasps else None
    grasp_identity_meta = graspnet_meta.get("target_identity_consistency", {}) if isinstance(graspnet_meta, dict) else {}
    selected_grasp_identity_ok, selected_grasp_identity_reason = selected_grasp_matches_identity(selected, frozen_target_identity) if selected is not None else (False, "missing_selected_grasp")
    if not target_identity_ok:
        identity_block_reason = target_identity_reason
    elif not selected_grasp_identity_ok:
        identity_block_reason = str(grasp_identity_meta.get("block_reason") or selected_grasp_identity_reason)
    else:
        identity_block_reason = str(grasp_identity_meta.get("block_reason") or "")
    target_identity_consistency = {
        "selected": bool(target_identity_ok),
        "vlm": bool(target is not None and target.vlm is not None and not target.vlm.error),
        "grasp_input": bool(grasp_identity_meta.get("grasp_input", False)),
        "selected_grasp": bool(grasp_identity_meta.get("selected_grasp", selected_grasp_identity_ok)),
        "execution": False,
        "consistent": False,
        "block_reason": identity_block_reason,
        "target_identity": target_identity_metadata(frozen_target_identity),
    }
    target_identity_consistency["consistent"] = bool(
        target_identity_consistency["selected"]
        and target_identity_consistency["vlm"]
        and target_identity_consistency["grasp_input"]
        and target_identity_consistency["selected_grasp"]
    )
    calibration_debug_meta = write_calibration_debug_outputs(scene, selected, rgb, intrinsics, t_camera_to_world, cycle_dir)
    gui_playback_tick(0.0)
    execution_meta = {"enabled": False}
    if args_cli.execute_grasp and pre_execution_blocked:
        execution_meta = {
            "enabled": True,
            "success": False,
            "grasp_success": False,
            "reason": (
                f"Pre-grasp standpoint navigation blocked execution: {pre_grasp_standpoint_nav_meta.get('reason', '')}"
                if standpoint_nav_blocked
                else f"Post-navigation reshoot blocked execution: {standpoint_reshoot_meta.get('target_reason', '')}"
                if reshoot_blocked
                else "Target pose debug blocked execution before GraspNet/IK."
            ),
            "attempts": [],
            "fsm_trace": [],
            "target_pose_debug": target_pose_debug_meta,
            "grasp_standpoint_plan": grasp_standpoint_plan_meta,
            "target_identity_consistency": target_identity_consistency,
        }
    elif args_cli.execute_grasp and not target_identity_consistency["consistent"]:
        execution_meta = {
            "enabled": True,
            "success": False,
            "grasp_success": False,
            "reason": f"Target identity consistency blocked execution: {target_identity_consistency['block_reason']}",
            "attempts": [],
            "fsm_trace": [],
            "target_identity_consistency": target_identity_consistency,
        }
    elif args_cli.execute_grasp:
        execution_meta = execute_grasp(
            sim,
            scene,
            selected_grasps,
            cycle_dir,
            calibration_debug_context={
                "camera_to_world": t_camera_to_world,
                "intrinsics": intrinsics,
                "projection_image": calibration_debug_meta.get("projection_image", ""),
            },
        )
        execution_scene_key = execution_meta.get("target_scene_key")
        expected_scene_key = frozen_target_identity.scene_key if frozen_target_identity is not None else None
        execution_identity_ok = bool(expected_scene_key is None or execution_scene_key == expected_scene_key)
        target_identity_consistency["execution"] = execution_identity_ok
        if not execution_identity_ok:
            target_identity_consistency["consistent"] = False
            target_identity_consistency["block_reason"] = f"execution_scene_key_mismatch:{execution_scene_key}!={expected_scene_key}"
        execution_meta["target_identity_consistency"] = target_identity_consistency
    update_target_failure_memory(target, execution_meta)
    sort_nav_meta: dict[str, Any] = {"enabled": bool(args_cli.enable_sort_nav), "success": False, "reason": "not_requested" if not args_cli.enable_sort_nav else "grasp_not_success"}
    if args_cli.enable_sort_nav and args_cli.execute_grasp:
        if bool(execution_meta.get("grasp_success", False)) and target is not None and target.vlm is not None:
            sort_nav_meta = run_sort_nav_after_grasp(sim, scene, run_dir, cycle_dir, target.vlm.waste_category)
        else:
            sort_nav_meta = {
                "enabled": True,
                "success": False,
                "reason": "Post-grasp sort navigation skipped because grasp_success is false or target category is missing.",
                "grasp_success": bool(execution_meta.get("grasp_success", False)),
            }
        execution_meta["sort_navigation"] = sort_nav_meta

    valid_vlm_count = sum(1 for inst in yolo_instances if inst.vlm is not None and not inst.vlm.error and inst.vlm.confidence >= args_cli.vlm_min_conf)
    planned_grasp_source = str(graspnet_meta.get("source") or (selected.source if selected is not None else ""))
    execution_grasp_source = str(execution_meta.get("grasp_source") or "")
    grasp_source = execution_grasp_source or planned_grasp_source
    graspnet_has_valid_candidate = planned_grasp_source == args_cli.grasp_backend and int(graspnet_meta.get("filtered_count", 0)) > 0
    execution_grasp_success = bool(execution_meta.get("grasp_success", False))
    strict_success = bool(
        target is not None
        and target.source in VISUAL_TARGET_SOURCES
        and target.vlm is not None
        and not target.vlm.error
        and target.vlm.confidence >= args_cli.vlm_min_conf
        and not args_cli.skip_yolo
        and not args_cli.skip_vlm
        and not args_cli.skip_graspnet
        and execution_grasp_source == args_cli.grasp_backend
        and int(graspnet_meta.get("filtered_count", 0)) > 0
        and execution_grasp_success
    )
    fallback_success = bool(execution_grasp_source == "fallback" and execution_grasp_success)
    vision_modules = {
        "yolo": {
            "enabled": not args_cli.skip_yolo,
            "status": "disabled" if args_cli.skip_yolo else ("ok" if len(yolo_instances) > 0 else "no_instances"),
            "instance_count": len(yolo_instances),
            "weights": args_cli.yolo_weights,
        },
        "vlm": {
            "enabled": not args_cli.skip_vlm,
            "status": "disabled" if args_cli.skip_vlm else ("ok" if valid_vlm_count > 0 else "no_valid_instances"),
            "valid_instance_count": valid_vlm_count,
            "model": args_cli.vlm_model,
        },
        "graspnet": {
            "enabled": not args_cli.skip_graspnet,
            "backend": args_cli.grasp_backend,
            "status": (
                "disabled"
                if args_cli.skip_graspnet
                else ("ok" if graspnet_has_valid_candidate else "no_valid_grasp")
            ),
            "candidate_count": int(graspnet_meta.get("candidate_count", 0)),
            "filtered_count": int(graspnet_meta.get("filtered_count", 0)),
            "error": graspnet_meta.get("error", ""),
            "hint": graspnet_meta.get("hint", ""),
        },
        "grasp_backend": {
            "enabled": not args_cli.skip_graspnet,
            "backend": args_cli.grasp_backend,
            "status": (
                "disabled"
                if args_cli.skip_graspnet
                else ("ok" if graspnet_has_valid_candidate else "no_valid_grasp")
            ),
            "candidate_count": int(graspnet_meta.get("candidate_count", 0)),
            "filtered_count": int(graspnet_meta.get("filtered_count", 0)),
            "error": graspnet_meta.get("error", ""),
            "hint": graspnet_meta.get("hint", ""),
        },
    }
    metadata = {
        "cycle_index": cycle_index,
        "camera_source": "head_rgbd",
        "strict_model_chain": bool(args_cli.strict_model_chain),
        "strict_success": strict_success,
        "fallback_success": fallback_success,
        "target_source": target.source if target else None,
        "grasp_backend": args_cli.grasp_backend,
        "grasp_source": grasp_source or None,
        "planned_grasp_source": planned_grasp_source or None,
        "execution_grasp_source": execution_grasp_source or None,
        "grasp_candidate_count": int(graspnet_meta.get("candidate_count", 0)),
        "grasp_filtered_count": int(graspnet_meta.get("filtered_count", 0)),
        "graspnet_candidate_count": int(graspnet_meta.get("candidate_count", 0)),
        "graspnet_filtered_count": int(graspnet_meta.get("filtered_count", 0)),
        "vision_modules": vision_modules,
        "yolo_all_debug": yolo_all_debug_meta,
        "perception_source": args_cli.perception_source,
        "target_source_policy": "visual_perception_only",
        "target_geometry_policy": "visual_mask_current_head_rgbd_geometric_center",
        "category_source_policy": "v28_qwen_vlm_only",
        "intrinsics": intrinsics.tolist(),
        "camera_to_world": t_camera_to_world.tolist(),
        "camera_self_occlusion_guard": camera_self_occlusion_guard_meta,
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth_m.shape),
        "yolo_instance_count": len(yolo_instances),
        "target_candidate_count": len(instances),
        "total_instance_count": len(all_debug_instances),
        "target_pose_debug": target_pose_debug_meta,
        "rgbd_vs_truth_debug": rgbd_vs_truth_meta,
        "scene_guided_rgbd_alignment": scene_guided_alignment_meta,
        "target_pose_debug_blocked": target_pose_blocked,
        "grasp_standpoint_plan": grasp_standpoint_plan_meta,
        "dynamic_grasp_standpoint": dynamic_grasp_standpoint_meta,
        "pre_grasp_standpoint_nav": pre_grasp_standpoint_nav_meta,
        "standpoint_reshoot": standpoint_reshoot_meta,
        "standpoint_nav_blocked": standpoint_nav_blocked,
        "target_identity": target_identity_metadata(frozen_target_identity),
        "target_identity_consistency": target_identity_consistency,
        "target_failure_memory": dict(_RECENT_TARGET_FAILURES),
        "right_arm_reachability": right_arm_reachability_records,
        "target_selection": {
            "reason": target_reason,
            "selected_yolo_index": target.index if target else None,
            "target_instance_id": frozen_target_identity.instance_id if frozen_target_identity else None,
            "target_mask_hash": frozen_target_identity.mask_hash if frozen_target_identity else None,
            "source": target.source if target else None,
            "scene_key": target.scene_key if target else None,
            "scene_object_name": target.scene_object_name if target else None,
            "object_name": target.vlm.object_name if target and target.vlm else None,
            "waste_category": target.vlm.waste_category if target and target.vlm else None,
            "category_source": "v28_qwen_vlm" if target and target.vlm and not target.vlm.error else None,
            "vlm_confidence": target.vlm.confidence if target and target.vlm else None,
            "mask_pixels": int(target.mask.sum()) if target else 0,
            "center_3d_world": target.center_3d.tolist() if target is not None and target.center_3d is not None else None,
            "reachability": instance_reachability_metadata(target),
            "requires_reachable_target": require_reachable_auto_target(),
        },
        "scene_guided_target_count": len(scene_guided_instances),
        "depth_component_count": len(depth_instances),
        "depth_components_debug_only": True,
        "grasp": graspnet_meta,
        "graspnet": graspnet_meta,
        "calibration_debug": calibration_debug_meta,
        "execution": execution_meta,
        "sort_navigation": sort_nav_meta,
        "observer_video": _VIDEO_RECORDER.metadata() if _VIDEO_RECORDER is not None else {"enabled": False},
    }
    (cycle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    (run_dir / "latest_cycle.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(
        f"[INFO] Saved cycle {cycle_index:04d}: camera=head_rgbd "
        f"perception={args_cli.perception_source}:{len(yolo_instances)} target={metadata['target_selection']['object_name']} "
        f"{args_cli.grasp_backend}={graspnet_meta.get('filtered_count', 0)}/{graspnet_meta.get('candidate_count', 0)} "
        f"execute={execution_meta.get('success', False)} strict={strict_success} -> {cycle_dir}",
        flush=True,
    )
    return bool(args_cli.execute_grasp)


def motion_only_categories() -> list[str]:
    if args_cli.motion_test_all_categories:
        return list(MOTION_SORT_CATEGORIES)
    return [canonical_category(args_cli.motion_test_category or args_cli.target_category or "其他垃圾")]


def run_motion_stub_grasp(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    gripper: AttachedParallelGripper,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
) -> dict[str, Any]:
    tracker.enter("PICK_STUB_OR_HOLD_OBJECT", mode="motion_only_stub")
    sim_dt = sim.get_physics_dt()
    steps = max(1, int(args_cli.grasp_steps))
    carry_target = arm_pose_target(robot, locked_joint_target, RIGHT_ARM_CARRY_JOINT_POS)
    close_steps = max(1, min(steps, int(round(0.35 * steps))))
    raise_steps = max(1, steps - close_steps)
    for step in range(steps):
        if step < close_steps:
            motion_target = locked_joint_target
        else:
            alpha = min_jerk((step - close_steps + 1) / raise_steps)
            motion_target = locked_joint_target + alpha * (carry_target - locked_joint_target)
        hold_non_wheel_joints(robot, motion_target, wheel_ids)
        apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0)
        gripper.close()
        stabilize_robot_base_for_nav(robot)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        tracker.tick()
    locked_joint_target.copy_(carry_target)
    return {"success": True, "mode": "motion_only_stub", "steps": steps, "arm_pose": "right_arm_carry"}


def run_motion_stub_drop(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    gripper: AttachedParallelGripper,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
) -> dict[str, Any]:
    tracker.enter("DROP", mode="motion_only_stub")
    sim_dt = sim.get_physics_dt()
    steps = max(1, int(args_cli.drop_steps))
    natural_target = arm_pose_target(robot, locked_joint_target, ARM_NATURAL_DOWN_JOINT_POS)
    for step in range(steps):
        alpha = min_jerk((step + 1) / steps)
        motion_target = locked_joint_target + alpha * (natural_target - locked_joint_target)
        hold_non_wheel_joints(robot, motion_target, wheel_ids)
        apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0)
        gripper.set_width(gripper.limits.max_width_m)
        stabilize_robot_base_for_nav(robot)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        tracker.tick()
    locked_joint_target.copy_(natural_target)
    return {"success": True, "mode": "motion_only_stub", "steps": steps, "arm_pose": "natural_down"}


def run_nav2_goal(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    bridge: ExternalRos2NavBridgeClient,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
    state_name: str,
    label: str,
    target_pose: tuple[float, float, float],
    step_hook: Callable[[], None] | None = None,
    final_dock: bool = False,
) -> dict[str, Any]:
    tracker.enter(state_name, backend="nav2", label=label, target_pose=target_pose)
    process = nav2_goal_process(target_pose, label)
    result = step_nav2_until_goal_done(
        sim,
        scene,
        robot,
        bridge,
        process,
        wheel_ids,
        locked_joint_target,
        tracker,
        target_pose,
        step_hook=step_hook,
    )
    if bool(final_dock) and bool(args_cli.nav_final_dock) and bool(result.get("success", False)):
        raw_success = bool(result.get("success", False))
        dock = final_dock_to_pose(sim, scene, robot, wheel_ids, locked_joint_target, tracker, target_pose, step_hook=step_hook)
        result["nav2_raw_final_pose"] = result.get("final_pose")
        result["nav2_raw_position_error_m"] = result.get("position_error_m")
        result["nav2_raw_yaw_error_rad"] = result.get("yaw_error_rad")
        result["final_dock"] = dock
        result["final_pose"] = dock.get("final_pose", result.get("final_pose"))
        result["position_error_m"] = dock.get("position_error_m", result.get("position_error_m"))
        result["yaw_error_rad"] = dock.get("yaw_error_rad", result.get("yaw_error_rad"))
        result["success"] = bool(dock.get("success", False))
        if bool(dock.get("success", False)):
            result["status"] = "SUCCEEDED_WITH_FINAL_DOCK"
        else:
            result["status"] = "SUCCEEDED_FINAL_DOCK_PARTIAL"
    locked_joint_target.copy_(robot.data.joint_pos.clone())
    tracker.event("nav2_goal_result", label=label, **result)
    return {"label": label, "state": state_name, **result}


def nav2_goal_reached_or_soft_docked(nav_result: dict[str, Any]) -> bool:
    if bool(nav_result.get("success", False)):
        return True
    return str(nav_result.get("status", "")) == "SUCCEEDED_FINAL_DOCK_PARTIAL"


def mind_sort_perception_pass(
    scene: InteractiveScene,
    out_dir: Path,
    *,
    completed_scene_keys: set[str] | None = None,
    prefix: str = "observe",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb, depth_m, intrinsics, t_camera_to_world = capture_head_camera(scene)
    Image.fromarray(rgb).save(out_dir / f"{prefix}_rgb.png")
    save_depth_images(depth_m, out_dir, prefix=prefix)
    instances = run_active_perception_model(rgb, out_dir)
    attach_rgbd_geometry_to_instances(scene, instances, depth_m, intrinsics, t_camera_to_world)
    depth_instances = detect_tabletop_depth_components(scene, rgb, depth_m, intrinsics, t_camera_to_world, out_dir)
    seen_scene_keys = {inst.scene_key for inst in instances if inst.scene_key}
    for depth_inst in depth_instances:
        if depth_inst.scene_key and depth_inst.scene_key in seen_scene_keys:
            continue
        instances.append(depth_inst)
        if depth_inst.scene_key:
            seen_scene_keys.add(depth_inst.scene_key)
    completed = completed_scene_keys or set()
    excluded_scene_keys = set(completed)
    if not bool(args_cli.debug_cube_grasp_demo):
        excluded_scene_keys.add("debug_cube")
    filtered: list[YoloInstance] = []
    rejected: list[dict[str, Any]] = []
    for inst in instances:
        candidate_scene_key = inst.scene_key
        candidate_scene_name = inst.scene_object_name
        match_distance = None
        if (not candidate_scene_key or candidate_scene_key not in scene.keys()) and inst.center_3d is not None:
            candidate_scene_key, candidate_scene_name, match_distance = locate_target_rigid_object(scene, inst.center_3d)
            if match_distance > max(0.10, scene_target_radius(candidate_scene_key) + 0.04):
                candidate_scene_key = None
                candidate_scene_name = None
        if candidate_scene_key in excluded_scene_keys:
            rejected.append(
                {
                    "index": int(inst.index),
                    "label": inst.yolo_label,
                    "scene_key": candidate_scene_key,
                    "scene_object_name": candidate_scene_name,
                    "match_distance_m": None if match_distance is None else float(match_distance),
                    "reason": "completed_or_debug_scene_key",
                }
            )
            continue
        filtered.append(inst)
    return {
        "rgb": rgb,
        "depth_m": depth_m,
        "intrinsics": intrinsics,
        "camera_to_world": t_camera_to_world,
        "instances": instances,
        "depth_instances": depth_instances,
        "filtered_instances": filtered,
        "completed_scene_keys": sorted(completed),
        "excluded_scene_keys": sorted(excluded_scene_keys),
        "rejected_instance_records": rejected,
        "instance_records": [target_candidate_record(inst) for inst in instances],
        "depth_instance_records": [target_candidate_record(inst) for inst in depth_instances],
        "filtered_instance_records": [target_candidate_record(inst) for inst in filtered],
        "rgb_path": str(out_dir / f"{prefix}_rgb.png"),
    }


def mind_sort_table_pose_from_selected(selected: dict[str, Any]) -> tuple[float, float, float]:
    pose = selected.get("pose") or TABLE_APPROACH_POSE
    x, y = float(pose[0]), float(pose[1])
    yaw = float(pose[2]) if len(pose) >= 3 else None
    if yaw is None or not math.isfinite(float(yaw)):
        yaw = yaw_facing_table(x, y)
    return (x, y, float(yaw))


def write_rigid_object_pose(
    scene: InteractiveScene,
    scene_key: str,
    pos_w: tuple[float, float, float] | np.ndarray,
    yaw: float,
) -> None:
    obj = scene[scene_key]
    root_state = obj.data.root_state_w.clone()
    root_state[:, 0:3] = torch.tensor(pos_w, dtype=root_state.dtype, device=root_state.device).reshape(1, 3)
    root_state[:, 3:7] = torch.tensor(quat_wxyz_from_yaw(float(yaw)), dtype=root_state.dtype, device=root_state.device).reshape(1, 4)
    root_state[:, 7:] = 0.0
    obj.write_root_pose_to_sim(root_state[:, :7])
    obj.write_root_velocity_to_sim(root_state[:, 7:])


_MIND_SORT_GRIPPER_CARRY_ATTACHMENTS: dict[str, dict[str, Any]] = {}


def yaw_from_rotation_matrix_xy(rot: np.ndarray, fallback_yaw: float = 0.0) -> float:
    rot = np.asarray(rot, dtype=np.float32).reshape(3, 3)
    x_axis = rot[:2, 0]
    if float(np.linalg.norm(x_axis)) < 1.0e-5:
        return float(fallback_yaw)
    return float(math.atan2(float(x_axis[1]), float(x_axis[0])))


def mind_sort_carried_pose(robot: Articulation) -> tuple[tuple[float, float, float], float]:
    root = robot.data.root_state_w[0].detach().cpu()
    yaw = yaw_from_quat_wxyz(root[3:7])
    forward = float(args_cli.mind_sort_attach_forward_m)
    lateral = float(args_cli.mind_sort_attach_lateral_m)
    x = float(root[0]) + forward * math.cos(yaw) - lateral * math.sin(yaw)
    y = float(root[1]) + forward * math.sin(yaw) + lateral * math.cos(yaw)
    z = float(args_cli.mind_sort_attach_height_m)
    return (x, y, z), yaw


def mind_sort_gripper_carried_pose(
    scene: InteractiveScene,
    robot: Articulation,
    scene_key: str | None,
) -> tuple[tuple[float, float, float], float, dict[str, Any]]:
    if not scene_key:
        pos_w, yaw = mind_sort_carried_pose(robot)
        return pos_w, yaw, {"mode": "legacy_robot_root_offset", "reason": "missing_scene_key"}
    attachment = _MIND_SORT_GRIPPER_CARRY_ATTACHMENTS.get(scene_key)
    if not attachment:
        pos_w, yaw = mind_sort_carried_pose(robot)
        return pos_w, yaw, {"mode": "legacy_robot_root_offset", "reason": "no_gripper_attachment"}
    try:
        ee_body_id = int(attachment.get("right_ee_body_id") or resolve_right_ee_body_id(scene))
        tcp_pose_w = tcp_pose_matrix(robot, ee_body_id).astype(np.float32)
        offset_tcp = np.asarray(attachment.get("object_root_offset_tcp_m", [0.0, 0.0, -0.03]), dtype=np.float32).reshape(3)
        pos_w = tcp_pose_w[:3, 3] + tcp_pose_w[:3, :3] @ offset_tcp
        robot_yaw = yaw_from_quat_wxyz(robot.data.root_state_w[0, 3:7].detach().cpu())
        yaw = yaw_from_rotation_matrix_xy(tcp_pose_w[:3, :3], fallback_yaw=robot_yaw)
        return (
            (float(pos_w[0]), float(pos_w[1]), float(pos_w[2])),
            float(yaw),
            {
                "mode": "right_gripper_tcp_attachment",
                "right_ee_body_id": ee_body_id,
                "tcp_world_m": tcp_pose_w[:3, 3].astype(float).tolist(),
                "object_root_offset_tcp_m": offset_tcp.astype(float).tolist(),
                "attachment_source": attachment.get("source", ""),
            },
        )
    except Exception as exc:
        pos_w, yaw = mind_sort_carried_pose(robot)
        return pos_w, yaw, {"mode": "legacy_robot_root_offset", "reason": f"gripper_attachment_failed:{exc!r}"}


def mind_sort_update_carried_object(scene: InteractiveScene, robot: Articulation, scene_key: str | None) -> None:
    if not scene_key or scene_key not in scene.keys():
        return
    pos_w, yaw, _ = mind_sort_gripper_carried_pose(scene, robot, scene_key)
    write_rigid_object_pose(scene, scene_key, pos_w, yaw)


def mind_sort_align_object_over_bin_opening(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
    scene_key: str,
    category: str,
    *,
    steps: int | None = None,
) -> dict[str, Any]:
    tracker.enter("MIND_DROP_ALIGN", scene_key=scene_key, category=category, mode="align_release_over_hardcoded_bin")
    sim_dt = sim.get_physics_dt()
    gripper = AttachedParallelGripper(robot)
    alignment_steps = max(1, int(steps) if steps is not None else min(max(1, int(args_cli.drop_steps)), 60))
    bounds = bin_opening_bounds_for_category(category)
    hardcoded_release = bin_drop_position_for_category(category)
    attachment = _MIND_SORT_GRIPPER_CARRY_ATTACHMENTS.get(scene_key)
    object_start = tensor_to_numpy(scene[scene_key].data.root_state_w[0, :3]).astype(np.float32)
    release_yaw = math.pi
    carried_pose_meta: dict[str, Any] = {}
    carried_pos = object_start.copy()
    if attachment is not None:
        carried_tuple, carried_yaw, carried_pose_meta = mind_sort_gripper_carried_pose(scene, robot, scene_key)
        carried_pos = np.asarray(carried_tuple, dtype=np.float32).reshape(3)
        release_yaw = float(carried_yaw)

    tcp_pos = None
    tcp_pose_meta: dict[str, Any] = {}
    try:
        ee_body_id = int((attachment or {}).get("right_ee_body_id") or resolve_right_ee_body_id(scene))
        tcp_pose_w = tcp_pose_matrix(robot, ee_body_id).astype(np.float32)
        tcp_pos = tcp_pose_w[:3, 3].astype(np.float32)
        tcp_pose_meta = {
            "right_ee_body_id": ee_body_id,
            "tcp_world_m": tcp_pos.astype(float).tolist(),
            "tcp_inside_bin_opening_xy": bool(point_inside_bin_opening_xy(tcp_pos, category)),
        }
    except Exception as exc:
        tcp_pose_meta = {"error": repr(exc)}

    carried_inside = bool(point_inside_bin_opening_xy(carried_pos, category))
    tcp_inside = bool(tcp_pos is not None and point_inside_bin_opening_xy(tcp_pos, category))
    release_pos = hardcoded_release.copy()
    release_mode = "hardcoded_bin_center_fallback"
    if attachment is not None and carried_inside:
        release_pos[:2] = carried_pos[:2]
        release_mode = "gripper_carried_pose_inside_bin_opening"
    elif tcp_inside and tcp_pos is not None:
        release_pos[:2] = tcp_pos[:2]
        release_mode = "tcp_inside_bin_opening"
    release_pos[2] = float(BIN_DROP_Z)

    start_pos = object_start.copy()
    for step in range(alignment_steps):
        alpha = min_jerk(float(step + 1) / float(alignment_steps))
        pos = (1.0 - alpha) * start_pos + alpha * release_pos
        hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
        apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0, sim_dt)
        gripper.close()
        stabilize_robot_base_for_nav(robot)
        write_rigid_object_pose(scene, scene_key, pos, release_yaw)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        tracker.tick()

    final_pos = tensor_to_numpy(scene[scene_key].data.root_state_w[0, :3]).astype(np.float32)
    tcp_to_release = None if tcp_pos is None else (tcp_pos - release_pos).astype(float).tolist()
    return {
        "success": True,
        "mode": release_mode,
        "scene_key": scene_key,
        "category": category,
        "canonical_category": canonical_category(category),
        "target_bin_name": BIN_NAME_BY_CATEGORY.get(canonical_category(category), "other"),
        "bin_y_m": float(bin_y_for_category(category)),
        "bin_opening_bounds": bounds,
        "hardcoded_bin_release_pose_world": hardcoded_release.astype(float).tolist(),
        "release_pose_world": release_pos.astype(float).tolist(),
        "object_start_world_m": object_start.astype(float).tolist(),
        "object_final_pre_release_world_m": final_pos.astype(float).tolist(),
        "carried_pose_world_m": carried_pos.astype(float).tolist(),
        "carried_inside_bin_opening_xy": carried_inside,
        "tcp_inside_bin_opening_xy": tcp_inside,
        "tcp_to_release_offset_m": tcp_to_release,
        "steps": int(alignment_steps),
        "gripper_attachment": attachment,
        "carried_pose_meta": carried_pose_meta,
        "tcp_pose_meta": tcp_pose_meta,
    }


def mind_sort_settle_at_table(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
    target_pose: tuple[float, float, float],
) -> dict[str, Any]:
    tracker.enter("FACE_TABLE_AND_SETTLE", target_pose=target_pose, settle_steps=int(args_cli.mind_sort_settle_steps))
    sim_dt = sim.get_physics_dt()
    steps = max(1, int(args_cli.mind_sort_settle_steps))
    for _ in range(steps):
        hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
        apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0, sim_dt)
        stabilize_robot_base_for_nav(robot)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        tracker.tick()
    locked_joint_target.copy_(robot.data.joint_pos.clone())
    return {
        "success": True,
        "steps": steps,
        "target_pose": [float(v) for v in target_pose],
        "final_robot_pose": robot_planar_pose(robot),
    }


def mind_sort_move_wrist_to_local_view(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
    target: YoloInstance,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker.enter("MOVE_WRIST_TO_LOCAL_VIEW", target=target_candidate_record(target))
    meta: dict[str, Any] = {
        "enabled": bool(args_cli.wrist_refine_grasp),
        "success": False,
        "reason": "",
        "target_center_world_m": target.center_3d.tolist() if target.center_3d is not None else None,
        "isaaclab_grasp_ik": isaaclab_grasp_ik_metadata(),
    }
    if not bool(args_cli.wrist_refine_grasp):
        meta["reason"] = "disabled"
        return meta
    if target.center_3d is None:
        meta["reason"] = "target_has_no_center"
        return meta
    gripper = AttachedParallelGripper(robot)
    pose_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    pose_ik_controller = DifferentialIKController(pose_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    entity_cfg = SceneEntityCfg("robot", joint_names=isaaclab_grasp_ik_joint_exprs(), body_names=[RIGHT_EE_BODY])
    entity_cfg.resolve(scene)
    ee_body_id = int(entity_cfg.body_ids[0])
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
    center = np.asarray(target.center_3d, dtype=np.float32).reshape(3)
    tcp_view_pose = np.eye(4, dtype=np.float32)
    wrist_rot_w = angled_top_down_wrist_rotation() if args_cli.arm_motion_wrist_orientation == "angled_top_down" else top_down_wrist_rotation()
    tcp_view_pose[:3, :3] = wrist_rot_w
    tcp_view_pose[:3, 3] = center
    tcp_view_pose[2, 3] = max(float(center[2] + args_cli.wrist_refine_view_height_m), float(TABLE_SURFACE_Z + 0.22))
    wrist_view_pose = tcp_pose_to_wrist_pose(tcp_view_pose, wrist_rot_w)
    locked_root_pose_w = robot.data.root_pose_w.clone()
    gripper.open_for_grasp(0.06)
    segment = run_ik_segment_until_converged(
        sim,
        scene,
        robot,
        pose_ik_controller,
        entity_cfg,
        ee_body_id,
        ee_jacobi_idx,
        wrist_view_pose,
        int(args_cli.wrist_refine_steps),
        max(0.05, float(args_cli.pregrasp_error_threshold_m)),
        gripper=gripper,
        gripper_width=gripper_command_width(gripper, 0.06),
        locked_root_pose_w=locked_root_pose_w,
        target_tcp_pose_w=tcp_view_pose,
        control_tcp_position=False,
    )
    locked_joint_target.copy_(robot.data.joint_pos.clone())
    meta.update(
        {
            "success": bool(segment.get("converged", False)),
            "reason": "" if bool(segment.get("converged", False)) else "wrist_view_ik_not_converged",
            "tcp_view_pose_world": tcp_view_pose.tolist(),
            "wrist_view_pose_world": wrist_view_pose.tolist(),
            "segment": segment,
            "output_dir": str(out_dir),
        }
    )
    (out_dir / "wrist_view_motion.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def mind_sort_snap_to_observe_pose(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
    reason: str,
) -> dict[str, Any]:
    global _KINEMATIC_STABLE_POSE_XYYAW
    target_pose = waypoint_pose("home")
    before_pose = robot_planar_pose(robot)
    enabled = bool(args_cli.mind_sort_snap_observe_pose)
    if enabled:
        set_robot_planar_pose_for_dry_run(robot, target_pose)
        if args_cli.nav_actuation_mode == "wheel" and args_cli.wheel_ground_coupling == "kinematic_stable":
            _KINEMATIC_STABLE_POSE_XYYAW = [
                (float(target_pose[0]), float(target_pose[1]), float(target_pose[2]))
                for _ in range(int(robot.data.root_state_w.shape[0]))
            ]
    sim_dt = sim.get_physics_dt()
    for _ in range(30 if enabled else 1):
        hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
        if not enabled:
            apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0, sim_dt)
        stabilize_robot_base_for_nav(robot)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        tracker.tick()
    locked_joint_target.copy_(robot.data.joint_pos.clone())
    after_pose = robot_planar_pose(robot)
    meta = {
        "enabled": enabled,
        "reason": reason,
        "target_pose": [float(v) for v in target_pose],
        "before_pose": before_pose,
        "after_pose": after_pose,
        "before_position_error_m": float(math.hypot(float(target_pose[0]) - before_pose[0], float(target_pose[1]) - before_pose[1])),
        "before_yaw_error_rad": float(wrap_to_pi(float(target_pose[2]) - before_pose[2])),
    }
    tracker.event("observe_pose_snap", **meta)
    return meta


def mind_sort_attach_object(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
    scene_key: str,
) -> dict[str, Any]:
    tracker.enter("MIND_PICK_ATTACH", scene_key=scene_key)
    sim_dt = sim.get_physics_dt()
    steps = max(1, int(args_cli.grasp_steps))
    for _ in range(steps):
        hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
        apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0, sim_dt)
        stabilize_robot_base_for_nav(robot)
        mind_sort_update_carried_object(scene, robot, scene_key)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        tracker.tick()
    pos_w = tensor_to_numpy(scene[scene_key].data.root_state_w[0, :3]).astype(np.float32).tolist()
    return {"success": True, "scene_key": scene_key, "steps": steps, "carried_pose_world": pos_w}


def mind_sort_suction_assist_attach(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
    scene_key: str,
    target: YoloInstance,
    physical_grasp: dict[str, Any] | None,
) -> dict[str, Any]:
    proximity_assist_enabled = bool(args_cli.mind_sort_gripper_proximity_assist or args_cli.mind_sort_suction_assist)
    legacy_suction_mode = bool(args_cli.mind_sort_suction_assist and not args_cli.mind_sort_gripper_proximity_assist)
    max_distance = (
        float(args_cli.mind_sort_suction_assist_max_distance_m)
        if legacy_suction_mode
        else float(args_cli.mind_sort_gripper_proximity_assist_max_distance_m)
    )
    assist_steps = (
        int(args_cli.mind_sort_suction_assist_steps)
        if legacy_suction_mode and int(args_cli.mind_sort_suction_assist_steps) > 0
        else int(args_cli.mind_sort_gripper_proximity_assist_steps)
    )
    tracker.enter("GRIPPER_PROXIMITY_ASSIST", scene_key=scene_key, mode="demo_2cm_gripper_proximity_carry")
    meta: dict[str, Any] = {
        "enabled": proximity_assist_enabled,
        "success": False,
        "suction_assisted": False,
        "gripper_proximity_assisted": False,
        "mode": "gripper_proximity_assisted_pickup",
        "policy": "if_final_gripper_or_object_distance_within_threshold_then_carry",
        "legacy_suction_alias": legacy_suction_mode,
        "scene_key": scene_key,
        "target": target_candidate_record(target),
        "physical_failure_reason": str((physical_grasp or {}).get("reason", "")),
        "physical_grasp_success": bool((physical_grasp or {}).get("grasp_success", False)),
        "max_distance_m": max_distance,
        "reason": "",
    }
    if not proximity_assist_enabled:
        meta["reason"] = "disabled"
        return meta
    if scene_key not in scene.keys():
        meta["reason"] = "scene_key_not_found"
        return meta

    object_pos = tensor_to_numpy(scene[scene_key].data.root_state_w[0, :3]).astype(np.float32)
    target_center = np.asarray(target.center_3d, dtype=np.float32).reshape(3) if target.center_3d is not None else object_pos
    try:
        ee_body_id = resolve_right_ee_body_id(scene)
        tcp_pos = tcp_pose_matrix(robot, ee_body_id)[:3, 3].astype(np.float32)
    except Exception as exc:
        ee_body_id = None
        tcp_pos = None
        meta["tcp_pose_error"] = repr(exc)

    def point3(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            return None
        if arr.shape == (4, 4):
            point = arr[:3, 3]
        else:
            flat = arr.reshape(-1)
            if flat.size < 3:
                return None
            point = flat[:3]
        if not np.isfinite(point).all():
            return None
        return point.astype(np.float32)

    def pose4(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            return None
        if arr.shape != (4, 4) or not np.isfinite(arr).all():
            return None
        return arr.astype(np.float32)

    def iter_physical_records(record: dict[str, Any] | None) -> Iterable[tuple[str, dict[str, Any]]]:
        if not isinstance(record, dict):
            return
        yield "physical_grasp", record
        attempts = record.get("attempts", [])
        if isinstance(attempts, list):
            for attempt_idx, attempt in enumerate(attempts):
                if isinstance(attempt, dict):
                    yield f"attempt_{attempt_idx}", attempt

    target_points: list[tuple[str, np.ndarray]] = [
        ("visual_rgbd_center", target_center.astype(np.float32)),
        ("current_object_root", object_pos.astype(np.float32)),
    ]
    tcp_points: list[tuple[str, np.ndarray]] = []
    if tcp_pos is not None:
        tcp_points.append(("current_tcp_after_failure", tcp_pos.astype(np.float32)))
    low_grasp_tcp_pose_candidates: list[tuple[str, np.ndarray]] = []

    for record_label, record in iter_physical_records(physical_grasp):
        top_target = point3((record.get("top_grasp") or {}).get("target_world_m"))
        if top_target is not None:
            target_points.append((f"{record_label}.top_grasp_target", top_target))
        tcp_grasp = point3(record.get("tcp_grasp_pose_world"))
        if tcp_grasp is not None:
            target_points.append((f"{record_label}.tcp_grasp_pose", tcp_grasp))
        tcp_after_grasp = point3(record.get("tcp_pose_after_grasp_world"))
        if tcp_after_grasp is not None:
            tcp_points.append((f"{record_label}.tcp_after_grasp", tcp_after_grasp))
        tcp_after_grasp_pose = pose4(record.get("tcp_pose_after_grasp_world"))
        if tcp_after_grasp_pose is not None:
            low_grasp_tcp_pose_candidates.append((f"{record_label}.tcp_pose_after_grasp_world", tcp_after_grasp_pose))
        segments = record.get("segments", [])
        if not isinstance(segments, list):
            continue
        for seg_idx, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            seg_name = str(segment.get("name", f"segment_{seg_idx}"))
            seg_name_l = seg_name.lower()
            # Only low grasp records prove object proximity. Pregrasp hover targets
            # can be close to their own waypoint while still far above the object.
            if "grasp" not in seg_name_l or "pregrasp" in seg_name_l:
                continue
            final_tcp = point3(segment.get("final_tcp_world_m"))
            if final_tcp is not None:
                tcp_points.append((f"{record_label}.{seg_name}.final_tcp", final_tcp))
            for key in ("target_tcp_world_m", "object_grasp_target_tcp_world_m"):
                seg_target = point3(segment.get(key))
                if seg_target is not None:
                    target_points.append((f"{record_label}.{seg_name}.{key}", seg_target))

    proximity_checks: list[dict[str, Any]] = []
    closest_distance = float("inf")
    closest_source = ""
    closest_tcp = None
    closest_target = None
    for tcp_label, tcp_point in tcp_points:
        for target_label, target_point in target_points:
            distance = float(np.linalg.norm(tcp_point - target_point))
            check = {
                "tcp_source": tcp_label,
                "target_source": target_label,
                "distance_m": distance,
                "tcp_world_m": tcp_point.astype(float).tolist(),
                "target_world_m": target_point.astype(float).tolist(),
            }
            proximity_checks.append(check)
            if distance < closest_distance:
                closest_distance = distance
                closest_source = f"{tcp_label} -> {target_label}"
                closest_tcp = tcp_point
                closest_target = target_point
    proximity_checks.sort(key=lambda item: float(item.get("distance_m", float("inf"))))
    tcp_to_target = float(np.linalg.norm(tcp_pos - target_center)) if tcp_pos is not None else float("inf")
    tcp_to_object = float(np.linalg.norm(tcp_pos - object_pos)) if tcp_pos is not None else float("inf")
    meta.update(
        {
            "right_ee_body_id": ee_body_id,
            "tcp_world_m": tcp_pos.astype(float).tolist() if tcp_pos is not None else None,
            "target_center_world_m": target_center.astype(float).tolist(),
            "object_root_world_before_m": object_pos.astype(float).tolist(),
            "tcp_to_target_distance_m": tcp_to_target,
            "tcp_to_object_root_distance_m": tcp_to_object,
            "closest_suction_distance_m": closest_distance,
            "closest_gripper_proximity_distance_m": closest_distance,
            "closest_gripper_proximity_source": closest_source,
            "closest_gripper_proximity_tcp_world_m": closest_tcp.astype(float).tolist() if closest_tcp is not None else None,
            "closest_gripper_proximity_target_world_m": closest_target.astype(float).tolist() if closest_target is not None else None,
            "gripper_proximity_checks_top5": proximity_checks[:5],
        }
    )
    if max_distance > 0.0 and (not math.isfinite(closest_distance) or closest_distance > max_distance):
        meta["reason"] = f"gripper proximity assist gate failed: {closest_distance:.3f} m > {max_distance:.3f} m"
        tracker.event("gripper_proximity_assist_rejected", scene_key=scene_key, reason=meta["reason"], closest_distance_m=closest_distance)
        return meta

    gripper = AttachedParallelGripper(robot)
    sim_dt = sim.get_physics_dt()
    steps = max(1, assist_steps if assist_steps > 0 else int(args_cli.grasp_steps))
    start_pos = object_pos.copy()
    locked_joint_target.copy_(robot.data.joint_pos.clone())
    current_tcp_pose = None
    if ee_body_id is not None:
        try:
            current_tcp_pose = tcp_pose_matrix(robot, int(ee_body_id)).astype(np.float32)
        except Exception:
            current_tcp_pose = None
    reference_label = "current_tcp_after_failure"
    reference_tcp_pose = current_tcp_pose
    if low_grasp_tcp_pose_candidates:
        reference_label, reference_tcp_pose = low_grasp_tcp_pose_candidates[0]
    attachment_reason = "object_root_relative_to_low_grasp_tcp_pose"
    if reference_tcp_pose is not None:
        offset_tcp = reference_tcp_pose[:3, :3].T @ (object_pos - reference_tcp_pose[:3, 3])
        if float(np.linalg.norm(offset_tcp)) > 0.12 and closest_target is not None:
            offset_tcp = reference_tcp_pose[:3, :3].T @ (object_pos - closest_target.astype(np.float32))
            attachment_reason = "object_root_relative_to_closest_low_grasp_target"
        if float(np.linalg.norm(offset_tcp)) > 0.12:
            offset_tcp = np.array([0.0, 0.0, -0.035], dtype=np.float32)
            attachment_reason = "fallback_small_tcp_local_offset"
    else:
        offset_tcp = np.array([0.0, 0.0, -0.035], dtype=np.float32)
        attachment_reason = "fallback_no_tcp_pose_available"
    _MIND_SORT_GRIPPER_CARRY_ATTACHMENTS[scene_key] = {
        "right_ee_body_id": ee_body_id,
        "object_root_offset_tcp_m": offset_tcp.astype(float).tolist(),
        "source": reference_label,
        "reason": attachment_reason,
        "closest_gripper_proximity_source": closest_source,
        "closest_gripper_proximity_distance_m": closest_distance,
    }
    _, _, initial_carry_pose_meta = mind_sort_gripper_carried_pose(scene, robot, scene_key)
    meta["gripper_attachment"] = {
        "mode": "right_gripper_tcp_attachment",
        "right_ee_body_id": ee_body_id,
        "reference_tcp_pose_source": reference_label,
        "object_root_offset_tcp_m": offset_tcp.astype(float).tolist(),
        "reason": attachment_reason,
        "initial_carried_pose": initial_carry_pose_meta,
    }
    for step in range(steps):
        alpha = float(step + 1) / float(steps)
        carried_pos, carried_yaw, carried_pose_meta = mind_sort_gripper_carried_pose(scene, robot, scene_key)
        carried = np.asarray(carried_pos, dtype=np.float32)
        ease = 0.5 - 0.5 * math.cos(math.pi * alpha)
        pos = (1.0 - ease) * start_pos + ease * carried
        hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
        apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0, sim_dt)
        gripper.close()
        stabilize_robot_base_for_nav(robot)
        write_rigid_object_pose(scene, scene_key, pos, carried_yaw)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        tracker.tick()

    mind_sort_update_carried_object(scene, robot, scene_key)
    final_pos = tensor_to_numpy(scene[scene_key].data.root_state_w[0, :3]).astype(np.float32)
    meta.update(
        {
            "success": True,
            "suction_assisted": legacy_suction_mode,
            "gripper_proximity_assisted": True,
            "reason": "Physical gripper lift verification failed, but the gripper reached the object within the proximity threshold; continuing with gripper-proximity assisted pickup.",
            "steps": steps,
            "carried_pose_world": final_pos.astype(float).tolist(),
            "final_carried_pose": carried_pose_meta,
            "report_label": "gripper proximity assisted pickup",
        }
    )
    tracker.event("gripper_proximity_assist_attached", scene_key=scene_key, closest_distance_m=closest_distance, steps=steps)
    return meta


def mind_sort_drop_object(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
    scene_key: str,
    category: str,
) -> dict[str, Any]:
    tracker.enter("MIND_DROP_RELEASE", scene_key=scene_key, category=category)
    sim_dt = sim.get_physics_dt()
    drop_alignment = mind_sort_align_object_over_bin_opening(
        sim,
        scene,
        robot,
        wheel_ids,
        locked_joint_target,
        tracker,
        scene_key,
        category,
    )
    tracker.enter(
        "MIND_DROP_RELEASE",
        scene_key=scene_key,
        category=category,
        release_pose=drop_alignment.get("release_pose_world"),
        target_bin_name=drop_alignment.get("target_bin_name"),
    )
    released_attachment = _MIND_SORT_GRIPPER_CARRY_ATTACHMENTS.pop(scene_key, None)
    gripper = AttachedParallelGripper(robot)
    drop_pos = np.asarray(drop_alignment.get("release_pose_world", bin_drop_position_for_category(category)), dtype=np.float32).reshape(3)
    write_rigid_object_pose(scene, scene_key, drop_pos, math.pi)
    for _ in range(max(1, int(args_cli.drop_steps))):
        hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
        apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0, sim_dt)
        gripper.set_width(gripper.limits.max_width_m)
        stabilize_robot_base_for_nav(robot)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        tracker.tick()
    final_pos = tensor_to_numpy(scene[scene_key].data.root_state_w[0, :3]).astype(np.float32).tolist()
    return {
        "success": True,
        "scene_key": scene_key,
        "category": category,
        "canonical_category": canonical_category(category),
        "target_bin_name": BIN_NAME_BY_CATEGORY.get(canonical_category(category), "other"),
        "drop_alignment": drop_alignment,
        "drop_pose_world": [float(v) for v in drop_pos.tolist()],
        "final_pose_world": final_pos,
        "released_gripper_attachment": released_attachment,
    }


def mind_sort_release_physical_object(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
    scene_key: str,
    category: str,
) -> dict[str, Any]:
    tracker.enter("DROP", mode="physical_gripper_release", scene_key=scene_key, category=category)
    gripper = AttachedParallelGripper(robot)
    sim_dt = sim.get_physics_dt()
    steps = max(1, int(args_cli.drop_steps))
    drop_alignment = None
    if scene_key in scene.keys():
        drop_alignment = mind_sort_align_object_over_bin_opening(
            sim,
            scene,
            robot,
            wheel_ids,
            locked_joint_target,
            tracker,
            scene_key,
            category,
        )
    tracker.enter(
        "DROP",
        mode="physical_gripper_release",
        scene_key=scene_key,
        category=category,
        release_pose=None if drop_alignment is None else drop_alignment.get("release_pose_world"),
        target_bin_name=None if drop_alignment is None else drop_alignment.get("target_bin_name"),
    )
    for _ in range(steps):
        hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
        apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0, sim_dt)
        gripper.set_width(gripper.limits.max_width_m)
        stabilize_robot_base_for_nav(robot)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        tracker.tick()
    locked_joint_target.copy_(robot.data.joint_pos.clone())
    final_pos = tensor_to_numpy(scene[scene_key].data.root_state_w[0, :3]).astype(np.float32).tolist() if scene_key in scene.keys() else None
    return {
        "success": True,
        "mode": "physical_gripper_release",
        "scene_key": scene_key,
        "category": category,
        "canonical_category": canonical_category(category),
        "target_bin_name": BIN_NAME_BY_CATEGORY.get(canonical_category(category), "other"),
        "drop_alignment": drop_alignment,
        "steps": int(steps),
        "final_pose_world": final_pos,
    }


def mind_sort_prepare_physical_grasp_posture(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
) -> dict[str, Any]:
    tracker.event("prepare_grasp_posture", mode="natural_down_ik_seed")
    gripper = AttachedParallelGripper(robot)
    sim_dt = sim.get_physics_dt()
    steps = max(0, int(args_cli.mind_sort_grasp_posture_reset_steps))
    start_target = locked_joint_target.clone()
    natural_target = arm_pose_target(robot, locked_joint_target, ARM_NATURAL_DOWN_JOINT_POS)
    root_pose_before = robot_planar_pose(robot)
    for step in range(steps):
        alpha = min_jerk((step + 1) / max(1, steps))
        motion_target = start_target + alpha * (natural_target - start_target)
        hold_non_wheel_joints(robot, motion_target, wheel_ids)
        apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0, sim_dt)
        gripper.set_width(gripper.limits.max_width_m)
        stabilize_robot_base_for_nav(robot)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        tracker.tick()
    if steps <= 0:
        hold_non_wheel_joints(robot, natural_target, wheel_ids)
        gripper.set_width(gripper.limits.max_width_m)
    locked_joint_target.copy_(natural_target)
    root_pose_after = robot_planar_pose(robot)
    return {
        "success": True,
        "steps": int(steps),
        "mode": "natural_down_ik_seed",
        "root_pose_before": [float(v) for v in root_pose_before],
        "root_pose_after": [float(v) for v in root_pose_after],
        "arm_pose": "natural_down",
        "gripper_width_m": float(gripper.limits.max_width_m),
    }


def mind_sort_execute_physical_grasp(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    wheel_ids: list[int],
    locked_joint_target: torch.Tensor,
    tracker: Task319StateTracker,
    target: YoloInstance,
    scene_key: str,
    cycle_dir: Path,
) -> dict[str, Any]:
    grasp_dir = cycle_dir / "physical_grasp"
    grasp_dir.mkdir(parents=True, exist_ok=True)
    tracker.enter("PLAN_PHYSICAL_GRASP", scene_key=scene_key, target=target_candidate_record(target))
    target.scene_key = scene_key
    if target.scene_object_name is None:
        target.scene_object_name = scene_object_name_for_key(scene_key)
    grasp_proposal = str(args_cli.mind_sort_grasp_proposal)
    selected_center = np.asarray(target.center_3d, dtype=np.float32).reshape(3) if target.center_3d is not None else None
    selected_points = target.points_world
    pre_grasp_rgb, pre_grasp_depth_m, pre_grasp_intrinsics, pre_grasp_camera_to_world = capture_head_camera(scene)
    Image.fromarray(pre_grasp_rgb).save(grasp_dir / "pre_physical_grasp_head_rgb.png")
    save_depth_images(pre_grasp_depth_m, grasp_dir, prefix="pre_physical_grasp_head_depth")
    refine_meta: dict[str, Any] = {"enabled": False, "success": False, "reason": "not_requested"}
    view_meta: dict[str, Any] = {"enabled": False, "success": False, "reason": "not_requested"}
    refine_source = "head_reshoot"
    if bool(args_cli.wrist_refine_grasp) and grasp_proposal == "rgbd_center":
        view_dir = grasp_dir / "wrist_local_view"
        view_meta = mind_sort_move_wrist_to_local_view(sim, scene, robot, wheel_ids, locked_joint_target, tracker, target, view_dir)
        tracker.enter("WRIST_RGBD_REFINE_TARGET", scene_key=scene_key)
        refined_center, refined_points, refine_meta = wrist_refine_target_center(scene, target, view_dir)
        if refined_center is not None:
            selected_center = refined_center
            selected_points = refined_points
            refine_source = "wrist_rgbd"
    elif bool(args_cli.wrist_refine_grasp) and grasp_proposal == "graspnet_baseline":
        refine_meta = {
            "enabled": False,
            "success": False,
            "reason": "disabled_for_graspnet_baseline_head_rgbd_mask",
        }
    if selected_center is None and grasp_proposal != "graspnet_baseline":
        meta = {
            "enabled": True,
            "success": False,
            "grasp_success": False,
            "reason": "No RGB-D target center is available for physical mind-sort grasp.",
            "wrist_view": view_meta,
            "wrist_refine": refine_meta,
        }
        (grasp_dir / "physical_grasp_execution.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta
    graspnet_meta: dict[str, Any] | None = None
    if grasp_proposal == "graspnet_baseline":
        tracker.enter("PLAN_PHYSICAL_GRASP", scene_key=scene_key, stage="graspnet_baseline")
        graspnet_dir = grasp_dir / "graspnet_baseline"
        graspnet_dir.mkdir(parents=True, exist_ok=True)
        old_save_legacy_grasp_debug = bool(args_cli.save_legacy_grasp_debug)
        if bool(args_cli.mind_sort_graspnet_save_debug):
            args_cli.save_legacy_grasp_debug = True
        try:
            selected_candidates, graspnet_meta = run_graspnet_stage(
                pre_grasp_rgb,
                pre_grasp_depth_m,
                pre_grasp_intrinsics,
                pre_grasp_camera_to_world,
                target,
                graspnet_dir,
                freeze_target_identity(target),
            )
        finally:
            args_cli.save_legacy_grasp_debug = old_save_legacy_grasp_debug
        if not selected_candidates:
            meta = {
                "enabled": True,
                "success": False,
                "grasp_success": False,
                "mode": "physical_mind_sort_grasp",
                "grasp_proposal": grasp_proposal,
                "reason": "GraspNet Baseline produced no executable grasp candidate.",
                "graspnet_baseline": graspnet_meta or {},
                "wrist_view": view_meta,
                "wrist_refine": refine_meta,
                "allow_bin_navigation": False,
            }
            (grasp_dir / "physical_grasp_execution.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            return meta
        candidate = selected_candidates[0]
    else:
        selected_candidates = [
            selected_grasp_from_rgbd_center(
                target,
                selected_center,
                selected_points,
                source=f"mind_sort_{refine_source}_center",
                metadata={
                    "mind_sort_physical_grasp": True,
                    "refine_source": refine_source,
                    "wrist_view_success": bool(view_meta.get("success", False)),
                    "wrist_refine_success": bool(refine_meta.get("success", False)),
                },
            )
        ]
        candidate = selected_candidates[0]
    target_alignment_meta = write_physical_grasp_target_alignment(
        scene,
        pre_grasp_rgb,
        pre_grasp_intrinsics,
        pre_grasp_camera_to_world,
        candidate,
        scene_key,
        grasp_dir,
    )
    rgbd_vs_truth_meta = write_rgbd_vs_truth_debug(
        scene,
        pre_grasp_rgb,
        target,
        pre_grasp_intrinsics,
        pre_grasp_camera_to_world,
        grasp_dir,
        label="mind_sort_physical_grasp_target",
    )
    calibration_camera_to_world = refine_meta.get("camera_to_world") if refine_meta.get("camera_to_world") is not None else pre_grasp_camera_to_world
    calibration_intrinsics = refine_meta.get("intrinsics") if refine_meta.get("intrinsics") is not None else pre_grasp_intrinsics
    try:
        execution = execute_grasp(
            sim,
            scene,
            selected_candidates,
            grasp_dir,
            calibration_debug_context={
                "camera_to_world": calibration_camera_to_world,
                "intrinsics": calibration_intrinsics,
                "projection_image": refine_meta.get("overlay", ""),
            },
        )
    except Exception as exc:
        execution = {
            "enabled": True,
            "success": False,
            "grasp_success": False,
            "drop_success": False,
            "reason": f"execute_grasp_exception:{exc!r}",
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "attempts": [],
            "segments": [],
        }
    execution.update(
        {
            "mode": "physical_mind_sort_grasp",
            "grasp_proposal": grasp_proposal,
            "refine_source": refine_source,
            "wrist_view": view_meta,
            "wrist_refine": refine_meta,
            "graspnet_baseline": graspnet_meta,
            "target_alignment": target_alignment_meta,
            "rgbd_vs_truth_debug": rgbd_vs_truth_meta,
            "selected_candidate": selected_grasp_metadata(candidate),
            "selected_candidate_count": len(selected_candidates),
            "allow_bin_navigation": bool(execution.get("grasp_success", False)),
        }
    )
    if bool(execution.get("grasp_success", False)):
        locked_joint_target.copy_(robot.data.joint_pos.clone())
    (grasp_dir / "physical_grasp_execution.json").write_text(json.dumps(execution, indent=2, ensure_ascii=False))
    return execution


def run_sort_nav_after_grasp(
    sim: SimulationContext,
    scene: InteractiveScene,
    run_dir: Path,
    cycle_dir: Path,
    category: str | None,
) -> dict[str, Any]:
    category = canonical_category(category)
    robot: Articulation = scene["robot"]
    gripper = AttachedParallelGripper(robot)
    tracker = Task319StateTracker(sim.get_physics_dt(), sim.device)
    tracker.enter("REST", mode="post_grasp_sort_nav", category=category, nav_backend=args_cli.nav_backend)
    wheel_ids, wheel_names = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
    metadata: dict[str, Any] = {
        "enabled": True,
        "mode": "post_grasp_sort_nav",
        "nav_backend": args_cli.nav_backend,
        "category": category,
        "target_bin_name": BIN_NAME_BY_CATEGORY.get(category, "other"),
        "target_bin_pose": nav2_bin_pose_for_category(category),
        "bin_staging_pose": waypoint_pose("bin_center"),
        "home_pose": waypoint_pose("home"),
        "return_home_via_bin_staging": True,
        "return_home_side_corridor_y": SIDE_CORRIDOR_Y,
        "return_home_approach_backoff_m": TABLE_RETURN_APPROACH_BACKOFF_M,
        "wheel_drive": wheel_drive_metadata(),
        "wheel_joints": wheel_names,
        "navigation": [],
        "drop": None,
        "success": False,
        "reason": "",
        "nav2_stack": {
            "auto_start": bool(args_cli.start_nav2_stack),
            "script": args_cli.nav2_stack_script,
            "startup_s": float(args_cli.nav2_stack_startup_s),
            "log": "",
        },
        "ros2_bridge": {
            "host": args_cli.ros2_bridge_host,
            "port": int(args_cli.ros2_bridge_port),
            "setup": args_cli.ros2_setup,
            "python": args_cli.ros2_python,
        },
    }
    if args_cli.nav_backend != "nav2":
        metadata["reason"] = f"Unsupported nav backend: {args_cli.nav_backend}"
        return metadata
    if len(wheel_ids) != 4:
        metadata["reason"] = f"Post-grasp navigation requires 4 wheel joints, found {wheel_names}."
        tracker.enter("RECOVER", reason=metadata["reason"])
        metadata["fsm_trace"] = tracker.trace
        (cycle_dir / "post_grasp_sort_nav.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return metadata
    locked_joint_target = robot.data.joint_pos.clone()
    bridge = ExternalRos2NavBridgeClient(run_dir)
    nav2_stack_process: subprocess.Popen[Any] | None = None
    try:
        bridge.start()
        nav2_stack_process, nav2_stack_log = start_nav2_stack(run_dir)
        if nav2_stack_log is not None:
            metadata["nav2_stack"]["log"] = str(nav2_stack_log)
        sim_dt = sim.get_physics_dt()
        for _ in range(args_cli.warmup_steps):
            vx, wz = bridge.exchange(robot)
            vx, wz = limit_external_nav_cmd(vx, wz, sim_dt)
            hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
            apply_wheel_velocity(robot, wheel_ids, vx, wz, sim_dt)
            stabilize_robot_base_for_nav(robot)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim_dt)
            record_video_frame(scene)
            gui_playback_tick(sim_dt)
            tracker.tick()

        nav_staging = run_nav2_goal(
            sim, scene, robot, bridge, wheel_ids, locked_joint_target, tracker,
            "NAV_TO_BIN_STAGING", f"post_grasp_bin_staging_{category}", waypoint_pose("bin_center")
        )
        metadata["navigation"].append(nav_staging)
        if not nav_staging.get("success", False):
            metadata["reason"] = f"Nav2 failed to reach bin staging: {nav_staging.get('status')}"
            tracker.enter("RECOVER", reason=metadata["reason"])
            return metadata

        tracker.enter("SELECT_BIN_BY_CATEGORY", category=category, target_pose=nav2_bin_pose_for_category(category))
        nav_bin = run_nav2_goal(
            sim, scene, robot, bridge, wheel_ids, locked_joint_target, tracker,
            "NAV_TO_BIN", f"post_grasp_bin_{BIN_NAME_BY_CATEGORY.get(category, 'other')}_{category}", nav2_bin_pose_for_category(category)
        )
        metadata["navigation"].append(nav_bin)
        if not nav_bin.get("success", False):
            metadata["reason"] = f"Nav2 failed to reach target bin: {nav_bin.get('status')}"
            tracker.enter("RECOVER", reason=metadata["reason"])
            return metadata

        drop = run_motion_stub_drop(sim, scene, robot, gripper, wheel_ids, locked_joint_target, tracker)
        drop["category"] = category
        metadata["drop"] = drop
        if not bool(drop.get("success", False)):
            metadata["success"] = False
            metadata["reason"] = "Drop failed after Nav2 bin navigation."
            tracker.enter("RECOVER", reason=metadata["reason"])
            return metadata

        nav_return_staging = run_nav2_goal(
            sim,
            scene,
            robot,
            bridge,
            wheel_ids,
            locked_joint_target,
            tracker,
            "RETURN_TO_BIN_STAGING",
            f"post_drop_return_to_bin_staging_{category}",
            waypoint_pose("bin_center"),
        )
        metadata["navigation"].append(nav_return_staging)
        if not nav_return_staging.get("success", False):
            metadata["reason"] = f"Nav2 failed to return to bin staging after drop: {nav_return_staging.get('status')}"
            tracker.enter("RECOVER", reason=metadata["reason"])
            return metadata

        table_pose = waypoint_pose("home")
        corridor_y = SIDE_CORRIDOR_Y if bin_y_for_category(category) >= 0.0 else -SIDE_CORRIDOR_Y
        side_corridor_pose = nav2_pose_with_travel_yaw(robot, (BIN_DRIVE_X, corridor_y))
        nav_side_corridor = run_nav2_goal(
            sim,
            scene,
            robot,
            bridge,
            wheel_ids,
            locked_joint_target,
            tracker,
            "RETURN_TO_SIDE_CORRIDOR",
            f"post_drop_return_side_corridor_{category}",
            side_corridor_pose,
        )
        metadata["navigation"].append(nav_side_corridor)
        if not nav_side_corridor.get("success", False):
            metadata["reason"] = f"Nav2 failed to reach side corridor after drop: {nav_side_corridor.get('status')}"
            tracker.enter("RECOVER", reason=metadata["reason"])
            return metadata

        home_approach_xy = (table_pose[0] - TABLE_RETURN_APPROACH_BACKOFF_M, corridor_y)
        home_approach_pose = nav2_pose_with_travel_yaw(robot, home_approach_xy)
        nav_home_approach = run_nav2_goal(
            sim,
            scene,
            robot,
            bridge,
            wheel_ids,
            locked_joint_target,
            tracker,
            "RETURN_HOME_APPROACH",
            f"post_drop_return_home_approach_{category}",
            home_approach_pose,
        )
        metadata["navigation"].append(nav_home_approach)
        if not nav_home_approach.get("success", False):
            metadata["reason"] = f"Nav2 failed to reach home approach after drop: {nav_home_approach.get('status')}"
            tracker.enter("RECOVER", reason=metadata["reason"])
            return metadata

        home_align_xy = (table_pose[0] - TABLE_RETURN_APPROACH_BACKOFF_M, table_pose[1])
        home_align_pose = nav2_pose_with_travel_yaw(robot, home_align_xy)
        nav_home_align = run_nav2_goal(
            sim,
            scene,
            robot,
            bridge,
            wheel_ids,
            locked_joint_target,
            tracker,
            "RETURN_HOME_ALIGN",
            f"post_drop_return_home_align_{category}",
            home_align_pose,
        )
        metadata["navigation"].append(nav_home_align)
        if not nav_home_align.get("success", False):
            metadata["reason"] = f"Nav2 failed to align before home after drop: {nav_home_align.get('status')}"
            tracker.enter("RECOVER", reason=metadata["reason"])
            return metadata

        nav_return_home = run_nav2_goal(
            sim,
            scene,
            robot,
            bridge,
            wheel_ids,
            locked_joint_target,
            tracker,
            "RETURN_HOME",
            f"post_drop_return_home_{category}",
            table_pose,
        )
        metadata["navigation"].append(nav_return_home)
        if not nav_return_home.get("success", False):
            metadata["reason"] = f"Nav2 failed to return home after drop: {nav_return_home.get('status')}"
            tracker.enter("RECOVER", reason=metadata["reason"])
            return metadata

        metadata["success"] = True
        metadata["reason"] = "Post-grasp Nav2 bin-side drop completed and robot returned home."
        tracker.enter("DONE", reason=metadata["reason"])
        return metadata
    except Exception as exc:
        metadata["success"] = False
        metadata["reason"] = repr(exc)
        tracker.enter("RECOVER", reason=repr(exc))
        return metadata
    finally:
        bridge.close()
        stop_external_process(nav2_stack_process, "Nav2 stack")
        metadata["final_pose"] = robot_planar_pose(robot)
        metadata["nav2_failure_hints"] = nav2_failure_hints(run_dir)
        metadata["fsm_trace"] = tracker.trace
        (cycle_dir / "post_grasp_sort_nav.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


def wheel_drive_metadata() -> dict[str, Any]:
    return {
        "actuation_mode": args_cli.nav_actuation_mode,
        "model": args_cli.wheel_drive_model,
        "wheel_radius": float(args_cli.wheel_radius),
        "track_width": float(args_cli.track_width),
        "mecanum_wheel_sign": float(args_cli.mecanum_wheel_sign),
        "wheel_velocity_scale": float(args_cli.wheel_velocity_scale),
        "ground_coupling": args_cli.wheel_ground_coupling,
        "stable_integrated_pose": args_cli.wheel_ground_coupling == "kinematic_stable",
        "wheel_root_stabilization": bool(args_cli.wheel_root_stabilization),
        "max_linear_speed_mps": float(args_cli.nav_max_linear_speed),
        "max_angular_speed_radps": float(args_cli.nav_max_angular_speed),
        "cmd_angular_scale": float(args_cli.nav_cmd_angular_scale),
        "cmd_smoothing": bool(args_cli.nav_cmd_smoothing),
        "cmd_linear_accel_limit_mps2": float(args_cli.nav_cmd_linear_accel_limit),
        "cmd_angular_accel_limit_radps2": float(args_cli.nav_cmd_angular_accel_limit),
        "cmd_filter_alpha": float(args_cli.nav_cmd_filter_alpha),
        "final_dock_position_tolerance_m": float(args_cli.nav_final_dock_position_tolerance),
        "final_dock_yaw_tolerance_rad": float(args_cli.nav_final_dock_yaw_tolerance),
    }


def run_wheel_open_loop_demo(sim: SimulationContext, scene: InteractiveScene, run_dir: Path) -> None:
    cycle_dir = run_dir / "wheel_open_loop"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    robot: Articulation = scene["robot"]
    tracker = Task319StateTracker(sim.get_physics_dt(), sim.device)
    tracker.enter(
        "REST",
        mode="wheel_open_loop_demo",
        nav_actuation_mode=args_cli.nav_actuation_mode,
        wheel_drive_model=args_cli.wheel_drive_model,
    )
    reset_scene(scene)
    wheel_ids, wheel_names = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
    if len(wheel_ids) != 4:
        raise RuntimeError(f"Wheel open-loop demo requires 4 wheel joints, found {wheel_names}.")
    locked_joint_target = robot.data.joint_pos.clone()
    sim_dt = sim.get_physics_dt()
    raw_velocity = parse_raw_wheel_velocity(args_cli.wheel_open_loop_raw_velocity)
    kinematic_targets = wheel_velocity_targets(float(args_cli.wheel_open_loop_linear_speed), float(args_cli.wheel_open_loop_angular_speed))
    command_targets = raw_velocity if raw_velocity is not None else kinematic_targets
    metadata: dict[str, Any] = {
        "mode": "wheel_open_loop_demo",
        "vision_paused": True,
        "nav_backend": "none",
        "wheel_drive": wheel_drive_metadata(),
        "wheel_joints": wheel_names,
        "command": {
            "linear_x_mps": float(args_cli.wheel_open_loop_linear_speed),
            "angular_z_radps": float(args_cli.wheel_open_loop_angular_speed),
            "wheel_velocity_targets_radps": command_targets,
            "kinematic_wheel_velocity_targets_radps": kinematic_targets,
            "raw_velocity_override": raw_velocity,
            "steps": int(args_cli.wheel_open_loop_steps),
            "sweep": bool(args_cli.wheel_open_loop_sweep),
            "sweep_steps": int(args_cli.wheel_open_loop_sweep_steps),
            "sweep_speed_radps": float(args_cli.wheel_open_loop_sweep_speed),
            "min_translation_m": float(args_cli.wheel_open_loop_min_translation_m),
            "min_yaw_rad": float(args_cli.wheel_open_loop_min_yaw_rad),
        },
        "initial_pose": robot_planar_pose(robot),
        "initial_wheel_state": wheel_state_snapshot(robot, wheel_ids, wheel_names),
        "samples": [],
        "final_pose": None,
        "final_wheel_state": None,
        "translation_m": 0.0,
        "yaw_delta_rad": 0.0,
        "sweep_results": [],
        "best_sweep_result": None,
        "success": False,
        "reason": "",
    }
    try:
        if args_cli.wheel_open_loop_sweep:
            tracker.enter(
                "WHEEL_OPEN_LOOP",
                mode="sweep",
                sweep_speed_radps=float(args_cli.wheel_open_loop_sweep_speed),
                sweep_steps=int(args_cli.wheel_open_loop_sweep_steps),
            )
            signs = (-1.0, 1.0)
            patterns: list[list[float]] = []
            speed = float(args_cli.wheel_open_loop_sweep_speed)
            for lf in signs:
                for lb in signs:
                    for rf in signs:
                        for rb in signs:
                            patterns.append([lf * speed, lb * speed, rf * speed, rb * speed])
            for pattern_idx, targets in enumerate(patterns):
                reset_scene(scene)
                locked_joint_target = robot.data.joint_pos.clone()
                initial_pose = robot_planar_pose(robot)
                initial_wheels = wheel_state_snapshot(robot, wheel_ids, wheel_names)
                for step in range(max(1, int(args_cli.wheel_open_loop_sweep_steps))):
                    hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
                    apply_raw_wheel_velocity(robot, wheel_ids, targets)
                    stabilize_robot_base_for_nav(robot)
                    scene.write_data_to_sim()
                    sim.step(render=True)
                    scene.update(sim_dt)
                    record_video_frame(scene)
                    gui_playback_tick(sim_dt)
                    tracker.tick()
                apply_raw_wheel_velocity(robot, wheel_ids, [0.0, 0.0, 0.0, 0.0])
                final_pose = robot_planar_pose(robot)
                translation = math.hypot(float(final_pose[0]) - float(initial_pose[0]), float(final_pose[1]) - float(initial_pose[1]))
                yaw_delta = wrap_to_pi(float(final_pose[2]) - float(initial_pose[2]))
                result = {
                    "index": pattern_idx,
                    "wheel_velocity_targets_radps": [float(value) for value in targets],
                    "initial_pose": initial_pose,
                    "final_pose": final_pose,
                    "translation_m": translation,
                    "yaw_delta_rad": yaw_delta,
                    "initial_wheel_state": initial_wheels,
                    "final_wheel_state": wheel_state_snapshot(robot, wheel_ids, wheel_names),
                }
                metadata["sweep_results"].append(result)
                tracker.event("wheel_open_loop_sweep_result", **result)
            best = max(metadata["sweep_results"], key=lambda item: float(item["translation_m"]))
            metadata["best_sweep_result"] = best
            metadata["initial_pose"] = best["initial_pose"]
            metadata["final_pose"] = best["final_pose"]
            metadata["translation_m"] = float(best["translation_m"])
            metadata["yaw_delta_rad"] = float(best["yaw_delta_rad"])
            metadata["final_wheel_state"] = best["final_wheel_state"]
            metadata["success"] = metadata["translation_m"] >= float(args_cli.wheel_open_loop_min_translation_m)
            if metadata["success"]:
                metadata["reason"] = "Wheel sign sweep found a pattern with usable linear displacement."
                tracker.enter("DONE", reason=metadata["reason"], best_sweep_result=best)
            else:
                metadata["reason"] = "Wheel sign sweep did not find any pattern with usable linear displacement."
                tracker.enter("RECOVER", reason=metadata["reason"], best_sweep_result=best)
            return

        tracker.enter(
            "WHEEL_OPEN_LOOP",
            vx=float(args_cli.wheel_open_loop_linear_speed),
            wz=float(args_cli.wheel_open_loop_angular_speed),
            raw_velocity=raw_velocity,
            wheel_targets=command_targets,
        )
        for step in range(max(1, int(args_cli.wheel_open_loop_steps))):
            hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
            if raw_velocity is None:
                apply_wheel_velocity(
                    robot,
                    wheel_ids,
                    float(args_cli.wheel_open_loop_linear_speed),
                    float(args_cli.wheel_open_loop_angular_speed),
                    sim_dt,
                )
            else:
                apply_raw_wheel_velocity(robot, wheel_ids, command_targets)
            stabilize_robot_base_for_nav(robot)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim_dt)
            record_video_frame(scene)
            gui_playback_tick(sim_dt)
            if step % 60 == 0:
                sample = {
                    "step": step,
                    "pose": robot_planar_pose(robot),
                    "wheel_state": wheel_state_snapshot(robot, wheel_ids, wheel_names),
                }
                metadata["samples"].append(sample)
                tracker.event("wheel_open_loop_progress", **sample)
            tracker.tick()
        apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0)
        metadata["final_pose"] = robot_planar_pose(robot)
        metadata["final_wheel_state"] = wheel_state_snapshot(robot, wheel_ids, wheel_names)
        initial = metadata["initial_pose"]
        final = metadata["final_pose"]
        translation = math.hypot(float(final[0]) - float(initial[0]), float(final[1]) - float(initial[1]))
        yaw_delta = wrap_to_pi(float(final[2]) - float(initial[2]))
        metadata["translation_m"] = translation
        metadata["yaw_delta_rad"] = yaw_delta
        commanded_linear = abs(float(args_cli.wheel_open_loop_linear_speed)) > 1.0e-6
        commanded_angular = abs(float(args_cli.wheel_open_loop_angular_speed)) > 1.0e-6
        if raw_velocity is not None:
            linear_ok = translation >= float(args_cli.wheel_open_loop_min_translation_m)
            angular_ok = abs(yaw_delta) >= float(args_cli.wheel_open_loop_min_yaw_rad)
            metadata["success"] = bool(linear_ok or angular_ok)
        else:
            linear_ok = (not commanded_linear) or translation >= float(args_cli.wheel_open_loop_min_translation_m)
            angular_ok = (not commanded_angular) or abs(yaw_delta) >= float(args_cli.wheel_open_loop_min_yaw_rad)
            metadata["success"] = bool((commanded_linear or commanded_angular) and linear_ok and angular_ok)
        if metadata["success"]:
            metadata["reason"] = "Wheel open-loop command moved the robot through the configured wheel-ground coupling."
            tracker.enter("DONE", reason=metadata["reason"])
        else:
            metadata["reason"] = (
                "Wheel open-loop command did not produce the expected base displacement; "
                "check wheel joint order/sign, wheel actuator limits, contact friction, and base stabilization writes."
            )
            tracker.enter("RECOVER", reason=metadata["reason"])
    except Exception as exc:
        metadata["success"] = False
        metadata["reason"] = repr(exc)
        metadata["final_pose"] = robot_planar_pose(robot)
        metadata["final_wheel_state"] = wheel_state_snapshot(robot, wheel_ids, wheel_names)
        tracker.enter("RECOVER", reason=repr(exc))
    finally:
        apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0)
        metadata["fsm_trace"] = tracker.trace
        metadata["observer_video"] = _VIDEO_RECORDER.metadata() if _VIDEO_RECORDER is not None else {"enabled": False}
        (cycle_dir / "wheel_open_loop_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        (cycle_dir / "fsm_trace.json").write_text(json.dumps(tracker.trace, indent=2, ensure_ascii=False))
        (run_dir / "latest_cycle.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"[INFO] Wheel open-loop demo success={metadata['success']} reason={metadata['reason']} -> {cycle_dir}", flush=True)


def run_ros_cmd_vel_demo(sim: SimulationContext, scene: InteractiveScene, run_dir: Path) -> None:
    cycle_dir = run_dir / "ros_cmd_vel_demo"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    robot: Articulation = scene["robot"]
    tracker = Task319StateTracker(sim.get_physics_dt(), sim.device)
    tracker.enter("REST", mode="ros_cmd_vel_demo", command_source="ROS2 /cmd_vel")
    reset_scene(scene)
    wheel_ids, wheel_names = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
    if len(wheel_ids) != 4:
        raise RuntimeError(f"ROS cmd_vel demo requires 4 wheel joints, found {wheel_names}.")
    locked_joint_target = robot.data.joint_pos.clone()
    bridge = ExternalRos2NavBridgeClient(run_dir)
    publisher_process: subprocess.Popen[Any] | None = None
    publisher_log: Path | None = None
    sim_dt = sim.get_physics_dt()
    duration_s = max(float(args_cli.ros_cmd_vel_demo_steps) * float(sim_dt) * 1.35 + 1.0, 1.5)
    metadata: dict[str, Any] = {
        "mode": "ros_cmd_vel_demo",
        "vision_paused": True,
        "nav_backend": "ros2_cmd_vel_topic",
        "wheel_drive": wheel_drive_metadata(),
        "ros2_bridge": {
            "host": args_cli.ros2_bridge_host,
            "port": int(args_cli.ros2_bridge_port),
            "setup": args_cli.ros2_setup,
            "python": args_cli.ros2_python,
        },
        "test_publisher": {
            "auto_publish": bool(args_cli.ros_cmd_vel_demo_auto_publish),
            "linear_x_mps": float(args_cli.ros_cmd_vel_demo_linear_x),
            "linear_y_mps": float(args_cli.ros_cmd_vel_demo_linear_y),
            "angular_z_radps": float(args_cli.ros_cmd_vel_demo_angular_z),
            "rate_hz": float(args_cli.ros_cmd_vel_demo_rate_hz),
            "duration_s": float(duration_s),
            "log": "",
        },
        "steps": int(args_cli.ros_cmd_vel_demo_steps),
        "initial_pose": robot_planar_pose(robot),
        "final_pose": None,
        "translation_m": 0.0,
        "yaw_delta_rad": 0.0,
        "cmd_samples": [],
        "wheel_joints": wheel_names,
        "success": False,
        "reason": "",
    }
    try:
        bridge.start()
        publisher_process, publisher_log = start_ros2_cmd_vel_test_publisher(cycle_dir, duration_s)
        if publisher_log is not None:
            metadata["test_publisher"]["log"] = str(publisher_log)
        initial_pose = metadata["initial_pose"]
        last_nonzero_cmd = [0.0, 0.0]
        for step in range(max(1, int(args_cli.ros_cmd_vel_demo_steps))):
            vx_raw, wz_raw = bridge.exchange(robot)
            vx, wz = limit_external_nav_cmd(vx_raw, wz_raw, sim_dt)
            if abs(vx) > 1.0e-6 or abs(wz) > 1.0e-6:
                last_nonzero_cmd = [float(vx), float(wz)]
            hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
            apply_wheel_velocity(robot, wheel_ids, vx, wz, sim_dt)
            stabilize_robot_base_for_nav(robot)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim_dt)
            record_video_frame(scene)
            if step % 20 == 0:
                pose = robot_planar_pose(robot)
                sample = {
                    "step": int(step),
                    "pose": pose,
                    "cmd_raw": [float(vx_raw), float(wz_raw)],
                    "cmd_applied": [float(vx), float(wz)],
                    "wheel_state": wheel_state_snapshot(robot, wheel_ids, wheel_names),
                }
                metadata["cmd_samples"].append(sample)
                tracker.event("ros_cmd_vel_progress", **sample)
            tracker.tick()
            if getattr(args_cli, "headless", False):
                time.sleep(max(float(sim_dt), 0.0))
            else:
                gui_playback_tick(sim_dt)
        apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0)
        final_pose = robot_planar_pose(robot)
        metadata["final_pose"] = final_pose
        metadata["last_nonzero_cmd_applied"] = last_nonzero_cmd
        metadata["translation_m"] = float(math.hypot(final_pose[0] - initial_pose[0], final_pose[1] - initial_pose[1]))
        metadata["yaw_delta_rad"] = float(wrap_to_pi(final_pose[2] - initial_pose[2]))
        commanded_linear = abs(float(args_cli.ros_cmd_vel_demo_linear_x)) > 1.0e-6
        commanded_angular = abs(float(args_cli.ros_cmd_vel_demo_angular_z)) > 1.0e-6
        linear_ok = (not commanded_linear) or metadata["translation_m"] >= float(args_cli.ros_cmd_vel_demo_min_translation_m)
        angular_ok = (not commanded_angular) or abs(metadata["yaw_delta_rad"]) >= float(args_cli.ros_cmd_vel_demo_min_yaw_rad)
        if linear_ok and angular_ok and (abs(last_nonzero_cmd[0]) > 1.0e-6 or abs(last_nonzero_cmd[1]) > 1.0e-6):
            metadata["success"] = True
            metadata["reason"] = "ROS2 /cmd_vel moved the robot in Isaac."
            tracker.enter("DONE", reason=metadata["reason"])
        else:
            metadata["success"] = False
            metadata["reason"] = (
                "ROS2 /cmd_vel did not produce the expected robot displacement; "
                "check the ROS bridge, external publisher, and Isaac wheel-ground coupling."
            )
            tracker.enter("RECOVER", reason=metadata["reason"])
    except Exception as exc:
        metadata["success"] = False
        metadata["reason"] = repr(exc)
        metadata["final_pose"] = robot_planar_pose(robot)
        tracker.enter("RECOVER", reason=repr(exc))
    finally:
        apply_wheel_velocity(robot, wheel_ids, 0.0, 0.0)
        bridge.close()
        stop_external_process(publisher_process, "ROS2 cmd_vel test publisher")
        metadata["fsm_trace"] = tracker.trace
        metadata["observer_video"] = _VIDEO_RECORDER.metadata() if _VIDEO_RECORDER is not None else {"enabled": False}
        (cycle_dir / "ros_cmd_vel_demo_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        (cycle_dir / "fsm_trace.json").write_text(json.dumps(tracker.trace, indent=2, ensure_ascii=False))
        (run_dir / "latest_cycle.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"[INFO] ROS cmd_vel demo success={metadata['success']} reason={metadata['reason']} -> {cycle_dir}", flush=True)


def run_dynamic_standpoint_nav_demo(sim: SimulationContext, scene: InteractiveScene, run_dir: Path) -> None:
    if args_cli.nav_backend != "nav2":
        raise RuntimeError("Dynamic standpoint navigation demo supports only --nav_backend nav2.")
    cycle_dir = run_dir / "dynamic_standpoint_nav"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    robot: Articulation = scene["robot"]
    tracker = Task319StateTracker(sim.get_physics_dt(), sim.device)
    tracker.enter("REST", mode="dynamic_standpoint_nav_demo", nav_backend="nav2")
    reset_scene(scene)
    wheel_ids, wheel_names = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
    if len(wheel_ids) != 4:
        raise RuntimeError(f"Dynamic standpoint navigation requires 4 wheel joints, found {wheel_names}.")
    locked_joint_target = robot.data.joint_pos.clone()
    allowed_sides = parse_dynamic_table_sides(args_cli.dynamic_allowed_table_sides)
    bridge = ExternalRos2NavBridgeClient(run_dir)
    nav2_stack_process: subprocess.Popen[Any] | None = None
    candidate_meta = compute_grasp_standpoint_candidates(
        args_cli.dynamic_target_world_xyz,
        robot_planar_pose(robot),
        allowed_sides,
    )
    metadata: dict[str, Any] = {
        "mode": "dynamic_standpoint_nav_demo",
        "vision_paused": True,
        "target_source": "manual_world_xyz",
        "target_world_xyz": [float(v) for v in args_cli.dynamic_target_world_xyz],
        "nav_backend": "nav2",
        "wheel_drive": wheel_drive_metadata(),
        "dynamic_allowed_table_sides": allowed_sides,
        "dynamic_ik_diagnostics": bool(args_cli.dynamic_ik_diagnostics),
        "dynamic_require_ik_reachable": bool(args_cli.dynamic_require_ik_reachable),
        "standpoint_candidates": candidate_meta,
        "navigation": None,
        "initial_pose": robot_planar_pose(robot),
        "final_pose": None,
        "nav2_stack": {
            "auto_start": bool(args_cli.start_nav2_stack),
            "script": args_cli.nav2_stack_script,
            "startup_s": float(args_cli.nav2_stack_startup_s),
            "log": "",
        },
        "nav2_action_name": args_cli.nav2_action_name,
        "nav2_goal_frame": args_cli.nav2_goal_frame,
        "nav_publish_synthetic_obstacles": bool(args_cli.nav_publish_synthetic_obstacles),
        "ros2_bridge": {
            "host": args_cli.ros2_bridge_host,
            "port": int(args_cli.ros2_bridge_port),
            "setup": args_cli.ros2_setup,
            "python": args_cli.ros2_python,
        },
        "success": False,
        "reason": "",
    }
    try:
        candidate_meta = validate_dynamic_standpoint_candidates_with_ik(sim, scene, candidate_meta)
        metadata["standpoint_candidates"] = candidate_meta
        (cycle_dir / "dynamic_standpoint_candidates.json").write_text(json.dumps(candidate_meta, indent=2, ensure_ascii=False))
        selected = candidate_meta.get("selected")
        if selected is None:
            metadata["reason"] = "No dynamic standpoint candidate satisfied geometric and IK reachability constraints."
            tracker.enter("RECOVER", reason=metadata["reason"])
            return

        bridge.start()
        nav2_stack_process, nav2_stack_log = start_nav2_stack(run_dir)
        if nav2_stack_log is not None:
            metadata["nav2_stack"]["log"] = str(nav2_stack_log)
        sim_dt = sim.get_physics_dt()
        for _ in range(args_cli.warmup_steps):
            vx, wz = bridge.exchange(robot)
            vx, wz = limit_external_nav_cmd(vx, wz, sim_dt)
            hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
            apply_wheel_velocity(robot, wheel_ids, vx, wz, sim_dt)
            stabilize_robot_base_for_nav(robot)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim_dt)
            record_video_frame(scene)
            gui_playback_tick(sim_dt)
            tracker.tick()

        target_pose = tuple(float(v) for v in selected["pose"])
        nav_result = run_nav2_goal(
            sim,
            scene,
            robot,
            bridge,
            wheel_ids,
            locked_joint_target,
            tracker,
            "WAYPOINT_NAV",
            f"dynamic_standpoint_{selected['id']}",
            target_pose,
        )
        nav_result["selected_candidate_id"] = selected["id"]
        nav_result["selected_side"] = selected["side"]
        metadata["navigation"] = nav_result
        if nav_result["success"]:
            metadata["success"] = True
            metadata["reason"] = "Dynamic standpoint Nav2 demo completed."
            tracker.enter("DONE", reason=metadata["reason"])
        else:
            metadata["reason"] = f"Nav2 failed for dynamic standpoint {selected['id']}: {nav_result['status']}"
            tracker.enter("RECOVER", reason=metadata["reason"])
    except Exception as exc:
        metadata["success"] = False
        metadata["reason"] = repr(exc)
        tracker.enter("RECOVER", reason=repr(exc))
    finally:
        bridge.close()
        stop_external_process(nav2_stack_process, "Nav2 stack")
        metadata["final_pose"] = robot_planar_pose(robot)
        metadata["nav2_failure_hints"] = nav2_failure_hints(run_dir)
        metadata["fsm_trace"] = tracker.trace
        metadata["observer_video"] = _VIDEO_RECORDER.metadata() if _VIDEO_RECORDER is not None else {"enabled": False}
        (cycle_dir / "dynamic_standpoint_nav_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        (cycle_dir / "dynamic_standpoint_candidates.json").write_text(json.dumps(metadata["standpoint_candidates"], indent=2, ensure_ascii=False))
        (cycle_dir / "fsm_trace.json").write_text(json.dumps(tracker.trace, indent=2, ensure_ascii=False))
        (run_dir / "latest_cycle.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"[INFO] Dynamic standpoint Nav2 demo success={metadata['success']} reason={metadata['reason']} -> {cycle_dir}", flush=True)


def run_waypoint_nav_demo(sim: SimulationContext, scene: InteractiveScene, run_dir: Path) -> None:
    if args_cli.nav_backend != "nav2":
        raise RuntimeError("Waypoint navigation demo supports only --nav_backend nav2.")
    route = parse_waypoint_route(args_cli.waypoint_route)
    cycle_dir = run_dir / "waypoint_nav"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    robot: Articulation = scene["robot"]
    tracker = Task319StateTracker(sim.get_physics_dt(), sim.device)
    tracker.enter("REST", mode="waypoint_nav_demo", nav_backend="nav2", route=route)
    reset_scene(scene)
    wheel_ids, wheel_names = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
    if len(wheel_ids) != 4:
        raise RuntimeError(f"Waypoint navigation requires 4 wheel joints, found {wheel_names}.")
    locked_joint_target = robot.data.joint_pos.clone()
    bridge = ExternalRos2NavBridgeClient(run_dir)
    nav2_stack_process: subprocess.Popen[Any] | None = None
    metadata: dict[str, Any] = {
        "mode": "waypoint_nav_demo",
        "vision_paused": True,
        "nav_backend": "nav2",
        "wheel_drive": wheel_drive_metadata(),
        "waypoints": waypoint_registry_metadata(),
        "route": route,
        "transitions": [],
        "navigation": [],
        "initial_pose": robot_planar_pose(robot),
        "final_pose": None,
        "nav2_stack": {
            "auto_start": bool(args_cli.start_nav2_stack),
            "script": args_cli.nav2_stack_script,
            "startup_s": float(args_cli.nav2_stack_startup_s),
            "log": "",
        },
        "nav2_action_name": args_cli.nav2_action_name,
        "nav2_goal_frame": args_cli.nav2_goal_frame,
        "nav_publish_synthetic_obstacles": bool(args_cli.nav_publish_synthetic_obstacles),
        "ros2_bridge": {
            "host": args_cli.ros2_bridge_host,
            "port": int(args_cli.ros2_bridge_port),
            "setup": args_cli.ros2_setup,
            "python": args_cli.ros2_python,
        },
        "success": False,
        "reason": "",
    }
    try:
        bridge.start()
        nav2_stack_process, nav2_stack_log = start_nav2_stack(run_dir)
        if nav2_stack_log is not None:
            metadata["nav2_stack"]["log"] = str(nav2_stack_log)
        sim_dt = sim.get_physics_dt()
        for _ in range(args_cli.warmup_steps):
            vx, wz = bridge.exchange(robot)
            vx, wz = limit_external_nav_cmd(vx, wz, sim_dt)
            hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
            apply_wheel_velocity(robot, wheel_ids, vx, wz, sim_dt)
            stabilize_robot_base_for_nav(robot)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim_dt)
            record_video_frame(scene)
            gui_playback_tick(sim_dt)
            tracker.tick()

        previous = "current"
        for idx, waypoint in enumerate(route):
            target_pose = waypoint_pose(waypoint)
            label = f"waypoint_{idx:02d}_{previous}_to_{waypoint}".replace(" ", "_")
            nav_result = run_nav2_goal(
                sim,
                scene,
                robot,
                bridge,
                wheel_ids,
                locked_joint_target,
                tracker,
                "WAYPOINT_NAV",
                label,
                target_pose,
            )
            nav_result["from_waypoint"] = previous
            nav_result["to_waypoint"] = waypoint
            metadata["transitions"].append(nav_result)
            metadata["navigation"].append(nav_result)
            if not nav_result["success"]:
                metadata["reason"] = f"Nav2 failed on transition {previous}->{waypoint}: {nav_result['status']}"
                break
            previous = waypoint
        else:
            metadata["success"] = True
            metadata["reason"] = "Waypoint Nav2 demo completed."
            tracker.enter("DONE", reason=metadata["reason"])
    except Exception as exc:
        metadata["success"] = False
        metadata["reason"] = repr(exc)
        tracker.enter("RECOVER", reason=repr(exc))
    finally:
        bridge.close()
        stop_external_process(nav2_stack_process, "Nav2 stack")
        metadata["final_pose"] = robot_planar_pose(robot)
        metadata["nav2_failure_hints"] = nav2_failure_hints(run_dir)
        metadata["fsm_trace"] = tracker.trace
        metadata["observer_video"] = _VIDEO_RECORDER.metadata() if _VIDEO_RECORDER is not None else {"enabled": False}
        (cycle_dir / "waypoint_nav_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        (cycle_dir / "fsm_trace.json").write_text(json.dumps(tracker.trace, indent=2, ensure_ascii=False))
        (run_dir / "latest_cycle.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"[INFO] Waypoint Nav2 demo success={metadata['success']} reason={metadata['reason']} -> {cycle_dir}", flush=True)


def run_motion_only_sort_demo(sim: SimulationContext, scene: InteractiveScene, run_dir: Path) -> None:
    if args_cli.nav_backend != "nav2":
        raise RuntimeError("Motion-only sort demo supports only --nav_backend nav2.")
    cycle_dir = run_dir / "motion_only_nav2"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    robot: Articulation = scene["robot"]
    gripper = AttachedParallelGripper(robot)
    tracker = Task319StateTracker(sim.get_physics_dt(), sim.device)
    tracker.enter("REST", mode="motion_only_sort_demo", nav_backend="nav2")
    reset_scene(scene)
    wheel_ids, wheel_names = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
    if len(wheel_ids) != 4:
        raise RuntimeError(f"Motion-only navigation requires 4 wheel joints, found {wheel_names}.")
    locked_joint_target = robot.data.joint_pos.clone()
    bridge = ExternalRos2NavBridgeClient(run_dir)
    nav2_stack_process: subprocess.Popen[Any] | None = None
    metadata: dict[str, Any] = {
        "mode": "motion_only_sort_demo",
        "vision_paused": True,
        "nav_backend": "nav2",
        "wheel_drive": wheel_drive_metadata(),
        "nav2_stack": {
            "auto_start": bool(args_cli.start_nav2_stack),
            "script": args_cli.nav2_stack_script,
            "startup_s": float(args_cli.nav2_stack_startup_s),
            "log": "",
        },
        "nav2_action_name": args_cli.nav2_action_name,
        "nav2_goal_frame": args_cli.nav2_goal_frame,
        "nav_publish_synthetic_obstacles": bool(args_cli.nav_publish_synthetic_obstacles),
        "ros2_bridge": {
            "host": args_cli.ros2_bridge_host,
            "port": int(args_cli.ros2_bridge_port),
            "setup": args_cli.ros2_setup,
            "python": args_cli.ros2_python,
        },
        "table_standpoints": TABLE_STANDPOINTS,
        "bin_staging_pose": waypoint_pose("bin_center"),
        "return_home_via_bin_staging": True,
        "return_to_bin_staging_yaw_policy": "travel_direction",
        "return_home_side_corridor_y": SIDE_CORRIDOR_Y,
        "return_home_approach_backoff_m": TABLE_RETURN_APPROACH_BACKOFF_M,
        "categories": motion_only_categories(),
        "navigation": [],
        "pick": [],
        "drop": [],
        "success": False,
        "reason": "",
    }
    try:
        bridge.start()
        nav2_stack_process, nav2_stack_log = start_nav2_stack(run_dir)
        if nav2_stack_log is not None:
            metadata["nav2_stack"]["log"] = str(nav2_stack_log)
        sim_dt = sim.get_physics_dt()
        for _ in range(args_cli.warmup_steps):
            vx, wz = bridge.exchange(robot)
            vx, wz = limit_external_nav_cmd(vx, wz, sim_dt)
            hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
            apply_wheel_velocity(robot, wheel_ids, vx, wz, sim_dt)
            stabilize_robot_base_for_nav(robot)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim_dt)
            record_video_frame(scene)
            gui_playback_tick(sim_dt)
            tracker.tick()

        table_pose = waypoint_pose("table_front")
        for category in motion_only_categories():
            category = canonical_category(category)
            nav_table = run_nav2_goal(sim, scene, robot, bridge, wheel_ids, locked_joint_target, tracker, "NAV_TO_TABLE_STANDPOINT", f"table_front_center_{category}", table_pose)
            metadata["navigation"].append(nav_table)
            if not nav_table["success"]:
                metadata["reason"] = f"Nav2 failed to reach table standpoint for {category}: {nav_table['status']}"
                break
            pick = run_motion_stub_grasp(sim, scene, robot, gripper, wheel_ids, locked_joint_target, tracker)
            pick["category"] = category
            metadata["pick"].append(pick)

            nav_staging = run_nav2_goal(sim, scene, robot, bridge, wheel_ids, locked_joint_target, tracker, "NAV_TO_BIN_STAGING", f"bin_staging_{category}", waypoint_pose("bin_center"))
            metadata["navigation"].append(nav_staging)
            if not nav_staging["success"]:
                metadata["reason"] = f"Nav2 failed to reach bin staging for {category}: {nav_staging['status']}"
                break

            tracker.enter("SELECT_BIN_BY_CATEGORY", category=category, target_pose=nav2_bin_pose_for_category(category))
            nav_bin = run_nav2_goal(sim, scene, robot, bridge, wheel_ids, locked_joint_target, tracker, "NAV_TO_BIN", f"bin_{BIN_NAME_BY_CATEGORY.get(category, 'other')}_{category}", nav2_bin_pose_for_category(category))
            metadata["navigation"].append(nav_bin)
            if not nav_bin["success"]:
                metadata["reason"] = f"Nav2 failed to reach target bin for {category}: {nav_bin['status']}"
                break

            drop = run_motion_stub_drop(sim, scene, robot, gripper, wheel_ids, locked_joint_target, tracker)
            drop["category"] = category
            metadata["drop"].append(drop)

            return_staging_pose = waypoint_pose("bin_center")
            nav_return_staging = run_nav2_goal(
                sim,
                scene,
                robot,
                bridge,
                wheel_ids,
                locked_joint_target,
                tracker,
                "RETURN_TO_BIN_STAGING",
                f"return_to_bin_staging_{category}",
                return_staging_pose,
            )
            metadata["navigation"].append(nav_return_staging)
            if not nav_return_staging["success"]:
                metadata["reason"] = f"Nav2 failed to return to bin staging for {category}: {nav_return_staging['status']}"
                break

            corridor_y = SIDE_CORRIDOR_Y if bin_y_for_category(category) >= 0.0 else -SIDE_CORRIDOR_Y
            side_corridor_pose = nav2_pose_with_travel_yaw(robot, (BIN_DRIVE_X, corridor_y))
            nav_side_corridor = run_nav2_goal(
                sim,
                scene,
                robot,
                bridge,
                wheel_ids,
                locked_joint_target,
                tracker,
                "RETURN_TO_SIDE_CORRIDOR",
                f"return_to_side_corridor_{category}",
                side_corridor_pose,
            )
            metadata["navigation"].append(nav_side_corridor)
            if not nav_side_corridor["success"]:
                metadata["reason"] = f"Nav2 failed to reach side corridor for {category}: {nav_side_corridor['status']}"
                break

            home_approach_xy = (table_pose[0] - TABLE_RETURN_APPROACH_BACKOFF_M, corridor_y)
            home_approach_pose = nav2_pose_with_travel_yaw(robot, home_approach_xy)
            nav_home_approach = run_nav2_goal(
                sim,
                scene,
                robot,
                bridge,
                wheel_ids,
                locked_joint_target,
                tracker,
                "RETURN_HOME_APPROACH",
                f"return_home_approach_{category}",
                home_approach_pose,
            )
            metadata["navigation"].append(nav_home_approach)
            if not nav_home_approach["success"]:
                metadata["reason"] = f"Nav2 failed to reach home approach for {category}: {nav_home_approach['status']}"
                break

            home_align_xy = (table_pose[0] - TABLE_RETURN_APPROACH_BACKOFF_M, table_pose[1])
            home_align_pose = nav2_pose_with_travel_yaw(robot, home_align_xy)
            nav_home_align = run_nav2_goal(
                sim,
                scene,
                robot,
                bridge,
                wheel_ids,
                locked_joint_target,
                tracker,
                "RETURN_HOME_ALIGN",
                f"return_home_align_{category}",
                home_align_pose,
            )
            metadata["navigation"].append(nav_home_align)
            if not nav_home_align["success"]:
                metadata["reason"] = f"Nav2 failed to align before home for {category}: {nav_home_align['status']}"
                break

            nav_return = run_nav2_goal(sim, scene, robot, bridge, wheel_ids, locked_joint_target, tracker, "RETURN_HOME", f"return_home_{category}", table_pose)
            metadata["navigation"].append(nav_return)
            if not nav_return["success"]:
                metadata["reason"] = f"Nav2 failed to return home for {category}: {nav_return['status']}"
                break
        else:
            metadata["success"] = True
            metadata["reason"] = "Motion-only Nav2 sort demo completed."
            tracker.enter("DONE", reason=metadata["reason"])
    except Exception as exc:
        metadata["success"] = False
        metadata["reason"] = repr(exc)
        tracker.enter("RECOVER", reason=repr(exc))
    finally:
        bridge.close()
        stop_external_process(nav2_stack_process, "Nav2 stack")
        metadata["fsm_trace"] = tracker.trace
        metadata["observer_video"] = _VIDEO_RECORDER.metadata() if _VIDEO_RECORDER is not None else {"enabled": False}
        (cycle_dir / "motion_only_nav2_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        (cycle_dir / "fsm_trace.json").write_text(json.dumps(tracker.trace, indent=2, ensure_ascii=False))
        (run_dir / "latest_cycle.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"[INFO] Motion-only Nav2 demo success={metadata['success']} reason={metadata['reason']} -> {cycle_dir}", flush=True)


def run_mind_sort_demo(sim: SimulationContext, scene: InteractiveScene, run_dir: Path) -> None:
    if args_cli.nav_backend != "nav2":
        raise RuntimeError("Mind-sort demo supports only --nav_backend nav2.")
    root_dir = run_dir / "mind_sort_demo"
    root_dir.mkdir(parents=True, exist_ok=True)
    robot: Articulation = scene["robot"]
    tracker = Task319StateTracker(sim.get_physics_dt(), sim.device)
    tracker.enter("REST", mode="mind_sort_demo", nav_backend="nav2")
    reset_scene(scene)
    wheel_ids, wheel_names = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
    if len(wheel_ids) != 4:
        raise RuntimeError(f"Mind-sort navigation requires 4 wheel joints, found {wheel_names}.")
    locked_joint_target = robot.data.joint_pos.clone()
    bridge = ExternalRos2NavBridgeClient(run_dir)
    nav2_stack_process: subprocess.Popen[Any] | None = None
    completed_scene_keys: set[str] = set()
    failed_scene_keys: set[str] = set()
    cycles: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "mode": "mind_sort_demo",
        "description": "Visual multi-object sorting with Nav2; requires real physical grasp/lift before bin navigation unless --mind_sort_simulated_pick is explicitly enabled.",
        "perception_source": args_cli.perception_source,
        "physical_grasp": {
            "enabled": bool(args_cli.mind_sort_physical_grasp),
            "grasp_proposal": str(args_cli.mind_sort_grasp_proposal),
            "grasp_backend": str(args_cli.grasp_backend),
            "rgbd_center_grasp_base": bool(args_cli.rgbd_center_grasp_base),
            "centroid_fallback_enabled": bool(centroid_fallback_enabled()),
            "graspnet_force_objectness_top_k": int(args_cli.graspnet_force_objectness_top_k),
            "wrist_refine": bool(args_cli.wrist_refine_grasp),
            "max_retries": int(args_cli.mind_sort_grasp_retries),
            "force_stable_grasp_profile": bool(args_cli.mind_sort_force_stable_grasp_profile),
            "motion_backend": str(args_cli.arm_motion_backend),
            "motion_profile": STABLE_GRASP_MOTION_PROFILE if args_cli.arm_motion_backend == "local_position_primitive" else str(args_cli.arm_motion_backend),
            "curobo_grasp_local_tcp_descent": bool(args_cli.curobo_grasp_local_tcp_descent),
            "curobo_grasp_servo_descent": bool(args_cli.curobo_grasp_servo_descent),
            "curobo_grasp_servo_stop_distance_m": float(args_cli.curobo_grasp_servo_stop_distance_m),
            "curobo_lift_local_tcp_ascent": bool(args_cli.curobo_lift_local_tcp_ascent),
            "gripper_post_close_hold_steps": int(args_cli.gripper_post_close_hold_steps),
            "curobo_prefer_kuavo_seed_for_position_only": bool(args_cli.curobo_prefer_kuavo_seed_for_position_only),
            "rgbd_top_grasp_directionless_envelope": bool(args_cli.rgbd_top_grasp_directionless_envelope),
            "rgbd_top_grasp_envelope_jaw_axis": [float(v) for v in args_cli.rgbd_top_grasp_envelope_jaw_axis],
            "rgbd_top_grasp_object_aware_orientation": bool(args_cli.rgbd_top_grasp_object_aware_orientation),
            "rgbd_top_grasp_enforce_orientation": bool(args_cli.rgbd_top_grasp_enforce_orientation),
            "action_generation_logic": (
                "graspnet_baseline_v28_target_mask_current_standpoint_rgbd_to_curobo_position_only"
                if str(args_cli.mind_sort_grasp_proposal) == "graspnet_baseline"
                else (
                    "v28_rgbd_directionless_envelope_max_open_fixed_jaw_curobo_hover_then_position_servo_descent"
                    if bool(args_cli.rgbd_top_grasp_directionless_envelope)
                    else "v28_rgbd_top_grasp_pca_jaw_axis_to_kuavo_seed_curobo_hover_then_fixed_wrist_vertical_descent"
                )
            ),
        },
        "simulated_pick": {
            "enabled": bool(args_cli.mind_sort_simulated_pick),
            "fallback": False,
        },
        "gripper_proximity_assist": {
            "enabled": bool(args_cli.mind_sort_gripper_proximity_assist or args_cli.mind_sort_suction_assist),
            "after_physical_failure_only": True,
            "max_distance_m": float(args_cli.mind_sort_gripper_proximity_assist_max_distance_m),
            "steps": int(args_cli.mind_sort_gripper_proximity_assist_steps) if int(args_cli.mind_sort_gripper_proximity_assist_steps) > 0 else int(args_cli.grasp_steps),
            "report_label": "gripper proximity assisted pickup",
            "carry_frame": "right_gripper_tcp",
            "visual_policy": "object_root_keeps_low_grasp_tcp_local_offset_so_it_appears_inside_the_closed_gripper",
            "drop_policy": "before_release_align_object_to_the_selected_hardcoded_bin_opening_then_open_gripper",
        },
        "suction_assist": {
            "enabled": bool(args_cli.mind_sort_suction_assist),
            "legacy_alias_for_gripper_proximity_assist": True,
            "max_distance_m": float(args_cli.mind_sort_suction_assist_max_distance_m),
            "steps": int(args_cli.mind_sort_suction_assist_steps) if int(args_cli.mind_sort_suction_assist_steps) > 0 else int(args_cli.grasp_steps),
            "report_label": "legacy suction alias",
        },
        "nav_backend": "nav2",
        "wheel_drive": wheel_drive_metadata(),
        "wheel_joints": wheel_names,
        "observe_pose": [float(v) for v in waypoint_pose("home")],
        "bin_staging_pose": [float(v) for v in waypoint_pose("bin_center")],
        "bin_layout": sort_bin_layout_metadata(),
        "max_objects": int(args_cli.mind_sort_max_objects),
        "settle_steps": int(args_cli.mind_sort_settle_steps),
        "legacy_simulated_pick_attach_offset": {
            "forward_m": float(args_cli.mind_sort_attach_forward_m),
            "lateral_m": float(args_cli.mind_sort_attach_lateral_m),
            "height_world_m": float(args_cli.mind_sort_attach_height_m),
        },
        "nav2_stack": {
            "auto_start": bool(args_cli.start_nav2_stack),
            "script": args_cli.nav2_stack_script,
            "startup_s": float(args_cli.nav2_stack_startup_s),
            "log": "",
        },
        "navigation": [],
        "cycles": cycles,
        "completed_scene_keys": [],
        "failed_scene_keys": [],
        "success": False,
        "reason": "",
    }
    try:
        bridge.start()
        nav2_stack_process, nav2_stack_log = start_nav2_stack(run_dir)
        if nav2_stack_log is not None:
            metadata["nav2_stack"]["log"] = str(nav2_stack_log)
        sim_dt = sim.get_physics_dt()
        for _ in range(args_cli.warmup_steps):
            vx, wz = bridge.exchange(robot)
            vx, wz = limit_external_nav_cmd(vx, wz, sim_dt)
            hold_non_wheel_joints(robot, locked_joint_target, wheel_ids)
            apply_wheel_velocity(robot, wheel_ids, vx, wz, sim_dt)
            stabilize_robot_base_for_nav(robot)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim_dt)
            record_video_frame(scene)
            gui_playback_tick(sim_dt)
            tracker.tick()

        max_objects = int(args_cli.mind_sort_max_objects)
        cycle_index = 0
        while simulation_app.is_running():
            if max_objects > 0 and cycle_index >= max_objects:
                metadata["success"] = len(failed_scene_keys) == 0
                metadata["reason"] = (
                    f"Reached --mind_sort_max_objects={max_objects}."
                    if metadata["success"]
                    else f"Reached --mind_sort_max_objects={max_objects} with physical grasp failures: {sorted(failed_scene_keys)}."
                )
                tracker.enter("DONE", reason=metadata["reason"])
                break

            cycle_dir = root_dir / f"cycle_{cycle_index:04d}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            cycle_meta: dict[str, Any] = {
                "cycle_index": cycle_index,
                "completed_scene_keys_before": sorted(completed_scene_keys),
                "failed_scene_keys_before": sorted(failed_scene_keys),
                "navigation": [],
                "success": False,
                "reason": "",
            }
            cycles.append(cycle_meta)

            observe_pose = waypoint_pose("home")
            tracker.enter("OBSERVE_TABLE_BACKOFF", target_pose=observe_pose)
            nav_observe = run_nav2_goal(
                sim,
                scene,
                robot,
                bridge,
                wheel_ids,
                locked_joint_target,
                tracker,
                "OBSERVE_TABLE_BACKOFF",
                f"mind_observe_backoff_{cycle_index:04d}",
                observe_pose,
            )
            cycle_meta["navigation"].append(nav_observe)
            metadata["navigation"].append(nav_observe)
            if not nav_observe.get("success", False):
                cycle_meta["reason"] = f"Nav2 failed to reach observe pose: {nav_observe.get('status')}"
                metadata["reason"] = cycle_meta["reason"]
                tracker.enter("RECOVER", reason=metadata["reason"])
                break
            cycle_meta["observe_pose_snap_before_perception"] = mind_sort_snap_to_observe_pose(
                sim,
                scene,
                robot,
                wheel_ids,
                locked_joint_target,
                tracker,
                "before_table_perception",
            )

            tracker.enter("PERCEIVE_TABLE_OBJECTS", cycle_index=cycle_index)
            perception = mind_sort_perception_pass(scene, cycle_dir, completed_scene_keys=completed_scene_keys | failed_scene_keys, prefix="observe")
            cycle_meta["perception"] = {
                "instance_count": len(perception["instances"]),
                "filtered_instance_count": len(perception["filtered_instances"]),
                "rgb": perception["rgb_path"],
                "instances": perception["instance_records"],
                "depth_instances": perception["depth_instance_records"],
                "filtered_instances": perception["filtered_instance_records"],
                "rejected_instances": perception["rejected_instance_records"],
            }

            tracker.enter("PLAN_GRASP_STANDPOINT", cycle_index=cycle_index)
            target, target_reason, standpoint_meta = plan_dynamic_grasp_standpoint(
                scene,
                perception["filtered_instances"],
                perception["rgb"],
                perception["intrinsics"],
                perception["camera_to_world"],
                cycle_dir,
            )
            cycle_meta["standpoint_plan"] = standpoint_meta
            cycle_meta["target_reason"] = target_reason
            if target is None or standpoint_meta.get("selected") is None:
                cycle_meta["reason"] = target_reason
                metadata["success"] = True
                metadata["reason"] = "No remaining valid table object for mind-sort demo."
                tracker.enter("DONE", reason=metadata["reason"])
                break

            tracker.enter("SELECT_NEXT_OBJECT", cycle_index=cycle_index, target=target_candidate_record(target))
            scene_key = target.scene_key
            scene_object_name = target.scene_object_name
            if scene_key is None:
                scene_key, scene_object_name, match_distance = locate_target_rigid_object(scene, target.center_3d)
                cycle_meta["scene_match_fallback"] = {
                    "scene_key": scene_key,
                    "scene_object_name": scene_object_name,
                    "match_distance_m": float(match_distance),
                }
            if scene_key is None or scene_key not in scene.keys():
                cycle_meta["reason"] = "Selected visual target could not be matched to a simulated rigid object for mind-pick attach."
                metadata["reason"] = cycle_meta["reason"]
                tracker.enter("RECOVER", reason=metadata["reason"])
                break
            category = canonical_category(target.vlm.waste_category if target.vlm else category_for_scene_key(scene_key))
            cycle_meta["selected_target"] = target_candidate_record(target)
            cycle_meta["selected_scene_key"] = scene_key
            cycle_meta["selected_scene_object_name"] = scene_object_name
            cycle_meta["category"] = category
            cycle_meta["selected_bin"] = {
                "category": category,
                "bin_name": BIN_NAME_BY_CATEGORY.get(category, "other"),
                "nav2_standpoint_pose": [float(v) for v in nav2_bin_pose_for_category(category)],
                "drop_pose_world": bin_drop_position_for_category(category).astype(float).tolist(),
                "opening_bounds": bin_opening_bounds_for_category(category),
            }

            selected_standpoint = dict(standpoint_meta.get("selected") or {})
            table_pose = mind_sort_table_pose_from_selected(selected_standpoint)
            selected_standpoint["table_facing_nav_pose"] = [float(v) for v in table_pose]
            cycle_meta["selected_standpoint"] = selected_standpoint
            nav_table = run_nav2_goal(
                sim,
                scene,
                robot,
                bridge,
                wheel_ids,
                locked_joint_target,
                tracker,
                "NAV_TO_TABLE_STANDPOINT",
                f"mind_table_{cycle_index:04d}_{scene_key}",
                table_pose,
                final_dock=bool(args_cli.mind_sort_physical_grasp),
            )
            cycle_meta["navigation"].append(nav_table)
            metadata["navigation"].append(nav_table)
            if not nav2_goal_reached_or_soft_docked(nav_table):
                cycle_meta["reason"] = f"Nav2 failed to reach dynamic table standpoint: {nav_table.get('status')}"
                metadata["reason"] = cycle_meta["reason"]
                tracker.enter("RECOVER", reason=metadata["reason"])
                break
            if bool(args_cli.mind_sort_physical_grasp):
                standpoint_error = float(nav_table.get("position_error_m", float("inf")))
                max_standpoint_error = float(args_cli.mind_sort_physical_max_standpoint_error_m)
                if not math.isfinite(standpoint_error) or standpoint_error > max_standpoint_error:
                    reason = (
                        "Physical grasp skipped because Nav2 stopped outside the allowed pre-grasp standpoint error: "
                        f"{standpoint_error:.3f} m > {max_standpoint_error:.3f} m."
                    )
                    failed_scene_keys.add(scene_key)
                    metadata["failed_scene_keys"] = sorted(failed_scene_keys)
                    cycle_meta["failed_scene_keys_after"] = sorted(failed_scene_keys)
                    cycle_meta["success"] = False
                    cycle_meta["reason"] = reason
                    cycle_meta["physical_grasp"] = {
                        "enabled": True,
                        "success": False,
                        "grasp_success": False,
                        "drop_success": False,
                        "reason": reason,
                        "blocked_before_grasp": True,
                        "nav_table_position_error_m": standpoint_error,
                        "max_standpoint_error_m": max_standpoint_error,
                    }
                    tracker.enter(
                        "RECOVER",
                        reason=reason,
                        nav_table_position_error_m=standpoint_error,
                        max_standpoint_error_m=max_standpoint_error,
                    )
                    (cycle_dir / "mind_sort_cycle.json").write_text(json.dumps(cycle_meta, indent=2, ensure_ascii=False))
                    (root_dir / "mind_sort_task_queue.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
                    cycle_index += 1
                    continue

            cycle_meta["settle"] = mind_sort_settle_at_table(sim, scene, robot, wheel_ids, locked_joint_target, tracker, table_pose)
            tracker.enter("RESHOOT_AT_STANDPOINT", cycle_index=cycle_index)
            reshoot_dir = cycle_dir / "standpoint_reshoot"
            reshoot = mind_sort_perception_pass(scene, reshoot_dir, completed_scene_keys=completed_scene_keys | failed_scene_keys, prefix="reshoot")
            frozen_identity = freeze_target_identity(target)
            reshoot_reachability_records = []
            if bool(args_cli.mind_sort_physical_grasp) and bool(args_cli.mind_sort_physical_reselect_after_reshoot):
                reshoot_ik_pool, reshoot_ik_pool_meta = select_reshoot_ik_pool(reshoot["filtered_instances"], robot, reshoot_dir)
                if bool(args_cli.target_reachability_ik_check):
                    reshoot_reachability_records = evaluate_right_arm_target_reachability(sim, scene, reshoot_ik_pool, reshoot_dir)
                    reachable_reshoot_instances = [
                        inst
                        for inst in reshoot_ik_pool
                        if bool(instance_reachability_metadata(inst).get("reachable", False))
                    ]
                    reachability_policy = "current_root_geometry_top_k_then_right_arm_ik"
                else:
                    geometry_records = reshoot_ik_pool_meta.get("selected_for_ik", [])
                    geometry_by_index = {int(item.get("instance_index", -1)): item for item in geometry_records}
                    reshoot_reachability_records = []
                    reachable_reshoot_instances = []
                    for inst in reshoot_ik_pool:
                        geom = geometry_by_index.get(int(inst.index), current_root_geometric_reachability_record(inst, robot))
                        reachability = {
                            "method": "current_root_geometry_window",
                            "reachable": bool(geom.get("reachable", False)),
                            "reason": "inside_current_root_right_arm_window" if bool(geom.get("reachable", False)) else str(geom.get("reject_reason", "")),
                            "reach_cost": float(geom.get("score", 1.0e6)),
                            "relative_to_robot_m": {
                                "dx": float((geom.get("target_local_xy") or [float("nan"), float("nan")])[0]),
                                "dy": float((geom.get("target_local_xy") or [float("nan"), float("nan")])[1]),
                                "z_world": float(geom.get("target_local_z_world_m", float("nan"))),
                                "source": "current_robot_root_after_navigation_geometry",
                            },
                            "frame_diagnostics": point_world_frame_diagnostics(robot, inst.center_3d, resolve_right_ee_body_id(scene)),
                            "geometric_record": geom,
                            "ik_check_enabled": False,
                        }
                        inst.right_arm_reachability = reachability
                        if bool(reachability.get("reachable", False)):
                            reachable_reshoot_instances.append(inst)
                        reshoot_reachability_records.append({
                            "instance_index": int(inst.index),
                            "source": inst.source,
                            "yolo_label": inst.yolo_label,
                            "scene_key": inst.scene_key,
                            "scene_object_name": inst.scene_object_name,
                            "center_3d_world": inst.center_3d.tolist() if inst.center_3d is not None else None,
                            "reachability": reachability,
                        })
                    (reshoot_dir / "right_arm_reachability.json").write_text(json.dumps(reshoot_reachability_records, indent=2, ensure_ascii=False))
                    reachability_policy = "current_root_geometry_top_k_no_ik_prescreen"
                target2, choose_reason = choose_target(reachable_reshoot_instances, reshoot_dir)
                reshoot_match_meta = {
                    "policy": reachability_policy,
                    "ik_candidate_pool": reshoot_ik_pool_meta,
                    "original_identity": target_identity_metadata(frozen_identity),
                    "right_arm_reachability": reshoot_reachability_records,
                    "reachable_instance_count": len(reachable_reshoot_instances),
                    "selected": target_candidate_record(target2) if target2 is not None else None,
                    "reason": choose_reason,
                }
                target_reason2 = f"standpoint_reshoot_reachable_reselect:{choose_reason}"
                (reshoot_dir / "reshoot_target_match.json").write_text(json.dumps(reshoot_match_meta, indent=2, ensure_ascii=False))
            else:
                target2, target_reason2, reshoot_match_meta = choose_reshoot_target(
                    reshoot["filtered_instances"],
                    frozen_identity,
                    target,
                    reshoot_dir,
                )
            cycle_meta["reshoot"] = {
                "instance_count": len(reshoot["instances"]),
                "filtered_instance_count": len(reshoot["filtered_instances"]),
                "rgb": reshoot["rgb_path"],
                "selected": target_candidate_record(target2) if target2 is not None else None,
                "target_reason": target_reason2,
                "match": reshoot_match_meta,
                "right_arm_reachability": reshoot_reachability_records,
            }
            if (
                bool(args_cli.mind_sort_physical_grasp)
                and target2 is None
            ):
                reason = "Post-navigation standpoint reshoot did not provide a current-frame target for physical grasp; refusing stale observe-frame coordinates."
                failed_scene_keys.add(scene_key)
                metadata["failed_scene_keys"] = sorted(failed_scene_keys)
                cycle_meta["failed_scene_keys_after"] = sorted(failed_scene_keys)
                cycle_meta["success"] = False
                cycle_meta["reason"] = reason
                cycle_meta["physical_grasp"] = {
                    "enabled": True,
                    "success": False,
                    "grasp_success": False,
                    "drop_success": False,
                    "reason": reason,
                    "blocked_before_grasp": True,
                    "reshoot_target_reason": target_reason2,
                    "reshoot_match": reshoot_match_meta,
                    "right_arm_reachability": reshoot_reachability_records,
                }
                tracker.enter("RECOVER", reason=reason)
                (cycle_dir / "mind_sort_cycle.json").write_text(json.dumps(cycle_meta, indent=2, ensure_ascii=False))
                (root_dir / "mind_sort_task_queue.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
                cycle_index += 1
                continue

            target_for_grasp = target2 if target2 is not None else target
            if target_for_grasp is not target:
                retarget_scene_key = target_for_grasp.scene_key
                retarget_scene_object_name = target_for_grasp.scene_object_name
                retarget_match_distance = None
                if retarget_scene_key is None:
                    retarget_scene_key, retarget_scene_object_name, retarget_match_distance = locate_target_rigid_object(
                        scene,
                        target_for_grasp.center_3d,
                    )
                if retarget_scene_key is not None:
                    scene_key = retarget_scene_key
                    scene_object_name = retarget_scene_object_name
                    category = canonical_category(target_for_grasp.vlm.waste_category if target_for_grasp.vlm else category_for_scene_key(scene_key))
                    cycle_meta["reshoot_retarget"] = {
                        "enabled": True,
                        "scene_key": scene_key,
                        "scene_object_name": scene_object_name,
                        "category": category,
                        "match_distance_m": None if retarget_match_distance is None else float(retarget_match_distance),
                        "target": target_candidate_record(target_for_grasp),
                    }
                else:
                    cycle_meta["reshoot_retarget"] = {
                        "enabled": True,
                        "success": False,
                        "reason": "reachable reshoot target could not be matched to a simulated rigid object",
                        "target": target_candidate_record(target_for_grasp),
                    }
            carried_by_gripper_proximity_assist = False
            if bool(args_cli.mind_sort_physical_grasp):
                cycle_meta["physical_grasp_posture_reset"] = mind_sort_prepare_physical_grasp_posture(
                    sim,
                    scene,
                    robot,
                    wheel_ids,
                    locked_joint_target,
                    tracker,
                )
                physical_grasp = mind_sort_execute_physical_grasp(
                    sim,
                    scene,
                    robot,
                    wheel_ids,
                    locked_joint_target,
                    tracker,
                    target_for_grasp,
                    scene_key,
                    cycle_dir,
                )
                cycle_meta["physical_grasp"] = physical_grasp
                if not bool(physical_grasp.get("grasp_success", False)):
                    proximity_attach = mind_sort_suction_assist_attach(
                        sim,
                        scene,
                        robot,
                        wheel_ids,
                        locked_joint_target,
                        tracker,
                        scene_key,
                        target_for_grasp,
                        physical_grasp,
                    )
                    cycle_meta["gripper_proximity_assist_attach"] = proximity_attach
                    cycle_meta["suction_assist_attach"] = proximity_attach
                    if not bool(proximity_attach.get("success", False)):
                        failed_scene_keys.add(scene_key)
                        metadata["failed_scene_keys"] = sorted(failed_scene_keys)
                        cycle_meta["failed_scene_keys_after"] = sorted(failed_scene_keys)
                        cycle_meta["success"] = False
                        cycle_meta["reason"] = f"Physical grasp failed and gripper proximity assist unavailable: {physical_grasp.get('reason', '')}; {proximity_attach.get('reason', '')}"
                        (cycle_dir / "mind_sort_cycle.json").write_text(json.dumps(cycle_meta, indent=2, ensure_ascii=False))
                        (root_dir / "mind_sort_task_queue.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
                        cycle_index += 1
                        continue
                    carried_by_gripper_proximity_assist = True
                    carry_hook = lambda key=scene_key: mind_sort_update_carried_object(scene, robot, key)
                else:
                    carry_hook = None
            elif bool(args_cli.mind_sort_simulated_pick):
                attach = mind_sort_attach_object(sim, scene, robot, wheel_ids, locked_joint_target, tracker, scene_key)
                cycle_meta["mind_pick_attach"] = attach
                if not attach.get("success", False):
                    cycle_meta["reason"] = "Mind-pick attach failed."
                    metadata["reason"] = cycle_meta["reason"]
                    tracker.enter("RECOVER", reason=metadata["reason"])
                    break
                carry_hook = lambda key=scene_key: mind_sort_update_carried_object(scene, robot, key)
            else:
                cycle_meta["reason"] = "No pickup mode enabled: physical grasp is disabled and simulated pick was not explicitly requested."
                metadata["reason"] = cycle_meta["reason"]
                tracker.enter("RECOVER", reason=metadata["reason"])
                break

            nav_staging = run_nav2_goal(
                sim,
                scene,
                robot,
                bridge,
                wheel_ids,
                locked_joint_target,
                tracker,
                "NAV_TO_BIN_STAGING",
                f"mind_bin_staging_{cycle_index:04d}_{category}",
                waypoint_pose("bin_center"),
                step_hook=carry_hook,
            )
            cycle_meta["navigation"].append(nav_staging)
            metadata["navigation"].append(nav_staging)
            if not nav_staging.get("success", False):
                cycle_meta["reason"] = f"Nav2 failed to reach bin staging while carrying: {nav_staging.get('status')}"
                metadata["reason"] = cycle_meta["reason"]
                tracker.enter("RECOVER", reason=metadata["reason"])
                break

            tracker.enter("SELECT_BIN_BY_CATEGORY", category=category, target_pose=nav2_bin_pose_for_category(category))
            nav_bin = run_nav2_goal(
                sim,
                scene,
                robot,
                bridge,
                wheel_ids,
                locked_joint_target,
                tracker,
                "NAV_TO_BIN",
                f"mind_bin_{cycle_index:04d}_{BIN_NAME_BY_CATEGORY.get(category, 'other')}",
                nav2_bin_pose_for_category(category),
                step_hook=carry_hook,
            )
            cycle_meta["navigation"].append(nav_bin)
            metadata["navigation"].append(nav_bin)
            if not nav_bin.get("success", False):
                cycle_meta["reason"] = f"Nav2 failed to reach target bin while carrying: {nav_bin.get('status')}"
                metadata["reason"] = cycle_meta["reason"]
                tracker.enter("RECOVER", reason=metadata["reason"])
                break

            if bool(args_cli.mind_sort_physical_grasp) and not carried_by_gripper_proximity_assist:
                drop = mind_sort_release_physical_object(sim, scene, robot, wheel_ids, locked_joint_target, tracker, scene_key, category)
                cycle_meta["physical_drop_release"] = drop
            elif bool(args_cli.mind_sort_simulated_pick) or carried_by_gripper_proximity_assist:
                drop = mind_sort_drop_object(sim, scene, robot, wheel_ids, locked_joint_target, tracker, scene_key, category)
                if carried_by_gripper_proximity_assist:
                    cycle_meta["gripper_proximity_assist_drop_release"] = drop
                    cycle_meta["suction_assist_drop_release"] = drop
                else:
                    cycle_meta["mind_drop_release"] = drop
            else:
                cycle_meta["reason"] = "No drop mode enabled after pickup."
                metadata["reason"] = cycle_meta["reason"]
                tracker.enter("RECOVER", reason=metadata["reason"])
                break
            nav_return_observe = run_nav2_goal(
                sim,
                scene,
                robot,
                bridge,
                wheel_ids,
                locked_joint_target,
                tracker,
                "RETURN_TO_OBSERVE_BACKOFF",
                f"mind_return_observe_{cycle_index:04d}_{scene_key}",
                waypoint_pose("home"),
            )
            cycle_meta["navigation"].append(nav_return_observe)
            metadata["navigation"].append(nav_return_observe)
            if not nav_return_observe.get("success", False):
                cycle_meta["reason"] = f"Nav2 failed to return to observe pose after drop: {nav_return_observe.get('status')}"
                metadata["reason"] = cycle_meta["reason"]
                tracker.enter("RECOVER", reason=metadata["reason"])
                break
            cycle_meta["observe_pose_snap_after_drop"] = mind_sort_snap_to_observe_pose(
                sim,
                scene,
                robot,
                wheel_ids,
                locked_joint_target,
                tracker,
                "after_drop_return_before_next_cycle",
            )
            completed_scene_keys.add(scene_key)
            cycle_meta["completed_scene_keys_after"] = sorted(completed_scene_keys)
            cycle_meta["failed_scene_keys_after"] = sorted(failed_scene_keys)
            cycle_meta["success"] = True
            cycle_meta["reason"] = "Mind-sort cycle completed."
            (cycle_dir / "mind_sort_cycle.json").write_text(json.dumps(cycle_meta, indent=2, ensure_ascii=False))
            (root_dir / "mind_sort_task_queue.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
            cycle_index += 1
        else:
            metadata["success"] = True
            metadata["reason"] = "Simulation stopped after completing available mind-sort loop iterations."
            tracker.enter("DONE", reason=metadata["reason"])
    except Exception as exc:
        metadata["success"] = False
        metadata["reason"] = repr(exc)
        tracker.enter("RECOVER", reason=repr(exc))
    finally:
        bridge.close()
        stop_external_process(nav2_stack_process, "Nav2 stack")
        metadata["completed_scene_keys"] = sorted(completed_scene_keys)
        metadata["failed_scene_keys"] = sorted(failed_scene_keys)
        metadata["final_pose"] = robot_planar_pose(robot)
        metadata["nav2_failure_hints"] = nav2_failure_hints(run_dir)
        metadata["fsm_trace"] = tracker.trace
        metadata["observer_video"] = _VIDEO_RECORDER.metadata() if _VIDEO_RECORDER is not None else {"enabled": False}
        (root_dir / "mind_sort_task_queue.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        (root_dir / "fsm_trace.json").write_text(json.dumps(tracker.trace, indent=2, ensure_ascii=False))
        (run_dir / "latest_cycle.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"[INFO] Mind-sort demo success={metadata['success']} reason={metadata['reason']} -> {root_dir}", flush=True)


def hold_final_scene(sim: SimulationContext, scene: InteractiveScene, robot: Articulation) -> None:
    if args_cli.post_action_hold_steps <= 0:
        return
    sim_dt = sim.get_physics_dt()
    print(f"[INFO] Holding final scene for {args_cli.post_action_hold_steps} GUI playback steps.", flush=True)
    for _ in range(args_cli.post_action_hold_steps):
        stabilize_robot_base(robot)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)


def run_loop(sim: SimulationContext, scene: InteractiveScene, run_dir: Path) -> None:
    if args_cli.ros_cmd_vel_demo:
        run_ros_cmd_vel_demo(sim, scene, run_dir)
        hold_final_scene(sim, scene, scene["robot"])
        return
    if args_cli.wheel_open_loop_demo:
        run_wheel_open_loop_demo(sim, scene, run_dir)
        hold_final_scene(sim, scene, scene["robot"])
        return
    if args_cli.dynamic_standpoint_nav_demo:
        run_dynamic_standpoint_nav_demo(sim, scene, run_dir)
        hold_final_scene(sim, scene, scene["robot"])
        return
    if args_cli.waypoint_nav_demo:
        run_waypoint_nav_demo(sim, scene, run_dir)
        hold_final_scene(sim, scene, scene["robot"])
        return
    if args_cli.motion_only_sort_demo:
        run_motion_only_sort_demo(sim, scene, run_dir)
        hold_final_scene(sim, scene, scene["robot"])
        return
    if args_cli.mind_sort_demo:
        run_mind_sort_demo(sim, scene, run_dir)
        hold_final_scene(sim, scene, scene["robot"])
        return
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    reset_scene(scene)
    for _ in range(args_cli.warmup_steps):
        stabilize_robot_base(robot)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
    cycle = 0
    steps_since_capture = args_cli.cycle_interval_steps
    while simulation_app.is_running():
        stabilize_robot_base(robot)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        record_video_frame(scene)
        gui_playback_tick(sim_dt)
        steps_since_capture += 1
        if steps_since_capture >= args_cli.cycle_interval_steps:
            should_stop = write_cycle(sim, scene, run_dir, cycle)
            cycle += 1
            steps_since_capture = 0
            if should_stop or (args_cli.num_cycles > 0 and cycle >= args_cli.num_cycles):
                hold_final_scene(sim, scene, robot)
                break


def main() -> None:
    global _VIDEO_RECORDER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (WORKSPACE_ROOT / args_cli.output_dir / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 120.0, physics_material=high_friction_material())
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.15, 4.65, 3.10], [2.05, 0.0, 0.62])
    scene_cfg_cls = IsolatedDebugCubeGraspSceneCfg if bool(args_cli.debug_cube_isolated_scene) else HeadCameraGraspSceneCfg
    scene = InteractiveScene(scene_cfg_cls(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False))
    if not bool(args_cli.debug_cube_isolated_scene):
        apply_ycb_visual_runtime_physics()
    apply_gripper_contact_material()
    sim.reset()
    if args_cli.debug_reposition_scene_object:
        scene_key, scene_object_name = scene_key_and_name_for_alias(args_cli.debug_reposition_scene_object)
        if scene_key is None or scene_key not in scene.keys():
            raise ValueError(f"Unknown --debug_reposition_scene_object: {args_cli.debug_reposition_scene_object}")
        debug_pos = tuple(float(v) for v in args_cli.debug_reposition_scene_object_pos)
        debug_yaw = float(args_cli.debug_reposition_scene_object_yaw)
        write_rigid_object_pose(scene, scene_key, debug_pos, debug_yaw)
        scene.write_data_to_sim()
        print(
            "[DEBUG] Repositioned scene object "
            f"{scene_key}/{scene_object_name} to pos={debug_pos}, yaw={debug_yaw:.3f} rad.",
            flush=True,
        )
    _VIDEO_RECORDER = ObserverVideoRecorder(run_dir)
    print(f"[INFO] Recording head-camera visual grasp chain to: {run_dir}", flush=True)
    print("[INFO] Camera source is fixed to scene['head_rgbd']; YOLO labels are weak and VLM provides object/category labels.", flush=True)
    print(
        "[INFO] Wheel/nav constraints: "
        f"max_vx={float(args_cli.nav_max_linear_speed):.2f} m/s, "
        f"max_wz={float(args_cli.nav_max_angular_speed):.2f} rad/s, "
        f"angular_scale={float(args_cli.nav_cmd_angular_scale):.2f}.",
        flush=True,
    )
    run_failed = False
    try:
        run_loop(sim, scene, run_dir)
    except BaseException:
        run_failed = True
        raise
    finally:
        video_saved = False
        if _VIDEO_RECORDER is not None:
            _VIDEO_RECORDER.finalize()
            video_saved = bool(_VIDEO_RECORDER.enabled and _VIDEO_RECORDER.path.exists())
        if bool(args_cli.exit_after_video_saved) and bool(args_cli.record_video) and video_saved and not run_failed:
            print("[INFO] Exiting directly after observer video saved; skipping slow Isaac/Kit shutdown. Use --no-exit_after_video_saved to disable.", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
    simulation_app.close()
