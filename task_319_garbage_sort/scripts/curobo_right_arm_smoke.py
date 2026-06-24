#!/usr/bin/env python3
"""Standalone cuRobo smoke test for the Task 319 Kuavo right arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from task_319_garbage_sort.curobo_right_arm import (  # noqa: E402
    RIGHT_ARM_CARRY_CONFIG,
    RIGHT_ARM_JOINT_NAMES,
    KuavoRightArmCuroboPlanner,
)


def parse_xyz(value: str) -> tuple[float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,z.")
    return (parts[0], parts[1], parts[2])


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test cuRobo on the Kuavo S62 right arm.")
    parser.add_argument("--kuavo_base_urdf", default=str(WORKSPACE_ROOT / "kuavo-ros-opensource/src/kuavo_assets/models/biped_s62/urdf/biped_s62.urdf"))
    parser.add_argument("--gripper_urdf", default=str(WORKSPACE_ROOT / "task_319_garbage_sort/two_finger_gripper.urdf"))
    parser.add_argument("--tcp_offset", type=parse_xyz, default=(0.115, 0.0, 0.0), help="TCP offset in gripper_base local frame; the URDF mount rotates gripper local +X inline with wrist local -Z.")
    parser.add_argument("--target_delta", type=parse_xyz, default=(0.03, 0.02, 0.04))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_warmup", action="store_true")
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--output", default="task_319_garbage_sort/output/curobo_right_arm_smoke.json")
    args = parser.parse_args()

    planner = KuavoRightArmCuroboPlanner(
        Path(args.kuavo_base_urdf),
        Path(args.gripper_urdf),
        tcp_offset_m=args.tcp_offset,
        device=args.device,
        warmup=not args.no_warmup,
        use_cuda_graph=False,
    )
    q0 = np.asarray(RIGHT_ARM_CARRY_CONFIG, dtype=np.float32)
    q_t = planner.tensor_args.to_device(q0.reshape(1, -1))
    kin_state = planner.motion_gen.kinematics.get_state(q_t)
    start_pos = kin_state.ee_position.detach().cpu().numpy().reshape(-1, 3)[0]
    start_quat = kin_state.ee_quaternion.detach().cpu().numpy().reshape(-1, 4)[0]

    target_pose = np.eye(4, dtype=np.float32)
    target_pose[:3, 3] = start_pos + np.asarray(args.target_delta, dtype=np.float32)
    # Use the current TCP orientation. For this task, position is the key check.
    qw, qx, qy, qz = start_quat.astype(np.float64)
    target_pose[:3, :3] = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )

    result = planner.plan_to_pose(q0, target_pose, max_attempts=args.max_attempts, enable_graph=False, timeout_s=8.0)
    summary = {
        "success": bool(result.success),
        "reason": result.reason,
        "joint_names": list(RIGHT_ARM_JOINT_NAMES),
        "start_q": q0.astype(float).tolist(),
        "start_tcp_position_base_m": start_pos.astype(float).tolist(),
        "target_tcp_position_base_m": target_pose[:3, 3].astype(float).tolist(),
        "target_delta_m": list(args.target_delta),
        "plan_steps": int(result.joint_positions.shape[0]),
        "metadata": result.metadata,
    }
    if result.success and result.joint_positions.size:
        summary["first_q"] = result.joint_positions[0].astype(float).tolist()
        summary["last_q"] = result.joint_positions[-1].astype(float).tolist()
        summary["max_abs_joint_delta_rad"] = float(np.max(np.abs(result.joint_positions - q0.reshape(1, -1))))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
