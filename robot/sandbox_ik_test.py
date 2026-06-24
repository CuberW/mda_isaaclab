#!/usr/bin/env python3
"""Minimal ctypes probe for the standalone Kuavo IK bridge.

Run this inside the Kuavo container after building:
  source /opt/ros/noetic/setup.bash
  /root/standalone_ik/build/libstandalone_ik.so
"""

from __future__ import annotations

import ctypes
import math
import pathlib
import re
import sys


LIB_PATH = pathlib.Path("/root/standalone_ik/build/libstandalone_ik.so")
URDF_PATH = pathlib.Path("/root/kuavo_ws_linux/src/kuavo_assets/models/biped_s62/urdf/biped_s62.urdf")
TASK_INFO_PATH = pathlib.Path(
    "/root/kuavo_ws_linux/src/humanoid-wheel-control/humanoid_wheel_interface/config/kuavo_s62/task.info"
)

TARGET_POSE = [0.35, -0.25, 0.40, 0.0, 0.0, 0.0]  # [x, y, z, yaw, pitch, roll]
ARM_INDEX = 1  # 0 = left, 1 = right
IS_WHOLE_BODY = 1  # 0 = single-arm, 1 = whole-body


def load_task_info(task_info_path: pathlib.Path) -> tuple[int, int]:
    text = task_info_path.read_text(encoding="utf-8", errors="ignore")
    arm_dim_match = re.search(r"\bmodelDof\s+(\d+)", text)
    model_type_match = re.search(r"\bmanipulatorModelType\s+(\d+)", text)
    if not arm_dim_match or not model_type_match:
        raise RuntimeError("failed to parse modelDof from task.info")
    return int(arm_dim_match.group(1)), int(model_type_match.group(1))


def infer_state_dim(arm_dim: int, manipulator_model_type: int) -> int:
    if manipulator_model_type == 4:
        return arm_dim + 3
    if manipulator_model_type == 1:
        return arm_dim + 3
    if manipulator_model_type == 3:
        return arm_dim + 6
    if manipulator_model_type == 2:
        return arm_dim + 6
    if manipulator_model_type == 0:
        return arm_dim
    raise RuntimeError(f"unsupported manipulatorModelType={manipulator_model_type}")


def build_initial_q(state_dim: int, arm_dim: int) -> list[float]:
    initial_q = [0.0] * state_dim

    # For kuavo_s62 task.info this is a wheel-world model: [base_x, base_y, base_yaw] + 18 joints.
    base_dim = state_dim - arm_dim
    if base_dim != 3:
        raise RuntimeError(f"unexpected base_dim={base_dim}, expected wheel-world base_dim=3")

    leg_biases = [0.08, -0.12, 0.08, -0.12]
    left_arm_biases = [0.05, -0.08, 0.10, 0.06, -0.06, 0.08, 0.04]
    right_arm_biases = [-0.05, 0.08, -0.10, -0.06, 0.06, -0.08, -0.04]
    arm_biases = leg_biases + left_arm_biases + right_arm_biases
    if len(arm_biases) != arm_dim:
        raise RuntimeError(f"bias vector length {len(arm_biases)} does not match arm_dim={arm_dim}")

    initial_q[base_dim:] = arm_biases
    return initial_q


def main() -> int:
    for path in (LIB_PATH, URDF_PATH, TASK_INFO_PATH):
        if not path.exists():
            print(f"[sandbox_ik_test] missing required path: {path}", file=sys.stderr)
            return 2

    arm_dim, manipulator_model_type = load_task_info(TASK_INFO_PATH)
    state_dim = infer_state_dim(arm_dim, manipulator_model_type)
    initial_q_values = build_initial_q(state_dim, arm_dim)
    lib = ctypes.CDLL(str(LIB_PATH))

    lib.ik_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.ik_create.restype = ctypes.c_void_p

    lib.ik_solve.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.ik_solve.restype = ctypes.c_int

    lib.ik_destroy.argtypes = [ctypes.c_void_p]
    lib.ik_destroy.restype = None

    handle = lib.ik_create(
        str(URDF_PATH).encode("utf-8"),
        str(TASK_INFO_PATH).encode("utf-8"),
        ARM_INDEX,
        IS_WHOLE_BODY,
    )
    if not handle:
        print("[sandbox_ik_test] ik_create failed", file=sys.stderr)
        return 3

    target_pose = (ctypes.c_double * 6)(*TARGET_POSE)
    initial_q = (ctypes.c_double * state_dim)(*initial_q_values)
    out_q = (ctypes.c_double * arm_dim)()
    best_linear_error = ctypes.c_double(math.inf)
    best_angular_error = ctypes.c_double(math.inf)

    try:
        status = lib.ik_solve(
            handle,
            target_pose,
            initial_q,
            state_dim,
            out_q,
            arm_dim,
            ctypes.byref(best_linear_error),
            ctypes.byref(best_angular_error),
        )
    finally:
        lib.ik_destroy(handle)

    print("[sandbox_ik_test] request:")
    print(f"  lib={LIB_PATH}")
    print(f"  urdf={URDF_PATH}")
    print(f"  task_info={TASK_INFO_PATH}")
    print(f"  arm_index={ARM_INDEX}")
    print(f"  is_whole_body={IS_WHOLE_BODY}")
    print(f"  target_pose={TARGET_POSE}")
    print(f"  manipulator_model_type={manipulator_model_type}")
    print(f"  state_dim={state_dim}")
    print(f"  arm_dim={arm_dim}")
    print(f"  initial_q={initial_q_values}")

    print("[sandbox_ik_test] response:")
    print(f"  status={status}")
    print(f"  bestLinearError={best_linear_error.value}")
    print(f"  bestAngularError={best_angular_error.value}")
    print(f"  qBest={[out_q[i] for i in range(arm_dim)]}")

    return 0 if status == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
