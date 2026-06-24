"""本地 MuJoCo 里查看 ReKep 接入 3.19 后的规划效果。

这个脚本不依赖 Isaac Sim，也不需要 Lula。它复用当前 3.19 Kuavo/Stretch 场景，
把目标垃圾物体的世界坐标包装成 ReKepKeypointInput，然后生成：

- Hover: 目标正上方 15cm，末端 Z 轴向下
- Grasp: 沿局部 Z 轴竖直下降到目标
- Lift: 抓取后竖直抬升

输出：
- results/rekep_319_demo/rekep_plan.json
- results/rekep_319_demo/rekep_overlay.png

可选：
- 加 --execute 可以让当前 3.19 Kuavo 控制器执行 hover/grasp/lift 手臂轨迹。
  该模式只演示 ReKep 航点控制，不闭爪、不 attach、不移动物体。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control import MuJoCoEnv
from planning.rekep_isaac_adapter import (
    ArmSide,
    ReKepIntegration,
    ReKepKeypointInput,
    TwoStageGraspConstraints,
)
from robot_common.infra.logging import logger
from task_319_garbage_sort.kuavo_controller import KuavoWheelGarbageController

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except Exception:
    HAS_PIL = False


class LocalMuJoCoIKAdapter:
    """把 ReKep 的 IK 协议桥接到当前 KuavoWheelGarbageController。"""

    def __init__(self, controller: KuavoWheelGarbageController):
        self.controller = controller

    def solve(self, side: ArmSide, target_pose, seed_joint_positions=None) -> np.ndarray:
        planned = self.controller.plan_arm_to(side.value, target_pose.position)
        if planned.positions.size == 0:
            raise RuntimeError(f"Local IK failed for {side.value}: {target_pose.position.tolist()}")
        return planned.positions[-1]


class LocalMuJoCoReKepAdapter:
    """ReKepIntegration 需要的最小机器人接口。"""

    def __init__(self, controller: KuavoWheelGarbageController):
        self.controller = controller
        self.world = None

    def get_joint_positions(self, joint_names):
        return np.array([self.controller._qpos_for_joint(name) for name in joint_names], dtype=float)

    def apply_joint_positions(self, joint_names, joint_positions):
        for joint, value in zip(joint_names, joint_positions):
            self.controller._set_actuator_for_joint(joint, float(value))


def _make_target_keypoints(center: np.ndarray, radius: float = 0.03) -> np.ndarray:
    cx, cy, cz = np.asarray(center, dtype=float).reshape(3)
    return np.array(
        [
            [cx - radius, cy - radius, cz],
            [cx + radius, cy - radius, cz],
            [cx - radius, cy + radius, cz],
            [cx + radius, cy + radius, cz],
            [cx, cy, cz],
        ],
        dtype=float,
    )


def _draw_overlay(env: MuJoCoEnv, image: np.ndarray, camera: str, plan, out_path: Path) -> None:
    if not HAS_PIL:
        return
    img = Image.fromarray(np.asarray(image, dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    colors = {
        "hover": (255, 180, 0),
        "grasp": (255, 40, 40),
        "lift": (40, 220, 80),
    }
    prev_pixel = None
    for phase, pose in zip(plan.phases, plan.ee_waypoints):
        proj = env.project_world_point(pose.position, camera)
        if not proj["visible"]:
            continue
        u, v = [float(x) for x in proj["pixel"]]
        color = colors.get(phase.value, (255, 255, 255))
        r = 7
        draw.ellipse([u - r, v - r, u + r, v + r], outline=color, width=3)
        draw.text((u + 9, v - 8), phase.value, fill=color)
        if prev_pixel is not None:
            draw.line([prev_pixel, (u, v)], fill=color, width=2)
        prev_pixel = (u, v)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local MuJoCo ReKep demo for Task 3.19")
    parser.add_argument("--scene", default="simulation/scenes/task_319_kuavo_wheel_s62.xml")
    parser.add_argument("--camera", default="camera_rgb")
    parser.add_argument("--target", default="trash_01")
    parser.add_argument("--execute", action="store_true", help="execute hover/grasp/lift arm waypoints without grasp attach")
    parser.add_argument("--out", default="results/rekep_319_demo")
    args = parser.parse_args()

    env = MuJoCoEnv(str(PROJECT_ROOT / args.scene), camera_name=args.camera, width=640, height=480)
    controller = KuavoWheelGarbageController(env)
    for _ in range(60):
        controller.hold_base_stationary()
        controller._hold_posture()
        env.step()

    if args.target not in env._body_names:
        raise ValueError(f"Target body not found: {args.target}")

    target_center = env.get_body_position(args.target).copy()
    keypoints = ReKepKeypointInput(target_obj_keypoints=_make_target_keypoints(target_center))

    # 这里只使用 ReKepIntegration 的两阶段约束规划能力；IK/执行用本地 MuJoCo adapter。
    local_robot = LocalMuJoCoReKepAdapter(controller)
    local_ik = LocalMuJoCoIKAdapter(controller)

    class _Config:
        right_arm_joint_names = controller.RIGHT_ARM_JOINTS
        left_arm_joint_names = controller.LEFT_ARM_JOINTS

        def arm_joint_names(self, side):
            return self.left_arm_joint_names if side == ArmSide.LEFT else self.right_arm_joint_names

    integration = ReKepIntegration(
        isaac_adapter=local_robot,
        ik_solver=local_ik,
        kuavo_config=_Config(),
        constraints=TwoStageGraspConstraints(hover_height_m=0.15, grasp_depth_m=0.0),
    )
    plan = integration.build_two_stage_plan(keypoints, side=ArmSide.RIGHT)
    q_sparse = integration.solve_joint_waypoints(plan)
    dense_t, dense_q = integration.smoother.smooth_joint_waypoints(q_sparse, total_time_s=2.5)

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = env.render(args.camera)
    _draw_overlay(env, rgb, args.camera, plan, out_dir / "rekep_overlay.png")

    payload = {
        "target_body": args.target,
        "target_center_world": target_center.tolist(),
        "phases": [phase.value for phase in plan.phases],
        "ee_waypoints": [
            {
                "phase": phase.value,
                "position_world": pose.position.tolist(),
                "quat_xyzw": pose.quat_xyzw.tolist(),
                "projection": {
                    "pixel": env.project_world_point(pose.position, args.camera)["pixel"].tolist(),
                    "visible": bool(env.project_world_point(pose.position, args.camera)["visible"]),
                },
            }
            for phase, pose in zip(plan.phases, plan.ee_waypoints)
        ],
        "joint_waypoints_shape": list(q_sparse.shape),
        "dense_trajectory_shape": list(dense_q.shape),
        "dense_time_s": dense_t.tolist(),
    }
    (out_dir / "rekep_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.execute:
        joint_names = controller.RIGHT_ARM_JOINTS
        for q in dense_q:
            controller.hold_base_pose(controller.get_base_pose())
            for joint, value in zip(joint_names, q):
                controller._set_actuator_for_joint(joint, float(value))
            controller._hold_posture(exclude=set(joint_names))
            env.step()
        logger.info("Executed local ReKep hover/grasp/lift demo trajectory.")

    print(f"ReKep local demo written to: {out_dir}")
    print(f"- overlay: {out_dir / 'rekep_overlay.png'}")
    print(f"- plan:    {out_dir / 'rekep_plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
