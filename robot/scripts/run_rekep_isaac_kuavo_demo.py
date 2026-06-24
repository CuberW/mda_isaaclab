"""在 Isaac Sim 5.1 中运行 Kuavo + ReKep 两阶段抓取 demo。

运行方式（在 Isaac Sim Python 环境中）：

    python scripts/run_rekep_isaac_kuavo_demo.py --config configs/rekep_isaac_kuavo.yaml

这个脚本尽量自动完成接入：
- 自动读取 Kuavo v62 默认关节名。
- 自动从 articulation.dof_names 校正左右臂关节列表。
- 如果配置的 articulation prim path 不存在，尝试在 /World 下找名字含 kuavo/biped/s62 的 articulation。
- 自动把 demo 世界系关键点转成 ReKep Hover -> Grasp -> Lift 轨迹。

你仍需要在 Isaac 工作站中准备好：
- Kuavo USD/URDF 已加载到 stage。
- Lula 的 robot_description_path 可用。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from planning.rekep_isaac_adapter import (
    ArmSide,
    CubicSplineTrajectorySmoother,
    DummyIKSolver,
    IsaacArticulationAdapter,
    KuavoIsaacConfig,
    LulaKuavoIKSolver,
    ReKepIntegration,
    ReKepKeypointInput,
    TwoStageGraspConstraints,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _make_world() -> Any:
    """创建或获取 Isaac World。"""

    try:
        from isaacsim.core.api import World
    except Exception:
        try:
            from omni.isaac.core import World
        except Exception as exc:
            raise ImportError(
                "未检测到 Isaac Sim Python API。请在 Isaac Sim 5.1 的 Python/Script Editor "
                "里运行本脚本，而不是普通 conda 环境。"
            ) from exc
    return World.instance() if hasattr(World, "instance") and World.instance() is not None else World()


def _prim_exists(path: str) -> bool:
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        return bool(stage and stage.GetPrimAtPath(path).IsValid())
    except Exception:
        return False


def _auto_find_articulation_path(preferred: str) -> str:
    if preferred and _prim_exists(preferred):
        return preferred

    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("当前没有打开 USD stage")
        candidates: list[str] = []
        for prim in stage.Traverse():
            name = prim.GetName().lower()
            path = str(prim.GetPath())
            if any(token in name or token in path.lower() for token in ("kuavo", "biped", "s62")):
                candidates.append(path)
        if candidates:
            print(f"[ReKepIsaac] 自动发现 Kuavo 候选 articulation: {candidates[0]}")
            return candidates[0]
    except Exception as exc:
        print(f"[ReKepIsaac] 自动发现 articulation 失败: {exc}")

    raise RuntimeError(
        "找不到 Kuavo articulation prim。请在 configs/rekep_isaac_kuavo.yaml 中设置 "
        "kuavo.articulation_prim_path，例如 /World/Kuavo。"
    )


def _make_config(raw: dict[str, Any]) -> KuavoIsaacConfig:
    kuavo = raw.get("kuavo", {})
    cfg = KuavoIsaacConfig.default_s62(
        articulation_prim_path=kuavo.get("articulation_prim_path", "/World/Kuavo")
    )
    cfg.articulation_prim_path = _auto_find_articulation_path(cfg.articulation_prim_path)
    cfg.right_ee_frame_name = kuavo.get("right_ee_frame_name", cfg.right_ee_frame_name)
    cfg.left_ee_frame_name = kuavo.get("left_ee_frame_name", cfg.left_ee_frame_name)
    cfg.right_arm_joint_names = kuavo.get("right_arm_joint_names", cfg.right_arm_joint_names)
    cfg.left_arm_joint_names = kuavo.get("left_arm_joint_names", cfg.left_arm_joint_names)
    cfg.control_hz = float(kuavo.get("control_hz", cfg.control_hz))
    return cfg


def _make_keypoints(raw: dict[str, Any]) -> ReKepKeypointInput:
    demo = raw.get("demo_keypoints", {})
    target = np.asarray(demo.get("target_obj_keypoints", []), dtype=float)
    container_raw = demo.get("container_keypoints")
    container = np.asarray(container_raw, dtype=float) if container_raw is not None else None
    return ReKepKeypointInput(target_obj_keypoints=target, container_keypoints=container)


def _resolve_project_path(path_text: str) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ReKep two-stage grasp demo in Isaac Sim 5.1")
    parser.add_argument("--config", default="configs/rekep_isaac_kuavo.yaml")
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--dry-run", action="store_true", help="只构建计划和 IK，不执行 world.step 控制循环")
    parser.add_argument(
        "--mock-ik",
        action="store_true",
        help="skip Lula and reuse current joint positions; useful for first Isaac articulation smoke test",
    )
    args = parser.parse_args()

    raw = _load_yaml((PROJECT_ROOT / args.config).resolve())
    world = _make_world()
    config = _make_config(raw)

    isaac = IsaacArticulationAdapter(world, config)
    isaac.initialize()
    config = isaac.config
    print(f"[ReKepIsaac] articulation={config.articulation_prim_path}")
    print(f"[ReKepIsaac] right joints={config.right_arm_joint_names}")
    print(f"[ReKepIsaac] left joints={config.left_arm_joint_names}")

    ik_cfg = raw.get("ik", {})
    if args.mock_ik:
        print("[ReKepIsaac] --mock-ik enabled: testing trajectory dispatch without real IK.")
        ik_solver = DummyIKSolver(num_joints=len(config.arm_joint_names(ArmSide(args.side))))
    else:
        if ik_cfg.get("backend", "lula").lower() != "lula":
            raise ValueError("当前脚本自动模式只实现 Lula；Pinocchio 可按 KuavoIKSolverProtocol 另接。")
        robot_description_path = ik_cfg.get("robot_description_path", "")
        if not robot_description_path:
            raise RuntimeError(
                "缺少 ik.robot_description_path。临时联调可加 --mock-ik；正式 IK 请在 "
                "configs/rekep_isaac_kuavo.yaml 中填写 Isaac Lula robot descriptor yaml 路径。"
            )
        ik_solver = LulaKuavoIKSolver(
            config=config,
            robot_description_path=_resolve_project_path(robot_description_path),
            urdf_path=_resolve_project_path(ik_cfg.get("urdf_path", "")),
        )
        ik_solver.initialize(isaac.robot)

    c_cfg = raw.get("constraints", {})
    constraints = TwoStageGraspConstraints(
        hover_height_m=float(c_cfg.get("hover_height_m", 0.15)),
        grasp_depth_m=float(c_cfg.get("grasp_depth_m", 0.0)),
        xy_tolerance_m=float(c_cfg.get("xy_tolerance_m", 0.01)),
    )
    t_cfg = raw.get("trajectory", {})
    smoother = CubicSplineTrajectorySmoother(min_points=int(t_cfg.get("min_points", 120)))

    integration = ReKepIntegration(
        isaac_adapter=isaac,
        ik_solver=ik_solver,
        kuavo_config=config,
        constraints=constraints,
        smoother=smoother,
    )
    keypoints = _make_keypoints(raw)
    side = ArmSide(args.side)
    plan = integration.build_two_stage_plan(keypoints, side=side)
    print(f"[ReKepIsaac] target_center={keypoints.target_center().round(4).tolist()}")
    for phase, pose in zip(plan.phases, plan.ee_waypoints):
        print(f"[ReKepIsaac] {phase.value}: pos={pose.position.round(4).tolist()} quat_xyzw={pose.quat_xyzw.round(4).tolist()}")

    q_sparse = integration.solve_joint_waypoints(plan)
    print(f"[ReKepIsaac] sparse joint waypoints shape={q_sparse.shape}")
    if args.dry_run:
        return 0

    def on_phase(phase):
        # TODO: 在这里接 Kuavo 真实夹爪控制。
        # HOVER: open gripper
        # GRASP: close gripper after vertical descent
        # LIFT: verify contact
        print(f"[ReKepIsaac] phase boundary: {phase.value}")

    integration.execute_plan(
        plan,
        total_time_s=float(t_cfg.get("total_time_s", 2.5)),
        on_phase_boundary=on_phase,
    )
    print("[ReKepIsaac] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
