#!/usr/bin/env python3
"""Validate strict physical graspability in the Task319 local RL environment.

This is a pre-training gate. It runs a deterministic local top-down grasp in
the same env used by RL, with teacher mixing and hidden latch disabled. Passing
means the simulated object can be lifted by contact physics, not by attachment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
for source_dir in (
    WORKSPACE_ROOT / "IsaacLab/source/isaaclab",
    WORKSPACE_ROOT / "IsaacLab/source/isaaclab_assets",
    WORKSPACE_ROOT / "IsaacLab/source/isaaclab_tasks",
    WORKSPACE_ROOT / "IsaacLab/source/isaaclab_rl",
):
    if source_dir.exists() and str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from isaaclab.app import AppLauncher

from task319_local_grasp_rl import TASK_ID  # noqa: E402


parser = argparse.ArgumentParser(description="Strict physical graspability check for Task319 local RL env.")
parser.add_argument("--task", type=str, default=TASK_ID)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--settle_steps", type=int, default=40)
parser.add_argument("--hover_steps", type=int, default=120)
parser.add_argument("--descend_steps", type=int, default=120)
parser.add_argument("--close_steps", type=int, default=120)
parser.add_argument("--lift_steps", type=int, default=160)
parser.add_argument("--hover_height_m", type=float, default=0.105)
parser.add_argument("--lift_height_m", type=float, default=0.120)
parser.add_argument("--target_x_bias_m", type=float, default=0.0)
parser.add_argument("--target_y_bias_m", type=float, default=0.0)
parser.add_argument("--target_z_bias_m", type=float, default=0.0)
parser.add_argument("--object_spawn_xy_range_m", type=float, default=0.0)
parser.add_argument("--target_noise_scale", type=float, default=0.0)
parser.add_argument("--disable_topdown_orientation", action="store_true")
parser.add_argument("--lift_along_tcp_minus_x", action="store_true")
parser.add_argument("--track_live_target", action="store_true", help="Track live object target during scripted check. Default locks the initial target.")
parser.add_argument("--success_rate_threshold", type=float, default=0.80)
parser.add_argument("--output_json", type=str, default="rl_training_analysis/strict_physical_grasp_check.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import task319_local_grasp_rl  # noqa: E402,F401
from task319_local_grasp_rl.local_suction_grasp_env import Task319LocalSuctionGraspEnvCfg  # noqa: E402


def _to_float(value) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(value)


def _action_to_desired(env, desired_tcp_w: torch.Tensor, close_action: torch.Tensor | float) -> torch.Tensor:
    unwrapped = env.unwrapped
    tcp_pos = unwrapped._tcp_pos_w()
    actions = torch.zeros((unwrapped.num_envs, 4), device=unwrapped.device)
    actions[:, 0:3] = torch.clamp(
        (desired_tcp_w - tcp_pos) / max(float(unwrapped.cfg.max_delta_pos_m), 1.0e-6),
        -1.0,
        1.0,
    )
    if isinstance(close_action, torch.Tensor):
        actions[:, 3] = close_action.to(device=unwrapped.device)
    else:
        actions[:, 3] = float(close_action)
    return actions


def _step_toward(env, desired_tcp_w: torch.Tensor, close_action: torch.Tensor | float):
    action = _action_to_desired(env, desired_tcp_w, close_action)
    return env.step(action)


def _metrics(env) -> dict[str, float]:
    unwrapped = env.unwrapped
    tcp_pos = unwrapped._tcp_pos_w()
    tcp_rot = unwrapped._gripper_base_rot_w()
    object_pos = unwrapped._object.data.root_pos_w
    object_lift = object_pos[:, 2] - unwrapped._object_initial_z
    success, retained = unwrapped._grasp_success_mask(tcp_pos, object_lift, object_pos)
    tcp_object_dist = torch.linalg.norm(tcp_pos - object_pos, dim=1)
    finger_width = unwrapped._finger_width_m()
    pushed = torch.linalg.norm(object_pos[:, 0:2] - unwrapped._object_initial_xy, dim=1)
    object_rel_local = torch.bmm(torch.transpose(tcp_rot, 1, 2), (object_pos - tcp_pos).unsqueeze(-1)).squeeze(-1)
    target_rel_local = torch.bmm(
        torch.transpose(tcp_rot, 1, 2),
        (unwrapped._object_grasp_target_w() - tcp_pos).unsqueeze(-1),
    ).squeeze(-1)
    tcp_x_world = tcp_rot[:, :, 0]
    tcp_y_world = tcp_rot[:, :, 1]
    tcp_z_world = tcp_rot[:, :, 2]
    return {
        "success_rate": _to_float(torch.mean(success.float())),
        "retained_rate": _to_float(torch.mean(retained.float())),
        "max_object_lift_m": _to_float(torch.max(object_lift)),
        "mean_object_lift_m": _to_float(torch.mean(object_lift)),
        "mean_tcp_z_m": _to_float(torch.mean(tcp_pos[:, 2])),
        "mean_object_z_m": _to_float(torch.mean(object_pos[:, 2])),
        "mean_tcp_object_dist_m": _to_float(torch.mean(tcp_object_dist)),
        "mean_finger_width_m": _to_float(torch.mean(finger_width)),
        "mean_object_xy_displacement_m": _to_float(torch.mean(pushed)),
        "mean_object_rel_tcp_local_x_m": _to_float(torch.mean(object_rel_local[:, 0])),
        "mean_object_rel_tcp_local_y_m": _to_float(torch.mean(object_rel_local[:, 1])),
        "mean_object_rel_tcp_local_z_m": _to_float(torch.mean(object_rel_local[:, 2])),
        "mean_target_rel_tcp_local_x_m": _to_float(torch.mean(target_rel_local[:, 0])),
        "mean_target_rel_tcp_local_y_m": _to_float(torch.mean(target_rel_local[:, 1])),
        "mean_target_rel_tcp_local_z_m": _to_float(torch.mean(target_rel_local[:, 2])),
        "mean_tcp_local_x_world_z": _to_float(torch.mean(tcp_x_world[:, 2])),
        "mean_tcp_local_y_world_x": _to_float(torch.mean(tcp_y_world[:, 0])),
        "mean_tcp_local_y_world_y": _to_float(torch.mean(tcp_y_world[:, 1])),
        "mean_tcp_local_y_world_z": _to_float(torch.mean(tcp_y_world[:, 2])),
        "mean_tcp_local_z_world_z": _to_float(torch.mean(tcp_z_world[:, 2])),
        "latched_rate": _to_float(torch.mean(unwrapped._grasp_latched.float())),
    }


def main() -> None:
    env_cfg = Task319LocalSuctionGraspEnvCfg()
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.debug_vis = not bool(args_cli.headless)
    env_cfg.episode_length_s = 20.0
    env_cfg.object_spawn_xy_range_m = float(args_cli.object_spawn_xy_range_m)
    env_cfg.target_estimate_xy_noise_std_m *= float(args_cli.target_noise_scale)
    env_cfg.target_estimate_z_noise_std_m *= float(args_cli.target_noise_scale)
    env_cfg.teacher_enabled = False
    env_cfg.teacher_only = False
    env_cfg.grasp_latch_enabled = False
    env_cfg.strict_physical_success = True
    env_cfg.enforce_topdown_grasp_orientation = not bool(args_cli.disable_topdown_orientation)

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    del obs
    unwrapped = env.unwrapped
    device = unwrapped.device
    locked_target = unwrapped._estimated_grasp_target_w().detach().clone()

    def target_w() -> torch.Tensor:
        if bool(args_cli.track_live_target):
            target = unwrapped._estimated_grasp_target_w()
        else:
            target = locked_target.clone()
        target[:, 0] += float(args_cli.target_x_bias_m)
        target[:, 1] += float(args_cli.target_y_bias_m)
        return target

    total_steps = 0
    phase_metrics: dict[str, dict[str, float]] = {}

    for _ in range(int(args_cli.settle_steps)):
        target = target_w()
        desired = target.clone()
        desired[:, 2] = target[:, 2] + float(args_cli.hover_height_m)
        _step_toward(env, desired, -1.0)
        total_steps += 1
    phase_metrics["settle"] = _metrics(env)

    for _ in range(int(args_cli.hover_steps)):
        target = target_w()
        desired = target.clone()
        desired[:, 2] = target[:, 2] + float(args_cli.hover_height_m)
        _step_toward(env, desired, -1.0)
        total_steps += 1
    phase_metrics["hover"] = _metrics(env)

    for _ in range(int(args_cli.descend_steps)):
        target = target_w()
        desired = target.clone()
        desired[:, 2] = target[:, 2] + float(args_cli.target_z_bias_m)
        _step_toward(env, desired, -1.0)
        total_steps += 1
    phase_metrics["descend"] = _metrics(env)

    for step in range(int(args_cli.close_steps)):
        target = target_w()
        desired = target.clone()
        desired[:, 2] = target[:, 2] + float(args_cli.target_z_bias_m)
        alpha = min(1.0, max(0.0, float(step) / max(1.0, 0.35 * float(args_cli.close_steps))))
        close_action = torch.full((unwrapped.num_envs,), -1.0 + 2.0 * alpha, device=device)
        _step_toward(env, desired, close_action)
        total_steps += 1
    phase_metrics["close"] = _metrics(env)

    lift_axis_w = None
    if bool(args_cli.lift_along_tcp_minus_x):
        lift_axis_w = -unwrapped._gripper_base_rot_w()[:, :, 0].detach().clone()
        flip = lift_axis_w[:, 2] < 0.0
        if torch.any(flip):
            lift_axis_w[flip] *= -1.0
        lift_axis_w = lift_axis_w / torch.linalg.norm(lift_axis_w, dim=1, keepdim=True).clamp_min(1.0e-6)

    for _ in range(int(args_cli.lift_steps)):
        target = target_w()
        desired = target.clone()
        desired[:, 2] = target[:, 2] + float(args_cli.target_z_bias_m)
        if lift_axis_w is None:
            desired[:, 2] = target[:, 2] + float(args_cli.lift_height_m)
        else:
            desired = desired + lift_axis_w * float(args_cli.lift_height_m)
        _step_toward(env, desired, 1.0)
        total_steps += 1
    phase_metrics["lift"] = _metrics(env)

    final_metrics = _metrics(env)
    payload = {
        "task": args_cli.task,
        "num_envs": int(args_cli.num_envs),
        "total_steps": total_steps,
        "strict_physical_success": True,
        "teacher_enabled": False,
        "grasp_latch_enabled": False,
        "object_spawn_xy_range_m": float(args_cli.object_spawn_xy_range_m),
        "target_noise_scale": float(args_cli.target_noise_scale),
        "target_x_bias_m": float(args_cli.target_x_bias_m),
        "target_y_bias_m": float(args_cli.target_y_bias_m),
        "target_z_bias_m": float(args_cli.target_z_bias_m),
        "disable_topdown_orientation": bool(args_cli.disable_topdown_orientation),
        "lift_along_tcp_minus_x": bool(args_cli.lift_along_tcp_minus_x),
        "track_live_target": bool(args_cli.track_live_target),
        "success_rate_threshold": float(args_cli.success_rate_threshold),
        "passed": final_metrics["success_rate"] >= float(args_cli.success_rate_threshold),
        "phase_metrics": phase_metrics,
        "final_metrics": final_metrics,
    }

    output_path = Path(args_cli.output_json)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    env.close()
    if not payload["passed"]:
        print(
            f"Strict physical grasp check failed: success_rate={final_metrics['success_rate']:.3f} "
            f"< threshold={float(args_cli.success_rate_threshold):.3f}. See {output_path}",
            flush=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
