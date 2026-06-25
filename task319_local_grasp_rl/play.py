"""Replay a trained Task319 hover-descent physical gripper policy."""

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
    WORKSPACE_ROOT / "IsaacLab/source/isaaclab_rl",
):
    if source_dir.exists() and str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from isaaclab.app import AppLauncher

from task319_local_grasp_rl import TASK_ID  # noqa: E402


parser = argparse.ArgumentParser(description="Play a Task319 hover-descent physical gripper skrl policy.")
parser.add_argument("--task", type=str, default=TASK_ID)
parser.add_argument("--agent", type=str, default=None, help="skrl agent config entry point.")
parser.add_argument("--algorithm", type=str, default="PPO", help="skrl algorithm name.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--stochastic", action="store_true", help="Sample stochastic actions instead of using policy mean actions.")
parser.add_argument("--print_metrics_interval", type=int, default=0, help="Print env log metrics every N steps.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if args_cli.agent is None:
    agent_cfg_entry_point = "skrl_cfg_entry_point" if args_cli.algorithm.lower() == "ppo" else f"skrl_{args_cli.algorithm.lower()}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner  # noqa: E402
else:
    from skrl.utils.runner.jax import Runner  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import task319_local_grasp_rl  # noqa: E402,F401


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.pop("wandb", None)
    agent_cfg["trainer"]["timesteps"] = args_cli.steps
    agent_cfg["agent"]["experiment"]["wandb"] = False
    # Evaluation must never enter SAC's random warmup branch. Passing timestep=0
    # every frame makes SAC act randomly while timestep < random_timesteps.
    agent_cfg["agent"]["random_timesteps"] = 0
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(raw_env, ml_framework=args_cli.ml_framework)
    runner = Runner(env, agent_cfg)
    runner.agent.load(args_cli.checkpoint)
    runner.agent.enable_training_mode(False, apply_to_models=True)
    obs, _ = env.reset()
    for timestep in range(args_cli.steps):
        actions, outputs = runner.agent.act(obs, None, timestep=timestep, timesteps=args_cli.steps)
        if not args_cli.stochastic and isinstance(outputs, dict) and "mean_actions" in outputs:
            actions = outputs["mean_actions"]
        obs, _, terminated, truncated, _ = env.step(actions)
        if args_cli.print_metrics_interval > 0 and (timestep + 1) % args_cli.print_metrics_interval == 0:
            logs = getattr(raw_env.unwrapped, "extras", {}).get("log", {})
            if logs:
                items = []
                for key in (
                    "Metrics/success_rate",
                    "Metrics/min_tcp_target_distance_m",
                    "Metrics/stage_reach_rate",
                    "Metrics/stage_descend_rate",
                    "Metrics/stage_close_rate",
                    "Metrics/grasp_latch_rate",
                    "Metrics/stage_lift_rate",
                    "Metrics/max_object_lift_m",
                    "Metrics/teacher_action_fraction",
                ):
                    value = logs.get(key)
                    if value is not None:
                        if hasattr(value, "detach"):
                            value = value.detach().cpu().item()
                        items.append(f"{key}={float(value):.4g}")
                if items:
                    print(f"[play step {timestep + 1}] " + " ".join(items), flush=True)
        if bool((terminated | truncated).all()):
            obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
