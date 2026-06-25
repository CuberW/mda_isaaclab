"""Train Task319 hover-descent physical gripper policy with skrl."""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime
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


parser = argparse.ArgumentParser(description="Train Task319 hover-descent physical gripper RL policy with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between training videos.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default=TASK_ID, help="Gym task id.")
parser.add_argument("--agent", type=str, default=None, help="skrl agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment and agent seed.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume.")
parser.add_argument(
    "--max_iterations",
    type=int,
    default=None,
    help="Training budget. On-policy algorithms use iterations * rollouts; off-policy algorithms use env steps.",
)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"], help="skrl ML framework.")
parser.add_argument("--algorithm", type=str, default="PPO", help="skrl algorithm name.")
parser.add_argument("--task319_quick_test", action="store_true", help="Use tiny settings for a fast smoke test.")
parser.add_argument(
    "--teacher_mixing",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable scripted teacher action mixing. Defaults to enabled for SAC and disabled otherwise.",
)
parser.add_argument("--teacher_only", action="store_true", help="Run with scripted teacher actions only. Useful for teacher sanity checks.")
parser.add_argument("--teacher_warmup_steps", type=int, default=10000, help="Env steps with 100% scripted teacher actions.")
parser.add_argument("--teacher_decay_steps", type=int, default=80000, help="Env steps used to decay teacher action probability.")
parser.add_argument("--teacher_mix_start", type=float, default=0.80, help="Teacher probability after warmup.")
parser.add_argument("--teacher_mix_end", type=float, default=0.0, help="Final teacher probability after decay.")
parser.add_argument("--wandb", action="store_true", help="Upload training metrics to Weights & Biases.")
parser.add_argument("--wandb_project", type=str, default=None, help="W&B project. Defaults to WANDB_PROJECT or YAML.")
parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team. Defaults to WANDB_ENTITY or YAML.")
parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name. Defaults to skrl experiment name.")
parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated W&B tags.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.agent is None:
    cli_algorithm = args_cli.algorithm.lower()
else:
    cli_algorithm = args_cli.agent.split("_cfg")[0].split("skrl_")[-1].lower()

if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.task319_quick_test:
    args_cli.num_envs = args_cli.num_envs or 8
    args_cli.max_iterations = args_cli.max_iterations or (2 if cli_algorithm == "ppo" else 96)
    args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import skrl  # noqa: E402
from packaging import version  # noqa: E402

if version.parse(skrl.__version__) < version.parse("2.0.0"):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}. Install skrl>=2.0.0.")
    raise SystemExit(1)

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner  # noqa: E402
else:
    from skrl.utils.runner.jax import Runner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import task319_local_grasp_rl  # noqa: E402,F401

logger = logging.getLogger(__name__)

if args_cli.agent is None:
    algorithm = cli_algorithm
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = cli_algorithm


def _resolve_wandb_cfg(agent_cfg: dict, experiment_name: str) -> None:
    yaml_wandb_cfg = agent_cfg.pop("wandb", {}) or {}
    experiment_cfg = agent_cfg["agent"]["experiment"]
    if not args_cli.wandb:
        experiment_cfg["wandb"] = False
        experiment_cfg["wandb_kwargs"] = {}
        return

    project = args_cli.wandb_project or os.environ.get("WANDB_PROJECT") or yaml_wandb_cfg.get("project")
    entity = args_cli.wandb_entity or os.environ.get("WANDB_ENTITY") or yaml_wandb_cfg.get("entity")
    run_name = args_cli.wandb_name or yaml_wandb_cfg.get("name") or experiment_name
    tags_raw = args_cli.wandb_tags or yaml_wandb_cfg.get("tags") or []
    tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()] if isinstance(tags_raw, str) else list(tags_raw)

    if not project:
        raise RuntimeError("W&B enabled but project is empty. Set WANDB_PROJECT or pass --wandb_project.")
    if "WANDB_API_KEY" not in os.environ and not (Path.home() / ".netrc").exists():
        print("[WARN] W&B is enabled but WANDB_API_KEY is not set and ~/.netrc was not found.")
        print("[WARN] Run `wandb login` or export WANDB_API_KEY before long training.")

    experiment_cfg["wandb"] = True
    experiment_cfg["wandb_kwargs"] = {
        "project": project,
        "entity": entity or None,
        "name": run_name,
        "tags": tags,
        "sync_tensorboard": True,
    }


def _training_timesteps_from_budget(agent_cfg: dict, budget: int) -> int:
    steps_per_iteration = int(agent_cfg["agent"].get("rollouts", 1))
    return int(budget) * steps_per_iteration


def _apply_teacher_cfg(env_cfg: DirectRLEnvCfg) -> None:
    teacher_enabled = (algorithm == "sac") if args_cli.teacher_mixing is None else bool(args_cli.teacher_mixing)
    env_cfg.teacher_enabled = bool(teacher_enabled or args_cli.teacher_only)
    env_cfg.teacher_only = bool(args_cli.teacher_only)
    env_cfg.teacher_warmup_steps = max(0, int(args_cli.teacher_warmup_steps))
    env_cfg.teacher_decay_steps = max(1, int(args_cli.teacher_decay_steps))
    env_cfg.teacher_mix_start = float(args_cli.teacher_mix_start)
    env_cfg.teacher_mix_end = float(args_cli.teacher_mix_end)
    if args_cli.teacher_only:
        env_cfg.teacher_mix_start = 1.0
        env_cfg.teacher_mix_end = 1.0


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _apply_teacher_cfg(env_cfg)
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = _training_timesteps_from_budget(agent_cfg, args_cli.max_iterations)

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line: {log_dir_name}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir_name += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir_name
    _resolve_wandb_cfg(agent_cfg, log_dir_name)
    log_dir = os.path.join(log_root_path, log_dir_name)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    runner = Runner(env, agent_cfg)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
    runner.run()
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
