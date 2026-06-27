"""Collect Task319 hover-to-grasp demonstrations with human teleoperation.

This collector is intentionally scoped to the local grasp stage.  The robot
starts with the right gripper near a hover pose above a randomized table object.
The operator commands small TCP deltas and gripper open/close, while the script
records policy observations and actions into an HDF5 dataset.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


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

from isaaclab.app import AppLauncher  # noqa: E402

from task319_local_grasp_rl import TASK_ID  # noqa: E402


DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "task_319_garbage_sort/output/teleop_grasp_datasets"


parser = argparse.ArgumentParser(description="Collect Task319 local hover-to-grasp teleoperation demos.")
parser.add_argument("--task", type=str, default=TASK_ID, help="Gym task id.")
parser.add_argument("--dataset_file", type=str, default=None, help="Output HDF5 file. Defaults to timestamped output dir.")
parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output root if --dataset_file is omitted.")
parser.add_argument("--teleop_device", choices=("keyboard", "spacemouse", "mouse"), default="keyboard", help="Motion teleoperation device.")
parser.add_argument("--num_success_demos", type=int, default=0, help="Stop after N successful demos. 0 means unlimited.")
parser.add_argument("--max_episodes", type=int, default=0, help="Stop after N saved/discarded episodes. 0 means unlimited.")
parser.add_argument("--steps", type=int, default=0, help="Stop after N control steps. 0 means unlimited.")
parser.add_argument("--episode_length_s", type=float, default=30.0, help="Manual collection episode length.")
parser.add_argument("--pos_sensitivity", type=float, default=0.35, help="Keyboard/SpaceMouse normalized translation sensitivity.")
parser.add_argument("--rot_sensitivity", type=float, default=0.0, help="Rotation sensitivity. Keep 0 for first top-down dataset.")
parser.add_argument("--mouse_xy_sensitivity", type=float, default=0.012, help="Ordinary mouse drag sensitivity for XY action.")
parser.add_argument("--mouse_z_sensitivity", type=float, default=0.20, help="Ordinary mouse wheel sensitivity for Z action.")
parser.add_argument("--action_scale", type=float, default=1.0, help="Extra multiplier before clipping the 3D delta action.")
parser.add_argument("--keyboard_full_step", action=argparse.BooleanOptionalAction, default=True, help="Map pressed keyboard translation keys to full +/-1 actions.")
parser.add_argument("--tcp_step_m", type=float, default=0.020, help="Maximum TCP translation per control step in meters.")
parser.add_argument("--joint_step_rad", type=float, default=0.060, help="Maximum right-arm joint update per control step.")
parser.add_argument("--print_interval", type=int, default=30, help="Print status every N steps. 0 disables periodic prints.")
parser.add_argument("--min_steps_to_save", type=int, default=8, help="Skip manual saves shorter than this many samples.")
parser.add_argument("--save_failures", action=argparse.BooleanOptionalAction, default=True, help="Save failed demos too.")
parser.add_argument("--enable_grasp_latch", action="store_true", help="Enable the env's hidden grasp latch for pipeline debugging only.")
parser.add_argument("--object_spawn_xy_range_m", type=float, default=0.04, help="Object XY randomization half range.")
parser.add_argument("--target_noise_scale", type=float, default=1.0, help="Scale env target-estimate noise.")
parser.add_argument("--disable_debug_vis", action="store_true", help="Hide target/TCP debug markers.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import torch  # noqa: E402
import carb  # noqa: E402
import omni  # noqa: E402
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg  # noqa: E402

import task319_local_grasp_rl  # noqa: E402,F401
from task319_local_grasp_rl.local_suction_grasp_env import (  # noqa: E402
    GRIPPER_CLOSED_M,
    GRIPPER_OPEN_M,
    Task319LocalSuctionGraspEnvCfg,
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(WORKSPACE_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _default_dataset_file() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(args_cli.output_dir) / timestamp / "task319_hover_grasp_teleop.hdf5"


def _to_numpy(value: torch.Tensor | np.ndarray | float | int | bool) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().flatten()[0].cpu().item())
    arr = np.asarray(value)
    return bool(arr.reshape(-1)[0])


@dataclass
class DemoBuffer:
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[np.ndarray] = field(default_factory=list)
    terminated: list[np.ndarray] = field(default_factory=list)
    truncated: list[np.ndarray] = field(default_factory=list)
    command_raw: list[np.ndarray] = field(default_factory=list)
    obs_policy: list[np.ndarray] = field(default_factory=list)
    tcp_pose_w: list[np.ndarray] = field(default_factory=list)
    target_rel_tcp: list[np.ndarray] = field(default_factory=list)
    right_q: list[np.ndarray] = field(default_factory=list)
    right_qd: list[np.ndarray] = field(default_factory=list)
    gripper_width: list[np.ndarray] = field(default_factory=list)
    object_pose_w: list[np.ndarray] = field(default_factory=list)
    lift_m: list[np.ndarray] = field(default_factory=list)
    xy_error_m: list[np.ndarray] = field(default_factory=list)
    grasp_latched: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def clear(self) -> None:
        for value in self.__dict__.values():
            value.clear()



class Se3OrdinaryMouse:
    """Small ordinary-mouse device for Task319 local teleoperation.

    Hold left mouse button and drag for XY.  Use wheel for Z.  Click middle
    button, or press K on the control keyboard, to toggle the gripper.
    """

    def __init__(self, *, xy_sensitivity: float, z_sensitivity: float):
        self.xy_sensitivity = float(xy_sensitivity)
        self.z_sensitivity = float(z_sensitivity)
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._mouse = self._appwindow.get_mouse()
        self._last_x = None
        self._last_y = None
        self._close_gripper = False
        self._callbacks = {}

    def reset(self):
        self._last_x = None
        self._last_y = None
        self._close_gripper = False

    def add_callback(self, key: str, func):
        self._callbacks[key] = func

    def toggle_gripper(self):
        self._close_gripper = not self._close_gripper

    def advance(self) -> torch.Tensor:
        x = float(self._input.get_mouse_value(self._mouse, carb.input.MouseInput.X))
        y = float(self._input.get_mouse_value(self._mouse, carb.input.MouseInput.Y))
        wheel = float(self._input.get_mouse_value(self._mouse, carb.input.MouseInput.WHEEL))
        left = float(self._input.get_mouse_value(self._mouse, carb.input.MouseInput.LEFT_BUTTON)) > 0.5
        middle = float(self._input.get_mouse_value(self._mouse, carb.input.MouseInput.MIDDLE_BUTTON)) > 0.5

        dx = 0.0
        dy = 0.0
        if self._last_x is not None and self._last_y is not None and left:
            dx = (y - self._last_y) * self.xy_sensitivity
            dy = -(x - self._last_x) * self.xy_sensitivity
        self._last_x = x
        self._last_y = y

        if middle:
            self.toggle_gripper()

        dz = wheel * self.z_sensitivity
        command = torch.zeros(7, dtype=torch.float32)
        command[0] = float(np.clip(dx, -1.0, 1.0))
        command[1] = float(np.clip(dy, -1.0, 1.0))
        command[2] = float(np.clip(dz, -1.0, 1.0))
        command[6] = -1.0 if self._close_gripper else 1.0
        return command


class Hdf5DemoWriter:
    def __init__(self, dataset_file: Path):
        dataset_file.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_file = dataset_file
        self.file = h5py.File(dataset_file, "a")
        self.data_group = self.file.require_group("data")
        self.string_dtype = h5py.string_dtype(encoding="utf-8")
        self.next_demo_id = self._find_next_demo_id()
        self.file.attrs["task"] = args_cli.task
        self.file.attrs["created_or_appended_at"] = datetime.now().isoformat(timespec="seconds")
        self.file.attrs["format"] = "task319_hover_grasp_teleop_v1"
        self.file.attrs["git_commit"] = _git_commit()

    def _find_next_demo_id(self) -> int:
        max_id = -1
        for key in self.data_group.keys():
            if key.startswith("demo_"):
                try:
                    max_id = max(max_id, int(key.split("_", 1)[1]))
                except ValueError:
                    continue
        return max_id + 1

    def close(self) -> None:
        self.file.flush()
        self.file.close()

    def write(self, buffer: DemoBuffer, *, success: bool, reason: str, metadata: dict[str, Any]) -> str | None:
        if len(buffer) < int(args_cli.min_steps_to_save):
            print(f"[SKIP] demo too short: {len(buffer)} steps < {args_cli.min_steps_to_save}", flush=True)
            return None

        name = f"demo_{self.next_demo_id:06d}"
        self.next_demo_id += 1
        group = self.data_group.create_group(name)

        def stack(values: list[np.ndarray], dtype=np.float32) -> np.ndarray:
            if not values:
                return np.zeros((0,), dtype=dtype)
            return np.asarray(values, dtype=dtype)

        group.create_dataset("actions", data=stack(buffer.actions), compression="gzip")
        group.create_dataset("rewards", data=stack(buffer.rewards), compression="gzip")
        group.create_dataset("terminated", data=stack(buffer.terminated, dtype=np.bool_), compression="gzip")
        group.create_dataset("truncated", data=stack(buffer.truncated, dtype=np.bool_), compression="gzip")
        group.create_dataset("success", data=np.asarray(success, dtype=np.bool_))
        group.create_dataset("reason", data=str(reason), dtype=self.string_dtype)
        group.create_dataset("metadata_json", data=json.dumps(metadata, ensure_ascii=False, sort_keys=True), dtype=self.string_dtype)

        obs = group.create_group("obs")
        obs.create_dataset("policy", data=stack(buffer.obs_policy), compression="gzip")
        obs.create_dataset("tcp_pose_w", data=stack(buffer.tcp_pose_w), compression="gzip")
        obs.create_dataset("target_rel_tcp", data=stack(buffer.target_rel_tcp), compression="gzip")
        obs.create_dataset("right_q", data=stack(buffer.right_q), compression="gzip")
        obs.create_dataset("right_qd", data=stack(buffer.right_qd), compression="gzip")
        obs.create_dataset("gripper_width", data=stack(buffer.gripper_width), compression="gzip")
        obs.create_dataset("command_raw_se3", data=stack(buffer.command_raw), compression="gzip")

        debug = group.create_group("debug")
        debug.create_dataset("object_pose_w", data=stack(buffer.object_pose_w), compression="gzip")
        debug.create_dataset("lift_m", data=stack(buffer.lift_m), compression="gzip")
        debug.create_dataset("xy_error_m", data=stack(buffer.xy_error_m), compression="gzip")
        debug.create_dataset("grasp_latched", data=stack(buffer.grasp_latched, dtype=np.bool_), compression="gzip")

        group.attrs["success"] = bool(success)
        group.attrs["reason"] = str(reason)
        group.attrs["num_steps"] = len(buffer)
        self.file.flush()
        print(f"[SAVE] {name}: success={success} steps={len(buffer)} reason={reason}", flush=True)
        return name


def _make_motion_device() -> tuple[object, object | None]:
    control_keyboard = None
    if args_cli.teleop_device == "keyboard":
        motion_device = Se3Keyboard(
            Se3KeyboardCfg(
                pos_sensitivity=float(args_cli.pos_sensitivity),
                rot_sensitivity=float(args_cli.rot_sensitivity),
                sim_device="cpu",
            )
        )
    elif args_cli.teleop_device == "mouse":
        motion_device = Se3OrdinaryMouse(
            xy_sensitivity=float(args_cli.mouse_xy_sensitivity),
            z_sensitivity=float(args_cli.mouse_z_sensitivity),
        )
        control_keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.0, rot_sensitivity=0.0, sim_device="cpu"))
        control_keyboard.add_callback("K", motion_device.toggle_gripper)
    else:
        motion_device = Se3SpaceMouse(
            Se3SpaceMouseCfg(
                pos_sensitivity=float(args_cli.pos_sensitivity),
                rot_sensitivity=float(args_cli.rot_sensitivity),
                sim_device="cpu",
            )
        )
        # SpaceMouse has no N/M hotkeys.  Keep a keyboard listener for save/reset/quit commands.
        control_keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.0, rot_sensitivity=0.0, sim_device="cpu"))
    return motion_device, control_keyboard


def _install_callbacks(devices: list[object], requests: dict[str, str | None]) -> None:
    def set_request(name: str):
        def _cb():
            requests["event"] = name

        return _cb

    for device in devices:
        if device is None or not hasattr(device, "add_callback"):
            continue
        device.add_callback("N", set_request("save_success"))
        device.add_callback("M", set_request("save_failure"))
        device.add_callback("R", set_request("reset"))
        device.add_callback("ESCAPE", set_request("quit"))


def _map_command_to_action(command: torch.Tensor) -> torch.Tensor:
    command_cpu = command.detach().flatten().cpu()
    action = torch.zeros((1, 4), dtype=torch.float32)
    translation = command_cpu[0:3] * float(args_cli.action_scale)
    if args_cli.teleop_device == "keyboard" and bool(args_cli.keyboard_full_step):
        pressed = torch.abs(command_cpu[0:3]) > 1.0e-6
        translation = torch.where(pressed, torch.sign(command_cpu[0:3]), torch.zeros_like(command_cpu[0:3]))
    action[0, 0:3] = torch.clamp(translation, -1.0, 1.0)
    # IsaacLab SE(3) devices return +1 for open and -1 for close.  The Task319
    # env uses -1 for open and +1 for close, so the gripper command is inverted.
    if command_cpu.numel() >= 7:
        action[0, 3] = float(np.clip(-float(command_cpu[6]), -1.0, 1.0))
    else:
        action[0, 3] = -1.0
    return action


def _snapshot(env, obs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    unwrapped = env.unwrapped
    with torch.no_grad():
        tcp_pos = unwrapped._tcp_pos_w()[0]
        tcp_quat = unwrapped._robot.data.body_quat_w[0, unwrapped._ee_body_id]
        tcp_pose = torch.cat((tcp_pos, tcp_quat), dim=0)
        target = unwrapped._estimated_grasp_target_w()[0]
        target_rel = target - tcp_pos
        right_q = unwrapped._robot.data.joint_pos[0, unwrapped._right_joint_ids]
        right_qd = unwrapped._robot.data.joint_vel[0, unwrapped._right_joint_ids]
        if unwrapped._finger_joint_ids:
            finger_pos = unwrapped._robot.data.joint_pos[0, unwrapped._finger_joint_ids]
            gripper_width = torch.mean(finger_pos).reshape(1)
        else:
            gripper_width = (GRIPPER_OPEN_M - unwrapped._gripper_close_cmd[0] * (GRIPPER_OPEN_M - GRIPPER_CLOSED_M)).reshape(1)
        object_pose = torch.cat((unwrapped._object.data.root_pos_w[0], unwrapped._object.data.root_quat_w[0]), dim=0)
        lift = (unwrapped._object.data.root_pos_w[0, 2] - unwrapped._object_initial_z[0]).reshape(1)
        xy_error = torch.linalg.norm(target_rel[0:2]).reshape(1)
        latched = unwrapped._grasp_latched[0].reshape(1) if hasattr(unwrapped, "_grasp_latched") else torch.zeros(1, dtype=torch.bool)
        return {
            "obs_policy": _to_numpy(obs["policy"][0]).astype(np.float32),
            "tcp_pose_w": _to_numpy(tcp_pose).astype(np.float32),
            "target_rel_tcp": _to_numpy(target_rel).astype(np.float32),
            "right_q": _to_numpy(right_q).astype(np.float32),
            "right_qd": _to_numpy(right_qd).astype(np.float32),
            "gripper_width": _to_numpy(gripper_width).astype(np.float32),
            "object_pose_w": _to_numpy(object_pose).astype(np.float32),
            "lift_m": _to_numpy(lift).astype(np.float32),
            "xy_error_m": _to_numpy(xy_error).astype(np.float32),
            "grasp_latched": _to_numpy(latched).astype(np.bool_),
        }


def _append_step(
    buffer: DemoBuffer,
    *,
    obs: dict[str, torch.Tensor],
    env,
    command: torch.Tensor,
    action: torch.Tensor,
) -> None:
    snap = _snapshot(env, obs)
    buffer.actions.append(_to_numpy(action[0]).astype(np.float32))
    buffer.command_raw.append(_to_numpy(command).astype(np.float32))
    buffer.obs_policy.append(snap["obs_policy"])
    buffer.tcp_pose_w.append(snap["tcp_pose_w"])
    buffer.target_rel_tcp.append(snap["target_rel_tcp"])
    buffer.right_q.append(snap["right_q"])
    buffer.right_qd.append(snap["right_qd"])
    buffer.gripper_width.append(snap["gripper_width"])
    buffer.object_pose_w.append(snap["object_pose_w"])
    buffer.lift_m.append(snap["lift_m"])
    buffer.xy_error_m.append(snap["xy_error_m"])
    buffer.grasp_latched.append(snap["grasp_latched"])


def _episode_metadata(env, *, episode_index: int, reason: str) -> dict[str, Any]:
    cfg = env.unwrapped.cfg
    return {
        "episode_index": int(episode_index),
        "reason": reason,
        "teleop_device": args_cli.teleop_device,
        "sim_device": args_cli.device,
        "pos_sensitivity": float(args_cli.pos_sensitivity),
        "rot_sensitivity": float(args_cli.rot_sensitivity),
        "action_scale": float(args_cli.action_scale),
        "sim_dt": float(cfg.sim.dt),
        "decimation": int(cfg.decimation),
        "episode_length_s": float(cfg.episode_length_s),
        "object_spawn_xy_range_m": float(cfg.object_spawn_xy_range_m),
        "target_estimate_xy_noise_std_m": float(cfg.target_estimate_xy_noise_std_m),
        "target_estimate_z_noise_std_m": float(cfg.target_estimate_z_noise_std_m),
        "grasp_latch_enabled": bool(cfg.grasp_latch_enabled),
        "git_commit": _git_commit(),
    }


def _print_instructions(dataset_file: Path) -> None:
    print("", flush=True)
    print("=== Task319 hover-to-grasp teleop collector ===", flush=True)
    print(f"Dataset: {dataset_file}", flush=True)
    print("GUI 窗口打开后，先用鼠标点一下 Isaac Sim 视窗，让键盘输入生效。", flush=True)
    if args_cli.teleop_device == "mouse":
        print("鼠标模式：按住左键拖动 = TCP 平面 XY；滚轮 = TCP 上/下；K = 开/合夹爪。", flush=True)
    else:
        print("按键：W/S = TCP X，A/D = TCP Y，Q/E = TCP Z，K = 开/合夹爪。", flush=True)
    print("采集控制：N = 保存成功，M = 保存失败，R = 丢弃并重置，ESC 或 Ctrl+C = 退出。", flush=True)
    print("第一版只采竖直向下抓取，不要用旋转键；默认也不启用隐藏 latch。", flush=True)
    print(f"当前 TCP 每步最大位移: {args_cli.tcp_step_m:.3f} m；键盘满步模式: {args_cli.keyboard_full_step}", flush=True)
    print("", flush=True)


def main() -> None:
    dataset_file = Path(args_cli.dataset_file).expanduser() if args_cli.dataset_file else _default_dataset_file()
    _print_instructions(dataset_file)

    env_cfg = Task319LocalSuctionGraspEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.episode_length_s = float(args_cli.episode_length_s)
    env_cfg.debug_vis = not bool(args_cli.disable_debug_vis)
    env_cfg.grasp_latch_enabled = bool(args_cli.enable_grasp_latch)
    env_cfg.max_delta_pos_m = float(args_cli.tcp_step_m)
    env_cfg.max_joint_step_rad = float(args_cli.joint_step_rad)
    env_cfg.object_spawn_xy_range_m = float(args_cli.object_spawn_xy_range_m)
    env_cfg.target_estimate_xy_noise_std_m *= float(args_cli.target_noise_scale)
    env_cfg.target_estimate_z_noise_std_m *= float(args_cli.target_noise_scale)

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    motion_device, control_keyboard = _make_motion_device()
    requests: dict[str, str | None] = {"event": None}
    _install_callbacks([motion_device, control_keyboard], requests)

    writer = Hdf5DemoWriter(dataset_file)
    buffer = DemoBuffer()
    episode_index = 0
    saved_success = 0
    global_step = 0

    try:
        while simulation_app.is_running():
            event = requests.get("event")
            requests["event"] = None
            if event == "quit":
                print("[QUIT] requested from keyboard.", flush=True)
                break
            if event in ("save_success", "save_failure"):
                success = event == "save_success"
                reason = "manual_success" if success else "manual_failure"
                if success or bool(args_cli.save_failures):
                    metadata = _episode_metadata(env, episode_index=episode_index, reason=reason)
                    name = writer.write(buffer, success=success, reason=reason, metadata=metadata)
                    if name is not None:
                        saved_success += int(success)
                buffer.clear()
                episode_index += 1
                obs, _ = env.reset()
                motion_device.reset()
                if control_keyboard is not None:
                    control_keyboard.reset()
                print(f"[RESET] episode={episode_index} success_demos={saved_success}", flush=True)
                continue
            if event == "reset":
                print(f"[RESET] discarded episode={episode_index} steps={len(buffer)}", flush=True)
                buffer.clear()
                episode_index += 1
                obs, _ = env.reset()
                motion_device.reset()
                if control_keyboard is not None:
                    control_keyboard.reset()
                continue

            command = motion_device.advance()
            action_cpu = _map_command_to_action(command)
            action = action_cpu.to(env.unwrapped.device)
            _append_step(buffer, obs=obs, env=env, command=command, action=action_cpu)
            obs, reward, terminated, truncated, _ = env.step(action)
            buffer.rewards.append(_to_numpy(reward[0]).astype(np.float32).reshape(1))
            buffer.terminated.append(np.asarray(_first_bool(terminated), dtype=np.bool_).reshape(1))
            buffer.truncated.append(np.asarray(_first_bool(truncated), dtype=np.bool_).reshape(1))

            global_step += 1
            done = _first_bool(terminated) or _first_bool(truncated)
            auto_success = bool(env.unwrapped._success[0].detach().cpu().item()) if hasattr(env.unwrapped, "_success") else False
            if args_cli.print_interval > 0 and global_step % int(args_cli.print_interval) == 0:
                snap = _snapshot(env, obs)
                print(
                    "[STEP "
                    f"{global_step}] ep={episode_index} samples={len(buffer)} "
                    f"action_xyz=({float(action_cpu[0, 0]):+.1f},{float(action_cpu[0, 1]):+.1f},{float(action_cpu[0, 2]):+.1f}) "
                    f"xy_err={float(snap['xy_error_m'][0]):.3f}m "
                    f"lift={float(snap['lift_m'][0]):.3f}m "
                    f"gripper_action={float(action_cpu[0, 3]):+.2f} "
                    f"success_saved={saved_success}",
                    flush=True,
                )

            if done:
                logs = getattr(env.unwrapped, "extras", {}).get("log", {})
                logged_success = logs.get("Metrics/success_rate")
                if logged_success is not None:
                    if hasattr(logged_success, "detach"):
                        logged_success = logged_success.detach().cpu().item()
                    auto_success = float(logged_success) > 0.5
                reason = "auto_success" if auto_success else "auto_timeout_or_abort"
                if auto_success or bool(args_cli.save_failures):
                    metadata = _episode_metadata(env, episode_index=episode_index, reason=reason)
                    name = writer.write(buffer, success=auto_success, reason=reason, metadata=metadata)
                    if name is not None:
                        saved_success += int(auto_success)
                buffer.clear()
                episode_index += 1
                obs, _ = env.reset()
                motion_device.reset()
                if control_keyboard is not None:
                    control_keyboard.reset()
                print(f"[AUTO RESET] episode={episode_index} success_demos={saved_success}", flush=True)

            if args_cli.num_success_demos > 0 and saved_success >= int(args_cli.num_success_demos):
                print(f"[DONE] collected {saved_success} successful demos.", flush=True)
                break
            if args_cli.max_episodes > 0 and episode_index >= int(args_cli.max_episodes):
                print(f"[DONE] reached max_episodes={args_cli.max_episodes}.", flush=True)
                break
            if args_cli.steps > 0 and global_step >= int(args_cli.steps):
                print(f"[DONE] reached steps={args_cli.steps}.", flush=True)
                break
    except KeyboardInterrupt:
        print("[QUIT] Ctrl+C received.", flush=True)
    finally:
        if len(buffer) > 0:
            print(f"[INFO] Unsaved current episode has {len(buffer)} samples. Press N/M next time to save it.", flush=True)
        writer.close()
        env.close()
        print(f"[INFO] Dataset closed: {dataset_file}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
