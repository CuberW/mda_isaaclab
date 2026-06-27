#!/usr/bin/env python3
"""Summarize and plot Task319 local-grasp RL training artifacts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs" / "skrl"
TELEOP_ROOT = ROOT / "task_319_garbage_sort" / "output" / "teleop_grasp_datasets"
OUT_DIR = Path(__file__).resolve().parent / "rl_training_analysis"

RUN_FAMILIES = (
    "task319_hover_descent_grasp",
    "task319_hover_descent_grasp_sac",
    "task319_local_suction_grasp",
)

PREFERRED_SCALARS = (
    "Metrics/success_rate",
    "Metrics/grasp_retained_rate",
    "Metrics/grasp_latch_rate",
    "Reward / Total reward (mean)",
    "Episode_Reward/success_bonus",
    "Episode_Reward/lift",
    "Episode_Reward/stage_lift",
    "Episode_Reward/stage_close",
    "Episode_Reward/close_near",
    "Episode_Reward/z_align",
    "Episode_Reward/xy_align",
    "Episode_Reward/penalty",
)


def _event_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("events.out.tfevents*"))


def _checkpoint_count(run_dir: Path) -> int:
    return len(list((run_dir / "checkpoints").glob("*.pt")))


def _run_algorithm(run_dir: Path, family: str) -> str:
    name = run_dir.name.lower()
    if "sac" in name or family.endswith("_sac"):
        return "SAC"
    if "ppo" in name:
        return "PPO"
    return "unknown"


def load_scalars() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for family in RUN_FAMILIES:
        family_dir = LOG_ROOT / family
        if not family_dir.exists():
            continue
        for run_dir in sorted([p for p in family_dir.iterdir() if p.is_dir()]):
            events = _event_files(run_dir)
            if not events:
                continue
            accumulator = event_accumulator.EventAccumulator(
                str(run_dir),
                size_guidance={event_accumulator.SCALARS: 0},
            )
            try:
                accumulator.Reload()
            except Exception as exc:
                run_rows.append(
                    {
                        "family": family,
                        "run": run_dir.name,
                        "algorithm": _run_algorithm(run_dir, family),
                        "error": repr(exc),
                    }
                )
                continue
            tags = accumulator.Tags().get("scalars", [])
            max_step = 0
            for tag in tags:
                scalars = accumulator.Scalars(tag)
                if scalars:
                    max_step = max(max_step, max(s.step for s in scalars))
                for scalar in scalars:
                    rows.append(
                        {
                            "family": family,
                            "run": run_dir.name,
                            "algorithm": _run_algorithm(run_dir, family),
                            "tag": tag,
                            "step": int(scalar.step),
                            "wall_time": float(scalar.wall_time),
                            "value": float(scalar.value),
                        }
                    )
            run_rows.append(
                {
                    "family": family,
                    "run": run_dir.name,
                    "algorithm": _run_algorithm(run_dir, family),
                    "event_files": len(events),
                    "checkpoints": _checkpoint_count(run_dir),
                    "tags": len(tags),
                    "max_step": int(max_step),
                    "has_success_rate": "Metrics/success_rate" in tags,
                    "has_grasp_retained_rate": "Metrics/grasp_retained_rate" in tags,
                    "has_grasp_latch_rate": "Metrics/grasp_latch_rate" in tags,
                    "path": str(run_dir.relative_to(ROOT)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(run_rows)


def summarize_runs(scalars: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    if scalars.empty:
        return runs.copy()
    finals: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (family, run, tag), group in scalars.sort_values("step").groupby(["family", "run", "tag"]):
        tail = group.tail(min(10, len(group)))
        finals[(family, run)][f"final10::{tag}"] = float(tail["value"].mean())
        best = float(group["value"].max())
        finals[(family, run)][f"max::{tag}"] = best
    rows = []
    for _, row in runs.iterrows():
        item = row.to_dict()
        item.update(finals.get((row["family"], row["run"]), {}))
        rows.append(item)
    return pd.DataFrame(rows)


def _smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    if values.size < 3:
        return values
    window = min(window, values.size)
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_scalars(scalars: pd.DataFrame, output: Path) -> None:
    if scalars.empty:
        return
    available = [tag for tag in PREFERRED_SCALARS if tag in set(scalars["tag"])]
    if not available:
        available = sorted(scalars["tag"].unique())[:12]

    ncols = 2
    nrows = math.ceil(len(available) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(4, 3.2 * nrows)), squeeze=False)
    color_by_algo = {"PPO": "#2563eb", "SAC": "#dc2626", "unknown": "#525252"}
    family_style = {
        "task319_hover_descent_grasp": "-",
        "task319_hover_descent_grasp_sac": "-",
        "task319_local_suction_grasp": "--",
    }

    for ax, tag in zip(axes.flat, available):
        tagged = scalars[scalars["tag"] == tag]
        for (family, run, algorithm), group in tagged.groupby(["family", "run", "algorithm"]):
            group = group.sort_values("step")
            steps = group["step"].to_numpy()
            values = group["value"].to_numpy()
            label = f"{algorithm} {run}"
            if family == "task319_local_suction_grasp":
                label = f"old suction {run}"
            ax.plot(
                steps,
                _smooth(values),
                color=color_by_algo.get(algorithm, "#525252"),
                linestyle=family_style.get(family, "-"),
                alpha=0.45,
                linewidth=1.4,
                label=label,
            )
        ax.set_title(tag)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
        if "rate" in tag:
            ax.set_ylim(-0.02, 1.02)

    for ax in axes.flat[len(available) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        # Keep only a compact legend. The full run list is in CSV/JSON.
        by_algo = {}
        for handle, label in zip(handles, labels):
            key = label.split()[0] if not label.startswith("old") else "old suction"
            by_algo.setdefault(key, handle)
        fig.legend(by_algo.values(), by_algo.keys(), loc="upper center", ncol=min(4, len(by_algo)))
    fig.suptitle("Task319 Local Grasp RL Training Curves", y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summarize_teleop() -> tuple[pd.DataFrame, pd.DataFrame]:
    file_rows = []
    demo_rows = []
    for path in sorted(TELEOP_ROOT.glob("*/task319_hover_grasp_teleop.hdf5")):
        with h5py.File(path, "r") as h5:
            data = h5.get("data")
            demos = sorted(data.keys()) if data is not None else []
            success_count = 0
            total_steps = 0
            for demo_name in demos:
                group = data[demo_name]
                actions = np.asarray(group["actions"]) if "actions" in group else np.empty((0, 4))
                rewards = np.asarray(group["rewards"]) if "rewards" in group else np.empty((0,))
                success = bool(np.asarray(group["success"]).item()) if "success" in group else False
                success_count += int(success)
                total_steps += int(actions.shape[0])
                demo_rows.append(
                    {
                        "file": str(path.relative_to(ROOT)),
                        "demo": demo_name,
                        "success": success,
                        "steps": int(actions.shape[0]),
                        "reward_sum": float(np.asarray(rewards, dtype=np.float32).sum()),
                        "reward_mean": float(np.asarray(rewards, dtype=np.float32).mean()) if rewards.size else 0.0,
                        "close_cmd_mean": float(actions[:, 3].mean()) if actions.ndim == 2 and actions.shape[1] >= 4 else np.nan,
                    }
                )
            file_rows.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "demos": len(demos),
                    "success_demos": success_count,
                    "steps": total_steps,
                    "size_bytes": path.stat().st_size,
                    "format": h5.attrs.get("format", ""),
                    "task": h5.attrs.get("task", ""),
                }
            )
    return pd.DataFrame(file_rows), pd.DataFrame(demo_rows)


def plot_teleop(file_df: pd.DataFrame, demo_df: pd.DataFrame, output: Path) -> None:
    if file_df.empty and demo_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    if not file_df.empty:
        labels = [Path(p).parts[-2] for p in file_df["file"]]
        axes[0].bar(labels, file_df["demos"], color="#2563eb", label="all")
        axes[0].bar(labels, file_df["success_demos"], color="#16a34a", label="success")
        axes[0].set_title("Teleop demos")
        axes[0].set_ylabel("episodes")
        axes[0].tick_params(axis="x", rotation=25)
        axes[0].legend()
    if not demo_df.empty:
        colors = np.where(demo_df["success"].to_numpy(), "#16a34a", "#dc2626")
        axes[1].bar(np.arange(len(demo_df)), demo_df["steps"], color=colors)
        axes[1].set_title("Steps per demo")
        axes[1].set_xlabel("demo")
        axes[1].set_ylabel("steps")
        axes[2].bar(np.arange(len(demo_df)), demo_df["reward_sum"], color=colors)
        axes[2].set_title("Reward sum per demo")
        axes[2].set_xlabel("demo")
        axes[2].set_ylabel("sum reward")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_decision_summary(summary: pd.DataFrame, teleop_files: pd.DataFrame, output: Path) -> None:
    if summary.empty:
        return

    df = summary.copy()
    def short_label(row: pd.Series) -> str:
        parts = str(row["run"]).split("_")
        if len(parts) >= 2:
            return f"{parts[0][5:]} {parts[1][:5]}\n{row['algorithm']}"
        return f"{row['run']}\n{row['algorithm']}"

    df["label"] = df.apply(short_label, axis=1)
    df["autonomous_signal"] = (
        df.get("final10::Metrics/teacher_action_fraction", pd.Series(np.nan, index=df.index)).isna()
        | (df.get("final10::Metrics/teacher_action_fraction", pd.Series(np.nan, index=df.index)) < 0.5)
    )
    df["is_current_hover"] = df["family"].isin(["task319_hover_descent_grasp", "task319_hover_descent_grasp_sac"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    current = df[df["is_current_hover"]].sort_values(["algorithm", "max_step", "run"])
    x = np.arange(len(current))
    colors = np.where(current["autonomous_signal"], "#2563eb", "#dc2626")

    axes[0, 0].bar(x, current["final10::Reward / Total reward (mean)"].fillna(0.0), color=colors)
    axes[0, 0].set_title("Final mean reward by run")
    axes[0, 0].set_ylabel("final-10 scalar mean")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(current["label"], rotation=65, ha="right", fontsize=7)
    axes[0, 0].grid(True, axis="y", alpha=0.25)

    metric_cols = [
        ("success", "final10::Metrics/success_rate"),
        ("retained", "final10::Metrics/grasp_retained_rate"),
        ("latch", "final10::Metrics/grasp_latch_rate"),
        ("teacher", "final10::Metrics/teacher_action_fraction"),
    ]
    metric_df = current[["run"] + [col for _, col in metric_cols if col in current.columns]].copy()
    if not metric_df.empty:
        width = 0.20
        for i, (name, col) in enumerate(metric_cols):
            if col not in metric_df.columns:
                continue
            axes[0, 1].bar(x + (i - 1.5) * width, current[col].fillna(0.0), width=width, label=name)
        axes[0, 1].set_title("Success metrics vs teacher usage")
        axes[0, 1].set_ylim(0, 1.08)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(current["label"], rotation=65, ha="right", fontsize=7)
        axes[0, 1].legend(ncol=2)
        axes[0, 1].grid(True, axis="y", alpha=0.25)

    lift_col = "final10::Metrics/max_object_lift_m"
    if lift_col in current.columns:
        axes[1, 0].bar(x, 1000.0 * current[lift_col].fillna(0.0), color=colors)
        axes[1, 0].axhline(25.0, color="#111827", linestyle="--", linewidth=1.1, label="success threshold 25 mm")
        axes[1, 0].set_title("Observed object lift")
        axes[1, 0].set_ylabel("mm")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(current["label"], rotation=65, ha="right", fontsize=7)
        axes[1, 0].legend()
        axes[1, 0].grid(True, axis="y", alpha=0.25)

    axes[1, 1].axis("off")
    total_checkpoints = int(df["checkpoints"].fillna(0).sum())
    hover_checkpoints = int(current["checkpoints"].fillna(0).sum())
    teleop_demos = int(teleop_files["demos"].sum()) if not teleop_files.empty else 0
    teleop_success = int(teleop_files["success_demos"].sum()) if not teleop_files.empty else 0
    autonomous_hover = int(current["autonomous_signal"].sum())
    teacher_hover = int((~current["autonomous_signal"]).sum())
    notes = [
        f"Runs parsed: {len(df)}",
        f"Hover-descent runs: {len(current)}",
        f"Checkpoints: {total_checkpoints} total / {hover_checkpoints} current hover",
        f"Scalar rows: see rl_scalars.csv",
        f"No-teacher-metric/low-teacher hover runs: {autonomous_hover}",
        f"Teacher-heavy hover runs: {teacher_hover}",
        f"Teleop demos: {teleop_demos} total / {teleop_success} success",
        "Blue bars: no logged teacher fraction or < 0.5",
        "Red bars: teacher fraction >= 0.5; not deployable proof",
    ]
    axes[1, 1].text(0.02, 0.98, "\n".join(notes), va="top", ha="left", fontsize=12)

    fig.suptitle("Task319 RL evidence for mainline grasp strategy", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scalars, runs = load_scalars()
    summary = summarize_runs(scalars, runs)
    teleop_files, teleop_demos = summarize_teleop()

    scalars.to_csv(OUT_DIR / "rl_scalars.csv", index=False)
    runs.to_csv(OUT_DIR / "rl_runs.csv", index=False)
    summary.to_csv(OUT_DIR / "rl_run_summary.csv", index=False)
    teleop_files.to_csv(OUT_DIR / "teleop_files.csv", index=False)
    teleop_demos.to_csv(OUT_DIR / "teleop_demos.csv", index=False)
    plot_scalars(scalars, OUT_DIR / "rl_training_curves.png")
    plot_teleop(teleop_files, teleop_demos, OUT_DIR / "teleop_dataset_summary.png")
    plot_decision_summary(summary, teleop_files, OUT_DIR / "mainline_rl_decision_summary.png")

    payload = {
        "log_root": str(LOG_ROOT.relative_to(ROOT)),
        "teleop_root": str(TELEOP_ROOT.relative_to(ROOT)),
        "num_scalar_rows": int(len(scalars)),
        "num_runs": int(len(runs)),
        "num_teleop_files": int(len(teleop_files)),
        "num_teleop_demos": int(len(teleop_demos)),
        "run_summary": summary.replace({np.nan: None}).to_dict(orient="records"),
        "teleop_files": teleop_files.replace({np.nan: None}).to_dict(orient="records"),
        "teleop_demos": teleop_demos.replace({np.nan: None}).to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
