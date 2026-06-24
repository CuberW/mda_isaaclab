"""
Motion tracing for MuJoCo task execution.

The trace records enough state to tell whether the viewer should have shown
continuous motion, and whether a task relied on sudden state jumps.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass
class MotionTraceConfig:
    """Thresholds used when summarizing motion quality."""

    max_base_step: float = 0.08
    max_joint_step: float = 0.35
    max_object_step: float = 0.18
    max_ctrl_step: float = 0.75
    first_motion_eps: float = 1e-4


@dataclass
class MotionSample:
    """One sampled MuJoCo control step."""

    step: int
    sim_time: float
    wall_time: float
    base_pos: Optional[list[float]]
    qpos: list[float]
    ctrl: list[float]
    bodies: dict[str, list[float]]
    tilts_deg: dict[str, float]


@dataclass
class MotionTrace:
    """Collects motion samples and jump metrics for one task episode."""

    task_name: str
    tracked_bodies: list[str] = field(default_factory=list)
    tilt_bodies: list[str] = field(default_factory=list)
    controlled_joint_prefixes: Optional[list[str]] = None
    track_base: bool = False
    joint_qpos_addresses: Optional[list[int]] = None
    config: MotionTraceConfig = field(default_factory=MotionTraceConfig)
    samples: list[MotionSample] = field(default_factory=list)
    first_motion_time: Optional[float] = None
    max_base_step: float = 0.0
    max_joint_step: float = 0.0
    max_object_step: float = 0.0
    max_ctrl_step: float = 0.0
    max_tilt_deg: float = 0.0

    def __post_init__(self):
        self._started_at = time.time()
        self._prev: Optional[MotionSample] = None
        self._initial_rotations: dict[str, np.ndarray] = {}

    @classmethod
    def from_task_config(
        cls,
        task_name: str,
        raw_config: Optional[dict] = None,
        tracked_bodies: Optional[Iterable[str]] = None,
        tilt_bodies: Optional[Iterable[str]] = None,
        controlled_joint_prefixes: Optional[Iterable[str]] = None,
        track_base: bool = False,
    ) -> "MotionTrace":
        task_cfg = (raw_config or {}).get("task", {})
        cfg = MotionTraceConfig(
            max_base_step=float(task_cfg.get("max_base_step", 0.08)),
            max_joint_step=float(task_cfg.get("max_joint_step", 0.35)),
            max_object_step=float(task_cfg.get("max_object_step", 0.18)),
            max_ctrl_step=float(task_cfg.get("max_ctrl_step", 0.75)),
        )
        return cls(
            task_name=task_name,
            tracked_bodies=list(tracked_bodies or []),
            tilt_bodies=list(tilt_bodies or []),
            controlled_joint_prefixes=(
                list(controlled_joint_prefixes)
                if controlled_joint_prefixes is not None else None
            ),
            track_base=track_base,
            config=cfg,
        )

    def reset(self):
        self.samples.clear()
        self.first_motion_time = None
        self.max_base_step = 0.0
        self.max_joint_step = 0.0
        self.max_object_step = 0.0
        self.max_ctrl_step = 0.0
        self.max_tilt_deg = 0.0
        self._started_at = time.time()
        self._prev = None
        self._initial_rotations.clear()

    def record(self, env):
        wall_time = time.time() - self._started_at
        base_pos = None
        if self.track_base and "base_link" in getattr(env, "_body_names", []):
            base_pos = env.get_body_position("base_link").astype(float).tolist()

        bodies = {}
        for name in self.tracked_bodies:
            if name in getattr(env, "_body_names", []):
                bodies[name] = env.get_body_position(name).astype(float).tolist()

        tilts = {}
        for name in self.tilt_bodies:
            if name not in getattr(env, "_body_names", []):
                continue
            rot = env.get_body_rotation(name)
            if name not in self._initial_rotations:
                self._initial_rotations[name] = rot.copy()
            rel = self._initial_rotations[name].T @ rot
            angle = np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
            tilt_deg = float(np.rad2deg(angle))
            tilts[name] = tilt_deg
            self.max_tilt_deg = max(self.max_tilt_deg, tilt_deg)

        if self.joint_qpos_addresses is None:
            self.joint_qpos_addresses = []
            for jid in range(env.model.njnt):
                jtype = env.model.jnt_type[jid]
                name = env.model.joint(jid).name or ""
                if self.controlled_joint_prefixes is not None:
                    if not any(name.startswith(prefix) for prefix in self.controlled_joint_prefixes):
                        continue
                if jtype == 0:
                    continue
                if jtype not in (2, 3):
                    continue
                if name.endswith("_joint") and (name.startswith("obj") or name in ("rod_joint", "box_joint")):
                    continue
                self.joint_qpos_addresses.append(int(env.model.jnt_qposadr[jid]))

        sample = MotionSample(
            step=int(env.step_count),
            sim_time=float(env.data.time),
            wall_time=float(wall_time),
            base_pos=base_pos,
            qpos=env.data.qpos.astype(float).tolist(),
            ctrl=env.data.ctrl.astype(float).tolist(),
            bodies=bodies,
            tilts_deg=tilts,
        )

        if self._prev is not None:
            motion = 0.0
            if sample.base_pos is not None and self._prev.base_pos is not None:
                base_delta = float(np.linalg.norm(
                    np.asarray(sample.base_pos[:2]) - np.asarray(self._prev.base_pos[:2])
                ))
                self.max_base_step = max(self.max_base_step, base_delta)
                motion = max(motion, base_delta)

            if self.joint_qpos_addresses:
                q_now = np.asarray(sample.qpos)[self.joint_qpos_addresses]
                q_prev = np.asarray(self._prev.qpos)[self.joint_qpos_addresses]
                q_delta = float(np.max(np.abs(q_now - q_prev)))
            else:
                q_delta = 0.0
            c_delta = float(np.max(np.abs(
                np.asarray(sample.ctrl) - np.asarray(self._prev.ctrl)
            ))) if sample.ctrl else 0.0
            self.max_joint_step = max(self.max_joint_step, q_delta)
            self.max_ctrl_step = max(self.max_ctrl_step, c_delta)
            motion = max(motion, q_delta)

            for body, pos in sample.bodies.items():
                prev_pos = self._prev.bodies.get(body)
                if prev_pos is None:
                    continue
                obj_delta = float(np.linalg.norm(np.asarray(pos) - np.asarray(prev_pos)))
                self.max_object_step = max(self.max_object_step, obj_delta)
                motion = max(motion, obj_delta)

            if self.first_motion_time is None and motion > self.config.first_motion_eps:
                self.first_motion_time = sample.wall_time

        self.samples.append(sample)
        self._prev = sample

    def summary(self) -> dict:
        thresholds = {
            "max_base_step": self.config.max_base_step,
            "max_joint_step": self.config.max_joint_step,
            "max_object_step": self.config.max_object_step,
            "max_ctrl_step": self.config.max_ctrl_step,
        }
        observed = {
            "max_base_step": round(self.max_base_step, 5),
            "max_joint_step": round(self.max_joint_step, 5),
            "max_object_step": round(self.max_object_step, 5),
            "max_ctrl_step": round(self.max_ctrl_step, 5),
        }
        violations = {
            key: observed[key]
            for key, limit in thresholds.items()
            if observed[key] > limit
        }
        return {
            "task_name": self.task_name,
            "samples": len(self.samples),
            "first_motion_time": (
                round(self.first_motion_time, 3)
                if self.first_motion_time is not None else None
            ),
            "observed": observed,
            "thresholds": thresholds,
            "max_tilt_deg": round(self.max_tilt_deg, 3),
            "violations": violations,
            "smooth": not violations,
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "samples": [sample.__dict__ for sample in self.samples],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
