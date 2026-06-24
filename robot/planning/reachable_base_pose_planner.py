"""Reachable base pose planner for mobile manipulation.

This module answers the State 3 question: where should the mobile base stand
so the arm can actually reach hover/grasp/lift and Nav2 can actually reach the
base pose?  It is robot-agnostic and uses injected callbacks for arm IK and
base planning so it can run against ROS2 full mode or MuJoCo light mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


ReachabilityFn = Callable[[str, np.ndarray, np.ndarray | None], dict]
BaseFeasibilityFn = Callable[[np.ndarray, np.ndarray], dict]


@dataclass(frozen=True)
class MobileManipulationGoal:
    object_world_pose: np.ndarray
    bin_world_pose: np.ndarray
    pregrasp_pose: np.ndarray
    grasp_pose: np.ndarray
    lift_pose: np.ndarray
    side: str = "right"


@dataclass(frozen=True)
class ReachableBaseCandidate:
    base_pose: np.ndarray
    score: float
    pregrasp: dict
    grasp: dict
    lift: dict
    base_path: dict
    source: str = "reachable_base_pose_search"


@dataclass(frozen=True)
class ReachableBasePlan:
    success: bool
    candidate: ReachableBaseCandidate | None = None
    candidates: tuple[ReachableBaseCandidate, ...] = ()
    reason: str = ""


@dataclass
class ReachableBasePosePlanner:
    radii: Sequence[float] = (0.42, 0.50, 0.58, 0.68, 0.78)
    angle_count: int = 24
    yaw_offsets: Sequence[float] = (0.0, 0.14, -0.14)
    max_candidates: int = 96
    max_nav_path_m: float = 8.0
    ik_error_limit_m: float = 0.05

    def plan(
        self,
        *,
        start_base_pose: np.ndarray,
        goal: MobileManipulationGoal,
        arm_reachability: ReachabilityFn,
        base_feasibility: BaseFeasibilityFn,
    ) -> ReachableBasePlan:
        start = np.asarray(start_base_pose[:3], dtype=float)
        obj = np.asarray(goal.object_world_pose[:3], dtype=float)
        scored: list[ReachableBaseCandidate] = []
        for base_pose in self._sample_base_poses(obj, start):
            base_path = base_feasibility(start, base_pose)
            if not bool(base_path.get("reachable", False)):
                continue
            pre = arm_reachability(goal.side, goal.pregrasp_pose, base_pose)
            grasp = arm_reachability(goal.side, goal.grasp_pose, base_pose)
            lift = arm_reachability(goal.side, goal.lift_pose, base_pose)
            if not (pre.get("reachable") and grasp.get("reachable") and lift.get("reachable")):
                continue
            max_ik = max(
                float(pre.get("ik_error", float("inf"))),
                float(grasp.get("ik_error", float("inf"))),
                float(lift.get("ik_error", float("inf"))),
            )
            if not np.isfinite(max_ik) or max_ik > self.ik_error_limit_m:
                continue
            travel = float(base_path.get("path_length", np.linalg.norm(base_pose[:2] - start[:2])))
            clearance_penalty = float(base_path.get("clearance_penalty", 0.0))
            standoff = float(np.linalg.norm(obj[:2] - base_pose[:2]))
            score = float(
                2.5 * max_ik
                + 0.10 * travel
                + 0.35 * abs(standoff - 0.58)
                + clearance_penalty
            )
            scored.append(ReachableBaseCandidate(
                base_pose=base_pose.copy(),
                score=score,
                pregrasp=pre,
                grasp=grasp,
                lift=lift,
                base_path=base_path,
            ))
        if not scored:
            return ReachableBasePlan(False, None, tuple(), "no_nav_and_arm_reachable_base_pose")
        scored.sort(key=lambda item: item.score)
        return ReachableBasePlan(True, scored[0], tuple(scored), "")

    def _sample_base_poses(self, object_pos: np.ndarray, start: np.ndarray) -> list[np.ndarray]:
        candidates: list[np.ndarray] = [start.copy()]
        angles = np.linspace(-math.pi, math.pi, int(self.angle_count), endpoint=False)
        for radius in self.radii:
            for angle in angles:
                base_xy = object_pos[:2] - float(radius) * np.array(
                    [math.cos(float(angle)), math.sin(float(angle))],
                    dtype=float,
                )
                yaw = math.atan2(float(object_pos[1] - base_xy[1]), float(object_pos[0] - base_xy[0]))
                for offset in self.yaw_offsets:
                    candidates.append(np.array([base_xy[0], base_xy[1], yaw + float(offset)], dtype=float))
        candidates.sort(key=lambda pose: (
            float(np.linalg.norm(pose[:2] - start[:2])),
            abs(float(math.atan2(math.sin(pose[2] - start[2]), math.cos(pose[2] - start[2])))),
        ))
        out: list[np.ndarray] = []
        seen: set[tuple[float, float, float]] = set()
        for pose in candidates:
            key = (round(float(pose[0]), 3), round(float(pose[1]), 3), round(float(pose[2]), 3))
            if key in seen:
                continue
            seen.add(key)
            out.append(pose)
            if len(out) >= int(self.max_candidates):
                break
        return out


__all__ = [
    "MobileManipulationGoal",
    "ReachableBaseCandidate",
    "ReachableBasePlan",
    "ReachableBasePosePlanner",
]
