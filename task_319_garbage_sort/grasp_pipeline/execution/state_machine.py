"""Simple grasp execution state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class GraspState(str, Enum):
    IDLE = "IDLE"
    APPROACH = "APPROACH"
    PRE_GRASP = "PRE_GRASP"
    GRASP = "GRASP"
    LIFT = "LIFT"
    HOLD = "HOLD"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass(slots=True)
class GraspStateMachine:
    approach_offset_m: float = 0.15
    lift_height_m: float = 0.20
    max_retries: int = 3
    state: GraspState = GraspState.IDLE
    retry_count: int = 0
    grasp_pose_world: np.ndarray | None = None
    grasp_width_m: float = 0.0
    failure_reason: str = ""
    history: list[GraspState] = field(default_factory=list)

    def start(self, grasp_pose_world: np.ndarray, grasp_width_m: float) -> None:
        self.grasp_pose_world = np.array(grasp_pose_world, dtype=np.float32)
        self.grasp_width_m = float(grasp_width_m)
        self.state = GraspState.APPROACH
        self.history = [self.state]

    def transition(self, state: GraspState) -> None:
        self.state = state
        self.history.append(state)

    def fail(self, reason: str) -> None:
        self.failure_reason = reason
        self.transition(GraspState.FAILED)

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def retry(self) -> None:
        self.retry_count += 1
        self.failure_reason = ""
        self.transition(GraspState.APPROACH if self.can_retry() else GraspState.FAILED)

    def pre_grasp_pose(self) -> np.ndarray:
        if self.grasp_pose_world is None:
            raise RuntimeError("No grasp pose set.")
        pose = self.grasp_pose_world.copy()
        pose[:3, 3] -= pose[:3, :3] @ np.array([0.0, 0.0, self.approach_offset_m], dtype=np.float32)
        return pose

    def lift_pose(self) -> np.ndarray:
        if self.grasp_pose_world is None:
            raise RuntimeError("No grasp pose set.")
        pose = self.grasp_pose_world.copy()
        pose[2, 3] += self.lift_height_m
        return pose


def check_lift_success(object_z_m: float, table_surface_z_m: float, threshold_m: float = 0.05) -> bool:
    return float(object_z_m) > float(table_surface_z_m) + float(threshold_m)
