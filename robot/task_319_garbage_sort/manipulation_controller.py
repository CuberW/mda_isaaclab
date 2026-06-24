"""Robot manipulation controller interface for garbage sorting.

This seam lets the MuJoCo Kuavo implementation and a future Isaac Sim
implementation expose the same high-level control surface. Task code should
call these methods instead of moving scene objects directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class PlannedArmPath:
    side: str
    joint_names: list[str]
    positions: np.ndarray
    source: str


class RobotManipulationController(ABC):
    """Minimal controller contract shared by MuJoCo and Isaac implementations."""

    @abstractmethod
    def get_base_pose(self) -> np.ndarray:
        """Return base pose as [x, y, yaw] in world frame."""

    @abstractmethod
    def get_camera_frame(self) -> str:
        """Return the primary robot RGB-D camera name."""

    @abstractmethod
    def move_base(self, forward_vel: float, turn_vel: float):
        """Command a short-horizon base velocity."""

    @abstractmethod
    def move_base_to(self, goal_xy_yaw: np.ndarray, steps: int = 300) -> bool:
        """Drive the base toward a world-frame [x, y, yaw] goal."""

    @abstractmethod
    def plan_arm_to(self, side: str, target_pose_world: np.ndarray) -> PlannedArmPath:
        """Plan an arm path to a world-frame end-effector pose."""

    def estimate_arm_reachability(self, side: str, target_pose_world: np.ndarray) -> dict:
        """Estimate whether an arm pose is reachable without executing it."""
        planned = self.plan_arm_to(side, target_pose_world)
        return {
            "reachable": bool(planned.positions.size),
            "source": planned.source,
            "ik_error": 0.0 if planned.positions.size else float("inf"),
            "joint_names": planned.joint_names,
        }

    @abstractmethod
    def execute_arm_trajectory(self, side: str, trajectory: PlannedArmPath) -> bool:
        """Execute an arm trajectory without moving scene objects."""

    @abstractmethod
    def open_gripper(self, side: str = "right"):
        """Open the selected gripper."""

    @abstractmethod
    def close_gripper(self, side: str = "right"):
        """Close the selected gripper."""

    @abstractmethod
    def verify_grasp_contact(self, object_id: str) -> bool:
        """Return True only when real finger/contact evidence exists."""

    @abstractmethod
    def carry_to(self, target_pose_world: np.ndarray) -> bool:
        """Move the carried object toward a release pose through robot motion."""

    @abstractmethod
    def release(self):
        """Release the grasped object."""
