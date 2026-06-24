"""Portable grasp execution plan generation.

The module intentionally returns robot goals and constraints, not object state
edits. A MuJoCo or Isaac controller must execute the returned poses through
real robot motion and verify contact before attaching or carrying.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraspExecutionPlan:
    object_world_pose: np.ndarray
    bin_world_pose: np.ndarray
    pregrasp_pose: np.ndarray
    grasp_pose: np.ndarray
    lift_pose: np.ndarray
    carry_pose: np.ndarray
    release_pose: np.ndarray
    contact_required: bool = True
    source: str = "geometry_keypoint_grasp_execution"


class GraspExecutionModule:
    """Generate portable grasp/carry/release goals from world-frame poses."""

    def __init__(
        self,
        approach_offset: float = 0.10,
        grasp_z_offset: float = 0.00,
        lift_height: float = 0.12,
        release_height: float = 0.28,
    ):
        self.approach_offset = float(approach_offset)
        self.grasp_z_offset = float(grasp_z_offset)
        self.lift_height = float(lift_height)
        self.release_height = float(release_height)

    def plan(
        self,
        object_world_pose: np.ndarray,
        bin_world_pose: np.ndarray,
        robot_state: dict | None = None,
        object_mask_or_points=None,
    ) -> GraspExecutionPlan:
        obj = np.asarray(object_world_pose, dtype=float).reshape(-1)
        bin_pose = np.asarray(bin_world_pose, dtype=float).reshape(-1)
        if obj.size < 3 or bin_pose.size < 3:
            raise ValueError("object_world_pose and bin_world_pose must contain at least xyz")

        obj_xyz = obj[:3].copy()
        bin_xyz = bin_pose[:3].copy()
        base_xy = np.asarray((robot_state or {}).get("base_xy", [0.0, 0.0]), dtype=float)
        approach = obj_xyz[:2] - base_xy[:2]
        norm = float(np.linalg.norm(approach))
        if norm < 1e-6:
            approach_dir = np.array([1.0, 0.0])
        else:
            approach_dir = approach / norm

        pregrasp = obj_xyz.copy()
        pregrasp[:2] -= approach_dir * self.approach_offset
        pregrasp[2] += max(0.02, self.grasp_z_offset + 0.03)

        grasp = obj_xyz.copy()
        grasp[2] += self.grasp_z_offset

        lift = grasp.copy()
        lift[2] += self.lift_height

        carry = bin_xyz.copy()
        carry[2] += self.release_height

        release = carry.copy()
        release[2] += 0.03

        return GraspExecutionPlan(
            object_world_pose=obj_xyz,
            bin_world_pose=bin_xyz,
            pregrasp_pose=pregrasp,
            grasp_pose=grasp,
            lift_pose=lift,
            carry_pose=carry,
            release_pose=release,
        )
