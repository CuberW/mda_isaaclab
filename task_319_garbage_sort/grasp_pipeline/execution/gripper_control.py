"""Width-based control helpers for the attached two-finger gripper."""

from __future__ import annotations

from dataclasses import dataclass

import torch


LEFT_FINGER_JOINT = "left_finger_joint"
RIGHT_FINGER_JOINT = "right_finger_joint"


@dataclass(slots=True)
class GripperLimits:
    min_width_m: float = 0.0
    max_width_m: float = 0.120
    default_open_width_m: float = 0.110
    grasp_extra_clearance_m: float = 0.018


class AttachedParallelGripper:
    """Map desired jaw width to the two prismatic finger joints."""

    def __init__(self, robot, *, limits: GripperLimits | None = None) -> None:
        self.robot = robot
        self.limits = limits or GripperLimits()
        self.joint_ids, self.joint_names = robot.find_joints([LEFT_FINGER_JOINT, RIGHT_FINGER_JOINT], preserve_order=True)
        if len(self.joint_ids) != 2:
            raise RuntimeError(f"Expected attached gripper joints, got {self.joint_names}.")

    def width_to_joint_positions(self, width_m: float) -> torch.Tensor:
        width_m = min(self.limits.max_width_m, float(width_m))
        per_finger = 0.5 * width_m
        return torch.full((self.robot.num_instances, 2), per_finger, device=self.robot.device)

    def width_to_joint_positions_unclamped(self, width_m: float) -> torch.Tensor:
        per_finger = 0.5 * float(width_m)
        return torch.full((self.robot.num_instances, 2), per_finger, device=self.robot.device)

    def clamp_width(self, width_m: float) -> float:
        return max(self.limits.min_width_m, min(self.limits.max_width_m, float(width_m)))

    def set_width(self, width_m: float) -> None:
        self.robot.set_joint_position_target(self.width_to_joint_positions(width_m), joint_ids=self.joint_ids)

    def set_width_unclamped(self, width_m: float) -> None:
        self.robot.set_joint_position_target(self.width_to_joint_positions_unclamped(width_m), joint_ids=self.joint_ids)

    def current_width(self) -> float:
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        return float(torch.sum(joint_pos[0]).detach().cpu().item())

    def open_width_for_grasp(self, requested_width_m: float, *, clearance_m: float | None = None) -> float:
        clearance = self.limits.grasp_extra_clearance_m if clearance_m is None else float(clearance_m)
        return self.clamp_width(float(requested_width_m) + clearance)

    def open_for_grasp(self, requested_width_m: float) -> None:
        self.set_width(self.open_width_for_grasp(requested_width_m))

    def close(self, width_m: float | None = None) -> None:
        if width_m is None:
            self.set_width(self.limits.min_width_m)
        else:
            self.set_width_unclamped(width_m)
