"""DualArmState — data holder for synchronized dual-arm pose."""

from dataclasses import dataclass

import numpy as np


@dataclass
class DualArmState:
    """State of both arms."""
    left_joints: np.ndarray
    right_joints: np.ndarray
    left_ee_pos: np.ndarray   # (x, y, z)
    left_ee_quat: np.ndarray  # (w, x, y, z)
    right_ee_pos: np.ndarray
    right_ee_quat: np.ndarray
    nominal_separation: float = 0.0

    @property
    def sync_error(self) -> float:
        """Deviation from the desired two-gripper relative distance."""
        current = float(np.linalg.norm(self.left_ee_pos - self.right_ee_pos))
        return abs(current - float(self.nominal_separation))
