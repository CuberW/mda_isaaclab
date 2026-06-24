"""Trajectory utilities shared by grasp execution scripts."""

from __future__ import annotations

import numpy as np


def min_jerk(alpha: float) -> float:
    alpha = max(0.0, min(1.0, float(alpha)))
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def interpolate_pose_linear(start: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Linear position interpolation with matrix-valued orientation blend placeholder."""

    beta = min_jerk(alpha)
    pose = np.array(start, dtype=np.float32).copy()
    pose[:3, 3] = (1.0 - beta) * start[:3, 3] + beta * target[:3, 3]
    pose[:3, :3] = target[:3, :3]
    return pose
