"""Shared dataclasses for the task-319 grasp pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class CameraFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    t_camera_to_world: np.ndarray


@dataclass(slots=True)
class PerceivedObject:
    object_id: int
    object_name: str
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray
    center_2d: tuple[float, float]
    center_3d: np.ndarray | None = None
    waste_category: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class GraspCandidates:
    poses: np.ndarray
    scores: np.ndarray
    widths: np.ndarray
    centers_2d: np.ndarray
    depths: np.ndarray | None = None


@dataclass(slots=True)
class SelectedGrasp:
    pose_world: np.ndarray
    width: float
    score: float
    center_2d: tuple[float, float]
    target: PerceivedObject
    source: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class GraspResult:
    success: bool
    reason: str
    target_name: str
    target_category: str | None
    lift_height_m: float = 0.0
    base_drift_m: float = 0.0
