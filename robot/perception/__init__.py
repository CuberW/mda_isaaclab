"""Perception seam for vision, VLM, RGB-D, and grasp-pose perception.

The implementation still lives in ``robot_common`` for backward compatibility;
task code should import through this package so perception changes stay local.
"""

from robot_common.perception import DetectionResult, PerceptionHub

__all__ = [
    "DetectionResult",
    "PerceptionHub",
]

