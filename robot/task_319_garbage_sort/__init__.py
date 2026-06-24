"""
Task 3.19: Wheeled Robot Garbage Sorting and Disposal

Stretch robot (hello-robot) performs:
  1. Scene scanning (RGB-D camera)
  2. YOLOv8 garbage detection + CLIP second-pass classification
  3. GraspNet RGB-D grasp planning
  4. Differential drive navigation to garbage
  5. Arm grasp execution
  6. Navigation to correct trash bin
  7. Release into categorized bin
  8. Loop until all garbage is sorted

Robot: Stretch RE1/RE2 (wheeled differential drive + 4-DoF telescoping arm)
Simulator: MuJoCo
"""

from .constants import (
    CATEGORY_RECYCLABLE, CATEGORY_KITCHEN, CATEGORY_HAZARDOUS, CATEGORY_OTHER,
    GARBAGE_CATEGORIES, OBJECT_TO_CATEGORY, SCENE_TRASH_TO_CATEGORY,
    COCO_CLASS_TO_CATEGORY, BIN_POSITIONS, BIN_BODY_BY_CATEGORY,
    GARBAGE_DETECTION_PROMPTS,
)
from .controller import StretchGarbageController
from .kuavo_controller import KuavoWheelGarbageController
from .manipulation_controller import PlannedArmPath, RobotManipulationController
from .pipeline import GarbageSortingPipeline

__all__ = [
    "CATEGORY_RECYCLABLE",
    "CATEGORY_KITCHEN",
    "CATEGORY_HAZARDOUS",
    "CATEGORY_OTHER",
    "GARBAGE_CATEGORIES",
    "OBJECT_TO_CATEGORY",
    "SCENE_TRASH_TO_CATEGORY",
    "COCO_CLASS_TO_CATEGORY",
    "BIN_POSITIONS",
    "BIN_BODY_BY_CATEGORY",
    "GARBAGE_DETECTION_PROMPTS",
    "KuavoWheelGarbageController",
    "PlannedArmPath",
    "RobotManipulationController",
    "StretchGarbageController",
    "GarbageSortingPipeline",
]
