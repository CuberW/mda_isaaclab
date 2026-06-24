"""
Robot Common Infrastructure - Unified architecture for active robot tasks.

Three-layer architecture:
  Layer 1: Perception - Detection, segmentation, 3D localization
  Layer 2: Decision  - LLM parsing, VLA, task planning
  Layer 3: Execution - Motion planning, IK, grasping, navigation

Tasks:
  2.2  - Dual-arm VLA collaborative transport (Panda x2)
  3.19 - Stretch garbage sorting and disposal
"""

__version__ = "1.0.0"
