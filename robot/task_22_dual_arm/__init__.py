"""
Task 2.2: Dual-Arm VLA Collaborative Transport

Dual Franka Panda arms perform:
  1. LLM instruction parsing (Qwen2.5-7B + DAG-Plan architecture)
  2. Scene perception (GroundingDINO + SAM + 3D localization)
  3. Dual-arm task planning (DAG-based sub-task decomposition)
  4. Dual-arm trajectory generation (MoveIt-style RRT-Connect + WBC)
  5. VLA fine control (OpenVLA-style closed-loop adjustment)
  6. Synchronized execution with collision avoidance

Robot: Dual Franka Panda (7-DoF x 2)
Simulator: MuJoCo (robosuite-style environment)
Architecture: CogACT-style cognition-action decoupling
  - System 2 (cognition): LLM parses roles + DAG plans subtasks (~10Hz)
  - System 1 (action): IK + trajectory + collision avoidance (~50Hz+)
"""

from .state import DualArmState
from .coordinator import DualArmCoordinator
from .dag_planner import DAGTaskPlanner
from .pipeline import DualArmVLAPipeline

__all__ = [
    "DualArmState",
    "DualArmCoordinator",
    "DAGTaskPlanner",
    "DualArmVLAPipeline",
]
