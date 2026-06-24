"""
Unified configuration system with YAML-based task composition.

Each task can be configured by combining different modules:
  task.type: dual_arm_vla | garbage_sort
  robot.type: franka_panda_dual | stretch
  perception.detector: grounding_dino | yolo_world | yolov8
  ...etc
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ── Project paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SIMULATION_DIR = PROJECT_ROOT / "simulation"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


# ── Robot configs ──────────────────────────────────────────────
@dataclass
class RobotConfig:
    type: str = "stretch"           # franka_panda_dual | stretch
    simulator: str = "mujoco"       # mujoco | isaac_sim
    mjcf_path: str = ""
    camera_name: str = "camera_rgb"
    dof: int = 4
    control_freq: float = 50.0      # Hz


@dataclass
class CameraConfig:
    name: str = "head_camera"
    type: str = "rgbd"
    width: int = 640
    height: int = 480
    fov: float = 60.0
    position: list = field(default_factory=lambda: [0.0, 0.0, 1.0])
    rotation: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class PerceptionConfig:
    detector: str = "yolov8"        # grounding_dino | yolo_world | yolov8
    segmentor: str = "sam"          # sam | sam2 | none
    grasp_estimator: str = "graspnet"  # anygrasp | graspnet | none
    depth_method: str = "mujoco"    # mujoco | depth_anything_v2
    cameras: list = field(default_factory=lambda: [
        CameraConfig("head_camera", "rgbd", 640, 480),
    ])
    confidence_threshold: float = 0.5


@dataclass
class DecisionConfig:
    llm: str = "qwen2.5-7b"         # qwen2.5 | deepseek | gpt4o
    vla: str = "none"               # openvla | rdt-1b | octo | none
    task_planner: str = "state_machine"  # dag_plan | rekep | state_machine
    use_code_as_policies: bool = False
    llm_endpoint: str = ""          # API endpoint for LLM


@dataclass
class ExecutionConfig:
    motion_planner: str = "ik_solver"  # robosuite_wbc | ik_solver
    collision_checker: str = "mujoco"  # fcl | mujoco_contact
    ik_solver: str = "analytical"      # kdl | trac_ik | analytical
    official_ik: dict = field(default_factory=dict)
    official_control: dict = field(default_factory=dict)
    max_velocity: float = 1.0
    max_acceleration: float = 2.0


@dataclass
class TrainingConfig:
    framework: str = "robomimic"    # robomimic | lerobot
    policy: str = "act"             # act | diffusion_policy | bc
    data_augmentation: str = "none" # mimicgen | none
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 100


@dataclass
class TaskConfig:
    type: str = "garbage_sort"      # dual_arm_vla | garbage_sort
    name: str = ""
    max_steps: int = 500
    random_seed: int = 42
    robot: RobotConfig = field(default_factory=RobotConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scene_objects: list = field(default_factory=list)
    trash_bins: list = field(default_factory=list)


# ── Config loader ──────────────────────────────────────────────
def load_config(config_path: str) -> TaskConfig:
    """Load task configuration from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    task_data = raw.get("task", {})
    robot_data = raw.get("robot", {})
    perception_data = raw.get("perception", {})
    decision_data = raw.get("decision", {})
    execution_data = raw.get("execution", {})
    training_data = raw.get("training", {})

    cameras = []
    for cam in perception_data.get("cameras", []):
        cameras.append(CameraConfig(**cam) if isinstance(cam, dict) else cam)

    return TaskConfig(
        type=task_data.get("type", "garbage_sort"),
        name=task_data.get("name", ""),
        max_steps=task_data.get("max_steps", 500),
        random_seed=task_data.get("random_seed", 42),
        robot=RobotConfig(**robot_data) if robot_data else RobotConfig(),
        perception=PerceptionConfig(
            detector=perception_data.get("detector", "yolov8"),
            segmentor=perception_data.get("segmentor", "sam"),
            grasp_estimator=perception_data.get("grasp_estimator", "graspnet"),
            depth_method=perception_data.get("depth_method", "mujoco"),
            cameras=cameras,
            confidence_threshold=perception_data.get("confidence_threshold", 0.5),
        ),
        decision=DecisionConfig(**decision_data) if decision_data else DecisionConfig(),
        execution=ExecutionConfig(**execution_data) if execution_data else ExecutionConfig(),
        training=TrainingConfig(**training_data) if training_data else TrainingConfig(),
        scene_objects=task_data.get("scene_objects", []),
        trash_bins=task_data.get("trash_bins", []),
    )


def _dict_to_dataclass(d, dc):
    """Helper: recursively convert dict to dataclass instance."""
    import dataclasses
    field_types = {f.name: f.type for f in dataclasses.fields(dc)}
    kwargs = {}
    for k, v in d.items():
        if k in field_types:
            kwargs[k] = v
    return dc(**kwargs)
