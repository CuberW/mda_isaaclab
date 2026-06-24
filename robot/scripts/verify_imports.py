#!/usr/bin/env python
"""
Verify that all modules import correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test importing all modules."""
    errors = []
    successes = []

    modules = [
        # Infrastructure
        ("robot_common.infra.config", "Config system"),
        ("robot_common.infra.logging", "Logging"),
        ("robot_common.infra.metrics", "Metrics"),
        ("robot_common.infra.task_lifecycle", "Task lifecycle helpers"),
        # Environment
        ("robot_common.env", "MuJoCo Env"),
        # Perception
        ("perception", "Perception seam"),
        ("perception.vision", "Vision adapters"),
        ("perception.grasp_estimators", "Grasp perception adapters"),
        ("perception.backends", "Perception backends"),
        ("robot_common.perception", "Perception Hub"),
        ("robot_common.perception.yolo_detector", "YOLO Detector"),
        ("robot_common.perception.grounding_dino", "GroundingDINO"),
        ("robot_common.perception.sam_segmentor", "SAM Segmentor"),
        ("robot_common.perception.classifier", "CLIP Classifier"),
        # Planning
        ("planning", "Planning seam"),
        ("planning.ik", "IK adapters"),
        ("planning.navigation", "Navigation adapters"),
        ("planning.trajectory", "Trajectory adapters"),
        ("planning.collision", "Collision adapters"),
        ("planning.constraints", "Constraint specs"),
        ("planning.backends", "Planning backends"),
        # Control
        ("control", "Control seam"),
        ("control.simulation", "Simulation adapters"),
        ("control.gripper", "Gripper adapters"),
        ("control.end_effectors", "End-effector specs"),
        ("control.grasp_quality", "Grasp quality checks"),
        ("control.wbc", "WBC adapters"),
        ("control.stance", "Stance adapters"),
        ("control.backends", "Control backends"),
        # Decision
        ("robot_common.decision", "Decision Hub"),
        ("robot_common.decision.state_machine", "State Machine"),
        # Execution
        ("robot_common.execution", "Execution Hub"),
        # Tasks
        ("task_319_garbage_sort", "Task 3.19"),
        ("task_22_dual_arm", "Task 2.2"),
    ]

    for module_name, description in modules:
        try:
            __import__(module_name)
            successes.append(f"  [OK] {description} ({module_name})")
        except Exception as e:
            errors.append(f"  [FAIL] {description} ({module_name}): {e}")

    print("\n" + "=" * 60)
    print("Module Import Verification")
    print("=" * 60)
    for s in successes:
        print(s)
    if errors:
        print("\n--- Errors ---")
        for e in errors:
            print(e)
    print(f"\nResults: {len(successes)}/{len(modules)} passed")

    return len(errors) == 0


if __name__ == "__main__":
    ok = test_imports()
    sys.exit(0 if ok else 1)
