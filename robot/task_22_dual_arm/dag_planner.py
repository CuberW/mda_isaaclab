"""DAGTaskPlanner — DAG-based task planning for dual-arm manipulation."""

from typing import List

from robot_common.decision import (
    TaskRouter, LLMTaskParser, TaskPlan, SubTask,
    TaskPhase, ArmRole,
)

from .coordinator import DualArmCoordinator


class DAGTaskPlanner:
    """DAG-based task planner for dual-arm manipulation.

    Generates directed acyclic graph of subtasks with arm assignments.
    Inspired by DAG-Plan architecture.
    """

    def __init__(self, coordinator: DualArmCoordinator):
        self.coordinator = coordinator

    def plan(self, instruction: str, objects: List[str]) -> TaskPlan:
        """Generate a DAG task plan for dual-arm operation.

        Args:
            instruction: Natural language instruction
            objects: Detected objects in scene

        Returns:
            TaskPlan with DAG subtasks
        """
        task_type = TaskRouter.classify(instruction)
        parser = LLMTaskParser()
        plan = parser.parse(instruction, task_type)

        # Detect arm roles from instruction
        if "左手" in instruction and "右手" in instruction:
            plan.arm_role = ArmRole.MASTER
        elif "双手" in instruction or "一起" in instruction or "协同" in instruction:
            plan.arm_role = ArmRole.SYNC
        else:
            plan.arm_role = ArmRole.SINGLE

        # Generate DAG subtasks based on instruction type. The API parser may
        # return generic "target_object" subtasks, so dual-arm keywords still
        # force concrete scene-aware DAG generation here.
        if (plan.task_type in ("carry", "dual_arm_vla")
                or any(kw in instruction for kw in ["搬运", "双手", "一起", "协同", "长杆"])
                or "carry" in instruction.lower()):
            plan.sub_tasks = self._generate_carry_dag(instruction, objects)
        elif "扶" in instruction or "hold" in instruction.lower():
            plan.sub_tasks = self._generate_hold_place_dag(instruction, objects)
        elif "抱" in instruction or "lift" in instruction.lower():
            plan.sub_tasks = self._generate_lift_dag(instruction, objects)

        return plan

    def _resolve_target(self, target_object: str, objects: List[str]) -> str:
        """Resolve parser-level target names to concrete scene object names."""
        if target_object in objects:
            return target_object
        if target_object in ("target_object", "object", "scene", "instruction"):
            return objects[0] if objects else target_object
        if target_object in ("container", "box"):
            return "box_obj" if "box_obj" in objects else target_object
        return target_object

    def _generate_carry_dag(self, instruction: str,
                            objects: List[str]) -> List[SubTask]:
        """Generate DAG for 'carry object together' task.

        DAG:
          left_approach ──┐
                          ├── dual_grasp ── sync_lift ── sync_carry ── sync_place
          right_approach ─┘

        """
        target = self._resolve_target(objects[0] if objects else "target_object", objects)
        return [
            SubTask(id="detect", phase=TaskPhase.SCAN, arm="single",
                    target_object="scene"),
            SubTask(id="left_approach", phase=TaskPhase.APPROACH, arm="left",
                    arm_role=ArmRole.MASTER, target_object=target),
            SubTask(id="right_approach", phase=TaskPhase.APPROACH, arm="right",
                    arm_role=ArmRole.SLAVE, target_object=target),
            SubTask(id="dual_grasp", phase=TaskPhase.GRASP, arm="both",
                    arm_role=ArmRole.SYNC, target_object=target,
                    depends_on=["left_approach", "right_approach"],
                    constraints={"sync_error": 0.02}),
            SubTask(id="sync_lift", phase=TaskPhase.LIFT, arm="both",
                    arm_role=ArmRole.SYNC, target_object=target,
                    depends_on=["dual_grasp"],
                    constraints={"max_tilt": 5.0}),
            SubTask(id="sync_carry", phase=TaskPhase.CARRY, arm="both",
                    arm_role=ArmRole.SYNC, target_object="target_region",
                    depends_on=["sync_lift"],
                    constraints={"max_tilt": 5.0, "sync_error": 0.02}),
            SubTask(id="sync_place", phase=TaskPhase.PLACE, arm="both",
                    arm_role=ArmRole.SYNC, target_object="target_region",
                    depends_on=["sync_carry"]),
        ]

    def _generate_hold_place_dag(self, instruction: str,
                                  objects: List[str]) -> List[SubTask]:
        """Generate DAG for 'left holds, right places' task.

        DAG:
          left_approach ── left_grasp ── left_hold (steady)
          right_approach ── right_grasp ── right_place
        """
        container = self._resolve_target(objects[1] if len(objects) > 1 else "container", objects)
        item = self._resolve_target(objects[2] if len(objects) > 2 else "item", objects)
        return [
            SubTask(id="detect", phase=TaskPhase.SCAN, arm="single",
                    target_object="scene"),
            SubTask(id="left_approach", phase=TaskPhase.APPROACH, arm="left",
                    arm_role=ArmRole.MASTER, target_object=container),
            SubTask(id="right_approach", phase=TaskPhase.APPROACH, arm="right",
                    arm_role=ArmRole.SLAVE, target_object=item),
            SubTask(id="left_grasp", phase=TaskPhase.GRASP, arm="left",
                    target_object=container, depends_on=["left_approach"]),
            SubTask(id="left_hold", phase=TaskPhase.CARRY, arm="left",
                    target_object=container, depends_on=["left_grasp"],
                    constraints={"hold_steady": True}),
            SubTask(id="right_grasp", phase=TaskPhase.GRASP, arm="right",
                    target_object=item, depends_on=["right_approach"]),
            SubTask(id="right_place", phase=TaskPhase.PLACE, arm="right",
                    target_object=container, depends_on=["right_grasp", "left_hold"]),
        ]

    def _generate_lift_dag(self, instruction: str,
                           objects: List[str]) -> List[SubTask]:
        """Generate DAG for 'coordinated lift' task."""
        target = objects[0] if objects else "box"
        return self._generate_carry_dag(instruction, objects)
