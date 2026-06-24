"""
Task state machine for structured execution control.

Manages task phases with transitions: approach → grasp → lift → carry → place
Supports state transitions, error recovery, and progress tracking.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable

from robot_common.decision import TaskPhase, TaskPlan, SubTask
from robot_common.infra.logging import logger


class TaskState(Enum):
    """Global task states."""
    IDLE = auto()
    INITIALIZING = auto()
    SCANNING = auto()
    DETECTING = auto()
    CLASSIFYING = auto()
    PLANNING = auto()
    APPROACHING = auto()
    GRASPING = auto()
    LIFTING = auto()
    CARRYING = auto()
    NAVIGATING = auto()
    PLACING = auto()
    RELEASING = auto()
    VERIFYING = auto()
    ERROR_RECOVERY = auto()
    COMPLETED = auto()
    FAILED = auto()


# State transition map
TRANSITIONS = {
    TaskState.IDLE: [TaskState.INITIALIZING],
    TaskState.INITIALIZING: [TaskState.SCANNING, TaskState.DETECTING],
    TaskState.SCANNING: [TaskState.DETECTING, TaskState.ERROR_RECOVERY],
    TaskState.DETECTING: [TaskState.CLASSIFYING, TaskState.PLANNING, TaskState.ERROR_RECOVERY],
    TaskState.CLASSIFYING: [TaskState.PLANNING, TaskState.ERROR_RECOVERY],
    TaskState.PLANNING: [TaskState.APPROACHING, TaskState.GRASPING, TaskState.ERROR_RECOVERY],
    TaskState.APPROACHING: [TaskState.GRASPING, TaskState.NAVIGATING, TaskState.ERROR_RECOVERY],
    TaskState.GRASPING: [TaskState.LIFTING, TaskState.ERROR_RECOVERY],
    TaskState.LIFTING: [TaskState.CARRYING, TaskState.NAVIGATING, TaskState.PLACING, TaskState.ERROR_RECOVERY],
    TaskState.CARRYING: [TaskState.PLACING, TaskState.NAVIGATING, TaskState.ERROR_RECOVERY],
    TaskState.NAVIGATING: [TaskState.APPROACHING, TaskState.PLACING, TaskState.RELEASING, TaskState.ERROR_RECOVERY],
    TaskState.PLACING: [TaskState.RELEASING, TaskState.ERROR_RECOVERY],
    TaskState.RELEASING: [TaskState.VERIFYING, TaskState.COMPLETED, TaskState.SCANNING, TaskState.ERROR_RECOVERY],
    TaskState.VERIFYING: [TaskState.COMPLETED, TaskState.DETECTING, TaskState.ERROR_RECOVERY],
    TaskState.ERROR_RECOVERY: [TaskState.IDLE, TaskState.COMPLETED, TaskState.FAILED],
}


class TaskStateMachine:
    """Manages task execution through a state machine."""

    def __init__(self, plan: TaskPlan):
        self.plan = plan
        self.state = TaskState.IDLE
        self.current_subtask_idx = 0
        self.completed_subtasks: list[str] = []
        self.errors: list[str] = []
        self.start_time: float = 0.0
        self._state_handlers: dict[TaskState, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default state handlers."""
        self._state_handlers = {
            TaskState.INITIALIZING: self._on_init,
            TaskState.SCANNING: self._on_scan,
            TaskState.DETECTING: self._on_detect,
            TaskState.CLASSIFYING: self._on_classify,
            TaskState.PLANNING: self._on_plan,
            TaskState.APPROACHING: self._on_approach,
            TaskState.GRASPING: self._on_grasp,
        }

    def register_handler(self, state: TaskState, handler: Callable):
        """Register a custom state handler."""
        self._state_handlers[state] = handler

    def transition(self, new_state: TaskState) -> bool:
        """Attempt to transition to a new state."""
        valid = TRANSITIONS.get(self.state, [])
        if new_state not in valid and new_state != TaskState.ERROR_RECOVERY:
            logger.warning(f"Invalid transition: {self.state} → {new_state}")
            # Allow everything → ERROR_RECOVERY
            if new_state != TaskState.ERROR_RECOVERY:
                return False
        old_state = self.state
        self.state = new_state
        logger.debug(f"State: {old_state.name} → {new_state.name}")
        return True

    def run_handler(self) -> bool:
        """Run the handler for the current state. Returns True if more work remains."""
        handler = self._state_handlers.get(self.state)
        if handler:
            try:
                return handler()
            except Exception as e:
                logger.error(f"Handler error in state {self.state}: {e}")
                self.errors.append(str(e))
                self.transition(TaskState.ERROR_RECOVERY)
                return False
        return True

    def _on_init(self) -> bool:
        logger.info(f"Task initialized: {self.plan.instruction}")
        return True

    def _on_scan(self) -> bool:
        return True

    def _on_detect(self) -> bool:
        return True

    def _on_classify(self) -> bool:
        return True

    def _on_plan(self) -> bool:
        return True

    def _on_approach(self) -> bool:
        return True

    def _on_grasp(self) -> bool:
        return True

    def mark_subtask_done(self, subtask_id: str):
        """Mark a subtask as completed."""
        if subtask_id not in self.completed_subtasks:
            self.completed_subtasks.append(subtask_id)

    def next_subtask(self) -> Optional[SubTask]:
        """Get the next pending subtask."""
        subtasks = self.plan.sub_tasks
        for st in subtasks:
            if st.id not in self.completed_subtasks:
                # Check dependencies
                deps_met = all(d in self.completed_subtasks for d in st.depends_on)
                if deps_met:
                    return st
        return None

    def is_complete(self) -> bool:
        """Check if all subtasks are done."""
        return all(st.id in self.completed_subtasks for st in self.plan.sub_tasks)

    def progress(self) -> float:
        """Return progress as fraction (0.0 - 1.0)."""
        if not self.plan.sub_tasks:
            return 1.0
        return len(self.completed_subtasks) / len(self.plan.sub_tasks)

    def summary(self) -> dict:
        return {
            "state": self.state.name,
            "progress": f"{self.progress()*100:.0f}%",
            "completed": len(self.completed_subtasks),
            "total": len(self.plan.sub_tasks),
            "errors": self.errors[-3:] if self.errors else [],
        }
