"""Explicit mobile-manipulation state machine for Task 3.19.

The state machine is intentionally independent from perception internals.  The
pipeline feeds it clean perception outputs and controller callbacks; the state
machine records what happened and refuses to hide missing planning/control
steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable


class MobileManipulationState(Enum):
    INIT_AND_PATROL = auto()
    DETECT_AND_CLASSIFY = auto()
    REACHABILITY_AND_WHOLE_BODY_PLAN = auto()
    COORDINATED_NAVIGATION = auto()
    GRASP_CARRY_RELEASE = auto()
    ERROR_RECOVERY = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class MobileManipulationEvent:
    state: MobileManipulationState
    ok: bool
    message: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class MobileManipulationRun:
    success: bool
    events: list[MobileManipulationEvent]
    failure: str = ""


class MobileManipulationStateMachine:
    """Run the six-state mobile-manipulation blueprint with injected actions."""

    def __init__(self):
        self.events: list[MobileManipulationEvent] = []

    def record(self, state: MobileManipulationState, ok: bool, message: str = "", **details) -> bool:
        self.events.append(MobileManipulationEvent(state, bool(ok), str(message), dict(details)))
        return bool(ok)

    def run(
        self,
        *,
        init_and_patrol: Callable[[], dict],
        detect_and_classify: Callable[[], dict],
        reachability_and_plan: Callable[[dict], dict],
        coordinated_navigation: Callable[[dict], dict],
        grasp_carry_release: Callable[[dict], dict],
        error_recovery: Callable[[str], dict] | None = None,
    ) -> MobileManipulationRun:
        self.events.clear()
        try:
            init = init_and_patrol()
            if not self.record(MobileManipulationState.INIT_AND_PATROL, init.get("ok", False), init.get("message", ""), **init):
                return self._fail("init_and_patrol_failed", error_recovery)

            detected = detect_and_classify()
            if not self.record(MobileManipulationState.DETECT_AND_CLASSIFY, detected.get("ok", False), detected.get("message", ""), **detected):
                return self._fail("detect_and_classify_failed", error_recovery)

            whole_body = reachability_and_plan(detected)
            if not self.record(MobileManipulationState.REACHABILITY_AND_WHOLE_BODY_PLAN, whole_body.get("ok", False), whole_body.get("message", ""), **whole_body):
                return self._fail("reachability_and_whole_body_plan_failed", error_recovery)

            nav = coordinated_navigation(whole_body)
            if not self.record(MobileManipulationState.COORDINATED_NAVIGATION, nav.get("ok", False), nav.get("message", ""), **nav):
                return self._fail("coordinated_navigation_failed", error_recovery)

            manipulation = grasp_carry_release({**whole_body, **nav})
            if not self.record(MobileManipulationState.GRASP_CARRY_RELEASE, manipulation.get("ok", False), manipulation.get("message", ""), **manipulation):
                return self._fail("grasp_carry_release_failed", error_recovery)

            self.record(MobileManipulationState.COMPLETED, True, "mobile manipulation complete")
            return MobileManipulationRun(True, list(self.events))
        except Exception as exc:
            return self._fail(f"exception: {exc}", error_recovery)

    def _fail(
        self,
        reason: str,
        error_recovery: Callable[[str], dict] | None,
    ) -> MobileManipulationRun:
        details = error_recovery(reason) if error_recovery is not None else {"ok": False, "message": reason}
        self.record(MobileManipulationState.ERROR_RECOVERY, bool(details.get("ok", False)), details.get("message", reason), **details)
        self.record(MobileManipulationState.FAILED, False, reason)
        return MobileManipulationRun(False, list(self.events), reason)


__all__ = [
    "MobileManipulationEvent",
    "MobileManipulationRun",
    "MobileManipulationState",
    "MobileManipulationStateMachine",
]
