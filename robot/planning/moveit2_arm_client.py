"""MoveIt2 readiness seam for Kuavo arm planning.

The implementation is deliberately a probe/adapter boundary.  It verifies the
ROS2 graph exposes a real MoveIt2 planning/execution stack.  It does not
generate local trajectories when MoveIt2 is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ros2_nav2_client import ROS2CLI, ROS2BridgeConfig, ROS2GraphSnapshot


@dataclass(frozen=True)
class MoveIt2Status:
    ready: bool
    ros2_ready: bool
    moveit_node_ready: bool
    joint_state_ready: bool
    trajectory_action_ready: bool
    planning_service_ready: bool
    missing: tuple[str, ...] = ()
    snapshot: ROS2GraphSnapshot | None = None
    message: str = ""


class MoveIt2ArmClient:
    """Probe seam for MoveIt2 arm planning and trajectory execution."""

    def __init__(self, cli: ROS2CLI | None = None, config: ROS2BridgeConfig | None = None):
        self.config = config or ROS2BridgeConfig()
        self.cli = cli or ROS2CLI(self.config)

    def ready_status(self) -> MoveIt2Status:
        snap = self.cli.snapshot()
        missing: list[str] = []
        if not snap.ros2_ready:
            missing.append("ros2 CLI / graph")
        moveit_node_ready = snap.has_any_node(("move_group", "moveit", "planning_scene"))
        if not moveit_node_ready:
            missing.append("MoveIt2 move_group/planning node")
        joint_state_ready = snap.has_topic("/joint_states")
        if not joint_state_ready:
            missing.append("/joint_states topic")
        trajectory_action_ready = snap.has_any_action((
            "follow_joint_trajectory",
            "execute_trajectory",
            "arm_controller",
        ))
        if not trajectory_action_ready:
            missing.append("joint trajectory execution action")
        planning_service_ready = (
            snap.has_service("/compute_ik")
            or snap.has_service("/check_state_validity")
            or snap.has_any_node(("move_group", "moveit"))
        )
        if not planning_service_ready:
            missing.append("MoveIt2 planning/IK service")
        return MoveIt2Status(
            ready=bool(
                snap.ros2_ready
                and moveit_node_ready
                and joint_state_ready
                and trajectory_action_ready
                and planning_service_ready
            ),
            ros2_ready=bool(snap.ros2_ready),
            moveit_node_ready=bool(moveit_node_ready),
            joint_state_ready=bool(joint_state_ready),
            trajectory_action_ready=bool(trajectory_action_ready),
            planning_service_ready=bool(planning_service_ready),
            missing=tuple(missing),
            snapshot=snap,
            message=snap.message,
        )

    def require_ready(self) -> None:
        status = self.ready_status()
        if status.ready:
            return
        raise RuntimeError(
            "MoveIt2 arm planning is not ready. "
            f"Missing: {', '.join(status.missing) or 'unknown'}. "
            f"Detail: {status.message}"
        )


__all__ = ["MoveIt2ArmClient", "MoveIt2Status"]
