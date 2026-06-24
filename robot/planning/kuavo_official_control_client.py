"""ROS2-level Kuavo official control readiness adapter.

This complements the Python SDK bridge.  The old SDK bridge may import ROS1
``rospy`` packages from Kuavo's workspace; this adapter checks the ROS2 graph
required by the current mobile-manipulation full path.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ros2_nav2_client import ROS2CLI, ROS2BridgeConfig, ROS2GraphSnapshot


@dataclass(frozen=True)
class KuavoROS2ControlStatus:
    ready: bool
    ros2_ready: bool
    ik_service_ready: bool
    base_control_ready: bool
    arm_control_ready: bool
    gripper_control_ready: bool
    tf_ready: bool
    missing: tuple[str, ...] = ()
    snapshot: ROS2GraphSnapshot | None = None
    message: str = ""


class KuavoOfficialROS2ControlClient:
    """Probe seam for Kuavo official ROS2 IK/base/arm/gripper control."""

    def __init__(
        self,
        cli: ROS2CLI | None = None,
        config: ROS2BridgeConfig | None = None,
        ik_service: str = "/mobile_manipulator_ik_accessibility_check",
    ):
        self.config = config or ROS2BridgeConfig()
        self.cli = cli or ROS2CLI(self.config)
        self.ik_service = str(ik_service)

    def ready_status(self) -> KuavoROS2ControlStatus:
        snap = self.cli.snapshot()
        missing: list[str] = []
        if not snap.ros2_ready:
            missing.append("ros2 CLI / graph")
        tf_ready = snap.has_topic("/tf") and snap.has_topic("/tf_static")
        if not tf_ready:
            missing.append("/tf and /tf_static topics")
        ik_ready = snap.has_service(self.ik_service)
        if not ik_ready:
            missing.append(self.ik_service)
        base_ready = (
            snap.has_any_action(("navigate_to_pose", "follow_path"))
            or snap.has_any_node(("controller_server", "kuavo_base", "chassis"))
            or any("cmd_vel" in topic for topic in snap.topics)
        )
        if not base_ready:
            missing.append("Kuavo/Nav2 base control")
        arm_ready = snap.has_any_action(("follow_joint_trajectory", "execute_trajectory", "arm"))
        if not arm_ready:
            missing.append("Kuavo arm trajectory action")
        gripper_ready = (
            any("claw" in topic.lower() or "gripper" in topic.lower() for topic in snap.topics)
            or any("claw" in service.lower() or "gripper" in service.lower() for service in snap.services)
            or any("claw" in action.lower() or "gripper" in action.lower() for action in snap.actions)
        )
        if not gripper_ready:
            missing.append("Kuavo claw/gripper command interface")
        return KuavoROS2ControlStatus(
            ready=bool(snap.ros2_ready and tf_ready and ik_ready and base_ready and arm_ready and gripper_ready),
            ros2_ready=bool(snap.ros2_ready),
            ik_service_ready=bool(ik_ready),
            base_control_ready=bool(base_ready),
            arm_control_ready=bool(arm_ready),
            gripper_control_ready=bool(gripper_ready),
            tf_ready=bool(tf_ready),
            missing=tuple(missing),
            snapshot=snap,
            message=snap.message,
        )

    def require_ready(self) -> None:
        status = self.ready_status()
        if status.ready:
            return
        raise RuntimeError(
            "Kuavo official ROS2 control stack is not ready. "
            f"Missing: {', '.join(status.missing) or 'unknown'}. "
            f"Detail: {status.message}"
        )


__all__ = ["KuavoOfficialROS2ControlClient", "KuavoROS2ControlStatus"]
