"""ROS2 full-stack adapter for Task 3.19 mobile manipulation.

The existing Task 3.19 perception code remains in ``pipeline.py``.  This file
owns only the full-stack readiness contract required before an official run:
ROS2 graph, Nav2, MoveIt2, and Kuavo official control must all be present.
"""

from __future__ import annotations

from dataclasses import dataclass

from planning import (
    KuavoOfficialROS2ControlClient,
    MobileManipulationStateMachine,
    MoveIt2ArmClient,
    Nav2Client,
    ROS2BridgeConfig,
    ROS2CLI,
)


@dataclass(frozen=True)
class ROS2MobileManipulationStatus:
    ready: bool
    missing: tuple[str, ...]
    details: dict


class ROS2MobileManipulationPipeline:
    """Full-stack checker and future ROS2 execution seam for 3.19."""

    def __init__(self, config: dict | None = None):
        raw = config or {}
        ros_cfg = raw.get("ros2", {})
        execution_cfg = raw.get("execution", {})
        official_ik = execution_cfg.get("official_ik", {})
        bridge_cfg = ROS2BridgeConfig(
            distro_setup=str(ros_cfg.get("distro_setup", "/opt/ros/jazzy/setup.bash")),
            workspace_setup=str(ros_cfg.get("workspace_setup", "~/ros2_ws/install/setup.bash")),
            wsl_distro=str(ros_cfg.get("wsl_distro", "UbuntuRobot")),
            use_wsl_on_windows=bool(ros_cfg.get("use_wsl_on_windows", True)),
            command_timeout_s=float(ros_cfg.get("command_timeout_s", 8.0)),
        )
        self.cli = ROS2CLI(bridge_cfg)
        self.nav2 = Nav2Client(self.cli, bridge_cfg)
        self.moveit2 = MoveIt2ArmClient(self.cli, bridge_cfg)
        self.kuavo = KuavoOfficialROS2ControlClient(
            self.cli,
            bridge_cfg,
            ik_service=str(official_ik.get("service_name", "/mobile_manipulator_ik_accessibility_check")),
        )
        self.state_machine = MobileManipulationStateMachine()

    def ready_status(self) -> ROS2MobileManipulationStatus:
        nav = self.nav2.ready_status()
        moveit = self.moveit2.ready_status()
        kuavo = self.kuavo.ready_status()
        missing = []
        if not nav.ready:
            missing.extend(f"Nav2: {item}" for item in nav.missing)
        if not moveit.ready:
            missing.extend(f"MoveIt2: {item}" for item in moveit.missing)
        if not kuavo.ready:
            missing.extend(f"Kuavo: {item}" for item in kuavo.missing)
        return ROS2MobileManipulationStatus(
            ready=bool(nav.ready and moveit.ready and kuavo.ready),
            missing=tuple(missing),
            details={
                "nav2": nav,
                "moveit2": moveit,
                "kuavo": kuavo,
            },
        )

    def require_ready(self) -> None:
        status = self.ready_status()
        if status.ready:
            return
        raise RuntimeError(
            "Task 3.19 --full requires ROS2 mobile manipulation backends. "
            f"Missing: {', '.join(status.missing) or 'unknown'}"
        )


__all__ = ["ROS2MobileManipulationPipeline", "ROS2MobileManipulationStatus"]
