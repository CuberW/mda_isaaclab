#!/usr/bin/env python
"""Probe the external Kuavo official ROS/SDK workspace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from planning import KuavoOfficialControlBridge, KuavoOfficialIKClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        default="E:/workspace/kuavo-ros-opensource",
        help="Path to kuavo-ros-opensource; can also be provided by KUAVO_ROS_WS",
    )
    parser.add_argument("--use-docker", action="store_true", help="Probe the official ROS1 stack inside Docker")
    parser.add_argument("--docker-container", default="kuavo_official_ros")
    parser.add_argument("--docker-workspace", default="/root/kuavo_ws_linux")
    parser.add_argument("--ik-service", default="/ik/two_arm_hand_pose_cmd_srv")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    print(f"Kuavo workspace: {workspace}")
    print(f"  exists: {workspace.exists()}")
    print(f"  sdk source: {(workspace / 'src' / 'kuavo_humanoid_sdk').exists()}")
    print(f"  installed setup: {(workspace / 'installed' / 'setup.bash').exists()}")
    print(f"  kuavo_msgs source: {(workspace / 'src' / 'kuavo_msgs').exists()}")

    ik = KuavoOfficialIKClient(
        enabled=True,
        timeout_s=0.5,
        service_name=str(args.ik_service),
        workspace=str(workspace),
        use_docker=bool(args.use_docker),
        docker_container=str(args.docker_container),
        docker_workspace=str(args.docker_workspace),
    )
    print("\nOfficial IK:")
    print(f"  available: {ik.available()}")
    if not ik.available():
        print(f"  detail: {ik.solve('right', [0, 0, 0, 0, 0, 0]).message}")

    bridge = KuavoOfficialControlBridge(
        enabled=True,
        workspace=str(workspace),
        use_docker=bool(args.use_docker),
        docker_container=str(args.docker_container),
        docker_workspace=str(args.docker_workspace),
    )
    status = bridge.ready_status()
    print("\nOfficial SDK control:")
    print(f"  ready: {status.ready}")
    print(f"  sdk_import: {status.sdk_import}")
    print(f"  ros_ready: {status.ros_ready}")
    print(f"  arm_trajectory_ready: {status.arm_trajectory_ready}")
    print(f"  base_control_ready: {status.base_control_ready}")
    print(f"  claw_ready: {status.claw_ready}")
    print(f"  detail: {status.message}")

    return 0 if ik.available() and status.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
