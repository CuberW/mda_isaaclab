#!/usr/bin/env python3
"""Send one NavigateToPose goal to Nav2 and print a JSON result.

This helper must run in a ROS 2 environment with nav2_msgs installed. It is
called by the Isaac task process while the socket bridge keeps odom/scan/tf and
/cmd_vel flowing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from rclpy.node import Node
except Exception as exc:
    print(json.dumps({"success": False, "status": "ROS2_NAV2_IMPORT_ERROR", "error": repr(exc)}, ensure_ascii=False), flush=True)
    sys.exit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task319 Nav2 NavigateToPose client")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--yaw", type=float, required=True)
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--action-name", default="/navigate_to_pose")
    parser.add_argument("--timeout-s", type=float, default=75.0, help="Goal timeout in seconds. Use 0 or a negative value to wait indefinitely.")
    parser.add_argument("--goal-attempts", type=int, default=4)
    parser.add_argument("--retry-delay-s", type=float, default=2.0)
    parser.add_argument("--label", default="goal")
    return parser.parse_args()


def quat_xyzw_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return (0.0, 0.0, math.sin(half), math.cos(half))


class Nav2GoalClient(Node):
    def __init__(self, action_name: str) -> None:
        super().__init__("task319_nav2_goal_client")
        self.client = ActionClient(self, NavigateToPose, action_name)

    def send_goal(self, args: argparse.Namespace) -> tuple[bool, dict[str, object]]:
        timeout_s = float(args.timeout_s)
        deadline = None if timeout_s <= 0.0 else time.time() + timeout_s
        wait_for_server_timeout = 10.0 if deadline is None else min(10.0, timeout_s)
        if not self.client.wait_for_server(timeout_sec=wait_for_server_timeout):
            return False, {"success": False, "status": "ACTION_SERVER_UNAVAILABLE", "label": args.label}

        last_result: dict[str, object] = {"success": False, "status": "NOT_SENT", "label": args.label}
        attempts = max(1, int(args.goal_attempts))
        for attempt in range(1, attempts + 1):
            remaining = None if deadline is None else deadline - time.time()
            if remaining is not None and remaining <= 0.0:
                    last_result.update({"status": "SEND_TIMEOUT", "attempt": attempt})
                    return False, last_result
            success, result = self._send_goal_once(args, deadline, attempt)
            if success or result.get("status") not in {"REJECTED"} or attempt >= attempts:
                return success, result
            last_result = result
            if deadline is None:
                time.sleep(float(args.retry_delay_s))
            else:
                time.sleep(min(float(args.retry_delay_s), max(0.0, deadline - time.time())))
        return False, last_result

    def _send_goal_once(self, args: argparse.Namespace, deadline: float | None, attempt: int) -> tuple[bool, dict[str, object]]:
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = args.frame_id
        goal.pose.pose.position.x = float(args.x)
        goal.pose.pose.position.y = float(args.y)
        goal.pose.pose.position.z = 0.0
        qx, qy, qz, qw = quat_xyzw_from_yaw(args.yaw)
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        send_future = self.client.send_goal_async(goal)
        while rclpy.ok() and not send_future.done():
            if deadline is not None and time.time() > deadline:
                return False, {"success": False, "status": "SEND_TIMEOUT", "label": args.label}
            rclpy.spin_once(self, timeout_sec=0.05)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False, {"success": False, "status": "REJECTED", "label": args.label, "attempt": attempt}

        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            if deadline is not None and time.time() > deadline:
                cancel_future = goal_handle.cancel_goal_async()
                while rclpy.ok() and not cancel_future.done():
                    rclpy.spin_once(self, timeout_sec=0.05)
                return False, {"success": False, "status": "TIMEOUT", "label": args.label}
            rclpy.spin_once(self, timeout_sec=0.05)
        result = result_future.result()
        status_code = int(getattr(result, "status", -1))
        success = status_code == 4  # GoalStatus.STATUS_SUCCEEDED
        return success, {
            "success": success,
            "status": "SUCCEEDED" if success else f"STATUS_{status_code}",
            "status_code": status_code,
            "label": args.label,
            "attempt": attempt,
            "target_pose": [float(args.x), float(args.y), float(args.yaw)],
            "frame_id": args.frame_id,
        }


def main() -> int:
    args = parse_args()
    rclpy.init(args=None)
    node = Nav2GoalClient(args.action_name)
    try:
        success, result = node.send_goal(args)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if success else 2
    except Exception as exc:
        print(json.dumps({"success": False, "status": "EXCEPTION", "label": args.label, "error": repr(exc)}, ensure_ascii=False), flush=True)
        return 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
