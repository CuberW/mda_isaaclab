#!/usr/bin/env python3
"""End-to-end Nav2 smoke test with a tiny simulated differential base.

The test publishes odom/tf/scan, subscribes to /cmd_vel, sends one
NavigateToPose goal, and integrates the commanded velocity until Nav2 reports a
result. It is a test harness only; the navigation algorithm remains Nav2.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task319 Nav2 smoke test")
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--goal-x", type=float, default=0.60)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-yaw", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=40.0)
    parser.add_argument("--server-timeout-s", type=float, default=20.0)
    parser.add_argument("--goal-attempts", type=int, default=4)
    parser.add_argument("--retry-delay-s", type=float, default=2.0)
    parser.add_argument("--publish-hz", type=float, default=30.0)
    parser.add_argument("--action-name", default="/navigate_to_pose")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--lidar-frame", default="lidar")
    parser.add_argument("--goal-tolerance-m", type=float, default=0.12)
    return parser.parse_args()


def quat_xyzw_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def quat_wxyz_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class Task319SmokeRobot(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("task319_nav2_smoke_robot")
        self.args = args
        self.x = float(args.start_x)
        self.y = float(args.start_y)
        self.yaw = float(args.start_yaw)
        self.vx = 0.0
        self.wz = 0.0
        self.last_time = time.monotonic()
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.tf_pub = TransformBroadcaster(self)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        period = 1.0 / max(float(args.publish_hz), 1.0)
        self.create_timer(period, self._on_timer)
        self.client = ActionClient(self, NavigateToPose, args.action_name)

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.vx = float(msg.linear.x)
        self.wz = float(msg.angular.z)

    def _on_timer(self) -> None:
        now = time.monotonic()
        dt = max(0.0, min(0.1, now - self.last_time))
        self.last_time = now
        self.x += self.vx * math.cos(self.yaw) * dt
        self.y += self.vx * math.sin(self.yaw) * dt
        self.yaw = wrap_to_pi(self.yaw + self.wz * dt)
        self.publish_state()

    def publish_state(self) -> None:
        stamp = self.get_clock().now().to_msg()
        qx, qy, qz, qw = quat_xyzw_from_yaw(self.yaw)
        qw2, qx2, qy2, qz2 = quat_wxyz_from_yaw(self.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.args.odom_frame
        odom.child_frame_id = self.args.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.angular.z = self.wz
        self.odom_pub.publish(odom)

        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = self.args.lidar_frame
        scan.angle_min = -math.pi
        scan.angle_max = math.pi
        scan.angle_increment = math.radians(1.0)
        scan.time_increment = 0.0
        scan.scan_time = 1.0 / max(float(self.args.publish_hz), 1.0)
        scan.range_min = 0.05
        scan.range_max = 8.0
        scan.ranges = [float("inf")] * 361
        self.scan_pub.publish(scan)

        transforms = [
            self._transform(stamp, self.args.map_frame, self.args.odom_frame, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            self._transform(stamp, self.args.odom_frame, self.args.base_frame, (self.x, self.y, 0.0), (qw2, qx2, qy2, qz2)),
            self._transform(stamp, self.args.base_frame, self.args.lidar_frame, (0.0, 0.0, 0.32), (1.0, 0.0, 0.0, 0.0)),
        ]
        self.tf_pub.sendTransform(transforms)

    @staticmethod
    def _transform(stamp: Any, parent: str, child: str, pos: tuple[float, float, float], quat_wxyz: tuple[float, float, float, float]) -> TransformStamped:
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.translation.x = float(pos[0])
        tf.transform.translation.y = float(pos[1])
        tf.transform.translation.z = float(pos[2])
        tf.transform.rotation.w = float(quat_wxyz[0])
        tf.transform.rotation.x = float(quat_wxyz[1])
        tf.transform.rotation.y = float(quat_wxyz[2])
        tf.transform.rotation.z = float(quat_wxyz[3])
        return tf

    def send_goal(self) -> tuple[bool, dict[str, Any]]:
        self.publish_state()
        if not self.client.wait_for_server(timeout_sec=float(self.args.server_timeout_s)):
            return False, {"success": False, "status": "ACTION_SERVER_UNAVAILABLE"}

        deadline = time.time() + float(self.args.timeout_s)
        attempts = max(1, int(self.args.goal_attempts))
        last_result: dict[str, Any] = {"success": False, "status": "NOT_SENT"}
        for attempt in range(1, attempts + 1):
            success, result = self._send_goal_once(deadline, attempt)
            if success or result.get("status") != "REJECTED" or attempt >= attempts:
                return success, result
            last_result = result
            time.sleep(min(float(self.args.retry_delay_s), max(0.0, deadline - time.time())))
        return False, last_result

    def _send_goal_once(self, deadline: float, attempt: int) -> tuple[bool, dict[str, Any]]:
        self.publish_state()
        if time.time() > deadline:
            return False, {"success": False, "status": "SEND_TIMEOUT", "attempt": attempt}

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = self.args.map_frame
        goal.pose.pose.position.x = float(self.args.goal_x)
        goal.pose.pose.position.y = float(self.args.goal_y)
        qx, qy, qz, qw = quat_xyzw_from_yaw(float(self.args.goal_yaw))
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        send_future = self.client.send_goal_async(goal)
        while rclpy.ok() and not send_future.done():
            if time.time() > deadline:
                return False, {"success": False, "status": "SEND_TIMEOUT", "attempt": attempt}
            rclpy.spin_once(self, timeout_sec=0.05)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False, {"success": False, "status": "REJECTED", "attempt": attempt}

        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            if time.time() > deadline:
                cancel_future = goal_handle.cancel_goal_async()
                while rclpy.ok() and not cancel_future.done():
                    rclpy.spin_once(self, timeout_sec=0.05)
                return False, self._result("TIMEOUT", success=False, attempt=attempt)
            rclpy.spin_once(self, timeout_sec=0.05)

        result = result_future.result()
        status_code = int(getattr(result, "status", -1))
        nav_success = status_code == 4
        success = bool(nav_success)
        status = "SUCCEEDED" if success else f"STATUS_{status_code}"
        return success, self._result(status, success=success, status_code=status_code, attempt=attempt)

    def _result(self, status: str, *, success: bool, status_code: int | None = None, attempt: int | None = None) -> dict[str, Any]:
        return {
            "success": success,
            "status": status,
            "status_code": status_code,
            "attempt": attempt,
            "start_pose": [float(self.args.start_x), float(self.args.start_y), float(self.args.start_yaw)],
            "goal_pose": [float(self.args.goal_x), float(self.args.goal_y), float(self.args.goal_yaw)],
            "final_pose": [self.x, self.y, self.yaw],
            "final_distance_m": math.hypot(float(self.args.goal_x) - self.x, float(self.args.goal_y) - self.y),
            "last_cmd_vel": [self.vx, self.wz],
        }


def main() -> int:
    args = parse_args()
    rclpy.init(args=None)
    node = Task319SmokeRobot(args)
    try:
        success, result = node.send_goal()
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if success else 2
    except Exception as exc:
        print(json.dumps({"success": False, "status": "EXCEPTION", "error": repr(exc)}, ensure_ascii=False), flush=True)
        return 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
