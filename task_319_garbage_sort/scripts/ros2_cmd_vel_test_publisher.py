#!/usr/bin/env python3
"""Publish a bounded ROS 2 Twist command for motion-control validation."""

from __future__ import annotations

import argparse
import time

import rclpy
from geometry_msgs.msg import Twist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a finite ROS 2 cmd_vel command")
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--linear-x", type=float, default=0.20)
    parser.add_argument("--linear-y", type=float, default=0.0)
    parser.add_argument("--angular-z", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--startup-delay-s", type=float, default=0.5)
    parser.add_argument("--stop-count", type=int, default=10)
    return parser.parse_args()


def make_twist(linear_x: float, linear_y: float, angular_z: float) -> Twist:
    msg = Twist()
    msg.linear.x = float(linear_x)
    msg.linear.y = float(linear_y)
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = float(angular_z)
    return msg


def main() -> None:
    args = parse_args()
    rclpy.init(args=None)
    node = rclpy.create_node("task319_cmd_vel_test_publisher")
    pub = node.create_publisher(Twist, args.topic, 10)
    try:
        time.sleep(max(0.0, float(args.startup_delay_s)))
        rate_hz = max(1.0e-6, float(args.rate_hz))
        period = 1.0 / rate_hz
        end_time = time.time() + max(0.0, float(args.duration_s))
        cmd = make_twist(args.linear_x, args.linear_y, args.angular_z)
        count = 0
        while time.time() < end_time:
            pub.publish(cmd)
            rclpy.spin_once(node, timeout_sec=0.0)
            count += 1
            time.sleep(period)
        stop = make_twist(0.0, 0.0, 0.0)
        for _ in range(max(1, int(args.stop_count))):
            pub.publish(stop)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
        print(
            f"[ROS2_CMD_TEST_PUB] topic={args.topic} count={count} "
            f"linear=({args.linear_x:.4f},{args.linear_y:.4f},0.0000) angular_z={args.angular_z:.4f}",
            flush=True,
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
