#!/usr/bin/env python3
"""Socket-to-ROS2 bridge for Task319 Isaac/Nav2 integration.

The Isaac process runs in a conda Python that cannot import ROS Jazzy rclpy.
This helper runs under the ROS 2 Python interpreter, publishes odom/scan/tf
from JSON state packets, subscribes to /cmd_vel, and returns the latest command
to Isaac. It intentionally does not implement navigation or path planning.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from typing import Any

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task319 ROS2 Nav2 socket bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31970)
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--lidar-frame", default="lidar")
    parser.add_argument("--cmd-timeout-s", type=float, default=0.5)
    return parser.parse_args()


class Task319SocketBridge(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("task319_socket_nav2_bridge")
        self.args = args
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.tf_pub = TransformBroadcaster(self)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.cmd_linear_x = 0.0
        self.cmd_angular_z = 0.0
        self.cmd_stamp = 0.0

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.cmd_linear_x = float(msg.linear.x)
        self.cmd_angular_z = float(msg.angular.z)
        self.cmd_stamp = time.time()

    def latest_cmd(self) -> dict[str, Any]:
        age = time.time() - self.cmd_stamp if self.cmd_stamp > 0.0 else float("inf")
        if age > float(self.args.cmd_timeout_s):
            linear_x = 0.0
            angular_z = 0.0
        else:
            linear_x = self.cmd_linear_x
            angular_z = self.cmd_angular_z
        return {
            "type": "cmd_vel",
            "linear_x": linear_x,
            "angular_z": angular_z,
            "age_s": age if math.isfinite(age) else None,
        }

    def publish_state(self, payload: dict[str, Any]) -> None:
        stamp = self.get_clock().now().to_msg()
        pose = [float(v) for v in payload.get("pose", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])]
        twist = [float(v) for v in payload.get("twist", [0.0] * 6)]
        x, y, z, qw, qx, qy, qz = pose[:7]

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.args.odom_frame
        odom.child_frame_id = self.args.base_frame
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation.w = qw
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.twist.twist.linear.x = twist[0] if len(twist) > 0 else 0.0
        odom.twist.twist.linear.y = twist[1] if len(twist) > 1 else 0.0
        odom.twist.twist.angular.z = twist[5] if len(twist) > 5 else 0.0
        self.odom_pub.publish(odom)

        ranges = payload.get("scan", [])
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = self.args.lidar_frame
        scan.angle_min = float(payload.get("scan_angle_min", -math.pi))
        scan.angle_max = float(payload.get("scan_angle_max", math.pi))
        scan.angle_increment = float(payload.get("scan_angle_increment", math.radians(1.0)))
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = 0.05
        scan.range_max = 8.0
        scan.ranges = [float(item) if math.isfinite(float(item)) else float("inf") for item in ranges]
        self.scan_pub.publish(scan)

        transforms = [
            self._transform(stamp, self.args.map_frame, self.args.odom_frame, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            self._transform(stamp, self.args.odom_frame, self.args.base_frame, (x, y, z), (qw, qx, qy, qz)),
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


def serve(args: argparse.Namespace, node: Task319SocketBridge) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, int(args.port)))
        server.listen(1)
        print(f"[ROS2_BRIDGE] listening on {args.host}:{args.port}", flush=True)
        conn, addr = server.accept()
        print(f"[ROS2_BRIDGE] connected by {addr}", flush=True)
        with conn, conn.makefile("rw", encoding="utf-8", newline="\n") as stream:
            for line in stream:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    stream.write(json.dumps({"type": "error", "error": repr(exc)}) + "\n")
                    stream.flush()
                    continue
                if payload.get("type") == "shutdown":
                    break
                if payload.get("type") == "state":
                    node.publish_state(payload)
                    rclpy.spin_once(node, timeout_sec=0.0)
                    stream.write(json.dumps(node.latest_cmd(), separators=(",", ":")) + "\n")
                    stream.flush()
                else:
                    stream.write(json.dumps({"type": "error", "error": "unknown packet type"}) + "\n")
                    stream.flush()


def main() -> None:
    args = parse_args()
    rclpy.init(args=None)
    node = Task319SocketBridge(args)
    try:
        serve(args, node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
