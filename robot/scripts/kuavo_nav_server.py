#!/usr/bin/env python3
"""WSL-side ROS2 navigation server for Kuavo mobile base.

Polls .bridge/nav_goal.json for goals, reads odometry from
.bridge/odom.json, publishes /cmd_vel, writes velocity to
.bridge/cmd_vel.json, and writes .bridge/nav_result.json on completion.

Run inside WSL:
  cd /mnt/e/workspace/bigdata/project
  source /opt/ros/jazzy/setup.bash
  python3 scripts/kuavo_nav_server.py

Requires: ROS2 Jazzy with rclpy, geometry_msgs, nav_msgs, tf2_ros
"""

import json
import math
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class KuavoNavServer(Node):
    def __init__(self, bridge_dir: str = "/mnt/e/workspace/bigdata/project/.bridge"):
        super().__init__("kuavo_nav_server")

        self._bridge = Path(bridge_dir)
        self._bridge.mkdir(parents=True, exist_ok=True)
        self._goal_file = self._bridge / "nav_goal.json"
        self._odom_file = self._bridge / "odom.json"
        self._cmd_vel_file = self._bridge / "cmd_vel.json"
        self._result_file = self._bridge / "nav_result.json"

        # ROS2 publishers
        self._cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        # PID state
        self._goal: dict | None = None
        self._active = False
        self._odom: dict = {"x": 0.0, "y": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0, "vw": 0.0}
        self._goal_start_time = 0.0
        self._last_goal_write_ts = 0.0

        # PID gains (env-overridable)
        self._kp_lin = float(os.environ.get("KUAVO_NAV_KP_LIN", "1.2"))
        self._kp_ang = float(os.environ.get("KUAVO_NAV_KP_ANG", "2.5"))
        self._max_lin = float(os.environ.get("KUAVO_NAV_MAX_LIN", "0.35"))
        self._max_ang = float(os.environ.get("KUAVO_NAV_MAX_ANG", "0.8"))
        self._pos_tol = float(os.environ.get("KUAVO_NAV_POS_TOL", "0.05"))
        self._yaw_tol = float(os.environ.get("KUAVO_NAV_YAW_TOL", "0.08"))
        self._timeout = float(os.environ.get("KUAVO_NAV_TIMEOUT", "30.0"))

        self._integral_xy = 0.0
        self._integral_yaw = 0.0

        # 50 Hz control loop
        self._timer = self.create_timer(0.02, self._control_loop)

        self.get_logger().info(
            f"KuavoNavServer ready (bridge={bridge_dir}, "
            f"tol=±{self._pos_tol}m/±{self._yaw_tol}rad, timeout={self._timeout}s)"
        )

    def _control_loop(self) -> None:
        # Always read odometry
        self._odom = self._read_json(self._odom_file)

        # Publish /odom and /tf for ROS2 graph health
        now = time.perf_counter()
        self._publish_odom_tf()

        # Check for new goal
        goal = self._read_json(self._goal_file)
        if goal is not None:
            goal_ts = float(goal.get("timestamp", 0))
            is_cancel = bool(goal.get("cancel", False))
            if is_cancel and self._active:
                self.get_logger().info("Goal cancelled")
                self._active = False
                self._result(False, float("inf"), float("inf"), "cancelled")
            elif goal_ts > self._last_goal_write_ts + 0.01 and not is_cancel:
                self._goal = goal
                self._active = True
                self._integral_xy = 0.0
                self._integral_yaw = 0.0
                self._goal_start_time = now
                # Clear stale result
                try:
                    self._result_file.unlink()
                except FileNotFoundError:
                    pass
                self.get_logger().info(
                    f"Goal: ({goal['x']:.3f}, {goal['y']:.3f}, {goal['yaw']:.3f})"
                )
            self._last_goal_write_ts = goal_ts

        if not self._active or self._goal is None:
            self._publish_cmd_vel(0.0, 0.0, 0.0)
            return

        gx = float(self._goal["x"])
        gy = float(self._goal["y"])
        gyaw = float(self._goal["yaw"])

        # Check timeout
        if now - self._goal_start_time > self._timeout:
            self.get_logger().warn("Goal timed out")
            self._active = False
            dx = gx - self._odom["x"]
            dy = gy - self._odom["y"]
            final_err = math.hypot(dx, dy)
            final_yaw_err = _yaw_diff(gyaw, self._odom["yaw"])
            self._result(False, final_err, final_yaw_err, "timeout")
            self._publish_cmd_vel(0.0, 0.0, 0.0)
            return

        # Check arrival
        dx = gx - self._odom["x"]
        dy = gy - self._odom["y"]
        dist = math.hypot(dx, dy)
        yaw_err = _yaw_diff(gyaw, self._odom["yaw"])

        if dist <= self._pos_tol and abs(yaw_err) <= self._yaw_tol:
            self.get_logger().info(f"Goal reached: err=({dist:.3f}m, {yaw_err:.3f}rad)")
            self._active = False
            self._result(True, dist, abs(yaw_err), "reached")
            self._publish_cmd_vel(0.0, 0.0, 0.0)
            return

        # PID: face target first, then drive
        target_heading = math.atan2(dy, dx)
        heading_err = _yaw_diff(target_heading, self._odom["yaw"])

        # When close to target, align to final yaw instead
        if dist < 0.15:
            heading_err = yaw_err

        # Slow down when facing away or close to target
        vx = self._kp_lin * dist
        if abs(heading_err) > 0.5:
            vx *= 0.3  # turn in place first
        if dist < 0.3:
            vx *= max(0.15, dist / 0.3)

        vw = self._kp_ang * heading_err

        vx = max(-self._max_lin, min(self._max_lin, vx))
        vw = max(-self._max_ang, min(self._max_ang, vw))

        self._publish_cmd_vel(vx, 0.0, vw)

    def _publish_cmd_vel(self, vx: float, vy: float, vw: float) -> None:
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.angular.z = vw
        self._cmd_vel_pub.publish(twist)

        data = {"vx": vx, "vy": vy, "vw": vw, "timestamp": time.time()}
        self._write_json(self._cmd_vel_file, data)

    def _publish_odom_tf(self) -> None:
        stamp = self.get_clock().now().to_msg()
        x = float(self._odom.get("x", 0))
        y = float(self._odom.get("y", 0))
        yaw = float(self._odom.get("yaw", 0))
        vx = float(self._odom.get("vx", 0))
        vy = float(self._odom.get("vy", 0))
        vw = float(self._odom.get("vw", 0))

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = vw
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        odom.pose.pose.orientation.z = sy
        odom.pose.pose.orientation.w = cy
        self._odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_footprint"
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.rotation.z = sy
        tf.transform.rotation.w = cy
        self._tf_broadcaster.sendTransform(tf)

    def _result(self, success: bool, xy_err: float, yaw_err: float, msg: str) -> None:
        data = {
            "success": success,
            "final_error_xy": xy_err,
            "final_error_yaw": yaw_err,
            "message": msg,
        }
        self._write_json(self._result_file, data)
        self._publish_cmd_vel(0.0, 0.0, 0.0)

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def _yaw_diff(target: float, current: float) -> float:
    return math.atan2(math.sin(target - current), math.cos(target - current))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kuavo WSL ROS2 Navigation Server")
    parser.add_argument(
        "--bridge-dir",
        default="/mnt/e/workspace/bigdata/project/.bridge",
        help="Bridge directory shared with Windows",
    )
    args = parser.parse_args()

    rclpy.init()
    node = KuavoNavServer(bridge_dir=args.bridge_dir)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
