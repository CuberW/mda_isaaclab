#!/usr/bin/env python3
"""Expose a ROS 2 Twist topic over a small TCP JSON protocol.

This is intended for bridging Nav2's `/cmd_vel` output to an external process
that runs in a different ROS environment, for example a ROS 1 Kuavo controller
publisher. The process deliberately does not know anything about Kuavo.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
from typing import Any

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 /cmd_vel TCP JSON server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31971)
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--cmd-timeout-s", type=float, default=0.5)
    parser.add_argument("--socket-timeout-s", type=float, default=1.0)
    return parser.parse_args()


class CmdVelSnapshot(Node):
    def __init__(self, topic: str, cmd_timeout_s: float) -> None:
        super().__init__("task319_ros2_cmd_vel_socket_server")
        self.topic = topic
        self.cmd_timeout_s = float(cmd_timeout_s)
        self._lock = threading.Lock()
        self._seq = 0
        self._stamp = 0.0
        self._linear = [0.0, 0.0, 0.0]
        self._angular = [0.0, 0.0, 0.0]
        self.create_subscription(Twist, topic, self._on_twist, 10)

    def _on_twist(self, msg: Twist) -> None:
        with self._lock:
            self._seq += 1
            self._stamp = time.time()
            self._linear = [float(msg.linear.x), float(msg.linear.y), float(msg.linear.z)]
            self._angular = [float(msg.angular.x), float(msg.angular.y), float(msg.angular.z)]

    def latest(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            age = now - self._stamp if self._stamp > 0.0 else float("inf")
            stale = age > self.cmd_timeout_s
            linear = list(self._linear)
            angular = list(self._angular)
            seq = self._seq
            stamp = self._stamp
        if stale:
            linear = [0.0, 0.0, 0.0]
            angular = [0.0, 0.0, 0.0]
        return {
            "type": "cmd_vel",
            "topic": self.topic,
            "seq": seq,
            "stale": stale,
            "age_s": age if math.isfinite(age) else None,
            "source_stamp_unix": stamp if stamp > 0.0 else None,
            "server_time_unix": now,
            "linear": {"x": linear[0], "y": linear[1], "z": linear[2]},
            "angular": {"x": angular[0], "y": angular[1], "z": angular[2]},
        }


def _write_json(stream: Any, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
    stream.flush()


def serve_client(conn: socket.socket, addr: tuple[str, int], node: CmdVelSnapshot) -> None:
    print(f"[ROS2_CMD_SERVER] connected by {addr}", flush=True)
    with conn, conn.makefile("rw", encoding="utf-8", newline="\n") as stream:
        _write_json(stream, {"type": "hello", "topic": node.topic})
        for line in stream:
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                _write_json(stream, {"type": "error", "error": repr(exc)})
                continue
            request_type = request.get("type")
            if request_type == "get":
                _write_json(stream, node.latest())
            elif request_type == "shutdown":
                _write_json(stream, {"type": "bye"})
                break
            else:
                _write_json(stream, {"type": "error", "error": f"unknown request type: {request_type!r}"})
    print(f"[ROS2_CMD_SERVER] disconnected {addr}", flush=True)


def serve(args: argparse.Namespace, node: CmdVelSnapshot) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(float(args.socket_timeout_s))
        server.bind((args.host, int(args.port)))
        server.listen(4)
        print(
            f"[ROS2_CMD_SERVER] listening on {args.host}:{args.port}, topic={args.topic}, "
            f"timeout={args.cmd_timeout_s}s",
            flush=True,
        )
        while rclpy.ok():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            serve_client(conn, addr, node)


def spin_node(node: CmdVelSnapshot) -> None:
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass


def main() -> None:
    args = parse_args()
    rclpy.init(args=None)
    node = CmdVelSnapshot(args.topic, args.cmd_timeout_s)
    spin_thread = threading.Thread(target=spin_node, args=(node,), daemon=True)
    spin_thread.start()
    try:
        serve(args, node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
