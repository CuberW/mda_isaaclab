#!/usr/bin/env python3
"""Publish ROS 2 Nav2 cmd_vel snapshots to Kuavo's official ROS 1 base topic.

The companion `ros2_cmd_vel_socket_server.py` runs in the ROS 2 environment and
exposes Nav2's `/cmd_vel` as JSON. This script runs in the Kuavo ROS 1
environment and republishes that command to `/cmd_vel` or `/cmd_vel_world`.
Use `--dry-run` to validate the TCP bridge on machines without ROS 1.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CmdVel:
    linear_x: float
    linear_y: float
    angular_z: float
    stale: bool
    seq: int
    age_s: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge ROS 2 Nav2 cmd_vel to Kuavo ROS 1 cmd_vel")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=31971)
    parser.add_argument("--kuavo-topic", default="/cmd_vel", choices=["/cmd_vel", "/cmd_vel_world"])
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true", help="Do not import rospy; print commands instead.")
    parser.add_argument("--max-messages", type=int, default=0, help="0 means run until interrupted.")
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--linear-scale", type=float, default=1.0)
    parser.add_argument("--angular-scale", type=float, default=1.0)
    parser.add_argument("--max-linear-x", type=float, default=0.45)
    parser.add_argument("--max-linear-y", type=float, default=0.25)
    parser.add_argument("--max-angular-z", type=float, default=0.8)
    parser.add_argument("--control-mode", type=int, default=-1, help="-1 skips mode switch; 2=BaseOnly, 3=BaseArm.")
    parser.add_argument("--zero-on-exit", action="store_true", default=True)
    parser.add_argument("--no-zero-on-exit", dest="zero_on_exit", action="store_false")
    return parser.parse_args()


def clamp(value: float, limit: float) -> float:
    limit = abs(float(limit))
    return max(-limit, min(limit, float(value)))


def decode_cmd(payload: dict[str, Any], args: argparse.Namespace) -> CmdVel:
    linear = payload.get("linear", {})
    angular = payload.get("angular", {})
    stale = bool(payload.get("stale", True))
    linear_x = clamp(float(linear.get("x", 0.0)) * float(args.linear_scale), args.max_linear_x)
    linear_y = clamp(float(linear.get("y", 0.0)) * float(args.linear_scale), args.max_linear_y)
    angular_z = clamp(float(angular.get("z", 0.0)) * float(args.angular_scale), args.max_angular_z)
    if stale:
        linear_x = 0.0
        linear_y = 0.0
        angular_z = 0.0
    age_s = payload.get("age_s")
    if age_s is not None:
        age_s = float(age_s)
    return CmdVel(
        linear_x=linear_x,
        linear_y=linear_y,
        angular_z=angular_z,
        stale=stale,
        seq=int(payload.get("seq", 0)),
        age_s=age_s,
    )


def request_cmd(stream: Any) -> dict[str, Any]:
    stream.write(json.dumps({"type": "get"}, separators=(",", ":")) + "\n")
    stream.flush()
    line = stream.readline()
    if not line:
        raise RuntimeError("cmd_vel socket server closed the connection")
    payload = json.loads(line)
    if payload.get("type") != "cmd_vel":
        raise RuntimeError(f"unexpected server response: {payload}")
    return payload


def connect(args: argparse.Namespace) -> tuple[socket.socket, Any]:
    sock = socket.create_connection((args.server_host, int(args.server_port)), timeout=float(args.connect_timeout_s))
    stream = sock.makefile("rw", encoding="utf-8", newline="\n")
    hello = json.loads(stream.readline())
    if hello.get("type") != "hello":
        raise RuntimeError(f"unexpected server hello: {hello}")
    return sock, stream


def import_ros1() -> tuple[Any, Any, Any]:
    import rospy  # type: ignore
    from geometry_msgs.msg import Twist  # type: ignore

    return rospy, Twist, None


def maybe_switch_control_mode(rospy: Any, control_mode: int) -> None:
    if control_mode < 0:
        return
    from kuavo_msgs.srv import changeTorsoCtrlMode, changeTorsoCtrlModeRequest  # type: ignore

    rospy.wait_for_service("/mobile_manipulator_mpc_control", timeout=5.0)
    client = rospy.ServiceProxy("/mobile_manipulator_mpc_control", changeTorsoCtrlMode)
    req = changeTorsoCtrlModeRequest()
    req.control_mode = int(control_mode)
    resp = client(req)
    if not getattr(resp, "result", False):
        raise RuntimeError(f"Kuavo control-mode switch failed: {getattr(resp, 'message', '')}")
    rospy.loginfo("Kuavo control mode switched to %s: %s", control_mode, getattr(resp, "message", ""))


def make_twist(Twist: Any, cmd: CmdVel) -> Any:
    msg = Twist()
    msg.linear.x = cmd.linear_x
    msg.linear.y = cmd.linear_y
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = cmd.angular_z
    return msg


def run_dry(args: argparse.Namespace, stream: Any) -> None:
    period = 1.0 / max(1e-6, float(args.rate_hz))
    count = 0
    while args.max_messages <= 0 or count < args.max_messages:
        cmd = decode_cmd(request_cmd(stream), args)
        if args.print_every <= 1 or count % int(args.print_every) == 0:
            print(
                "[DRY_RUN] "
                f"topic={args.kuavo_topic} seq={cmd.seq} stale={cmd.stale} age={cmd.age_s} "
                f"linear=({cmd.linear_x:.4f},{cmd.linear_y:.4f},0.0000) angular_z={cmd.angular_z:.4f}",
                flush=True,
            )
        count += 1
        time.sleep(period)


def run_ros1(args: argparse.Namespace, stream: Any) -> None:
    rospy, Twist, _ = import_ros1()
    rospy.init_node("task319_kuavo_cmd_vel_bridge", anonymous=True)
    maybe_switch_control_mode(rospy, int(args.control_mode))
    pub = rospy.Publisher(args.kuavo_topic, Twist, queue_size=10)
    rate = rospy.Rate(float(args.rate_hz))
    count = 0
    last_msg = make_twist(Twist, CmdVel(0.0, 0.0, 0.0, True, 0, None))
    try:
        while not rospy.is_shutdown() and (args.max_messages <= 0 or count < args.max_messages):
            cmd = decode_cmd(request_cmd(stream), args)
            last_msg = make_twist(Twist, cmd)
            pub.publish(last_msg)
            if args.print_every <= 1 or count % int(args.print_every) == 0:
                rospy.loginfo(
                    "Published %s seq=%s stale=%s vx=%.4f vy=%.4f wz=%.4f",
                    args.kuavo_topic,
                    cmd.seq,
                    cmd.stale,
                    cmd.linear_x,
                    cmd.linear_y,
                    cmd.angular_z,
                )
            count += 1
            rate.sleep()
    finally:
        if args.zero_on_exit:
            stop_msg = make_twist(Twist, CmdVel(0.0, 0.0, 0.0, True, 0, None))
            for _ in range(5):
                pub.publish(stop_msg)
                rospy.sleep(0.05)


def main() -> None:
    args = parse_args()
    try:
        sock, stream = connect(args)
    except Exception as exc:
        print(f"[ERROR] Could not connect to ROS2 cmd_vel server: {exc}", file=sys.stderr)
        raise SystemExit(2)
    with sock, stream:
        try:
            if args.dry_run:
                run_dry(args, stream)
            else:
                run_ros1(args, stream)
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            print(f"[ERROR] Kuavo cmd_vel bridge failed: {exc}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
