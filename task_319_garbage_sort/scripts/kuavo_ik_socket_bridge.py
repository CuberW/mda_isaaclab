#!/usr/bin/env python3
"""Socket-to-ROS1 bridge for Kuavo official arm IK services.

The Isaac process usually runs in a conda environment that cannot import the
ROS1 Kuavo message packages.  This helper runs inside a sourced Kuavo ROS1
shell and exposes a small newline-delimited JSON protocol to Isaac.
"""

from __future__ import annotations

import argparse
import json
import socket
import traceback
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task319 Kuavo IK socket bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31975)
    parser.add_argument("--service", default="/ik/two_arm_hand_pose_cmd_srv_muli_refer")
    parser.add_argument("--service-timeout-s", type=float, default=2.0)
    parser.add_argument("--node-name", default="task319_kuavo_ik_socket_bridge")
    return parser.parse_args()


def pose_payload_to_msg(pose_msg: Any, payload: dict[str, Any]) -> None:
    pose_msg.pos_xyz = [float(item) for item in payload.get("pos_xyz", [0.0, 0.0, 0.0])[:3]]
    quat = [float(item) for item in payload.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0])[:4]]
    pose_msg.quat_xyzw = quat
    pose_msg.elbow_pos_xyz = [float(item) for item in payload.get("elbow_pos_xyz", [0.0, 0.0, 0.0])[:3]]
    joint_angles = payload.get("joint_angles")
    if joint_angles is not None:
        pose_msg.joint_angles = [float(item) for item in joint_angles[:7]]


def solve_request(payload: dict[str, Any], args: argparse.Namespace, rospy: Any, two_arm_srv_type: Any, two_arm_cmd_type: Any) -> dict[str, Any]:
    cmd = two_arm_cmd_type()
    cmd.frame = int(payload.get("frame", 2))
    cmd.use_custom_ik_param = True
    q_arm = payload.get("q_arm")
    if isinstance(q_arm, list) and len(q_arm) >= 14:
        cmd.joint_angles_as_q0 = True
        cmd.hand_poses.left_pose.joint_angles = [float(item) for item in q_arm[:7]]
        cmd.hand_poses.right_pose.joint_angles = [float(item) for item in q_arm[7:14]]
    else:
        cmd.joint_angles_as_q0 = False

    pose_payload_to_msg(cmd.hand_poses.left_pose, payload.get("left_pose", {}))
    pose_payload_to_msg(cmd.hand_poses.right_pose, payload.get("right_pose", {}))

    params = payload.get("params", {}) or {}
    cmd.ik_param.major_optimality_tol = float(params.get("major_optimality_tol", 1e-3))
    cmd.ik_param.major_feasibility_tol = float(params.get("major_feasibility_tol", 1e-3))
    cmd.ik_param.minor_feasibility_tol = float(params.get("minor_feasibility_tol", 1e-3))
    cmd.ik_param.major_iterations_limit = float(params.get("major_iterations_limit", 100))
    cmd.ik_param.oritation_constraint_tol = float(params.get("oritation_constraint_tol", 0.5))
    cmd.ik_param.pos_constraint_tol = float(params.get("pos_constraint_tol", 0.006))
    cmd.ik_param.pos_cost_weight = float(params.get("pos_cost_weight", 80.0))
    cmd.ik_param.constraint_mode = int(params.get("constraint_mode", 2))
    if hasattr(cmd.ik_param, "elbow_cost_scale"):
        cmd.ik_param.elbow_cost_scale = float(params.get("elbow_cost_scale", 0.1))

    service_name = str(payload.get("service") or args.service)
    rospy.wait_for_service(service_name, timeout=float(args.service_timeout_s))
    proxy = rospy.ServiceProxy(service_name, two_arm_srv_type)
    response = proxy(cmd)
    hand_poses = response.hand_poses
    return {
        "type": "solve_result",
        "label": payload.get("label", ""),
        "success": bool(response.success),
        "with_torso": bool(response.with_torso),
        "q_arm": [float(item) for item in response.q_arm],
        "q_torso": [float(item) for item in response.q_torso],
        "time_cost_ms": float(response.time_cost),
        "error_reason": str(response.error_reason),
        "hand_poses": {
            "left_pose": {
                "pos_xyz": [float(item) for item in hand_poses.left_pose.pos_xyz],
                "quat_xyzw": [float(item) for item in hand_poses.left_pose.quat_xyzw],
                "joint_angles": [float(item) for item in hand_poses.left_pose.joint_angles],
            },
            "right_pose": {
                "pos_xyz": [float(item) for item in hand_poses.right_pose.pos_xyz],
                "quat_xyzw": [float(item) for item in hand_poses.right_pose.quat_xyzw],
                "joint_angles": [float(item) for item in hand_poses.right_pose.joint_angles],
            },
        },
    }


def serve(args: argparse.Namespace) -> None:
    import rospy
    from kuavo_msgs.msg import twoArmHandPoseCmd
    from kuavo_msgs.srv import twoArmHandPoseCmdSrv

    rospy.init_node(args.node_name, anonymous=True, disable_signals=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, int(args.port)))
        server.listen(1)
        print(f"[KUAVO_IK_BRIDGE] listening on {args.host}:{args.port}", flush=True)
        conn, addr = server.accept()
        print(f"[KUAVO_IK_BRIDGE] connected by {addr}", flush=True)
        with conn, conn.makefile("rw", encoding="utf-8", newline="\n") as stream:
            for line in stream:
                try:
                    payload = json.loads(line)
                    packet_type = payload.get("type")
                    if packet_type == "shutdown":
                        break
                    if packet_type == "health":
                        service_available = True
                        service_error = ""
                        try:
                            rospy.wait_for_service(args.service, timeout=0.1)
                        except Exception as exc:
                            service_available = False
                            service_error = repr(exc)
                        response = {
                            "type": "health",
                            "ok": True,
                            "service": args.service,
                            "service_available": service_available,
                            "service_error": service_error,
                        }
                    elif packet_type == "solve":
                        response = solve_request(payload, args, rospy, twoArmHandPoseCmdSrv, twoArmHandPoseCmd)
                    else:
                        response = {"type": "error", "error": f"unknown packet type: {packet_type}"}
                except Exception as exc:
                    response = {"type": "error", "error": repr(exc), "traceback": traceback.format_exc(limit=8)}
                stream.write(json.dumps(response, separators=(",", ":"), ensure_ascii=False) + "\n")
                stream.flush()


def main() -> None:
    serve(parse_args())


if __name__ == "__main__":
    main()
