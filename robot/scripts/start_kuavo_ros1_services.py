"""Start the official Kuavo ROS1 services used by the 3.19 Kuavo bridge.

The official Kuavo repository is ROS1/Noetic while the Windows/WSL developer
environment is ROS2/Jazzy.  For this project we run the official ROS1 nodes
inside the vendor Docker image and let the Python task bridge call those ROS1
services from the container.
"""

from __future__ import annotations

import argparse
import subprocess
import time
import uuid


def run(cmd: list[str], *, timeout: float = 60.0, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def docker_bash(container: str, command: str, *, detached: bool = False, timeout: float = 60.0):
    cmd = ["docker", "exec"]
    if detached:
        cmd.append("-d")
    cmd += [container, "bash", "-lc", command]
    return run(cmd, timeout=timeout, check=not detached)


def docker_ros(container: str, docker_workspace: str, command: str, *, timeout: float = 10.0):
    setup = (
        f"cd {docker_workspace} && "
        "source /opt/ros/noetic/setup.bash && "
        "test -f devel/setup.bash && source devel/setup.bash; "
        "export ROS_MASTER_URI=http://localhost:11311; "
    )
    return run(["docker", "exec", container, "bash", "-lc", setup + command], timeout=timeout, check=False)


def wait_for(container: str, docker_workspace: str, command: str, label: str, *, timeout_s: float = 20.0) -> bool:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        proc = docker_ros(container, docker_workspace, command, timeout=5.0)
        last = (proc.stdout or proc.stderr or "").strip()
        if proc.returncode == 0:
            print(f"[OK] {label}")
            return True
        time.sleep(1.0)
    print(f"[MISS] {label}: {last}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", default="kuavo_official_ros")
    parser.add_argument("--docker-workspace", default="/root/kuavo_ws_linux")
    parser.add_argument("--robot-version", default="62")
    parser.add_argument("--skip-arm-trajectory", action="store_true")
    parser.add_argument("--skip-wheel-bridge", action="store_true")
    parser.add_argument("--no-container-restart", action="store_true")
    args = parser.parse_args()

    urdf_file = (
        f"{args.docker_workspace}/src/kuavo_assets/models/"
        f"biped_s{args.robot_version}/urdf/biped_s{args.robot_version}.urdf"
    )
    stand_joint_state = (
        "[0.349066, 0.0, 0.0, -0.523599, 0.0, 0.0, 0.0, "
        "0.349066, 0.0, 0.0, -0.523599, 0.0, 0.0, 0.0]"
    )
    run_id = str(uuid.uuid4())

    if not args.no_container_restart:
        run(["docker", "restart", args.container], timeout=45)
    else:
        docker_bash(
            args.container,
            "pkill -x arms_ik_node || true; "
            "pkill -x humanoid_plan_arm_trajectory_node || true; "
            "pkill -x rosout || true; "
            "pkill -x rosmaster || true; "
            "pkill -x roscore || true; "
            "sleep 2; true",
            timeout=10,
        )
    docker_bash(
        args.container,
        "source /opt/ros/noetic/setup.bash && export ROS_MASTER_URI=http://localhost:11311 && "
        "roscore >/tmp/kuavo_roscore.log 2>&1",
        detached=True,
    )
    if not wait_for(args.container, args.docker_workspace, "rosnode list >/dev/null", "ROS1 master", timeout_s=20):
        docker_bash(args.container, "tail -80 /tmp/kuavo_roscore.log || true", timeout=10)
        return 1

    common = (
        f"cd {args.docker_workspace} && source /opt/ros/noetic/setup.bash && "
        "source devel/setup.bash && export ROS_MASTER_URI=http://localhost:11311 && "
        f"export ROSLAUNCH_UUID={run_id} && "
        f"export ROBOT_VERSION={args.robot_version} && "
    )

    docker_bash(
        args.container,
        common
        + f"rosparam set /run_id '{run_id}' && "
        + f"test -f '{urdf_file}' && "
        + f"rosparam set /urdfFile '{urdf_file}' && "
        + f"rosparam set /standJointState '{stand_joint_state}' && "
        + "rosparam set /end_effector_joints_num 12 && "
        + f"rosparam set /robot_description \"$(cat '{urdf_file}')\" && "
        + f"rosparam set /humanoid_description \"$(cat '{urdf_file}')\" && "
        + "rosparam set /only_half_up_body true && "
        + "echo URDF_READY",
        timeout=30,
    )

    docker_bash(
        args.container,
        common
        + f"roslaunch motion_capture_ik ik_node.launch robot_version:={args.robot_version} "
        "visualize:=false >/tmp/kuavo_ik_node.log 2>&1",
        detached=True,
    )
    if not args.skip_arm_trajectory:
        docker_bash(
            args.container,
            common
            + f"rosparam set /urdfFile '{urdf_file}' && "
            + f"rosparam set /standJointState '{stand_joint_state}' && "
            + "rosparam set /end_effector_joints_num 12 && "
            + "rosrun humanoid_plan_arm_trajectory arm_trajectory_bezier_process.py "
            ">/tmp/kuavo_arm_traj_bezier.log 2>&1",
            detached=True,
        )
    if not args.skip_wheel_bridge:
        docker_bash(
            args.container,
            common
            + "python3 src/kuavo_wheel/scripts/wheel_bridge.py "
            "--kuavo-master http://localhost:11311 "
            "--wheel-master http://localhost:11311 "
            "--kuavo-slave-ip 127.0.0.1 "
            "--wheel-slave-ip 127.0.0.1 "
            ">/tmp/kuavo_wheel_bridge.log 2>&1",
            detached=True,
        )
        docker_bash(
            args.container,
            common
            + f"rosparam set /urdfFile '{urdf_file}' && "
            + f"rosparam set /standJointState '{stand_joint_state}' && "
            + "rosparam set /end_effector_joints_num 12 && "
            + "rosrun humanoid_plan_arm_trajectory humanoid_plan_arm_trajectory_node "
            ">/tmp/kuavo_arm_traj_node.log 2>&1",
            detached=True,
        )

    ik_ok = wait_for(
        args.container,
        args.docker_workspace,
        "rosservice list | grep -qx '/ik/two_arm_hand_pose_cmd_srv'",
        "official IK service /ik/two_arm_hand_pose_cmd_srv",
        timeout_s=25,
    )
    traj_ok = True
    if not args.skip_arm_trajectory:
        traj_ok = wait_for(
            args.container,
            args.docker_workspace,
            "rosservice list | grep -Eq '/execute_arm_action|/interrupt_arm_traj|/bezier/plan_arm_trajectory'",
            "official arm trajectory interface",
            timeout_s=25,
        )
    base_ok = True
    if not args.skip_wheel_bridge:
        base_ok = wait_for(
            args.container,
            args.docker_workspace,
            "rostopic info /move_base/base_cmd_vel | grep -q 'Subscribers:'",
            "official wheel bridge command topic /move_base/base_cmd_vel",
            timeout_s=12,
        )

    docker_bash(
        args.container,
        common
        + "echo NODES; rosnode list; echo SERVICES; "
        "rosservice list | grep -E '/ik/|plan_arm|traj|claw|mobile_manipulator|cmd_vel' || true; "
        "echo TOPICS; rostopic list | grep -E 'claw|cmd_vel|joint|base|traj|mobile|arm' || true; "
        "echo IK_LOG; tail -30 /tmp/kuavo_ik_node.log; "
        "echo ARM_TRAJ_BEZIER_LOG; tail -25 /tmp/kuavo_arm_traj_bezier.log 2>/dev/null || true; "
        "echo ARM_TRAJ_NODE_LOG; tail -25 /tmp/kuavo_arm_traj_node.log 2>/dev/null || true; "
        "echo WHEEL_BRIDGE_LOG; tail -30 /tmp/kuavo_wheel_bridge.log 2>/dev/null || true",
    )
    return 0 if ik_ok and traj_ok and base_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
