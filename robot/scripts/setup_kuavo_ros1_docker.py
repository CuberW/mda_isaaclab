"""Prepare the official Kuavo ROS1 Docker backend for Task 3.19.

This script does not implement robot control.  It only makes the official
kuavo-ros-opensource workspace usable inside the official Noetic Docker image:
start container, copy the Windows-mounted workspace to a Linux-native path,
normalize line endings, build message types, and verify ROS imports.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str], *, timeout: float = 120.0, check: bool = True) -> subprocess.CompletedProcess[str]:
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


def docker_bash(container: str, command: str, *, timeout: float = 120.0, check: bool = True):
    return run(["docker", "exec", container, "bash", "-lc", command], timeout=timeout, check=check)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=r"E:\workspace\kuavo-ros-opensource")
    parser.add_argument("--image", default="kuavo_opensource_mpc_wbc_img:0.6.1")
    parser.add_argument("--container", default="kuavo_official_ros")
    parser.add_argument("--docker-workspace", default="/root/kuavo_ws_linux")
    parser.add_argument("--robot-version", default="62")
    parser.add_argument("--refresh-copy", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    if not workspace.exists():
        raise SystemExit(f"Kuavo workspace does not exist: {workspace}")

    run(["docker", "rm", "-f", args.container], check=False)
    run([
        "docker",
        "run",
        "-d",
        "--name",
        args.container,
        "--net",
        "host",
        "--privileged",
        "-v",
        "/dev:/dev",
        "-v",
        f"{workspace.as_posix()}:/root/kuavo_ws",
        "-w",
        "/root/kuavo_ws",
        "-e",
        f"ROBOT_VERSION={args.robot_version}",
        args.image,
        "sleep",
        "infinity",
    ], timeout=60)

    copy_guard = f"test -f {args.docker_workspace}/.mda_prepared"
    if args.refresh_copy:
        copy_guard = "false"
    docker_bash(
        args.container,
        (
            f"if {copy_guard}; then echo 'Linux copy already prepared'; "
            "else "
            f"rm -rf {args.docker_workspace}; "
            f"cp -a /root/kuavo_ws {args.docker_workspace}; "
            f"find {args.docker_workspace} -type f "
            "\\( -name '*.sh' -o -name '*.bash' -o -name '*.zsh' -o -name 'env.sh' "
            "-o -name '_setup_util.py' -o -name '*.py' -o -name '*.launch' "
            "-o -name '*.xml' -o -name '*.cmake' -o -name 'CMakeLists.txt' "
            "-o -name 'package.xml' -o -name '*.srv' -o -name '*.msg' "
            "-o -name '*.action' -o -name '*.yaml' -o -name '*.yml' "
            "-o -name '*.txt' \\) "
            "-print0 | xargs -0 perl -pi -e 's/\\r$//'; "
            f"touch {args.docker_workspace}/.mda_prepared; "
            "fi"
        ),
        timeout=600,
    )

    docker_bash(
        args.container,
        (
            "curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc "
            "-o /tmp/ros.asc && apt-key add /tmp/ros.asc; "
            "find /etc/apt -type f -name '*.list' -print "
            "-exec sed -i 's#http://mirrors.ustc.edu.cn/ros/ubuntu#http://packages.ros.org/ros/ubuntu#g' {} ';'; "
            "apt-get update -o Acquire::AllowInsecureRepositories=true || true"
        ),
        timeout=240,
        check=False,
    )
    docker_bash(
        args.container,
        (
            f"mkdir -p {args.docker_workspace}/installed/lib "
            f"{args.docker_workspace}/src/kuavo-ros-control-lejulib/hardware_plant/lib/dexhand_sdk/protobuf_sdk/dist/linux/shared; "
            f"if test -f /root/kuavo_ws/tools/check_tool/dexhand/libstark.so; then "
            f"cp /root/kuavo_ws/tools/check_tool/dexhand/libstark.so {args.docker_workspace}/installed/lib/libstark.so; "
            f"cp /root/kuavo_ws/tools/check_tool/dexhand/libstark.so "
            f"{args.docker_workspace}/src/kuavo-ros-control-lejulib/hardware_plant/lib/dexhand_sdk/protobuf_sdk/dist/linux/shared/libstark.so; "
            "fi"
        ),
        timeout=60,
    )
    docker_bash(args.container, "python3 -m pip install transitions -i https://pypi.tuna.tsinghua.edu.cn/simple", timeout=180)
    docker_bash(
        args.container,
        (
            f"cd {args.docker_workspace} && "
            "source /opt/ros/noetic/setup.bash && "
            "catkin init && "
            "catkin config -DCMAKE_ASM_COMPILER=/usr/bin/as -DCMAKE_BUILD_TYPE=Release && "
            "catkin build kuavo_msgs humanoid_plan_arm_trajectory mobile_manipulator_controllers --no-status"
        ),
        timeout=1200,
        check=False,
    )
    docker_bash(
        args.container,
        (
            f"cd {args.docker_workspace} && "
            "python3 - <<'PY'\n"
            "from pathlib import Path\n"
            "p = Path('src/manipulation_nodes/motion_capture_ik/CMakeLists.txt')\n"
            "s = p.read_text()\n"
            "for dep in ['  noitom_hi5_hand_udp_python\\n', '  teach_pendant\\n', '  biped_s40_hand\\n', '  humanoid_wheel_interface\\n']:\n"
            "    s = s.replace(dep, '')\n"
            "s = s.replace('add_subdirectory(test)\\n\\n', '')\n"
            "start = s.find('# topic_logger')\n"
            "if start != -1:\n"
            "    s = s[:start]\n"
            "p.write_text(s.rstrip() + '\\n')\n"
            "p = Path('src/manipulation_nodes/motion_capture_ik/package.xml')\n"
            "lines = [line for line in p.read_text().splitlines() if not any(dep in line for dep in ['noitom_hi5_hand_udp_python', 'teach_pendant', 'biped_s40_hand', 'humanoid_wheel_interface'])]\n"
            "p.write_text('\\n'.join(lines) + '\\n')\n"
            "PY\n"
            "source /opt/ros/noetic/setup.bash && source devel/setup.bash && "
            "catkin build motion_capture_ik --no-deps --no-status"
        ),
        timeout=420,
    )
    docker_bash(
        args.container,
        (
            f"cd {args.docker_workspace} && "
            "source /opt/ros/noetic/setup.bash && source devel/setup.bash && "
            "python3 -c 'from kuavo_msgs.srv import accessIkSolve; print(\"KUAVO_MSGS_OK\")' && "
            "rossrv show kuavo_msgs/accessIkSolve"
        ),
        timeout=60,
    )
    print("\nPrepared. Next command:")
    print(
        "python run_all.py --task 319 --config configs\\task_319_kuavo_wheel.yaml "
        "--headless --full --demo-objects 1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
