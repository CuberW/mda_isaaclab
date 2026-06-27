#!/usr/bin/env python3
"""Launch a minimal Nav2 stack for the Task319 Isaac bridge.

This script runs in a sourced ROS 2 environment. It generates a small blank map,
a simple behavior tree, and a parameter file, then starts the standard Nav2
planner/controller/bt_navigator/map_server/lifecycle_manager processes.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task319 minimal Nav2 stack launcher")
    parser.add_argument("--output-dir", default="task_319_garbage_sort/output/nav2_runtime")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--lidar-frame", default="lidar")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--width-m", type=float, default=6.0)
    parser.add_argument("--height-m", type=float, default=6.0)
    parser.add_argument("--origin-x", type=float, default=-1.0)
    parser.add_argument("--origin-y", type=float, default=-3.0)
    parser.add_argument("--planner-tolerance", type=float, default=0.12)
    parser.add_argument("--xy-goal-tolerance", type=float, default=0.06)
    parser.add_argument("--yaw-goal-tolerance", type=float, default=0.12)
    parser.add_argument(
        "--lifecycle-start-delay-s",
        type=float,
        default=4.0,
        help="Delay lifecycle manager startup so the Isaac bridge has time to publish fresh tf/odom/scan data.",
    )
    return parser.parse_args()


def write_blank_map(out_dir: Path, args: argparse.Namespace) -> Path:
    width = max(10, int(round(float(args.width_m) / float(args.resolution))))
    height = max(10, int(round(float(args.height_m) / float(args.resolution))))
    pgm_path = out_dir / "task319_blank_map.pgm"
    yaml_path = out_dir / "task319_blank_map.yaml"
    pixels = bytearray([254] * width * height)

    def mark_rect(x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        ix0 = max(0, int(math.floor((x_min - float(args.origin_x)) / float(args.resolution))))
        ix1 = min(width - 1, int(math.ceil((x_max - float(args.origin_x)) / float(args.resolution))))
        iy0 = max(0, int(math.floor((y_min - float(args.origin_y)) / float(args.resolution))))
        iy1 = min(height - 1, int(math.ceil((y_max - float(args.origin_y)) / float(args.resolution))))
        for iy in range(iy0, iy1 + 1):
            row = height - 1 - iy
            for ix in range(ix0, ix1 + 1):
                pixels[row * width + ix] = 0

    # Known static scene geometry. Nav2 inflates these cells by the configured
    # robot radius, so the rectangles represent physical extents plus only a
    # small modeling margin.
    mark_rect(0.88, 2.72, -0.47, 0.47)  # sorting table
    mark_rect(2.75, 3.25, -0.97, -0.47)  # recycle bin
    mark_rect(2.75, 3.25, -0.49, 0.01)  # kitchen bin
    mark_rect(2.75, 3.25, -0.01, 0.49)  # hazardous bin
    mark_rect(2.75, 3.25, 0.47, 0.97)  # other bin

    with pgm_path.open("wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(bytes(pixels))
    yaml_path.write_text(
        "\n".join(
            [
                f"image: {pgm_path.name}",
                "mode: trinary",
                f"resolution: {float(args.resolution):.6f}",
                f"origin: [{float(args.origin_x):.6f}, {float(args.origin_y):.6f}, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def write_behavior_tree(out_dir: Path) -> Path:
    bt_path = out_dir / "task319_simple_nav_to_pose.xml"
    bt_path.write_text(
        """<root BTCPP_format="4" main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <PipelineSequence name="Task319NavigateWithReplanning">
      <RateController hz="1.0">
        <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
      </RateController>
      <FollowPath path="{path}" controller_id="FollowPath"/>
    </PipelineSequence>
  </BehaviorTree>
</root>
""",
        encoding="utf-8",
    )
    return bt_path


def write_nav2_params(out_dir: Path, map_yaml: Path, bt_xml: Path, args: argparse.Namespace) -> Path:
    params_path = out_dir / "task319_nav2_params.yaml"
    params_path.write_text(
        f"""map_server:
  ros__parameters:
    use_sim_time: false
    yaml_filename: "{map_yaml}"

planner_server:
  ros__parameters:
    use_sim_time: false
    expected_planner_frequency: 5.0
    costmap_update_timeout: 1.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: {float(args.planner_tolerance):.6f}
      use_astar: false
      allow_unknown: true

controller_server:
  ros__parameters:
    use_sim_time: false
    odom_topic: "/odom"
    controller_frequency: 20.0
    failure_tolerance: 0.3
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    progress_checker_plugins: ["progress_checker"]
    goal_checker_plugins: ["general_goal_checker"]
    controller_plugins: ["FollowPath"]
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.03
      movement_time_allowance: 45.0
    general_goal_checker:
      stateful: true
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: {float(args.xy_goal_tolerance):.6f}
      yaw_goal_tolerance: {float(args.yaw_goal_tolerance):.6f}
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      desired_linear_vel: 0.32
      lookahead_dist: 0.25
      min_lookahead_dist: 0.10
      max_lookahead_dist: 0.35
      lookahead_time: 0.8
      use_velocity_scaled_lookahead_dist: false
      min_approach_linear_velocity: 0.04
      approach_velocity_scaling_dist: 0.5
      use_collision_detection: true
      max_allowed_time_to_collision_up_to_carrot: 0.20
      use_rotate_to_heading: true
      rotate_to_heading_min_angle: 0.785
      rotate_to_heading_angular_vel: 0.55
      max_angular_accel: 0.75
      transform_tolerance: 0.3

local_costmap:
  local_costmap:
    ros__parameters:
      use_sim_time: false
      global_frame: "{args.odom_frame}"
      robot_base_frame: "{args.base_frame}"
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.28
      update_frequency: 10.0
      publish_frequency: 5.0
      transform_tolerance: 0.3
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: true
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: "{args.scan_topic}"
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 8.0
          raytrace_min_range: 0.05
          obstacle_max_range: 8.0
          obstacle_min_range: 0.05
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        inflation_radius: 0.35
        cost_scaling_factor: 4.0
      always_send_full_costmap: true

global_costmap:
  global_costmap:
    ros__parameters:
      use_sim_time: false
      global_frame: "{args.map_frame}"
      robot_base_frame: "{args.base_frame}"
      resolution: 0.05
      robot_radius: 0.28
      track_unknown_space: false
      update_frequency: 5.0
      publish_frequency: 2.0
      transform_tolerance: 0.3
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: true
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: "{args.scan_topic}"
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 8.0
          raytrace_min_range: 0.05
          obstacle_max_range: 8.0
          obstacle_min_range: 0.05
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        inflation_radius: 0.40
        cost_scaling_factor: 4.0
      always_send_full_costmap: true

bt_navigator:
  ros__parameters:
    use_sim_time: false
    global_frame: "{args.map_frame}"
    robot_base_frame: "{args.base_frame}"
    odom_topic: "/odom"
    bt_loop_duration: 10
    default_server_timeout: 20
    wait_for_service_timeout: 1000
    always_reload_bt_xml: true
    navigators: ["navigate_to_pose"]
    navigate_to_pose:
      plugin: "nav2_bt_navigator::NavigateToPoseNavigator"
    default_nav_to_pose_bt_xml: "{bt_xml}"

lifecycle_manager_navigation:
  ros__parameters:
    use_sim_time: false
    autostart: true
    bond_timeout: 4.0
    attempt_respawn_reconnection: false
    node_names:
      - map_server
      - planner_server
      - controller_server
      - bt_navigator
""",
        encoding="utf-8",
    )
    return params_path


def start_process(name: str, command: list[str], log_dir: Path) -> subprocess.Popen[str]:
    log_path = log_dir / f"{name}.log"
    log_file = log_path.open("w", encoding="utf-8")
    print(f"[NAV2_STACK] starting {name}: {' '.join(command)}", flush=True)
    return subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)


def terminate_processes(processes: list[tuple[str, subprocess.Popen[str]]]) -> None:
    for _, process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 5.0
    for _, process in reversed(processes):
        if process.poll() is not None:
            continue
        remaining = max(0.1, deadline - time.time())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()


def run_lifecycle_command(command: list[str], log_file, timeout_s: float = 12.0) -> subprocess.CompletedProcess[str] | None:
    log_file.write(f"[LIFECYCLE] run: {' '.join(command)}\n")
    log_file.flush()
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        log_file.write(f"[LIFECYCLE] timeout: {exc!r}\n")
        log_file.flush()
        return None
    log_file.write(completed.stdout)
    log_file.write(f"[LIFECYCLE] returncode={completed.returncode}\n")
    log_file.flush()
    return completed


def parse_lifecycle_state(output: str) -> str | None:
    text = str(output or "").strip().lower()
    for state in ("unconfigured", "inactive", "active", "finalized"):
        if text.startswith(state):
            return state
    return None


def get_lifecycle_state(node_name: str, log_file, timeout_s: float = 6.0) -> str | None:
    completed = run_lifecycle_command(["ros2", "lifecycle", "get", node_name], log_file, timeout_s=timeout_s)
    if completed is None or completed.returncode != 0:
        return None
    return parse_lifecycle_state(completed.stdout)


def wait_for_lifecycle_state(
    node_name: str,
    desired_states: set[str],
    log_file,
    timeout_s: float = 20.0,
    poll_s: float = 0.5,
) -> str | None:
    deadline = time.time() + max(0.1, timeout_s)
    while time.time() < deadline:
        state = get_lifecycle_state(node_name, log_file)
        if state in desired_states:
            return state
        time.sleep(max(0.05, poll_s))
    return get_lifecycle_state(node_name, log_file)


def set_lifecycle_transition(
    node_name: str,
    transition: str,
    desired_state: str,
    log_file,
    attempts: int = 5,
) -> bool:
    command = ["ros2", "lifecycle", "set", node_name, transition]
    for attempt in range(1, max(1, attempts) + 1):
        log_file.write(f"[LIFECYCLE] {node_name} {transition} attempt {attempt}\n")
        log_file.flush()
        completed = run_lifecycle_command(command, log_file)
        state = get_lifecycle_state(node_name, log_file)
        if state == desired_state:
            return True
        if completed is not None and completed.returncode == 0 and "Transitioning successful" in str(completed.stdout):
            state = wait_for_lifecycle_state(node_name, {desired_state}, log_file, timeout_s=8.0)
            if state == desired_state:
                return True
        time.sleep(0.8)
    return False


def bring_lifecycle_node_active(node_name: str, log_file) -> bool:
    state = wait_for_lifecycle_state(node_name, {"unconfigured", "inactive", "active"}, log_file, timeout_s=30.0)
    log_file.write(f"[LIFECYCLE] {node_name} initial_state={state}\n")
    log_file.flush()
    if state == "active":
        return True
    if state == "unconfigured":
        if not set_lifecycle_transition(node_name, "configure", "inactive", log_file):
            log_file.write(f"[LIFECYCLE] {node_name} configure did not reach inactive\n")
            log_file.flush()
            return False
        state = "inactive"
    if state == "inactive":
        if not set_lifecycle_transition(node_name, "activate", "active", log_file):
            log_file.write(f"[LIFECYCLE] {node_name} activate did not reach active\n")
            log_file.flush()
            return False
        return True
    log_file.write(f"[LIFECYCLE] {node_name} unsupported state={state}\n")
    log_file.flush()
    return False


def manual_lifecycle_bringup(log_dir: Path, start_delay_s: float) -> bool:
    nodes = ["/map_server", "/planner_server", "/controller_server", "/bt_navigator"]
    log_path = log_dir / "lifecycle_manual_bringup.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        delay = max(0.0, float(start_delay_s))
        log_file.write(f"[LIFECYCLE] manual bringup delay_s={delay:.3f}\n")
        log_file.flush()
        time.sleep(delay)
        for node_name in nodes:
            if not bring_lifecycle_node_active(node_name, log_file):
                log_file.write(f"[LIFECYCLE] failed to activate {node_name}\n")
                log_file.flush()
                return False
            time.sleep(0.35)
    return True


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    map_yaml = write_blank_map(out_dir, args)
    bt_xml = write_behavior_tree(out_dir)
    params_path = write_nav2_params(out_dir, map_yaml, bt_xml, args)

    processes: list[tuple[str, subprocess.Popen[str]]] = []
    commands = [
        ("map_server", ["ros2", "run", "nav2_map_server", "map_server", "--ros-args", "--params-file", str(params_path)]),
        ("planner_server", ["ros2", "run", "nav2_planner", "planner_server", "--ros-args", "--params-file", str(params_path)]),
        ("controller_server", ["ros2", "run", "nav2_controller", "controller_server", "--ros-args", "--params-file", str(params_path)]),
        ("bt_navigator", ["ros2", "run", "nav2_bt_navigator", "bt_navigator", "--ros-args", "--params-file", str(params_path)]),
    ]

    stop = False

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal stop
        print(f"[NAV2_STACK] received signal {signum}; terminating", flush=True)
        stop = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        for name, command in commands:
            processes.append((name, start_process(name, command, log_dir)))
            time.sleep(0.35)
        (out_dir / "nav2_stack_runtime.json").write_text(
            "{\n"
            f'  "params": "{params_path}",\n'
            f'  "map": "{map_yaml}",\n'
            f'  "behavior_tree": "{bt_xml}",\n'
            f'  "pid": {os.getpid()}\n'
            "}\n",
            encoding="utf-8",
        )
        print(f"[NAV2_STACK] runtime files: {out_dir}", flush=True)
        if not manual_lifecycle_bringup(log_dir, float(args.lifecycle_start_delay_s)):
            print("[NAV2_STACK] manual lifecycle bringup failed; terminating", flush=True)
            return 2
        print("[NAV2_STACK] manual lifecycle bringup complete", flush=True)
        while not stop:
            failed = [(name, process.returncode) for name, process in processes if process.poll() is not None]
            if failed:
                print(f"[NAV2_STACK] child process exited: {failed}", flush=True)
                return 2
            time.sleep(0.5)
        return 0
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    sys.exit(main())
