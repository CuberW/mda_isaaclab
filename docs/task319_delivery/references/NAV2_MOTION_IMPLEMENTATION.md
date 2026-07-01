# Task 319 Nav2 Motion Implementation Notes

Last updated: 2026-06-25

This document is the running source of truth for the mobile-base/navigation part
of Task 319. Future changes to navigation, standpoints, Nav2 parameters, bridge
interfaces, or motion-only validation results should update this file.

## Current Status

- Motion-only validation remains available, but the active full-task path is
  `--mind_sort_demo`: VISS v28/Qwen perception, dynamic RGB-D standpoint
  planning, Nav2 navigation, right-gripper grasp attempt, bin-side navigation,
  and gripper-to-bin release.
- The formal visual-grasp path now defaults to dynamic RGB-D table-side station
  planning with `--dynamic_grasp_standpoint_nav`. It evaluates all valid visual
  targets, computes a continuous table-side robot pose from each target's RGB-D
  world center, then re-shoots RGB-D after Nav2 reaches the selected station.
  The older preset planner remains as fallback.
- Navigation uses ROS 2 Nav2, not a handwritten path follower.
- `--mind_sort_demo` uses real v28/VISS perception and Nav2 navigation. It
  first attempts physical right-gripper grasp/lift; by default, the demo also
  enables the 2 cm gripper-proximity carry continuation so the full sorting
  loop can proceed when the gripper reached the object but lift verification is
  still unreliable. Strict physical-grasp validation must pass
  `--no-mind_sort_gripper_proximity_assist`.
- Mind-sort recording should use the global high observer view by default:
  `observer_camera_pos=(2.15, 4.65, 3.10)`,
  `observer_camera_target=(2.05, 0.0, 0.62)`. Demo videos should run with
  `--mind_sort_max_objects 0` unless intentionally debugging a single item.
- After every mind-sort drop, the state machine must explicitly navigate back
  to the backed-off observation pose with `RETURN_TO_OBSERVE_BACKOFF`; the next
  cycle then re-runs perception from that pose. This keeps the delivered video
  aligned with the intended observe -> plan -> sort -> observe loop.
- Latest full mind-sort validation:
  - run directory:
    `task_319_garbage_sort/output/head_camera_grasp_records/20260625_150100`
  - video:
    `task_319_garbage_sort/output/head_camera_grasp_records/20260625_150100/external_grasp_demo.mp4`
  - completed scene keys:
    `trash_00`, `trash_01`, `trash_02`, `trash_04`, `trash_05`, `trash_06`,
    `trash_07`, `trash_08`
  - failed scene keys: none
  - drop policy: right gripper TCP moves over the selected bin opening before
    release; per-cycle metadata records `object_teleport_used=false`
  - final result: stopped when returning to the observation pose after the
    completed object cycles because the robot did not reach the target within
    the configured time limit
- Earlier mind-sort validation:
  - run directory:
    `task_319_garbage_sort/output/head_camera_grasp_records/20260624_001406`
  - selected object: `trash_05 / trash_potted_meat_can_0`
  - category/bin: `可回收物` -> recycle
  - table-facing standpoint:
    `[0.4849134088, 0.0074041486, 0.0]`
  - Nav2 result: all four demo goals returned `SUCCEEDED`
  - output video:
    `task_319_garbage_sort/output/head_camera_grasp_records/20260624_001406/external_grasp_demo.mp4`
- ROS 2 `/cmd_vel` can now be validated as a real Isaac robot motion input:
  `--ros_cmd_vel_demo` starts a ROS 2 `/cmd_vel` publisher, receives that topic
  through the existing Isaac ROS bridge, applies the command to the wheel-mode
  base, and checks the robot's measured pose displacement.
- `--waypoint_nav_demo` now validates movement between named ground waypoints.
  The route is an explicit comma-separated sequence of endpoints; table-left,
  table-right, bin-center, and home are not hard-coded mandatory intermediate
  points.
- The waypoint/open-loop navigation demos force wheel-mode execution:
  - `--nav_actuation_mode wheel`
  - `--wheel_drive_model mecanum45`
  - `--wheel_ground_coupling kinematic`
- Pure wheel-contact validation was attempted first and is not currently usable:
  a 16-pattern wheel sign sweep moved at most about `0.0025 m`. The wheel joints
  respond, but the current USD wheel/contact setup does not produce reliable
  planar base motion. The accepted validation path therefore keeps Nav2 and the
  wheel velocity interface, while using a kinematic wheel-ground coupling layer
  to integrate the commanded chassis motion in Isaac.
- The local Nav2 runtime is installed under:
  - `task_319_garbage_sort/output/nav2_user_install/root`
  - setup script: `task_319_garbage_sort/scripts/setup_nav2_user_install.bash`
- The Isaac motion-only sorting state machine can auto-start the bundled Nav2
  stack and complete one full category route:
  - table standpoint
  - stub pick/hold with the right arm raised into a carry pose
  - bin staging point
  - target bin point
  - stub drop
  - return via bin staging and a side corridor around the table
  - table standpoint
- The robot initial arm posture uses natural-down S62 arm joint targets. The
  motion-only pick stub closes the gripper, raises the right arm to a carry
  posture, and the drop stub opens the gripper then returns the arm to natural
  down.
- Verified successful run:
  - run directory: `task_319_garbage_sort/output/head_camera_grasp_records/20260622_220145`
  - video: `task_319_garbage_sort/output/head_camera_grasp_records/20260622_220145/external_grasp_demo.mp4`
  - result: `Motion-only Nav2 demo success=True`
- Latest waypoint-route validation:
  - route: `home,table_left,bin_center,table_right,home`
  - run directory: `task_319_garbage_sort/output/head_camera_grasp_records/20260622_225252`
  - video: `task_319_garbage_sort/output/head_camera_grasp_records/20260622_225252/external_grasp_demo.mp4`
  - result: `Waypoint Nav2 demo success=True`
- Important waypoint kinematic baseline:
  - command family: `--waypoint_nav_demo --wheel_drive_model mecanum45 --wheel_ground_coupling kinematic`
  - route: `home,table_left,bin_center,table_right,home`
  - this is the current reproducible Nav2 movement success marker and should be
    kept available while newer wheel mappings are tested.
- The high-friction per-wheel material binding experiment was reverted. The
  current baseline intentionally keeps the earlier yaw-wobble behavior while a
  separate official Isaac wheeled-controller test isolates the wheel geometry
  and joint-drive mapping.
- Latest official wheeled-controller diagnostic:
  - script: `task_319_garbage_sort/official_holonomic_wheel_test.py`
  - run directory: `task_319_garbage_sort/output/official_holonomic_wheel_tests/20260622_234208`
  - result: official `DifferentialController` produced wheel velocity commands
    and the robot yawed about `0.148 rad` with negligible translation, indicating
    wheel control reaches the articulation but the four-wheel geometry/sign
    mapping is not calibrated for straight travel.
- Latest ROS 2 `/cmd_vel` Isaac-control validation:
  - run directory: `task_319_garbage_sort/output/head_camera_grasp_records/20260623_000756`
  - video: `task_319_garbage_sort/output/head_camera_grasp_records/20260623_000756/external_grasp_demo.mp4`
  - publisher: `/cmd_vel`, `linear.x=0.20 m/s`, `angular.z=0.0 rad/s`, `111` messages
  - measured robot motion: initial pose `[0.53, 0.0, 0.0]`, final pose
    `[0.776834, -0.001488, -0.138321]`, translation `0.246839 m`
  - result: `ROS cmd_vel demo success=True`, reason
    `ROS2 /cmd_vel moved the robot in Isaac.`
- Latest stable wheel-adapter validation:
  - new command family:
    `--wheel_drive_model urdf_diagonal --wheel_ground_coupling kinematic_stable --wheel_velocity_scale 0.35`
  - S62 wheel geometry source: URDF wheel joint origins/yaws, with LF `+45 deg`,
    LB `+135 deg`, RF `-45 deg`, RB `-135 deg`, wheel radius `0.13035 m`,
    wheel half-span `0.23248871 m`
  - straight ROS 2 `/cmd_vel` run:
    `task_319_garbage_sort/output/head_camera_grasp_records/20260623_002628`
  - straight-run video:
    `task_319_garbage_sort/output/head_camera_grasp_records/20260623_002628/external_grasp_demo.mp4`
  - straight-run measured motion: translation `0.243330 m`, yaw drift
    `-0.0000507 rad`
  - Nav2 waypoint run:
    `task_319_garbage_sort/output/head_camera_grasp_records/20260623_002725`
  - Nav2 waypoint video:
    `task_319_garbage_sort/output/head_camera_grasp_records/20260623_002725/external_grasp_demo.mp4`
  - route: `home,table_left`
  - result: `Waypoint Nav2 demo success=True`, both transitions returned
    `SUCCEEDED`
- Dynamic table-side standpoint interface is now available through
  `--dynamic_standpoint_nav_demo`.
  - input: `--dynamic_target_world_xyz x,y,z`
  - allowed grasp sides: `front,left,right`
  - output: continuous table-side robot pose `(x, y, yaw)` sent to Nav2
  - diagnostics: right-arm geometric reachability plus optional dry-run IK
  - first verified run with measured final-error fields:
    `task_319_garbage_sort/output/head_camera_grasp_records/20260623_005732`
  - video:
    `task_319_garbage_sort/output/head_camera_grasp_records/20260623_005732/external_grasp_demo.mp4`
  - selected target/object coordinate: `[1.20, 0.00, 0.60]`
  - selected robot standpoint: `[0.53, 0.16, 0.0]`
  - Nav2 result: `SUCCEEDED`
  - measured final pose: `[0.540, 0.011, 0.311]`
  - measured error: `position_error_m=0.148929`,
    `yaw_error_rad=-0.311376`
  - limitation: the dynamic candidate generation and Nav2 action path work,
    but command-to-wheel tracking still leaves measurable pose error and needs
    further controller tuning.
- Preset pre-grasp standpoint interface:
  - enable with `--grasp_standpoint_nav`
  - default candidates:
    `table_front_left,table_front,table_front_right,table_left_near,table_left_far,table_right_near,table_right_far`
  - `table_left_near/far` and `table_right_near/far` are close grasp
    standpoints at `TABLE_Y_LIMITS +/- 0.42 m`; the older `table_left` and
    `table_right` remain side-corridor waypoints and are not part of the default
    grasp candidate list.
  - selection input: the selected object's current head RGB-D geometric center
  - selection rule: transform the object center into each candidate robot base
    frame and prefer stations where the object lands in the right-arm local
    window: forward `0.35-0.92 m`, lateral `-0.46-0.08 m`
  - debug outputs per cycle:
    `planned_grasp_standpoint.json`, `planned_standpoint_rgb.png`,
    `planned_standpoint_topdown.png`, `pre_grasp_standpoint_nav.json`, and
    `standpoint_reshoot/`
  - if Nav2 cannot reach the station, grasp execution is blocked.
- Dynamic pre-grasp standpoint interface:
  - enabled by default with `--dynamic_grasp_standpoint_nav`
  - observation/home pose defaults to `(0.18, 0.0, 0.0)` so the head RGB-D
    camera starts farther from the table
  - outputs:
    `dynamic_grasp_standpoint.json`, `dynamic_standpoint_rgb.png`,
    `dynamic_standpoint_topdown.png`
  - candidate sides: front, left, right
  - station yaw faces the selected object center
  - right-arm local target window remains forward `0.35-0.92 m`, lateral
    `-0.46-0.08 m`
  - `--standpoint_nav_only` validates the main visual path without grasping:
    v28/VISS RGB-D perception -> dynamic station computation -> Nav2
    `NavigateToPose` -> station re-shoot -> stop.
  - latest visual-to-Nav2 verification:
    `task_319_garbage_sort/output/head_camera_grasp_records/20260623_234801`
  - selected object: `black marker` / `trash_large_marker_0`
  - computed and sent Nav2 pose:
    `[0.4442109466, 0.2268338203, -0.2830957735]`
  - Nav2 result: `SUCCEEDED`
  - measured final pose:
    `[0.3610131443, 0.1041492596, 0.0372942650]`
  - measured error: `position_error_m=0.148234`,
    `yaw_error_rad=-0.320390`
  - interpretation: the RGB-D point-to-ground-station-to-Nav2 chain is wired,
    but the final stop accuracy is not yet good enough to treat this pose as
    grasp-ready without another local correction stage or tighter controller
    tolerance.

## Control Chain

```text
state-machine target pose (x, y, yaw)
  -> ROS 2 Nav2 NavigateToPose action
  -> Nav2 global planner + controller
  -> /cmd_vel
  -> socket bridge back to Isaac
  -> wheel inverse kinematics for the four S62 wheel joints
  -> configured wheel-ground coupling in Isaac
```

The planner/controller are Nav2 components. The Python code only launches Nav2,
publishes simulated robot state, sends target poses, and applies the returned
velocity command inside Isaac. In the current validated waypoint path, the
wheel-ground coupling is kinematic because the pure wheel-contact model is not
yet reliable.

For direct ROS command validation, the shorter chain is:

```text
ROS 2 geometry_msgs/Twist on /cmd_vel
  -> scripts/ros2_nav_socket_bridge.py
  -> ExternalRos2NavBridgeClient.exchange(robot)
  -> apply_wheel_velocity(...)
  -> four wheel joint velocity targets plus current kinematic wheel-ground coupling
  -> measured Isaac robot base pose changes
```

The accepted `20260623_000756` run proves that a ROS 2 velocity topic is not
only being received: it changes the simulated robot's measured base pose. The
current Isaac demo consumes `linear.x` and `angular.z`; `linear.y` is preserved
in the separate Kuavo ROS 1 bridge but is not applied by this differential-style
Isaac validation mode.

For dynamic table-side grasp standpoints, the goal chain is:

```text
target object world/table coordinate (x, y, z)
  -> continuous candidate standpoints on front/left/right table sides
  -> right-arm local geometry check, plus optional dry-run IK diagnostics
  -> selected robot ground pose (x, y, yaw)
  -> run_nav2_goal(...)
  -> Nav2 NavigateToPose
```

This first interface deliberately uses a manual target coordinate. The future
vision binding should replace `--dynamic_target_world_xyz` with the RGB-D
object center from the perception chain, without changing the navigation goal
interface.

The newer `kinematic_stable` coupling is separate from the `kinematic` baseline.
It keeps an internally integrated planar pose for the kinematic layer, so wheel
contact yaw drift from one physics step is not fed back into the next command
integration step. This is why the straight `/cmd_vel` test drops yaw drift from
about `0.138 rad` in the old `mecanum45 + kinematic` baseline to about
`0.00005 rad` in the `urdf_diagonal + kinematic_stable` test. The old baseline
remains intentionally available as the reproducible Nav2 success marker.

## Official Kuavo ROS/SDK Interface Findings

Official Kuavo wheel-arm interfaces exist and should be preferred over
handwritten four-wheel velocity mapping for future motion integration.

Sources checked:

- Kuavo Manual: `Kuavo 5-W 接口使用文档`
- Local official examples:
  `kuavo-ros-opensource/src/demo/test_kuavo_wheel_real`
- Local wheel-arm API doc:
  `kuavo-ros-opensource/src/demo/test_kuavo_wheel_real/轮臂控制API.md`
- Local SDK examples:
  `kuavo-ros-opensource/src/kuavo_humanoid_websocket_sdk/examples`

Relevant base interfaces:

| Interface | Type | Frame | Meaning |
| --- | --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/Twist` | robot body/base | velocity command, highest base-command priority |
| `/cmd_vel_world` | `geometry_msgs/Twist` | startup/world | velocity command in world axes |
| `/cmd_pose` | `geometry_msgs/Twist` | robot body/base | relative base pose command |
| `/cmd_pose_world` | `geometry_msgs/Twist` | startup/world/odom | absolute base pose command |
| `/lb_cmd_pose_reach_time` | `std_msgs/Float32` | n/a | estimated execution time after pose commands |
| `/mobile_manipulator_mpc_control` | `kuavo_msgs/changeTorsoCtrlMode` | n/a | wheel-arm MPC control-mode switch |
| `/mobile_manipulator/lb_mpc_control_mode` | `std_msgs/Int8` | n/a | current wheel-arm MPC mode feedback |
| `/mobile_manipulator_timed_single_cmd` | `kuavo_msgs/lbTimedPosCmd` | planner-dependent | timed command service; `planner_index=0` is base world pose |

Official wheel-arm control modes used by the examples:

| Mode | Meaning |
| --- | --- |
| `0` | `NoControl` |
| `1` | `ArmOnly` |
| `2` | `BaseOnly` |
| `3` | `BaseArm` |
| `4` | `ArmEeOnly` |

Notes:

- The official docs state that pose commands use MPC/time-optimal planning and
  publish estimated reach time on the corresponding reach-time topic.
- `cmd_vel`/`cmd_vel_world` should be continuously published; zero command or no
  command for about one second stops the base.
- The `.srv` inline comment for `changeTorsoCtrlMode` is not fully consistent
  with the Kuavo 5-W docs and examples. For wheel-arm mode switching, follow
  `lb_ctrl_api.py` and verify the result via `/mobile_manipulator/lb_mpc_control_mode`.
- The generic websocket SDK also exposes `walk`, `control_command_pose`, and
  `control_command_pose_world`, but for the current wheel-arm robot the Kuavo
  5-W ROS topics/services are the more direct integration point.
- Existing official Isaac integration expects `/sensors_data_raw` and
  `sim_start`. Local stubs already exist under
  `kuavo-ros-opensource/src/humanoid-control/humanoid_controllers/scripts/` for
  host-Isaac experiments, but they publish neutral state and are not a complete
  physical feedback bridge.

Preferred next integration order:

1. Add a small ROS1 official-interface smoke test that starts the wheel-arm
   controller, switches to `BaseOnly`, publishes `/cmd_pose_world`, waits for
   `/lb_cmd_pose_reach_time`, and records whether the official controller accepts
   the command.
2. If accepted, bridge the official controller's output/state into the current
   Isaac process instead of computing four wheel speeds directly.
3. For Nav2, bridge ROS2 `/cmd_vel` to official ROS1 `/cmd_vel` or
   `/cmd_vel_world`; keep Nav2 as the planner, but let the Kuavo controller own
   base kinematics.
4. Keep the current kinematic wheel-ground coupling only as a fallback
   simulation validation path until official controller feedback is connected.

Implemented bridge prototype:

```text
ROS 2 Nav2 /cmd_vel
  -> scripts/ros2_cmd_vel_socket_server.py
  -> TCP JSON snapshots
  -> scripts/kuavo_ros1_cmd_vel_socket_client.py
  -> Kuavo ROS 1 /cmd_vel or /cmd_vel_world
```

The bridge is split into two processes because the current Task319 workstation
has ROS 2 Jazzy available system-wide, while Kuavo's official wheel-arm stack is
ROS 1/Noetic-oriented. Keeping `rclpy` and `rospy` in separate processes avoids
mixed Python/ROS environment conflicts.

ROS 2 side command:

```bash
cd mda_isaaclab
/usr/bin/python3 task_319_garbage_sort/scripts/ros2_cmd_vel_socket_server.py \
  --host 127.0.0.1 \
  --port 31971 \
  --topic /cmd_vel \
  --cmd-timeout-s 1.0
```

Kuavo ROS 1 side command on a machine/container where `rospy` and `kuavo_msgs`
are available:

```bash
cd mda_isaaclab
source kuavo-ros-opensource/devel/setup.bash
python3 task_319_garbage_sort/scripts/kuavo_ros1_cmd_vel_socket_client.py \
  --server-host 127.0.0.1 \
  --server-port 31971 \
  --kuavo-topic /cmd_vel \
  --control-mode 2 \
  --rate-hz 50
```

Use `--kuavo-topic /cmd_vel_world` if the Nav2 command should be interpreted in
the Kuavo startup/world frame rather than the base frame. Use `--dry-run` to
test the TCP bridge without importing ROS 1.

Local dry-run verification on this workstation:

```text
published ROS 2 /cmd_vel: linear.x=0.18, linear.y=0.02, angular.z=0.11
dry-run Kuavo output:    linear=(0.1800,0.0200,0.0000), angular_z=0.1100
timeout output:          linear=(0.0000,0.0000,0.0000), angular_z=0.0000
```

Local ROS 1 topic verification:

```text
ROS 2 test publisher sent: linear.x=0.18, linear.y=0.02, angular.z=0.11
ROS 1 /cmd_vel subscriber received: vx=0.1800, vy=0.0200, wz=0.1100
```

This used a local Noetic `rosmaster` from a historical sandbox plus the
generated Kuavo message bindings in `kuavo-ros-opensource/devel`. It verifies
the ROS 2 -> TCP -> ROS 1 `/cmd_vel` publishing chain. It does not verify the
actual Kuavo wheel-arm MPC controller, because `/mobile_manipulator_mpc_control`
and the real wheel-arm controller nodes are not running in this workstation
environment. The same client is ready to run inside the Kuavo ROS 1 controller
environment with `--control-mode 2` or `--control-mode 3`.

## Main Files

| File | Role |
| --- | --- |
| `visual_grasp_record_demo.py` | Isaac scene, motion-only sorting state machine, Nav2 bridge client, cmd_vel application |
| `task319_grasp_sort_sm.py` | Thin entrypoint that runs `visual_grasp_record_demo.py` |
| `scripts/setup_nav2_user_install.bash` | Sources ROS Jazzy and user-local Nav2 packages |
| `scripts/ros2_nav2_stack_launcher.py` | Generates map/BT/params and starts Nav2 processes |
| `scripts/ros2_nav_socket_bridge.py` | Bridges Isaac JSON state to ROS topics and returns `/cmd_vel` |
| `scripts/ros2_nav_goal_client.py` | Sends one `NavigateToPose` goal and prints a JSON result |
| `scripts/ros2_nav2_smoke_test.py` | ROS-only Nav2 smoke test without Isaac |
| `scripts/ros2_cmd_vel_test_publisher.py` | Finite-duration ROS 2 `/cmd_vel` publisher for direct Isaac motion validation |
| `scripts/ros2_cmd_vel_socket_server.py` | Exposes ROS 2 `/cmd_vel` over TCP JSON for external controllers |
| `scripts/kuavo_ros1_cmd_vel_socket_client.py` | Publishes TCP cmd_vel snapshots to Kuavo ROS 1 `/cmd_vel` or `/cmd_vel_world` |

## State Machine Targets

Defined in `visual_grasp_record_demo.py`.

| Symbol | Current Value | Meaning |
| --- | --- | --- |
| `TABLE_STANDPOINTS["front_center"]` | `(0.53, 0.0, 0.0)` | robot table-side standpoint |
| `compute_grasp_standpoint_candidates(...)` | continuous | converts target world/table coordinate into front/left/right candidate standpoints |
| `BIN_STAGING_POSE` | `(3.75, 0.0, pi)` | shared staging point before selecting a bin |
| `nav2_bin_pose_for_category(category)` | `(3.75, bin_y, pi)` | category-specific bin approach point |
| `SIDE_CORRIDOR_Y` | `1.35` | side corridor used when returning from positive-y bins |
| `TABLE_RETURN_APPROACH_BACKOFF_M` | `0.45` | x backoff before final table return |

Named waypoint demo registry:

| Waypoint | Pose | Meaning |
| --- | --- | --- |
| `home` | `(0.53, 0.0, 0.0)` | initial/front table pose |
| `table_front` | `(0.53, 0.0, 0.0)` | front table grasp-standpoint candidate |
| `table_left` | `(1.80, -1.35, pi/2)` | left-side table grasp-standpoint candidate |
| `table_right` | `(1.80, 1.35, -pi/2)` | right-side table grasp-standpoint candidate |
| `bin_center` | `(3.75, 0.0, pi)` | shared bin-side staging pose |
| `bin_recycle` | `(3.75, -0.72, pi)` | recycle-bin standpoint |
| `bin_kitchen` | `(3.75, -0.24, pi)` | kitchen-bin standpoint |
| `bin_hazard` | `(3.75, 0.24, pi)` | hazard-bin standpoint |
| `bin_other` | `(3.75, 0.72, pi)` | other-bin standpoint |

Current category-to-bin names:

| Category | Bin Name |
| --- | --- |
| `可回收物` | `recycle` |
| `厨余垃圾` | `kitchen` |
| `有害垃圾` | `hazard` |
| `其他垃圾` | `other` |

Motion-only route order:

```text
NAV_TO_TABLE_STANDPOINT
PICK_STUB_OR_HOLD_OBJECT
NAV_TO_BIN_STAGING
SELECT_BIN_BY_CATEGORY
NAV_TO_BIN
DROP
RETURN_TO_BIN_STAGING
RETURN_TO_SIDE_CORRIDOR
RETURN_HOME_APPROACH
RETURN_HOME_ALIGN
RETURN_HOME
DONE
```

## Interfaces And I/O

### 1. State Machine To Nav2 Goal

Function:

```python
run_nav2_goal(..., state_name, label, target_pose)
```

Input:

```python
target_pose = (x_m, y_m, yaw_rad)
label = "human-readable-goal-label"
```

Output:

```python
{
  "label": label,
  "state": state_name,
  "success": true_or_false,
  "status": "SUCCEEDED" | "TIMEOUT" | "STATUS_<code>" | ...,
  "target_pose": [x, y, yaw],
  "final_pose": [x, y, yaw],
  "position_error_m": measured_xy_error,
  "yaw_error_rad": measured_yaw_error,
  "nav2_result": {...}
}
```

Results are written into:

```text
<run_dir>/motion_only_nav2/motion_only_nav2_metadata.json
<run_dir>/motion_only_nav2/fsm_trace.json
<run_dir>/waypoint_nav/waypoint_nav_metadata.json
<run_dir>/waypoint_nav/fsm_trace.json
<run_dir>/dynamic_standpoint_nav/dynamic_standpoint_nav_metadata.json
<run_dir>/dynamic_standpoint_nav/dynamic_standpoint_candidates.json
<run_dir>/dynamic_standpoint_nav/fsm_trace.json
```

### 2. Nav2 Action Goal Client

Script:

```text
task_319_garbage_sort/scripts/ros2_nav_goal_client.py
```

CLI input:

```bash
--x <meters>
--y <meters>
--yaw <radians>
--frame-id map
--action-name /navigate_to_pose
--timeout-s <seconds>
--label <label>
```

ROS output:

```text
NavigateToPose.Goal on /navigate_to_pose
```

Process stdout:

```json
{
  "success": true,
  "status": "SUCCEEDED",
  "status_code": 4,
  "label": "bin_other_其他垃圾",
  "attempt": 1,
  "target_pose": [3.75, 0.72, 3.141593],
  "frame_id": "map"
}
```

### 3. Isaac To ROS Socket Bridge

Script:

```text
task_319_garbage_sort/scripts/ros2_nav_socket_bridge.py
```

Socket:

```text
host: 127.0.0.1
port: 31970
```

Isaac sends one JSON packet per simulation step:

```json
{
  "type": "state",
  "stamp": 123.0,
  "pose": [x, y, z, qw, qx, qy, qz],
  "twist": [vx, vy, vz, wx, wy, wz],
  "scan": [null_or_inf_or_range_values],
  "scan_angle_min": -3.141592653589793,
  "scan_angle_max": 3.141592653589793,
  "scan_angle_increment": 0.017453292519943295
}
```

Bridge publishes to ROS:

| Topic | Type | Source |
| --- | --- | --- |
| `/odom` | `nav_msgs/Odometry` | Isaac root pose/twist |
| `/scan` | `sensor_msgs/LaserScan` | synthetic or empty scan |
| `/tf` | `map->odom`, `odom->base_link`, `base_link->lidar` | bridge-generated transforms |

Bridge subscribes to ROS:

| Topic | Type | Meaning |
| --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/Twist` | Nav2 velocity command |

Bridge response to Isaac:

```json
{
  "type": "cmd_vel",
  "linear_x": 0.35,
  "angular_z": 0.2,
  "age_s": 0.01
}
```

### 4. Isaac Velocity Application

Function:

```python
apply_wheel_velocity(robot, wheel_ids, vx, wz, dt=None)
```

Input:

```python
vx = cmd_vel.linear.x
wz = cmd_vel.angular.z
```

Current default mode:

```text
--nav_actuation_mode wheel
--wheel_drive_model mecanum45
--wheel_ground_coupling kinematic
```

Behavior:

- Computes wheel joint velocity targets for the four S62 wheel joints.
- In `wheel + kinematic` mode, integrates the planar base pose from Nav2
  `cmd_vel` and writes the root pose/velocity in Isaac:
  - world `vx = body_vx * cos(yaw)`
  - world `vy = body_vx * sin(yaw)`
  - world `wz = cmd_vel.angular.z`
- In `wheel + contact` mode, only wheel joint velocity targets are written and
  Isaac contact physics must move the base. This mode is available for future
  tuning but did not pass the current validation.
- In legacy `root_velocity` mode, the root velocity is written directly without
  forcing the navigation demo into wheel mode.

Reason for current default:

- Nav2 path planning/control is already used.
- The S62 wheel/contact model in Isaac is not yet stable enough to rely on pure
  wheel physics for validation.
- `wheel + kinematic` keeps the interface wheel-driven while avoiding an
  unvalidated tire-contact model.

To test pure wheel actuation later:

```bash
--nav_actuation_mode wheel --wheel_ground_coupling contact
```

## Nav2 Stack Configuration

Generated by:

```text
task_319_garbage_sort/scripts/ros2_nav2_stack_launcher.py
```

Started processes:

```text
nav2_map_server / map_server
nav2_planner / planner_server
nav2_controller / controller_server
nav2_bt_navigator / bt_navigator
nav2_lifecycle_manager / lifecycle_manager_navigation
```

Map:

- 6 m x 6 m generated occupancy map.
- Static occupied rectangles for the table and four bins.
- Resolution: `0.05 m`.

Controller:

- plugin: `nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController`
- desired linear velocity: `0.32 m/s`
- lookahead distance: `0.25 m`
- rotate-to-heading angular velocity: `0.75 rad/s`
- max angular acceleration: `2.4 rad/s^2`
- goal xy tolerance: `0.15 m`
- goal yaw tolerance: `0.35 rad`
- collision detection enabled
- collision lookahead window: `0.20 s`
- progress checker movement allowance: `45 s`
- y velocity is not commanded by RPP; `min_y_velocity_threshold` is set to `0.001`
  and the Isaac bridge consumes only `cmd_vel.linear.x` and `cmd_vel.angular.z`.

Isaac execution limits:

- wheel/ground material baseline: `static_friction=1.45`,
  `dynamic_friction=1.10`
- no per-wheel material binding is applied in the current baseline.
- mass check for the loaded S62 URDF: `29` mass entries, total model mass about
  `204.3 kg`; each wheel is `6.0 kg`, so the robot is not using an obviously
  underweight base model.
- max linear speed: `0.45 m/s`
- max angular speed: `0.75 rad/s`
- external Nav2 angular command scale: `2.5`, clamped to max angular speed

## Commands

ROS-only Nav2 smoke test:

```bash
cd mda_isaaclab
source task_319_garbage_sort/scripts/setup_nav2_user_install.bash
/usr/bin/python3 task_319_garbage_sort/scripts/ros2_nav2_stack_launcher.py --output-dir task_319_garbage_sort/output/nav2_smoke_stack
```

Second terminal:

```bash
cd mda_isaaclab
source task_319_garbage_sort/scripts/setup_nav2_user_install.bash
/usr/bin/python3 task_319_garbage_sort/scripts/ros2_nav2_smoke_test.py --goal-x 0.60 --goal-y 0.0 --goal-yaw 0.0 --timeout-s 70 --server-timeout-s 25 --goal-attempts 5 --retry-delay-s 2
```

Isaac motion-only single-category test:

```bash
cd mda_isaaclab
python task_319_garbage_sort/task319_grasp_sort_sm.py --motion_only_sort_demo --nav_backend nav2 --motion_test_category 其他垃圾 --record_debug --record_video --warmup_steps 80
```

Headless verified command:

```bash
cd mda_isaaclab
timeout 900s python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --motion_only_sort_demo --nav_backend nav2 --motion_test_category 其他垃圾 --record_debug --record_video --warmup_steps 80 --nav2_stack_startup_s 2
```

Visible waypoint-route demo:

```bash
cd mda_isaaclab
python task_319_garbage_sort/task319_grasp_sort_sm.py --waypoint_nav_demo --waypoint_route home,table_left,bin_center,table_right,home --nav_backend nav2 --record_debug --record_video --warmup_steps 80 --nav2_stack_startup_s 2 --post_action_hold_steps 240
```

Headless waypoint-route verification:

```bash
cd mda_isaaclab
timeout 900s python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --waypoint_nav_demo --waypoint_route home,table_left,bin_center,table_right,home --nav_backend nav2 --record_debug --record_video --warmup_steps 80 --nav2_stack_startup_s 2
```

Wheel open-loop sanity check:

```bash
cd mda_isaaclab
timeout 600s python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --wheel_open_loop_demo --record_debug --record_video --warmup_steps 40 --wheel_open_loop_steps 240 --wheel_open_loop_linear_speed 0.25 --wheel_open_loop_angular_speed 0.0
```

Official Isaac wheeled-controller diagnostic:

```bash
cd mda_isaaclab
timeout 260s python task_319_garbage_sort/official_holonomic_wheel_test.py --headless --record_debug --controller_backend differential --warmup_steps 5 --drive_steps 40 --cmd_vx 0.18 --cmd_vy 0.0 --cmd_wz 0.0
```

## Latest Verified Result

Official wheeled-controller diagnostic:

```text
task_319_garbage_sort/output/official_holonomic_wheel_tests/20260622_234208
```

Summary:

```text
success=True
controller_backend=differential
wheel_targets_radps=[1.380898, 1.380898, 1.380898, 1.380898]
translation_m=0.000116
yaw_delta_rad=-0.147768
```

Waypoint route run:

```text
task_319_garbage_sort/output/head_camera_grasp_records/20260622_225252
```

Summary:

```text
success=True
reason=Waypoint Nav2 demo completed.
route=home,table_left,bin_center,table_right,home
video=task_319_garbage_sort/output/head_camera_grasp_records/20260622_225252/external_grasp_demo.mp4
```

Waypoint transitions:

| Transition | Status | Target | Final |
| --- | --- | --- | --- |
| `current -> home` | `SUCCEEDED` | `[0.53, 0.0, 0.0]` | `[0.5300005, -0.0000002, -0.0002189]` |
| `home -> table_left` | `SUCCEEDED` | `[1.8, -1.35, pi/2]` | `[1.6629844, -1.2966428, 1.2681111]` |
| `table_left -> bin_center` | `SUCCEEDED` | `[3.75, 0.0, pi]` | `[3.7609110, -0.1486804, 2.8583613]` |
| `bin_center -> table_right` | `SUCCEEDED` | `[1.8, 1.35, -pi/2]` | `[1.9317502, 1.4172001, -1.8667554]` |
| `table_right -> home` | `SUCCEEDED` | `[0.53, 0.0, 0.0]` | `[0.4403284, 0.1163044, -0.2839006]` |

Nav2 failure hints for this run:

```text
collision_ahead=False
failed_to_make_progress=False
controller_patience_exceeded=False
```

Earlier motion-only sorting run:

Run:

```text
task_319_garbage_sort/output/head_camera_grasp_records/20260622_220145
```

Summary:

```text
success=True
reason=Motion-only Nav2 sort demo completed.
```

Navigation results:

| State | Label | Status | Target | Final |
| --- | --- | --- | --- | --- |
| `NAV_TO_TABLE_STANDPOINT` | `table_front_center_其他垃圾` | `SUCCEEDED` | `[0.53, 0.0, 0.0]` | `[0.5300004, -0.0000004, -0.0001555]` |
| `NAV_TO_BIN_STAGING` | `bin_staging_其他垃圾` | `SUCCEEDED` | `[3.75, 0.0, pi]` | `[3.7799368, -0.1470969, 2.8219863]` |
| `NAV_TO_BIN` | `bin_other_其他垃圾` | `SUCCEEDED` | `[3.75, 0.72, pi]` | `[3.7245781, 0.5721859, 2.9131293]` |
| `RETURN_TO_BIN_STAGING` | `return_to_bin_staging_其他垃圾` | `SUCCEEDED` | `[3.75, 0.0, -1.5263384]` | `[3.8059864, 0.1381317, -1.8522578]` |
| `RETURN_TO_SIDE_CORRIDOR` | `return_to_side_corridor_其他垃圾` | `SUCCEEDED` | `[3.75, 1.35, 1.6169619]` | `[3.7946284, 1.2073190, 1.8403748]` |
| `RETURN_HOME_APPROACH` | `return_home_approach_其他垃圾` | `SUCCEEDED` | `[0.08, 1.35, 3.1032010]` | `[0.2263152, 1.3231666, 3.0024144]` |
| `RETURN_HOME_ALIGN` | `return_home_align_其他垃圾` | `SUCCEEDED` | `[0.08, 0.0, -1.6809285]` | `[0.0392926, 0.1438615, -1.3428086]` |
| `RETURN_HOME` | `return_home_其他垃圾` | `SUCCEEDED` | `[0.53, 0.0, 0.0]` | `[0.3839919, -0.0313541, 0.1774371]` |

Stub actions:

| Action | Status |
| --- | --- |
| Pick/hold stub | success, `arm_pose=right_arm_carry` |
| Drop stub | success, `arm_pose=natural_down` |

Video:

```text
task_319_garbage_sort/output/head_camera_grasp_records/20260622_220145/external_grasp_demo.mp4
```

## Nav2 Failure Semantics

The return-to-observe failure in the motion metadata is a Nav2 action failure,
not a Python crash. In the runs that failed before `20260622_220145`,
`controller_server` printed `Failed to make progress`: the local controller kept
sending angular-only commands near the bin/table obstacle region, the robot did
not make enough translational progress, and Nav2 aborted the current navigation
goal.

The fix was to keep using Nav2 but split the return route into safer
intermediate goals around the table side corridor instead of asking Nav2 to
return directly from a bin-side pose to the table standpoint.

Additional waypoint-debug failures before the current fix:

- `20260622_224341`: `home -> table_left` failed with controller
  `collision_ahead` and `controller_patience_exceeded`. The bundled RPP
  controller was then tuned to shorter lookahead and lower desired linear
  velocity.
- `20260622_224621`: `bin_center -> table_right` failed with
  `failed_to_make_progress`. Nav2 was still sending nonzero `/cmd_vel`, but the
  Isaac execution layer allowed the unstable wheel-contact model to kick the
  base yaw away from the path. The fix was to integrate planar base pose in
  `wheel + kinematic` coupling while still computing wheel joint velocity
  targets from `/cmd_vel`.

## Known Limitations

- Current waypoint validation uses wheel-mode kinematic wheel-ground coupling in
  Isaac. This is a controlled wheel-interface validation, not final pure
  wheel-contact validation.
- Static map geometry is hard-coded in the Nav2 stack launcher.
- Table/bin standpoints are fixed presets. Dynamic selection from object
  geometry is not wired into navigation yet.
- Perception and grasp are paused in `--motion_only_sort_demo`.
- Real object pickup/drop is stubbed in this mode; it only validates base motion
  and state-machine routing.

## Next Planned Work

- Re-enable perception after the movement state machine is stable.
- Replace fixed table standpoints with a computed reachable stand position around
  the table.
- Keep Nav2 as the navigation backend; do not add a custom path follower.
- Validate `--nav_actuation_mode wheel --wheel_ground_coupling contact` after
  tuning S62 wheel physics/contact.
- Integrate grasp success and carried-object state before driving to bins.
