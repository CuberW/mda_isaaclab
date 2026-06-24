# Task 320: Kuavo Dual-Arm Floor Sorting

This task is intentionally separate from `../task_319_garbage_sort`.

It creates a new IsaacLab scene with the Kuavo S62 robot, ten large floor
objects, four open sorting areas, head-camera perception, dual-arm carry poses,
strict contact-gated pickup, dynamic waist targets, and outside-obstacle
navigation.

The default mode is intentionally physically honest:

- `--carry_mode pure_contact`
- `--grasp_gate_mode contact_only`
- `--use_task320_contact_pads`
- no object pose writes
- no object velocity writes
- no object wrench/servo carry

In this mode, the object can move only through real Isaac/PhysX contact with
the robot links. If bilateral arm contact is not generated, the run stops and
reports the grasp failure instead of pretending the object was carried.

Task 320 does not edit the official Kuavo URDF. By default it generates a local
URDF variant at `generated_assets/biped_s62_task320_contact_pads.urdf` and uses
that only for IsaacLab spawning. The patch adds finite box collision pads to the
left/right wrist and end-effector helper links so the bare arms have real
contact surfaces. IK still uses the official Kuavo URDF and task.info.

Kuavo's official standalone IK is available without starting ROS through
`../robot/planning/kuavo_kinematics_solver.py`, but the default execution
backend is currently the conservative `--ik_backend heuristic` path. The
standalone IK still needs frame calibration against the IsaacLab-imported S62
links before it should drive the visible robot by default. Enable it explicitly
with `--ik_backend official_standalone` when debugging that calibration.

The local C ABI library is built at:

```bash
PKG_CONFIG_PATH=/home/zhxm/miniconda3/envs/my_task319_safe/lib/python3.11/site-packages/cmeel.prefix/lib/pkgconfig:/opt/ros/jazzy/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH \
CMAKE_PREFIX_PATH=/home/zhxm/miniconda3/envs/my_task319_safe/lib/python3.11/site-packages/cmeel.prefix:$CMAKE_PREFIX_PATH \
cmake -S ../robot/standalone_ik -B ../robot/standalone_ik/build_local \
  -DKUAVO_ROOT=/home/zhxm/workspace/mda_isaaclab/kuavo-ros-opensource \
  -DCMAKE_BUILD_TYPE=Release

cmake --build ../robot/standalone_ik/build_local -j2
```

The official IK returns 18 controlled joints in this order:
`knee_joint`, `leg_joint`, `waist_pitch_joint`, `waist_yaw_joint`, left arm
1-7, right arm 1-7. Task 320 reads that order from the C ABI and applies the
returned lower-body, waist, and arm targets when `--unlock_whole_body` is true.
For this IsaacLab demo, `--official_ik_conservative_lower_body_limits` is
enabled by default so the lower body participates without driving the waist or
lift joints to extreme URDF limits during short scripted motions.
`--official_ik_reject_large_joint_delta` is also enabled by default: if one IK
stage requires a large lower-body/waist jump or excessive arm-joint jump, the
stage is rejected and the grasp fails instead of executing a twisted posture.
The grasp distance diagnostics use end-effector-to-object-surface distance; the
strict default gate still requires bilateral contact force.

The previous visualization path is still available for debugging with
`--carry_mode physical_soft_constraint --grasp_gate_mode contact_or_distance`,
but that mode is not strict physical grasping because it servo-controls the
object after the gate passes.

Use `--no-use_task320_contact_pads` to compare against the original Kuavo URDF;
that model has tiny or empty end-effector collision geometry and usually cannot
produce stable bilateral arm contact.

Default run shape:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_320_dual_arm_floor_sort/floor_dual_arm_sort_demo.py \
  --headless --max_objects 10 --perception_source mock_scene --force_exit
```

Use `--perception_source viss_qwen_first` to call the same VISS Qwen-first
perception script used by the parent garbage sorting task.

Visualization-only carry demo:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_320_dual_arm_floor_sort/floor_dual_arm_sort_demo.py \
  --headless --max_objects 1 \
  --carry_mode physical_soft_constraint --grasp_gate_mode contact_or_distance \
  --force_exit
```

Use `--ik_backend official_standalone` to test the extracted Kuavo standalone IK
path.
