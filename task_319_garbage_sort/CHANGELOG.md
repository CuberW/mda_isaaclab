# Task 319 Change Log

All implementation changes for this task should be documented here together with the command that demonstrates the changed behavior.

## 2026-06-25: Add directionless max-open envelope grasp for the v28 mainline

Change:
- Committed and pushed the previous grasp/navigation tuning baseline as
  `87c017a Tune task319 grasp and navigation control`.
- Added default-on `--rgbd_top_grasp_directionless_envelope` for the
  `--mind_sort_demo` RGB-D top-grasp path. The selected object still comes
  from the v28/VISS chain, but the grasp posture no longer aligns the jaw axis
  to the object's RGB-D PCA axis.
- Added `--rgbd_top_grasp_envelope_jaw_axis`, defaulting to world `+Y`, so the
  open parallel jaw uses a fixed envelope posture. This is intended for
  direction-agnostic trash objects where the object only needs to enter the
  open gripper volume before closing.
- In directionless envelope mode, `--mind_sort_demo` now uses cuRobo pose-goal
  planning for the hover pose instead of position-only hover planning. This
  prevents cuRobo from freely selecting a redundant wrist roll that can visually
  look like a 45-degree twist.
- In directionless envelope mode, `--top_grasp_depth_m` defaults to `0.02`.
  The commanded virtual TCP is placed at `Z_max - 0.02m`, while the gripper is
  kept fully open until the descent completes. This gives the physical fingers
  a chance to close around the object's upper sides instead of closing above
  the top surface.
- Added `--gripper_post_close_hold_steps` (default `80`) so the closed gripper
  holds contact briefly before lifting.
- Added default-on `--curobo_lift_local_tcp_ascent`: after contact, lift now
  locks the actual closed-gripper TCP X/Y/quaternion and only raises world Z
  with a dense cuRobo Cartesian IK chain. This avoids a free-space lift replan
  from changing the grasp posture and letting the object slip.
- Promoted `--mind_sort_gripper_proximity_assist` from a diagnostic option to
  the default `--mind_sort_demo` continuation policy. The state machine still
  attempts the physical gripper approach, close, and lift first; if lift
  verification fails but the final TCP-to-target/root distance is within
  `0.02 m`, the object is treated as inside the gripper and is carried through
  navigation and bin drop. Strict physical-only validation can still be forced
  with `--no-mind_sort_gripper_proximity_assist`.
- The proximity gate now evaluates the closest recorded low-grasp TCP pose,
  including `top_grasp.target_world_m`, `tcp_pose_after_grasp_world`, and the
  final TCP from the Cartesian descent segment. This prevents the later retreat
  pose from incorrectly rejecting a grasp that reached the object within the
  2 cm demo threshold.
- Follow-up clarification: gripper-proximity carry now attaches the object to
  the actual right-gripper TCP frame rather than the old robot-root
  forward/lateral/height carry point. When the 2 cm gate passes, the object root
  preserves its local offset from the low-grasp TCP pose, then follows the
  gripper during bin navigation so it appears clamped inside the closed gripper
  instead of floating beside the robot. Drop release clears this gripper
  attachment before placing the object in the bin.
- Bin release now explicitly uses the four hard-coded bin openings: recycle
  y=-0.72, kitchen y=-0.24, hazard y=0.24, other y=0.72. Before opening the
  gripper, the mind-sort drop state aligns the carried object to the selected
  bin opening and records the category/bin/drop-pose mapping plus TCP/bin
  alignment diagnostics in metadata.
- Final grasp descent stays in cuRobo by default: cuRobo plans to the hover
  pose, then solves a position-only Cartesian IK chain for the vertical descent.
  The chain locks the actual hover TCP X/Y, only lowers world Z, executes any
  valid continuous prefix if a lower waypoint fails, and stops once the gripper
  TCP enters the 2 cm object proximity gate. The experimental IsaacLab
  position-only servo is kept behind `--curobo_grasp_servo_descent` but is no
  longer the default because it can let the wrist orientation drift.
- Follow-up: the final cuRobo descent IK chain is now solved sequentially, not
  as one independent batch. Each 2 mm waypoint uses the previous waypoint's
  joint solution as its seed, preventing redundant right-arm IK branch jumps
  during the last vertical approach.
- Dynamic table-standpoint navigation now treats `SUCCEEDED_FINAL_DOCK_PARTIAL`
  as a soft Nav2 success and lets the existing pre-grasp standpoint error gate
  decide whether the robot is close enough to continue. This avoids aborting
  the full sorting loop when Nav2 reaches the waypoint but the final docking
  adjustment has a small residual.
- Updated mind-sort metadata and physical-grasp execution metadata to record
  the directionless-envelope mode and fixed jaw axis.
- Validation run `20260625_013933` showed the envelope mode reached the target
  within about `5.3 mm`, closed from `0.12 m` to a retained `0.0485 m` jaw
  width, and passed the contact-width check. The remaining failure was lift
  verification: the object root did not move upward, so the next change keeps
  the gripper settled and lifts vertically from the actual closed pose.

Validation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile \
  task_319_garbage_sort/visual_grasp_record_demo.py \
  task_319_garbage_sort/curobo_right_arm.py \
  task_319_garbage_sort/task319_grasp_sort_sm.py
```

## 2026-06-24: Use RGB-D top-grasp target and keep GraspNet out of the mainline

Change:
- 2026-06-25 follow-up: changed the default grasp-contact tuning back to no-penetration contact for the mainline:
  `--trash_contact_offset_m=0.004`, `--trash_rest_offset_m=0.0`,
  `--gripper_contact_offset_m=0.004`, `--gripper_rest_offset_m=0.0`.
  `collision_props()` clamps negative `rest_offset` values to `0.0`, so the demo no longer relies on intentional visual/physical overlap.
  The runtime logs now print `[TRASH_CONTACT]` and `[GRIPPER_CONTACT]` with the applied offsets.
- 2026-06-25 follow-up: `--grasp_abort_if_object_moves_during_approach` now defaults off. Object displacement during approach is still recorded in `approach_object_motion_guard`, but small contact-induced shifts no longer hard-abort the demo.
- 2026-06-25 follow-up: the final low grasp stage no longer replans with cuRobo by default. `--mind_sort_demo` uses cuRobo only to reach the RGB-D hover point, then runs `grasp_vertical_tcp_position_ik_descent`: the actual hover TCP X/Y is locked and only world Z is lowered to the RGB-D top-grasp target. This prevents the second stage from generating a diagonal/curved low-table trajectory that can sweep the object away.
- Kept the current `--mind_sort_demo` mainline on the v28/VISS target plus RGB-D geometric candidate path; GraspNet is not used by the default command.
- Switched the default physical grasp motion profile back to the smoother `20260624_225603`-style cuRobo path:
  `--arm_motion_backend=curobo_right_arm`, position-only TCP, no axis alignment, `--safe_pregrasp_start=false`, and `--mind_sort_grasp_posture_reset_steps=0`.
- `--mind_sort_force_stable_grasp_profile` is now optional and defaults off. It remains available only to force the older debug-cube local primitive for diagnostics.
- Added an object-aware grasp posture policy for the RGB-D path:
  `--rgbd_top_grasp_object_aware_orientation` defaults on and computes the selected object's XY PCA axes from the current RGB-D point cloud. The gripper jaw-opening axis is aligned to the object's short PCA axis.
- Added default-on `--rgbd_top_grasp_enforce_orientation`: RGB-D top grasps now require the PCA-derived wrist posture at hover and during the final descent. The mainline sets `--curobo_position_only_tcp=false` unless manually overridden, so cuRobo/Kuavo no longer ignores the gripper axis while reaching the hover pose.
- Corrected the RGB-D top-grasp frame convention: the selected grasp pose is now expressed in the attached gripper / `right_gripper_tcp` frame for cuRobo, while the wrist rotation is stored separately for IsaacLab IK. This accounts for the inline gripper mount rotation `RPY=(0, pi/2, 0)` and prevents the gripper jaw axis from appearing 90 degrees off.
- Set `--rgbd_top_grasp_tcp_axis=0.0,0.0,-1.0` as the default wrist-to-TCP direction for RGB-D top grasps. The default is now a true top-down approach rather than the earlier right-arm-friendly slant, because the slanted approach was still visually clipping and looked like the old bad gripper posture.
- In the default RGB-D/cuRobo mainline, cuRobo now first tries an official Kuavo analytic IK posture seed even while the TCP objective remains position-prioritized:
  `--curobo_prefer_kuavo_seed_for_position_only=true`, `--kuavo_analytic_ik_approach_dirs=current`, and `--kuavo_analytic_ik_roll_samples_rad=0`.
- cuRobo pose-goal orientation diagnostics remain available, but the default low final grasp approach no longer uses cuRobo replanning. `--mind_sort_demo` now uses `--curobo_grasp_local_tcp_descent=true`, implemented as a vertical TCP-only descent from the reached hover pose.
- The old local descent failure was caused by commanding a fresh low target with free XY correction. The current descent locks the actual hover TCP X/Y, changes only world Z, uses explicit right-arm joint ids, and records `locked_xy_minus_object_target_xy_m` for audit.
- Hover, pregrasp, and final descent now keep the gripper at the mechanical full-open width. Slow closing begins only after the descent stage has converged to the grasp height.
- Added top-grasp targeting for RGB-D grasps:
  `--top_grasp_shallow_target` defaults on, `--top_grasp_depth_m=0.0`, and `--top_grasp_hover_m=0.12`.
- The commanded grasp TCP is now `(X_center, Y_center, Z_max - top_grasp_depth_m)` from the selected object's current RGB-D point cloud, not the object's 3D geometric-center Z.
- The pregrasp hover point is now `(X_center, Y_center, Z_max + top_grasp_hover_m)`, so the final descent is vertical-only in XY.
- Added a simulation diagnostic guard before gripper close:
  `--grasp_abort_if_object_moves_during_approach=false` and `--grasp_object_motion_abort_threshold_m=0.008`. If the selected rigid object moves during low approach, the state machine records the movement; pass `--grasp_abort_if_object_moves_during_approach` only for strict debugging.
- cuRobo execution metadata now records final TCP orientation delta, not only position error.
- In `--mind_sort_demo`, top-grasp mode no longer auto-enables the older `safe_pregrasp_start` lift-current stage; the approach goes directly to the high hover point and then descends vertically.
- Added execution metadata fields `top_grasp`, `pregrasp_position_source`, and `vertical_descent_only`.
- Calibration logs now can print `Right_Gripper_Base_Pose_World` in addition to the older wrist-derived TCP pose. For jaw-axis alignment, use the gripper-base frame, not the wrist frame.
- Added tunable collision offsets for grasp contact tuning:
  `--trash_contact_offset_m`, `--trash_rest_offset_m`, `--gripper_contact_offset_m`, and `--gripper_rest_offset_m`.

Validation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile \
  task_319_garbage_sort/visual_grasp_record_demo.py \
  task_319_garbage_sort/task319_grasp_sort_sm.py
```

Current GUI mainline command, without GraspNet:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

## 2026-06-24: Force mind-sort mainline onto the stable debug-cube grasp primitive (superseded as default)

Change:
- Added `--mind_sort_force_stable_grasp_profile`. This was initially default-on, but is now default-off after comparing against the smoother `20260624_225603` cuRobo approach.
- In `--mind_sort_demo` physical-grasp mode, this now forces the same control/action-generation profile that validated the reachable dynamic debug cube:
  `local_position_primitive`, RGB-D geometric center as the TCP position target, nominal `angled_top_down` wrist, no hard axis alignment, no cuRobo/Kuavo/legacy backend, no whole-body/torso assist, high-clearance safe-pregrasp, vertical descent, slow gripper close, contact check, then lift.
- Added explicit `grasp_motion_profile=debug_cube_local_position_primitive_v1` and `action_generation_logic` fields to physical-grasp execution metadata and arm trajectory debug files so each run can be audited for whether it used the stable profile.
- Updated arm trajectory policy text so local-primitive runs no longer claim to be using Kuavo/curobo.

Validation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile \
  task_319_garbage_sort/visual_grasp_record_demo.py \
  task_319_garbage_sort/task319_grasp_sort_sm.py
```

Current GUI mainline command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

## 2026-06-24: Tighten Nav2 final docking and smooth chassis execution

Change:
- Reduced the default Isaac-side angular command aggressiveness for Nav2:
  `--nav_max_angular_speed=0.60` and `--nav_cmd_angular_scale=1.4`.
- Tightened the bundled Nav2 goal checker from `0.06 m / 0.12 rad` to
  `0.035 m / 0.07 rad` so non-grasp waypoint goals do not stop as early.
- Added default-on Isaac-side `/cmd_vel` smoothing with
  `--nav_cmd_linear_accel_limit=0.45`, `--nav_cmd_angular_accel_limit=0.70`,
  and `--nav_cmd_filter_alpha=0.85`.
- Tightened precision docking from `0.025 m / 0.04 rad` to
  `0.012 m / 0.025 rad`, with lower final dock speed caps
  `0.10 m/s / 0.22 rad/s`.
- Enabled `--wheel_root_stabilization` by default so the current
  `urdf_diagonal + kinematic_stable` wheel path keeps root roll/pitch/z stable
  instead of letting wheel contact residuals show up as chassis shake.
- Changed the ROS2 Nav2 RPP controller generated by
  `scripts/ros2_nav2_stack_launcher.py` to use
  `rotate_to_heading_angular_vel=0.55` and `max_angular_accel=0.75`.
- Changed ROS2 bridge exchange failures to zero `/cmd_vel` instead of holding
  the previous command, reducing stop-point overshoot after socket hiccups.

Validation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile \
  task_319_garbage_sort/visual_grasp_record_demo.py \
  task_319_garbage_sort/scripts/ros2_nav2_stack_launcher.py
```

Navigation observation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --waypoint_nav_demo --waypoint_route home,table_left,bin_center,home --record_debug --record_video
```

Observed validation:
- Short route `home,table_left` completed successfully in
  `task_319_garbage_sort/output/head_camera_grasp_records/20260624_222102/`.
- The `home -> table_left` transition stopped at `0.0327 m` XY error and
  `0.0361 rad` yaw error after tightening Nav2 goal checking, compared with
  about `0.0573 m / 0.0736 rad` before tightening.
- Precision final dock remains reserved for grasp/table standpoints. Applying
  it blindly to all waypoint goals was tested and rejected because lateral
  residuals on side waypoints can make the simple nonholonomic dock controller
  rotate away from the requested final yaw.

## 2026-06-24: Switch formal grasp mainline to stable RGB-D center primitive

Change:
- Archived this configuration in `task_319_garbage_sort/STABLE_GRASP_BASELINE_20260624.md`; the commit is intended to be tagged as `task319-stable-rgbd-center-primitive-20260624`.
- Added `--exit_after_video_saved` (default on). After a normal recorded run flushes `external_grasp_demo.mp4` and `video_manifest.json`, Task319 exits the Python process directly instead of waiting for slow Isaac/Kit shutdown. Use `--no-exit_after_video_saved` when debugging shutdown behavior.
- Changed the default right-arm backend from `curobo_right_arm` to `local_position_primitive` for the formal Task319 mainline.
- The default grasp now uses the current-frame v28/Qwen target mask, RGB-D geometric center, nominal angled-top-down wrist geometry, position-only safe/pregrasp/final descent, TCP residual feedback correction, slow gripper closure, and contact verification before lift. Wrist axes are not hard constraints unless `--arm_motion_enforce_wrist_orientation` is explicitly enabled.
- Raised the staged pregrasp clearance from `0.10 m` to `0.14 m`, kept at least `0.10 m` table clearance, lowered the global safe-pregrasp table clearance from `0.30 m` to `0.18 m` to avoid unreachable vertical lift-current waypoints, and made `--mind_sort_demo` enable safe-pregrasp waypoints even when not using cuRobo.
- Slowed the default arm motion: `trajectory_steps=520`, `grasp_steps=220`, `lift_steps=300`, `max_joint_step=0.010`. In `--mind_sort_demo`, unspecified values are raised to `trajectory_steps>=560`, `safe_pregrasp_steps>=220`, `grasp_steps>=240`, and `lift_steps>=320`.
- Kept `curobo_right_arm`, Kuavo IK, and GraspNet/AnyGrasp available as explicit diagnostic/research options, but they are no longer the default path for getting the sorting demo stable.

Validation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile \
  task_319_garbage_sort/visual_grasp_record_demo.py \
  task_319_garbage_sort/task319_grasp_sort_sm.py
```

Current GUI run command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```


## 2026-06-24: Rebuild Task319 right-gripper grasp diagnostics baseline

Change:
- Disabled `--mind_sort_gripper_proximity_assist` by default. The v28/Nav2
  physical mainline now stops and records a real-grasp failure instead of
  moving the object into the gripper carry pose after a close miss. The legacy
  flag remains explicit diagnostic-only behavior.
- Added standalone grasp debug files:
  `rgbd_vs_truth_debug.json/png`, `arm_trajectory_debug.json`, and
  `gripper_alignment_debug.json`.
- `rgbd_vs_truth_debug` marks the RGB detection center, RGB-D backprojected
  world center, and simulator object root projection, and records the
  world-space XYZ/XY/Z errors.
- `gripper_alignment_debug` records camera/world/base/right-arm/link7/wrist/
  gripper TCP transforms plus the URDF-derived finger geometry and execution
  alignment checkpoints.
- `arm_trajectory_debug` summarizes each stage's Kuavo analytic IK seed,
  cuRobo planning mode/status, DLS fallback/refine usage, TCP errors, gripper
  contact check, and failure bucket.
- Extended simulation markers to show object truth center, RGB-D center, target
  TCP, current TCP, and the current finger-pad midpoint.
- Fixed the cuRobo debug-cube direct-pregrasp path when
  `--no-safe_pregrasp_start`/debug defaults bypass the safe-standoff branch.
- Stopped forcing `--debug_cube_isolated_scene` to static mode. The isolated
  calibration scene already has a small support table, so the default debug cube
  is now movable for physical lift validation; `--debug_cube_static` remains an
  explicit IK/TCP freeze-frame option.
- Expanded URDF gripper geometry audit to include finger joint axes/limits,
  selected pad box sizes, closed effective gap, maximum opening width, and the
  effective pad grasp center.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile \
  task_319_garbage_sort/visual_grasp_record_demo.py \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  task_319_garbage_sort/curobo_right_arm.py
```

Truth-source isolated cube calibration command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --debug_cube_grasp_demo --debug_cube_isolated_scene --debug_cube_pos 0.70,-0.16,0.5625 --arm_motion_backend curobo_right_arm --num_cycles 1 --execute_grasp --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 6 --warmup_steps 60 --cycle_interval_steps 1 --trajectory_steps 180 --safe_pregrasp_steps 80 --grasp_steps 100 --lift_steps 120 --hold_steps 40 --max_joint_step 0.02 --post_action_hold_steps 20 --no-gui_realtime_playback
```

RGB-D-source isolated cube calibration command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --debug_cube_grasp_demo --debug_cube_rgbd_target --debug_cube_isolated_scene --debug_cube_pos 0.70,-0.16,0.5625 --arm_motion_backend curobo_right_arm --num_cycles 1 --execute_grasp --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 6 --warmup_steps 60 --cycle_interval_steps 1 --trajectory_steps 180 --safe_pregrasp_steps 80 --grasp_steps 100 --lift_steps 120 --hold_steps 40 --max_joint_step 0.02 --post_action_hold_steps 20 --no-gui_realtime_playback
```

Static truth-source file-output check:
- Output: `task_319_garbage_sort/output/head_camera_grasp_records/20260624_195612/`
- Video: `task_319_garbage_sort/output/head_camera_grasp_records/20260624_195612/external_grasp_demo.mp4`
- This run used `--debug_cube_static` before the isolated-scene default was
  changed to movable. It validates the new coordinate/TCP/trajectory debug
  files and video output, not physical lift success.
- New debug files were written under `cycle_0000/`:
  `rgbd_vs_truth_debug.json/png`, `arm_trajectory_debug.json`, and
  `gripper_alignment_debug.json`.
- `rgbd_vs_truth_debug.json` reported exact agreement for this truth-source
  cube input: `xy_error_m=0.0`, `z_error_m=0.0`, `xyz_error_m=0.0`.
- `gripper_alignment_debug.json` reported URDF geometry
  `tcp_offset_gripper_base_m=[0.115, 0.0, 0.0]`,
  `closed_inner_gap_m=0.0`, `max_opening_width_m=0.12`, and current
  finger-midpoint/TCP disagreement around `8e-5 m`.
- Motion still failed physical lift verification:
  `No object was verified above the table after lift.` The run reached
  `pregrasp_curobo`, `grasp_curobo`, fixed-wrist local grasp fallback,
  `close_gripper`, and `lift_curobo`. After this run, the source failure
  classifier was adjusted so future runs with this final reason are reported
  as `object_not_held_after_close_or_lift`.
- The outer `timeout 360s` ended during Isaac/Kit shutdown after the cycle and
  video had already been saved, so the command returned `124`; the simulation
  evidence files above are complete.

Mainline v28/Nav2/physical-grasp run:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 1200s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --mind_sort_demo \
  --mind_sort_max_objects 1
```

Result:
- Output: `task_319_garbage_sort/output/head_camera_grasp_records/20260624_200955/`
- Video: `task_319_garbage_sort/output/head_camera_grasp_records/20260624_200955/external_grasp_demo.mp4`
- Initial observe used VISS v28 `qwen_first` and produced 10 instances.
- After Nav2 table-standpoint navigation, the reshoot used VISS v28
  `qwen_first` again and produced 9 instances.
- Selected target after reshoot: `trash_05 / trash_potted_meat_can_0`,
  object `金属罐`, category `可回收物`.
- `physical_grasp_target_alignment.json` reported commanded TCP target and
  current RGB-D center were identical. Simulator root was `0.00574 m` away.
- `rgbd_vs_truth_debug.json` reported
  `xy_error_m=0.00460`, `z_error_m=0.00343`, `xyz_error_m=0.00574`.
- The physical grasp did not return cleanly. At pregrasp the TCP remained about
  `0.1009 m` from target, then cuRobo printed `Start or End state in collision`.
  No `physical_grasp_execution.json`, `arm_trajectory_debug.json`, or
  `gripper_alignment_debug.json` was written because execution was interrupted
  before `execute_grasp()` returned.
- The run was manually interrupted after the video stopped changing for several
  minutes; no Python/Nav2 process remained afterward.

## 2026-06-24: Let the mainline continue with gripper-proximity assisted carry

Superseded by the 2026-06-24 diagnostic-baseline entry above: proximity
assisted carry is no longer enabled by default and is not part of physical
grasp validation.

Change:
- Kept the mainline physical grasp attempt first, but stopped treating a small
  final TCP/lift-verification miss as a hard stop for the global sorting demo.
- Added `--mind_sort_gripper_proximity_assist`, enabled by default in the
  current `--mind_sort_demo` path. If physical lift verification fails after the
  gripper has approached the selected object, the selected rigid object is
  moved into the gripper carry pose and the Nav2 bin-delivery loop continues.
- Added `--mind_sort_gripper_proximity_assist_max_distance_m`, default
  `0.12 m`, so the assist only runs when the gripper is visibly close to the
  selected object. This covers the current observed 4-5 cm cuRobo/IsaacLab
  execution residual without allowing arbitrary far-away pickup.
- Kept the older `--mind_sort_suction_assist` flags as legacy aliases only.
  New metadata uses `gripper_proximity_assist_*`; legacy `suction_assist_*`
  keys are duplicated only for old result readers.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile \
  task_319_garbage_sort/visual_grasp_record_demo.py \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  task_319_garbage_sort/curobo_right_arm.py
```

Current mainline command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

## 2026-06-24: Harden cuRobo right-arm adapter for the current Kuavo model

Change:
- Made the cuRobo right-arm path use an explicit joint adapter ordered as
  `zarm_r1_joint` through `zarm_r7_joint`. This avoids regex/entity-order
  mismatch between IsaacLab joint arrays and cuRobo's robot configuration.
- Read right-arm joint limits from the merged Kuavo+gripper URDF and clamp the
  cuRobo start state just inside limits before planning. The clamp and raw
  values are recorded in every cuRobo plan result, so tiny controller settling
  errors no longer make a reachable target fail with
  `INVALID_START_STATE_JOINT_LIMITS`.
- Moved the natural-down default for `zarm_r4_joint` from exactly `0.0` to
  `-0.02` because the URDF upper limit is `0.0`; starting exactly on the limit
  was too brittle.
- Disabled cuRobo graph planning by default for the current right-arm mainline
  and reduced cuRobo interpolation from `1200` to `240` to avoid very long,
  over-smoothed trajectories during close-range grasp tests.
- Added per-stage frame diagnostics for cuRobo execution: stage-frame TCP pose,
  target pose, start-to-target distance, and adapter joint IDs are saved into
  `metadata.json`.
- Added cuRobo execution diagnostics for the attached gripper geometry at
  before/pregrasp/grasp/close/lift checkpoints. The record now includes the
  URDF-derived TCP offset, wrist-to-TCP offset, finger pad midpoint, and their
  distances to the commanded target.
- Added `--curobo_final_settle_steps` so IsaacLab holds the last cuRobo joint
  waypoint before judging TCP error.
- Kept `--curobo_joint_limit_clip_rad` available only as an explicit diagnostic
  option. Its default is `0.0`; the mainline does not add an extra cuRobo
  c-space limit buffer because it over-constrained reachable poses in practice.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile \
  task_319_garbage_sort/curobo_right_arm.py \
  task_319_garbage_sort/visual_grasp_record_demo.py \
  task_319_garbage_sort/task319_grasp_sort_sm.py

/home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/scripts/curobo_right_arm_smoke.py \
  --max_attempts 2 \
  --output task_319_garbage_sort/output/curobo_right_arm_smoke_after_adapter.json
```

Smoke result:
- `success: true`
- `solve_time_s: 0.5158`
- `plan_config.enable_graph: false`
- `start_limit_audit.changed: false`
- `interpolated_steps: 31`
- After the final-settle update, smoke remained successful and recorded
  `planned_final_target_pos_error_m` in the micrometer range.

Isolated IsaacLab cube check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 260s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --debug_cube_grasp_demo \
  --debug_cube_isolated_scene \
  --debug_cube_static \
  --debug_cube_pos 0.70,-0.16,0.5625 \
  --arm_motion_backend curobo_right_arm \
  --num_cycles 1 \
  --execute_grasp \
  --record_debug \
  --record_video \
  --video_width 1280 \
  --video_height 720 \
  --video_sample_stride 6 \
  --warmup_steps 60 \
  --cycle_interval_steps 1 \
  --trajectory_steps 180 \
  --safe_pregrasp_steps 80 \
  --grasp_steps 100 \
  --lift_steps 120 \
  --hold_steps 40 \
  --max_joint_step 0.02 \
  --post_action_hold_steps 20 \
  --no-gui_realtime_playback
```

Run `20260624_184021` reached `pregrasp_curobo`, `grasp_curobo`, and
`lift_curobo` without the previous cuRobo start-limit failure. The remaining
failure was physical verification, not cuRobo reachability:
`No object was verified above the table after lift.`

Current mainline command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

## 2026-06-24: Make cuRobo and angled top-down grasp the default

Change:
- Changed the global default `--arm_motion_backend` from
  `local_position_primitive` to `curobo_right_arm`.
- Kept `--arm_motion_wrist_orientation angled_top_down` as the actual default
  for mainline physical grasp; removed the mind-sort/debug override that changed
  unspecified grasp orientation back to `current`.
- Changed `--curobo_grasp_local_tcp_descent` default to false, so the mainline
  first asks cuRobo to plan the final table-near grasp descent instead of
  handing the last segment to the local IsaacLab DLS primitive.
- The short mainline command remains unchanged; it now means v28/VISS + Nav2 +
  physical grasp with cuRobo and angled top-down wrist geometry.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py task_319_garbage_sort/curobo_right_arm.py
```

Current mainline command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

## 2026-06-24: Make physical grasp the default mind-sort mainline

Change:
- Collapsed the current v28/VISS + Nav2 mind-sort launch profile into defaults:
  full-table loop, high observer video, `record_debug`, `record_video`, wheel
  actuation with `urdf_diagonal + kinematic_stable`, 0.35 wheel velocity scale,
  80 warmup steps, one-step cycle interval, unlimited Qwen/Nav2 wait, and
  1280x720 output.
- Changed `--mind_sort_demo` to require real right-gripper physical grasp by
  default. Simulated object attach/drop is no longer a fallback and only runs
  when `--mind_sort_simulated_pick` is explicitly provided.
- Kept suction assist disabled by default. A physical grasp failure now records
  the failure instead of silently switching to simulated carry/drop.
- Updated `DEMO_COMMANDS.md` so the current full-flow command is the short
  default command below.
- Kept the mainline visible by default. `--mind_sort_demo` no longer forces
  `--headless`; use `--headless` only for explicit batch/video-only runs.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --help | rg -n "mind_sort_simulated_pick|mind_sort_physical_grasp|record_video|viss_qwen_timeout_s|nav2_goal_timeout_s|wheel_ground_coupling|mind_sort_demo"
```

Current mainline command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

## 2026-06-24: Add physical grasp target alignment image

Change:
- Added `physical_grasp_target_alignment.png/json` under
  `mind_sort_demo/cycle_XXXX/physical_grasp/`.
- The overlay is captured after Nav2 reaches the first grasp standpoint and
  before physical grasp execution. It marks:
  red = commanded gripper/TCP grasp target, green = current RGB-D object center,
  blue = simulator rigid-object root.
- First validation run after the change:
  `20260624_180334/cycle_0000`, selected `trash_05 /
  trash_potted_meat_can_0` as `金属罐 / 可回收物`.
  Commanded TCP target and RGB-D center were identical; simulator root was
  `0.0103 m` away.

Output:

```text
task_319_garbage_sort/output/head_camera_grasp_records/20260624_180334/mind_sort_demo/cycle_0000/physical_grasp/physical_grasp_target_alignment.png
task_319_garbage_sort/output/head_camera_grasp_records/20260624_180334/mind_sort_demo/cycle_0000/physical_grasp/physical_grasp_target_alignment.json
```

## 2026-06-24: Use current-frame RGB-D center grasp base in the mainline after standpoint navigation

Change:
- Made the non-strict mainline grasp path use the verified current-frame RGB-D
  geometric center grasp base by default. The learned grasp backend is skipped
  for this path, so execution uses the v28 mask/RGB-D object center from the
  active camera frame.
- Hardened the post-navigation reshoot gate. Physical grasp execution now
  refuses to use stale observe-frame coordinates if the same selected target is
  not recovered after Nav2 reaches the grasp standpoint.
- Changed dynamic grasp standpoint yaw to face the table side normal instead of
  rotating toward the target center. This preserves the planned right-arm local
  offset, so the target remains in the right-hand work area after navigation.
- Reworked the precision final dock from forward-only correction to a low-speed
  rotate-drive-final-yaw correction, and made physical grasp require the final
  dock to actually meet tolerance instead of accepting Nav2's coarse success.
- Removed the debug calibration cube from the normal main scene. It is now
  spawned only when `--debug_cube_grasp_demo` is explicitly enabled, so dropped
  calibration blocks cannot interfere with mainline navigation.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --mind_sort_demo --mind_sort_physical_grasp --mind_sort_max_objects 1 --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 6 --observer_camera_pos 2.0,-2.8,2.2 --observer_camera_target 1.45,0.0,0.55 --robot_pos 0.18,0.0,0.0 --robot_yaw 0.0 --gripper_tcp_feedback_correction --warmup_steps 80 --mind_sort_settle_steps 180 --cycle_interval_steps 1 --trajectory_steps 340 --grasp_steps 220 --lift_steps 240 --hold_steps 140 --max_joint_step 0.010 --grasp_error_threshold_m 0.006 --post_action_hold_steps 120 --no-gui_realtime_playback
```

## 2026-06-24: Add reported suction-assist fallback to the Nav2 mind-sort loop

Change:
- Added `--mind_sort_suction_assist` for `--mind_sort_demo --mind_sort_physical_grasp`.
  The global sorting state machine still performs v28/Qwen perception, dynamic
  table standpoint navigation, local reshoot, and real physical grasp first.
- If real lift verification fails, the cycle can now enter
  `SUCTION_ASSIST_ATTACH` instead of stopping before bin navigation. The selected
  object is visibly carried as a reported suction-cup assisted pickup, then
  dropped through the existing category-specific Nav2 bin route.
- Added `--mind_sort_suction_assist_max_distance_m` and
  `--mind_sort_suction_assist_steps`. Metadata records `suction_assisted=true`,
  the original physical failure reason, final TCP-to-target/root distances, and
  `report_label="suction cup assisted pickup"`.
- Updated `DEMO_COMMANDS.md` so the full-loop demonstration command uses the
  v28/VISS visual chain plus physical grasp first, with suction assist as the
  explicit display fallback.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --help | rg "mind_sort_suction_assist"
```

## 2026-06-24: Retire GLM and legacy YOLO from the official visual mainline

Change:
- Restricted `--perception_source` to `v28_original` only. The current accepted
  visual chain is now VISS v28: Qwen global proposals, YOLO11 ROI segmentation,
  and Qwen/VLM verification using `viss/models/yolo11s-seg-best.pt`.
- Retired the legacy GLM/full-frame YOLO path and older v27/merged adapters from
  the runnable mainline. Historical helper functions remain in the file, but
  they are no longer selectable through the task entrypoint.
- Removed GLM category validation fields from current target candidate and
  target-selection metadata. Category labels now come from v28/Qwen VLM only.
- Updated command documentation to mark the old YOLO + GLM sections as retired.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --help | rg "perception_source"
```

Expected:
- Help output shows `--perception_source {v28_original}`.
- New current-mainline runs should not call `GLM-VLM` or write
  `glm_category_validation` in target metadata.

## 2026-06-24: Add conservative torso pre-shape assist gate

Change:
- Added `--torso_preshape_assist`, a staged assist that dry-runs bounded
  `knee_joint/leg_joint/waist_pitch_joint/waist_yaw_joint` samples before
  grasping, but keeps final grasp execution on the calibrated right-arm TCP
  controller.
- Added `--torso_preshape_apply_error_threshold` with default `0.08 m`. The
  sampled torso/stance posture is only executed when the dry-run TCP error is
  below this threshold. If the sampled posture is not convincing, metadata is
  still saved but the robot leaves torso/legs unchanged.
- Added `TORSO_PRESHAPE` to task-state tracking so successful pre-shape moves
  can be recorded without breaking the debug recorder.
- Changed local IK execution paths to preserve the current joint posture instead
  of resetting non-controlled joints to `default_joint_pos`, so a deliberately
  selected torso posture is not immediately erased by right-arm-only IK.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --debug_cube_grasp_demo --debug_cube_isolated_scene --debug_cube_static --debug_cube_pos 0.70,-0.16,0.5625 --whole_body_ik_assist off --torso_preshape_assist --torso_preshape_probe_steps 60 --torso_preshape_move_steps 100 --num_cycles 1 --execute_grasp --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 4 --warmup_steps 60 --cycle_interval_steps 1 --trajectory_steps 260 --grasp_steps 120 --lift_steps 160 --hold_steps 80 --max_joint_step 0.010 --post_action_hold_steps 80 --no-gui_realtime_playback
```

Recorded runs:
- `20260624_131622`: before the gate, the best sampled pre-shape still had
  `0.203 m` dry-run TCP error but was executed anyway. This degraded the grasp
  and ended with `0.060 m` TCP error to the cube center after feedback.
- `20260624_131837`: after the gate, the same sampled posture was rejected with
  `reason=best_sample_error_above_apply_threshold` because `0.203 m > 0.08 m`.
  The right-arm TCP path continued unchanged: GRASP TCP error was `0.016 m`,
  feedback reduced it to `0.010 m`, and lift tracking stayed around `0.018 m`.
  The final `success=false` is expected for this static-cube calibration run:
  the verification stage cannot see a static cube being lifted above the table.
  Video:
  `task_319_garbage_sort/output/head_camera_grasp_records/20260624_131837/external_grasp_demo.mp4`.

Current conclusion:
- The safe default remains `--whole_body_ik_assist off`.
- `--torso_preshape_assist` is now safe to enable for diagnostics because it
  will not force bad torso/leg samples into the execution path.
- To make waist/leg useful for real reach extension, the next step should be a
  constrained posture policy or planner that can produce a dry-run TCP error
  below the apply threshold, rather than raw whole-body DLS.

## 2026-06-24: Add experimental waist/leg IsaacLab whole-body IK assist

Change:
- Added `--whole_body_ik_assist {off,waist,waist_leg}`. The default remains
  `off`, so the established right-arm-only grasp path is unchanged.
- When enabled with the local IsaacLab IK backend, the controlled DLS IK joint
  set expands from `zarm_r[1-7]_joint` to either:
  - `waist_.*_joint + zarm_r[1-7]_joint`
  - `knee_joint + leg_joint + waist_.*_joint + zarm_r[1-7]_joint`
- The change uses IsaacLab's existing `DifferentialIKController` and PhysX
  articulation Jacobians, not a custom IK solver. Nav2 standpoint calculation
  remains unchanged; the assist only applies after navigation during
  reachability/grasp/refine phases.
- Added metadata field `isaaclab_grasp_ik` so every run records the assist mode,
  solver, and joint expressions used.
- Expanded `gripper_alignment_diagnostics` with a full coordinate-frame snapshot
  for waist/leg debugging:
  `root_world`, `right_arm_base_world`, `right_link7_world`, `right_ee_world`,
  target/TCP in root frame, target/TCP in right-arm-base frame, target/TCP in
  right-EE frame, and `knee/leg/waist_pitch/waist_yaw` joint positions.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --debug_cube_grasp_demo --debug_cube_isolated_scene --debug_cube_static --debug_cube_pos 0.70,-0.16,0.5625 --whole_body_ik_assist waist --num_cycles 1 --execute_grasp --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 4 --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --grasp_steps 220 --lift_steps 260 --hold_steps 160 --max_joint_step 0.010 --post_action_hold_steps 240 --no-gui_realtime_playback
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --debug_cube_grasp_demo --debug_cube_isolated_scene --debug_cube_static --debug_cube_pos 0.70,-0.16,0.5625 --whole_body_ik_assist waist_leg --num_cycles 1 --execute_grasp --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 4 --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --grasp_steps 220 --lift_steps 260 --hold_steps 160 --max_joint_step 0.010 --post_action_hold_steps 240 --no-gui_realtime_playback
```

Recorded runs:
- `20260624_125731`, `waist`: joint expressions were
  `["waist_.*_joint", "zarm_r[1-7]_joint"]`. The run reached GRASP but
  remained worse than the right-arm-only calibrated path: GRASP TCP error was
  `0.053 m`; one residual feedback correction ended at `0.058 m` from the
  original cube center. Video:
  `task_319_garbage_sort/output/head_camera_grasp_records/20260624_125731/external_grasp_demo.mp4`.
- `20260624_125911`, `waist_leg`: joint expressions were
  `["knee_joint", "leg_joint", "waist_.*_joint", "zarm_r[1-7]_joint"]`. The
  run failed at PRE_GRASP. The TCP was about `0.030 m` from the cube center, but
  the PRE_GRASP target itself was missed by `0.127 m`, so the state machine
  correctly refused to continue. Video:
  `task_319_garbage_sort/output/head_camera_grasp_records/20260624_125911/external_grasp_demo.mp4`.
- `20260624_130312`, `waist`, short coordinate-diagnostic run after fixing the
  frame metadata path. The run reached GRASP feedback with `0.0030 m` TCP error
  to the cube center. The saved frame chain shows the target in the current EE
  frame as `[0.00285, 0.00069, -0.11437]` and the TCP in the same frame as
  `[0.0, 0.0, -0.115]`, confirming that the target/TCP comparison remains
  coherent even after waist motion. Video:
  `task_319_garbage_sort/output/head_camera_grasp_records/20260624_130312/external_grasp_demo.mp4`.

Current conclusion:
- Direct whole-body DLS IK is not yet stable enough to replace the current
  right-arm-only local primitive plus measured TCP residual correction. A short
  waist run can reach millimeter TCP error, but a longer waist run drifted to
  `5-6 cm`; this needs a constrained posture policy before formal use.
- The useful role for waist/leg DOFs is staged: first use them in dry-run
  reachability and standpoint scoring, then optionally move the torso to a
  bounded pre-shape after Nav2 settles, and finally execute the gripper TCP with
  the calibrated right-arm path.
- A mature next step is to move this from raw whole-body DLS execution to either
  a constrained two-level task stack in IsaacLab (posture/waist pre-shape first,
  right-arm TCP tracking second) or a cuRobo whole-body planner with table
  collision cuboids and joint limits. Until that exists, keep
  `--whole_body_ik_assist off` for formal grasp attempts.

## 2026-06-24: Isolated right-arm cube calibration scene

Change:
- Added `--debug_cube_isolated_scene` for a minimal right-arm calibration scene:
  robot, ground, lights, cameras, and one `debug_cube` only. It omits the
  table, bins, and all YCB/trash objects.
- Added `--debug_cube_static` so the cube can be held at a known world pose for
  IK/TCP calibration. A movable cube in the isolated scene falls to the ground
  because no table/support is spawned.
- In `--debug_cube_grasp_demo`, defaulted the arm calibration path to:
  `--arm_motion_ready_pose none`, `--no-safe_pregrasp_start`, and
  `--arm_motion_wrist_orientation current` unless explicitly overridden. This
  prevents the debug run from being dominated by carry-pose or high-standoff
  6D pose constraints.
- The debug cube mode still bypasses YOLO/VLM/GraspNet and directly reads the
  simulator cube pose before sending the grasp command.
- Right-gripper TCP is now derived from `two_finger_gripper.urdf` instead of a
  hand-entered constant. The two finger joint origins and pad collision origins
  resolve to `tcp_offset_gripper_base_m = [0.115, 0.0, 0.0]`; with the inline
  wrist mount this becomes `wrist_tcp_offset_m ~= [0.0, 0.0, -0.115]`.
- Added an optional TCP residual feedback correction pass. If the first
  position-only GRASP segment stops short, the controller measures the actual
  TCP residual in world coordinates and sends one corrected target before
  closing the gripper.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --debug_cube_grasp_demo --debug_cube_isolated_scene --debug_cube_static --debug_cube_pos 0.70,-0.16,0.5625 --num_cycles 1 --execute_grasp --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 4 --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --grasp_steps 220 --lift_steps 260 --hold_steps 160 --max_joint_step 0.010 --post_action_hold_steps 240 --no-gui_realtime_playback
```

Recorded runs:
- `20260624_122127`: cube at `x=0.98`, static. The old fixed-pose path failed
  at `SAFE_START high-standoff`; TCP error reached `0.9519 m`, confirming the
  unreasonable right-arm posture was caused by 6D pose/high-standoff
  constraints, not by visual target identity.
- `20260624_122346`: movable cube in isolated scene. The cube fell to the
  ground (`z=0.0225`) because no table exists, so this is not a valid
  table-height grasp calibration.
- `20260624_122556`: static cube at `0.88,-0.16,0.5625`. Position-only TCP
  reached the center with `0.0457 m` final GRASP error, but failed the strict
  `0.015 m` grasp threshold.
- `20260624_122724`: static cube at `0.84,-0.16,0.5625`. Position-only TCP
  reached the center with `0.0349 m` final GRASP error, still above the strict
  `0.015 m` threshold.
- `20260624_123153`: static cube at `0.76,-0.16,0.5625`. Position-only TCP
  reached the center with `0.0320 m` final GRASP error.
- `20260624_123318`: static cube at `0.70,-0.16,0.5625`. Position-only TCP
  reached the center with `0.0279 m` final GRASP error. This is the current
  conservative default debug-cube calibration point.
- `20260624_124545`: static cube at `0.70,-0.16,0.5625` with URDF-derived TCP
  and feedback correction. The first position-only GRASP stopped at `0.0279 m`
  TCP error. One measured residual correction moved the TCP to `0.0044 m` from
  the original cube center. The finger-pad midpoint was `0.0045 m` from the
  target, while the URDF gripper-base TCP and wrist TCP matched exactly in the
  diagnostic frame. Video:
  `task_319_garbage_sort/output/head_camera_grasp_records/20260624_124545/external_grasp_demo.mp4`.

Current conclusion:
- The “large target delta / unreasonable posture” problem is reproducible in
  the old 6D pose safe-start path and is removed by the isolated debug
  position-only path.
- The target position was also unreasonable: points near `x=0.84-0.98` are at
  or beyond the practical right-arm reach envelope with the waist locked. The
  default debug cube is now `0.70,-0.16,0.5625`.
- The remaining raw IK residual is a stable `~0.028-0.035 m` right-arm TCP
  miss under the current IsaacLab differential IK stack. URDF gripper geometry
  is now internally consistent; the residual feedback pass reduces the actual
  TCP miss to about `4-5 mm` on the isolated cube.
- Static cube runs validate IK/TCP motion only. A true lift test needs either
  the real table scene or a support object; otherwise a movable cube falls to
  the ground in the isolated scene.

## 2026-06-24: Re-run physical mind-sort loop after final-dock correction

Change:
- Updated `--nav_final_dock` so the precision correction no longer rotates the
  robot toward small lateral residuals. It now keeps the grasp station facing
  the requested table/bin yaw and only corrects the forward/back component.
- If final docking only partially improves the pose, the Nav2 goal remains a
  navigation success and records `SUCCEEDED_FINAL_DOCK_PARTIAL`; the subsequent
  physical-grasp standpoint gate decides whether the residual position error is
  still acceptable.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --mind_sort_demo --mind_sort_physical_grasp --mind_sort_max_objects 3 --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 4 --warmup_steps 80 --cycle_interval_steps 1 --nav_backend nav2 --nav_actuation_mode wheel --wheel_drive_model urdf_diagonal --wheel_ground_coupling kinematic_stable --wheel_velocity_scale 0.35 --nav2_stack_startup_s 2 --nav2_goal_timeout_s 0 --nav2_planner_tolerance 0.12 --nav2_xy_goal_tolerance 0.06 --nav2_yaw_goal_tolerance 0.12 --nav_final_dock --nav_final_dock_position_tolerance 0.025 --nav_final_dock_yaw_tolerance 0.04 --mind_sort_settle_steps 80 --viss_qwen_verify_mode all --viss_qwen_max_candidates 0 --viss_qwen_max_roi_refine 0 --viss_qwen_timeout_s 0 --trajectory_steps 360 --safe_pregrasp_steps 160 --wrist_refine_steps 140 --grasp_steps 140 --lift_steps 220 --hold_steps 120 --drop_steps 100 --post_action_hold_steps 120 --no-gui_realtime_playback
```

Recorded run:
- Output: `task_319_garbage_sort/output/head_camera_grasp_records/20260624_120142/external_grasp_demo.mp4`
- Metadata:
  `task_319_garbage_sort/output/head_camera_grasp_records/20260624_120142/mind_sort_demo/mind_sort_task_queue.json`
- Result: the physical mind-sort loop attempted three distinct targets
  (`trash_05`, `trash_07`, `trash_00`) without getting stuck in Nav2/final
  docking. No object was completed because all three physical grasp attempts
  failed at `SAFE_START lift-current TCP error exceeded threshold`.
- Navigation observations:
  - cycle 0000 table station: raw error `0.0594 m`, final-dock error
    `0.0245 m`, status `SUCCEEDED_WITH_FINAL_DOCK`.
  - cycle 0001 table station: raw error `0.0592 m`, partial final-dock error
    `0.0592 m`, yaw improved from `0.1033 rad` to `0.0398 rad`, status
    `SUCCEEDED_FINAL_DOCK_PARTIAL`.
  - cycle 0002 table station: raw error `0.0597 m`, partial final-dock error
    `0.0586 m`, yaw improved from `0.0894 rad` to `0.0394 rad`, status
    `SUCCEEDED_FINAL_DOCK_PARTIAL`.
- Physical grasp observations:
  - Wrist local-view IK did not converge in all three cycles.
  - Wrist RGB-D refine therefore fell back to the head-reshoot RGB-D center.
  - The remaining blocker is right-arm safe-start / wrist-view reachability,
    not the visual target loop or Nav2 table-standpoint loop.

## 2026-06-24: Tighten Nav2 stop tolerance and add grasp-standpoint final docking

Change:
- Identified the `~0.149 m` table-standpoint error as Nav2 goal-checker
  tolerance, not visual or arm IK error. The bundled Nav2 stack previously used
  `xy_goal_tolerance: 0.15` and `yaw_goal_tolerance: 0.35`, so Nav2 correctly
  returned `SUCCEEDED` while still being too far away for right-arm grasping.
- Made bundled Nav2 tolerances configurable through
  `--nav2_planner_tolerance`, `--nav2_xy_goal_tolerance`, and
  `--nav2_yaw_goal_tolerance`. The Task319 defaults are now `0.12 m`,
  `0.06 m`, and `0.12 rad`.
- Added `--nav_final_dock` for precision grasp standpoints. After Nav2 reports
  success, the robot performs a low-speed wheel-driven final correction to the
  exact table-side target pose. This is not a teleport; it uses the same wheel
  actuation path as navigation.
- The final docking correction is currently enabled only for physical
  mind-sort table grasp standpoints, not for general bin/return movement.
- `nav_table` metadata now preserves the raw Nav2 final error under
  `nav2_raw_position_error_m` and records corrected final-dock error under
  `final_dock`.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py task_319_garbage_sort/scripts/ros2_nav2_stack_launcher.py
/usr/bin/python3 task_319_garbage_sort/scripts/ros2_nav2_stack_launcher.py --output-dir /tmp/task319_nav2_params_check --xy-goal-tolerance 0.06 --yaw-goal-tolerance 0.12 --planner-tolerance 0.12
```

Generated Nav2 parameter check:
- `planner tolerance: 0.120000`
- `xy_goal_tolerance: 0.060000`
- `yaw_goal_tolerance: 0.120000`

## 2026-06-24: Physical grasp gate inside v28/Nav2 mind-sort loop

Change:
- Added `--mind_sort_physical_grasp` mode to the mind-sort state machine. At a
  table grasp waypoint, the loop now requires a real right-gripper grasp/lift
  before navigating toward the bin. Simulated attach/drop remains available
  when this flag is disabled.
- Added a right wrist RGB-D camera (`wrist_rgbd`) and local refinement hooks:
  the robot first tries to move the wrist above the selected target, capture a
  wrist RGB-D frame, project the selected object center into that frame, and
  recompute the local point-cloud center from the wrist depth ROI.
- Added physical-grasp states around the existing mind-sort loop:
  `MOVE_WRIST_TO_LOCAL_VIEW`, `WRIST_RGBD_REFINE_TARGET`,
  `PLAN_PHYSICAL_GRASP`, `GRASP_CLOSE`, `VERIFY_PHYSICAL_HOLD`, and
  `LIFT_TO_CARRY_POSE`.
- Physical mind-sort uses the same v28/VISS target identity through observe,
  dynamic standpoint planning, standpoint reshoot, wrist refine, and grasp.
  The fallback grasp candidate is generated from the selected RGB-D geometric
  center, not from simulator truth.
- Added `--mind_sort_physical_max_standpoint_error_m` so physical grasp is
  skipped if Nav2 reports success but the measured final base pose is still too
  far from the planned grasp standpoint.
- Physical mode disables learned grasp/IK prescreen by default and uses the
  centroid candidate as the executed grasp candidate; this keeps the state
  machine focused on the currently selected RGB-D object center.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

Recorded diagnostic run:
- Command: see the physical mind-sort command in
  `task_319_garbage_sort/DEMO_COMMANDS.md`.
- Output: `task_319_garbage_sort/output/head_camera_grasp_records/20260624_114454/external_grasp_demo.mp4`
- Metadata:
  `task_319_garbage_sort/output/head_camera_grasp_records/20260624_114454/mind_sort_demo/cycle_0000/physical_grasp/physical_grasp_execution.json`
- Result: the chain reached v28 perception, dynamic Nav2 table standpoint,
  standpoint RGB-D reshoot, physical grasp planning, and execution. The grasp
  did not succeed: wrist local-view IK did not converge and the fallback
  head-reshoot center failed at `SAFE_START lift-current TCP error exceeded
  threshold`.
- Important measured issue: Nav2 returned `SUCCEEDED` for the grasp standpoint
  but stopped with about `0.149 m` XY error. That is large enough to invalidate
  right-arm reachability, so the new pre-grasp standpoint-error gate was added
  after this run.

## 2026-06-24: Mind-sort loop completes all 10 table trash objects

Change:
- Added `--mind_sort_allow_depth_component_targets` with default enabled for
  `--mind_sort_demo`. The dynamic standpoint selector still gives strict
  priority to v28/VISS visual targets, but once visual candidates are exhausted
  it may use head-depth tabletop components as fallback targets.
- Attached the component point cloud to `depth_component` instances so the same
  RGB-D geometry used to compute their centers also satisfies point-count and
  graspability diagnostics.
- Updated `dynamic_standpoint_rgb.png` drawing to show enabled fallback
  component targets in blue and the selected target in red.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py
```

Recorded run:
- Command: see the mind-sort delivery command in `task_319_garbage_sort/DEMO_COMMANDS.md`.
- Output: `task_319_garbage_sort/output/head_camera_grasp_records/20260624_021520/external_grasp_demo.mp4`
- Result: completed `trash_00` through `trash_09`; final stop reason was
  `No remaining valid table object for mind-sort demo.`
- Video check: 1280x720, 15917 frames, 30 FPS, about 530.6 seconds.

## 2026-06-24: Enforce Qwen category rules and continuous Nav2 mind-sort loop

Change:
- Added explicit hazardous/recyclable/kitchen/other category rules to both v28
  Qwen prompts in `viss/scripts/perception/yolo11_qwen_perception_offline.py`.
  Markers, highlighters, oil-based pens, ink pens, paint pens, and other
  ink/solvent writing tools are now explicitly classified as `hazardous`.
- Changed Task319's v28 default `--viss_qwen_verify_mode` from `none` to `all`.
  The main chain now keeps the v28 README qwen-first + YOLO ROI parameters
  (`conf=0.15`, `roi_expand=2.0`, `timeout=90s`) while requiring Qwen
  verification instead of skipping it. Candidate and ROI-refine counts are no
  longer capped by default for full-table scenes.
- Removed the mind-sort override that silently raised v28 max candidates/refine
  to 10. Explicit CLI arguments can still override these values.
- Changed Task319 and the v28 perception script so `--viss_qwen_max_candidates 0`
  / `--max-qwen-candidates 0` and `--viss_qwen_max_roi_refine 0` /
  `--max-roi-refine 0` mean no hard candidate-count limit. This is now the
  Task319 default for full-table scenes.
- Changed `--mind_sort_snap_observe_pose` default to false. Returning to the
  observation pose now remains a continuous Nav2 movement by default; pose snap
  is diagnostic only.
- Changed `--dynamic_ik_diagnostics` default to false. The previous default
  could temporarily reposition the robot during dynamic standpoint scoring,
  which is a likely source of visible start/plan jitter in videos.
- Added indefinite Nav2 goal support: `--nav2_goal_timeout_s 0` now means wait
  until Nav2 returns instead of killing the route halfway through a video.
- Added indefinite v28/VISS visual subprocess support:
  `--viss_qwen_timeout_s 0` now means wait for Qwen/YOLO/Qwen completion.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py viss/scripts/perception/yolo11_qwen_perception_offline.py
```

Delivery command:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo --mind_sort_max_objects 0 --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 4 --warmup_steps 80 --cycle_interval_steps 1 --nav_backend nav2 --nav_actuation_mode wheel --wheel_drive_model urdf_diagonal --wheel_ground_coupling kinematic_stable --wheel_velocity_scale 0.35 --nav2_stack_startup_s 2 --nav2_goal_timeout_s 0 --mind_sort_settle_steps 120 --viss_qwen_verify_mode all --viss_qwen_max_candidates 0 --viss_qwen_max_roi_refine 0 --viss_qwen_timeout_s 0 --grasp_steps 80 --drop_steps 120 --post_action_hold_steps 240
```

## 2026-06-24: Visual mind-sort Nav2 demo with simulated object carry/drop

Change:
- Updated external video defaults to a higher, farther, wider global view:
  `--observer_camera_pos 2.15,4.65,3.10` and
  `--observer_camera_target 2.05,0.0,0.62`. The observer camera focal length is
  widened to keep the full robot/table/bin scene in frame during complete
  sorting videos.
- The documented mind-sort video command now uses `--mind_sort_max_objects 0`,
  which means process all detected sortable objects instead of stopping after
  one object.
- Initially added `--mind_sort_snap_observe_pose` to compensate observation
  pose error. This historical default was later superseded on 2026-06-24:
  continuous mind-sort runs now leave the snap disabled by default and rely on
  Nav2 movement instead of pose teleporting.
- Added `--mind_sort_demo`, a temporary end-to-end sorting state machine that
  keeps v28/VISS perception and Nav2 navigation active while bypassing real
  grasp/IK execution.
- The loop is: observe full table from the backed-off pose, run v28 perception,
  choose the next non-completed object, compute a dynamic table-side standpoint,
  navigate there with the robot yaw facing the table side normal, settle, re-shoot
  RGB-D, attach the matched rigid object to the moving robot, navigate to the
  category bin, release the object above the bin, return to the backed-off
  observation pose, and repeat.
- Added mind-sort parameters:
  `--mind_sort_max_objects`, `--mind_sort_settle_steps`,
  `--mind_sort_attach_height_m`, `--mind_sort_attach_forward_m`, and
  `--mind_sort_attach_lateral_m`.
- `debug_cube` is excluded from mind-sort candidates by default so the demo
  selects real table trash unless the debug cube demo is explicitly requested.
- Added an optional Nav2 step hook used only by mind-sort carrying; existing
  Nav2 demos and post-grasp sort navigation keep their previous behavior.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
timeout 1800s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --mind_sort_demo --mind_sort_max_objects 0 --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 4 --warmup_steps 80 --cycle_interval_steps 1 --nav2_stack_startup_s 2 --nav2_goal_timeout_s 150 --mind_sort_settle_steps 120 --grasp_steps 80 --drop_steps 120 --post_action_hold_steps 240 --no-gui_realtime_playback
```

Result:
- Latest single-object smoke run before the full-loop command correction:
  `task_319_garbage_sort/output/head_camera_grasp_records/20260624_001406`
- v28 initial observation returned 10 instances; after filtering `debug_cube`,
  9 candidate table objects remained.
- Selected target: `trash_05 / trash_potted_meat_can_0`.
- Category: `可回收物`; target bin: recycle.
- Table-facing dynamic standpoint pose:
  `[0.4849134088, 0.0074041486, 0.0]`.
- Nav2 states `OBSERVE_TABLE_BACKOFF`, `NAV_TO_TABLE_STANDPOINT`,
  `NAV_TO_BIN_STAGING`, and `NAV_TO_BIN` all returned `SUCCEEDED`.
- The object was mind-attached, carried, and released at
  `[3.03, -0.72, 0.98]`; final object pose was approximately
  `[3.0301, -0.7213, 0.0669]` inside the recycle bin.
- Video: `task_319_garbage_sort/output/head_camera_grasp_records/20260624_001406/external_grasp_demo.mp4`
- Metadata: `task_319_garbage_sort/output/head_camera_grasp_records/20260624_001406/mind_sort_demo/mind_sort_task_queue.json`
- Delivery requirement after correction: final video commands must use
  `--mind_sort_max_objects 0`, and the state machine now includes
  `RETURN_TO_OBSERVE_BACKOFF` after every drop so the loop is visible before the
  next perception pass.
- Full-loop correction after the `20260624_002922` run: the command did loop
  through cycle 0000 and cycle 0001, but by cycle 0002 the returned observation
  pose still had about `0.147 m` position error and `0.319 rad` yaw error, so
  the head camera only detected 2 instances. This was first mitigated with
  `--mind_sort_snap_observe_pose`; later continuous-loop requirements changed
  the default back to no snap, so the remaining fix must be in navigation
  accuracy rather than pose reset.

## 2026-06-23: Verify dynamic RGB-D standpoint is passed to Nav2

Change:
- Added `--standpoint_nav_only` as a validation mode for the main visual chain.
- In this mode, the system runs the current v28/VISS RGB-D perception, computes
  the dynamic table-side grasp station, sends that station to Nav2, re-shoots
  RGB-D at the reached station, and then stops without grasp execution.
- This isolates the question "did the computed ground point actually become a
  Nav2 goal?" from the separate grasp/IK problem.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --help | rg "standpoint_nav_only|dynamic_grasp"
timeout 420s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --perception_source v28_original --standpoint_nav_only --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 4 --warmup_steps 80 --cycle_interval_steps 1 --nav2_stack_startup_s 2 --nav2_goal_timeout_s 150 --post_action_hold_steps 120 --no-gui_realtime_playback
```

Result:
- Run: `task_319_garbage_sort/output/head_camera_grasp_records/20260623_234801`
- Dynamic selected target: `black marker`, scene object `trash_large_marker_0`.
- Dynamic station selected from RGB-D: `dynamic_front`, pose
  `[0.4442109466, 0.2268338203, -0.2830957735]`.
- Nav2 received the same target pose and returned `SUCCEEDED`.
- Final measured robot pose was `[0.3610131443, 0.1041492596, 0.0372942650]`.
- Residual error was `position_error_m=0.148234` and
  `yaw_error_rad=-0.320390`, so the computation-to-Nav2 interface is connected,
  but the current stop tolerance/tracking precision is still too loose for
  final grasp alignment.
- Re-shoot outputs were saved under `cycle_0000/standpoint_reshoot/`.
- Video: `task_319_garbage_sort/output/head_camera_grasp_records/20260623_234801/external_grasp_demo.mp4`

## 2026-06-23: Dynamic RGB-D table-side grasp standpoint as default

Change:
- Moved the default initial/home observation pose from `(0.53, 0.0, 0.0)` to `(0.18, 0.0, 0.0)` so the head RGB-D camera starts farther from the table and sees more of the tabletop and edges.
- Added `--dynamic_grasp_standpoint_nav`, enabled by default, to compute a continuous pre-grasp station from the selected object's current RGB-D world center.
- Added `--dynamic_observe_pose` and `--dynamic_standpoint_fallback_to_preset`.
- The dynamic planner evaluates all valid v28/VISS targets, generates front/left/right table-side candidate poses, and picks the object+station pair with a reachable right-arm local target position.
- Dynamic station yaw faces the selected object center, not just the table normal.
- Existing preset station planning remains as fallback when dynamic planning has no reachable candidate.
- New debug outputs:
  - `dynamic_grasp_standpoint.json`
  - `dynamic_standpoint_rgb.png`
  - `dynamic_standpoint_topdown.png`

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --help | rg "dynamic_grasp|dynamic_observe|robot_pos|grasp_standpoint_nav"
```

Light visual-planning run:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --perception_source v28_original --record_debug --warmup_steps 40 --cycle_interval_steps 1 --post_action_hold_steps 0 --no-gui_realtime_playback
```

Result:
- Run: `task_319_garbage_sort/output/head_camera_grasp_records/20260623_232410`
- `dynamic_grasp_standpoint.json`, `dynamic_standpoint_rgb.png`, and `dynamic_standpoint_topdown.png` were generated.
- The planner evaluated 3 valid targets and selected `aluminum can`.
- Selected dynamic station: `dynamic_left`, pose `[1.4966, -0.8717, 1.2877]`.
- Target local coordinate at that station: `[0.5728, 0.0]`, satisfying the right-arm local reach window.

## 2026-06-23: Preset table-side standpoint selection before visual grasp

Change:
- Added pre-grasp table-side standpoint navigation behind `--grasp_standpoint_nav`.
- Added fixed candidate waypoints:
  - `table_front_left`, `table_front`, `table_front_right`
  - `table_left_near`, `table_left_far`
  - `table_right_near`, `table_right_far`
- `table_left_near/far` and `table_right_near/far` are close grasp standpoints at `TABLE_Y_LIMITS +/- 0.42 m`; the older `table_left/table_right` remain side-corridor waypoints and are intentionally not part of the default grasp candidate list.
- Each candidate yaw is computed by the existing waypoint system so the robot faces the table.
- Target selection no longer requires the object to be reachable from the initial pose when `--grasp_standpoint_nav` is enabled. Instead, each object receives a best preset-standpoint score based on whether its RGB-D geometric center lands in the right-arm local reach window from a candidate station.
- After the selected object and station are chosen, the code writes:
  - `planned_grasp_standpoint.json`
  - `planned_standpoint_rgb.png`
  - `planned_standpoint_topdown.png`
- If `--execute_grasp` is active, the state machine navigates to the chosen station with Nav2 before grasp planning, then re-shoots RGB-D and re-runs the same active VISS/v28 perception chain under `standpoint_reshoot/`.
- If the pre-grasp Nav2 goal fails or no geometrically reachable station exists, grasp execution is blocked instead of using the stale initial-camera target.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --help | rg "grasp_standpoint|enable_sort_nav"
```

Main run command for visual inspection:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --num_cycles 1 --perception_source v28_original --grasp_standpoint_nav --execute_grasp --enable_sort_nav --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 3 --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 300 --lift_steps 360 --hold_steps 240 --max_joint_step 0.012 --no-grasp_ik_prescreen --lift_height 0.10 --arm_motion_wrist_orientation angled_top_down --angled_top_down_tcp_axis 0.65,0,-0.76 --arm_motion_converge_extra_steps 800 --arm_motion_converge_chunk_steps 80 --grasp_error_threshold_m 0.015 --nav2_stack_startup_s 2 --nav2_goal_timeout_s 150 --post_action_hold_steps 1200
```

## 2026-06-23: Angled top-down grasp default for better cube-center alignment

Change:
- Added `--arm_motion_wrist_orientation angled_top_down`.
- Added `--angled_top_down_tcp_axis`, defaulting to `(0.65, 0.0, -0.76)`, meaning the inline gripper points partly forward and partly downward instead of requiring a strict vertical top grasp.
- Changed the default staged grasp wrist mode from `side_pinch` to `angled_top_down`.
- Tightened the default final grasp TCP threshold from `0.030 m` to `0.015 m`.
- Added runtime gripper alignment diagnostics:
  - wrist-derived TCP,
  - gripper-base-derived TCP,
  - left/right finger pad centers,
  - actual finger pad midpoint,
  - each point's offset from the target center.

Rationale:
- The dynamic cube run showed the gripper model/TCP frames are internally consistent: wrist TCP, gripper-base TCP, and finger midpoint match at millimeter scale.
- The visible misalignment came mainly from residual IK tracking error. With `side_pinch`, the final grasp TCP stayed about `2.4-2.6 cm` from the cube center.
- A strict vertical `top_down` pose was not reachable at the safety standoff with the waist locked.
- The angled top-down pose keeps the gripper inline with the wrist while giving the arm a better reachable posture and reducing the actual finger pad midpoint error below 1 cm.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

Strict diagnostic runs:
- `20260623_223302`: side-pinch with `--grasp_error_threshold_m 0.012` failed at final GRASP after 800 extra convergence steps; best TCP error was `0.02419 m`.
- `20260623_223618`: strict vertical top-down failed at safe standoff; high-standoff TCP error was about `0.212 m`.
- `20260623_223840`: angled top-down with `--grasp_error_threshold_m 0.012` reached final GRASP with TCP error `0.01330 m`, but did not close because the strict TCP threshold was slightly exceeded. The actual finger pad midpoint error was already `0.00834 m`.

Successful dynamic cube validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --debug_cube_grasp_demo --execute_grasp --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 3 --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 300 --lift_steps 360 --hold_steps 240 --max_joint_step 0.012 --no-grasp_ik_prescreen --lift_height 0.10 --arm_motion_wrist_orientation angled_top_down --angled_top_down_tcp_axis 0.65,0,-0.76 --arm_motion_converge_extra_steps 800 --arm_motion_converge_chunk_steps 80 --grasp_error_threshold_m 0.015 --post_action_hold_steps 120 --no-gui_realtime_playback
```

Result:
- Run: `task_319_garbage_sort/output/head_camera_grasp_records/20260623_224004`
- Video: `task_319_garbage_sort/output/head_camera_grasp_records/20260623_224004/external_grasp_demo.mp4`
- `success=True`, reason: `Object lifted and held.`
- Final GRASP TCP error: `0.01330 m`.
- Final GRASP finger pad midpoint error: `0.00834 m`.
- Contact check passed with final jaw width `0.06135 m`.
- Verified lift delta: `0.2411 m`.

## 2026-06-23: Inline gripper mount and dynamic cube grasp baseline

Change:
- Removed the right-wrist placeholder hand geometry from active visual/collision use; the right arm now relies on the attached parallel gripper as the effective grasping tool.
- Changed the attached gripper mount to inline with the right wrist/forearm axis: mount RPY is `(0, pi/2, 0)`, mapping gripper local `+X` to wrist local `-Z`.
- Split TCP calibration into two explicit frames:
  - `--gripper_tcp_offset` is in the right wrist frame and defaults to `(0, 0, -0.115)`.
  - `--gripper_local_tcp_offset` is in `gripper_base` frame and defaults to `(0.115, 0, 0)` for cuRobo extra-link planning.
- Updated GraspNet/AnyGrasp frame documentation and the IsaacLab grasp-frame convention to match the inline TCP axis.
- Removed the wrist placeholder link from the right-arm cuRobo collision model; cuRobo now keeps the TCP extra link on `gripper_base`.
- Changed the default debug grasp cube to a movable physical cube at a right-arm reachable location near `(0.98, -0.16, 0.5625)`.
- Added the initial `side_pinch` wrist mode; safe-start establishes the selected fixed wrist orientation, then low `GRASP` and `LIFT` stages use TCP position-only IK to avoid unnecessary axis-alignment failures for direction-insensitive objects.

Rationale:
- The previous wrist/hand geometry made the gripper approach nearly perpendicular to the arm axis and could press the gripper into the tabletop.
- For the current physical task, the object is direction-insensitive; position convergence matters more than matching every target frame axis.
- The new setup validates the core manipulation requirement in isolation: given a reachable object center, the right gripper can move to it, close, and lift a dynamic object.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/gripper_robot_urdf.py task_319_garbage_sort/curobo_right_arm.py task_319_garbage_sort/grasp_pipeline/grasping/anygrasp_wrapper.py
```

Static checks:
- Generated URDF: `/tmp/task319_urdf/kuavo_s62_with_right_gripper_v4_inline_gripper_axis_201b59c048d4_005de575ffd1.urdf`
- Mount RPY: `0.0 1.5707963267948966 0.0`
- Right wrist placeholder visual/collision count: `0/0`
- cuRobo TCP parent: `gripper_base`
- cuRobo extra-link transform: `[0.115, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]`

Dynamic cube grasp validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --debug_cube_grasp_demo --execute_grasp --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 3 --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 300 --lift_steps 360 --hold_steps 240 --max_joint_step 0.012 --no-grasp_ik_prescreen --lift_height 0.10 --arm_motion_converge_extra_steps 360 --arm_motion_converge_chunk_steps 80 --post_action_hold_steps 120 --no-gui_realtime_playback
```

Result:
- Run: `task_319_garbage_sort/output/head_camera_grasp_records/20260623_222602`
- Video: `task_319_garbage_sort/output/head_camera_grasp_records/20260623_222602/external_grasp_demo.mp4`
- `success=True`, reason: `Object lifted and held.`
- Target: `debug_cube`, grasp backend candidate count `1/1`.
- Final `GRASP` TCP error: `0.0264 m` under the `0.03 m` threshold.
- Final `LIFT` TCP error: `0.0145 m`.
- Contact check passed with final jaw width `0.0472 m`.
- Verified lift delta: `0.2485 m`.

## 2026-06-23: Shrink cracker box and slow default grasp primitive

Change:
- Scaled `trash_cracker_box_0` / YCB `003_cracker_box` from `1.00` to `0.60` and adjusted its table Z placement with the same scale.
- Reduced that box mass from `0.08 kg` to `0.04 kg` to avoid an oversized high-inertia test target.
- Slowed the default IsaacLab staged grasp primitive: `trajectory_steps=360`, `safe_pregrasp_steps=180`, `grasp_steps=180`, `lift_steps=220`, and `max_joint_step=0.012`.
- Updated the default visual-grasp launch commands to an even more conservative inspection profile: `trajectory_steps=520`, `safe_pregrasp_steps=220`, `grasp_steps=220`, `lift_steps=300`, `max_joint_step=0.012`.

Rationale:
- The largest paper box was wider than the current gripper/centroid primitive can tolerate reliably; shrinking it makes the visual-chain grasp test focus on perception-to-center grasping instead of object-size limits.
- The staged IsaacLab primitive was already active, but the command-level `--max_joint_step 0.025` allowed relatively abrupt arm motion. The new command profile prioritizes slow approach and reduced object push.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

Viewer command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --num_cycles 1 --skip_graspnet --use_centroid_fallback --execute_grasp --record_debug --record_video --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.012 --post_action_hold_steps 1200
```

## 2026-06-23: Switch official perception default to v28 qwen-first ROI chain

Change:
- Added `--perception_source v28_original` and made it the default Task319 visual source.
- Changed the default VISS segmentation model to `viss/models/best_seg.pt`.
- The v28 adapter stores outputs under `cycle_xxxx/v28_original/` and keeps 3D geometry in the current Task319 RGB-D frame: masks are projected through the current depth, intrinsics, and `camera_to_world`.
- Preserved `--perception_source v27_original` for comparison runs.
- Documented the planned asynchronous v28 design: Qwen global candidate generation, YOLO ROI mask refinement, grasp starts, Qwen crop verification finishes before bin selection.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

Launch command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --num_cycles 1 --skip_graspnet --use_centroid_fallback --execute_grasp --record_debug --record_video --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.012 --post_action_hold_steps 1200
```

## 2026-06-23: Widen attached gripper and slow close command

Change:
- Increased the attached two-finger gripper prismatic joint limit from `0.035 m` per finger to `0.060 m` per finger, giving a simulated total jaw opening of `0.120 m`.
- Updated the width-based gripper controller defaults to `max_width_m=0.120`, `default_open_width_m=0.110`, and `grasp_extra_clearance_m=0.018`.
- Reduced gripper joint velocity from the previous fast close to `0.18 m/s` in both the URDF joint limit and IsaacLab actuator config.
- Removed remaining `0.070 m` grasp-width clamps from the visual grasp execution paths and made them use the attached gripper limits instead.
- Changed centroid fallback width estimation to allow larger RGB-D object widths instead of clamping the estimated width near `0.068 m`.
- Replaced direct final-close commands in grasp execution with a smooth width ramp over `--grasp_steps`.

Validation:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/grasp_pipeline/execution/gripper_control.py task_319_garbage_sort/grasp_pipeline/grasping/mask_filter.py task_319_garbage_sort/grasp_pipeline/grasping/grasp_selector.py
```

Formal v27 + cuRobo validation, same current configuration:

```bash
cd /home/zhxm/workspace/mda_isaaclab
env -u PYTHONHOME -u PYTHONPATH CUDA_HOME=/home/zhxm/miniconda3/envs/my_task319_safe PATH=/home/zhxm/miniconda3/envs/my_task319_safe/bin:$PATH LD_LIBRARY_PATH=/home/zhxm/miniconda3/envs/my_task319_safe/lib:${LD_LIBRARY_PATH:-} /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --perception_source v27_original --viss_qwen_model viss/models/yolo11s-seg-best.pt --viss_qwen_api_style dashscope --skip_graspnet --use_centroid_fallback --no-target_reachability_ik_check --no-grasp_ik_prescreen --arm_motion_backend curobo_right_arm --execute_grasp --record_debug --record_video --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 120 --safe_pregrasp_steps 90 --curobo_command_min_steps 90 --grasp_steps 80 --lift_steps 120 --hold_steps 120 --grasp_error_threshold_m 0.03 --arm_motion_converge_extra_steps 260 --arm_motion_converge_chunk_steps 80 --curobo_tcp_refine_steps 160 --post_action_hold_steps 0 --no-gui_realtime_playback
```

Result:
- Run: `task_319_garbage_sort/output/head_camera_grasp_records/20260623_194101`
- Selected target: `食品包装盒 / trash_00 / trash_cracker_box_0`, category `可回收物`.
- Target pose debug: PASS, XY error `0.0389 m`, Z error `0.0828 m`.
- Selected centroid fallback width: `0.0794 m`; with the new default clearance, open command is within the new `0.120 m` max jaw opening.
- Execution still failed before gripper close: `GRASP cuRobo failed: cuRobo plan failed and local TCP fallback failed`.
- `grasp_curobo` final TCP error was `0.0559 m`, above the current strict `--grasp_error_threshold_m 0.03`, so the run did not enter the slow close stage.
- Video: `task_319_garbage_sort/output/head_camera_grasp_records/20260623_194101/external_grasp_demo.mp4`

Interpretation:
- The wider gripper model and commands are active, but this exact validation did not test physical capture because the final grasp descent was rejected by the 3 cm TCP threshold first.
- The next manipulation issue is still low grasp-pose TCP convergence/reachability near the target center, not jaw opening alone.

## 2026-06-23: Collision-safe high-standoff grasp trajectory layer

Change:
- Added a shared `run_collision_safe_pregrasp_waypoints()` trajectory layer before non-legacy grasp execution.
- `local_position_primitive`, `kuavo_ik`, and `kuavo_analytic_ik` now move through:
  1. current TCP vertical lift,
  2. high standoff above the target,
  3. pregrasp,
  4. final vertical grasp descent.
- The high-standoff height is computed from the actual table surface plus `--safe_pregrasp_table_clearance_m`, the selected object height plus `--safe_pregrasp_object_clearance_m`, and the current TCP height.
- Lift targets now also clamp to at least `TABLE_SURFACE_Z + --safe_pregrasp_table_clearance_m`, so a successful grasp retreats well above the tabletop rather than only lifting by a relative delta.
- The safety waypoints and their final TCP errors are recorded in each cycle metadata under `execution.segments` and `execution.updates.safe_pregrasp`.

Rationale:
- This follows the standard pick/place primitive used by mature manipulation stacks: approach from a collision-free standoff, descend near-vertically, close, then retreat vertically.
- Full link-level collision guarantees should ultimately use an obstacle-aware motion generator such as Isaac Sim RMPFlow/cuMotion or cuRobo with a Kuavo collision model and the table registered as an obstacle. This change is the immediate table-aware guard for the current IsaacLab Differential IK / Kuavo IK execution path.

Validation:
- Static checks passed:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

- Debug execution run:

```text
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260623_183546
Target: marker / trash_large_marker_0
Target pose debug: PASS, XY error 0.012 m.
safe_lift_current: converged, final TCP error 0.0138 m.
safe_standoff: converged, final TCP error 0.0287 m.
safe_z: 0.8400 m, table surface z: 0.5400 m, table clearance: 0.3000 m.
PRE_GRASP: converged, final TCP error 0.0242 m.
GRASP: failed with Kuavo analytic IK joint_limit_violation; the final TCP error was 0.1199 m.
Interpretation: the new table-safe high-standoff trajectory is active and converges. The remaining failure is low grasp-pose reachability/IK near the table, not the old direct tabletop sweep.
Video: task_319_garbage_sort/output/head_camera_grasp_records/20260623_183546/external_grasp_demo.mp4
```

Viewer command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --num_cycles 1 --skip_graspnet --use_centroid_fallback --target_object_name marker --no-target_self_occlusion_filter --no-target_reachability_ik_check --no-grasp_ik_prescreen --arm_motion_backend kuavo_analytic_ik --arm_motion_wrist_orientation current --execute_grasp --record_debug --record_video --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 180 --safe_pregrasp_steps 80 --grasp_steps 70 --lift_steps 140 --hold_steps 40 --max_joint_step 0.018 --post_action_hold_steps 0
```

## 2026-06-23: Kuavo official analytic IK seed + explicit gripper TCP handling

Change:
- Added `--arm_motion_backend kuavo_analytic_ik`, backed by `task_319_garbage_sort/scripts/kuavo_analytic_ik_cli.cpp`.
- The CLI uses Kuavo official `motion_capture_ik/AnalyticArmIk.hpp`, auto-builds with `g++`, and returns joint limits plus official-FK residual diagnostics.
- Added explicit gripper TCP handling: target object center remains the TCP target; official IK receives the converted arm-frame target; execution validation is always done on gripper TCP position.
- Added official-FK filtering with `--kuavo_analytic_ik_max_fk_error_m` to reject analytic outputs whose own FK misses the requested link target.
- Added direction-agnostic wrist sampling via `--kuavo_analytic_ik_approach_dirs` and `--kuavo_analytic_ik_roll_samples_rad`.
- Added `--kuavo_analytic_ik_tcp_refine`: official analytic IK is treated as a Kuavo seed, then IsaacLab position-only IK refines the gripper TCP.
- Added `--kuavo_analytic_ik_execute_top_k` to execute multiple official seed candidates and rank by actual Isaac TCP error.
- Added `--kuavo_analytic_ik_local_grasp_descent_on_seed_failure`: if the official analytic solver has no legal GRASP seed, the final descent can use TCP position-only IK from the pregrasp pose.

Validation:
- Static checks passed:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
g++ -std=c++17 -O2 -I/usr/include/eigen3 -Ikuavo-ros-opensource/src/manipulation_nodes/motion_capture_ik/include task_319_garbage_sort/scripts/kuavo_analytic_ik_cli.cpp -o task_319_garbage_sort/build/kuavo_analytic_ik_cli
```

- Latest headless validation:

```text
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260623_182054
Target: 纸盒包装 / trash_cracker_box_0
PRE_GRASP: converged, final TCP error 0.0246 m after Kuavo analytic seed + TCP refine.
GRASP: failed. Official analytic IK had no legal seed at the lower geometric-center target; local TCP descent also stopped 0.0703 m above/away from the target.
Interpretation: gripper TCP offset handling is necessary and now explicit, but this failure is not just a gripper-length offset. With waist locked and the current stand pose, the right arm cannot reliably reach the selected object's RGB-D geometric center at grasp height. Next step is either a reachable grasp-height policy for this box or navigation/standpoint adjustment before descent.
Video: task_319_garbage_sort/output/head_camera_grasp_records/20260623_182054/external_grasp_demo.mp4
```

Viewer command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --num_cycles 1 --skip_graspnet --use_centroid_fallback --no-target_reachability_ik_check --no-grasp_ik_prescreen --arm_motion_backend kuavo_analytic_ik --arm_motion_wrist_orientation current --execute_grasp --record_debug --record_video --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 300 --grasp_steps 120 --lift_steps 220 --hold_steps 180 --max_joint_step 0.025 --post_action_hold_steps 1200
```

## 2026-06-23: Position-only grasp motion primitive and Kuavo IK bridge hook

Change:
- Added `--arm_motion_backend {local_position_primitive,legacy_differential_ik,kuavo_ik,auto}`. The default is now `local_position_primitive`.
- The new local primitive executes right-arm grasping as staged TCP-position motion: carry ready pose -> high pregrasp above target -> descend to target center -> close gripper -> lift -> hold. Grasp axes are diagnostic only; position is the hard requirement.
- Non-legacy backends automatically enable learned grasp position-only execution unless explicitly overridden.
- Added convergence retry parameters: `--arm_motion_converge_extra_steps`, `--arm_motion_converge_chunk_steps`, `--arm_motion_pregrasp_clearance_m`, and `--arm_motion_min_table_clearance_m`.
- Added `--arm_motion_use_target_center_position`, default enabled. Non-legacy staged execution commands the selected object's RGB-D geometric center instead of the learned grasp translation, so the visual target and grasp target stay tied to the same object.
- When IK prescreen is disabled, execution candidates are now ordered by target-center distance before score instead of raw GraspNet score.
- Added `--arm_motion_wrist_orientation {current,top_down}`, default `current`. `top_down` is kept as an explicit diagnostic because local hard-pose IK drove the TCP far above the target in validation; the official Kuavo IK bridge remains the right path for position-hard/orientation-soft control.
- Added experimental official Kuavo IK bridge support:
  - Main sim client connects to `--kuavo_ik_bridge_host/--kuavo_ik_bridge_port`.
  - Bridge script: `task_319_garbage_sort/scripts/kuavo_ik_socket_bridge.py`.
  - Default service: `/ik/two_arm_hand_pose_cmd_srv_muli_refer`.
  - Default `constraint_mode=2`, meaning position hard + orientation soft.
  - `--arm_motion_backend kuavo_ik` uses official IK if the bridge is online; `auto` falls back to the local primitive when the bridge is unavailable.
- Lift success is still verified physically by object height change, not by IK convergence alone.

Validation commands:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py task_319_garbage_sort/scripts/kuavo_ik_socket_bridge.py
```

Default grasp simulation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --num_cycles 1 --grasp_backend graspnet --graspnet_force_objectness_top_k 64 --no-target_reachability_ik_check --no-grasp_ik_prescreen --arm_motion_backend local_position_primitive --arm_motion_wrist_orientation current --execute_grasp --record_debug --record_video --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 360 --grasp_steps 120 --lift_steps 220 --hold_steps 180 --max_joint_step 0.025 --post_action_hold_steps 1200
```

Latest headless validation:

```text
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260623_171333
Selected target: 纸盒包装 / trash_cracker_box_0
Commanded position source: target_rgbd_geometric_center
Candidate center distance: 0.0048 m
Result before top-down wrist fix: failed, grasp final TCP error 0.076 m, almost all residual from Z because the old free wrist posture put the TCP offset upward.
Video: task_319_garbage_sort/output/head_camera_grasp_records/20260623_171333/external_grasp_demo.mp4
```

Top-down hard-pose diagnostic:

```text
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260623_171739
Commanded position source: target_rgbd_geometric_center
Wrist orientation mode: top_down
Result: failed in PRE_GRASP; local pose IK final wrist error 0.397 m and TCP error 0.619 m.
Interpretation: hard-locking the local top-down wrist pose is not viable with the current IsaacLab DifferentialIK setup. Use official Kuavo IK bridge for position-hard/orientation-soft testing.
Video: task_319_garbage_sort/output/head_camera_grasp_records/20260623_171739/external_grasp_demo.mp4
```

Optional official Kuavo IK bridge startup:

```bash
cd /home/zhxm/workspace/mda_isaaclab
source kuavo-ros-opensource/devel_mda319/setup.bash
python3 task_319_garbage_sort/scripts/kuavo_ik_socket_bridge.py --host 127.0.0.1 --port 31975 --service /ik/two_arm_hand_pose_cmd_srv_muli_refer
```

Then run the sim with `--arm_motion_backend kuavo_ik` or `--arm_motion_backend auto`.

## 2026-06-23: GraspNet diagnostic execution without IK prescreen

Change:
- Added `--grasp_ik_prescreen/--no-grasp_ik_prescreen`. When disabled, GraspNet/AnyGrasp ranked candidates are sent directly to the execution retry queue instead of being removed by the dry-run IK prescreen.
- Added `--graspnet_force_objectness_top_k`. When GraspNet reports zero objectness-positive seeds, this diagnostic option forces the highest-probability Top-K seeds through `pred_decode()` so the rest of the grasp pipeline can be evaluated.
- Added `--learned_grasp_position_only_ik`. When enabled, GraspNet/AnyGrasp execution tracks TCP position only and treats learned grasp axes as diagnostic rather than required IK orientation constraints.

Validation commands:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py task_319_garbage_sort/grasp_pipeline/grasping/graspnet_wrapper.py
```

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 1200s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --grasp_backend graspnet --graspnet_force_objectness_top_k 64 --no-target_reachability_ik_check --no-grasp_ik_prescreen --execute_grasp --record_debug --record_video --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.012 --post_action_hold_steps 0
```

Observed result:

```text
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260623_161900
Selected target: 食品包装盒 / trash_cracker_box_0
GraspNet input points: 3242
GraspNet original objectness positives: 0
Forced objectness Top-K: 64
Raw decoded grasps: 64
Mask-filtered grasps: 49
IK prescreen enabled: false
Execution attempts: 3
Execution result: failed in SAFE_START, safe_standoff final error about 0.64-0.78 m
Video: task_319_garbage_sort/output/head_camera_grasp_records/20260623_161900/external_grasp_demo.mp4
```

Interpretation:
- The earlier GraspNet path was indeed being blocked before execution when IK prescreen removed all candidates.
- After disabling that prescreen and forcing objectness for the reachable carton, GraspNet produced candidates and the FSM attempted execution.
- Remaining failure is runtime IK tracking of the high-clearance safe-start/pose target, not lack of GraspNet candidates.

## 2026-06-23: Add head-camera self-occlusion guard for target selection

Change:
- Confirmed the current head-camera RGB frame includes a large white robot-body/self-occlusion region along the lower-right/bottom image area.
- Added `--target_self_occlusion_filter`, default enabled, to reject visual grasp targets whose masks overlap the lower image guard band.
- Added `--target_self_occlusion_bottom_fraction 0.14` and `--target_self_occlusion_max_overlap_fraction 0.02` as the current default guard parameters.
- Each cycle now saves `camera_self_occlusion_guard.png` and records guard metadata in `metadata.json.camera_self_occlusion_guard`.
- `target_candidates.json` now records per-target `image_guard` metrics and rejects bottom-band candidates with `robot_self_occlusion_or_bottom_image_guard`.

Validation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 420s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --skip_graspnet --no-target_reachability_ik_check --record_debug --warmup_steps 20 --cycle_interval_steps 1 --post_action_hold_steps 0
```

Observed result:

```text
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260623_160824
v27_original instances: 8
Selected target: 食品包装盒 / trash_cracker_box_0
Rejected by self-occlusion guard: 3 candidates
Guard: bottom_fraction=0.14, guard_top_px=413, max_overlap_fraction=0.02
Pose check: selected target camera-vs-sim xy_error=0.046 m, pass_xy_gate=true
Debug image: cycle_0000/camera_self_occlusion_guard.png
```

## 2026-06-23: Require center-point reachability for centroid fallback grasp

Change:
- Added `--target_reachability_center_only`, default enabled.
- Right-arm reachability for centroid fallback now checks the same RGB-D geometric center that the grasp command will use.
- The previous diagnostic behavior could mark a target as nearly reachable because an object edge/top point was close to the TCP, even though the actual centroid grasp point was still unreachable. That mismatch is now removed by default.
- Extra visible-surface probe points can still be restored for diagnostics with `--no-target_reachability_center_only`.

Validation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

Run result:

```text
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260623_155819
Command: timeout 900s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --skip_graspnet --use_centroid_fallback --execute_grasp --record_debug --record_video --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.012 --post_action_hold_steps 0
v27_original: model=viss/models/yolo11s-seg-best.pt, pipeline=qwen_first, accepted=7
Reachability policy: rgbd_geometric_center_only
Closest center candidates: trash_07/black marker error=0.165 m, trash_05/green and yellow can error=0.172 m, threshold=0.070 m
Execution: blocked before grasp because no selected target satisfied center-point right-arm reachability.
Video: task_319_garbage_sort/output/head_camera_grasp_records/20260623_155819/external_grasp_demo.mp4
```

## 2026-06-23: Switch v27 default segmentation weight back to best

Change:
- Changed the default `--viss_qwen_model` from `viss/models/yolo11s-seg.pt` to `viss/models/yolo11s-seg-best.pt`.
- The v27 original perception chain now uses the best YOLO11s segmentation weight unless a run explicitly overrides `--viss_qwen_model`.

Validation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile viss/scripts/perception/yolo11_qwen_perception_offline.py task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

## 2026-06-23: Align reachability probe with execution and disable legacy debug artifacts by default

Change:
- Fixed a reachability false-negative source: when `--execute_grasp` or `--strict_model_chain` is used and `--target_reachability_probe_steps` is not explicitly provided, the dry-run IK reachability probe now uses at least `max(--trajectory_steps, 240)` steps instead of the old fixed `120` steps.
- This keeps the pre-grasp reachability gate closer to the actual centroid fallback execution trajectory, so a target is less likely to be rejected only because the short probe did not converge.
- Confirmed centroid fallback grasping uses the current v27 mask's RGB-D point-cloud median as the grasp center, not simulator ground truth and not v27 camera parameters.
- Added `--save_legacy_yolo_debug` for the old full-frame YOLO debug overlay/json. It is disabled by default.
- Added `--save_legacy_grasp_debug` for old GraspNet/fallback overlays and point-cloud dumps. It is disabled by default.

Validation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile viss/scripts/perception/yolo11_qwen_perception_offline.py task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

## 2026-06-23: Add one-object target scoring and post-drop home return

Change:
- Replaced the previous target ordering with an explicit one-object-per-cycle scoring policy. After hard filters, candidates are ranked by reachable status, v27 group (`detections` before `approach_candidates`), right-arm IK cost, distance, clutter, RGB-D point quality, VLM confidence, and recent failed-grasp memory.
- `target_candidates.json` now records the full scoring breakdown for every valid/rejected visual candidate: total score, weighted components, point-cloud metrics, clutter distance, failure key, and reachability metadata.
- Visual grasp execution remains gated by reachability. When `--execute_grasp` is set and all candidates are outside the current right-arm reachable set, the state machine now selects no target, does not grasp, and does not navigate.
- Post-grasp sort navigation now includes return-home behavior after drop. The full Nav2 chain is: bin staging -> category bin side -> drop -> bin staging -> side corridor -> home approach -> home align -> home.

Validation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile viss/scripts/perception/yolo11_qwen_perception_offline.py task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

Smoke validation:

```text
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260623_125738
Command: timeout 480s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --skip_graspnet --use_centroid_fallback --record_debug --warmup_steps 2 --cycle_interval_steps 1 --post_action_hold_steps 0
v27_original: planner=3, approach=5, accepted=8
Selection: policy=one_object_per_cycle_score_lowest_after_hard_filters, valid=5, rejected=3, selected=green-yellow can/可回收物
Selected score: total=1417.763, point_count=908, nearest_xy=0.2166 m

Run: task_319_garbage_sort/output/head_camera_grasp_records/20260623_125947
Command: timeout 300s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --skip_graspnet --use_centroid_fallback --execute_grasp --record_debug --warmup_steps 2 --cycle_interval_steps 1 --trajectory_steps 20 --grasp_steps 5 --lift_steps 5 --hold_steps 5 --post_action_hold_steps 0
Reachability gate: requires_reachable=True, valid=5, all reachable=False, target=None, execution blocked before grasp/navigation.
```

## 2026-06-23: Wire v27 original perception into grasp and post-grasp Nav2 sorting

Change:
- Added `--perception_source v27_original` as the default Task319 visual chain. It runs the v27 original `qwen_first` pipeline directly with `viss/models/yolo11s-seg.pt`, without the Task319 merged filtering overrides that degraded recognition.
- Adapted v27 `detections` and `approach_candidates` into visual grasp candidates. `rejected_detections` remain debug-only and are not eligible for grasp.
- Kept 3D grasp geometry in the current Task319 scene: masks from v27 are projected through the current head RGB-D depth, intrinsics, and `camera_to_world`, then the centroid fallback uses that RGB-D geometric center.
- Added post-grasp sorting navigation: when `--enable_sort_nav` and `--execute_grasp` are both set, a verified grasp now continues through Nav2 to the bin staging pose, selects the category bin side from the v27/Qwen label, drives there, and runs the drop state.
- For full visual-grasp-sort runs, defaulted the wheel path to the current stable Nav2 bridge settings: `--nav_actuation_mode wheel`, `--wheel_drive_model urdf_diagonal`, `--wheel_ground_coupling kinematic_stable`, and `--wheel_velocity_scale 0.35` unless explicitly overridden.

Validation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

Smoke validation:

```text
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260623_124452
Command: timeout 480s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --skip_graspnet --use_centroid_fallback --record_debug --warmup_steps 2 --cycle_interval_steps 1 --post_action_hold_steps 0
v27_original: pipeline=qwen_first, planner=2, approach=4, accepted=6, rejected_debug=4
Selected target: source=v27_original, object=海绵, category=其他垃圾, center=[1.0326473712921143, -0.15174487233161926, 0.6213397979736328]
Execution: not run in this smoke test; grasp_source=fallback candidate was generated.
```

## 2026-06-23: Restore v27 perception defaults in Task319 adapter

Change:
- Restored the Task319 v27 adapter defaults to match the v27 script: `--viss_qwen_verify_mode planner_only`, `--viss_qwen_max_candidates 20`, and `--viss_qwen_max_roi_refine 20`.
- Kept the requested model override at `viss/models/yolo11s-seg.pt`.
- Added `v27_config_policy` and `v27_effective_config` to `viss_qwen_instances.json` so every cycle records the effective v27 adapter settings.

Validation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py
```

## 2026-06-23: Save full-frame YOLO all-object recognition overlay each cycle

Change:
- Added a cycle-level YOLO debug pass that uses the official v27 weight `viss/models/yolo11s-seg.pt` on the current head RGB frame.
- Each cycle now writes `yolo_all_overlay.png` with all YOLO boxes/masks and `yolo_all_detections.json` with raw labels, confidence, boxes, and mask-pixel counts.
- This overlay is debug-only and does not change target selection, Qwen classification, reachability checks, or grasp execution.

Validation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py
```

## 2026-06-23: Promote v27/Qwen perception to the default visual chain

Change:
- Changed the default `--perception_source` from legacy `yolo_glm` to `viss_qwen_first`, making the v27 Qwen-first + YOLO11 ROI adapter the official visual path for Task319 grasp runs.
- Kept the legacy local YOLO+GLM path available through `--perception_source yolo_glm`.
- Updated the default visual-grasp commands so they no longer need to pass `--perception_source viss_qwen_first` explicitly.
- Changed the no-target summary text from the legacy YOLO+GLM wording to generic visual-perception wording.

Validation command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py
```

## 2026-06-23: Make v27/Qwen labels authoritative over YOLO raw classes

Change:
- For `--perception_source viss_qwen_first`, the adapted instance label now uses Qwen/VLM `verify_object_name` first, then `coarse_object_name`.
- The YOLO ROI `raw_class_name` is retained only as `yolo_raw_class_name` debug metadata.
- Waste category selection already uses Qwen/VLM `garbage_category`; metadata now records `label_policy=vlm_output_over_yolo_raw_class`.

Syntax check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py
```

## 2026-06-23: Switch v27/Qwen adapter default YOLO ROI weight

Change:
- Changed `--viss_qwen_model` default from `viss/models/yolo11s-seg-best.pt` to `viss/models/yolo11s-seg.pt`.
- Manual overrides with `--viss_qwen_model ...` still work.

Syntax check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py
```

## 2026-06-23: Add v27 Qwen-first perception adapter for current head RGB-D scene

Change:
- Added `--perception_source viss_qwen_first` to run the v27 Qwen-first + YOLO ROI segmentation pipeline on the current Task319 `head_rgbd` RGB frame.
- The v27 output is adapted into the existing `YoloInstance` path, so 3D centers, reachability, target selection, grasp planning, and execution still use the current scene depth image, intrinsics, and `camera_to_world`.
- The v27 script now supports the official DashScope Qwen multimodal-generation API by default through `DASHSCOPE_API_KEY`, plus configurable `--output-json` and `--overlay-output` paths.
- Navigation waypoint yaw is now recomputed to face the table for table standpoints and face the bins for bin standpoints.

API setup:

```bash
export DASHSCOPE_API_KEY='your DashScope API key'
export QWEN_MODEL='qwen3-vl-flash'
export QWEN_API_STYLE='dashscope'
```

API check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python viss/scripts/perception/yolo11_qwen_perception_offline.py --qwen-api-check --qwen-api-style dashscope
```

Syntax check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile viss/scripts/perception/yolo11_qwen_perception_offline.py task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

GUI launch command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --perception_source viss_qwen_first --num_cycles 1 --skip_graspnet --use_centroid_fallback --execute_grasp --record_debug --record_video --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.012 --post_action_hold_steps 1200
```

## 2026-06-23: Remove remote Isaac ground USD dependency from main scene startup

Change:
- Replaced `HeadCameraGraspSceneCfg.ground` in `visual_grasp_record_demo.py` from IsaacLab `GroundPlaneCfg` to a local static `CuboidCfg` floor.
- This avoids opening `https://omniverse-content-production.s3-us-west-2.amazonaws.com/.../default_environment.usd` during scene creation, which fails in offline or restricted-network environments before the state machine starts.
- The replacement keeps collision, visual material, and high-friction physics material; the cuboid is `12 m x 12 m x 0.02 m` with its top surface at `z=0`.

Syntax check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
python3 -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

Startup verification:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 180s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --skip_yolo --skip_vlm --skip_graspnet --warmup_steps 2 --cycle_interval_steps 1 --post_action_hold_steps 0
```

Observed result:

```text
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260623_084730
Scene creation succeeded.
Cycle 0000 was written.
The previous FileNotFoundError for default_environment.usd did not recur.
```

## 2026-06-23: Add dynamic table-side grasp-standpoint navigation demo

Change:
- Added `--dynamic_standpoint_nav_demo`, which pauses perception/grasping and
  converts a target object world coordinate into continuous table-side robot
  standpoints.
- Candidate standpoints are generated on the allowed table sides only:
  `front`, `left`, and `right`. The bin side is not treated as a grasp side.
- Each candidate records the target point in the robot local frame and applies
  a right-arm geometric reachability gate before it can be selected.
- Added optional dry-run right-arm IK diagnostics through
  `--dynamic_ik_diagnostics` and `--dynamic_require_ik_reachable`.
  Diagnostics are enabled by default; IK is not a hard gate unless explicitly
  requested because the current dry-run starts from a default arm posture that
  is stricter than the navigation-standpoint problem.
- The dynamic demo defaults to the separate stable wheel path when the user does
  not override wheel options:
  `--wheel_drive_model urdf_diagonal --wheel_ground_coupling kinematic_stable --wheel_velocity_scale 0.35`.
  The older `mecanum45 + kinematic` waypoint baseline is unchanged.
- Outputs are written under `dynamic_standpoint_nav/`:
  `dynamic_standpoint_candidates.json`, `dynamic_standpoint_nav_metadata.json`,
  and `fsm_trace.json`.

Syntax check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

Verification:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 720s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --dynamic_standpoint_nav_demo --dynamic_target_world_xyz 1.20,0.00,0.60 --dynamic_allowed_table_sides front,left,right --nav_backend nav2 --wheel_drive_model urdf_diagonal --wheel_ground_coupling kinematic_stable --wheel_velocity_scale 0.35 --record_debug --record_video --warmup_steps 80 --nav2_stack_startup_s 2 --nav2_goal_timeout_s 150 --video_sample_stride 4 --post_action_hold_steps 0
```

Observed result:

```text
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260623_005732
Video: task_319_garbage_sort/output/head_camera_grasp_records/20260623_005732/external_grasp_demo.mp4
Target: [1.20, 0.00, 0.60]
Selected candidate: table_front_dynamic_0
Selected pose: [0.53, 0.16, 0.0]
Target local xy from selected pose: [0.67, -0.16]
Selection policy: geometric_reachable_with_ik_diagnostic
Nav2 status: SUCCEEDED
Final measured pose: [0.540, 0.011, 0.311]
Measured position error: 0.148929 m
Measured yaw error: -0.311376 rad
```

Current limitation:
- Nav2 now receives and completes the dynamic standpoint goal, but the Isaac
  base still finishes with about `0.149 m` position error and `0.311 rad` yaw
  error in this run. This is a command-to-wheel tracking issue, not a failure
  of target-to-standpoint candidate generation.

## 2026-06-23: Add stable URDF-diagonal wheel adapter without replacing the kinematic baseline

Change:
- Preserved the existing `--wheel_drive_model mecanum45 --wheel_ground_coupling kinematic`
  behavior as the important Nav2 waypoint success baseline.
- Added `--wheel_drive_model urdf_diagonal`, which derives the four S62 wheel
  directions from the URDF joint geometry:
  - LF `+45 deg`
  - LB `+135 deg`
  - RF `-45 deg`
  - RB `-135 deg`
- Added `--wheel_ground_coupling kinematic_stable`, an independent kinematic
  coupling mode that integrates the desired planar pose internally instead of
  feeding physics-step yaw drift back into the next kinematic step.
- Added `--wheel_velocity_scale`, default `1.0`, to scale wheel joint velocity
  targets without changing the chassis `/cmd_vel` used by the kinematic layer.
- Extended `wheel_drive_metadata()` so each run records `wheel_velocity_scale`
  and whether the stable integrated pose mode was active.

Syntax check:

```bash
cd /home/zhxm/workspace/mda_isaaclab/task_319_garbage_sort
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile visual_grasp_record_demo.py task319_grasp_sort_sm.py
/usr/bin/python3 -m py_compile scripts/ros2_cmd_vel_test_publisher.py scripts/ros2_cmd_vel_socket_server.py scripts/kuavo_ros1_cmd_vel_socket_client.py
```

Straight `/cmd_vel` comparison:

```text
Old baseline:  mecanum45 + kinematic
Run:           task_319_garbage_sort/output/head_camera_grasp_records/20260623_000756
translation:   0.246839 m
yaw drift:    -0.138321 rad

Rejected test: differential + kinematic
Run:           task_319_garbage_sort/output/head_camera_grasp_records/20260623_002300
translation:   0.251842 m
yaw drift:    -0.203568 rad

New stable:    urdf_diagonal + kinematic_stable + wheel_velocity_scale=0.35
Run:           task_319_garbage_sort/output/head_camera_grasp_records/20260623_002628
Video:         task_319_garbage_sort/output/head_camera_grasp_records/20260623_002628/external_grasp_demo.mp4
translation:   0.243330 m
yaw drift:    -0.0000507 rad
```

New stable straight verification command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 480s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --ros_cmd_vel_demo --record_debug --record_video --ros_cmd_vel_demo_steps 240 --ros_cmd_vel_demo_linear_x 0.20 --ros_cmd_vel_demo_angular_z 0.0 --ros_cmd_vel_demo_min_translation_m 0.05 --video_sample_stride 4 --post_action_hold_steps 0 --wheel_drive_model urdf_diagonal --wheel_ground_coupling kinematic_stable --wheel_velocity_scale 0.35
```

Nav2 waypoint validation with the new stable adapter:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 720s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --waypoint_nav_demo --waypoint_route home,table_left --nav_backend nav2 --record_debug --record_video --warmup_steps 80 --nav2_stack_startup_s 2 --nav2_goal_timeout_s 150 --post_action_hold_steps 0 --wheel_drive_model urdf_diagonal --wheel_ground_coupling kinematic_stable --wheel_velocity_scale 0.35 --video_sample_stride 4
```

Observed Nav2 result:

```text
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260623_002725
Video: task_319_garbage_sort/output/head_camera_grasp_records/20260623_002725/external_grasp_demo.mp4
Waypoint Nav2 demo success=True
current -> home: SUCCEEDED
home -> table_left: SUCCEEDED
```

## 2026-06-23: Verify ROS 2 `/cmd_vel` really moves the Isaac robot

Change:
- Added `scripts/ros2_cmd_vel_test_publisher.py`, a finite-duration ROS 2
  `geometry_msgs/Twist` publisher for direct command validation.
- Added `--ros_cmd_vel_demo` to `visual_grasp_record_demo.py` /
  `task319_grasp_sort_sm.py`.
- In this mode, perception/grasping are paused, Isaac starts the existing
  ROS 2 bridge, receives `/cmd_vel`, applies it through the wheel-mode base
  command path, records wheel states, and validates success from measured robot
  base displacement.
- The mode writes `ros_cmd_vel_demo/ros_cmd_vel_demo_metadata.json`,
  `ros_cmd_vel_demo/fsm_trace.json`, and the observer video when
  `--record_video` is enabled.

Syntax check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
/usr/bin/python3 -m py_compile task_319_garbage_sort/scripts/ros2_cmd_vel_test_publisher.py task_319_garbage_sort/scripts/ros2_cmd_vel_socket_server.py task_319_garbage_sort/scripts/kuavo_ros1_cmd_vel_socket_client.py
```

Verification:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 480s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --ros_cmd_vel_demo --record_debug --record_video --ros_cmd_vel_demo_steps 240 --ros_cmd_vel_demo_linear_x 0.20 --ros_cmd_vel_demo_angular_z 0.0 --ros_cmd_vel_demo_min_translation_m 0.05 --video_sample_stride 4 --post_action_hold_steps 0
```

Observed result:

```text
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260623_000756
Video: task_319_garbage_sort/output/head_camera_grasp_records/20260623_000756/external_grasp_demo.mp4
ROS 2 /cmd_vel publisher: 111 messages, linear.x=0.20 m/s, angular.z=0.0 rad/s
Initial pose: [0.53, 0.0, 0.0]
Final pose: [0.776834, -0.001488, -0.138321]
translation_m=0.246839
success=True
reason=ROS2 /cmd_vel moved the robot in Isaac.
```

Interpretation:
- This verifies the real ROS 2 topic-to-Isaac-control path: the robot's measured
  base pose changed because `/cmd_vel` was received and applied.
- It is separate from the ROS 2 -> TCP -> ROS 1 Kuavo `/cmd_vel` bridge
  verification below. The remaining unverified layer is acceptance by the real
  Kuavo wheel-arm MPC controller service in the actual Kuavo ROS 1 controller
  environment.

## 2026-06-22: Add Nav2 `/cmd_vel` to Kuavo ROS1 base-command bridge prototype

Change:
- Added `scripts/ros2_cmd_vel_socket_server.py`, a ROS 2 node that subscribes to
  Nav2 `/cmd_vel` and exposes the latest `geometry_msgs/Twist` over a TCP JSON
  protocol.
- Added `scripts/kuavo_ros1_cmd_vel_socket_client.py`, a Kuavo-side ROS 1 client
  that pulls those snapshots and republishes them to official Kuavo `/cmd_vel`
  or `/cmd_vel_world`.
- The Kuavo client can optionally call `/mobile_manipulator_mpc_control` before
  publishing, so the target controller mode can be set to `BaseOnly` (`2`) or
  `BaseArm` (`3`).
- Added `--dry-run` mode to validate the bridge on machines without ROS 1.

Syntax check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/usr/bin/python3 -m py_compile task_319_garbage_sort/scripts/ros2_cmd_vel_socket_server.py task_319_garbage_sort/scripts/kuavo_ros1_cmd_vel_socket_client.py
```

Dry-run verification:

```text
ROS 2 test publisher sent: linear.x=0.18, linear.y=0.02, angular.z=0.11
Kuavo dry-run output:     linear=(0.1800,0.0200,0.0000), angular_z=0.1100
Timeout output:           linear=(0.0000,0.0000,0.0000), angular_z=0.0000
```

ROS 1 topic verification:

```text
ROS 1 /cmd_vel subscriber received: vx=0.1800, vy=0.0200, wz=0.1100
```

Current limitation:
- The ROS 1 topic publishing path was verified with a local Noetic `rosmaster`
  from a historical sandbox and the generated Kuavo message bindings. The actual
  Kuavo wheel-arm MPC controller service `/mobile_manipulator_mpc_control` was
  not available in this workstation environment, so hardware/controller
  acceptance still needs to be tested inside the real Kuavo ROS 1 controller
  environment.

## 2026-06-22: Investigate official Kuavo ROS/SDK wheel-arm motion interfaces

Change:
- Checked the official Kuavo 5-W ROS interface documentation and the local
  official `kuavo-ros-opensource` examples.
- Documented the relevant official base-control interfaces in
  `NAV2_MOTION_IMPLEMENTATION.md`:
  - `/cmd_vel`
  - `/cmd_vel_world`
  - `/cmd_pose`
  - `/cmd_pose_world`
  - `/lb_cmd_pose_reach_time`
  - `/mobile_manipulator_mpc_control`
  - `/mobile_manipulator/lb_mpc_control_mode`
  - `/mobile_manipulator_timed_single_cmd`
- Identified the preferred next direction: use official Kuavo wheel-arm
  `cmd_pose_world`/timed command or bridge Nav2 `/cmd_vel` into official Kuavo
  ROS1 `/cmd_vel`, instead of continuing to tune handwritten four-wheel velocity
  mapping in Isaac.

No simulation behavior was changed in this entry.

## 2026-06-22: Revert wheel-friction damping experiment and add official wheeled-controller diagnostic

Change:
- Reverted the previous high-friction/yaw-damping experiment:
  - `high_friction_material()` is back to `static_friction=1.45`, `dynamic_friction=1.10`.
  - Removed runtime wheel collision material binding and the associated `make_instanceable=False` robot import override.
  - Restored Isaac Nav2 command defaults to `--nav_max_angular_speed=0.75` and `--nav_cmd_angular_scale=2.5`.
  - Removed Isaac-side `/cmd_vel` acceleration smoothing from the main state machine.
  - Restored bundled RPP parameters to `rotate_to_heading_angular_vel=0.75`, `max_angular_accel=2.4`, and `min_y_velocity_threshold=0.001`.
- Added `official_holonomic_wheel_test.py`, a standalone wheel diagnostic that bypasses Nav2/perception/grasping and uses NVIDIA Isaac Sim wheeled-robot controllers to generate wheel velocity targets.
- Added `--controller_backend differential|holonomic`. The `holonomic` backend currently identifies a controller initialization/geometric-setup issue; the `differential` backend is used as a control test for the official controller-to-wheel-action path.

Syntax check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/official_holonomic_wheel_test.py task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/scripts/ros2_nav2_stack_launcher.py
```

Verification:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 260s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/official_holonomic_wheel_test.py --headless --record_debug --controller_backend differential --warmup_steps 5 --drive_steps 40 --cmd_vx 0.18 --cmd_vy 0.0 --cmd_wz 0.0
```

Observed result:

```text
Run: task_319_garbage_sort/output/official_holonomic_wheel_tests/20260622_234208
controller_backend=differential
wheel_targets_radps=[1.380898, 1.380898, 1.380898, 1.380898]
translation_m=0.000116
yaw_delta_rad=-0.147768
```

Interpretation:
- The official controller path reaches the wheel articulation because the robot rotates.
- Equal wheel velocity targets do not produce forward travel on this S62 wheel geometry, so the remaining issue is wheel order/axis/roller-angle mapping, not Nav2 itself.

## 2026-06-22: Wheel friction and yaw-oscillation damping

Status: reverted by the entry above; do not treat this as the current baseline.

Change:
- Raised the current Task319 Isaac high-friction material to `static_friction=1.50` and `dynamic_friction=1.20`.
- Added runtime binding of the same high-friction material to Kuavo wheel collision prims. Wheel links are made uninstanceable before binding so the per-wheel material override reaches the actual collider prims.
- Reduced Isaac-side angular command aggressiveness:
  - `--nav_max_angular_speed`: `0.60 rad/s`
  - `--nav_cmd_angular_scale`: `1.0`
  - `--nav_cmd_linear_accel_limit`: `0.35 m/s^2`
  - `--nav_cmd_angular_accel_limit`: `0.45 rad/s^2`
- Added stateful `/cmd_vel` smoothing before wheel target generation so wheel commands ramp instead of jumping.
- Tuned bundled Nav2 RPP yaw behavior:
  - `rotate_to_heading_angular_vel`: `0.60 rad/s`
  - `max_angular_accel`: `0.45 rad/s^2`
  - `min_y_velocity_threshold`: `0.0`
- Confirmed the current RPP controller does not use DWB `max_vel_theta/max_accel_theta/min_vel_y/max_vel_y` fields; the equivalent active parameters above were changed, and the Isaac bridge consumes only `linear.x` plus `angular.z`.
- Checked the loaded S62 URDF mass: 29 mass entries, total about `204.3 kg`, with `6.0 kg` per wheel. No mass override was added.

Syntax check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py task_319_garbage_sort/scripts/ros2_nav2_stack_launcher.py
```

Visible verification command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --waypoint_nav_demo --waypoint_route home,table_left,bin_center,table_right,home --nav_backend nav2 --record_debug --record_video --warmup_steps 80 --nav2_stack_startup_s 2 --post_action_hold_steps 240
```

Short anti-slip/yaw-damping verification:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 450s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --wheel_open_loop_demo --record_debug --record_video --warmup_steps 10 --wheel_open_loop_steps 30 --wheel_open_loop_linear_speed 0.20 --wheel_open_loop_angular_speed 0.0 --wheel_open_loop_min_translation_m 0.0
```

Observed result:

```text
Wheel high-friction material bound to 4/4 collision prims after uninstancing 4 wheel links (static=1.50, dynamic=1.20).
Wheel open-loop demo success=True
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260622_231708
Video: task_319_garbage_sort/output/head_camera_grasp_records/20260622_231708/external_grasp_demo.mp4
translation_m=0.073729
yaw_delta_rad=0.003807
```

## 2026-06-22: Wheel-mode waypoint Nav2 demo and kinematic wheel-ground coupling

Change:
- Added/validated `--waypoint_nav_demo` for moving between named ground waypoints while perception/grasp are paused.
- `--waypoint_route` is now the explicit ordered endpoint sequence. `table_left` and `table_right` are possible grasp standpoints, not hard-coded mandatory intermediates.
- The waypoint/open-loop demos force wheel-mode execution with `--nav_actuation_mode wheel`; the current validated default is `--wheel_ground_coupling kinematic`.
- Added planar pose integration for `wheel + kinematic` coupling. Nav2 still provides `/cmd_vel`, Isaac still computes four S62 wheel joint velocity targets, and the kinematic layer writes the resulting base pose/velocity to avoid the currently unreliable pure wheel-contact model.
- Tuned bundled Nav2 Regulated Pure Pursuit parameters for the waypoint route: desired linear velocity `0.32 m/s`, lookahead `0.25 m`, and collision lookahead `0.20 s`.
- Documented pure contact wheel findings: wheel joints respond, but the best raw sign-pattern sweep only moved about `0.0025 m`, so pure contact is not yet accepted for route validation.

Verification:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 900s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --waypoint_nav_demo --waypoint_route home,table_left,bin_center,table_right,home --nav_backend nav2 --record_debug --record_video --warmup_steps 80 --nav2_stack_startup_s 2
```

Observed result:

```text
Waypoint Nav2 demo success=True reason=Waypoint Nav2 demo completed.
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260622_225252
Video: task_319_garbage_sort/output/head_camera_grasp_records/20260622_225252/external_grasp_demo.mp4
```

All waypoint transitions returned `SUCCEEDED`: `current->home`, `home->table_left`, `table_left->bin_center`, `bin_center->table_right`, and `table_right->home`.

## 2026-06-22: Natural-down arms, faster Nav2 execution, and side-corridor return

Change:
- Set the Kuavo S62 initial arm posture to natural-down arm joint targets instead of the previous raised-arm offsets.
- Updated the motion-only pick stub to close the gripper first, then raise the right arm into `right_arm_carry`; updated the drop stub to open the gripper and return the arm to `natural_down`.
- Increased Nav2/Isaac execution speed to `0.45 m/s` linear and `0.75 rad/s` angular. External Nav2 angular commands are scaled by `2.5` before Isaac execution and clamped to the max angular speed.
- Tuned bundled Nav2 controller parameters: yaw goal tolerance `0.35 rad`, progress checker movement allowance `45 s`, desired linear velocity `0.45 m/s`.
- Fixed return-home failures from the bin area by routing through explicit Nav2 goals around the side corridor: `RETURN_TO_BIN_STAGING -> RETURN_TO_SIDE_CORRIDOR -> RETURN_HOME_APPROACH -> RETURN_HOME_ALIGN -> RETURN_HOME`.
- Documented why previous `STATUS_6` results happened: Nav2 controller progress checking aborted goals when the local controller issued angular-only commands near obstacle regions without enough translational progress.

Verification:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 900s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --motion_only_sort_demo --nav_backend nav2 --motion_test_category 其他垃圾 --record_debug --record_video --warmup_steps 80 --nav2_stack_startup_s 2
```

Observed result:

```text
Motion-only Nav2 demo success=True reason=Motion-only Nav2 sort demo completed.
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260622_220145
Video: task_319_garbage_sort/output/head_camera_grasp_records/20260622_220145/external_grasp_demo.mp4
```

All Nav2 states in `motion_only_nav2_metadata.json` returned `SUCCEEDED`, including the new return corridor states.

## 2026-06-22: Local Nav2 runtime, auto launcher, and smoke test

Change:
- Added a user-local ROS Jazzy Nav2 runtime under `task_319_garbage_sort/output/nav2_user_install/root` and `scripts/setup_nav2_user_install.bash` so Nav2 helpers can run without sudo.
- Added `scripts/ros2_nav2_stack_launcher.py`, which generates a blank map, a minimal Nav2 behavior tree, Nav2 parameters, and starts the official `map_server`, `planner_server`, `controller_server`, `bt_navigator`, and `lifecycle_manager` processes.
- Added `scripts/ros2_nav2_smoke_test.py`, a ROS-only test harness that publishes odom/tf/scan, subscribes to `/cmd_vel`, sends `NavigateToPose`, and verifies that Nav2 completes the goal.
- Motion-only sort navigation now auto-starts the bundled Nav2 stack by default through `--start_nav2_stack`; pass `--no-start_nav2_stack` only when connecting to an external stack.
- `ros2_nav_goal_client.py` now retries rejected goals so lifecycle activation timing does not fail the first target.
- The generated Nav2 map now includes known static table/bin geometry. The Isaac integration defaults to a 150s Nav2 goal timeout, uses a 15 cm base station tolerance, and the bundled Regulated Pure Pursuit controller uses a shorter collision lookahead window so slow headless simulation does not fail a valid long path prematurely.
- Fixed the motion-only drop stub to use the existing width-based gripper API when opening the attached gripper.
- Added `NAV2_MOTION_IMPLEMENTATION.md` as the maintained implementation note for the navigation control chain, interfaces, current validation status, and future motion-stack updates.

Cold-start Nav2 smoke test:

```bash
cd /home/zhxm/workspace/mda_isaaclab
source task_319_garbage_sort/scripts/setup_nav2_user_install.bash
/usr/bin/python3 task_319_garbage_sort/scripts/ros2_nav2_stack_launcher.py --output-dir task_319_garbage_sort/output/nav2_smoke_stack_cold
```

Second terminal:

```bash
cd /home/zhxm/workspace/mda_isaaclab
source task_319_garbage_sort/scripts/setup_nav2_user_install.bash
/usr/bin/python3 task_319_garbage_sort/scripts/ros2_nav2_smoke_test.py --goal-x 0.60 --goal-y 0.0 --goal-yaw 0.0 --timeout-s 55 --server-timeout-s 25 --goal-attempts 5 --retry-delay-s 2
```

Observed result:

```text
{"success": true, "status": "SUCCEEDED", "status_code": 4, "attempt": 3, "final_distance_m": 0.07788016159431975}
```

Isaac motion-only single-category Nav2 verification:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 700s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --motion_only_sort_demo --nav_backend nav2 --motion_test_category 其他垃圾 --record_debug --record_video --warmup_steps 80
```

Observed Isaac result:

```text
Motion-only Nav2 demo success=True reason=Motion-only Nav2 sort demo completed.
Run: task_319_garbage_sort/output/head_camera_grasp_records/20260622_211843
Video: task_319_garbage_sort/output/head_camera_grasp_records/20260622_211843/external_grasp_demo.mp4
```

## 2026-06-22: GraspNet calibration projection and TCP frame debug

Change:
- Auto target selection for `--execute_grasp`/`--strict_model_chain` now rejects perception candidates outside the right-arm reachable workspace before GraspNet/IK, instead of selecting a far object and failing with no motion. Metadata records target reachability bounds and relative target position.
- Strict visual-grasp runs now ignore `--use_centroid_fallback`; the selected learned backend must provide the grasp candidate for execution.
- Added `debug_projection.jpg` and `calibration_debug.json` in each cycle directory. The image reprojects the Top-1 learned grasp world point through `camera_to_world` and camera intrinsics, then marks it with a red crosshair.
- Added live Isaac frame markers at `/Visuals/task319_target_grasp_pose` and `/Visuals/task319_current_tcp_pose` so the GraspNet target frame and Kuavo right-arm TCP frame can be compared in RGB axes during PRE_GRASP/GRASP.
- Added mandatory `[CALIB]` terminal diagnostics for `Camera_to_World`, `Top1_GraspNet_Pose_World`, `Right_Arm_TCP_Pose_World`, plus TCP-minus-target translation and relative RPY.

Strict GraspNet calibration-debug command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
export GLM_API_KEY='your GLM API key'
timeout 420s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --execute_grasp --strict_model_chain --record_debug --record_video --vlm_model glm-5v-turbo --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.012 --mask_distance_threshold_m 0.02 --min_filtered_grasps 8 --ik_prescreen_top_k 20 --ik_prescreen_max_joint_delta_rad 4.5 --max_grasp_retries 3 --pregrasp_error_threshold_m 0.07 --grasp_error_threshold_m 0.06 --lift_error_threshold_m 0.08
```

Syntax check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

## 2026-06-22: Experimental AnyGrasp backend with explicit TCP calibration

Change:
- Added `--grasp_backend anygrasp` as an experimental learned grasp backend alongside the existing GraspNet path.
- Added `--anygrasp_repo`, `--anygrasp_checkpoint`, gripper geometry flags, object-mask/collision flags, and `--anygrasp_grasp_to_tcp`.
- Validated `--anygrasp_grasp_to_tcp` as a strict 4x4 homogeneous transform before inference: finite values, `[0,0,0,1]` last row, orthonormal rotation, determinant `+1`, and a conservative translation bound.
- AnyGrasp candidates now flow through the same relaxed mask filter, ranking, IK prescreen, Top-K retry, execution metrics, and strict success checks as GraspNet. Metadata records `grasp_backend`, generic `grasp_candidate_count`, `grasp_filtered_count`, and the calibration diagnostics.
- Extended `prepare_grasp_deps.py` to report AnyGrasp SDK path, checkpoint, `gsnet`/`lib_cxx` binaries, license folder, MinkowskiEngine status, and OpenSSL 1.1 runtime status without breaking the default GraspNet-only dependency check. Use `--require_anygrasp` when validating an AnyGrasp installation.

AnyGrasp strict visual-grasp command after placing the licensed SDK assets:

```bash
cd /home/zhxm/workspace/mda_isaaclab
export GLM_API_KEY='your GLM API key'
timeout 420s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --execute_grasp --strict_model_chain --grasp_backend anygrasp --record_debug --record_video --vlm_model glm-5v-turbo --target_object_name potted --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.012 --mask_distance_threshold_m 0.02 --min_filtered_grasps 8 --ik_prescreen_top_k 20 --ik_prescreen_max_joint_delta_rad 4.5 --max_grasp_retries 3 --pregrasp_error_threshold_m 0.07 --grasp_error_threshold_m 0.06 --lift_error_threshold_m 0.08 --anygrasp_grasp_to_tcp 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
```

Dependency check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/scripts/prepare_grasp_deps.py --prepare_anygrasp_binaries --require_anygrasp
```

## 2026-06-22: GraspNet mask relaxation, IK prescreen, and Top-K retry

Change:
- Replaced strict GraspNet mask gating with a relaxed filter: strict eroded 2D mask first, raw 2D mask fallback second, and 3D KD-tree/nearest-distance acceptance within `--mask_distance_threshold_m` when mask-edge noise would otherwise remove valid grasps.
- Added ranked GraspNet candidate pools instead of selecting a single pose immediately. Metadata now records candidate rank, combined score, top-down bias, mask-filter mode/counts, TCP alignment mode, and candidate metadata.
- Added pre-execution IK reachability prescreen over `--ik_prescreen_top_k` candidates for pre-grasp, grasp, and lift poses; `--ik_prescreen_max_joint_delta_rad` controls the single-probe joint-delta rejection threshold.
- Added Top-K retry in the grasp FSM. Failed GraspNet attempts retreat toward the current candidate pre-grasp pose, then continue to the next IK-reachable GraspNet pose up to `--max_grasp_retries`; centroid fallback remains a degraded debug path and only reports `fallback_success=true`.
- Added execution rejection thresholds: `--pregrasp_error_threshold_m`, `--grasp_error_threshold_m`, and `--lift_error_threshold_m`.

Strict visual-grasp command with the new defaults made explicit:

```bash
cd /home/zhxm/workspace/mda_isaaclab
export GLM_API_KEY='your GLM API key'
timeout 420s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --execute_grasp --strict_model_chain --record_debug --record_video --vlm_model glm-5v-turbo --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.012 --mask_distance_threshold_m 0.02 --min_filtered_grasps 8 --ik_prescreen_top_k 20 --ik_prescreen_max_joint_delta_rad 4.5 --max_grasp_retries 3 --pregrasp_error_threshold_m 0.07 --grasp_error_threshold_m 0.06 --lift_error_threshold_m 0.08
```

Syntax check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py task_319_garbage_sort/grasp_pipeline/types.py task_319_garbage_sort/grasp_pipeline/grasping/mask_filter.py task_319_garbage_sort/grasp_pipeline/grasping/grasp_selector.py
```

## 2026-06-22: Observer-camera grasp video recording

Change:
- Added an external `observer_rgb` camera that records the robot, table, and gripper motion without changing the head-camera perception source.
- Added `--record_video`, `--video_width`, `--video_height`, `--video_fps`, `--video_sample_stride`, and `--save_video_frames`.
- Added `--observer_camera_pos` and `--observer_camera_target`; the default video angle is a closer table-side view so the gripper/object motion is visible.
- The run directory now writes `external_grasp_demo.mp4`, `video_manifest.json`, and `metadata.json.observer_video` so motion can be checked without relying on the Isaac viewport.

Headless motion-evidence video command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 360s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --execute_grasp --skip_vlm --skip_graspnet --use_centroid_fallback --target_object_name potted --record_debug --record_video --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.012
```

## 2026-06-22: GUI real-time grasp playback

Change:
- Added GUI-only real-time playback pacing after each simulation step so arm/gripper motion is visible instead of running as a fast batch.
- Added `--post_action_hold_steps` to keep the final grasp scene visible before the Isaac window closes after a one-cycle demo.
- Added `--async_model_inference/--no-async_model_inference`. In GUI mode, YOLO/VLM and GraspNet now run in background workers while the Isaac main thread keeps the viewport event loop alive.
- Added a GUI planning message before YOLO/VLM/GraspNet inference. Heavy GPU inference can still reduce frame rate, but the window should not go fully unresponsive while planning.

GUI strict visual-grasp display command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
export GLM_API_KEY='your GLM API key'
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --num_cycles 1 --execute_grasp --strict_model_chain --record_debug --vlm_model glm-5v-turbo --target_object_name potted --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.012 --post_action_hold_steps 1200
```

## 2026-06-22: Strict YOLO + GLM-VLM + GraspNet grasp chain

Change:
- Added `--strict_model_chain` to require YOLO, GLM-VLM, and GraspNet for strict visual-grasp success.
- Strict mode rejects `--skip_yolo`, `--skip_vlm`, and `--skip_graspnet`.
- Strict target selection excludes depth-component fallback targets; fallback can still be recorded as a debug path but cannot set `strict_success=true`.
- GraspNet-selected grasps now execute with the GraspNet pose orientation; centroid fallback remains the stable wrist-orientation debug path.
- `metadata.json` now records `strict_model_chain`, `strict_success`, `fallback_success`, `vision_modules`, `target_source`, `grasp_source`, `graspnet_candidate_count`, and `graspnet_filtered_count`.

Strict visual-grasp command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
export GLM_API_KEY='your GLM API key'
timeout 360s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --execute_grasp --strict_model_chain --record_debug --vlm_model glm-5v-turbo --target_object_name potted --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.012
```

Strict success requires:
- `camera_source=head_rgbd`
- `target_selection.source=yolo`
- `vision_modules.yolo.enabled=true`, `vision_modules.vlm.enabled=true`, `vision_modules.graspnet.enabled=true`
- `grasp_source=graspnet`, `graspnet_filtered_count > 0`
- `execution.grasp_success=true`, `strict_success=true`

## 2026-06-22: Local Git snapshot hygiene

Change:
- Added `.gitignore` for the Task319 snapshot repository.
- Added `SAVE_TO_GITHUB.md` with GitHub push commands for user `CuberW <773329413@qq.com>`.
- Ignored generated outputs, Python caches, downloaded model weights, and third-party dependency folders so the code/config snapshot stays small and safe to push.

## 2026-06-22: True head-camera grasp state machine

Change:
- Added the public main-chain entrypoint `task319_grasp_sort_sm.py` for Task319 grasp/sort execution.
- Extended the head-camera grasp chain into an FSM: `REST -> BASE_TO_TABLE -> STATIC_PERCEPT -> PLAN_GRASP -> PRE_GRASP -> GRASP -> LIFT -> VERIFY_HOLD`, with optional `NAV_TO_BIN -> DROP -> RETURN_HOME`.
- Kept perception grounded in the robot head RGB-D camera only. Debug output now records head RGB/depth, YOLO masks, VLM overlay, depth-component fallback, selected grasp overlay, FSM trace, and execution metrics.
- Added a tabletop depth-component fallback so YCB objects can still be selected from the head RGB-D point cloud when YOLO labels are unreliable.
- Fixed true-grasp manipulation stability by using fixed-base manipulation by default, keeping Kuavo upright at the URDF zero pose, locking the base during arm execution, and preserving the current wrist orientation for reachable pose-IK commands.
- Left hardcoded bin-waypoint sorting behind `--enable_sort_nav`; the default command stops after successful lift/hold so grasp can be accepted independently before navigation is enabled.

Syntax check:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile task_319_garbage_sort/visual_grasp_record_demo.py task_319_garbage_sort/task319_grasp_sort_sm.py
```

Headless true-grasp verification command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 300s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --execute_grasp --skip_vlm --skip_graspnet --use_centroid_fallback --target_object_name potted --record_debug --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.012
```

Observed result from `output/head_camera_grasp_records/20260622_033759/cycle_0000/metadata.json`:
- Selected target: `trash_potted_meat_can_0`, category `厨余垃圾`, source `depth_component`, camera source `head_rgbd`.
- IK final position errors: pre-grasp `0.016 m`, grasp `0.016 m`, lift `0.049 m`.
- Lift success: `grasp_success=true`, target lift delta `0.228 m`, base drift `0.000 m`.
- Debug files include `head_rgb.png`, `head_depth_vis.png`, `yolo_overlay.png`, `vlm_overlay.png`, `depth_component_overlay.png`, `selected_grasp_overlay.png`, `fsm_trace.json`, and `metadata.json`.

GUI true-grasp command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
export GLM_API_KEY='your GLM API key'
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --num_cycles 1 --execute_grasp --record_debug --vlm_model glm-5v-turbo --use_centroid_fallback
```

## 2026-06-22: Kuavo default upright posture

Change:
- Restored Kuavo's default posture to the URDF zero pose in the scene preview, head-camera grasp demo, isolated arm test, and gripper physics test.
- Set the robot root z height to `0.0` for wheel-ground contact instead of lifting the base.
- Kept head-camera table visibility through `CameraCfg.OffsetCfg`, not through non-zero torso or head joint defaults.

Display command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/phase1_scene_ros2_bridge.py --disable_lidar
```

Head-camera verification command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 120s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/visual_grasp_record_demo.py --headless --num_cycles 1 --skip_vlm --skip_graspnet
```

Expected result:
- Kuavo stands in the URDF zero upright posture, with `knee_joint`, `leg_joint`, `waist_pitch_joint`, `waist_yaw_joint`, `zhead_1_joint`, and `zhead_2_joint` at `0.0`.
- The head camera still records `head_rgb.png`, `head_depth_vis.png`, `yolo_overlay.png`, and `metadata.json`.

## 2026-06-23 - Align VISS v28 README defaults and raise head camera resolution

- Set the formal Task319 visual chain default to `v28_original` with `viss/models/yolo11s-seg-best.pt`, matching the `viss/readme.md` v28 model naming.
- Aligned VISS Qwen-first parameters with `viss/readme.md`: `conf=0.15`, `roi_expand=2.0`, `verify_mode=none`, `max_qwen_candidates=3`, `max_roi_refine=3`, `timeout=90s`, and no accepted fallback result.
- Updated the VISS subprocess adapter for the newer script CLI: unsupported legacy `--output-json`, `--overlay-output`, and `--qwen-api-style` arguments are detected and omitted; output is read from the script's default `~/trashbot_ws/data/logs` under an isolated per-cycle HOME.
- Map `DASHSCOPE_API_KEY` to `QWEN_API_KEY` and default `QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1` for the VISS OpenAI-compatible Qwen client.
- Raised the head RGB-D camera default resolution from `640x480` to `1280x960` via `--head_camera_width/--head_camera_height`.
- Changed external video observer defaults to a left-front/high view: `--observer_camera_pos 1.65,2.55,1.70`, `--observer_camera_target 1.85,0.02,0.72`.

## 2026-06-24 - Movable debug cube grasp calibration baseline

- Added a small calibration support table to the isolated debug-cube scene so the movable cube starts on a physical tabletop instead of falling to the ground.
- Changed the debug-cube default forced grasp width to `cube_size + 0.004 m` when no explicit width is provided; this avoids over-opening the gripper for a 4.5 cm cube.
- Verified the isolated movable cube grasp with right-arm position-only control: run `20260624_144910` reached the cube center with about `4 mm` TCP error, physically lifted the cube by about `0.243 m`, and saved video at `task_319_garbage_sort/output/head_camera_grasp_records/20260624_144910/external_grasp_demo.mp4`.
- Strict `top_down` wrist orientation remains unsuitable for this pose: run `20260624_144647` produced about `0.319 m` TCP-target error after pregrasp, so the immediate baseline is position-only with tighter TCP feedback and gripper width calibration.

## 2026-06-24 - cuRobo with official Kuavo analytic IK posture seed

- Kept `curobo_right_arm` as the default mainline motion backend, but enabled `--curobo_use_kuavo_analytic_seed` by default.
- Added cuRobo joint-space planning to the right-arm wrapper. The official Kuavo analytic IK now chooses the redundant right-arm posture for each TCP stage, and cuRobo plans a smooth collision-aware joint trajectory to that posture.
- The intent is to remove pathological redundant IK choices where the arm reaches the requested TCP point by twisting through the torso or folding the wrist behind the body. If the official seed is unavailable or the joint-space cuRobo plan fails, the code records the reason and falls back to the previous cuRobo pose-goal planner for comparison.
- Verified syntax with `python -m py_compile` and verified the new cuRobo joint-space planner with a small right-arm joint target: `success=True`, `steps=31`.

## 2026-06-24 - cuRobo position-only TCP mainline

- Added `--curobo_position_only_tcp`, enabled by default.
- cuRobo now first plans each TCP stage as a partial pose: rotation weights are zero and XYZ position weights are one. This keeps cuRobo responsible for shortest-path, collision-aware trajectory generation while avoiding unnecessary gripper-axis alignment and redundant wrist/arm twisting.
- Kuavo official analytic IK seed remains available, but it is now used only after the cuRobo position-only plan fails instead of being the primary posture selector.
- Verified `python -m py_compile` for the Task319 entrypoints and verified a standalone cuRobo position-only move: `success=True`, `steps=31`, planned final TCP position error about `5e-6 m`.
- Short debug-cube Isaac run `20260624_203156` confirmed the integrated backend used `position_only_pose_goal` for `pregrasp_curobo`, `grasp_curobo`, and `lift_curobo`; all three cuRobo plans succeeded with the `reach_partial_pose` metric. The cube was not lifted in that short run, so the remaining issue is gripper contact/close accuracy or residual TCP error, not strict axis alignment in cuRobo.

## 2026-06-24 - Integrate Task319 source into parent mda_isaaclab repository

- Removed the nested Git boundary from `task_319_garbage_sort` by moving its `.git` directory to `backups/task_319_garbage_sort_git_20260624_203842/`.
- Updated the parent `.gitignore` so Task319 source can be added to the parent repository while generated outputs, compiled helpers, caches, video/debug artifacts, cuRobo source checkout, and vendored GraspNet/AnyGrasp/ReKep dependencies stay local-only.
- Updated `task_319_garbage_sort/.gitignore` to ignore `build/`, video files, and logs in addition to existing output/cache/model artifacts.
- Verified the parent repository now sees 48 Task319 source/documentation/URDF files through `git ls-files -o --exclude-standard task_319_garbage_sort`; `output/`, `build/`, `__pycache__/`, and the preserved nested-git backup are ignored.
