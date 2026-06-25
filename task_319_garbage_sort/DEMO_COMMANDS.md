# Task 319 Demo Commands

All commands below use the required `my_task319_safe` environment. Run them from:

```bash
cd /home/zhxm/workspace/mda_isaaclab
```

Documentation rule:
- Every task-319 code change should update `task_319_garbage_sort/CHANGELOG.md`.
- Any visible behavior change should also add or update the launch/display command in this file.
- Repository-level setup and current status are summarized in `README.md`.
- The detailed Chinese development report is `TASK319_DEVELOPMENT_REPORT.md`.

## 1. Visual Scene Preview: robot + table + 10 textured YCB trash objects

This opens Isaac Sim with the phase-one scene. You should see Kuavo in the URDF zero upright posture, the table, four open bins, and 10 correctly-scaled textured YCB objects on the table.

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/phase1_scene_ros2_bridge.py --disable_lidar
```

Notes:
- Do not pass `--headless` if you want to see the window.
- Kuavo's default `knee_joint`, `leg_joint`, `waist_pitch_joint`, `waist_yaw_joint`, `zhead_1_joint`, and `zhead_2_joint` are all `0.0`.
- This script is a scene/ROS bridge entrypoint; it does not perform grasping by itself.
- Add `--ros2` only when you want `/cmd_vel`, `/odom`, `/scan`, and `/tf`.

## 2. Visible Gripper Physics Demo: attached gripper closes on a red block

This opens Isaac Sim close to the wrist gripper. The robot uses the URDF zero upright posture while the gripper stays open first, then a small red block appears between the finger pads, the gripper closes, and the block remains held.

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/gripper_physics_test.py --smoke_steps 0 --close_start_step 600
```

Expected visual sequence:
- First ~5 seconds: gripper open.
- Then: red block is inserted between the fingers.
- Then: fingers close and hold the block.

For a headless metric check instead of visual observation:

```bash
timeout 120s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/gripper_physics_test.py --headless --smoke_steps 180
```

## 3. Visible Manipulation-Control Demo: fixed-base arm tracks a smooth target

This opens Isaac Sim with only the robot in an empty scene. The robot starts from the URDF zero upright posture, the base is fixed/locked, navigation is disabled, and the right arm follows a smooth 6D target. Frame markers show current and target end-effector poses.

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/arm_chain_isolated_test.py --smoke_steps 0 --trajectory_steps 720
```

Expected visual sequence:
- Base does not drift.
- Right arm moves smoothly toward the target marker.
- No navigation or obstacle variables are involved.

For a headless metric check:

```bash
timeout 160s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/arm_chain_isolated_test.py --headless --smoke_steps 360
```

## 4A. Default Official Visual Grasp: v28 original on Current Head RGB-D Scene

This is the only accepted visual perception path for current Task319 work. It runs the v28 `qwen_first` pipeline on the current Task319 `head_rgbd` RGB frame with `viss/models/yolo11s-seg-best.pt`: Qwen first proposes global trash candidates, YOLO refines each ROI mask, and Qwen verification annotates the refined crop. Task319 keeps the v28 README defaults for confidence, ROI expansion, and timeout, but does not cap candidate count by default because the full-table scene can contain many objects. Qwen verification is enabled by default because classification is the authority for bin selection. `detections` and `approach_candidates` are eligible for grasp; `rejected_detections` are saved for debugging only. The returned 2D polygon/category is adapted back into the current RGB-D chain; 3D object centers and reachability still use the current camera intrinsics, depth, and `camera_to_world`. GLM, v27, the older merged adapter, and the legacy full-frame YOLO path are retired from the mainline.

Selection policy:
- Each cycle selects at most one object.
- Hard filters remove invalid mask/RGB-D/VLM candidates.
- When `--execute_grasp` is set, right-arm reachability is a hard gate; unreachable objects are not grabbed and no navigation starts.
- For RGB-D centroid fallback, reachability is checked at the same RGB-D geometric center used by the grasp command. Extra object-edge/top probe points are disabled by default and can be restored only for diagnostics with `--no-target_reachability_center_only`.
- Valid candidates are ranked by weighted score: planner group, IK cost, distance, clutter, depth point quality, VLM confidence, and recent failed-grasp memory.
- After a successful grasp and drop, `--enable_sort_nav` returns the robot to the home table-front pose through the bin staging and side-corridor waypoints.
- In the current `--mind_sort_demo` runtime profile, the state machine first
  attempts physical right-gripper grasp/lift. If physical lift verification
  fails but the final low-grasp TCP reached the object within the 2 cm gate,
  `--mind_sort_gripper_proximity_assist` is auto-enabled and the object is
  carried in the right-gripper TCP frame for demo continuity. Strict physical
  validation must pass `--no-mind_sort_gripper_proximity_assist`.
- In the current `--mind_sort_demo` runtime profile, right-arm motion is
  switched to `curobo_right_arm` unless the command explicitly overrides
  `--arm_motion_backend`. The older `local_position_primitive` can still be
  forced for the 20260624 stable debug-cube profile with
  `--mind_sort_force_stable_grasp_profile`.

API setup, in the same shell before launching:

```bash
export DASHSCOPE_API_KEY='your DashScope API key'
export QWEN_MODEL='qwen3-vl-flash'
export QWEN_API_STYLE='dashscope'
```

Qwen API check:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python viss/scripts/perception/yolo11_qwen_perception_offline.py --qwen-api-check --qwen-api-style dashscope
```

GUI visual-grasp test with default v28/Qwen perception and RGB-D centroid grasp:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --num_cycles 1 --skip_graspnet --use_centroid_fallback --execute_grasp --record_debug --record_video --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.010 --post_action_hold_steps 1200
```

GUI full-chain test: v28 original perception -> RGB-D centroid grasp -> Nav2 to matching bin side -> drop:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --num_cycles 1 --skip_graspnet --use_centroid_fallback --execute_grasp --enable_sort_nav --record_debug --record_video --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.010 --nav2_stack_startup_s 2 --nav2_goal_timeout_s 150 --post_action_hold_steps 1200
```

Current default mind-sort loop command, using formal v28/Qwen perception,
dynamic RGB-D station planning, Nav2, cuRobo right-arm motion, and the 2 cm
gripper-proximity continuation policy:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

Strict physical-grasp validation disables the demo carry gate:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo --no-mind_sort_gripper_proximity_assist
```

Latest full-loop video demonstration command. This uses the legacy suction flag
as an alias for the gripper-proximity carry path and is for complete sorting
video evidence, not physical-gripper acceptance:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --mind_sort_demo \
  --mind_sort_suction_assist \
  --no-mind_sort_gripper_proximity_assist \
  --mind_sort_allow_stale_reshoot_for_suction_demo \
  --no-target_reachability_ik_check \
  --record_video \
  --video_width 1280 \
  --video_height 720 \
  --video_sample_stride 4 \
  --no-gui_realtime_playback
```

Headless record/debug variant:

```bash
timeout 480s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --skip_graspnet --use_centroid_fallback --execute_grasp --record_debug --record_video --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.010 --post_action_hold_steps 0
```

Expected debug files in the cycle directory:
- `target_candidates.json`: hard-filter and weighted-score details for every visual candidate
- `v28_original/yolo_seg_offline_result.json`
- `v28_original/yolo11_qwen_overlay_latest.jpg`
- `v28_original/v28_original_instances.json`
- `v28_original/v28_original_instances_overlay.png`
- `v28_original/v28_mask_XX.png`
- `target_pose_debug.jpg/json`
- `post_grasp_sort_nav.json` when `--enable_sort_nav` is used and grasp verification succeeds

Disabled by default:
- Old full-frame YOLO debug files: `yolo_all_overlay.png`, `yolo_all_detections.json`. Re-enable only with `--save_legacy_yolo_debug`.
- Old grasp debug files: `graspnet_overlay.png`, `selected_grasp_overlay.png`, `debug_graspnet_input_world.ply/json`, `graspnet_workspace_mask.png`. Re-enable only with `--save_legacy_grasp_debug`.

## 4A-Calib. Isolated Right-Arm Single-Cube IK/TCP Calibration

This bypasses VLM and GraspNet. It creates only the robot, cameras, a small
calibration table, and one movable debug cube, then sends the right-arm grasp
command. Use this when debugging right-arm IK, TCP offset, and unreasonable
wrist posture without table/trash clutter.

Movable truth-source cube calibration with the default cuRobo right-arm adapter:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --debug_cube_grasp_demo --debug_cube_isolated_scene --debug_cube_pos 0.70,-0.16,0.5625 --arm_motion_backend curobo_right_arm --num_cycles 1 --execute_grasp --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 6 --warmup_steps 60 --cycle_interval_steps 1 --trajectory_steps 180 --safe_pregrasp_steps 80 --grasp_steps 100 --lift_steps 120 --hold_steps 40 --max_joint_step 0.02 --post_action_hold_steps 20 --no-gui_realtime_playback
```

RGB-D target-source calibration uses the same scene and controller, but drives
the grasp from the head RGB-D backprojected cube center while still auditing the
simulator root:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --debug_cube_grasp_demo --debug_cube_rgbd_target --debug_cube_isolated_scene --debug_cube_pos 0.70,-0.16,0.5625 --arm_motion_backend curobo_right_arm --num_cycles 1 --execute_grasp --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 6 --warmup_steps 60 --cycle_interval_steps 1 --trajectory_steps 180 --safe_pregrasp_steps 80 --grasp_steps 100 --lift_steps 120 --hold_steps 40 --max_joint_step 0.02 --post_action_hold_steps 20 --no-gui_realtime_playback
```

Expected calibration outputs:
- `cycle_0000/rgbd_vs_truth_debug.json`
- `cycle_0000/rgbd_vs_truth_debug.png`
- `cycle_0000/target_pose_debug.json`
- `cycle_0000/arm_trajectory_debug.json`
- `cycle_0000/gripper_alignment_debug.json`
- `external_grasp_demo.mp4`

Notes:
- The isolated scene now includes a small calibration table, so the default cube
  is movable and suitable for physical lift validation. Add
  `--debug_cube_static` only for pure IK/TCP freeze-frame checks.
- The current cuRobo adapter uses the explicit right-arm joint order
  `zarm_r1_joint` through `zarm_r7_joint`, clamps the planning start state
  against URDF limits, and writes `curobo_joint_adapter` plus per-stage frame
  diagnostics into `metadata.json`.
- The current conservative run at `0.70,-0.16,0.5625` derives the gripper TCP
  from `two_finger_gripper.urdf`: the local TCP is `[0.115, 0.0, 0.0]` in
  `gripper_base`, which maps to about `[0.0, 0.0, -0.115]` from the right wrist.
- On `20260624_184021`, cuRobo successfully planned/executed the pregrasp,
  grasp, and lift segments without the earlier
  `INVALID_START_STATE_JOINT_LIMITS` failure. The remaining failure was gripper
  contact/lift verification: `No object was verified above the table after lift.`
- On `20260624_124545`, the first position-only GRASP segment stopped with
  `0.0279 m` TCP error, but the measured TCP residual feedback correction
  reduced the final TCP error to `0.0044 m`; the finger-pad midpoint was
  `0.0045 m` from the cube center. This avoids the previous unreachable
  `x=0.84-0.98` edge-of-workspace targets and confirms the remaining offset is
  mainly IK tracking residual, not gripper URDF geometry.

Experimental waist/leg whole-body IK assist checks:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --debug_cube_grasp_demo --debug_cube_isolated_scene --debug_cube_static --debug_cube_pos 0.70,-0.16,0.5625 --whole_body_ik_assist waist --num_cycles 1 --execute_grasp --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 4 --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --grasp_steps 220 --lift_steps 260 --hold_steps 160 --max_joint_step 0.010 --post_action_hold_steps 240 --no-gui_realtime_playback

/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --debug_cube_grasp_demo --debug_cube_isolated_scene --debug_cube_static --debug_cube_pos 0.70,-0.16,0.5625 --whole_body_ik_assist waist_leg --num_cycles 1 --execute_grasp --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 4 --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --grasp_steps 220 --lift_steps 260 --hold_steps 160 --max_joint_step 0.010 --post_action_hold_steps 240 --no-gui_realtime_playback
```

Result summary:
- `waist` uses IsaacLab DLS IK with `waist_.*_joint + zarm_r[1-7]_joint`.
  Current isolated-cube run `20260624_125731` reached GRASP but ended around
  `5-6 cm` TCP error, worse than the right-arm-only calibrated path.
- Coordinate diagnostics are saved under
  `gripper_alignment_after_* .coordinate_frames`. The key fields are
  `target_in_root_m`, `tcp_in_root_m`, `target_in_right_arm_base_m`,
  `tcp_in_right_arm_base_m`, `target_in_right_ee_m`, and `tcp_in_right_ee_m`,
  plus `whole_body_joint_positions_rad`.
- Short coordinate validation run `20260624_130312` reached `0.0030 m` TCP
  error after feedback and verified that the target and TCP are coherent in the
  moving right-EE frame after waist motion.
- `waist_leg` uses `knee_joint + leg_joint + waist_.*_joint + zarm_r[1-7]_joint`.
  Current run `20260624_125911` failed at PRE_GRASP, so it is not ready for
  formal grasp execution.
- Recommended current formal setting remains `--whole_body_ik_assist off`.
  Treat waist/leg as reachability/standpoint diagnostics until a constrained
  posture pre-shape or cuRobo whole-body plan is added.

Conservative torso pre-shape assist, with right-arm TCP execution kept as the
formal grasp controller:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --debug_cube_grasp_demo --debug_cube_isolated_scene --debug_cube_static --debug_cube_pos 0.70,-0.16,0.5625 --whole_body_ik_assist off --torso_preshape_assist --torso_preshape_probe_steps 60 --torso_preshape_move_steps 100 --num_cycles 1 --execute_grasp --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 4 --warmup_steps 60 --cycle_interval_steps 1 --trajectory_steps 260 --grasp_steps 120 --lift_steps 160 --hold_steps 80 --max_joint_step 0.010 --post_action_hold_steps 80 --no-gui_realtime_playback
```

Result summary:
- Run `20260624_131837` sampled waist/stance postures but rejected the best one
  because its dry-run TCP error was `0.203 m`, above the default
  `--torso_preshape_apply_error_threshold 0.08`.
- The controller therefore kept the current posture and used the calibrated
  right-arm TCP path. GRASP TCP error was `0.016 m`; the feedback correction
  reduced it to `0.010 m`.
- Video:
  `task_319_garbage_sort/output/head_camera_grasp_records/20260624_131837/external_grasp_demo.mp4`.

## 4B. Current v28/Nav2 Mind-Sort Mainline

This keeps the v28/VISS perception and Nav2 dynamic-standpoint loop, and tries
the real right-gripper grasp/lift before bin navigation. The default runtime
also enables the 2 cm gripper-proximity carry continuation, so a full-loop demo
can continue if the gripper reached the object but lift verification failed.
Use `--no-mind_sort_gripper_proximity_assist` for strict physical-grasp
validation.

GUI mainline command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

Strict physical-grasp command:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo --no-mind_sort_gripper_proximity_assist
```

Explicit navigation-display command without physical grasp:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo --mind_sort_simulated_pick
```

Expected debug files:
- `mind_sort_demo/cycle_XXXX/standpoint_reshoot/v28_original/`
- `mind_sort_demo/cycle_XXXX/physical_grasp/physical_grasp_execution.json`
- `mind_sort_demo/cycle_XXXX/physical_grasp/rgbd_vs_truth_debug.json`
- `mind_sort_demo/cycle_XXXX/physical_grasp/arm_trajectory_debug.json`
- `mind_sort_demo/cycle_XXXX/physical_grasp/gripper_alignment_debug.json`
- `mind_sort_demo/cycle_XXXX/mind_sort_cycle.json`
- `mind_sort_demo/cycle_XXXX/physical_grasp/fsm_trace.json`
- `mind_sort_demo/cycle_XXXX/physical_grasp/wrist_local_view/wrist_refine_rgb.png`
- `mind_sort_demo/cycle_XXXX/physical_grasp/wrist_local_view/wrist_refine_debug.json`

Current diagnostic status:
- The physical state is wired into the loop and records video/metadata.
- The latest complete demo run is
  `task_319_garbage_sort/output/head_camera_grasp_records/20260625_150100`.
  It sorted 8 objects with `object_teleport_used=false` at drop time, then
  stopped on a final return-to-observe Nav2 `STATUS_6`.
- Real physical gripper pickup remains the main open issue. SAC training in
  `task319_local_grasp_rl/` is intended to replace the open-loop final descent,
  close, and lift stage.

## 4. Retired Legacy Strict Visual Grasp: YOLO + GLM-VLM

Retired. Do not use this for current Task319 work. The current mainline is section 4A/4B with VISS v28 and Qwen/VLM; GLM and the old full-frame YOLO chain are kept only as historical notes.

Dependency/model check before strict mode:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/scripts/prepare_grasp_deps.py --download_missing_models
```

Headless strict visual-grasp acceptance command using GraspNet:

```bash
export GLM_API_KEY='your GLM API key'
timeout 420s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --execute_grasp --strict_model_chain --record_debug --record_video --vlm_model glm-5v-turbo --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.010 --mask_distance_threshold_m 0.02 --min_filtered_grasps 8 --ik_prescreen_top_k 20 --ik_prescreen_max_joint_delta_rad 4.5 --max_grasp_retries 3 --pregrasp_error_threshold_m 0.07 --grasp_error_threshold_m 0.06 --lift_error_threshold_m 0.08
```

Expected result:
- Kuavo stays upright with fixed-base manipulation and `base_drift_m` near `0.0`.
- The target object is lifted more than `0.05 m` above the table and held through `VERIFY_HOLD`.
- `metadata.json` reports `strict_success: true`, `target_source: yolo`, `planned_grasp_source: graspnet`, `execution_grasp_source: graspnet`, `graspnet_filtered_count > 0`, and `execution.grasp_success: true`.

GUI strict visual-grasp command using GraspNet:

```bash
export GLM_API_KEY='your GLM API key'
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --num_cycles 1 --execute_grasp --strict_model_chain --record_debug --record_video --vlm_model glm-5v-turbo --target_object_name potted --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.010 --post_action_hold_steps 1200
```

Headless motion-evidence video command without VLM/GraspNet dependencies:

```bash
timeout 360s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --execute_grasp --skip_vlm --skip_graspnet --use_centroid_fallback --target_object_name potted --record_debug --record_video --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.010
```

GUI strict visual grasp + experimental bin-drop navigation:

```bash
export GLM_API_KEY='your GLM API key'
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --num_cycles 1 --execute_grasp --strict_model_chain --enable_sort_nav --record_debug --vlm_model glm-5v-turbo --target_object_name potted
```

Notes:
- `--enable_sort_nav` is being migrated to ROS2/Nav2. Do not use the old internal waypoint follower as the accepted navigation path.
- API keys must stay in environment variables: `GLM_API_KEY`, `ZHIPUAI_API_KEY`, or `BIGMODEL_API_KEY`.
- `fallback_success=true` is only a debug/degraded result. It does not count as strict visual grasp success unless `grasp_source` matches the selected `--grasp_backend`.
- Learned grasp mask filtering is relaxed by default: strict 2D mask, raw 2D mask, then 3D nearest-distance acceptance within `--mask_distance_threshold_m`.
- Execution prescreens up to `--ik_prescreen_top_k` learned grasp candidates with `--ik_prescreen_max_joint_delta_rad`, retreats toward the current candidate pre-grasp pose after rejected attempts, then retries up to `--max_grasp_retries` IK-reachable poses before entering `RECOVER` or degraded fallback.
- In GUI mode, YOLO/VLM and grasp planning run in background workers by default via `--async_model_inference`, while the main thread keeps the Isaac viewport responsive. Heavy GPU inference can still reduce frame rate; use `--no-async_model_inference` only for debugging.
- Arm/gripper execution after planning is paced by `--gui_realtime_playback`, and `--post_action_hold_steps` keeps the final scene visible.
- `--record_video` writes `external_grasp_demo.mp4` from an external observer camera in the run directory. This camera is only for evidence recording; perception still uses `head_rgbd`.
- The observer video angle can be changed with `--observer_camera_pos x,y,z --observer_camera_target x,y,z`; this does not affect grasp perception or execution.

Direct ROS2 `/cmd_vel` Isaac-control demo:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --ros_cmd_vel_demo \
  --record_debug \
  --record_video \
  --ros_cmd_vel_demo_steps 240 \
  --ros_cmd_vel_demo_linear_x 0.20 \
  --ros_cmd_vel_demo_angular_z 0.0 \
  --ros_cmd_vel_demo_min_translation_m 0.05 \
  --video_sample_stride 4 \
  --post_action_hold_steps 240
```

Headless ROS2 `/cmd_vel` verification:

```bash
timeout 480s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --ros_cmd_vel_demo \
  --record_debug \
  --record_video \
  --ros_cmd_vel_demo_steps 240 \
  --ros_cmd_vel_demo_linear_x 0.20 \
  --ros_cmd_vel_demo_angular_z 0.0 \
  --ros_cmd_vel_demo_min_translation_m 0.05 \
  --video_sample_stride 4 \
  --post_action_hold_steps 0
```

ROS2 `/cmd_vel` notes:
- This pauses vision/grasp and validates that a ROS2 `geometry_msgs/Twist` on
  `/cmd_vel` causes measured Isaac robot base motion.
- By default, the demo auto-starts `scripts/ros2_cmd_vel_test_publisher.py`.
  Add `--no-ros_cmd_vel_demo_auto_publish` to drive `/cmd_vel` from an external
  ROS2 node instead.
- Current Isaac validation consumes `linear.x` and `angular.z`. The separate
  Kuavo ROS1 bridge can preserve and republish `linear.y` to `/cmd_vel` or
  `/cmd_vel_world`, but the Isaac differential-style validation mode does not
  apply lateral velocity.

Motion-only ROS2/Nav2 sorting demo:

```bash
timeout 700s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --motion_only_sort_demo \
  --nav_backend nav2 \
  --motion_test_all_categories \
  --record_debug \
  --record_video \
  --warmup_steps 80 \
  --nav2_goal_timeout_s 150
```

Motion-only Nav2 notes:
- This pauses YOLO/GLM/depth/GraspNet and validates only the base movement/drop state machine.
- Isaac auto-sources `task_319_garbage_sort/scripts/setup_nav2_user_install.bash` and starts the bundled minimal Nav2 stack by default through `scripts/ros2_nav2_stack_launcher.py`.
- Isaac launches `scripts/ros2_nav_socket_bridge.py` under ROS Python to publish `/odom`, `/scan`, `/tf` and receive Nav2 `/cmd_vel`.
- Each movement goal is sent through `scripts/ros2_nav_goal_client.py` to Nav2 `NavigateToPose`; there is no fallback to the old internal follower.
- To connect to an already-running external Nav2 stack instead, add `--no-start_nav2_stack`.

Important waypoint Nav2 kinematic baseline:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --waypoint_nav_demo \
  --waypoint_route home,table_left,bin_center,table_right,home \
  --nav_backend nav2 \
  --wheel_drive_model mecanum45 \
  --wheel_ground_coupling kinematic \
  --record_debug \
  --record_video \
  --warmup_steps 80 \
  --nav2_stack_startup_s 2 \
  --post_action_hold_steps 240
```

Headless waypoint kinematic-baseline verification:

```bash
timeout 900s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --waypoint_nav_demo \
  --waypoint_route home,table_left,bin_center,table_right,home \
  --nav_backend nav2 \
  --wheel_drive_model mecanum45 \
  --wheel_ground_coupling kinematic \
  --record_debug \
  --record_video \
  --warmup_steps 80 \
  --nav2_stack_startup_s 2
```

Stable URDF-diagonal waypoint Nav2 demo:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --waypoint_nav_demo \
  --waypoint_route home,table_left \
  --nav_backend nav2 \
  --wheel_drive_model urdf_diagonal \
  --wheel_ground_coupling kinematic_stable \
  --wheel_velocity_scale 0.35 \
  --record_debug \
  --record_video \
  --warmup_steps 80 \
  --nav2_stack_startup_s 2 \
  --nav2_goal_timeout_s 150 \
  --post_action_hold_steps 240 \
  --video_sample_stride 4
```

Headless stable URDF-diagonal waypoint verification:

```bash
timeout 720s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --waypoint_nav_demo \
  --waypoint_route home,table_left \
  --nav_backend nav2 \
  --wheel_drive_model urdf_diagonal \
  --wheel_ground_coupling kinematic_stable \
  --wheel_velocity_scale 0.35 \
  --record_debug \
  --record_video \
  --warmup_steps 80 \
  --nav2_stack_startup_s 2 \
  --nav2_goal_timeout_s 150 \
  --post_action_hold_steps 0 \
  --video_sample_stride 4
```

Dynamic table-side standpoint Nav2 demo:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --dynamic_standpoint_nav_demo \
  --dynamic_target_world_xyz 1.20,0.00,0.60 \
  --dynamic_allowed_table_sides front,left,right \
  --nav_backend nav2 \
  --wheel_drive_model urdf_diagonal \
  --wheel_ground_coupling kinematic_stable \
  --wheel_velocity_scale 0.35 \
  --record_debug \
  --record_video \
  --warmup_steps 80 \
  --nav2_stack_startup_s 2 \
  --nav2_goal_timeout_s 150 \
  --post_action_hold_steps 240 \
  --video_sample_stride 4
```

Headless dynamic table-side standpoint verification:

```bash
timeout 720s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --dynamic_standpoint_nav_demo \
  --dynamic_target_world_xyz 1.20,0.00,0.60 \
  --dynamic_allowed_table_sides front,left,right \
  --nav_backend nav2 \
  --wheel_drive_model urdf_diagonal \
  --wheel_ground_coupling kinematic_stable \
  --wheel_velocity_scale 0.35 \
  --record_debug \
  --record_video \
  --warmup_steps 80 \
  --nav2_stack_startup_s 2 \
  --nav2_goal_timeout_s 150 \
  --post_action_hold_steps 0 \
  --video_sample_stride 4
```

Dynamic standpoint notes:
- This is the first interface for `target world/table coordinate -> candidate robot ground standpoint -> Nav2 goal`.
- Vision and grasping are paused in this mode; `--dynamic_target_world_xyz` is the manual stand-in for the future RGB-D object center.
- Only `front`, `left`, and `right` are valid grasp sides. The bin side remains a separate bin-staging/bin-target navigation role.
- The demo writes `dynamic_standpoint_nav/dynamic_standpoint_candidates.json` and `dynamic_standpoint_nav/dynamic_standpoint_nav_metadata.json`.
- `--dynamic_ik_diagnostics` is off by default. It temporarily repositions the robot for dry-run IK probes and is diagnostic only; do not enable it for continuous navigation-loop videos unless you are explicitly debugging IK reachability.

Main visual-chain standpoint Nav2 verification, without grasping:

```bash
timeout 420s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --num_cycles 1 \
  --perception_source v28_original \
  --standpoint_nav_only \
  --record_debug \
  --record_video \
  --video_width 1280 \
  --video_height 720 \
  --video_sample_stride 4 \
  --warmup_steps 80 \
  --cycle_interval_steps 1 \
  --nav_backend nav2 \
  --nav_actuation_mode wheel \
  --wheel_drive_model urdf_diagonal \
  --wheel_ground_coupling kinematic_stable \
  --wheel_velocity_scale 0.35 \
  --nav2_stack_startup_s 2 \
  --nav2_goal_timeout_s 150 \
  --post_action_hold_steps 120 \
  --no-gui_realtime_playback
```

This runs the real v28/VISS perception on the head RGB-D frame, computes the
dynamic grasp standpoint from the selected object's measured 3D center, sends
that pose to Nav2, saves `pre_grasp_standpoint_nav.json`, then re-shoots under
`standpoint_reshoot/`.

Mind-sort physical-grasp mainline:

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

This is the delivery command for the current mind-sort mainline. It combines the
high-accuracy v28/VISS visual chain (`qwen_first -> YOLO ROI segmentation ->
Qwen crop verification`) with the complete Nav2 video loop and real physical
gripper pickup. The physical grasp default is cuRobo right-arm planning with
the angled top-down wrist geometry (`--arm_motion_backend curobo_right_arm`,
`--arm_motion_wrist_orientation angled_top_down`) and cuRobo-first final grasp
descent. The long-form video/navigation defaults are now built in: all-object
loop, 1280x720 video, high observer camera, `wheel + urdf_diagonal +
kinematic_stable`, unlimited Qwen/Nav2 wait, and `record_video` enabled. Do not
wrap this command in shell `timeout` when recording the final video. The output
is written to `mind_sort_demo/mind_sort_task_queue.json`.

Waypoint notes:
- `--waypoint_route` is an explicit ordered list of route endpoints. `table_left` and `table_right` are possible grasp standpoints, not mandatory pass-through points.
- Available names: `home`, `table_front`, `table_left`, `table_right`, `bin_center`, `bin_recycle`, `bin_kitchen`, `bin_hazard`, `bin_other`.
- This mode pauses vision/grasp and forces wheel-mode execution. The important baseline is `--wheel_drive_model mecanum45 --wheel_ground_coupling kinematic`: Nav2 still generates `/cmd_vel`, the code computes four wheel joint velocity targets, and Isaac integrates the planar wheel-ground motion because pure wheel-contact physics is not yet stable.
- The newer stable test path is `--wheel_drive_model urdf_diagonal --wheel_ground_coupling kinematic_stable --wheel_velocity_scale 0.35`. It keeps the baseline intact while using the S62 URDF wheel yaws and stable internal pose integration to avoid feeding wheel-contact yaw drift back into the next kinematic step.
- The high-friction/yaw-damping experiment was reverted. Current baseline values are wheel/ground friction `static=1.45`, `dynamic=1.10`; Isaac angular speed cap `0.75 rad/s`; external Nav2 angular scale `2.5`; bundled RPP `rotate_to_heading_angular_vel=0.75` and `max_angular_accel=2.4`.
- Use `official_holonomic_wheel_test.py` to isolate official Isaac wheeled-controller behavior before changing Nav2 or friction again.

Official Isaac wheeled-controller diagnostic:

```bash
timeout 260s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/official_holonomic_wheel_test.py \
  --headless \
  --record_debug \
  --controller_backend differential \
  --warmup_steps 5 \
  --drive_steps 40 \
  --cmd_vx 0.18 \
  --cmd_vy 0.0 \
  --cmd_wz 0.0
```

Wheel open-loop sanity check:

```bash
timeout 600s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --wheel_open_loop_demo \
  --record_debug \
  --record_video \
  --warmup_steps 40 \
  --wheel_open_loop_steps 240 \
  --wheel_open_loop_linear_speed 0.25 \
  --wheel_open_loop_angular_speed 0.0
```

ROS-level Nav2 smoke test without Isaac:

```bash
source task_319_garbage_sort/scripts/setup_nav2_user_install.bash

/usr/bin/python3 task_319_garbage_sort/scripts/ros2_nav2_stack_launcher.py \
  --output-dir task_319_garbage_sort/output/nav2_smoke_stack
```

In a second terminal:

```bash
source task_319_garbage_sort/scripts/setup_nav2_user_install.bash

/usr/bin/python3 task_319_garbage_sort/scripts/ros2_nav2_smoke_test.py \
  --goal-x 0.60 \
  --goal-y 0.0 \
  --goal-yaw 0.0 \
  --timeout-s 55 \
  --server-timeout-s 25 \
  --goal-attempts 5
```

Fallback physical-grasp comparison command:

```bash
timeout 300s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --execute_grasp --skip_vlm --skip_graspnet --use_centroid_fallback --target_object_name potted --record_debug --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.010
```

AnyGrasp experimental command after installing the licensed SDK assets:

```bash
export GLM_API_KEY='your GLM API key'
timeout 420s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --num_cycles 1 --execute_grasp --strict_model_chain --grasp_backend anygrasp --record_debug --record_video --vlm_model glm-5v-turbo --target_object_name potted --warmup_steps 80 --cycle_interval_steps 1 --trajectory_steps 520 --safe_pregrasp_steps 220 --grasp_steps 220 --lift_steps 300 --hold_steps 180 --max_joint_step 0.010 --mask_distance_threshold_m 0.02 --min_filtered_grasps 8 --ik_prescreen_top_k 20 --ik_prescreen_max_joint_delta_rad 4.5 --max_grasp_retries 3 --pregrasp_error_threshold_m 0.07 --grasp_error_threshold_m 0.06 --lift_error_threshold_m 0.08 --anygrasp_grasp_to_tcp 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
```

AnyGrasp notes:
- `--anygrasp_grasp_to_tcp` is a row-major 4x4 calibration from the AnyGrasp grasp frame into the Kuavo TCP frame. The identity matrix is only a smoke-test default; replace it after confirming the AnyGrasp approach axis against the Kuavo wrist/TCP frame.
- The dependency checker reports the required SDK folder, checkpoint, `gsnet`/`lib_cxx` binaries, license folder, MinkowskiEngine status, and OpenSSL 1.1 runtime status. Add `--require_anygrasp` to fail the check when any AnyGrasp asset is missing:

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/scripts/prepare_grasp_deps.py --prepare_anygrasp_binaries --require_anygrasp
```

Each cycle writes:

```text
task_319_garbage_sort/output/head_camera_grasp_records/<timestamp>/cycle_0000/
  head_rgb.png
  head_depth_mm.png
  head_depth_vis.png
  head_depth_m.npy
  yolo_overlay.png
  yolo_union_mask.png
  yolo_instances.json
  depth_component_overlay.png
  depth_components.json
  vlm_overlay.png
  vlm_results.json
  target_mask.png
  graspnet_workspace_mask.png
  graspnet_overlay.png
  selected_grasp_overlay.png
  fsm_trace.json
  metadata.json
```

## 5. Retired Head-Camera YOLO / GLM-VLM / GraspNet Recording Demo

Retired. Do not use this for current Task319 work. The maintained visual chain is VISS v28 with Qwen/VLM classification and `viss/models/yolo11s-seg-best.pt`.

GUI repeat mode for visual inspection:

```bash
export GLM_API_KEY='your GLM API key'
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/visual_grasp_record_demo.py --num_cycles 0 --cycle_interval_steps 240 --vlm_model glm-5v-turbo
```

One visible grasp attempt:

```bash
export GLM_API_KEY='your GLM API key'
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/visual_grasp_record_demo.py --num_cycles 1 --execute_grasp --vlm_model glm-5v-turbo --graspnet_num_point 8000
```

Headless one-cycle verification:

```bash
export GLM_API_KEY='your GLM API key'
timeout 180s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/visual_grasp_record_demo.py --headless --num_cycles 1 --warmup_steps 80 --cycle_interval_steps 30 --vlm_model glm-5v-turbo --graspnet_num_point 4096
```

Fast RGB-D + YOLO check without network/API calls:

```bash
timeout 120s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/visual_grasp_record_demo.py --headless --num_cycles 1 --skip_vlm --skip_graspnet
```

Each cycle writes:

```text
task_319_garbage_sort/output/head_camera_grasp_records/<timestamp>/cycle_0000/
  head_rgb.png
  head_depth_mm.png
  head_depth_vis.png
  head_depth_m.npy
  yolo_overlay.png
  yolo_union_mask.png
  yolo_instances.json
  depth_component_overlay.png
  depth_components.json
  vlm_overlay.png
  vlm_results.json
  target_mask.png
  graspnet_workspace_mask.png
  graspnet_overlay.png
  selected_grasp_overlay.png
  metadata.json
```

Notes:
- `--num_cycles 0` means repeat until you close Isaac Sim.
- Do not write the GLM API key into tracked files. Keep it in `GLM_API_KEY`, `ZHIPUAI_API_KEY`, or `BIGMODEL_API_KEY`.
- `metadata.json` records `camera_source: head_rgbd`, camera intrinsics/extrinsics, VLM classifications, target selection, GraspNet status, and execution status.
- The old dry-run below is only a terminal pipeline check; use this recording demo when you want the actual head-camera images.

## 6. Perception / Grasp Planning Dry-Run

This does not open Isaac Sim. It verifies target selection, waste classification, mask filtering, and grasp selection.

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/visual_grasp_pipeline.py --dry_run_graspnet
```

## 7. Dependency / Model Check

This confirms GraspNet, YOLO, CUDA extensions, and optional LLM dependencies are present.

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/scripts/prepare_grasp_deps.py
```
