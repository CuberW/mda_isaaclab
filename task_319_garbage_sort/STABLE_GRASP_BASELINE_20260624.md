# Task319 Stable Grasp Baseline - 2026-06-24

Tag to create after commit:

```bash
task319-stable-rgbd-center-primitive-20260624
```

This baseline intentionally prioritizes a stable physical-grasp demonstration over generic 6D grasp generation.

## Main Policy

- Perception: VISS v28 / Qwen-first target detection and classification.
- Target position: current-frame RGB-D geometric center from the selected target mask.
- Grasp generator: centroid/RGB-D center, not GraspNet or AnyGrasp.
- Arm backend: `local_position_primitive`.
- IK objective: TCP position only by default.
- Wrist orientation: nominal `angled_top_down`; not a hard constraint unless `--arm_motion_enforce_wrist_orientation` is explicitly set.
- Approach: safe high pregrasp, position-only descent, slow gripper close, contact check, then lift.
- Video behavior: `--exit_after_video_saved` is on by default, so recorded runs exit right after `external_grasp_demo.mp4` and `video_manifest.json` are flushed.

## Main Command

```bash
cd mda_isaaclab
python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

## Debug-Cube Validation

Command:

```bash
cd mda_isaaclab
timeout 360s python task_319_garbage_sort/task319_grasp_sort_sm.py --headless --debug_cube_grasp_demo --debug_cube_isolated_scene --debug_cube_pos 0.70,-0.16,0.5625 --num_cycles 1 --execute_grasp --record_debug --record_video --video_width 1280 --video_height 720 --video_sample_stride 8 --warmup_steps 40 --cycle_interval_steps 1 --trajectory_steps 180 --safe_pregrasp_start --safe_pregrasp_steps 100 --grasp_steps 120 --lift_steps 140 --hold_steps 20 --post_action_hold_steps 0 --no-gui_realtime_playback
```

Result:

```text
run_dir: task_319_garbage_sort/output/head_camera_grasp_records/20260624_220310
video: task_319_garbage_sort/output/head_camera_grasp_records/20260624_220310/external_grasp_demo.mp4
execution_success: true
execution_grasp_success: true
reason: Object lifted and held.
motion_backend: local_position_primitive
```

Segment errors:

```text
safe_lift_current: 0.0307 m
safe_standoff:     0.0234 m
pregrasp:          0.0198 m
grasp:             0.0130 m
lift:              0.0264 m
```

## Notes

- `curobo_right_arm`, Kuavo IK, GraspNet, and AnyGrasp remain available as explicit diagnostic or research options.
- This baseline should be used as the rollback point if future grasp-planner experiments reintroduce twisted arm postures or table-collision-prone trajectories.
