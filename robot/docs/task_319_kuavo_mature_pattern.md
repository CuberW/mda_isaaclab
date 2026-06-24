# Task 3.19 Kuavo Feasible Pattern

## Conclusion

For the current Kuavo wheel-arm stack, the feasible architecture for Task 3.19
is not:

- MoveIt2 as the main manipulation layer
- dual-arm soft-constraint IK as the main grasp entry
- a large BT/Nav2-first refactor

The feasible path is:

1. Keep `task_319_garbage_sort/pipeline.py` as the thin task orchestrator.
2. Use Kuavo official wheel-arm precise single-arm IK as the core feasibility
   gate.
3. Use Kuavo official wheel-arm timed base/torso/arm services for macro motion.
4. Use Kuavo official `wheel_ik_ros_uni_cpp_node` for short-range incremental
   end-effector correction.
5. Use official claw services for grasp/release.
6. Keep perception, object selection, grasp candidate generation, and recovery
   policy in our own code.


## Why This Is The Right Downgrade

The key local finding is that Kuavo already contains a precise single-arm /
whole-body IK chain outside the soft dual-arm service:

- `humanoid_wheel_interface` implements
  `/mobile_manipulator_ik_accessibility_check`
- the service returns `qBest`, `bestLinearError`, `bestAngularError`
- its implementation calls:
  - `InverseKinematics::computeHandOnlyIK(...)`
  - `InverseKinematics::computeWholeBodyIK(...)`
- the solver is implemented in:
  - `third_party/kuavo-ros-opensource/src/humanoid-wheel-control/humanoid_wheel_interface/src/motion_planner/InverseKinematics.cpp`

This is much closer to what 3.19 needs than
`/ik/two_arm_hand_pose_cmd_srv`.

Also, the wheel-arm stack already exposes mature execution services from
`MobileManipulatorReferenceManager`, including:

- `/mobile_manipulator_timed_single_cmd`
- `/mobile_manipulator_timed_multi_cmd`
- `/mobile_manipulator_timed_offline_traj`
- `/mobile_manipulator_mpc_control`
- `/wheel_arm_change_arm_ctrl_mode`
- `/mobile_manipulator_ee_pose_reach_error`

So the practical architecture should be built around those official services,
not around Python-side Jacobian patching.


## What To Reuse Officially

### 1. Precise IK feasibility

Use official:

- `/mobile_manipulator_ik_accessibility_check`

Why:

- it already exposes exact-solution-or-best-solution semantics
- it returns quantitative errors
- it supports arm-only and whole-body modes
- it is backed by official wheel-arm solver code, not by our custom math

Recommended usage in 3.19:

- first try `isWholeBody=false`
- if arm-only fails but target is still promising, optionally try
  `isWholeBody=true`
- accept only when `bestLinearError` and `bestAngularError` are within our
  explicit thresholds


### 2. Macro base / torso / arm execution

Use official:

- `/mobile_manipulator_timed_single_cmd`
- `/mobile_manipulator_timed_multi_cmd`
- `/mobile_manipulator_timed_offline_traj`
- `/mobile_manipulator_mpc_control`
- `/wheel_arm_change_arm_ctrl_mode`

Why:

- these are already the official wheel-arm motion entry points
- they are easier to debug than reimplementing another controller in Python
- they let us separate "planning / decision" from "robot execution"

Recommended usage in 3.19:

- base positioning to pre-grasp: timed single/multi commands
- torso height / pitch shaping: timed commands
- retreat / lift / place macros: timed multi or offline traj


### 3. Short-range end-effector correction

Use official:

- `wheel_ik_ros_uni_cpp_node`
- launch source:
  `third_party/kuavo-ros-opensource/src/manipulation_nodes/noitom_hi5_hand_udp_python/launch/launch_quest3_ik.launch`

Why:

- this is the official C++ incremental wheel-arm IK path
- it already has the required `/quest3/*` parameters
- it is suitable for final short-range correction better than sending one large
  dual-arm soft IK jump

Recommended role in 3.19:

- only for the last 3 cm to 10 cm approach / alignment
- not for gross target acquisition
- not as the main global planner


### 4. Gripper

Use official:

- `/control_robot_leju_claw`
- `/leju_claw_state`

Why:

- grasp truth should come from real claw state and task observations
- not from simulation-only heuristics


## What We Should Write Ourselves

### 1. Task orchestration

Keep custom in Python:

- object selection
- category mapping
- retry policy
- fallback ordering
- task success bookkeeping

Reason:

- these are 3.19-specific semantics
- official Kuavo stack should not own competition logic


### 2. Perception and grasp candidate generation

Keep custom:

- detection
- mask / bbox filtering
- grasp candidate proposal
- bin mapping

Reason:

- these are scene-specific and task-specific


### 3. Pre-grasp candidate filtering

Write custom:

- generate several single-arm candidate poses around the object
- score them by reachability, collision margin proxy, and approach simplicity
- query official precise IK service to pick the first executable candidate

Reason:

- Kuavo provides the solver, but not the 3.19 object-level candidate policy


### 4. Thin adapters

Write custom wrappers around official services:

- `check_precise_ik(...)`
- `send_base_pregrasp(...)`
- `send_torso_preshape(...)`
- `execute_incremental_approach(...)`
- `close_claw_and_verify(...)`

Reason:

- this keeps `pipeline.py` readable
- this makes logging and debugging much cleaner


## Recommended 3.19 Runtime Architecture

```text
Perception / object selection / grasp proposal
  -> custom Python

Candidate reachability check
  -> official /mobile_manipulator_ik_accessibility_check

Pre-grasp base + torso motion
  -> official wheel-arm timed command services

Final short-range alignment
  -> official wheel_ik_ros_uni_cpp_node

Close claw / lift / retreat / place
  -> official claw + official timed motion services

Task retries / recovery / metrics
  -> custom Python
```


## The Feasible Execution Sequence

For each target object:

1. Detect and classify in Python.
2. Generate several single-arm grasp candidates in Python.
3. For each candidate, call
   `/mobile_manipulator_ik_accessibility_check`.
4. Select the first candidate that meets strict error thresholds.
5. Move base and torso to pre-grasp using official timed services.
6. Switch arm control mode through official wheel-arm control service.
7. Use `wheel_ik_ros_uni_cpp_node` for short-range approach refinement.
8. Close claw using official claw service.
9. Check claw state and, if needed, end-effector reach error.
10. Lift and retreat using official timed motion services.
11. Move to placement region and place with the same chain.


## The Most Important Engineering Decision

Do not make the dual-arm service the 3.19 mainline.

Mainline should be:

- single-arm precise IK check
- single-arm execution
- incremental short-range correction

Dual-arm soft IK may remain:

- for experiments
- for comparison
- for non-critical demos

But not as the core production path for 3.19.


## Plan A And Plan B

### Plan A: Use official wheel-arm service host directly

Best immediate path:

- start `humanoid_wheel_interface_ros/mobile_manipulator_mpc_node`
- or start its launch:
  `humanoid_wheel_interface_ros/launch/manipulator_kuavo_s60.launch`
- consume `/mobile_manipulator_ik_accessibility_check` and the timed motion
  services from Python

Why this is feasible:

- the service host already exists in official code
- the precise IK service is already wired into it
- no need to write a new solver first

Known prerequisites:

- `/taskFile`
- `/urdfFile`
- `/libFolder`
- `/robot_init_state_param`
- writable CppAD cache / generated library path under `/var/ocs2/...`


### Plan B: Strip and bridge the exact IK solver

If Plan A is too heavy or unstable, the best fallback is not dual-arm soft IK.
It is a thin bridge over the official wheel-arm solver code:

- `InverseKinematics.cpp`
- `FactoryFunctions.cpp`
- `ManipulatorModelInfo.h`

Recommended bridge order:

1. first build a tiny standalone C++ node or library that only loads URDF +
   task info + Pinocchio model + `InverseKinematics`
2. expose one function:
   `solve_single_arm_pose(arm, is_whole_body, pose, init_q, thresholds)`
3. only if needed, add `pybind11`

Why this order is better than jumping straight to pybind:

- easier to isolate dependencies
- easier to debug linker / Pinocchio / OCS2 issues
- easier to compare outputs against the official ROS service


## What Is Not Worth Doing Right Now

Not recommended as current priority:

- MoveIt2 as the main 3.19 manipulation layer
- MTC as the first integration step
- Nav2-first refactor
- rewriting base control from scratch
- Python Jacobian patching on top of soft dual-arm IK

These can come later, after precise single-arm execution is stable.


## Minimal Deliverable For 3.19

The smallest architecture that is realistically capable of finishing 3.19 is:

1. current Python pipeline keeps task logic
2. official wheel-arm precise IK service becomes the candidate gate
3. official timed wheel-arm motion services become the macro executor
4. official incremental wheel IK becomes the final alignment executor
5. official claw service becomes grasp/release executor

That is the architecture I recommend treating as the real target.
