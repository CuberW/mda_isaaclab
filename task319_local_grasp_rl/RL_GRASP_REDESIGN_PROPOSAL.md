# Task319 Local Grasp RL Redesign Proposal

Date: 2026-06-26

## 简明中文结论

现在不要直接开 RL 大训练。夹爪本身能物理夹住物体，`gripper_physics_test.py` 已验证；但当前 RL 环境里的桌面抓取脚本没有通过严格物理抓取门：无 teacher、无 latch 时，物体没有稳定进入两指之间并被抬起 25mm。

问题不是“奖励调一调”就能解决，而是任务设计不对：当前策略只控制 TCP 平移和夹爪开合，不控制抓取姿态；夹爪会斜着接近物体，物体容易被错过、推走或短暂抬起后滑脱。

更好的设计是：主线先负责算出可物理夹持的 top-down grasp frame，也就是夹爪姿态和 jaw axis 对齐物体短边；RL 只做最后几毫米的局部残差修正、闭合时机和 lift 判定。训练前必须先跑严格物理门，通过后再训。

最低训练前门槛：

```bash
cd mda_isaaclab/task319_local_grasp_rl
python strict_physical_grasp_check.py --headless --num_envs 8 --success_rate_threshold 0.80
```

这条不过，就不要开始正式 RL 训练。

## Verdict

The current RL result is weak mainly because the task design lets training
optimize the wrong thing. The policy is asked to learn reach, descend, close,
contact capture, and lift from a sparse/contact-heavy signal, while the logs can
be dominated by scripted teacher actions and the hidden grasp latch. That makes
the curves look better than the deployable policy really is.

The better design is not a larger end-to-end RL run. The better design is a
small residual local controller that sits after the existing mainline hover
planner, with behavior cloning first and RL only as a fine-tuning step.

## Current Problems

1. Hidden latch contaminates success.

   `grasp_latch_enabled=True` turns a near-target close command into an
   attachment-like object motion. That is useful for debugging a state machine,
   but it is not valid evidence that the physical gripper captured the object.

2. Teacher mixing contaminates policy metrics.

   SAC defaults to teacher mixing. Some successful runs have
   `teacher_action_fraction=1.0`, so the measured success belongs to the
   scripted controller, not the learned policy.

3. The action space is too broad for the data available.

   Four continuous outputs must discover the whole sequence:
   XY correction, final descent, close timing, and lift. The stable mainline
   already knows most of this sequence. The policy should not relearn it from
   scratch.

4. Observation is too thin for grasp timing.

   The actor sees a target point and proprioception, but not enough deployable
   object geometry: jaw-axis frame, object footprint size, top height margin,
   mask confidence, or whether the target is near a clampable width.

5. Reward has a high-risk scale mix.

   Large success and lift rewards, latch rewards, dense alignment rewards, and
   premature-close penalties are all active together. This makes it easy to get
   high reward from a shaped shortcut while still failing physical capture.

6. Evaluation is not separated from training assists.

   There is no hard gate saying: no teacher, no latch, no scripted action
   replacement, fixed evaluation seeds, and explicit physical lift metrics.

## Proposed Architecture

### Mainline Boundary

Keep the existing mainline responsible for:

- VISS/Qwen target selection.
- RGB-D top grasp target generation.
- Nav2 grasp standpoint.
- cuRobo/Kuavo IK to the hover pose.
- Safe fallback primitive if learning policy is disabled or fails.

The learned policy starts only after the right gripper TCP has reached the hover
pose above the selected object.

### Policy Role

Use RL as a residual local controller, not as the whole grasp policy.

Base primitive:

1. Lock hover TCP X/Y.
2. Descend vertically to `Z_max - grasp_depth`.
3. Close slowly at the low target.
4. Lift vertically.

Learned residual:

- Small XY correction around the locked hover point.
- Small Z-speed correction during descent.
- Close timing adjustment.
- Abort/hold signal when the target becomes unsafe.

This keeps the learned policy inside a narrow, physically meaningful operating
region.

## Environment V2

Create a new env id rather than mutating old logs:

```text
Task319-Residual-Hover-Grasp-Kuavo-Direct-v0
```

Recommended files:

```text
task319_local_grasp_rl/residual_hover_grasp_env.py
task319_local_grasp_rl/agents/skrl_ppo_residual_cfg.yaml
task319_local_grasp_rl/train_bc.py
task319_local_grasp_rl/play_bc.py
task319_local_grasp_rl/evaluate_policy.py
```

### State Machine

Keep an explicit phase variable outside the policy:

```text
APPROACH_HOVER -> DESCEND -> PRE_CLOSE -> CLOSE_HOLD -> LIFT -> DONE
```

The policy receives the phase as a one-hot observation. The phase machine
enforces safety and sequence order. The policy only adjusts local parameters
within each phase.

### Action

Use a bounded residual action:

```text
action_dim = 5

0: dx_residual       [-1, 1] -> +/- 8 mm
1: dy_residual       [-1, 1] -> +/- 8 mm
2: dz_speed_scale    [-1, 1] -> 0.25x to 1.25x primitive speed
3: close_bias        [-1, 1] -> close earlier/later inside PRE_CLOSE
4: hold_or_abort     [-1, 1] -> hold if positive, abort if strongly negative
```

Do not let the policy command arbitrary lift or arbitrary horizontal motion.
Lift remains a vertical primitive after a validated close/capture phase.

### Actor Observation

Actor inputs must stay deployable:

```text
target_rel_tcp_grasp_frame: 3
tcp_velocity_grasp_frame: 3
gripper_width_and_command: 2
right_arm_joint_pos_scaled: 7
right_arm_joint_vel_scaled: 7
phase_one_hot: 5
last_action: 5
object_geometry: 6
  - estimated footprint length
  - estimated footprint width
  - estimated top height above table
  - mask point count / confidence
  - target depth confidence
  - jaw-axis alignment confidence
contact_or_proxy: 2
  - finger closure residual
  - close-time object motion proxy
```

Approximate actor dimension: `40`.

The object geometry comes from the same RGB-D mask/candidate code used by the
mainline. It should not use simulator-only object root pose at inference.

### Privileged Critic

Use asymmetric actor-critic for RL fine-tuning:

- Actor: deployable observation only.
- Critic: actor observation plus simulator truth:
  object pose, object velocity, contact flags, object displacement, true lift,
  and true TCP-object distance.

This improves learning without leaking simulator truth into deployment.

### Physics And Success

Training and evaluation defaults:

```text
grasp_latch_enabled = false
teacher_mixing = false
teacher_only = false
```

Success condition:

- Gripper close command is high.
- Object is lifted at least `25 mm`.
- Object remains between/near the fingers for the lift hold window.
- Object XY displacement before close is below threshold.
- No table penetration or hidden object pose write was used.

Keep latch only as an explicit debug mode:

```text
--debug_grasp_latch
```

Never count latch-enabled runs as policy success.

## Reward V2

Use smaller normalized rewards. Avoid one huge `800` success bonus.

Recommended per-step reward:

```text
r =
  + 1.0 * potential_progress_to_preclose
  + 0.5 * jaw_axis_alignment
  + 0.8 * close_when_in_capture_window
  + 1.2 * opposing_finger_contact_or_proxy
  + 2.0 * lift_progress_after_capture
  + 8.0 * terminal_success
  - 1.0 * premature_close
  - 2.0 * object_push_before_close
  - 2.0 * table_contact
  - 0.1 * action_rate
  - 0.02 * time
```

Important gates:

- Closing reward is zero unless the TCP is inside the capture window.
- Lift reward is zero unless close/capture is validated.
- XY residual reward is clipped after the residual leaves the safe envelope.
- Object push before close terminates early in strict curriculum stages.

## Data Strategy

### Phase 1: Demonstration Dataset

Use the stable mainline primitive as the first teacher, but record real
executed actions and outcomes, not hidden latch outcomes.

Minimum useful dataset:

```text
30-50 successful episodes: validate collector and BC loop
200-300 successful episodes: first useful BC policy
800-1500 successful episodes: robust local policy
20-30% failed episodes: keep for analysis and later DAgger/filtering
```

The current teleop dataset has only `3` demos and `0` successes, so it is not
enough for BC or RL.

### Phase 2: Behavior Cloning

Train BC before RL:

```text
input: actor observation
target: residual action relative to the base primitive
loss: Huber/L1 for residual motion, BCE or focal loss for close/hold signals
split: by episode, not by frame
```

Acceptance before RL:

```text
>= 60% sim success on held-out random seeds
0 latch
0 teacher action replacement
object push before close < 10 mm median
```

### Phase 3: RL Fine-Tuning

Start from the BC policy. Fine-tune only inside the residual env.

Recommended first algorithm:

- PPO for residual controller stability.
- Short horizon: 80-120 control steps.
- 256-512 parallel envs.
- No teacher action replacement in metric runs.
- Optional imitation auxiliary loss to prevent policy drift.

If using SAC:

- Increase replay memory substantially; `4096` is too small for this problem.
- Do not use teacher-mixed rollouts as deployable success evidence.
- Keep separate evaluation runs every checkpoint with teacher/latch disabled.

## Curriculum

Stage A: residual reach only.

- Object fixed.
- No gripper close.
- Reward only target frame convergence and low action rate.

Stage B: descend without push.

- Object randomized in a small XY band.
- Close disabled.
- Penalize object motion before close.

Stage C: close timing.

- Enable close action.
- No lift reward until close is inside the capture window.

Stage D: lift after capture.

- Enable lift phase.
- Success requires physical retained lift.

Stage E: domain randomization.

- Object XY/yaw.
- Object footprint size.
- Mass/friction/contact offsets.
- RGB-D target noise.
- Hover pose error from mainline.

Do not advance curriculum by training steps alone. Advance only when evaluation
metrics pass.

## Evaluation Protocol

Every checkpoint must run:

```text
evaluate_policy.py \
  --checkpoint CHECKPOINT \
  --num_envs 256 \
  --episodes 2048 \
  --no-teacher_mixing \
  --no-debug_grasp_latch \
  --fixed_eval_seeds
```

Report:

- `success_rate`
- `lift_retained_rate`
- `opposing_contact_rate`
- `premature_close_rate`
- `object_push_before_close_median`
- `table_contact_rate`
- `mean_time_to_capture`
- `fallback_required_rate`

Promotion gates:

```text
sim random eval success >= 70%
premature close <= 5%
table contact <= 2%
median object push before close <= 8 mm
mainline A/B test improves over vertical primitive
```

## Mainline Integration

Add only an explicit opt-in flag:

```text
--mind_sort_local_grasp_policy path/to/policy.pt
```

Runtime behavior:

1. Mainline reaches hover with existing planner.
2. Build actor observation from live RGB-D target, grasp frame, TCP state, and
   gripper state.
3. Run residual policy at the local control rate.
4. Clamp residual XY and Z-speed.
5. Abort to the deterministic vertical primitive if:
   - policy asks to leave the safe envelope,
   - target confidence is low,
   - object moves too much before close,
   - timeout occurs,
   - TCP tracking error exceeds threshold.
6. Count success only after physical lift verification.

Default `--mind_sort_demo` should keep using the deterministic mainline until
the policy passes the promotion gates.

## Implementation Order

1. Add V2 residual env with latch disabled by default.
2. Add deterministic primitive wrapper and residual action conversion.
3. Extend recorder to save mainline primitive episodes as residual-action HDF5.
4. Add `train_bc.py` and `play_bc.py`.
5. Add strict `evaluate_policy.py`.
6. Train BC on 200+ successful episodes.
7. Fine-tune with PPO in V2 env.
8. Add explicit opt-in mainline integration flag.
9. Run A/B test against the current vertical TCP descent primitive.

## Expected Outcome

This design should make the learned policy answer a narrow question:

```text
Given a mostly correct top-down hover grasp, how should I make millimeter-scale
local corrections and decide the close timing?
```

That is a realistic learning problem. The current design asks the policy to
rediscover a full contact-rich manipulation sequence while training metrics can
be satisfied by teacher/latch shortcuts. The V2 design removes those shortcuts,
keeps the stable mainline as the safety baseline, and gives RL a scoped role
where it can actually improve final grasp robustness.
