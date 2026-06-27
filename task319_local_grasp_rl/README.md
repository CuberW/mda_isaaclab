# Task319 Hover-Descent Grasp RL

This folder contains a standalone IsaacLab DirectRLEnv for the final hover-to-grasp stage of Task 319.

Current status: SAC/PPO training is a candidate improvement path for the
Task319 mainline. It is not connected to `--mind_sort_demo` by default yet. The
intended integration point is after the mainline reaches the hover pose above a
selected object and before the final descent, gripper close, and lift.

## Scope

The policy is not a replacement for V28 perception, Nav2, cuRobo hover planning, or garbage sorting. It is a small local controller for the step after the mainline has already reached a hover pose above the selected object:

1. Main Task319 stack identifies the object and navigates to the grasp standpoint.
2. Main stack moves the right gripper to a hover pose above the object.
3. This RL policy receives the current TCP/proprioception and the RGB-D estimated grasp target relative to the TCP.
4. The policy outputs small TCP delta motions plus a continuous gripper close command.
5. Success is only counted when the physical object is lifted by the gripper. There is no suction/attachment teleport in this environment.

The actor observation is intentionally deployable in the 3.19 stack: no object root pose, object velocity, or simulator-only object state is fed to the policy. Training rewards still use simulator truth to score physical lift success and object push penalties.

During training reset, the object is randomized around the task table target, and the right arm is initialized from a cuRobo-computed hover joint pose for that reachable table region. The object is not spawned under the current TCP.

## Task

Gym id:

```text
Task319-Hover-Descent-Grasp-Kuavo-Direct-v0
```

The old `Task319-Local-Suction-Grasp-Kuavo-Direct-v0` id is still registered as a compatibility alias, but new training should use the hover-descent id above.

Environment file:

```text
task319_local_grasp_rl/local_suction_grasp_env.py
```

The robot asset is the existing Kuavo S62 URDF with the generated right gripper from:

```text
task_319_garbage_sort/gripper_robot_urdf.py
```

## Observation And Action

Observation dimension: `28`

Main observation components:

- RGB-D-style estimated grasp target relative to the current TCP
- TCP linear velocity
- right-arm 7 joint positions and velocities
- current gripper opening / close command
- TCP height and estimated target height
- previous action

Action dimension: `4`

- `dx, dy, dz`: small TCP delta command
- `gripper_close`: continuous close command; `-1` stays open, `+1` closes

The action is converted to right-arm joint targets with a damped least-squares Jacobian controller inside the environment. The controlled point is the gripper TCP derived from the two-finger gripper URDF, not the `gripper_base` body origin.

Reward terms encourage the full grasp loop, not just reaching the target:

- XY alignment with the true grasp target
- descent to `Z_max - grasp_depth`
- closing only when near the grasp target
- lifting the object physically after closure
- avoiding early closure, table contact, and pushing the object away
- success only when the close command is high, the object is lifted, and the lifted object remains near the gripper TCP

## Train

Fast smoke test:

```bash
cd mda_isaaclab
python task319_local_grasp_rl/train.py --task319_quick_test
```

Longer training:

```bash
cd mda_isaaclab
python task319_local_grasp_rl/train.py --headless --num_envs 256 --max_iterations 1500
```

SAC training:

```bash
cd mda_isaaclab
python task319_local_grasp_rl/train.py --algorithm SAC --headless --num_envs 256
```

With SAC, `--max_iterations` is interpreted as environment steps because SAC is off-policy and has no PPO rollout length.

Logs and checkpoints:

```text
logs/skrl/task319_hover_descent_grasp/
```

## W&B

W&B is off by default. Configure it with environment variables:

```bash
export WANDB_API_KEY="your_key"
export WANDB_ENTITY="your_account_or_team"
export WANDB_PROJECT="task319-hover-descent-grasp"
```

For local use, copy the example script and fill in your own account details:

```bash
cp task319_local_grasp_rl/wandb.example.sh task319_local_grasp_rl/wandb.sh
source task319_local_grasp_rl/wandb.sh
```

`task319_local_grasp_rl/wandb.sh` is ignored by git. Do not commit real API
keys.

Then run:

```bash
python task319_local_grasp_rl/train.py --headless --num_envs 256 --max_iterations 1500 --wandb
```

You can also override from CLI:

```bash
--wandb_project task319-hover-descent-grasp --wandb_entity your_account --wandb_name hover-descent-run-001
```

Default W&B metadata lives in:

```text
task319_local_grasp_rl/agents/skrl_ppo_cfg.yaml
```

## Play

After training, run a checkpoint:

```bash
python task319_local_grasp_rl/play.py --checkpoint path/to/checkpoint.pt --num_envs 16
```

## Integration Plan

After the local policy is trained:

1. Keep V28 perception and Nav2 as the global mainline.
2. Use cuRobo or the current stable hover planner to move the right gripper above the selected object.
3. Convert the live RGB-D target point (`Z_max - grasp_depth`) into the local observation format: estimated target minus current gripper TCP.
4. Run this policy only during the final descent, physical close, and lift.
5. Do not use suction/proximity attach for this trained physical-gripper policy; keep demo-only fallbacks outside the policy path.
6. Keep actor observations deployable: RGB-D estimated target, TCP/proprioception, gripper state, and previous action. Simulator truth may be used only in rewards/evaluation, not as an inference input.
