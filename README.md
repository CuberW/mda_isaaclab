# MDA IsaacLab Task319 Garbage Sorting

本仓库当前主线围绕 3.19 垃圾分类抓取任务展开：Kuavo 轮式机器人在 IsaacLab 场景中通过头部 RGB-D 相机识别桌面垃圾，选择合适站位，使用 Nav2 导航到抓取点，根据 VLM 四分类结果移动到对应垃圾桶，并完成投放。

当前正式视觉主线是 `VISS v28/Qwen-first + yolo11s-seg-best + 当前场景头部 RGB-D`。抓取主线仍在调试物理夹爪精度，演示闭环可使用夹爪近距 carry/旧吸取别名跑通全流程；该演示模式不能等同于真实物理夹取最终完成。局部 hover-to-grasp 阶段正在用 skrl SAC/PPO 训练，目标是后续替换开环下降与闭合阶段。

## Repository Layout

- `task_319_garbage_sort/`: 主任务场景、状态机、Nav2 bridge、抓取/投放逻辑、调试命令和历史记录。
- `viss/`: VISS 视觉链路。当前主线调用 `viss/scripts/perception/yolo11_qwen_perception_offline.py` 和 `viss/models/yolo11s-seg-best.pt`。
- `task319_local_grasp_rl/`: IsaacLab DirectRLEnv 局部抓取训练环境，当前用于 hover 后最后下降、闭合、抬升的 SAC/PPO 训练。
- `robot/`, `models/`: 机器人模型、配置和本地模型说明。
- `TASK319_DEVELOPMENT_REPORT.md`: 当前开发报告、问题复盘和后续路线。

大型外部依赖和生成产物不提交：`IsaacLab/`, `kuavo-ros-opensource/`, `task_319_garbage_sort/output/`, `logs/`, `wandb/`, 第三方研究仓库和本地训练日志均保持 local-only。

## Environment

默认工作目录：

```bash
cd /home/zhxm/workspace/mda_isaaclab
```

默认 Python 环境：

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python
```

如果使用交互 shell，也可以先激活环境：

```bash
conda activate my_task319_safe
```

本地需要具备：

- IsaacLab/Isaac Sim 可运行环境。
- ROS 2 + Nav2；任务脚本默认会使用 `task_319_garbage_sort/scripts/setup_nav2_user_install.bash` 和 bundled minimal Nav2 stack。
- CUDA GPU，用于 IsaacLab、cuRobo 和 RL 训练。
- VISS v28 权重：`viss/models/yolo11s-seg-best.pt`。
- DashScope/Qwen API key，用于 VLM 全局识别、分类和验证。

## API Configuration

Qwen/DashScope：

```bash
export DASHSCOPE_API_KEY="your_dashscope_key"
export QWEN_MODEL="qwen3-vl-flash"
export QWEN_API_STYLE="dashscope"
```

API 连通性检查：

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python viss/scripts/perception/yolo11_qwen_perception_offline.py --qwen-api-check --qwen-api-style dashscope
```

W&B 训练日志可选配置：

```bash
cp task319_local_grasp_rl/wandb.example.sh task319_local_grasp_rl/wandb.sh
source task319_local_grasp_rl/wandb.sh
```

`task319_local_grasp_rl/wandb.sh` 已被 `.gitignore` 忽略，请不要提交真实 API key。

## Main Commands

GUI 主线默认命令，不加 `--headless`：

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

这条命令使用当前默认配置：

- VISS v28/Qwen-first 视觉链路。
- `viss/models/yolo11s-seg-best.pt` 精确分割。
- 当前主场景头部 RGB-D 相机参数，不使用 VISS 原始偏置相机参数。
- 动态桌边站位规划，Nav2 导航到抓取点，并在站位处重新拍照。
- 四个硬编码垃圾桶位置和类别映射。
- 录制外部观察视频，视频保存后退出。

当前完整展示链路使用旧吸取别名/近距 carry 辅助时，可显式运行：

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

该命令用于展示“识别 -> 分类 -> 导航 -> 末端到桶口上方释放”的完整闭环。它不是物理夹爪验收命令。

仅验证动态站位和 Nav2，不抓取：

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py --standpoint_nav_only --record_video
```

小方块物理抓取校准：

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --debug_cube_grasp_demo \
  --debug_cube_isolated_scene \
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

输出目录：

```text
task_319_garbage_sort/output/head_camera_grasp_records/<timestamp>/
```

常见关键文件：

- `external_grasp_demo.mp4`: 外部观察视频。
- `video_manifest.json`: 视频保存状态。
- `mind_sort_demo/mind_sort_task_queue.json`: 全局循环状态、完成/失败物体、Nav2/抓取配置。
- `mind_sort_demo/cycle_XXXX/mind_sort_cycle.json`: 单个物体的选择、分类、抓取、导航、投放记录。
- `mind_sort_demo/cycle_XXXX/standpoint_reshoot/v28_original/`: 抓取站位重新拍照后的 VISS v28 结果。

## RL Training

局部抓取 RL 的目标不是替代视觉、Nav2 或全局状态机，而是接管 hover 点之后最后 10-15 cm 的局部闭环下降、闭合、抬升。

快速 smoke test：

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task319_local_grasp_rl/train.py --task319_quick_test
```

SAC 训练：

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task319_local_grasp_rl/train.py --algorithm SAC --headless --num_envs 256
```

PPO 训练：

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task319_local_grasp_rl/train.py --algorithm PPO --headless --num_envs 256 --max_iterations 1500
```

上传 W&B：

```bash
source task319_local_grasp_rl/wandb.sh
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task319_local_grasp_rl/train.py --algorithm SAC --headless --num_envs 256 --wandb
```

回放 checkpoint：

```bash
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python task319_local_grasp_rl/play.py --checkpoint /abs/path/to/checkpoint.pt --num_envs 16
```

训练日志和 checkpoint 默认在：

```text
logs/skrl/task319_hover_descent_grasp/
```

## Current Known Limits

- 真实物理夹爪主线仍不稳定，主要问题是主线物体抓取姿态、最后下降、夹爪闭合和物体接触耦合。
- GraspNet/AnyGrasp 已经过多轮诊断，当前不作为默认主线；RGB-D 几何和 VISS v28 负责选点与分类。
- Nav2 可驱动机器人完成大部分路线，但末端定位和 `STATUS_6` 偶发失败仍需要控制参数和重试策略继续优化。
- 当前完整演示可跑通多个物体分类投放，但依赖近距 carry/旧吸取别名保证展示稳定性。
- SAC 正在训练局部 hover-to-grasp 策略，后续计划接入到主线 physical grasp 阶段。

更多细节见 `TASK319_DEVELOPMENT_REPORT.md`、`task_319_garbage_sort/DEMO_COMMANDS.md`、`task_319_garbage_sort/NAV2_MOTION_IMPLEMENTATION.md` 和 `task319_local_grasp_rl/README.md`。
