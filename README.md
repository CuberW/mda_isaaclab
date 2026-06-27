# MDA IsaacLab Task319 Garbage Sorting

## 3.19 交付材料入口

本仓库已按 `mda_isaaclab` 项目整理 3.19 垃圾分类交付材料：

- 部署和运行说明：`docs/task319_delivery/README.md`（含环境版本、依赖安装、启动命令、模型下载位置、常见问题）
- 交付文件清单：`docs/task319_delivery/DELIVERY_CHECKLIST.md`
- 问题调试报告：`docs/task319_delivery/DEBUG_REPORT.md`
- 答辩 PPT 文稿：`docs/task319_delivery/PPT_OUTLINE.md`
- 答辩 PPT 文件：`docs/task319_delivery/TASK319_DEFENSE_PPT.pptx`
- 演示视频证据：`docs/task319_delivery/VIDEO_EVIDENCE.md`

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

进入仓库根目录：

```bash
cd mda_isaaclab
```

激活项目 Python 环境后再运行命令：

```bash
conda activate my_task319_safe
python --version
```

后续命令统一使用当前环境中的 `python`，不要写本机解释器绝对路径。

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
python viss/scripts/perception/yolo11_qwen_perception_offline.py --qwen-api-check --qwen-api-style dashscope
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
python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
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
python task_319_garbage_sort/task319_grasp_sort_sm.py \
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
python task_319_garbage_sort/task319_grasp_sort_sm.py --standpoint_nav_only --record_video
```

朴素桌面式小方块上方物理抓取测试，默认 GUI，不加 `--headless`：

```bash
python task_319_garbage_sort/topdown_cube_grasp_test.py \
  --record_video \
  --object_pos 0.70,-0.16,0.560 \
  --object_size 0.050,0.032,0.040 \
  --object_yaw_deg 25
```

也可以从主入口转发到同一套独立逻辑：

```bash
python task_319_garbage_sort/task319_grasp_sort_sm.py \
  --debug_cube_simple_topdown \
  --record_video \
  --object_pos 0.70,-0.16,0.560 \
  --object_size 0.050,0.032,0.040
```

旧主线小方块物理抓取校准：

```bash
python task_319_garbage_sort/task319_grasp_sort_sm.py \
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

分段物理展示视频，不使用吸附、近距 carry 辅助或录制过程物体瞬移：

```bash
# 固定最佳夹持位物理抬升，主线桌面场景保留 10 个物体
python task_319_garbage_sort/task319_grasp_sort_sm.py --physical_showcase_stage grasp_fixed --physical_showcase_object trash_battery_0 --physical_showcase_category 有害垃圾 --record_video --video_width 1280 --video_height 720 --video_sample_stride 4 --post_action_hold_steps 120

# 主线电池物体已被物理夹住作为初始条件，导航到有害垃圾桶并开爪投放
python task_319_garbage_sort/task319_grasp_sort_sm.py --physical_showcase_stage carry_drop --physical_showcase_object trash_battery_0 --physical_showcase_category 有害垃圾 --physical_showcase_start_waypoint table_front --mind_sort_realistic_bin_drop_demo --record_video --video_width 1280 --video_height 720 --video_sample_stride 4 --nav2_goal_timeout_s 180 --post_action_hold_steps 120

# 投放后从垃圾桶侧导航回 home
python task_319_garbage_sort/task319_grasp_sort_sm.py --physical_showcase_stage return_home --physical_showcase_object trash_battery_0 --physical_showcase_category 有害垃圾 --mind_sort_realistic_bin_drop_demo --record_video --video_width 1280 --video_height 720 --video_sample_stride 4 --nav2_goal_timeout_s 180 --post_action_hold_steps 120
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
- `physical_showcase/<stage>/physical_showcase_metadata.json`: 分段物理展示的物理约束、导航和投放记录。

## Entry Point Map

本节用于说明“当前到底该从哪里启动”。真正的 Task319 IsaacLab 主入口是
`task_319_garbage_sort/task319_grasp_sort_sm.py`；多数功能模式由它转到
`task_319_garbage_sort/visual_grasp_record_demo.py` 后根据参数选择。旧
`robot/` 包里的入口仍保留，但不是当前 IsaacLab 主线依据。

### 主任务入口

- `task_319_garbage_sort/task319_grasp_sort_sm.py`
  - 当前公开主入口。
  - 默认转发到 `visual_grasp_record_demo.py`。
  - 如果传入 `--debug_cube_simple_topdown`，会转发到 `topdown_cube_grasp_test.py`。

- `task_319_garbage_sort/visual_grasp_record_demo.py`
  - 当前最大、最完整的 IsaacLab 执行主体。
  - 包含正式场景、VISS/v28 视觉、Qwen 分类、Nav2、机械臂抓取、投放、视频录制和大量调试模式。
  - 一般优先通过 `task319_grasp_sort_sm.py` 调用，不建议随意绕过 wrapper。

- `--mind_sort_demo`
  - 当前完整垃圾分类主线模式。
  - 目标链路是：头部 RGB-D -> VISS v28/Qwen-first -> 目标选择/分类 -> 动态桌边站位 -> Nav2 到桌边 -> 抓取 -> 导航到垃圾桶 -> 投放/返回。
  - 这是当前主线判断入口。若只想跑当前完整任务，优先使用：

```bash
python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

- `--execute_grasp`
  - 在视觉链路中实际执行机械臂抓取。
  - 不加它时，很多命令只做识别、目标选择、规划和记录。

- `--enable_sort_nav`
  - 抓取后启用“去对应垃圾桶并投放”的导航流程。

- `--standpoint_nav_only`
  - 只跑视觉、目标选择、动态桌边站位计算、Nav2 到站位和重新拍照。
  - 不执行抓取。用于验证“目标点 -> 机器人站位 -> Nav2”链路。

### 主线调试和展示模式

- `--debug_cube_grasp_demo`
  - 旧的小方块抓取调试模式。
  - 仍走 `visual_grasp_record_demo.py` 的大主线逻辑，只是目标换成 debug cube。

- `--debug_cube_isolated_scene`
  - 配合 `--debug_cube_grasp_demo` 使用。
  - 只保留机器人、简化桌子、小方块和相机，去掉正式 10 个垃圾物体和桶等干扰。

- `--debug_cube_rgbd_target`
  - debug cube 仍存在，但抓取目标来自头部 RGB-D 反投影结果，而不是直接使用仿真物体 root pose。
  - 用于检查视觉目标和仿真真值之间的偏差。

- `--debug_cube_static`
  - 让 debug cube 静止/运动学。
  - 适合纯 IK/TCP 对位，不适合验证真实物理夹取。

- `--debug_cube_simple_topdown`
  - 新增实验入口。
  - 通过 `task319_grasp_sort_sm.py` 运行时会转发到 `topdown_cube_grasp_test.py`。
  - 当前这条朴素桌面式上方抓取实验已验证不稳定/失败，暂时不作为主线依据。

- `--physical_showcase_stage grasp_fixed|carry_drop|return_home`
  - 分段录视频展示入口。
  - `grasp_fixed`：固定场景下展示夹持/抬升段。
  - `carry_drop`：已持物状态下导航到桶并投放。
  - `return_home`：投放后返回桌边/home。
  - 这是为了录制分段 Demo，不等价于完整自主抓取闭环。

### 导航入口

- `--motion_only_sort_demo`
  - 关闭视觉和抓取，只测试基础导航/投放状态机。
  - 用于验证 Nav2 和路由，不验证机械臂。

- `--waypoint_nav_demo`
  - 按命名 waypoint 路线导航，例如 `home,table_left,bin_center,home`。
  - 用于验证 Nav2、轮式底盘和 waypoint 定义。

- `--dynamic_standpoint_nav_demo`
  - 手动给一个世界坐标目标点，系统计算合适桌边站位，再交给 Nav2。
  - 用于单独验证“目标点到抓取站位”的逻辑。

- `--ros_cmd_vel_demo`
  - 不跑 Nav2 目标，只验证 ROS2 `/cmd_vel` 能否驱动 Isaac 里的机器人底盘移动。

- `task_319_garbage_sort/phase1_scene_ros2_bridge.py`
  - 早期场景 + ROS2 bridge 入口。
  - 搭建机器人、桌子、垃圾物体、桶和基础 ROS2 topic。
  - 可暴露 `/cmd_vel`、`/odom`、`/scan`、`/tf`。
  - 不做视觉抓取和分拣闭环。

### 抓取和机械臂单项测试入口

- `task_319_garbage_sort/gripper_physics_test.py`
  - 只测夹爪物理。
  - 把小红块放进夹爪之间，闭合后观察能否保持。
  - 用于排查夹爪摩擦、刚度、闭合宽度和 PhysX 接触参数。

- `task_319_garbage_sort/arm_chain_isolated_test.py`
  - 只测 Kuavo 右臂控制链。
  - 固定底盘，右臂用 IsaacLab DLS IK 跟踪一个 6D 目标。
  - 无视觉、无导航、无物体抓取。

- `task_319_garbage_sort/topdown_cube_grasp_test.py`
  - 朴素桌面式上方抓取实验。
  - 小长方体、PCA 短轴、cuRobo 右臂、可选 Kuavo IK 审计/seed、闭爪和物理抬升验证。
  - 当前实验失败，暂时标记为非主线。

- `task_319_garbage_sort/scripts/curobo_right_arm_smoke.py`
  - cuRobo 右臂规划 smoke test。
  - 主要验证 cuRobo 配置、URDF、TCP link、右臂关节顺序和规划器能否工作。

- `task_319_garbage_sort/official_holonomic_wheel_test.py`
  - 单独测试 Isaac 官方 holonomic wheel controller。
  - 不涉及 Task319 视觉、抓取或 Nav2 主线。

### ROS、Nav2 和 Kuavo IK 辅助进程

- `task_319_garbage_sort/scripts/ros2_nav2_stack_launcher.py`
  - 启动本地 minimal Nav2 stack。
  - 主线通常自动调用。

- `task_319_garbage_sort/scripts/ros2_nav_socket_bridge.py`
  - Isaac conda 环境不能直接 import ROS2 时，用它发布 odom/scan/tf，并接收 Nav2 `/cmd_vel`。

- `task_319_garbage_sort/scripts/ros2_nav_goal_client.py`
  - 给 Nav2 发送单个 `NavigateToPose` goal。

- `task_319_garbage_sort/scripts/ros2_nav2_smoke_test.py`
  - Nav2 端到端 smoke test。
  - 不依赖完整 Task319 Isaac 场景。

- `task_319_garbage_sort/scripts/ros2_cmd_vel_test_publisher.py`
  - 发布有限时长 `/cmd_vel`。
  - 配合 `--ros_cmd_vel_demo` 验证底盘运动。

- `task_319_garbage_sort/scripts/kuavo_ik_socket_bridge.py`
  - ROS1 官方 Kuavo IK socket bridge。
  - Isaac 进程通过 JSON socket 调官方 `/ik/two_arm_hand_pose_cmd_srv` 和 FK 服务。

- `task_319_garbage_sort/scripts/kuavo_ros2_ik_socket_bridge.py`
  - ROS2 版本 IK bridge。
  - 主要用于 ros1_bridge 场景诊断。

- `task_319_garbage_sort/scripts/start_kuavo_official_ik_sidecar.sh`
  - 启动官方 Kuavo IK sidecar 的 shell 入口。

### 感知、依赖和离线工具

- `task_319_garbage_sort/visual_grasp_pipeline.py`
  - 离线视觉抓取 pipeline 入口。
  - 偏模型准备、目标选择、分类、候选过滤。
  - 不是完整 Isaac 执行主线。

- `task_319_garbage_sort/scripts/prepare_grasp_deps.py`
  - 检查/准备 GraspNet、AnyGrasp、权重和 CUDA 扩展等可选依赖。

### RL 局部抓取入口

- `task319_local_grasp_rl/train.py`
  - 训练 hover-to-grasp 局部策略。
  - 目标是接管 hover 点之后最后 10-15 cm 的局部闭环下降、闭爪和抬升。
  - 当前没有默认接入 `--mind_sort_demo`。

- `task319_local_grasp_rl/play.py`
  - 回放训练好的 RL checkpoint。

### 旧 robot 包入口

- `robot/run_all.py`
  - 较早的统一任务 runner，更偏 MuJoCo/通用任务框架。
  - 当前 IsaacLab Task319 主线不以它为准。

- `robot/task_319_garbage_sort/__main__.py`
  - 旧 `python -m task_319_garbage_sort` 风格入口。
  - 保留用于旧架构，不是当前 IsaacLab 主线入口。

### 当前推荐判断

- 跑当前完整主线：`task319_grasp_sort_sm.py --mind_sort_demo`
- 拆抓取站位问题：`--standpoint_nav_only`
- 拆导航问题：`--waypoint_nav_demo` 或 `--dynamic_standpoint_nav_demo`
- 拆底盘 `/cmd_vel` 问题：`--ros_cmd_vel_demo`
- 拆夹爪物理问题：`gripper_physics_test.py`
- 拆右臂控制链问题：`arm_chain_isolated_test.py`
- 拆 cuRobo 问题：`scripts/curobo_right_arm_smoke.py`
- `topdown_cube_grasp_test.py` 当前是失败实验入口，先不纳入主线判断。

### 具体使用指导

按下面顺序排查，不要一上来就跑完整主线。每一步只验证一个子系统，
通过后再进入下一步；失败时先看该步骤输出目录里的 JSON/视频，不要混入其他入口。

1. 验证 Isaac 场景是否正常

```bash
python task_319_garbage_sort/phase1_scene_ros2_bridge.py --disable_lidar
```

应该看到 Kuavo、桌子、四个垃圾桶和 10 个桌面物体。这个入口不抓取、不导航，只确认资产、场景和相机基础加载正常。

2. 验证夹爪物理本身

```bash
python task_319_garbage_sort/gripper_physics_test.py --smoke_steps 0 --close_start_step 600
```

如果这里都夹不住红色小块，优先调夹爪摩擦、关节刚度、闭合宽度和接触参数；不要进入主线抓取调试。

3. 验证右臂控制链

```bash
python task_319_garbage_sort/arm_chain_isolated_test.py --smoke_steps 0 --trajectory_steps 720
```

通过标准是底盘不漂、右臂平滑跟踪目标 marker。失败说明右臂关节、TCP、IK 或仿真控制链有问题，与视觉和 Nav2 无关。

4. 验证 cuRobo 右臂规划

```bash
python task_319_garbage_sort/scripts/curobo_right_arm_smoke.py
```

用于确认 cuRobo 配置、右臂关节顺序、URDF、TCP extra link 和 GPU 规划器可用。cuRobo smoke test 不通过时，不要用 `curobo_right_arm` 后端调主线抓取。

5. 验证底盘 `/cmd_vel`

```bash
python task_319_garbage_sort/task319_grasp_sort_sm.py \
  --ros_cmd_vel_demo \
  --record_debug \
  --record_video \
  --ros_cmd_vel_demo_steps 240 \
  --ros_cmd_vel_demo_linear_x 0.20 \
  --post_action_hold_steps 240
```

如果机器人不动，先查 ROS2 bridge、`/cmd_vel` 发布、轮子驱动模型和 Isaac wheel/root coupling。此时不要调 Nav2 目标或抓取。

6. 验证 Nav2 waypoint 导航

```bash
python task_319_garbage_sort/task319_grasp_sort_sm.py \
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
  --post_action_hold_steps 240
```

这个入口不做视觉和抓取，只看 Nav2 能否把机器人带到命名点。失败时看 Nav2 goal 状态、`nav2_*` JSON 和外部视频。

7. 验证目标点到动态桌边站位

```bash
python task_319_garbage_sort/task319_grasp_sort_sm.py \
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
  --post_action_hold_steps 240
```

通过后，才说明“给定物体位置 -> 算站位 -> Nav2 到站位”基本可用。

8. 验证真实视觉链路到站位，不抓取

```bash
python task_319_garbage_sort/task319_grasp_sort_sm.py \
  --standpoint_nav_only \
  --record_debug \
  --record_video \
  --warmup_steps 80 \
  --nav2_stack_startup_s 2 \
  --nav2_goal_timeout_s 150 \
  --post_action_hold_steps 120
```

这个入口会跑正式 VISS v28/Qwen 视觉，选择目标，计算动态抓取站位，导航过去并重新拍照，但不会抓取。失败时先看 `standpoint_reshoot/` 和 `pre_grasp_standpoint_nav.json`。

9. 验证旧 debug cube 抓取校准

```bash
python task_319_garbage_sort/task319_grasp_sort_sm.py \
  --debug_cube_grasp_demo \
  --debug_cube_isolated_scene \
  --debug_cube_pos 0.70,-0.16,0.5625 \
  --arm_motion_backend curobo_right_arm \
  --num_cycles 1 \
  --execute_grasp \
  --record_debug \
  --record_video \
  --warmup_steps 60 \
  --cycle_interval_steps 1 \
  --trajectory_steps 180 \
  --safe_pregrasp_steps 80 \
  --grasp_steps 100 \
  --lift_steps 120 \
  --hold_steps 40 \
  --max_joint_step 0.02 \
  --post_action_hold_steps 20
```

这个入口仍属于旧主线 debug cube 校准，适合看 TCP 对准、arm trajectory 和实际 cube motion。它不是正式垃圾物体抓取验收。

10. 尝试官方 Kuavo IK 后端

```bash
python task_319_garbage_sort/task319_grasp_sort_sm.py \
  --mind_sort_demo \
  --arm_motion_backend kuavo_ik \
  --kuavo_ik_auto_start \
  --kuavo_ik_auto_start_mode docker \
  --kuavo_ik_docker_container kuavo_official_ros \
  --record_debug \
  --record_video
```

用于验证官方 Kuavo IK sidecar 能否参与主线。失败时优先看 IK bridge 日志、`kuavo_ik_*` metadata、FK mapping audit 和 q_torso/q_arm 长度，不要先改抓取状态机。

11. 跑当前完整主线

```bash
python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

这是当前完整任务入口。只有前面的场景、夹爪、右臂、cuRobo/Nav2、视觉站位都基本通过后，才建议用它判断主线状态。主要结果看：

- `external_grasp_demo.mp4`
- `mind_sort_demo/mind_sort_task_queue.json`
- `mind_sort_demo/cycle_XXXX/mind_sort_cycle.json`
- `cycle_XXXX/arm_trajectory_debug.json`
- `cycle_XXXX/gripper_alignment_debug.json`
- `cycle_XXXX/standpoint_reshoot/v28_original/`

12. 暂时不要用作判断的入口

- `topdown_cube_grasp_test.py`：当前朴素上方抓取实验失败，先作为失败实验保留。
- `robot/run_all.py` 和 `robot/task_319_garbage_sort/__main__.py`：旧架构入口，不代表当前 IsaacLab Task319 主线。
- `visual_grasp_pipeline.py`：离线视觉/候选工具，不执行完整 Isaac 抓取闭环。

## 遥操作采集抓取数据指导

这里的目标不是让人遥操作完整“识别、导航、抓取、投放”流程，而是专门采集 `hover-to-grasp` 局部抓取数据，用来训练一个更合适的右臂末端闭环抓取策略。全局视觉、Nav2、cuRobo 到 hover 点、Kuavo IK/右臂控制仍然保留；学习策略只接管 hover 点之后最后 10-15 cm 的对准、下探、闭爪和抬升。

当前仓库还没有 Task319 专用遥操作采集脚本。IsaacLab 本地源码已经提供了可复用基础：`IsaacLab/source/isaaclab/isaaclab/devices/keyboard/se3_keyboard.py`、`Se3SpaceMouse` 和通用 HDF5 录制工具 `IsaacLab/scripts/tools/record_demos.py`。但通用 `record_demos.py` 需要环境接入 recorder，Task319 局部抓取环境还没完成这层接入，所以第一步应新增一个专用 collector，例如 `task319_local_grasp_rl/collect_teleop_demos.py`。

### 采集范围

只采局部抓取阶段：

1. 场景生成一个桌面物体，初期建议用小方块或尺寸明确的小长方体。
2. 主线或采集脚本用 cuRobo/Kuavo IK 把右夹爪移动到物体上方 hover pose。
3. 操作者只控制右夹爪 TCP 的小范围平移和夹爪开合。
4. 成功动作应包含：XY 对准、从上往下下探、闭爪、短暂停留、物理抬升。
5. 不采集导航、不采集全身走位、不采集投垃圾桶动作；这些阶段先由现有主线规则控制。

这样做的原因是：完整分拣流程状态太长，失败来源混杂；当前真正需要学习的是 hover 后最后一小段物理抓取。

### 遥操作设备和按键

优先使用 SpaceMouse，因为连续控制更平滑；没有 SpaceMouse 时先用键盘也可以。键盘不需要额外网卡，只需要 Isaac Sim 图形窗口获得输入焦点。

IsaacLab `Se3Keyboard` 默认按键：

- `W/S`：TCP 沿 X 方向移动。
- `A/D`：TCP 沿 Y 方向移动。
- `Q/E`：TCP 沿 Z 方向移动。
- `K`：夹爪开合切换。
- `Z/X`、`T/G`、`C/V`：姿态旋转。第一版采集建议先忽略旋转，固定竖直向下抓取，降低策略维度。

Task319 collector 还应额外注册几个采集控制键：

- `N`：标记当前 episode 成功并保存，然后 reset。
- `M`：标记当前 episode 失败并保存到 failed 分组，然后 reset。
- `R`：不保存，直接 reset。
- `ESC`：结束采集并关闭文件。

### 单条 Episode 流程

每条数据从“右夹爪已经在物体上方 hover”开始：

1. reset 场景，随机物体 XY、yaw、尺寸、质量和摩擦；初期随机范围要小。
2. 用 cuRobo 或 Kuavo IK 把右臂恢复到稳定 hover 姿态，夹爪打开，TCP 高于物体顶面约 8-12 cm。
3. 操作者用 `W/S/A/D` 做 XY 对准，目标是让 TCP 投影落在可夹取中心区域。
4. 操作者用 `Q/E` 缓慢下探，让物体尽可能进入夹爪包络，但不能主动穿模。
5. 按 `K` 闭爪，等待 0.3-0.8 s，让接触稳定。
6. 用 `Q/E` 或自动 lift primitive 抬升 5-10 cm。
7. 如果物体被真实物理夹住并离桌，按 `N` 保存成功；如果推飞、滑落、没夹住，按 `M` 保存失败。

成功判定建议使用仿真真值做标签，但不要作为策略输入：物体被抬高至少 2.5 cm，并且保持在 TCP 附近一小段时间，例如 0.5 s 内物体中心距离 TCP 不超过 12 cm。

### 每步应记录的数据

策略输入必须尽量贴近真实部署可获得的信息。不要把物体 root pose、物体速度这类仿真真值喂给策略；它们只能用于成功标签、调试和质量筛选。

推荐 HDF5 保存到：

`task_319_garbage_sort/output/teleop_grasp_datasets/YYYYMMDD_HHMMSS/task319_hover_grasp_teleop.hdf5`

推荐结构：

```text
data/demo_000000/actions              (T, 4)   dx, dy, dz, gripper_close
data/demo_000000/obs/policy           (T, 28)  与 task319_local_grasp_rl 当前策略一致
data/demo_000000/obs/tcp_pose_w       (T, 7)   TCP 世界位姿
data/demo_000000/obs/target_rel_tcp   (T, 3)   视觉估计抓取点相对 TCP
data/demo_000000/obs/right_q          (T, 7)   右臂关节位置
data/demo_000000/obs/right_qd         (T, 7)   右臂关节速度
data/demo_000000/obs/gripper_width    (T, 1)   夹爪开度
data/demo_000000/debug/object_pose_w  (T, 7)   仅调试/标签，不进策略
data/demo_000000/debug/lift_m         (T, 1)   仅调试/标签
data/demo_000000/success              scalar   bool
data/demo_000000/metadata_json        string   物体尺寸、随机种子、设备、git commit、sim dt
```

如果以后要训练视觉策略，再额外保存 head camera 或 wrist camera 的 RGB-D；第一版局部策略不建议直接上图像，先用现有 28 维 proprioception + 视觉估计目标点训练。

### 数据量建议

先小规模验证，再扩大采集：

- 30-50 条成功 demo：只用于验证 collector、HDF5、BC 训练和回放链路。
- 200-300 条成功 demo：可以训练第一版行为克隆策略，物体随机范围要覆盖桌面常见位置。
- 800-1500 条成功 demo：用于训练更稳的策略，并加入不同尺寸、yaw、质量、摩擦、轻微视觉误差。
- 失败 demo 单独保存，比例保留 20-30%；第一版 BC 只用成功 demo，失败 demo 用于分析、筛选和后续 DAgger/判别器。

训练、验证、测试按 episode 划分，例如 80/10/10，不能按 frame 随机切分，否则同一条操作轨迹会同时出现在训练和验证里。

### 训练路线

第一阶段先做行为克隆，不要直接用长流程 RL：

1. 新增 `task319_local_grasp_rl/train_bc.py`，读取 HDF5。
2. 输入使用 `obs/policy` 的 28 维观测，输出 4 维动作 `dx, dy, dz, gripper_close`。
3. 用 MLP 训练，损失先用 L1 或 MSE；动作和观测都要做归一化。
4. 新增 `task319_local_grasp_rl/play_bc.py`，在局部 env 中闭环回放策略，统计真实物理抬升成功率。
5. BC 成功率稳定后，再把 BC policy 作为 teacher 或初始化，接到现有 `task319_local_grasp_rl/train.py` 的 SAC/PPO 训练里微调。

第二阶段再接主线：

1. 主线仍由 VISS/Qwen 选择物体。
2. Nav2 到抓取站位。
3. cuRobo/Kuavo IK 把右夹爪送到 hover。
4. 切到学习到的局部策略，策略输出小 TCP delta 和夹爪闭合命令。
5. 成功物理抬升后，再交还给主线做运输和投放。

### 质量控制

采集时要主动丢弃这些数据：

- 初始 hover 点明显不在物体上方。
- 物体直径或短边超过夹爪可夹范围。
- 操作者下探时把物体推走。
- 闭爪前已经碰撞导致物体大幅移动。
- 使用了吸附、绑定、传送、悬浮修正等非物理辅助。
- 成功标签来自脚本误判，而视频里物体实际滑落。

核心原则是：训练数据要反映“真实右臂夹爪从上往下夹住并抬起”的闭环动作，而不是演示系统用隐藏规则把物体带走。

## RL Training

局部抓取 RL 的目标不是替代视觉、Nav2 或全局状态机，而是接管 hover 点之后最后 10-15 cm 的局部闭环下降、闭合、抬升。

快速 smoke test：

```bash
python task319_local_grasp_rl/train.py --task319_quick_test
```

SAC 训练：

```bash
python task319_local_grasp_rl/train.py --algorithm SAC --headless --num_envs 256
```

PPO 训练：

```bash
python task319_local_grasp_rl/train.py --algorithm PPO --headless --num_envs 256 --max_iterations 1500
```

上传 W&B：

```bash
source task319_local_grasp_rl/wandb.sh
python task319_local_grasp_rl/train.py --algorithm SAC --headless --num_envs 256 --wandb
```

回放 checkpoint：

```bash
python task319_local_grasp_rl/play.py --checkpoint path/to/checkpoint.pt --num_envs 16
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
