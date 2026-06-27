# Task319 垃圾分类抓取机器人开发报告

更新时间：2026-06-25

## 1. 当前主线概览

当前 3.19 任务主线目标是：机器人从观察位拍摄桌面 RGB-D，识别所有垃圾，选择一个可执行目标，导航到适合右臂抓取的桌边站位，重新拍照定位，执行抓取，根据 VLM 分类结果导航到对应垃圾桶，末端移动到桶口上方释放物体，然后回到观察位继续下一轮。

当前默认入口：

```bash
cd mda_isaaclab
python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

主线组成：

| 模块 | 当前正式选择 | 说明 |
| --- | --- | --- |
| 视觉识别 | VISS v28 Qwen-first | Qwen 全局识别候选，YOLO11s-seg-best 精分割，Qwen/VLM 给四分类标签 |
| 相机数据 | 当前主场景 `head_rgbd` | 使用 Task319 场景自身 intrinsics、depth、`camera_to_world`，不使用 VISS 原始偏置相机参数 |
| 3D 定位 | RGB-D mask/ROI 内几何中心和 top grasp 点 | 抓取点来自当前站位重新拍摄的 RGB-D，不使用上帝视角作为主线定位 |
| 目标选择 | 多候选排序 + 可达/站位约束 | 初始观察位尽量拍全桌，按对象状态、距离、候选站位和失败记忆排序 |
| 导航 | ROS2 Nav2 `NavigateToPose` | Nav2 规划和控制，Isaac bridge 接 `/cmd_vel`，再转轮式底盘运动 |
| 抓取 | 物理夹爪仍在调试 | 小方块和局部基线可验证；主线真实垃圾物体仍不稳定 |
| 演示闭环 | 夹爪近距 carry/旧吸取别名 | 用于跑通完整视觉、导航、分类、投放链路，不作为物理夹爪验收 |
| RL 后续 | skrl SAC/PPO hover-to-grasp | 正在训练接管最后下降、闭合和抬升阶段 |

最新完整演示验证 run：

```text
task_319_garbage_sort/output/head_camera_grasp_records/20260625_150100/
```

视频：

```text
task_319_garbage_sort/output/head_camera_grasp_records/20260625_150100/external_grasp_demo.mp4
```

该 run 完成了 8 个物体的视觉识别、分类、导航到桶侧和桶口上方释放；最终在返回观察位时 Nav2 返回 `STATUS_6`，因此全局 `success=false`，但 `failed_scene_keys=[]`，已完成物体没有投放失败。该 run 使用了旧吸取别名/夹爪近距 carry 辅助，必须作为展示闭环而非真实夹爪最终成功来解读。

| Cycle | Scene Key | VLM 名称 | 分类 | 目标桶 | Cycle 成功 | 桶口释放 | 物体瞬移 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0000 | `trash_05` | 金属罐 | 可回收物 | recycle | true | true | false |
| 0001 | `trash_00` | 纸盒 | 可回收物 | recycle | true | true | false |
| 0002 | `trash_02` | 金属罐 | 可回收物 | recycle | true | true | false |
| 0003 | `trash_08` | 红色砖块 | 其他垃圾 | other | true | true | false |
| 0004 | `trash_06` | 清洁剂 | 有害垃圾 | hazard | true | true | false |
| 0005 | `trash_07` | 笔 | 有害垃圾 | hazard | true | true | false |
| 0006 | `trash_01` | 纸盒 | 可回收物 | recycle | true | true | false |
| 0007 | `trash_04` | banana | 厨余垃圾 | kitchen | true | true | false |

## 2. 视觉与分类模块

### 当前方案

当前正式视觉链路来自 VISS v28，接入方式不是直接运行 VISS 原始仿真服务器，而是在 Task319 主场景中用本场景头部 RGB-D 采图，然后调用：

```text
viss/scripts/perception/yolo11_qwen_perception_offline.py
```

当前默认权重：

```text
viss/models/yolo11s-seg-best.pt
```

当前思路是 Qwen-first：

1. Qwen/VLM 先看全局 RGB 图，返回高可信度垃圾候选、语义名称和四分类类别。
2. YOLO11s-seg-best 对候选 ROI 做精确分割。
3. Qwen 对精分割 crop 做验证和分类确认。
4. Task319 将 2D mask/ROI 绑定到当前 RGB-D 深度图，计算该物体在世界坐标下的几何信息。
5. 分类标签以 VLM 输出为准，不再使用硬编码类别作为正式分类结果。

四分类目标为：

- 可回收物
- 有害垃圾
- 厨余垃圾
- 其他垃圾

Qwen/DashScope 配置：

```bash
export DASHSCOPE_API_KEY="your_dashscope_key"
export QWEN_MODEL="qwen3-vl-flash"
export QWEN_API_STYLE="dashscope"
```

### 遇到的问题

早期尝试过旧 YOLO、GLM、GraspNet mask 联动和融合后的 v27/v28 适配版本，主要问题包括：

- 视觉链路各模块选择的不是同一个物体，导致“YOLO 框的是 A，抓取选的是 B”。
- 旧 mask 边缘质量不稳定，直接用 mask 点云质心会漂移。
- 纯 RGB 分割容易被颜色欺骗，例如白色物体在浅色桌面上边界不可靠。
- VLM 如果只看抠得过干净的物体 crop，容易因为缺少上下文而误判。
- GLM/旧链路和当前场景相机参数不统一，导致分类、定位和抓取前后不一致。
- 早期有候选数量限制，后续多物体场景中容易漏检。

### 当前解决方案

- 主线统一成 VISS v28/Qwen-first，彻底弃用 GLM 和旧 YOLO 作为默认链路。
- 使用当前 Task319 场景的头部 RGB-D 和相机外参，不使用 VISS 原始偏置相机参数。
- 每个 cycle 先在观察位选目标，再导航到站位重新拍照，抓取定位使用站位处的最新 RGB-D。
- 分类标签绑定到当前选中的实例，不跨物体复用。
- 默认不限制识别个数，避免多物体桌面漏检。
- 保存 `v28_original/` 结果、overlay、mask 和 cycle JSON，便于追溯“标红目标”和分类标签来源。

## 3. RGB-D 定位与抓取点计算

### 当前方案

抓取点不再优先依赖 GraspNet 的 6D 姿态，而是使用当前相机拍摄得到的 RGB-D 点云几何：

- 通过 VISS v28 mask/ROI 取对应 RGB-D 像素。
- 使用当前 intrinsics 和 `camera_to_world` 将深度点投到世界坐标。
- 计算物体 3D 几何中心、点云范围和最高点 `Z_max`。
- top-grasp 路径使用 `(X_center, Y_center, Z_max - depth)` 或类似浅层顶部目标，而不是物体几何中心 Z。
- hover 点位于物体上方，最后阶段应尽量垂直下降，避免斜向扫到物体。

### 遇到的问题

- 早期直接用几何中心作为目标，夹爪会撞到物体顶部或桌面。
- 第二段下降如果重新做关节空间规划，会产生横向弧线，把物体碰走。
- 如果抓取前使用旧观察位坐标，而不是导航到站位后的相机坐标，会造成明显错位。
- 夹爪姿态约束过多时，IK 会选择病态解，出现手臂穿身、后扭或嵌入躯干。
- 夹爪最大开合和物体尺寸不匹配时，即使位置正确也夹不起来。

### 当前处理

- 要求抓取前必须在导航站位重新拍照，并使用当前机器人位姿和当前相机外参计算世界坐标。
- 记录 `rgbd_vs_truth_debug.json`、`gripper_alignment_debug.json`、`arm_trajectory_debug.json` 等诊断文件。
- 抓取高度转向 top-grasp，减少从几何中心下插导致的碰撞。
- 默认主线避免严格轴向对齐，优先保证位置接近和路径不撞物体。
- 物理夹爪不稳定时，演示链路使用近距 gate：末端进入夹爪附近范围后，将物体绑定到右夹爪 TCP frame carry 到垃圾桶。

## 4. 抓取与机械臂控制模块

### 已尝试方案

1. GraspNet/AnyGrasp：
   - 添加过点云 sanity check、PLY 保存、mask distance relaxation、objectness top-k、关闭 IK 预筛选等诊断。
   - 实际问题是 GraspNet 原始 objectness 经常为零，或候选进入执行后仍在 runtime IK/safe-start 阶段失败。
   - 当前不再作为默认主线，仅保留诊断/研究价值。

2. RGB-D 几何中心/顶部抓取：
   - 对小方块和简单物体更可控。
   - 结合缓慢下降、夹爪全开、慢闭合和抬升能形成稳定基线。
   - 主线复杂垃圾物体仍受姿态、尺寸、接触、导航残差影响。

3. cuRobo：
   - 用于右臂 hover pose 规划和部分 Cartesian descent 尝试。
   - 遇到的问题是冗余姿态和位置-only IK 分支选择可能让手臂出现视觉上不合理的扭转。
   - 后续策略是减少不必要姿态约束，更多用于避障和到 hover 的全局规划，最后下降交给局部闭环/RL。

4. Kuavo IK / IsaacLab IK：
   - 用作姿态 seed、对比和调试。
   - 单独解决不了主线“目标定位、末端姿态、接触闭合”耦合问题。

5. 小方块校准：
   - 用可移动小方块在可达位置验证过“给定明确世界坐标时可以接近并夹取”的能力。
   - 这说明问题不只是夹爪物理，而是主线中感知坐标、导航残差、目标尺寸/姿态和 IK 解共同造成。

### 当前结论

真实物理夹爪仍是当前最大未闭环问题。为了推进系统级展示，主线保留了近距 carry/旧吸取别名的演示路径，但报告和 README 必须明确区分：

- 真实物理夹爪验收：需要物体被手指闭合夹住、抬升、随末端运动。
- 当前完整演示闭环：使用真实 VISS v28 识别、真实 Nav2 导航、真实桶口末端释放，但抓取阶段可用近距 carry 辅助保证视频完整。

后续应将 SAC 训练出的局部 hover-to-grasp policy 接入真实物理夹爪路径，减少手写开环下降和接触参数地狱。

## 5. 站位规划与导航模块

### 当前方案

导航使用 ROS2 Nav2，而不是手写轨迹跟随。控制链路为：

```text
Task319 状态机目标位姿
-> ROS2 Nav2 NavigateToPose
-> Nav2 planner/controller
-> /cmd_vel
-> ros2_nav_socket_bridge.py
-> Isaac 中的轮速/底盘适配
-> kinematic_stable 轮地耦合
```

当前较稳定的轮式配置是：

```text
--wheel_drive_model urdf_diagonal
--wheel_ground_coupling kinematic_stable
--wheel_velocity_scale 0.35
```

历史重要 baseline 是：

```text
--wheel_drive_model mecanum45
--wheel_ground_coupling kinematic
```

该 baseline 能动但 yaw 晃动明显，保留作为成功移动标记。

### 动态站位

当前默认使用动态桌边站位规划：

1. 机器人先在后退观察位拍全桌。
2. 对所有识别物体计算世界坐标。
3. 根据右手可达窗口、桌边安全距离和候选边选择适合抓取的地面位姿。
4. 目标 yaw 朝向桌面/物体，避免到点后背对目标。
5. Nav2 导航到该站位。
6. 到站位后重新拍照，再定位并抓取。

这种设计比固定左右几个点更接近真实系统，但仍保留预设候选点思想作为约束来源。

### 遇到的问题与处理

- 纯轮地物理接触曾导致轮子几乎不驱动，因此采用 kinematic wheel-ground coupling 保证 Nav2 命令可验证。
- 高摩擦版本曾让轮子彻底不动，已回退。
- yaw 左右晃通过 `urdf_diagonal + kinematic_stable` 明显降低。
- Nav2 偶发 `STATUS_6`，例如最新完整演示在返回观察位时失败。当前需要增加更稳健的重试、软成功和定位容差处理。
- 导航到站位后的 residual error 会影响抓取，因此抓取前重新拍照是必要步骤。

## 6. 分类投放与垃圾桶模块

当前有四个硬编码垃圾桶位置，分类到桶的映射由 VLM 四分类决定：

| 分类 | Bin key | 说明 |
| --- | --- | --- |
| 可回收物 | `recycle` | 纸盒、金属罐等 |
| 厨余垃圾 | `kitchen` | 香蕉等 |
| 有害垃圾 | `hazard` | 清洁剂、电池、笔/马克笔等按当前 prompt 规则可归入有害 |
| 其他垃圾 | `other` | 无法归入以上三类的普通废弃物 |

重要修正：

- 早期投放阶段存在“垃圾自己瞬移到桶里”的问题，不符合物理展示。
- 当前演示链路已经改成：机器人先导航到目标桶侧，右夹爪 TCP 再移动到对应桶口上方，然后打开夹爪/释放 carry。
- metadata 中记录 `object_teleport_used: false`、`drop_pose_world`、`target_bin_name` 和 `final_pose_inside_bin_opening_xy`。

这解决了投放阶段的展示真实性，但抓取阶段是否完全物理夹住仍需后续 RL/闭环控制解决。

## 7. RL/SAC 局部抓取训练

当前新增了独立目录：

```text
task319_local_grasp_rl/
```

Gym id：

```text
Task319-Hover-Descent-Grasp-Kuavo-Direct-v0
```

训练目标非常明确：不训练完整“看图到扔垃圾”，只训练最后 hover 后 10-15 cm 的局部抓取小脑。

输入：

- RGB-D 风格估计目标相对当前 TCP 的位置。
- TCP 速度。
- 右臂关节位置和速度。
- 当前夹爪开合状态。
- 高度信息。
- 上一步动作。

输出：

- `dx, dy, dz`: 末端微小位移。
- `gripper_close`: 连续夹爪闭合命令。

奖励设计：

- XY 对齐目标点。
- 下降到 `Z_max - grasp_depth`。
- 只在接近目标时闭合。
- 成功物理抬起物体给大奖励。
- 提前闭合、推飞物体、碰桌、动作过大给惩罚。

当前 SAC 训练命令：

```bash
python task319_local_grasp_rl/train.py --algorithm SAC --headless --num_envs 256
```

W&B：

```bash
cp task319_local_grasp_rl/wandb.example.sh task319_local_grasp_rl/wandb.sh
source task319_local_grasp_rl/wandb.sh
python task319_local_grasp_rl/train.py --algorithm SAC --headless --num_envs 256 --wandb
```

接入计划：

1. 保持 VISS v28 和 Nav2 主线不变。
2. 用现有规划器把右夹爪移动到目标 hover 点。
3. 将当前 RGB-D 目标转换成 policy 的局部观测。
4. 最后下降、闭合和抬升由 SAC policy 每帧输出小动作。
5. 成功后再进入现有 Nav2 投放状态。

## 8. 主要问题复盘

| 问题 | 根因 | 已采取方案 | 当前状态 |
| --- | --- | --- | --- |
| 视觉模块选中物体不一致 | YOLO、VLM、抓取候选没有绑定同一实例 | 统一到 VISS v28/Qwen-first，cycle 内绑定实例 | 基本解决 |
| GLM/旧 YOLO 效果差 | 模型链路和当前场景相机/任务不匹配 | 弃用 GLM 和旧 YOLO 默认主线 | 已解决 |
| GraspNet 零候选 | objectness 低、点云薄、内部碰撞过滤、mask/坐标误差 | 加 PLY 保存、阈值放宽、top-k、关闭预筛选 | 不作为默认 |
| 可达性误判 | 用固定阈值或站位不合适 | 动态桌边站位，右臂局部窗口筛选 | 可用但仍需优化 |
| 相机外参疑似偏移 | 抓取点和图像目标不重合 | 加投影图、RGB-D vs truth debug、站位后重拍 | 继续监控 |
| 手臂姿态扭曲 | 过强姿态约束、IK 分支跳变、目标不可达 | 减少轴向约束、position-first、局部 RL 计划 | 未彻底解决 |
| 夹爪碰飞物体 | 下降非直线、闭合过早、目标高度不合理 | top-grasp、hover、慢下降、慢闭合 | 仍是核心问题 |
| 投放瞬移不真实 | 早期直接写物体 pose 到桶里 | 改为末端先到桶口上方再释放 | 已改善 |
| 底盘 yaw 抖 | 轮子物理接触和映射不稳定 | `urdf_diagonal + kinematic_stable` | 明显改善 |
| Nav2 `STATUS_6` | 终点残差、局部控制器退出状态、长循环累积 | 软成功、重试、返回观察位策略 | 仍偶发 |

## 9. 后续改进措施

优先级从高到低：

1. 接入 SAC hover-to-grasp policy：
   - 先在小方块和标准垃圾物体上评估成功率。
   - 再替换主线 physical grasp 的最后下降、闭合和抬升。
   - 保留近距 carry 只做展示 fallback，不用于物理验收。

2. 强化抓取前闭环重定位：
   - 导航到站位后必须重新拍 RGB-D。
   - 若末端 hover 后目标发生移动，再用局部相机/RGB-D 刷新目标。
   - 可以考虑手腕相机作为最后 10 cm 的视觉伺服输入。

3. 减少 IK 病态姿态：
   - cuRobo 负责到 hover 的避障规划。
   - 最后下降使用局部闭环或密集 Cartesian waypoint。
   - 姿态只保留必要约束，不强求无意义轴向完全对齐。

4. 提升 Nav2 稳定性：
   - 对 `STATUS_6` 增加局部 retry 和容差判断。
   - 在关键拍照位做更稳定的面向桌面/垃圾桶 yaw 对齐。
   - 继续保留 `kinematic_stable`，避免回到纯接触轮地导致不可动。

5. 完善文档和验收：
   - 每次视频保存后自动退出。
   - 每个完整 run 记录命令、版本、分类表、失败原因。
   - 明确区分“演示成功”“物理抓取成功”“RL 训练成功”三类结果。

## 10. 当前验收口径

短期展示验收：

- 能识别桌面多个垃圾。
- 能给每个垃圾绑定分类。
- 能选择目标并导航到桌边站位。
- 能移动到对应垃圾桶侧。
- 释放时末端在正确桶口上方。
- 视频能完整看到机器人、桌面和垃圾桶。

物理抓取验收：

- 不启用旧吸取别名或近距 carry。
- 夹爪真实闭合夹住物体。
- 物体随夹爪被抬起。
- 导航过程中物体保持在夹爪中。
- 到桶口上方打开夹爪后，物体自然落入对应垃圾桶。

RL 接入验收：

- policy 在 hover 后局部闭环能稳定抓起训练物体。
- 在主线 RGB-D 定位误差和导航残差下仍能纠偏。
- 对小方块、纸盒、罐体、电池/笔类至少分别验证。
- 不依赖 simulator-only privileged observation 作为部署输入。
