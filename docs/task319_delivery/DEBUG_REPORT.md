# Task 3.19 Problem Debugging Report

更新时间：2026-06-27

## 1. 视觉目标不一致

问题：早期 YOLO、VLM、GraspNet 各自选目标，出现“框选 A、分类 B、抓取 C”的错配。

处理：

- 主线统一为 VISS v28/Qwen-first。
- Qwen 先做全局候选，YOLO11 对 ROI 精分割，Qwen/VLM 再做验证和四分类。
- Task319 在 cycle 内绑定同一个 `scene_key`、mask、RGB-D 3D 中心和分类结果。
- 输出 `v28_original/`、overlay、mask、`mind_sort_cycle.json` 便于追溯。

状态：基本解决。偶发分类歧义仍存在，例如 `trash_potted_meat_can_0` 在不同 run 中可能受“金属罐”和“食物残留”上下文影响。

## 2. RGB-D 定位和相机外参误差

问题：使用观察位旧坐标或外部视觉链路相机参数时，抓取点和图像目标有明显偏差。

处理：

- 使用当前 IsaacLab 主场景 `head_rgbd` 的 intrinsics、depth 和 `camera_to_world`。
- 导航到抓取站位后重新拍照，用站位处 RGB-D 重新计算目标 3D 点。
- 保存 `rgbd_vs_truth_debug.json/png`、`target_pose_debug`、投影误差和真值对比。

状态：可追溯。20260625_150100 的 cycle 记录中，RGB-D 中心和仿真 root 的 XY 误差通常在厘米级以内。

## 3. GraspNet 候选不稳定

问题：GraspNet 在薄点云、mask 边缘噪声和桌面物体上经常产生零 objectness 或 runtime IK 失败。

处理：

- 保留 GraspNet Baseline 为诊断模式，不再作为默认主线。
- 默认使用 RGB-D top grasp：取点云 XY 中心、`Z_max - depth` 作为 TCP 目标。
- 加入点云质量、宽度估计、目标尺寸和可达性记录。

状态：GraspNet 仍用于研究和对比；主线默认 `rgbd_center`。

## 4. 右臂 IK 和轨迹姿态问题

问题：右臂末端姿态约束过强时，会出现手臂扭转、关节跳变、桌面附近斜向扫物体。

处理：

- cuRobo 负责到 hover 的高层规划，最后下降尽量保持垂直。
- 使用 Kuavo 解析 IK seed 和 cuRobo 规划结果做互相审计。
- 记录每段 `segments`、TCP 误差、joint limit、FK/IK 诊断。
- 对最后下降加入 sequential waypoint、关节跳变阈值、XY drift guard。

状态：能生成可执行段，但复杂物体上仍可能因 Cartesian final descent 未到目标而失败。最新严格物理 run `20260626_145846` 即失败在该阶段。

## 5. 真实物理夹爪抓取不稳定

问题：严格要求物体被双指真实夹住、抬升、随末端移动时，复杂垃圾物体成功率不足。

处理：

- 调整夹爪摩擦、接触参数、闭合宽度、闭合保持时间和 top grasp 深度。
- 用 `gripper_physics_test.py`、`topdown_cube_grasp_test.py` 分离验证夹爪物理。
- 完整系统展示时显式启用 `--mind_sort_suction_assist` 或 `--mind_sort_gripper_proximity_assist`，让到达近距阈值的物体绑定到右夹爪 TCP 后继续分类投放。
- 严格物理命令默认禁用辅助，并在失败时停止和写明原因。

状态：当前最大未闭环项。答辩中应明确辅助演示不等同于严格物理夹爪验收。

## 6. 底盘和 Nav2 问题

问题：纯轮地接触模式下底盘运动不稳定，早期有 yaw 抖动和目标点残差。

处理：

- 使用 ROS2 Nav2 `NavigateToPose`，不使用手写路径作为正式导航。
- Isaac 侧采用 `wheel_drive_model=urdf_diagonal` 和 `wheel_ground_coupling=kinematic_stable`。
- 保存 `nav2_stack/` 的地图、参数、行为树和日志。
- 导航到抓取站位后重新拍照，降低残差对抓取坐标的影响。

状态：Nav2 已能完成桌边站位和桶边移动。长循环末尾仍可能 `STATUS_6`，例如 `20260625_150100` 在第 9 次返回观察位失败，但此前 8 个 cycle 均完成投放。

## 7. 投放瞬移问题

问题：早期投放曾直接写物体 pose 到桶内，视觉上不符合“移动至桶口后释放”。

处理：

- 投放阶段先导航到目标桶侧。
- 右夹爪 TCP 再移动到对应桶口上方。
- 打开夹爪或释放 carry 绑定。
- metadata 中记录 `object_teleport_used=false`、`final_pose_inside_bin_opening_xy=true`。

状态：已改善。完整演示 run 的已完成 cycle 均记录为桶口释放。

## 8. 干扰样本鲁棒性

样本：

- `trash_potted_meat_can_0`: 带食物残留语义的金属罐，可回收和厨余之间存在语义冲突。
- 历史 USD 场景中的 `trash_dirty_bottle`: 脏污瓶，测试“可回收物上沾污物”。

现象：

- `20260625_150100` 中 `trash_potted_meat_can_0` 被识别为金属罐并投到可回收桶。
- `20260626_145846` 严格单轮中同一目标分类为厨余垃圾，同时物理抓取失败。

结论：鲁棒性样本已覆盖，但分类规则还需在 prompt/规则层明确“沾污可回收物”的判定策略，例如按比赛规则优先投其他/厨余或先清洁后可回收。

## 9. 后续计划

1. 接入 `task319_local_grasp_rl/` 的 hover-to-grasp policy，替换手写最后下降、闭合和抬升。
2. 对 strict physical grasp 建立固定 10 物体评测表，单独统计真实抓取成功率。
3. 为干扰样本增加 prompt 判定准则和固定测试集。
4. 对 Nav2 `STATUS_6` 增加软成功判定、重试和返回观察位局部修正。
5. 将 `video_manifest.json` 和人工整理视频名保持一致，减少交付路径混淆。
