# Task319 主线算法理论与底层实现说明

本文档说明当前 `task_319_garbage_sort` 主线中“视觉定位、动态站位、Nav2 导航、cuRobo 抓取执行”的数学原理和代码实现边界。这里不只描述功能，而是说明每一步输入、坐标系、公式、约束和实际调用位置。

## 1. 主线数据流

当前主线默认按以下闭环运行：

1. 头部 RGB-D 相机在当前机器人位姿下采集 `rgb / depth / K / camera_to_world`。
2. VISS v28 视觉链路给出候选物体实例：Qwen 全局识别、YOLO11 分割、VLM 分类与验证。
3. 对每个 YOLO mask 使用当前深度图反投影得到世界系点云。
4. 从点云估计物体 3D 几何中心、PCA 主轴、夹爪宽度和 top-grasp 目标点。
5. 根据目标点在桌面上的世界坐标，生成桌前、桌左、桌右候选站位，结合右臂局部可抓窗口打分。
6. 通过 Nav2 发送底盘目标位姿 `(x, y, yaw)`，机器人导航到抓取站位，朝向桌面法线方向。
7. 底盘到位后头部相机重新拍摄，重新运行 v28 视觉链路与 RGB-D 反投影，避免使用旧视角下的过期目标坐标。
8. 将更新后的抓取目标位姿、右臂当前关节初值、夹爪宽度交给 cuRobo/Isaac 执行。
9. cuRobo 第一阶段规划到预抓取悬停位；第二阶段构造笛卡尔竖直下降路点，逐点 IK，闭合夹爪并竖直抬升。
10. 后续根据 VLM 分类导航至对应垃圾桶侧，手臂移动至桶上方并释放。

主要实现入口：

- 视觉与主状态机：`task_319_garbage_sort/visual_grasp_record_demo.py`
- 右臂 cuRobo 封装：`task_319_garbage_sort/curobo_right_arm.py`
- 夹爪宽度控制：`task_319_garbage_sort/grasp_pipeline/execution/gripper_control.py`
- 动态站位记录：`cycle_*/dynamic_grasp_standpoint.json`

## 2. 坐标系定义

当前系统至少涉及四个核心坐标系：

- `W`：Isaac 世界坐标系。
- `C`：头部 RGB-D 相机光学坐标系。
- `B`：机器人 `base_link` 坐标系，也是 cuRobo 规划输入的基坐标系。
- `G`：右夹爪 TCP 坐标系，代码中 cuRobo 的 `ee_link` 为 `right_gripper_tcp`。

头部相机采集时保存：

```text
rgb: H x W x 3
depth: H x W, 单位 m, 来自 distance_to_image_plane
K: 3 x 3 camera intrinsics
T_WC: 4 x 4 camera_to_world
```

cuRobo 接收的目标不是世界系位姿，而是机器人基坐标系位姿：

```text
T_BG = inv(T_WB) @ T_WG
```

其中 `T_WB` 来自机器人 root pose，`T_WG` 是视觉链路构造出的目标 TCP 世界位姿。实现函数是 `world_pose_to_robot_base_pose()`。

## 3. YOLO Mask + 深度反投影到世界点云

代码入口是 `mask_points_world(mask, depth_m, intrinsics, t_camera_to_world)`。

对于 mask 中每个像素 `(u, v)`，读取深度 `d`。相机内参：

```text
K = [[fx, 0,  cx],
     [0,  fy, cy],
     [0,  0,  1 ]]
```

像素反投影到相机系：

```text
x_c = (u - cx) * d / fx
y_c = (v - cy) * d / fy
z_c = d
p_C = [x_c, y_c, z_c, 1]^T
```

再转到世界系：

```text
p_W = T_WC @ p_C
```

有效点过滤条件：

- 深度有限且 `0.05 m < d < 5.0 m`。
- 世界系高度在桌面物体窗口内：

```text
TABLE_SURFACE_Z + 0.006 < z_W < TABLE_SURFACE_Z + 0.45
```

- 若过滤后点数少于 25，则认为该 mask 无有效 3D 点云。

这一步只使用当前 RGB-D 相机数据。仿真真实物体 root pose 只用于 debug 对比和身份绑定，不作为抓取点来源。

## 4. 3D 几何中心估计

代码入口是 `rgbd_geometric_grasp_center(points_world)`。

输入点云：

```text
P = {p_i = (x_i, y_i, z_i)}
```

水平位置采用中位数而不是均值：

```text
x_center = median({x_i})
y_center = median({y_i})
```

这样做的原因是 YOLO mask 边缘、深度噪点、遮挡点会造成离群值；中位数对离群点更稳。

高度先计算：

```text
z_min = min({z_i})
z_max = max({z_i})
h = z_max - z_min
```

若 `h >= 0.012 m`，高度取物体厚度中点：

```text
z_center = 0.5 * (z_min + z_max)
```

否则使用点云高度中位数。最后施加桌面安全下限：

```text
z_center = max(z_center, TABLE_SURFACE_Z + 0.025)
```

因此“水平取中位数，高度取厚度中点”的准确含义是：

```text
center_W = [median(x), median(y), midpoint(z_min, z_max)].
```

若物体点云太薄，则高度中点退化为 `median(z)`，避免薄片噪声导致高度跳变。

## 5. PCA 主轴与夹爪宽度估计

代码入口：

- `estimate_rgbd_xy_principal_axes(points_world)`
- `estimate_parallel_jaw_grasp_width(points_world)`
- `estimate_rgbd_top_grasp_wrist_rotation(points_world)`

只在桌面平面 `XY` 上做 PCA：

```text
X = [[x_1, y_1],
     [x_2, y_2],
     ...
     [x_n, y_n]]

mu = median(X)
X_c = X - mu
Cov = cov(X_c)
```

对协方差矩阵做特征分解：

```text
Cov v_j = lambda_j v_j
```

小特征值方向对应点云短轴，大特征值方向对应长轴。为了抗离群点，轴向长度不是直接取 min/max，而是取 5% 到 95% 分位：

```text
short_extent = percentile(X_c @ v_short, 95) - percentile(X_c @ v_short, 5)
long_extent  = percentile(X_c @ v_long,  95) - percentile(X_c @ v_long,  5)
```

夹爪开口宽度估计：

```text
w_est = clamp(max(min_width, short_extent + calibration_clearance), 0, 0.120)
```

当前默认：

```text
min_width = 0.024 m
calibration_clearance = 0.004 m
max_width = 0.120 m
```

top-grasp 姿态中：

- 夹爪 TCP/forward 轴强制接近世界 `-Z`，避免斜扫桌面。
- 夹爪开口轴优先对齐物体点云短轴。
- 当前 cuRobo 主线可以使用 position-only 目标，因此姿态更多是初始姿态/偏好，不应被理解为必须让所有坐标轴严格重合。

## 6. Top-Grasp 目标点计算

代码入口是 `top_grasp_target_from_points(center_world, points_world)`，随后由 `selected_grasp_from_rgbd_center()` 写入 `SelectedGrasp`。

当前不会直接把 TCP 目标设为物体 3D 几何中心，而是从点云最高点向下插入一段深度：

```text
z_max = max({z_i})
z_min = min({z_i})
h = z_max - z_min
```

令：

```text
d_requested = top_grasp_depth_m
d_min       = top_grasp_min_depth_m
d_max       = top_grasp_max_depth_m
rho         = top_grasp_depth_height_fraction
d_wrap      = top_grasp_wrap_extra_depth_m
z_floor     = TABLE_SURFACE_Z + top_grasp_table_clearance_m
```

几何中心到顶部的深度：

```text
d_center = z_max - z_center
```

若启用 adaptive depth：

```text
d_raw_without_wrap = max(d_requested, d_min, d_center, rho * h)
d_raw = d_raw_without_wrap + d_wrap
```

再受最大插入深度和桌面安全高度裁剪：

```text
d_table = z_max - z_floor
d = min(d_raw, d_max, d_table)
d = max(d, 0)
```

最终抓取 TCP 目标点：

```text
p_grasp = [x_center, y_center, max(z_floor, z_max - d)]
```

悬停点：

```text
p_hover = [x_center, y_center, max(z_max + hover, p_grasp.z + hover, TABLE_SURFACE_Z + min_table_clearance)]
```

这就是“从物体上方进入、只插入顶部区域”的实现。它不是简单抓几何中心，也不是把夹爪目标放到桌面以下。

当前参数默认来自命令行参数：

- `top_grasp_min_depth_m`: 约 `0.038 m`
- `top_grasp_max_depth_m`: 约 `0.085 m`
- `top_grasp_depth_height_fraction`: 约 `0.55`
- `top_grasp_wrap_extra_depth_m`: 约 `0.012 m`
- `top_grasp_hover_m`: 约 `0.12 m`

在 `--mind_sort_demo` 相关默认配置中，部分参数会被主线逻辑收紧到更适合展示抓取的值。

## 7. 后导航重拍与坐标实时性

初始相机拍摄用于全桌观察和站位规划，但真正抓取前会在底盘到达站位后重新拍摄：

```text
rgb2, depth2, K2, T_WC2 = capture_head_camera(scene)
```

然后重新执行：

```text
v28 perception -> YOLO mask -> RGB-D point cloud -> center/top-grasp
```

如果能在重拍图像中找到同一个物体，则主流程会替换：

```text
rgb, depth, K, T_WC, target = rgb2, depth2, K2, T_WC2, target2
```

这一步对应代码中的 `standpoint_reshoot`。选择同一目标时优先使用冻结身份：

- `scene_key`
- `scene_object_name`
- VLM 类别与名字
- 与原始 3D center 最近的同类目标

这样做的目的：避免机器人移动后还拿“起点相机视角下的旧 camera_to_world 和旧目标点”去抓。

## 8. 动态抓取站位生成

代码入口是 `compute_grasp_standpoint_candidates(target_world_xyz, robot_pose_xyyaw)`。

目标点：

```text
p_t = [x_t, y_t, z_t]^T
```

候选站位来自桌子三个方向：

```text
front: yaw = 0
left:  yaw = pi / 2
right: yaw = -pi / 2
```

这些 yaw 让机器人到位后朝向桌面法线方向。

右臂局部偏好窗口定义在机器人 base 坐标系中：

```text
preferred_local = [0.50, -0.16] m
local_x_bounds = [0.35, 0.56] m
local_y_bounds = [-0.46, 0.08] m
```

世界系到候选站位局部坐标的转换为：

```text
delta = [x_t - x_s, y_t - y_s]

x_local =  cos(yaw) * delta_x + sin(yaw) * delta_y
y_local = -sin(yaw) * delta_x + cos(yaw) * delta_y
```

反过来，根据期望让目标落在右臂 preferred local 位置，候选站位初值为：

```text
stand_xy = target_xy - R(yaw) * preferred_local
```

然后按桌边约束裁剪：

- front：`x_s <= TABLE_X_MIN - clearance`，`y_s` 限制在桌前通道范围内。
- left：`x_s` 限制在桌子 X 范围内，`y_s <= TABLE_Y_MIN - clearance`。
- right：`x_s` 限制在桌子 X 范围内，`y_s >= TABLE_Y_MAX + clearance`。

当前桌面边界：

```text
TABLE_X_LIMITS = [0.92, 2.68]
TABLE_Y_LIMITS = [-0.42, 0.42]
clearance 默认约 0.42 m
```

候选站位打分：

```text
local_error =
    abs(x_local - 0.50)
  + 0.45 * abs(y_local - (-0.16))

travel_distance = norm(stand_xy - robot_xy)

score =
    local_error
  + 0.20 * travel_distance
  + 0.75 * number_of_right_arm_window_risks
```

硬拒绝项主要是：

- 目标不在桌面边界内。
- 目标高度不在桌面抓取窗口内。

右臂窗口目前是风险/惩罚项，不是完整 IK 可达性证明。真正能不能执行，最终仍由 cuRobo 在抓取阶段根据机器人当前关节状态、障碍物和目标 TCP 进行求解。

动态站位的完整记录写入：

```text
cycle_*/dynamic_grasp_standpoint.json
```

其中包括：

- `target_world_xyz`
- `allowed_sides`
- `right_arm_window`
- 每个 candidate 的 `pose / target_local_xy / score / reject_reason`
- `selected`

## 9. Nav2 导航接口与底盘执行

代码入口：

- `run_nav2_goal()`
- `nav2_goal_process()`
- `ExternalRos2NavBridgeClient.exchange()`
- `apply_wheel_velocity()`

状态机向 Nav2 发送的目标是：

```text
target_pose = (x_goal, y_goal, yaw_goal)
```

`nav2_goal_process()` 启动 `scripts/ros2_nav_goal_client.py`，通过 ROS2 Nav2 action 发送目标。Nav2 负责全局/局部规划，输出 `/cmd_vel`。

仿真侧通过 socket bridge 每步交换：

1. Isaac 发送当前机器人位姿、速度和合成 scan。
2. ROS2 bridge 返回 Nav2 的速度命令：

```text
cmd_vel = (linear_x, angular_z)
```

仿真侧不自己写导航路径，只做速度命令执行和物理/运动学耦合。

当前稳定版本底盘执行使用：

```text
nav_actuation_mode = wheel
wheel_ground_coupling = kinematic_stable
```

即：仍然给轮子写速度目标，同时使用稳定的平面运动积分来减少纯接触轮胎打滑和 yaw 抖动。平面积分公式：

```text
psi_mid = psi_k + 0.5 * omega * dt
psi_{k+1} = wrap_to_pi(psi_k + omega * dt)
x_{k+1} = x_k + v * cos(psi_mid) * dt
y_{k+1} = y_k + v * sin(psi_mid) * dt
```

因此当前 Nav2 仍是规划和控制源，Isaac 端只是把 `/cmd_vel` 接入轮式机器人执行层，并用 `final_dock_to_pose()` 在目标附近做小范围对准。

## 10. cuRobo 右臂模型与输入

代码入口是 `KuavoRightArmCuroboPlanner`。

cuRobo 使用 Kuavo URDF 与附加两指夹爪 URDF 合并后的模型。当前活动自由度只包含右臂 7 个关节：

```text
zarm_r1_joint
zarm_r2_joint
zarm_r3_joint
zarm_r4_joint
zarm_r5_joint
zarm_r6_joint
zarm_r7_joint
```

以下关节在 cuRobo 模型中锁定：

```text
knee_joint = 0
leg_joint = 0
waist_pitch_joint = 0
waist_yaw_joint = 0
left_finger_joint = 0.030
right_finger_joint = 0.030
```

cuRobo 的 `base_link` 是 `base_link`，`ee_link` 是通过 fixed joint 增加的 `right_gripper_tcp`。TCP 相对 `gripper_base` 的固定偏移来自夹爪几何：

```text
right_gripper_tcp = gripper_base + tcp_offset
```

当前配置还给右臂、夹爪添加 collision spheres，并把桌面与桌腿作为 cuRobo world cuboids。这样 cuRobo 规划时会考虑自碰撞和桌面障碍。

每次规划输入：

```text
q0: 当前 Isaac 右臂 7 关节角，顺序严格匹配 RIGHT_ARM_JOINT_NAMES
T_WG: 目标 TCP 世界位姿
T_BG: inv(T_WB) @ T_WG
w_open: 夹爪开口宽度
```

`q0` 会先按 URDF joint limit 做轻微裁剪，避免起点已经越界导致 cuRobo 查询无效。

夹爪宽度不是 cuRobo 的活动规划变量。cuRobo 只规划右臂 7 关节；夹爪宽度在 Isaac 执行轨迹时同步写入两个 finger joint。

## 11. cuRobo 第一阶段：规划到预抓取悬停位

代码入口是 `execute_curobo_right_arm_attempt()` 内部的 `plan_and_run_stage()`。

预抓取目标来自第 6 节的 `p_hover`：

```text
T_WG_hover = [R_grasp, p_hover]
T_BG_hover = inv(T_WB) @ T_WG_hover
```

规划调用：

```text
MotionGen.plan_single(start=q0, goal=T_BG_hover)
```

当前主线通常启用 position-only TCP 模式：

```text
reach_vec_weight_rot_xyz_pos_xyz = [0, 0, 0, 1, 1, 1]
```

含义是：目标位置是硬重点，姿态不作为同等刚性的三轴约束。这样可以减少“为了对齐某个轴导致手臂扭成病态姿态”的问题。

如果启用官方 Kuavo analytic IK seed，流程会先采样若干 wrist roll 姿态，调用 Kuavo IK 求一个右臂种子解；若种子有效，则 cuRobo 可先规划到该关节种子，或者在 position-only 失败后再用它作为兜底。最终执行的仍是 Isaac 中的右臂关节轨迹。

执行轨迹时使用 `run_curobo_joint_trajectory()`。它不会一次跳到目标关节，而是对 cuRobo 插值轨迹做 min-jerk 子步进：

```text
alpha = 10s^3 - 15s^4 + 6s^5
q_des = (1 - alpha) q_start + alpha q_goal
```

并限制单步最大关节变化，降低物理抖动。

## 12. cuRobo 第二阶段：竖直下降、闭合、抬升

最后下降段不再允许 cuRobo 自由重新规划一条弧线。当前实现手动构造世界系竖直 TCP 路点：

```text
T_start = actual TCP pose after hover
T_goal  = target grasp TCP pose
```

锁定实际 hover 的水平位置：

```text
x_i = x_start
y_i = y_start
```

只插值高度：

```text
z_i = (1 - alpha_i) * z_start + alpha_i * z_goal
alpha_i = i / N
```

姿态使用以下策略之一：

- position-only：只把位置作为硬约束，同时通过解选择评分保持 hover wrist 姿态连续。
- full-pose：保持固定目标旋转或 hover 旋转。

然后把每个世界系路点转换到机器人 base 系：

```text
T_BG_i = inv(T_WB) @ T_WG_i
```

逐点调用 cuRobo IK：

```text
solve_ik_chain_for_poses_sequential(q_start, [T_BG_1, ..., T_BG_N])
```

每个路点会有多个 seed 解。当前解选择评分为：

position-only 时：

```text
score =
    ||q - q_prev||
  + 10.0 * position_error
  + 5.0 * rotation_distance_from_hover
  + 2.0 * rotation_distance_from_previous
  + 0.75 * wrist_joint_delta_from_hover
```

full-pose 时：

```text
score =
    ||q - q_prev||
  + 10.0 * position_error
  + 0.25 * rotation_error
```

若相邻路点需要的关节跳变超过阈值，则判定该 IK 链不连续，防止手臂突然翻腕。

执行下降轨迹时还会监控：

```text
||TCP_xy_current - TCP_xy_locked|| <= max_cartesian_xy_error
```

如果 XY 漂移过大则中止下降，避免夹爪斜向扫过物体。

下降到位后执行慢速闭合：

```text
run_slow_gripper_close()
```

闭合后先做接触检查：

```text
evaluate_gripper_contact_before_lift()
```

只有夹爪接触条件满足，才进入 lift 阶段。

抬升阶段同样构造竖直路点：

```text
x_i = x_after_close
y_i = y_after_close
z_i: z_after_close -> z_after_close + lift_height
R_i = R_after_close
```

再用 cuRobo IK 链求解并执行。这样抬升不会横向拖拽物体。

## 13. 夹爪宽度到底如何发送

夹爪控制实现是 `AttachedParallelGripper`。

平行夹爪目标宽度 `w` 与两个 prismatic finger joint 的关系：

```text
left_finger_joint  = w / 2
right_finger_joint = w / 2
```

正常设置会裁剪：

```text
w = clamp(w, min_width, max_width)
max_width = 0.120 m
default_open_width = 0.110 m
```

抓取前打开宽度：

```text
w_open = clamp(w_est + grasp_extra_clearance)
```

当前 `grasp_extra_clearance_m` 默认约 `0.018 m`。

执行 cuRobo 轨迹时，右臂关节由 cuRobo 轨迹控制，夹爪宽度作为同步目标写入 Isaac：

```text
full_target[:, right_arm_joint_ids] = q_des
full_target[:, finger_joint_ids] = [w / 2, w / 2]
```

因此“将夹爪宽度发送至 cuRobo”严格说是不准确的：宽度进入的是 Isaac 执行控制层，不是 cuRobo 右臂 MotionGen 的优化变量。cuRobo 只规划右臂和 TCP。

## 14. 调试图与 8mm 定位误差如何计算

相关输出：

- `target_pose_debug.jpg/json`
- `rgbd_vs_truth_debug.png/json`
- `dynamic_standpoint_rgb.png`
- `dynamic_standpoint_topdown.png`
- `dynamic_grasp_standpoint.json`

世界点反投影回图像用于校验外参：

```text
p_C = inv(T_WC) @ p_W
u = fx * x_C / z_C + cx
v = fy * y_C / z_C + cy
```

`rgbd_vs_truth_debug` 中会画：

- 紫色：2D detection center。
- 青色：RGB-D mask 反投影得到的世界中心再投影回图像的位置。
- 黄色：仿真真实物体 root pose 投影位置。

定位误差计算：

```text
delta = p_rgbd_center_W - p_sim_truth_W
xy_error = norm(delta_xy)
xyz_error = norm(delta_xyz)
```

你提到的约 `8mm` 误差，就是这类 debug JSON 中 `xyz_error_m` 或 `xy_error_m` 的量级。它用于检查相机内外参、深度反投影、mask 与深度对齐是否可信。

注意：真实机器人没有 simulator truth。这个对比只在仿真中用于标定与排错，不能作为真实运行时抓取点来源。

## 15. 当前实现的真实边界

1. RGB-D 点云几何中心是当前抓取点的主要来源；它依赖 YOLO mask 和深度图对齐质量。
2. 动态站位的右臂窗口是几何启发式打分，不是完整全身 IK 可达性证明。
3. 真正的运动可达性由 cuRobo 在当前底盘位姿、当前关节初值、桌面障碍和 TCP 目标下求解。
4. 最后下降段已经从自由规划改成“手动笛卡尔竖直路点 + cuRobo IK 链”，理论上不应再斜向扫物体；若仍碰撞，优先检查 RGB-D 目标点是否偏、夹爪 TCP 标定是否偏、或实际 hover XY 是否未对齐。
5. 夹爪宽度控制是 Isaac finger joint 控制，不参与 cuRobo 优化；如果物体比最大开口宽，cuRobo 能到位也无法物理夹起。
6. `dynamic_grasp_standpoint.json` 是分析站位选择的核心记录；抓取误差则主要看 `target_pose_debug.json`、`rgbd_vs_truth_debug.json` 和 `execution.segments` 中的 cuRobo TCP error。
