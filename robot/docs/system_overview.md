# Kuavo 3.19 垃圾分类 — 系统全览

## 一、系统分布

```
┌─ Windows 11 本机 ─────────────────────────────────────────────┐
│  Python 3.12 (conda base)                                      │
│  MuJoCo 3.9.0 (仿真渲染 + 物理)                                 │
│  GroundingDINO (开放词汇目标检测)                                │
│  GraspNet (6-DOF 抓取姿态估计)                                   │
│  ReKep (关系关键点约束规划)                                      │
│  A* 路径规划 (MobileBaseAStarPlanner)                           │
│  Jacobian IK 精调 (20步迭代, MuJoCo mj_jacBody)                │
├───────────────────────────────────────────────────────────────┤
│  Docker Desktop (WSL2 backend)                                  │
│  ┌─ 容器 kuavo_official_ros (ROS1 Noetic) ──────────────────┐  │
│  │  roscore (主节点)                                          │  │
│  │  arms_ik_node (Drake IK 求解器)                            │  │
│  │  arm_trajectory_bezier_process (臂 Bezier 轨迹)            │  │
│  │  wheel_bridge.py (底盘 /cmd_vel 桥接)                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## 二、全局流程

```
[1] 加载场景 ──→ [2] 感知扫描 ──→ [3] 抓取规划 ──→ [4] 导航到位
                                         │
                    ┌────────────────────┘
                    ▼
[5] 臂运动执行 ──→ [6] 夹持 + 携带 ──→ [7] 投放 ──→ [8] 下一物体
```

## 三、每一步详解

### [1] 加载场景

**工具**: MuJoCo 3.9.0  
**模型**: `simulation/scenes/task_319_kuavo_wheel_s62.xml`  
**内容**: Kuavo 5-W v62 轮式臂机器人 + 桌面 + 4 个垃圾桶 + 4 个垃圾物体（塑料瓶/铝罐/香蕉皮/电池）+ 2 个相机

**实现**: `MuJoCoEnv.__init__()` 加载 XML → `mjModel` + `mjData` → `KuavoWheelGarbageController` 初始化

---

### [2] 感知扫描

**工具**: GroundingDINO (开放词汇检测)  
**模型**: `models/grounding-dino-base` (本地权重)  
**输入**: MuJoCo 相机 `camera_rgb` 的 RGB 图像 (640×480)  
**输出**: 检测到的物体列表 [{class, bbox, confidence}, ...]

**实现细节**:
1. `self.env.render("camera_rgb")` → RGB numpy (480,640,3)
2. `self.env.render_depth("camera_rgb")` → depth numpy (480,640)
3. `self.perception.detect(rgb, text_prompt, depth)` → GroundingDINO 前向推理 → 3D 边界框中心点
4. 物体按距离排序，最近优先

---

### [3] 抓取规划

**工具**: ReKep + GraspNet  
**输入**: 物体 3D 位置 + 点云  
**输出**: hover/grasp/lift 三个目标位姿 (世界坐标)

**实现细节**:
1. `GraspNetEstimator` 估计 6-DOF 抓取姿态
2. `ReKepIntegration.build_hover_pose()` → hover 位姿 (物体上方 0.15m)
3. `_kuavo_pinch_target_for_grasp_center()` → 指心→r_pinch 转换 (几何常量 +0.14m Z)
4. `_kuavo_safe_approach_pose()` → 安全接近位姿 (hover + 0.08m)
5. 输出三个 waypoint: approach → pregrasp(hover) → grasp

---

### [4] 导航到位

**工具**: A* 路径规划 + 平面基座 PD 控制  
**模型**: `MobileBaseAStarPlanner` (2D 栅格 A*)  
**输入**: 当前基座位姿 + 目标基座位姿  
**输出**: 基座到达目标

**实现细节**:
1. `_find_kuavo_feasible_grasp_pose()` — 搜索 240 个候选基座位姿 (5 半径 × 16 角度 × 3 偏航)
2. 每个候选: A* 路径规划 + IK 可达性检查
3. 排序选最优，`move_planar_to()` 执行 PD 导航：
   - P 增益: X/Y=3000, Yaw=1200
   - 力 ramp: max_delta=50
   - 底座阻尼: X/Y=20, Yaw=10 (MuJoCo XML)

---

### [5] 臂运动执行 ⭐ 核心链路

这是最复杂的部分，分 5 个子步骤：

#### 5a. 世界坐标 → base_local 变换

**输入**: 目标位姿 (世界坐标), 基座位姿 `[x, y, yaw]`  
**输出**: base_local 坐标

```python
dx = target_x - base_x
dy = target_y - base_y
local_x = dx*cos(yaw) + dy*sin(yaw)
local_y = -dx*sin(yaw) + dy*cos(yaw)
local_z = target_z  # Z 不变，基座无 Z 运动
```

#### 5b. 官方 IK 求解 (Docker ROS1)

**工具**: Drake IK (`arms_ik_node`, `motion_capture_ik` 包)  
**服务**: `/ik/two_arm_hand_pose_cmd_srv`  
**输入**: base_local 目标 `[x, y, z, roll, pitch, yaw]` + 当前关节角  
**输出**: 14 个关节角 (7 左臂 + 7 右臂) + 4 个躯干角

**实现细节** (`planning/kuavo_official_ik.py`):
1. 发布当前 MuJoCo 关节状态到 Docker `/joint_states` (ROS 名称映射: `zarm_r1_joint`→`r_arm_pitch` 等)
2. 构造 `twoArmHandPoseCmd` 请求 (对齐官方 demo `leju_claw_cylinder_pub.py`):
   - `use_custom_ik_param = False` (用 IK 节点默认参数)
   - 右臂: `pos_xyz`=目标位置, `quat_xyzw`=`quaternion_from_euler(roll, pitch, yaw)`
   - 左臂: `pos_xyz`=`[0, 0.3, -0.5]`, `quat_xyzw`=`[0,0,0,1]` (dummy 目标)
3. 通过 `docker exec` 执行内嵌 Python 脚本 → JSON 返回 `{q_arm, q_torso, success}`

**关节名映射** (Drake ↔ MuJoCo):
```
IK 输出:    r_arm_pitch  r_arm_roll  r_arm_yaw  r_forearm_pitch  r_hand_yaw  r_hand_pitch  r_hand_roll
MuJoCo:     zarm_r1(Y)   zarm_r2(X)  zarm_r3(Z) zarm_r4(Y)       zarm_r5(Z)  zarm_r6(X)    zarm_r7(Y)
```
注: IK 的 "hand_pitch"(索引5) 物理轴是 X=roll → zarm_r6, IK 的 "hand_roll"(索引6) 物理轴是 Y=pitch → zarm_r7

#### 5c. 躯干关节应用

IK 用躯干俯仰做高度调节。`q_torso = [?, waist_pitch, waist_yaw, ?]`

```python
self._posture_targets["waist_pitch_joint"] = q_torso[1]  # 躯干前后倾
self._posture_targets["waist_yaw_joint"] = q_torso[2]     # 躯干旋转
```

#### 5d. Jacobian 位置精调 (20 步迭代)

**工具**: MuJoCo `mj_jacBody()` 计算雅可比矩阵  
**输入**: IK 关节角 + 目标位置  
**输出**: 精调后的关节角

**实现细节** (`_jacobian_position_step`):
1. 将 IK 关节角写入 scratch `MjData` → `mj_forward()` 算 FK
2. 误差 = `target_pos - FK(seed_q)`
3. 若误差 < 0.005m 则结束
4. `mj_jacBody(model, data, jacp, jacr, body_id)` → 3×nv 雅可比
5. 截取激活关节列: `J = jacp[:, active_dofs]`
6. 阻尼最小二乘: `dq = J.T @ inv(J@J.T + 0.001*I) @ err`
7. 步长限幅 0.3 rad, 关节限位 clamp
8. 更新 `seed_q += dq`, 回到步骤 1

#### 5e. MuJoCo PD 执行

**工具**: MuJoCo 转矩电机  
**实现**: `_set_actuator_for_joint()` 用 PD 将关节驱动到目标角:
```python
torque = kp*(target - q) - kd*qd
kp=22(N·m/rad, 臂关节), kd=2.2
```

---

### [6] 夹持 + 携带

**工具**: MuJoCo weld 约束  
**实现**:
1. `close_gripper()` → 70 步渐进闭合 (MuJoCo PD + 官方 SDK 并行)
2. `evaluate_pinch_grasp()` → MuJoCo 几何检查 (指尖接触检测)
3. `grasp.attach()` → MuJoCo weld 约束 (物体粘到夹爪)
4. `plan_arm_to("right", lift_pose)` → 抬升物体

---

### [7] 投放

**工具**: 官方 SDK + MuJoCo PD  
**实现**: `_navigate_to(bin_pos)` → `open_gripper()` → 物体落入垃圾桶

---

### [8] 错误恢复

- 抓取失败 → 重新观测/重新规划 (最多 3 次)
- `--full` 模式禁止静默降级到本地 IK
- 每次任务结束输出 trace JSON 到 `results/traces/`

---

## 四、关键接口汇总

| 接口 | 方向 | 位置 | 协议 |
|------|------|------|------|
| 官方 IK | Windows → Docker | `planning/kuavo_official_ik.py` | `docker exec` + ROS1 service |
| 官方臂轨迹 | Windows → Docker | `planning/kuavo_official_control.py` | `docker exec` + ROS1 topic |
| 官方底盘 | Windows → Docker | `planning/kuavo_official_control.py` | `docker exec` + ROS1 topic |
| 官方夹爪 | Windows → Docker | `planning/kuavo_official_control.py` | `docker exec` + ROS1 topic |
| MuJoCo 物理 | 本地 | `task_319_garbage_sort/kuavo_controller.py` | 直接 API (mj_step, mj_jacBody, ctrl) |
| MuJoCo 渲染 | 本地 | `robot_common/env/` | 直接 API (mj_render) |
| 感知 | 本地 | `perception/` | GPU 推理 (GroundingDINO, GraspNet) |
| 规划 | 本地 | `task_319_garbage_sort/pipeline.py` | 纯 Python (A*, ReKep) |

## 五、启动命令

```powershell
# 1. 启动 Docker 服务
powershell -File scripts/start_docker_services.ps1

# 2. 运行 (带仿真窗口)
cd E:\workspace\bigdata\project
$env:PYTHONUNBUFFERED = '1'
python run_all.py --task 319 --config configs\task_319_kuavo_wheel_s62.yaml --full --demo-objects 1 --live-debug
```
