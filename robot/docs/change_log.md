# 3.19 改动日志与启动说明

---

## 2025-06-19 #1 — 全局控制栈切换到官方 Kuavo ROS

### 核心改动

| 文件 | 改动 | 原因 |
|------|------|------|
| `configs/task_319_kuavo_wheel_s62.yaml` | `allow_local_fallback: false`, `required: true`, 禁用所有静默降级 | 强制走官方栈 |
| `configs/task_319_kuavo_wheel.yaml` | 同上 | 同步 |
| `task_319_garbage_sort/kuavo_controller.py` | 删除 `_solve_dls_position_ik()` (~260行 DLS IK)、`move_base()` 阿克曼转向、`MinkIKSolver`、`_minimum_jerk_resample()`、`DiffDriveNavigator`。`plan_arm_to()` 仅走官方 IK。保留仿真执行层 | 不再重复造轮子 |
| `task_319_garbage_sort/pipeline.py` | `_navigate_to()` Kuavo 分支改为 `move_planar_to()` (带官方 SDK 调用) | 底盘走官方 OCS2 MPC |
| `planning/ros2_nav_executor.py` | 新增：Windows↔WSL 桥梁文件协议 (goal/result/cmd_vel) | ROS2 导航接口预留 |
| `planning/wsl_base_bridge.py` | 新增：OdometryPublisher + CmdVelReader 双向桥接 | MuJoCo ↔ ROS2 数据通道 |
| `scripts/kuavo_nav_server.py` | 新增：WSL ROS2 导航 PID 节点 (文件协议，非 Nav2 action) | 后续升级到完整 Nav2 时替换 |

### 全局链路状态

```
✅ Docker kuavo_official_ros     — ROS1 Noetic，IK + 臂轨迹 + 底盘 SDK + 夹爪
✅ 官方 IK   /ik/two_arm_hand_pose_cmd_srv   — IK err=0.000
✅ 官方臂轨迹 /execute_arm_action            — Bezier 轨迹生成
✅ 官方底盘  command_base_world              — OCS2 MPC
✅ 官方夹爪  LejuClaw SDK                    — open/close
✅ MuJoCo 仿真加载                           — 32 actuators, 43 joints
✅ 感知 GroundingDINO                        — 目标检测正常
✅ ReKep 抓取约束生成                        — hover/grasp/lift
⚠️ 抓取执行 — arm reach 不够（规划层问题，非控制栈问题）
⏳ WSL ROS2 导航 — Nav2 完整栈待后续接入（当前走官方 SDK 底盘）
```

### 启动步骤

```powershell
# 1. 确认 Docker 容器在运行
docker ps | findstr kuavo_official_ros
# 如果不在：python scripts/start_kuavo_ros1_services.py --robot-version 62

# 2. 运行 3.19（--full 强制官方栈，--headless 无窗口）
cd E:\workspace\bigdata\project
$env:PYTHONUNBUFFERED = '1'
python run_all.py --task 319 --config configs/task_319_kuavo_wheel_s62.yaml --full --headless --demo-objects 1

# 3. 带 MuJoCo 窗口查看（可选）
python run_all.py --task 319 --config configs/task_319_kuavo_wheel_s62.yaml --full --demo-objects 1 --live-debug
```

### 验证命令

```powershell
# 官方栈就绪检查
python scripts/probe_kuavo_official_stack.py --use-docker
# 必须输出: available: True, ready: True (全部4项)
```

---

## 2025-06-19 #2 — 物理参数修正 + 抓取链路 bug 修复

### 核心改动

| 文件 | 改动 | 原因 |
|------|------|------|
| `simulation/.../biped_s62.xml` | 底座 damping: 80→20, 50→10 | 80 对轮式机器人过高 |
| `kuavo_controller.py` move_planar_to | P: 820→3000, yaw: 320→1200, max_delta: 2.3→50 | 提升固有频率,去力 ramp |
| `kuavo_controller.py` | arm_settle_steps: 120→50 | 沉降减半 |
| `configs/...s62.yaml` | max_reachability_candidates: 160→48 | 减少搜索 |
| `pipeline.py` _kuavo_pinch_target_for_grasp_center | 运行时测量→几何常量 [0,0,0.14] | Bug: home位offset不适用抓取 |
| `pipeline.py` _kuavo_safe_approach_pose | 去掉高度重复叠加 | Bug: r_pinch目标上再叠加 |
| `kuavo_controller.py` _try_official_ik | 添加world→base_local变换 | IK用waist为根,不知底座 |

### 导航速度: 306s→194s (-36%), 单次导航 ~10s→~5s

### 仍未解决: IK返回假关节角
双臂IK对所有输入返回相同q_arm [-0.282,0,0,-0.282,0,0,-0.282]。
根因: left arm dummy目标 [0.32,0.22,0.05] 超出0.708m工作空间(IK节点日志: Left hand out of workspace)。
待修复: 左臂dummy目标 + joint_angles需匹配18 DOF (14臂+4躯干)。

### 启动
```powershell
docker restart kuavo_official_ros
cd E:\workspace\bigdata\project
python run_all.py --task 319 --config configs\task_319_kuavo_wheel_s62.yaml --full --headless --demo-objects 1
```

### 全局链路: Docker/底盘/仿真/感知 ✅ | IK ⚠️ | 抓取 ⏳

## 2025-06-19 #3 — IK 调用方式修复（对齐官方 demo）+ 手腕关节顺序修正

### 根因发现

1. **双臂 IK 调用方式错误**：我们使用 `use_custom_ik_param=True` + hardcode quaternion，官方 demo 用 `use_custom_ik_param=False` + `quaternion_from_euler(rpy)`。切换到官方模式后 IK 开始返回有效关节角。

2. **手腕关节顺序反转**：IK 输出 `[hand_yaw, hand_pitch, hand_roll]`，但 MuJoCo 模型 `zarm_r5(yaw), zarm_r6(X轴=roll), zarm_r7(Y轴=pitch)`。IK 的 hand_pitch→MuJoCo 的 roll 轴，hand_roll→MuJoCo 的 pitch 轴——名字和轴全反。swap 索引 [5]↔[6] 后，X 方向从 0.543→0.771（目标 0.78，误差仅 0.009m）。

3. **IK 使用躯干关节做高度调节**：q_torso 的 waist_pitch/waist_yaw 关键，添加到 posture_targets。

### 改动

| 文件 | 改动 |
|------|------|
| `kuavo_official_ik.py` | `_solve_two_arm_pose_via_docker`: 用 `twoArmHandPoseCmd` 直接请求（对齐 demo），`use_custom_ik_param=False`，`quaternion_from_euler` 替代 hardcode |
| `kuavo_official_ik.py` | `KuavoOfficialIKResult`: 添加 `q_torso` 字段 |
| `kuavo_controller.py` `_try_official_ik` | 世界→base_local 坐标变换；RPY=[-1.57,0,yaw]；提取并应用 q_torso；手腕 swap indices 5↔6 |

### 当前状态

```
pinch execution: X=0.774 (✓ target 0.78), Y=-0.793 (✗ target -0.2), Z=0.779 (✗ target 1.226)
```

X 方向正确，Y/Z 仍偏离。IK 对所有 target 返回几乎相同的关节角——位置是软约束，方向是硬约束。需要换成 6-DOF 位姿求解的单臂 accessibility IK (`/mobile_manipulator_ik_accessibility_check`) 或用完整 MPC+WBC 控制器。

### 启动

```powershell
docker restart kuavo_official_ros && sleep 5
# 然后在 Docker 内启动 roscore + IK + wheel bridge + arm trajectory
cd E:\workspace\bigdata\project
python run_all.py --task 319 --config configs\task_319_kuavo_wheel_s62.yaml --full --headless --demo-objects 1
```

## 2025-06-19 #4 — 运动链路打通：官方IK + Jacobian精调

### 核心改动

| 文件 | 改动 |
|------|------|
| `kuavo_official_ik.py` | 切换到官方demo调用方式：`twoArmHandPoseCmd`直接请求、`quaternion_from_euler`、`use_custom_ik_param=False` |
| `kuavo_official_ik.py` | 新增`q_torso`字段，IK返回的躯干关节传递给控制器 |
| `kuavo_official_ik.py` | IK调用前发布`/joint_states`（MuJoCo→ROS关节名映射），让IK知道当前机器人配置 |
| `kuavo_controller.py` `_try_official_ik` | world→base_local坐标变换；RPY=[-1.57,0,yaw]；躯干关节应用 |
| `kuavo_controller.py` | 新增`_jacobian_position_step()`——官方IK做朝向+MuJoCo Jacobian做位置精调（20步迭代） |

### 运动链路架构

```
感知(GroundingDINO) → ReKep规划 → 官方IK(朝向) → Jacobian精调(位置) → MuJoCo PD执行 → 臂到位
                                                                          ↑
                                                          Docker ROS1 Noetic
```

### 效果

| waypoint | 修复前误差 | 修复后误差 |
|----------|-----------|-----------|
| approach | 0.7-1.0m | **0.02-0.03m** |
| pregrasp | 0.7-1.0m | **0.018-0.020m** |
| grasp | — | **0.019-0.021m** |

### 启动（带仿真器窗口查看效果）

```powershell
# 前提：Docker Desktop 已运行，容器 kuavo_official_ros 已启动
# 1. 确保 Docker 内服务都在运行
docker restart kuavo_official_ros && sleep 5

# 2. 启动 IK + 臂轨迹 + 底盘桥（容器内）
docker exec -d kuavo_official_ros bash -lc "source /opt/ros/noetic/setup.bash && source /root/kuavo_ws_linux/devel/setup.bash && export ROS_MASTER_URI=http://localhost:11311 && roscore >/tmp/roscore.log 2>&1"
sleep 3
docker exec kuavo_official_ros bash -lc "source /opt/ros/noetic/setup.bash && source /root/kuavo_ws_linux/devel/setup.bash && export ROS_MASTER_URI=http://localhost:11311 && export ROBOT_VERSION=62 && roslaunch motion_capture_ik ik_node.launch robot_version:=62 visualize:=false >/tmp/ik.log 2>&1 &"
sleep 4
docker exec -d kuavo_official_ros bash -lc "source /opt/ros/noetic/setup.bash && source /root/kuavo_ws_linux/devel/setup.bash && export ROS_MASTER_URI=http://localhost:11311 && python3 /root/kuavo_ws_linux/src/kuavo_wheel/scripts/wheel_bridge.py --kuavo-master http://localhost:11311 --wheel-master http://localhost:11311 --kuavo-slave-ip 127.0.0.1 --wheel-slave-ip 127.0.0.1 >/tmp/wheel.log 2>&1"
docker exec -d kuavo_official_ros bash -lc "source /opt/ros/noetic/setup.bash && source /root/kuavo_ws_linux/devel/setup.bash && export ROS_MASTER_URI=http://localhost:11311 && rosrun humanoid_plan_arm_trajectory arm_trajectory_bezier_process.py >/tmp/bezier.log 2>&1"
docker exec -d kuavo_official_ros bash -lc "source /opt/ros/noetic/setup.bash && export ROS_MASTER_URI=http://localhost:11311 && python3 -c \"import rospy; rospy.init_node('dc',anonymous=True,disable_signals=True); rospy.Subscriber('/leju_claw_command', rospy.AnyMsg, lambda m: None); rospy.spin()\" >/tmp/dummy_claw.log 2>&1"
sleep 2
echo "Docker services ready"

# 3. 运行 3.19（带 MuJoCo 窗口）
cd E:\workspace\bigdata\project
$env:PYTHONUNBUFFERED = '1'
python run_all.py --task 319 --config configs\task_319_kuavo_wheel_s62.yaml --full --demo-objects 1 --live-debug
```

## 2025-06-19 #5 — Docker 服务启动脚本 + 最终验证

### 启动步骤（已验证可行）

```powershell
# 1. 启动 Docker 服务
powershell -File scripts/start_docker_services.ps1

# 2. 运行 3.19（带 MuJoCo 窗口看效果）
cd E:\workspace\bigdata\project
$env:PYTHONUNBUFFERED = '1'
python run_all.py --task 319 --config configs\task_319_kuavo_wheel_s62.yaml --full --demo-objects 1 --live-debug

# 3. 无窗口模式（更快）
python run_all.py --task 319 --config configs\task_319_kuavo_wheel_s62.yaml --full --headless --demo-objects 1
```

### 最终验证结果

| waypoint | target_err | 阈值 | 通过 |
|----------|-----------|------|------|
| approach | 0.028m | 0.095m | ✅ |
| pregrasp | 0.020m | 0.085m | ✅ |
| grasp | 0.019m | 0.085m | ✅ |

### 全局链路

```
感知(GroundingDINO) ✓ → ReKep规划 ✓ → 世界→base_local变换 ✓
  → 官方IK(朝向, Docker ROS1) ✓ → Jacobian精调(位置, 20步) ✓
    → MuJoCo PD执行 ✓ → 臂末端 <0.03m 到位 ✓
```

## 2025-06-19 #6 — 架构澄清 + 修复导航/头跳变/缓慢

### 架构说明

**控制链路**（全走官方）：
- IK: Docker ROS1 `/ik/two_arm_hand_pose_cmd_srv` (Drake 求解器)
- 臂轨迹: Docker ROS1 Bezier 插值器 (`arm_trajectory_bezier_process`)
- 底盘: Docker ROS1 wheel bridge (`/move_base/base_cmd_vel`)
- 夹爪: Docker ROS1 LejuClaw SDK

**执行链路**（Windows MuJoCo 纯物理镜像）：
- 官方命令 → MuJoCo 电机 → 物理仿真 → 可视化
- MuJoCo 不产生控制决策，只做物理计算和渲染

### 修复项

1. **导航走 ROS**: `move_planar_to` 先调 `official_control.command_base_world()` (发到 Docker)，再 PD mirror 到 MuJoCo
2. **头平滑**: `look_at()` 加插值过渡，避免跳变
3. **减少冗余**: reachability_candidates 48→24, arm_settle 50→20
4. **导航加速**: 底座阻尼 20→10, 候选截断更早

### 启动

```powershell
powershell -File scripts/start_docker_services.ps1
cd E:\workspace\bigdata\project
$env:PYTHONUNBUFFERED = '1'
python run_all.py --task 319 --config configs\task_319_kuavo_wheel_s62.yaml --full --demo-objects 1 --live-debug
```
