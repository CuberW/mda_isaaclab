# WSL2 + ROS2 + Kuavo 官方 ROS 栈接入执行说明

本文档用于当前电脑或另一个 Agent 逐条执行。要求：贴真实命令输出，不要编造，不要用本地 DLS/MuJoCo PD 冒充官方控制。

## 当前事实

- Windows 项目仓库：`E:\workspace\bigdata\project`
- Windows Kuavo 官方仓库：`E:\workspace\kuavo-ros-opensource`
- WSL 发行版：`UbuntuRobot`
- WSL 系统：Ubuntu 24.04 Noble，ROS2 Jazzy
- Kuavo 官方控制栈：ROS1 Noetic，不适合直接在 Ubuntu 24.04 原生构建
- 正式可用方案：Kuavo 官方 ROS1 节点在 Docker 容器 `kuavo_official_ros` 内运行
- Docker 容器内官方工作区：`/root/kuavo_ws_linux`
- WSL 原生 Kuavo 工作区：`/root/workspace/kuavo-ros-opensource`

## 1. 验证 WSL 和 ROS2

在 Windows PowerShell 执行：

```powershell
wsl --version
wsl -l -v
wsl -d UbuntuRobot -- bash -lc "lsb_release -a; source /opt/ros/jazzy/setup.bash; which ros2; ros2 node list || true; printenv ROS_DOMAIN_ID || true"
```

注意：Jazzy 下 `ros2 --version` 可能返回 `unrecognized arguments: --version`，这不代表 ROS2 不存在；以 `which ros2` 和 `ros2 node list` 为准。

## 2. 把 Kuavo 官方仓库同步到 WSL

```powershell
python scripts\sync_kuavo_ros_to_wsl.py --distro UbuntuRobot --source E:\workspace\kuavo-ros-opensource --target /root/workspace/kuavo-ros-opensource --delete
```

成功时必须看到：

```text
KUAVO_WSL_COPY_OK
WSL path: /root/workspace/kuavo-ros-opensource
Windows UNC path: \\wsl.localhost\UbuntuRobot\root\workspace\kuavo-ros-opensource
```

如果输出 `DOCKER_NOT_READY`，说明 Docker Desktop 没有对 `UbuntuRobot` 开 WSL integration。打开：

```text
Docker Desktop -> Settings -> Resources -> WSL integration -> UbuntuRobot
```

然后重启 Docker Desktop 或执行：

```powershell
wsl --shutdown
```

## 3. 准备 Kuavo 官方 ROS1 Docker 工作区

当前更稳的执行方式是在 Windows PowerShell 里启动 Docker。首次或刷新时执行：

```powershell
python scripts\setup_kuavo_ros1_docker.py --workspace E:\workspace\kuavo-ros-opensource --robot-version 62 --refresh-copy
```

如果 Docker Desktop 已经开放 WSL integration，也可以在 WSL 内用 WSL 原生仓库执行：

```bash
cd /mnt/e/workspace/bigdata/project
python3 scripts/setup_kuavo_ros1_docker.py --workspace /root/workspace/kuavo-ros-opensource --robot-version 62 --refresh-copy
```

## 4. 启动官方 Kuavo ROS1 服务

```powershell
python scripts\start_kuavo_ros1_services.py --container kuavo_official_ros --docker-workspace /root/kuavo_ws_linux --robot-version 62
```

成功时必须看到：

```text
[OK] ROS1 master
[OK] official IK service /ik/two_arm_hand_pose_cmd_srv
[OK] official arm trajectory interface
[OK] official wheel bridge command topic /move_base/base_cmd_vel
```

ROS graph 中应包含：

```text
/ik/two_arm_hand_pose_cmd_srv
/ik/fk_srv
/bezier/arm_traj
/kuavo_arm_traj
/move_base/base_cmd_vel
/base_cmd_vel
```

## 5. Probe 官方控制栈

```powershell
python scripts\probe_kuavo_official_stack.py --use-docker --docker-container kuavo_official_ros --docker-workspace /root/kuavo_ws_linux --ik-service /ik/two_arm_hand_pose_cmd_srv
```

通过标准：

```text
Official IK:
  available: True

Official SDK control:
  ready: True
  sdk_import: True
  ros_ready: True
  arm_trajectory_ready: True
  base_control_ready: True
  claw_ready: True
```

## 6. 运行 3.19 Kuavo full

无 viewer：

```powershell
$env:PYTHONUNBUFFERED='1'
python run_all.py --task 319 --config configs\task_319_kuavo_wheel.yaml --headless --full --demo-objects 1
```

带 viewer / live debug：

```powershell
$env:PYTHONUNBUFFERED='1'
python run_all.py --task 319 --config configs\task_319_kuavo_wheel.yaml --full --demo-objects 1 --live-debug
```

## 7. 3.19 状态机必须这样跑

1. `INIT_AND_PATROL`：检查 perception、GraspNet、Kuavo official IK/control graph。
2. `DETECT_AND_CLASSIFY`：只接收现有视觉链路输出，不重写视觉。
3. `REACHABILITY_AND_WHOLE_BODY_PLAN`：采样可达底盘位姿，调用官方 IK 验证 pregrasp/grasp/lift。
4. `COORDINATED_NAVIGATION`：底盘命令必须来自规划结果，不允许硬编码场景路线。
5. `GRASP_CARRY_RELEASE`：真实 finger contact 后 attach，保持抓取移动到对应垃圾桶，慢开夹爪自然释放。
6. `ERROR_RECOVERY`：失败就重新观测/重新规划；`--full` 不允许 fallback 成功。

## 8. 严格禁止

- 禁止直接写物体 pose 伪造投放。
- 禁止用大球/大距离阈值冒充夹取。
- 禁止在 `--full` 里把本地 DLS、MuJoCo PD、固定路线当官方控制成功。
- 禁止修改视觉链路来掩盖控制失败。
