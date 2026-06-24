# Kuavo 官方控制栈接入说明

3.19 Kuavo 分支的 `--full` 模式只接受官方 Kuavo 控制链路：

```text
感知目标位姿 -> ReKep hover/grasp/lift 约束
-> Kuavo 官方 IK service
-> Kuavo 官方 SDK 轨迹/底盘/LejuClaw 控制
-> MuJoCo/Isaac 仅做可视化与接触验证镜像
```

本地 DLS IK、MuJoCo PD、手写 base servo 只允许在 light/debug 模式中做离线调试，不能算 full 成功。

## 需要启动的官方组件

- Python 包：`kuavo_humanoid_sdk`
- ROS1 Python：`rospy`
- Kuavo 消息/服务包：`kuavo_msgs`
- IK 服务：`/mobile_manipulator_ik_accessibility_check`
- IK 节点：官方 SDK 提示为 `roslaunch motion_capture_ik ik_node.launch`
- 控制接口：`KuavoRobot.control_arm_joint_trajectory`
- 底盘接口：`KuavoRobot.control_command_pose_world`
- 夹爪接口：`LejuClaw`

当前项目默认会尝试引用外部官方工作区：

```text
E:\workspace\kuavo-ros-opensource
```

不建议把整个官方仓库复制进当前项目；它应该作为外部依赖存在。换机器或换路径时设置：

```powershell
$env:KUAVO_ROS_WS="你的 kuavo-ros-opensource 工作区路径"
```

当前 Windows conda 可以做静态探测和 MuJoCo light 调试，但不能直接提供 ROS1 的 `rospy` 和生成后的 `kuavo_msgs`。真正 full 模式应在官方 ROS1/WSL/Linux shell 中完成：

```bash
source devel/setup.bash
# 或者，如果使用官方 installed 空间：
source installed/setup.bash
roslaunch motion_capture_ik ik_node.launch
```

先用快速探测命令确认官方路径和依赖：

```powershell
python scripts\probe_kuavo_official_stack.py --workspace E:\workspace\kuavo-ros-opensource
```

如果看到 `No module named 'rospy'` 或 `kuavo_msgs` 不可用，说明当前 shell 不是 ROS1 运行环境，或者 `kuavo_msgs` 尚未编译生成 Python 包。

## 验证命令

```powershell
python scripts\verify_system_health.py --light
python scripts\verify_system_health.py --full
python run_all.py --task 319 --config configs\task_319_kuavo_wheel.yaml --headless --full --demo-objects 1
```

如果 `--full` 报缺 `rospy`、`kuavo_humanoid_sdk` 或 IK service 未 ready，这是正确行为：系统不会再用本地 fallback 冒充官方控制。
