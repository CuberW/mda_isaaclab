# 机器人任务统一系统

面向作业的当前有效链路：

- **3.19** Stretch 轮式机器人垃圾分类投放
- **2.2** 双臂协同搬运

Task 3.7 / Kuavo 开放词汇抓取已经弃用并从仓库移除。系统骨架统一到 `MuJoCoEnv`、当前任务 pipeline、`PerceptionHub`、Mink IK、`GraspManager`、`MotionTrace` 和 `run_all.py`。当前原则是：light 模式用于本机调试；full 模式必须使用成熟后端，不用 scene-truth 或自写启发式冒充作业链路。

## 快速检查

```powershell
python scripts\verify_imports.py
python scripts\verify_system_health.py --light
python scripts\verify_system_health.py --full
```

`--light` 检查当前 Windows conda 可运行的仿真骨架、相机、Mink 不污染现场 qpos、trace 和禁止瞬移写法。

`--full` 额外要求 GraspNet CUDA 推理和 robosuite WBC；这是作业完整链路验收入口。

## 资产和后端

```powershell
python scripts\setup_full_stack_assets.py
```

该脚本会准备：

- `third_party/graspnetAPI`
- `third_party/graspnet-baseline`
- `models/graspnet-rs/checkpoint-rs.tar`
- 已有 GroundingDINO / SAM / CLIP 权重检查

完整后端建议在 WSL2/Docker 中隔离，不污染当前 conda：

```powershell
# WSL2 Ubuntu 24.04 + GraspNet CUDA ops + robosuite
powershell -ExecutionPolicy Bypass -File scripts\robot_stack.ps1 wsl-setup

# Docker 方案仍保留，取决于 Docker Hub / CUDA 镜像网络
powershell -ExecutionPolicy Bypass -File scripts\robot_stack.ps1 build
```

## 运行任务

完整作业链路应在 full 后端通过后运行：

```powershell
python run_all.py --task 319 --headless
python run_all.py --task 22 --headless --instruction "用双手把长杆放到指定区域"
```

Viewer 验收：

```powershell
python run_all.py --task 319
python run_all.py --task 22 --instruction "用双手把长杆放到指定区域"
```

WSL full-stack 入口：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\robot_stack.ps1 wsl-health
powershell -ExecutionPolicy Bypass -File scripts\robot_stack.ps1 wsl-headless-all
powershell -ExecutionPolicy Bypass -File scripts\robot_stack.ps1 wsl-viewer 319
```

## 模块结构

```text
robot_common/
  infra/        配置、日志、指标、MotionTrace
  env/          MuJoCo 环境、相机、RGB-D、坐标变换
  perception/   YOLO、GroundingDINO、SAM、CLIP
  decision/     状态机、任务路由、DAG/VLA 策略边界
  execution/    Mink IK、GraspManager、GraspNet、robosuite WBC

task_319_garbage_sort/   Stretch 垃圾分类投放
task_22_dual_arm/        双臂协同搬运

configs/                 当前任务配置
simulation/              MuJoCo 场景和机器人模型
models/                  模型权重，gitignored
third_party/             成熟后端源码，gitignored
```

## 当前 full 链路要求

- 3.19：YOLO/CLIP 感知分类，GraspNet RGB-D 抓取位姿，Stretch 连续差速导航，接触后 weld attach。
- 2.2：DAG 分解，双臂同步抓取和搬运，robosuite WBC 成熟执行路径，trace 检查同步误差和倾角。

指标文件写入 `results/traces/`，health 和任务成功不能覆盖 trace 运动质量失败。
