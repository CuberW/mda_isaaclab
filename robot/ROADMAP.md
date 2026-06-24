# Robot System Roadmap

更新时间：2026-06-17

## 当前范围

当前仓库只保留两个有效作业链路：

- **3.19 Stretch 垃圾分类投放**
- **2.2 双臂协同搬运**

Task 3.7 / Kuavo 开放词汇抓取已经单独弃用，并从运行入口、任务包、配置、Kuavo 场景资产、MoveIt2 配置和健康检查中移除。

## 当前结构

```text
project/
├── run_all.py                       # 统一入口：--task 319 / 22 / all
├── task_319_garbage_sort/           # Stretch 垃圾分类投放
├── task_22_dual_arm/                # 双臂协同搬运
├── robot_common/                    # 共享基础设施
│   ├── env/                         # MuJoCo 环境、相机和坐标转换
│   ├── perception/                  # 检测、分割、分类、点云/抓取输入
│   ├── decision/                    # 任务解析、DAG、状态机
│   ├── execution/                   # Mink IK、GraspManager、GraspNet、WBC、导航
│   └── infra/                       # 配置、日志、指标、trace、debug artifacts
├── perception/                      # 感知层公共 facade
├── planning/                        # 规划层公共 facade
├── control/                         # 控制层公共 facade
├── configs/                         # 当前任务 YAML 配置
├── simulation/                      # MuJoCo 场景和机器人模型
├── scripts/                         # 验证、审计、资产装配和运行辅助脚本
├── models/                          # 模型权重，gitignored
└── third_party/                     # 第三方源码，gitignored
```

## 已验证能力

- 3.19：垃圾检测/分类、GraspNet 抓取位姿、Stretch 差速导航、真实夹爪接触后 attach、投放验证。
- 2.2：DAG 分解、双臂同步抓取、WBC torque 控制、真实 finger/pinch contact、同步误差和倾角 trace。
- 公共能力：MuJoCo 渲染、RGB-D 相机、坐标转换、Mink IK、MotionTrace、静态禁止 teleport 检查。

## 运行命令

```powershell
python scripts\verify_imports.py
python scripts\verify_system_health.py --light
python scripts\verify_system_health.py --full

python run_all.py --task 319 --headless
python run_all.py --task 22 --headless --instruction "用双手把长杆放到指定区域"
python run_all.py --task all --headless
```

## 后续优化方向

1. 改进 3.19 和 2.2 的轨迹平滑：minimum jerk、速度/加速度/jerk 限幅、零空间姿态偏好。
2. 加强真实夹爪接触模型：finger pad 摩擦、夹爪闭合力、物体滑移检测。
3. 完善 Dobot Magician 真机 adapter：目标位姿输出、Dobot IK、串口/SDK 控制、安全限位。
4. 将 debug artifacts 标准化：每个 episode 输出识别图、抓取位姿、坐标变换、轨迹和接触日志。
