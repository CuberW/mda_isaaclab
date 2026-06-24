# 项目状态总结

更新时间：2026-06-17

## 当前结论

- Task 3.7 / Kuavo 开放词汇抓取已经弃用并从仓库移除。
- 当前有效任务为 **3.19 Stretch 垃圾分类投放** 和 **2.2 双臂协同搬运**。
- 运行入口统一为 `run_all.py --task 319|22|all`。

## 已保留能力

- `MuJoCoEnv`、`PerceptionHub`、Mink IK、`GraspManager`、`MotionTrace`、debug artifacts。
- 3.19 的 GraspNet 抓取、Stretch 连续导航和真实夹爪接触链路。
- 2.2 的 DAG 分解、双臂 WBC torque 控制、真实 finger/pinch 接触和 trace 指标。

## 已移除内容

- `task_37_open_vocab/`
- `configs/task_37_open_vocab.yaml`
- Kuavo 标定、场景、机器人资产
- Kuavo stance 控制器
- Kuavo MoveIt2 配置包和 bridge 脚本
- `run_all.py --task 37` 入口
- `verify_system_health.py` 中的 3.7 / MoveIt2 验收项

## 当前验证命令

```powershell
python scripts\verify_imports.py
python scripts\verify_system_health.py --light
python scripts\verify_system_health.py --full
python run_all.py --task 319 --headless
python run_all.py --task 22 --headless --instruction "用双手把长杆放到指定区域"
```
