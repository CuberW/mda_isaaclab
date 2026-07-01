# Task 3.19 Delivery Checklist

本清单用于检查 `mda_isaaclab` 当前项目是否覆盖题目要求。

## 1. 源代码

| 模块 | 路径 | 说明 |
| --- | --- | --- |
| 主入口 | `task_319_garbage_sort/task319_grasp_sort_sm.py` | 任务统一启动入口 |
| IsaacLab 场景和状态机 | `task_319_garbage_sort/visual_grasp_record_demo.py` | 10 件垃圾、4 个桶、RGB-D、VISS、Nav2、抓取、投放 |
| 场景预览和 ROS2 bridge | `task_319_garbage_sort/phase1_scene_ros2_bridge.py` | 场景/ROS topic 基础验证 |
| 抓取 pipeline | `task_319_garbage_sort/grasp_pipeline/` | 类型、感知、抓取选择、执行工具 |
| cuRobo 右臂适配 | `task_319_garbage_sort/curobo_right_arm.py` | 右臂规划和 IK 适配 |
| Nav2/ROS 脚本 | `task_319_garbage_sort/scripts/ros2_*.py` | Nav2 stack、goal client、socket bridge、cmd_vel 测试 |
| Kuavo IK 脚本 | `task_319_garbage_sort/scripts/kuavo_*.py`, `task_319_garbage_sort/scripts/kuavo_analytic_ik_cli.cpp` | Kuavo 官方 IK bridge 和解析 IK 诊断 |
| 双指夹爪 URDF | `task_319_garbage_sort/two_finger_gripper.urdf` | 右手夹爪模型 |
| 局部抓取强化学习 | `task319_local_grasp_rl/` | 后续接入局部接近、闭合和抬升策略 |

## 2. 仿真工程文件

| 类型 | 路径 |
| --- | --- |
| Isaac/Usd 场景 | `isaac_projects/trashbot_scene.usd`, `isaac_projects/trashbot_scene_head_camera.usd`, `isaac_projects/trashbot_scene_real_models.usd` |
| 真实垃圾模型 | `isaac_projects/usd_trash_models/`, `isaac_projects/*.glb`, `isaac_projects/textures/` |
| 场景真值和恢复脚本 | `isaac_projects/v27_object_ground_truth_layout.json`, `isaac_projects/script/reset_rubbish.py` |
| 机器人/任务配置 | `robot/configs/task_319_kuavo_wheel.yaml`, `robot/configs/task_319_kuavo_wheel_s62.yaml`, `robot/configs/task_319_garbage_sort*.yaml` |
| 备用 robot 包 | `robot/task_319_garbage_sort/` |

当前正式 IsaacLab 主线直接在 `visual_grasp_record_demo.py` 中构造主场景；`robot/configs/task_319_garbage_sort.yaml` 是早期 Stretch/MuJoCo 风格配置，不是当前验收主入口。

## 3. 场景要求覆盖

| 题目要求 | 当前实现 |
| --- | --- |
| 桌面/地面至少 10 件垃圾 | `TRASH_SCENE_OBJECTS` 含 10 个桌面垃圾 |
| 包含 4 类 | 可回收物、厨余垃圾、有害垃圾、其他垃圾 |
| 四类垃圾桶颜色标识 | 可回收桶蓝色、厨余桶绿色、有害桶红色、其他桶灰色 |
| 识别类别、位置、抓取姿态 | VISS/Qwen 输出类别，RGB-D 输出三维位置，顶部抓取模块输出夹爪姿态 |
| 识别、抓取、移动至对应垃圾桶、投放 | `--mind_sort_demo` 状态机覆盖完整流程 |
| 统计准确率/成功率 | `mind_sort_task_queue.json`, `mind_sort_cycle.json`, `isaac_task_execution_result.json` |
| 干扰样本 | `trash_potted_meat_can_0` 食物残留金属罐、`trash_dirty_bottle` 历史场景脏瓶样本 |

主线 10 件垃圾：

| Key | 对象 | 类别 |
| --- | --- | --- |
| `trash_00` | cracker box | 可回收物 |
| `trash_01` | sugar box | 可回收物 |
| `trash_02` | tomato soup can | 可回收物 |
| `trash_03` | mustard bottle | 可回收物 |
| `trash_04` | banana | 厨余垃圾 |
| `trash_05` | potted meat can, food residue | 厨余/可回收干扰样本 |
| `trash_06` | bleach cleanser | 有害垃圾 |
| `trash_07` | battery block | 有害垃圾 |
| `trash_08` | foam brick | 其他垃圾 |
| `trash_09` | mug | 其他垃圾 |

## 4. 模型和权重

大型权重不纳入 Git 提交，按 `docs/task319_delivery/README.md` 的“模型下载位置”恢复到下列路径。

| 模型 | 路径 |
| --- | --- |
| 当前 YOLO11 分割 | `viss/models/yolo11s-seg-best.pt` |
| VISS 备用权重 | `viss/models/best_seg.pt`, `viss/models/yolo11s-seg.pt`, `viss/models/yoloe*.pt` |
| GraspNet Baseline | `models/graspnet-rs/checkpoint-rs.tar` |
| YOLOv8 备用 | `models/yolo/yolov8m-seg.pt`, `models/yolov8n.pt`, `models/yolov8s-worldv2.pt` |
| SAM/CLIP/DINO/DA3 | `models/sam-vit-b/`, `models/clip-vit-b32/`, `models/dinov2-base/`, `models/da3-small/`, `weights/clip/ViT-B-32.pt` |

## 5. 文档和答辩材料

| 材料 | 路径 |
| --- | --- |
| 部署 README | `docs/task319_delivery/README.md` |
| 课程报告 | `docs/task319_delivery/COURSE_REPORT.md` |
| 调试报告 | `docs/task319_delivery/DEBUG_REPORT.md` |
| PPT 文稿 | `docs/task319_delivery/PPT_OUTLINE.md` |
| PPT 文件 | `docs/task319_delivery/TASK319_DEFENSE_PPT.pptx` |
| 视频证据说明 | `docs/task319_delivery/VIDEO_EVIDENCE.md` |
| 开发过程报告 | `docs/task319_delivery/references/TASK319_DEVELOPMENT_REPORT.md` |
| 算法理论说明 | `docs/task319_delivery/references/task319_algorithm_theory.md` |
| 状态机讲解 | `docs/task319_delivery/references/task319_visual_grasp_state_machine_explainer.md` |
| 命令手册 | `docs/task319_delivery/references/DEMO_COMMANDS.md` |
| 导航实现记录 | `docs/task319_delivery/references/NAV2_MOTION_IMPLEMENTATION.md` |
| 稳定抓取 baseline | `docs/task319_delivery/references/STABLE_GRASP_BASELINE_20260624.md` |
| 参考论文 | `docs/task319_delivery/references/A_Deep-Intelligence_Framework_for_Online_Video_Processing.pdf` |

## 6. 视频和结果文件

| 用途 | 路径 |
| --- | --- |
| GitHub 提交版完整演示视频 | `docs/task319_delivery/videos/task319_system_demo_480p.mp4` |
| 原始完整多物体演示视频 | `task_319_garbage_sort/output/head_camera_grasp_records/20260625_150100/grasp8.mp4` |
| 完整多物体结果 | `task_319_garbage_sort/output/head_camera_grasp_records/20260625_150100/mind_sort_demo/mind_sort_task_queue.json` |
| 完整多物体压缩证据 | `task_319_garbage_sort/output/head_camera_grasp_records/20260625_150100/mind_sort_demo.zip` |
| 最新严格物理单轮视频 | `task_319_garbage_sort/output/head_camera_grasp_records/20260626_145846/external_grasp_demo.mp4` |
| 最新严格物理单轮结果 | `task_319_garbage_sort/output/head_camera_grasp_records/20260626_145846/mind_sort_demo/mind_sort_task_queue.json` |
| 早期逻辑动画指标 | `isaac_projects/isaac_task_execution_result.json` |

## 7. 当前结论

- 系统级完整链路已经具备：识别、分类、动态站位、Nav2、到桶口释放。
- 完整多物体演示完成 8 个成功物体处理轮次，投放阶段不使用物体瞬移。
- 严格物理夹爪仍是主要风险项，最新单轮严格验证未通过。
- 课程报告采用逻辑动画、完整主线演示、严格物理单轮三组代表性实验记录；同时说明开发期间另有约 10-20 倍历史调试测试，用作工程覆盖度和稳定性优化说明。
- 交付材料中已将系统级辅助演示和严格物理验收分开统计。
