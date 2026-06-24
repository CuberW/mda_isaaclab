# 基于深度方法 + VLM 的垃圾分类抓取方案

日期：2026-06-22

## 1. 可行性判断

这版方案整体可行，并且比当前 YOLO 主导链路更适合当前问题：

- 同色物体或弱纹理物体容易让 YOLO/SAM 产生错误分割，深度连通域更适合做物理物体分离。
- YOLO mask 边缘锯齿会导致 3D 质心漂移，抓取点应改为由 RGB-D 点云几何中心计算。
- VLM 不应参与几何定位，只负责语义分类；分类结果绑定到已选中的同一个深度目标。
- VLM crop 应使用膨胀后的原始 RGB ROI，保留少量桌面上下文，不应使用纯抠图或黑底图。
- VLM 延迟可以通过“夹取前拍照 + 异步分类 + 移动中等待”隐藏。

仓库里已有可复用基础：

- 头部 RGB-D 相机链路：`head_rgbd` 的 RGB、depth、intrinsics、camera_to_world 已接入。
- 深度组件方法：已有 `detect_tabletop_depth_components()`，目前作为 fallback/debug，可升级为主感知。
- GLM-VLM 四分类：已有 `GlmVlmClassifier`，prompt 已限制输出四类垃圾。
- 右臂 IK 可达性检查：已有 dry-run IK，可避免手写距离阈值。
- 抓取状态机：已有 pregrasp、grasp、lift、hold、retry 基础。
- 垃圾桶坐标和导航基础函数：已有分类到桶位的 y 坐标映射，以及 `drive_to_pose()` 等函数。

当前主要缺口：

- `depth_component` 现在不是主链路，且 debug 语义会借用仿真真值，不能作为真实分类结果。
- 当前 VLM crop 会把 mask 外背景涂黑，和新方案要求冲突。
- 当前分类在抓取前同步完成，还没有接成“抓取执行中异步 VLM”。
- `--enable_sort_nav` 的投桶导航函数存在，但抓取成功后的完整投放闭环尚未真正接入。
- 当前深度组件还没有 PCA 主方向估计，只能先用质心 + 宽度估计。


## 1A. v28 视觉链路修订：Qwen 全局候选 + YOLO ROI 精分割 + 抓取并行验证

当前正式视觉入口改为 `--perception_source v28_original`。它不是使用 v28 自带相机参数或站位，而是在 Task319 主场景中先用当前头部 RGB-D 相机拍照，然后把 RGB 帧交给 `viss/scripts/perception/yolo11_qwen_perception_offline.py` 的 v28 `qwen_first` 逻辑。

设计链条：

1. Qwen 对整张 RGB 图做全局识别，返回最多 `--viss_qwen_max_candidates` 个高可信度垃圾候选，候选包括语义名称、四分类类别、粗 bbox 和置信度。
2. YOLO 使用 `viss/models/best_seg.pt` 只在每个 Qwen ROI 内做精确分割，避免全图 YOLO 被背景或同色桌面干扰。
3. 主链路只采用 YOLO ROI mask 的像素范围，从当前 Task319 depth、intrinsics、`camera_to_world` 计算 RGB-D 点云几何中心；不使用 v28 的相机外参或仿真真值。
4. 类别以 Qwen/VLM 输出为准，YOLO raw class 只作为 debug 字段。
5. 目标选择仍然先做当前右臂可达性、点云质量、距离和失败记忆排序，保证视觉识别、定位、分类和抓取目标是同一个实例。

下一步状态机改造：

- 现在的 `v28_original` 调用仍是同步离线感知：Qwen coarse、YOLO refine、Qwen verify 全部结束后才返回实例列表。
- 为了实现“臂爪开始运动，同时 Qwen 验证分割后的结果”，需要把 v28 拆成两个阶段：第一阶段返回 coarse category + ROI mask 并立即开始抓取；第二阶段把 crop verify 放入后台 future。
- 抓取抬升完成后再读取 verify future：若验证返回有效四分类，则覆盖 coarse category 并导航到对应垃圾桶；若超时或 unknown，则保留 coarse category，仍不可用时默认 `其他垃圾`。
- 该并行验证只影响投桶分类，不允许在抓取执行中切换目标实例。

## 2. 目标链路

目标链路改为 `depth_vlm`：

1. 头部 RGB-D 拍摄一帧。
2. 深度方法分割桌面上独立物体，输出每个物体的 mask、bbox、点云、3D 几何中心和尺寸信息。
3. 对每个深度目标做右臂 dry-run IK 可达性检查。
4. 只在 IK 可达目标里选择一个最容易抓的物体。
5. 对该目标的 bbox 膨胀 50px，裁剪原始 RGB ROI，启动异步 GLM-VLM 四分类。
6. 机械臂立即使用该目标的 RGB-D 几何中心执行抓取和抬升。
7. 抓取成功后，底盘移动到垃圾桶前的中间点。
8. 读取 VLM 分类结果，超时则默认 `其他垃圾`。
9. 移动到对应垃圾桶，释放物体。
10. 返回桌前初始位置，重新拍照刷新待抓取列表。

## 3. 关键实现变更

### 3.1 感知模式

新增参数：

```text
--perception_mode {depth_vlm,yolo_glm}
```

默认使用：

```text
--perception_mode depth_vlm
```

行为要求：

- `depth_vlm` 下，目标选择只使用深度组件。
- YOLO 只保留为 debug/对照，不参与目标选择、抓取点计算或分类决策。
- `yolo_glm` 保留为旧链路回归测试用。

### 3.2 深度组件升级为主感知

将 `detect_tabletop_depth_components()` 从 fallback/debug 升级为主感知入口。

每个深度目标至少记录：

- `object_id`
- `source="depth_component"`
- `mask`
- `bbox_xyxy`
- `points_world`
- `center_3d`
- `xy_extent_m`
- `z_extent_m`
- `point_count`
- `mask_hash`

可选增强：

- 使用 PCA 估计桌面平面内长轴方向。
- 根据点云短轴估计夹爪开口宽度。
- 对扁平物体，不寻找最高点，直接使用点云质心或可达探测点。

重要约束：

- 不再用仿真真值生成 VLM 分类。
- 仿真真值只允许写入 validation/debug 字段，用于对比相机估计误差。

### 3.3 目标选择

目标选择规则：

1. 过滤点数过少、bbox 过大、触边严重、桌面残留等明显无效组件。
2. 对剩余组件计算右臂 dry-run IK 可达性。
3. 只允许选择 `reachable=True` 的组件。
4. 排序优先级：
   - IK TCP 误差更小；
   - 当前 TCP 或机器人右臂更容易接近；
   - mask/点云面积合理；
   - 目标不过大、不像桌面背景。

禁止事项：

- 不用 YOLO 或 GLM 判断物体是否可达。
- 不使用手写距离阈值作为最终可达性判断。
- 不选择当前右臂抓不到的远处物体。

### 3.4 VLM 分类

VLM 只做四分类：

```text
可回收物 / 厨余垃圾 / 其他垃圾 / 有害垃圾
```

crop 规则：

- 从深度组件 bbox 或 mask 外接矩形得到 ROI。
- bbox 向四周膨胀 `50px`，边界裁剪到图像内。
- 使用原始 RGB ROI，不涂黑背景，不做纯抠图。
- resize 到 `224x224` 后送入 VLM。

异步规则：

- 选中目标后立即启动 VLM future。
- 机械臂抓取、抬升时 VLM 后台执行。
- 分类结果绑定到该目标的 `object_id` 和 `mask_hash`。
- 到达中间点后读取结果，最多等待 `1s`。
- 超时或错误时，分类设为 `其他垃圾`，metadata 记录超时或错误原因。

### 3.5 抓取点计算

默认抓取点：

```text
grasp_point = depth_component.center_3d
```

执行方式：

- v1 使用几何中心 fallback 抓取，不依赖 GraspNet/AnyGrasp。
- 默认 position-only IK，不强制目标坐标轴方向完全对齐。
- 保留安全 pregrasp：先将夹爪移动到高于桌面和目标的安全点，再接近目标。
- 夹爪宽度根据点云 XY 尺寸估计，并做上下限 clamp。

后续可选：

- 对长条物体，沿 PCA 长轴从中心缩进 1/3 作为抓取点。
- 对易被碰动目标，在 pregrasp 后快速重算深度中心，但不重新分类。

### 3.6 投桶导航

分类到垃圾桶映射沿用现有坐标：

```text
可回收物 -> y = -0.72
厨余垃圾 -> y = -0.24
有害垃圾 -> y =  0.24
其他垃圾 -> y =  0.72
```

执行顺序：

1. 抓取成功并完成抬升。
2. 保持夹爪闭合。
3. 底盘移动到垃圾桶前中间点或侧边通道中间点。
4. 读取 VLM 分类结果。
5. 移动到对应桶位。
6. 手臂执行 drop pose。
7. 张开夹爪释放。
8. 返回桌前初始位。
9. 下一轮重新拍摄 RGB-D，刷新物体列表。

## 4. 调试输出要求

每个 cycle 至少输出：

- `head_rgb.png`
- depth 可视化图
- `depth_component_overlay.png`
- `depth_components.json`
- `vlm_crop_*.jpg`
- `vlm_results.json`
- `target_pose_debug.jpg`
- `target_pose_debug.json`
- `right_arm_reachability.json`
- `metadata.json`
- `external_grasp_demo.mp4`，执行仿真验证时默认录制

`metadata.json` 必须记录：

```json
{
  "target_source_policy": "depth_vlm_primary",
  "target_geometry_policy": "depth_component_rgbd_geometric_center",
  "category_source_policy": "glm_async_on_depth_roi",
  "target_source": "depth_component",
  "grasp_point_source": "depth_component_geometric_center",
  "category_source": "glm",
  "vlm_async": true
}
```

## 5. 测试计划

### 5.1 静态检查

```bash
cd /home/zhxm/workspace/mda_isaaclab
/home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m py_compile \
  task_319_garbage_sort/visual_grasp_record_demo.py \
  task_319_garbage_sort/task319_grasp_sort_sm.py
```

### 5.2 感知验证

运行 1 个 cycle，不执行抓取：

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 240s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --num_cycles 1 \
  --perception_mode depth_vlm \
  --record_debug \
  --record_video \
  --warmup_steps 80 \
  --cycle_interval_steps 1
```

验收点：

- `depth_components.json` 有合理组件。
- 选中目标来自 `depth_component`。
- YOLO 不参与目标选择。
- `vlm_crop_*.jpg` 是带背景的膨胀原图 ROI。
- `target_pose_debug.jpg` 中相机估计中心与仿真物体位置偏差可接受。

### 5.3 抓取验证

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 420s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --num_cycles 1 \
  --execute_grasp \
  --perception_mode depth_vlm \
  --skip_graspnet \
  --use_centroid_fallback \
  --record_debug \
  --record_video \
  --warmup_steps 80 \
  --cycle_interval_steps 1 \
  --trajectory_steps 420 \
  --grasp_steps 130 \
  --lift_steps 240 \
  --hold_steps 180 \
  --max_joint_step 0.025 \
  --fallback_position_only_ik \
  --target_reachability_position_only
```

验收点：

- 自动选择右臂 IK 可达范围内的深度目标。
- 夹爪初始运动不碰动物体。
- 抓取点为深度组件几何中心。
- 抓取后目标被抬升并保持。
- 视频可观察完整过程。

### 5.4 分类投桶验证

```bash
cd /home/zhxm/workspace/mda_isaaclab
timeout 700s /home/zhxm/miniconda3/envs/my_task319_safe/bin/python \
  task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --num_cycles 1 \
  --execute_grasp \
  --enable_sort_nav \
  --perception_mode depth_vlm \
  --skip_graspnet \
  --use_centroid_fallback \
  --record_debug \
  --record_video \
  --vlm_model glm-5v-turbo \
  --vlm_timeout_s 1.5 \
  --vlm_retries 0 \
  --warmup_steps 80 \
  --cycle_interval_steps 1
```

验收点：

- VLM 分类在抓取/移动期间异步执行。
- 到中间点后读取分类结果。
- 超时时默认 `其他垃圾`，并写入 metadata。
- 底盘移动到对应垃圾桶。
- 释放物体后返回初始位置。

## 6. 默认假设

- 本方案优先级高于之前的强制 YOLO/GLM 目标选择链路。
- v1 使用现有 GLM-5V，不新增本地 CLIP/Qwen-VL 模型。
- v1 抓取采用深度几何中心，不启用 GraspNet/AnyGrasp。
- Isaac 中 RGB-D 已对齐，但仍必须保留投影与外参 debug。
- 曝光/白平衡锁定属于真实硬件部署项，Isaac 仿真中先记录为约束。
- 第一阶段不承诺 `<6s` 和 `>95%`，先验收“同一深度目标抓取成功 + VLM 分类绑定 + 正确投桶链路可运行”。
