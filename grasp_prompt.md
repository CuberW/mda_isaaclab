# IsaacLab 视觉抓取 Pipeline — 详细实现提示词

## 任务目标

在 IsaacLab 场景中，实现一个**轮臂机器人**（移动底盘 + 机械臂，头部挂载 RGB-D 相机）的视觉引导抓取系统。场景中有 **10 个物体**，分属 **4 个类别**（例如可回收物、厨余垃圾、有害垃圾、其他垃圾）。要求机器人能够：

1. 通过 RGB-D 相机感知场景
2. 识别每个物体的**精确轮廓（mask）**、**类别名称**、**垃圾归属类别**
3. 为指定目标物体生成**精确抓取位姿**
4. 执行抓取动作，将物体抓起

导航到垃圾桶暂不实现，**抓起来就算成功**。

---

## 一、前置条件：坐标系统与标定

### 1.1 坐标系定义

```
world_frame:         世界参考系（场景原点，所有物体的父坐标系）
robot_base_frame:    机器人底盘坐标系（移动底座中心）
arm_base_frame:      机械臂基座坐标系（固定在底盘上）
ee_frame:            末端执行器坐标系（夹爪中心，TCP）
camera_frame:        相机光心坐标系（RGB-D 相机，固定在机械臂头部/腕部）
camera_color_frame:  RGB 图像坐标系（像素 u, v）
camera_depth_frame:  深度图像坐标系（通常与 color 对齐）
```

### 1.2 必须完成的标定（实现时硬编码或从 USD 读取）

```
1. camera → ee (手眼标定):
   - 变换矩阵 T_cam_to_ee: 4×4 rigid transform
   - 从 Isaac 场景中直接读取相机 prim 到末端连杆的 relative pose

2. ee → arm_base (正运动学):
   - 由机械臂关节角实时计算 forward kinematics

3. arm_base → robot_base:
   - 固定安装偏移，从场景中读取

4. robot_base → world:
   - 底盘里程计或 ground truth pose（Isaac 场景中直接获取）

5. depth_scale:
   - 深度图像像素值 → 实际距离（米）的转换系数
```

### 1.3 关键验证

在实现感知和控制之前，先写一个**标定验证脚本**：

- 在场景中放置一个颜色醒目的球体（已知 world 坐标）
- 用相机拍摄，通过像素坐标 + 深度值反算 world 坐标
- 对比误差，确保重投影误差 < 1cm

---

## 二、感知层：YOLOv8-seg 实例分割

### 2.1 模型选择

```python
from ultralytics import YOLO
model = YOLO("yolov8m-seg.pt")  # m 版本平衡精度与速度，场景复杂可用 l-seg 或 x-seg
```

如果 10 个物体中有 COCO 80 类之外的自定义物体（如针筒不属于 person/bottle/cup 等），需要做两件事之一：

- **方案 A（推荐）**：用 Grounding DINO / SAM 2 做 open-vocabulary 分割，不需要训练
- **方案 B**：标注数据 → 微调 YOLOv8-seg 支持自定义类别

### 2.2 推理接口

```python
def perceive(rgb_image: np.ndarray) -> list[dict]:
    """
    输入: RGB 图像 [H, W, 3] (uint8, BGR 或 RGB)
    输出: 物体列表，每个物体包含:
        {
            "id": int,                  # 物体编号
            "class_name": str,          # "bottle", "banana", ...
            "conf": float,              # 置信度 [0,1]
            "bbox": [x1,y1,x2,y2],      # 检测框 (像素坐标)
            "mask": np.ndarray,         # 二值 mask [H, W], dtype=bool
            "contour": np.ndarray,      # 轮廓顶点 [N, 2], 像素坐标
            "center_2d": (cx, cy),      # mask 重心 (像素坐标)
        }
    """
    results = model(rgb_image)
    objects = []
    for r in results:
        if r.masks is None:
            continue
        for i, (box, mask_tensor) in enumerate(zip(r.boxes, r.masks.data)):
            cls_id = int(box.cls.item())
            mask_np = mask_tensor.cpu().numpy().astype(bool)
            # 提取轮廓: 找最大连通域的外轮廓
            contours, _ = cv2.findContours(
                mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            main_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(main_contour)
            cx = M["m10"] / M["m00"] if M["m00"] > 0 else (box.xyxy[0] + box.xyxy[2]) / 2
            cy = M["m01"] / M["m00"] if M["m00"] > 0 else (box.xyxy[1] + box.xyxy[3]) / 2
            objects.append({
                "id": i,
                "class_name": r.names[cls_id],
                "conf": float(box.conf.item()),
                "bbox": box.xyxy[0].cpu().numpy().tolist(),
                "mask": mask_np,
                "contour": main_contour.squeeze(1).tolist(),
                "center_2d": (cx, cy),
            })
    return objects
```

### 2.3 使用深度图计算物体 3D 位置

```python
def pixel_to_world(u: float, v: float, depth_image: np.ndarray,
                   cam_intrinsics: np.ndarray, T_cam_to_world: np.ndarray,
                   depth_scale: float = 0.001) -> np.ndarray:
    """
    像素坐标 → 世界坐标
    cam_intrinsics: 3×3 内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    T_cam_to_world: 4×4 相机到世界的变换矩阵
    depth_scale: 深度的单位换算（如 mm → m 则为 0.001）
    """
    u_int, v_int = int(round(u)), int(round(v))
    z = depth_image[v_int, u_int] * depth_scale  # 转为米
    if z <= 0 or z > 5.0:  # 无效深度检查
        return None
    # 像素 → 相机坐标系
    x_cam = (u - cam_intrinsics[0, 2]) * z / cam_intrinsics[0, 0]
    y_cam = (v - cam_intrinsics[1, 2]) * z / cam_intrinsics[1, 1]
    point_cam = np.array([x_cam, y_cam, z, 1.0])
    # 相机 → 世界
    point_world = T_cam_to_world @ point_cam
    return point_world[:3] / point_world[3]
```

### 2.4 物体 3D 信息整合

对每个检测到的物体：

```python
# mask 中心点的 3D 坐标
obj["center_3d"] = pixel_to_world(obj["center_2d"][0], obj["center_2d"][1],
                                   depth_image, K, T_cam_to_world)

# 物体包围球半径（mask最大外接圆半径，用于粗略尺寸估计）
obj["radius_3d"] = estimate_3d_radius(obj["mask"], obj["center_3d"],
                                       depth_image, K, T_cam_to_world)
```

---

## 三、推理层：LLM 垃圾分类

### 3.1 接口设计

```python
def classify_waste(object_name: str) -> dict:
    """
    输入: 物体名称, 例如 "apple"
    输出: {
        "category": "厨余垃圾",       # 四大类之一
        "category_id": 1,             # 0-3
        "reason": "苹果属于食物残渣，易腐烂，归入厨余垃圾",
        "special_note": "苹果核对猫有毒，注意安全"  # 可选
    }
    """
```

### 3.2 提示词模板

```python
WASTE_CLASSIFY_PROMPT = """你是垃圾分类专家。中国大陆四大垃圾分类标准：
- 可回收物 (id=0): 废纸、塑料瓶、金属罐、玻璃瓶、旧衣物等
- 厨余垃圾 (id=1): 剩饭剩菜、果皮、茶叶渣、骨头等
- 有害垃圾 (id=2): 废电池、废灯管、废药品、废油漆等
- 其他垃圾 (id=3): 砖瓦陶瓷、渣土、卫生纸、烟蒂等

请判断以下物体属于哪类垃圾：
物体: {object_name}

只回复 JSON，不要其他内容：
{{"category": "类别名", "category_id": 数字, "reason": "判断依据"}}
"""
```

### 3.3 实现建议

- 优先接**本地小模型**（Qwen2.5-7B / Llama3-8B 量化版），延迟控制在 100ms 内
- 对常见 10 个物体，可直接硬编码映射表兜底（LLM 不可用时仍能工作）
- 批量查询：一次性把所有物体名发给 LLM，减少调用次数

---

## 四、抓取层：GraspNet + Mask 滤波

### 4.1 GraspNet 输入/输出

```
输入:
  - rgb_image:  [H, W, 3] uint8
  - depth_image: [H, W] float32, 单位米

输出 (全图所有抓取候选):
  - grasp_poses:  [N, 4, 4]  抓取位姿 (世界/相机坐标系)
  - grasp_scores: [N]         质量分数 [0, 1]
  - grasp_widths: [N]         夹爪张开宽度 (米)
  - grasp_centers_2d: [N, 2]  抓取中心在图像上的像素坐标
```

### 4.2 用 Mask 过滤抓取候选

**这是整个 pipeline 最关键的一步**：将 GraspNet 的全图候选限制在目标物体的 mask 范围内。

```python
def filter_grasps_by_mask(
    all_grasps: dict,
    object_mask: np.ndarray,    # [H, W] bool, 目标物体的mask
    margin: int = 5             # mask 向内收缩的像素，避免抓边缘
) -> dict:
    """
    只保留抓取中心落在物体mask内部的候选
    margin: mask向内腐蚀的像素，避免夹爪碰到物体边缘
    """
    eroded_mask = cv2.erode(
        object_mask.astype(np.uint8),
        np.ones((margin, margin), np.uint8)
    ).astype(bool)

    valid_indices = []
    for i, (u, v) in enumerate(all_grasps["centers_2d"]):
        u_int, v_int = int(round(u)), int(round(v))
        if 0 <= v_int < eroded_mask.shape[0] and 0 <= u_int < eroded_mask.shape[1]:
            if eroded_mask[v_int, u_int]:  # 抓取中心在物体mask内
                valid_indices.append(i)

    return {
        "poses": all_grasps["poses"][valid_indices],
        "scores": all_grasps["scores"][valid_indices],
        "widths": all_grasps["widths"][valid_indices],
        "centers_2d": all_grasps["centers_2d"][valid_indices],
    }
```

### 4.3 碰撞预检

```python
def check_grasp_collision_free(
    grasp_pose: np.ndarray,
    gripper_open_width: float,
    scene_objects: list,  # 场景中其他物体的 3D bbox/ mesh
    safety_margin: float = 0.02
) -> bool:
    """
    简易碰撞检测:
    1. 计算夹爪完全张开时两个手指在 world 空间的位置
    2. 检查是否与场景中非目标物体重叠
    3. 检查桌子/地面碰撞（z 不能为负或超过桌子高度 + 安全距离）

    IsaacLab 中可以直接用 PhysX 的碰撞查询或 Isaac's own collision checker
    """
    # 使用 IsaacLab 内置的物理引擎做碰撞查询更准确
    # 此处为占位逻辑，实际实现用 PhysX scene query
    return True
```

### 4.4 最优抓取选择

```python
def select_best_grasp(
    filtered_grasps: dict,
    object_center_3d: np.ndarray,
    prefer_top_down: bool = True
) -> dict | None:
    """
    从过滤后的候选中选择最优抓取:
    1. 按 score 排序
    2. 优先选自上而下的抓取（grasp pose 的 z 轴接近世界 z 轴反方向）
    3. 在 top_k 个候选中选距离物体中心最近的
    """
    if len(filtered_grasps["scores"]) == 0:
        return None

    # 综合打分: score * 0.6 + approach_bonus * 0.3 + centrality * 0.1
    top_k = min(10, len(filtered_grasps["scores"]))
    indices = np.argsort(filtered_grasps["scores"])[::-1][:top_k]

    best_idx = indices[0]
    best_score = -1
    for idx in indices:
        pose = filtered_grasps["poses"][idx]
        # 检查 approach direction 是否接近垂直向下
        approach_dir = pose[:3, 2]  # z-axis of grasp frame
        world_down = np.array([0, 0, -1])
        approach_bonus = max(0, np.dot(approach_dir, world_down))

        # 抓取中心与物体中心的距离
        grasp_pos = pose[:3, 3]
        dist = np.linalg.norm(grasp_pos - object_center_3d)
        centrality = max(0, 1.0 - dist / object_radius)

        combined = filtered_grasps["scores"][idx] * 0.6 + approach_bonus * 0.3 + centrality * 0.1
        if combined > best_score:
            best_score = combined
            best_idx = idx

    return {
        "pose": filtered_grasps["poses"][best_idx],
        "score": filtered_grasps["scores"][best_idx],
        "width": filtered_grasps["widths"][best_idx],
    }
```

---

## 五、动作执行层

### 5.1 抓取状态机

```
IDLE → APPROACH → PRE_GRASP → GRASP → LIFT → HOLD → DONE
                         ↑         |
                         └── RETRY ┘  (抓取失败时重试)
```

```python
class GraspStateMachine:
    STATES = ["IDLE", "APPROACH", "PRE_GRASP", "GRASP", "LIFT", "HOLD", "DONE", "FAILED"]

    def __init__(self):
        self.state = "IDLE"
        self.grasp_pose = None    # 世界坐标系下的抓取位姿
        self.grasp_width = None   # 夹爪张开宽度
        self.approach_offset = 0.15  # 预抓取点距抓取点的后退距离 (m)
        self.lift_height = 0.2    # 抓起后提升高度 (m)
        self.retry_count = 0
        self.max_retries = 3

    def compute_trajectory(self) -> list[np.ndarray]:
        """返回 waypoints: 一系列末端位姿 (世界坐标系)"""
        if self.state == "IDLE":
            return []
        elif self.state == "APPROACH":
            # waypoint 1: pre_grasp = grasp_pose 沿 approach 方向后退 15cm
            pre_grasp = self.grasp_pose.copy()
            pre_grasp[:3, 3] -= pre_grasp[:3, :3] @ np.array([0, 0, self.approach_offset])
            return [pre_grasp]
        elif self.state == "PRE_GRASP":
            # waypoint 2: 到达抓取点，夹爪张开
            return [self.grasp_pose, {"gripper_open": self.grasp_width + 0.01}]  # 多张开 1cm 容错
        elif self.state == "GRASP":
            # 闭合夹爪
            return [{"gripper_close": True}]
        elif self.state == "LIFT":
            # waypoint 3: 垂直提升
            lift_pose = self.grasp_pose.copy()
            lift_pose[2, 3] += self.lift_height
            return [lift_pose]
        elif self.state == "HOLD":
            return []  # 保持当前位置
        elif self.state == "DONE":
            return []
```

### 5.2 运动规划集成

```
IsaacLab 中推荐使用:
- RMPFlow 或 Lula (Isaac 内置): 用于无碰撞轨迹规划
- 或直接调用 MoveIt 2 的 ROS bridge (如果用 ROS 接口)
- 简单场景可用 IK solver + 直线插值

步骤:
1. 用 IK 求解目标末端位姿对应的关节角
2. 检查自碰撞和环境碰撞
3. 生成平滑轨迹
4. 通过 joint position controller 或 joint impedance controller 执行
```

### 5.3 抓取成功判定

```python
def check_grasp_success(ee_pose: np.ndarray, object_3d_center: np.ndarray,
                        gripper_closed: bool, lift_start_z: float) -> bool:
    """
    判定标准 (满足任一即可):
    1. 物体在夹爪之间（夹爪传感器检测到力/位置反馈不为零）
    2. 物体 z 坐标 > lift_start_z + 0.05m（物体已离开桌面）
    3. 简单判断: 物体不再在桌子平面上（z > 桌子 z + 阈值）
    """
    # 在 IsaacLab 中: 读取物体实际 pose，判断 z 是否超过桌面
    object_z = object_3d_center[2]  # 物体当前的 z 坐标
    table_z = 0.75  # 桌面高度，从场景读取
    return object_z > table_z + 0.05
```

---

## 六、主循环

```python
def main_grasp_loop(target_class: str | None = None):
    """
    主循环: 感知→推理→规划→执行

    Args:
        target_class: 指定要抓的类别（如"可回收物"），None表示按优先级自动选择
    """
    # === 初始化 ===
    model_seg = YOLO("yolov8m-seg.pt")
    graspnet = load_graspnet_model()  # 加载预训练 GraspNet
    gripper = RobotGripper()
    arm = RobotArm()
    camera = RGBDCamera()
    sm = GraspStateMachine()

    # === 标定数据 ===
    K = camera.get_intrinsics()  # 3×3
    T_cam_to_ee = np.load("calib/T_cam_to_ee.npy")
    T_ee_to_world = arm.get_ee_pose()  # 实时读取

    while True:
        # Step 1: 获取传感器数据
        rgb = camera.get_rgb()
        depth = camera.get_depth()  # 单位: 米
        T_ee_to_world = arm.get_ee_pose()
        T_cam_to_world = T_ee_to_world @ T_cam_to_ee

        # Step 2: YOLOv8-seg 感知
        objects = perceive(rgb)  # 返回物体列表，每个有 mask+class_name+center_2d

        if len(objects) == 0:
            print("[感知] 场景中无物体，等待...")
            time.sleep(0.5)
            continue

        # Step 3: 计算 3D 信息 + LLM 分类
        waste_list = []
        for obj in objects:
            obj["center_3d"] = pixel_to_world(
                obj["center_2d"][0], obj["center_2d"][1],
                depth, K, T_cam_to_world
            )
            # LLM 垃圾分类
            waste_info = classify_waste(obj["class_name"])
            obj.update(waste_info)
            waste_list.append(obj)

        # Step 4: 选择目标物体
        if target_class:
            candidates = [o for o in waste_list if o["category"] == target_class]
        else:
            # 默认优先级: 有害垃圾 > 厨余垃圾 > 可回收物 > 其他垃圾
            priority = {"有害垃圾": 0, "厨余垃圾": 1, "可回收物": 2, "其他垃圾": 3}
            candidates = sorted(waste_list, key=lambda o: (
                priority.get(o["category"], 99),
                -o["conf"]  # 同类中优先置信度高的
            ))

        target = candidates[0] if candidates else None
        if target is None:
            print("[选择] 无目标物体")
            continue

        print(f"[选择] 目标: {target['class_name']} → {target['category']} "
              f"(conf={target['conf']:.3f})")

        # Step 5: GraspNet 全图抓取检测 + Mask 过滤
        all_grasps = graspnet.detect(rgb, depth)
        filtered = filter_grasps_by_mask(all_grasps, target["mask"], margin=5)

        if len(filtered["scores"]) == 0:
            print("[抓取] 无有效抓取候选，尝试下一个物体")
            waste_list.remove(target)
            continue

        # Step 6: 选择最优抓取
        best = select_best_grasp(filtered, target["center_3d"], prefer_top_down=True)
        if best is None:
            continue

        grasp_pose_cam = best["pose"]
        grasp_pose_world = T_cam_to_world @ grasp_pose_cam  # 转到世界坐标系
        grasp_width = best["width"]
        print(f"[抓取] 最优抓取位姿 world:\n{grasp_pose_world}")
        print(f"[抓取] 夹爪宽度: {grasp_width:.3f}m, 分数: {best['score']:.3f}")

        # Step 7: 碰撞检测
        if not check_grasp_collision_free(grasp_pose_world, grasp_width, objects):
            print("[安全] 抓取存在碰撞风险，跳过")
            waste_list.remove(target)
            continue

        # Step 8: 执行抓取状态机
        sm = GraspStateMachine()
        sm.grasp_pose = grasp_pose_world
        sm.grasp_width = grasp_width

        for state in ["APPROACH", "PRE_GRASP", "GRASP", "LIFT"]:
            sm.state = state
            waypoints = sm.compute_trajectory()

            if state == "PRE_GRASP":
                gripper.open(sm.grasp_width + 0.01)
            elif state == "GRASP":
                gripper.close()

            for wp in waypoints:
                if isinstance(wp, np.ndarray) and wp.shape == (4, 4):
                    # 逆运动学求解 + 轨迹执行
                    joint_traj = arm.plan_trajectory(wp)
                    if joint_traj is None:
                        print(f"[运动] IK 无解 at {state}")
                        sm.state = "FAILED"
                        break
                    arm.execute(joint_traj)
                    time.sleep(0.1)

            if sm.state == "FAILED":
                break

        # Step 9: 抓取成功判定
        if sm.state != "FAILED":
            success = check_grasp_success(
                arm.get_ee_pose(),
                target["center_3d"],
                gripper.is_closed(),
                sm.grasp_pose[2, 3]
            )
            if success:
                print(f"✅ 抓取成功! 物体: {target['class_name']} 类别: {target['category']}")
                sm.state = "DONE"
                break
            else:
                print(f"❌ 抓取失败，物体可能滑落")
                sm.retry_count += 1
                gripper.open()
                if sm.retry_count < sm.max_retries:
                    sm.state = "IDLE"
                    continue
                else:
                    print(f"[放弃] 重试 {sm.max_retries} 次后放弃 {target['class_name']}")
                    waste_list.remove(target)
```

---

## 七、文件结构建议

```
grasp_pipeline/
├── calibration/
│   ├── verify_calibration.py    # 标定验证脚本
│   └── camera_params.py         # 相机内参、外参
├── perception/
│   ├── yolo_seg.py              # YOLOv8-seg 封装
│   └── depth_utils.py           # 深度图→世界坐标转换
├── reasoning/
│   └── waste_classifier.py      # LLM 垃圾分类
├── grasping/
│   ├── graspnet_wrapper.py      # GraspNet 封装
│   ├── mask_filter.py           # Mask 过滤 + 碰撞预检
│   └── grasp_selector.py        # 最优抓取选择
├── execution/
│   ├── state_machine.py         # 抓取状态机
│   ├── motion_planner.py        # IK + 轨迹规划 (IsaacLab 接口)
│   └── gripper_control.py       # 夹爪控制
├── main_pipeline.py             # 主循环
└── run_isaac_scene.py           # IsaacLab 场景启动 + Pipeline 集成
```

---

## 八、注意事项

1. **深度图质量**: GraspNet 对深度质量敏感，确保 IsaacLab 渲染的深度图无空洞、无噪声。如果用了仿真深度传感器，确认 `depth_range` 覆盖工作空间。

2. **GraspNet 模型选择**: 优先用 `graspnet-baseline` 的预训练权重（在大型抓取数据集上训练），不需要在自己的物体上微调即可工作。

3. **夹爪参数**: GraspNet 输出的 `width` 是建议值，需要 clamp 到实际夹爪的物理范围（如 [0, 0.08]米）。

4. **相机位置**: 推荐相机在机械臂腕部（eye-in-hand），抓前拍一次（look-then-move），而非持续跟踪。如果相机在头部（eye-to-hand），需要额外标定 camera → robot_base。

5. **多物体遮挡**: 如果 10 个物体有堆叠遮挡，建议加一个 `check_visibility` 步骤——用深度图判断 mask 中心附近是否有深度跳变（被遮挡），过滤掉被严重遮挡的物体。

6. **IsaacLab 特性利用**: IsaacLab 提供了 ground truth 语义分割和深度图，实现时可以用 GT 快速验证 pipeline 正确性，再切换到真实传感器模拟。
