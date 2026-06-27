# MDA IsaacLab 3.19 Garbage Sorting Delivery README

本目录是 `mda_isaaclab` 仓库内的 3.19 轮式人形机器人垃圾分类投放交付材料入口。当前验收主线是：

```text
IsaacLab/Isaac Sim 场景
-> Kuavo 轮式双臂机器人头部 RGB-D
-> VISS v28 Qwen-first + YOLO11 ROI 分割
-> RGB-D 目标 3D 定位和 top grasp 姿态
-> Nav2 动态桌边站位
-> 抓取或演示辅助抓取
-> Nav2 到对应垃圾桶
-> 右夹爪移动到桶口上方释放
```

严格物理抓取仍在调试。完整系统演示视频使用显式的 gripper proximity / legacy suction alias 辅助路径跑通全流程；严格物理抓取验证命令会在失败时停止并记录原因。

## 交付材料索引

| 交付项 | 路径 |
| --- | --- |
| 部署和运行说明 | `docs/task319_delivery/README.md` |
| 交付文件清单 | `docs/task319_delivery/DELIVERY_CHECKLIST.md` |
| 问题调试报告 | `docs/task319_delivery/DEBUG_REPORT.md` |
| 答辩 PPT 文稿 | `docs/task319_delivery/PPT_OUTLINE.md` |
| 答辩 PPT 文件 | `docs/task319_delivery/TASK319_DEFENSE_PPT.pptx` |
| 演示视频证据说明 | `docs/task319_delivery/VIDEO_EVIDENCE.md` |
| 原开发报告 | `TASK319_DEVELOPMENT_REPORT.md` |
| 主任务入口 | `task_319_garbage_sort/task319_grasp_sort_sm.py` |
| 主场景和状态机 | `task_319_garbage_sort/visual_grasp_record_demo.py` |
| 仿真工程资产 | `isaac_projects/`, `robot/`, `task_319_garbage_sort/two_finger_gripper.urdf` |

## 环境版本

本机已验证环境：

| 项目 | 版本或路径 |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS |
| Python | `conda activate my_task319_safe` 后使用 `python`, Python 3.11.14 |
| IsaacLab | 0.54.4, editable path `IsaacLab/source/isaaclab` |
| PyTorch | 2.7.0+cu128, CUDA available |
| GPU | NVIDIA GeForce RTX 4090, driver 580.159.03 |
| ROS 2 | `ROS_DISTRO=jazzy` |
| Ultralytics | 8.4.72 |
| OpenCV | 4.10.0.84 |
| NumPy | 1.26.0 |
| skrl | 2.1.0 |

进入仓库根目录：

```bash
cd mda_isaaclab
```

激活项目 Python 环境后再运行命令：

```bash
conda activate my_task319_safe
python --version
```

后续命令统一使用当前环境中的 `python`，避免绑定个人机器上的解释器绝对路径。

## 依赖安装

以下步骤按“新机器从零部署”编写。已经配好环境时可跳过已完成步骤，但最终仍建议执行本节末尾的检查命令。

### 1. 系统依赖

Ubuntu 24.04 / ROS 2 Jazzy 环境建议先安装基础工具：

```bash
sudo apt update
sudo apt install -y \
  git git-lfs curl wget unzip ffmpeg \
  build-essential cmake ninja-build \
  libgl1 libglib2.0-0
git lfs install
```

ROS 2 Jazzy 安装参考官方文档：

```text
https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
```

安装完成后确认：

```bash
source /opt/ros/jazzy/setup.bash
ros2 --version
```

本项目 Nav2 启动脚本会 source 系统 ROS 2，并叠加本地 Nav2 包：

```bash
bash task_319_garbage_sort/scripts/setup_nav2_user_install.bash
```

### 2. Conda 和 Python 依赖

建议使用 Python 3.11。不要在系统 Python 里安装 IsaacLab 相关依赖：

```bash
cd mda_isaaclab
conda create -n my_task319_safe python=3.11 -y
conda activate my_task319_safe
python -m pip install -U pip setuptools wheel
```

本机已验证的 PyTorch 是 `2.7.0+cu128`。如需复现该版本，可按 PyTorch 官方 previous versions 页面选择 CUDA 12.8 wheel：

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

PyTorch 官方版本页面：

```text
https://pytorch.org/get-started/previous-versions/
```

安装 Task319 主线 Python 依赖：

```bash
python -m pip install \
  ultralytics==8.4.72 \
  opencv-python==4.10.0.84 \
  numpy==1.26.0 \
  dashscope openai \
  skrl==2.1.0 \
  huggingface_hub gdown
```

抓取诊断和可选后端依赖由项目脚本检查/安装：

```bash
python task_319_garbage_sort/scripts/prepare_grasp_deps.py --install_python_deps
python task_319_garbage_sort/scripts/prepare_grasp_deps.py --download_missing_models
```

### 3. IsaacLab / Isaac Sim

IsaacLab 和 Isaac Sim 按 NVIDIA 官方安装文档配置：

```text
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
https://github.com/isaac-sim/IsaacLab
```

如果仓库根目录已经包含 `IsaacLab/`，在当前 conda 环境中安装 IsaacLab 扩展：

```bash
cd mda_isaaclab
conda activate my_task319_safe
cd IsaacLab
./isaaclab.sh --install
cd ..
```

检查 IsaacLab 是否可导入：

```bash
python - <<'PY'
import isaaclab
import torch
print("isaaclab import ok")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
```

### 4. Qwen/DashScope API

VLM 使用 DashScope/Qwen API。先到阿里云百炼控制台申请 API Key：

```text
https://help.aliyun.com/zh/model-studio/get-api-key
```

运行前在同一个 shell 中设置：

```bash
export DASHSCOPE_API_KEY="your_dashscope_key"
export QWEN_MODEL="qwen3-vl-flash"
export QWEN_API_STYLE="dashscope"
```

连通性检查：

```bash
python viss/scripts/perception/yolo11_qwen_perception_offline.py --qwen-api-check --qwen-api-style dashscope
```

## 模型下载位置

主线验收必须具备 `viss/models/yolo11s-seg-best.pt`。该文件是项目内训练/整理权重，不是公开官方模型；提交交付物时应随“模型附件/团队模型盘”一起提供。公开备用模型可按下面命令下载。

| 模型 | 获取位置 | 本地放置路径 | 用途 |
| --- | --- | --- | --- |
| YOLO11 垃圾分割主线权重 | 项目交付模型附件/团队模型盘 | `viss/models/yolo11s-seg-best.pt` | VISS v28 主线 ROI 分割 |
| VISS 备用分割权重 | 项目交付模型附件/团队模型盘 | `viss/models/best_seg.pt`, `viss/models/yolo11s-seg.pt` | 备用分割实验 |
| YOLOv8n | https://github.com/ultralytics/assets/releases | `models/yolov8n.pt` | 备用检测 |
| YOLOv8m-seg | https://github.com/ultralytics/assets/releases | `models/yolo/yolov8m-seg.pt` | 备用分割 |
| YOLO-World v2 | https://github.com/ultralytics/assets/releases | `models/yolov8s-worldv2.pt` | 开放词汇备用 |
| GraspNet RS checkpoint | https://github.com/graspnet/graspnet-baseline 或 Google Drive ID `1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk` | `models/graspnet-rs/checkpoint-rs.tar` | 抓取诊断备用 |
| SAM ViT-B | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth | `models/sam-vit-b/sam_vit_b_01ec64.pth` | 分割备用 |
| CLIP ViT-B/32 | https://huggingface.co/openai/clip-vit-base-patch32 | `weights/clip/ViT-B-32.pt` 或 `models/clip-vit-b32/` | 视觉语义备用 |
| DINOv2 Base | https://huggingface.co/facebook/dinov2-base | `models/dinov2-base/` | 视觉特征备用 |
| DA3 Small | https://huggingface.co/depth-anything/da3-small | `models/da3-small/` | 深度估计备用 |

### 1. 准备目录

```bash
mkdir -p \
  viss/models \
  models/yolo \
  models/sam-vit-b \
  models/graspnet-rs \
  models/clip-vit-b32 \
  models/dinov2-base \
  models/da3-small \
  weights/clip
```

### 2. 放置主线自训练权重

从交付模型附件或团队模型盘复制主线权重：

```bash
cp path/to/model_bundle/viss/models/yolo11s-seg-best.pt viss/models/yolo11s-seg-best.pt
```

可选备用权重同样从模型附件复制：

```bash
cp path/to/model_bundle/viss/models/best_seg.pt viss/models/best_seg.pt
cp path/to/model_bundle/viss/models/yolo11s-seg.pt viss/models/yolo11s-seg.pt
```

如果交付包里没有 `viss/models/yolo11s-seg-best.pt`，主线视觉链路不能复现；需要先补齐该权重。

### 3. 下载公开 YOLO / SAM 权重

```bash
curl -L -o models/yolov8n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

curl -L -o models/yolo/yolov8m-seg.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-seg.pt

curl -L -o models/yolov8s-worldv2.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt

curl -L -o models/sam-vit-b/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 4. 下载 HuggingFace 模型

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download

snapshot_download(
    repo_id="openai/clip-vit-base-patch32",
    local_dir="models/clip-vit-b32",
)
snapshot_download(
    repo_id="facebook/dinov2-base",
    local_dir="models/dinov2-base",
)
snapshot_download(
    repo_id="depth-anything/da3-small",
    local_dir="models/da3-small",
)
hf_hub_download(
    repo_id="dgrachev/a2_pretrained",
    filename="checkpoint-rs.tar",
    local_dir="models/graspnet-rs",
)
PY
```

如果网络无法访问 HuggingFace，可设置镜像后重试：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

GraspNet 官方 RealSense checkpoint 也可用 `gdown` 从 Google Drive 下载：

```bash
gdown --id 1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk -O models/graspnet-rs/checkpoint-rs.tar
```

### 5. 权重完整性检查

```bash
python - <<'PY'
from pathlib import Path

required = [
    "viss/models/yolo11s-seg-best.pt",
    "models/yolo/yolov8m-seg.pt",
]
optional = [
    "models/yolov8n.pt",
    "models/yolov8s-worldv2.pt",
    "models/sam-vit-b/sam_vit_b_01ec64.pth",
    "models/graspnet-rs/checkpoint-rs.tar",
    "models/clip-vit-b32/config.json",
    "models/dinov2-base/config.json",
    "models/da3-small/config.json",
]

ok = True
for name in required:
    path = Path(name)
    exists = path.is_file()
    size_mb = path.stat().st_size / 1024 / 1024 if exists else 0
    print(f"[{'OK' if exists else 'MISS'}] required {name} {size_mb:.1f} MB")
    ok = ok and exists
for name in optional:
    path = Path(name)
    exists = path.exists()
    size_mb = path.stat().st_size / 1024 / 1024 if path.is_file() else 0
    print(f"[{'OK' if exists else 'MISS'}] optional {name} {size_mb:.1f} MB")
raise SystemExit(0 if ok else 1)
PY
```

当前仓库中的 `models/README.md` 记录了部分公开模型来源，但其中 `python scripts/download_models.py` 是历史入口；交付运行以本节命令和现有本地权重为准。

## 启动命令

场景预览，只检查 Kuavo、桌子、4 个垃圾桶和 10 个桌面垃圾是否加载：

```bash
python task_319_garbage_sort/phase1_scene_ros2_bridge.py --disable_lidar
```

当前主线 GUI 命令，默认严格物理抓取，失败会记录并停止对应 cycle：

```bash
python task_319_garbage_sort/task319_grasp_sort_sm.py --mind_sort_demo
```

完整系统演示视频命令，显式启用 legacy suction alias / gripper proximity 辅助路径，用于展示识别、导航、分类、投放闭环：

```bash
python task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --mind_sort_demo \
  --mind_sort_suction_assist \
  --no-mind_sort_gripper_proximity_assist \
  --mind_sort_allow_stale_reshoot_for_suction_demo \
  --no-target_reachability_ik_check \
  --record_video \
  --video_width 1280 \
  --video_height 720 \
  --video_sample_stride 4 \
  --no-gui_realtime_playback
```

严格物理抓取单物体验证，不启用辅助抓取：

```bash
python task_319_garbage_sort/task319_grasp_sort_sm.py \
  --headless \
  --mind_sort_demo \
  --mind_sort_max_objects 1 \
  --no-mind_sort_gripper_proximity_assist \
  --no-mind_sort_suction_assist \
  --record_video \
  --video_width 1280 \
  --video_height 720 \
  --video_sample_stride 4 \
  --no-gui_realtime_playback
```

只验证视觉到动态站位和 Nav2，不抓取：

```bash
python task_319_garbage_sort/task319_grasp_sort_sm.py --standpoint_nav_only --record_video
```

简化小方块物理抓取校准：

```bash
python task_319_garbage_sort/topdown_cube_grasp_test.py \
  --record_video \
  --object_pos 0.70,-0.16,0.560 \
  --object_size 0.050,0.032,0.040 \
  --object_yaw_deg 25
```

## 输出位置

运行输出默认写入：

```text
task_319_garbage_sort/output/head_camera_grasp_records/<timestamp>/
```

重点文件：

- `video_manifest.json`: 视频录制状态。
- `external_grasp_demo.mp4` 或人工整理后的 `grasp8.mp4`: 演示视频。
- `mind_sort_demo/mind_sort_task_queue.json`: 全局任务队列、完成/失败物体和导航记录。
- `mind_sort_demo/cycle_XXXX/mind_sort_cycle.json`: 单个目标的识别、分类、站位、抓取和投放记录。
- `mind_sort_demo/cycle_XXXX/standpoint_reshoot/v28_original/`: 导航到抓取站位后的 VISS/Qwen/YOLO 结果。
- `nav2_stack/`: 本次 Nav2 参数、地图、行为树和日志。

## 当前实验结果

完整多物体演示 run：

```text
task_319_garbage_sort/output/head_camera_grasp_records/20260625_150100/
```

当前可见视频：

```text
docs/task319_delivery/videos/task319_system_demo_480p.mp4
```

原始 720p 本地证据视频：

```text
task_319_garbage_sort/output/head_camera_grasp_records/20260625_150100/grasp8.mp4
```

GitHub 提交版视频由原始 `grasp8.mp4` 压缩生成，保留完整时长，分辨率 854x480，帧率 10 fps。该 run 的 `video_manifest.json` 仍记录原始输出名 `external_grasp_demo.mp4`，但当前目录中实际保留的视频文件名是 `grasp8.mp4`。该 run 完成 8 个物体的识别、分类、桶口释放，最终在第 9 次返回观察位时 Nav2 返回 `STATUS_6`，所以全局 `success=false`，但 `failed_scene_keys=[]`，已完成 cycle 均为成功。

| Cycle | 物体 | 分类 | Cycle | 严格物理抓取 | 辅助抓取 | 投放 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `trash_potted_meat_can_0` | 可回收物 | 成功 | 成功 | 成功 | 成功 |
| 1 | `trash_cracker_box_0` | 可回收物 | 成功 | 失败 | 成功 | 成功 |
| 2 | `trash_tomato_soup_can_0` | 可回收物 | 成功 | 失败 | 成功 | 成功 |
| 3 | `trash_foam_brick_0` | 其他垃圾 | 成功 | 失败 | 成功 | 成功 |
| 4 | `trash_bleach_cleanser_0` | 有害垃圾 | 成功 | 失败 | 成功 | 成功 |
| 5 | `trash_battery_0` | 有害垃圾 | 成功 | 失败 | 成功 | 成功 |
| 6 | `trash_sugar_box_0` | 可回收物 | 成功 | 失败 | 成功 | 成功 |
| 7 | `trash_banana_0` | 厨余垃圾 | 成功 | 失败 | 成功 | 成功 |

最新严格物理单物体验证 run：

```text
task_319_garbage_sort/output/head_camera_grasp_records/20260626_145846/
```

结果：`trash_potted_meat_can_0` 单物体严格物理抓取失败，原因是 cuRobo Cartesian final descent 未到达目标/近距阈值，未启用辅助 attach/carry。

早期逻辑动画验收记录：

```text
isaac_projects/isaac_task_execution_result.json
```

该记录是逻辑 pick-and-place 动画，不使用真实物理夹爪接触或 IK。10 个任务的逻辑指标均为 100%，只能作为场景和分类动作映射的早期证据。

## 常见问题

1. `viss/models/yolo11s-seg-best.pt` 缺失：
   这是本项目主线自训练/整理权重，不是公开官方模型。必须从交付模型附件或团队模型盘复制到 `viss/models/yolo11s-seg-best.pt`；没有该文件时，VISS v28 主线视觉链路不能复现。

2. HuggingFace 或 GitHub 权重下载失败：
   先确认网络能访问对应域名。HuggingFace 可设置 `HF_ENDPOINT=https://hf-mirror.com` 后重试；GitHub release 下载失败时，可在浏览器打开表格中的链接手动下载，再放到 README 指定的本地路径。

3. Qwen API 检查失败：
   检查 `DASHSCOPE_API_KEY`、`QWEN_MODEL`、`QWEN_API_STYLE` 是否在同一个 shell 中导出。

4. 没有视频：
   检查是否传入 `--record_video`，并查看 `video_manifest.json` 的 `error` 字段。GUI 模式下不要提前关闭 Isaac 窗口。

5. Nav2 返回 `STATUS_6`：
   这是已知问题，通常发生在长循环返回观察位或末端 docking 残差阶段。单个已完成 cycle 的投放记录仍应以对应 `mind_sort_cycle.json` 为准。

6. 物理抓取失败：
   当前严格物理夹爪仍是主要未闭环项。调试入口包括 `topdown_cube_grasp_test.py`、`gripper_physics_test.py`、`task319_local_grasp_rl/`。

7. 不应把辅助演示当作物理夹爪验收：
   启用 `--mind_sort_suction_assist` 或 `--mind_sort_gripper_proximity_assist` 的 run 只证明系统级闭环，不证明真实夹爪已经稳定抓住所有垃圾。
