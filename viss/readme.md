#启动仿真：
$ISAAC = "D:\isaacsim"
$env:ROS_DISTRO = "jazzy"
$env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp"
$env:ROS_DOMAIN_ID = "0"
$env:PATH = "$env:PATH;$ISAAC\exts\isaacsim.ros2.core\jazzy\lib"

& "$ISAAC\isaac-sim.bat" --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge

#python环境:
source ~/envs/yolo/bin/activate

#先运行仿真服务器程序v26_active_view_search_isaac_server.py|v28_head_camera_two_stage_action_server.py 然后开启：
python3 ~/trashbot_ws/scripts/loop/v28_head_camera_two_stage_loop.py   --perception-backend qwen_first   --qwen-first-model ~/trashbot_ws/models/yolo11s-seg-best.pt   --qwen-first-conf 0.15   --qwen-first-roi-expand 2.0   --qwen-first-verify-mode none   --qwen-first-max-candidates 3   --qwen-first-max-roi-refine 3   --qwen-first-timeout 90   --qwen-first-fallback none   --max-cycles 10

## Task319 主场景集成说明

Task319 当前不直接使用 VISS 原始相机位姿。主状态机先在
`task_319_garbage_sort` 场景中用自己的 `head_rgbd` 相机拍 RGB-D，再调用
VISS v28/Qwen-first 的离线感知脚本：

```bash
python viss/scripts/perception/yolo11_qwen_perception_offline.py --qwen-api-check --qwen-api-style dashscope
```

当前默认权重：

```text
viss/models/yolo11s-seg-best.pt
```

Task319 主线保持 v28 的 Qwen-first 思路：Qwen 全局识别候选，YOLO 对 ROI
精分割，Qwen/VLM 输出垃圾四分类。不同之处是 3D 几何中心、可达性和抓取点
一律使用 Task319 当前相机的 intrinsics、depth 和 `camera_to_world` 计算，
不使用 VISS 原始仿真相机参数。
