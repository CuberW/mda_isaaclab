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
