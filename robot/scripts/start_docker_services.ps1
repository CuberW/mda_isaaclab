# Start all Docker ROS1 services for Kuavo 3.19
$container = "kuavo_official_ros"
$ws = "/root/kuavo_ws_linux"

Write-Host "Restarting container..." -ForegroundColor Yellow
docker restart $container | Out-Null
Start-Sleep 5

Write-Host "Starting roscore..." -ForegroundColor Yellow
docker exec -d $container bash -c "source /opt/ros/noetic/setup.bash; export ROS_HOSTNAME=localhost; export ROS_MASTER_URI=http://localhost:11311; exec roscore"
Start-Sleep 4

Write-Host "Starting IK node..." -ForegroundColor Yellow
docker exec -d $container bash -c "source /opt/ros/noetic/setup.bash; source $ws/devel/setup.bash; export ROS_MASTER_URI=http://localhost:11311; export ROBOT_VERSION=62; exec roslaunch motion_capture_ik ik_node.launch robot_version:=62 visualize:=false"
Start-Sleep 10

Write-Host "Starting wheel bridge..." -ForegroundColor Yellow
docker exec -d $container bash -c "source /opt/ros/noetic/setup.bash; source $ws/devel/setup.bash; export ROS_MASTER_URI=http://localhost:11311; exec python3 $ws/src/kuavo_wheel/scripts/wheel_bridge.py --kuavo-master http://localhost:11311 --wheel-master http://localhost:11311 --kuavo-slave-ip 127.0.0.1 --wheel-slave-ip 127.0.0.1"
Start-Sleep 3

Write-Host "Starting arm trajectory..." -ForegroundColor Yellow
docker exec -d $container bash -c "source /opt/ros/noetic/setup.bash; source $ws/devel/setup.bash; export ROS_MASTER_URI=http://localhost:11311; exec rosrun humanoid_plan_arm_trajectory arm_trajectory_bezier_process.py"
Start-Sleep 3

Write-Host "Starting dummy claw..." -ForegroundColor Yellow
docker exec -d $container bash -c "source /opt/ros/noetic/setup.bash; export ROS_MASTER_URI=http://localhost:11311; exec python3 -c 'import rospy;rospy.init_node(\"dc\",anonymous=True,disable_signals=True);rospy.Subscriber(\"/leju_claw_command\",rospy.AnyMsg,lambda m:None);rospy.spin()'"
Start-Sleep 2

Write-Host "`nVerifying..." -ForegroundColor Green
$retry = 0
do {
    Start-Sleep 2
    $ik = docker exec $container bash -lc "source /opt/ros/noetic/setup.bash; export ROS_MASTER_URI=http://localhost:11311; rosservice list 2>&1 | grep two_arm_hand_pose_cmd_srv"
    $base = docker exec $container bash -lc "source /opt/ros/noetic/setup.bash; export ROS_MASTER_URI=http://localhost:11311; rostopic list 2>&1 | grep base_cmd_vel"
    $retry++
} while ($retry -lt 5 -and (-not $ik -or -not $base))

Write-Host "IK: $ik"
Write-Host "Base: $base"

if ($ik -and $base) {
    Write-Host "`nALL DOCKER SERVICES READY" -ForegroundColor Green
} else {
    Write-Host "`nFAILED - IK or base missing" -ForegroundColor Red
}
