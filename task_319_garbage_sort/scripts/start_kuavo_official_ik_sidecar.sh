#!/usr/bin/env bash
# Start the official Kuavo ROS1 IK node plus the Task319 JSON socket bridge.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/setup_kuavo_official_ik_env.sh"

HOST="${KUAVO_IK_BRIDGE_HOST:-127.0.0.1}"
PORT="${KUAVO_IK_BRIDGE_PORT:-31975}"
SERVICE="${KUAVO_IK_SERVICE:-/ik/two_arm_hand_pose_cmd_srv_muli_refer}"
LOG_DIR="${KUAVO_IK_LOG_DIR:-$WORKSPACE_ROOT/task_319_garbage_sort/output/kuavo_ik_sidecar}"
ROBOT_VERSION_ARG="${ROBOT_VERSION:-43}"
mkdir -p "$LOG_DIR"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$LOG_DIR/ros_logs}"
mkdir -p "$ROS_LOG_DIR"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

ros_master_ready() {
  "$KUAVO_ROS1_PYTHON" - <<'PY' >/dev/null 2>&1
import os
import socket
import sys
import xmlrpc.client

socket.setdefaulttimeout(1.0)
uri = os.environ.get("ROS_MASTER_URI", "http://127.0.0.1:11311")
try:
    xmlrpc.client.ServerProxy(uri).getSystemState("/task319_master_probe")
except Exception:
    sys.exit(1)
PY
}

ik_service_ready() {
  SERVICE_TO_CHECK="$SERVICE" "$KUAVO_ROS1_PYTHON" - <<'PY' >/dev/null 2>&1
import os
import socket
import sys
import xmlrpc.client

socket.setdefaulttimeout(1.0)
uri = os.environ.get("ROS_MASTER_URI", "http://127.0.0.1:11311")
service = os.environ["SERVICE_TO_CHECK"]
try:
    _, _, state = xmlrpc.client.ServerProxy(uri).getSystemState("/task319_service_probe")
except Exception:
    sys.exit(1)
services = {name for name, _providers in state[2]}
sys.exit(0 if service in services else 1)
PY
}

ROSCORE_PID=""
if ! ros_master_ready; then
  "$KUAVO_ROS1_PYTHON" "$KUAVO_NOETIC_ROOT/bin/rosmaster" --core -p 11311 -w 3 >"$LOG_DIR/rosmaster.log" 2>&1 &
  ROSCORE_PID="$!"
  PIDS+=("$ROSCORE_PID")
  for _ in $(seq 1 80); do
    ros_master_ready && break
    if [[ -n "$ROSCORE_PID" ]] && ! kill -0 "$ROSCORE_PID" >/dev/null 2>&1; then
      break
    fi
    sleep 0.1
  done
fi

if ! ros_master_ready; then
  echo "[kuavo_ik_sidecar] rosmaster did not become available." >&2
  echo "[kuavo_ik_sidecar] See logs in: $LOG_DIR/rosmaster.log" >&2
  exit 1
fi

IK_PID=""
if ! ik_service_ready; then
  if [[ "$ROBOT_VERSION_ARG" == "15" ]]; then
    MODEL_VERSION=14
  else
    MODEL_VERSION="$ROBOT_VERSION_ARG"
  fi
  MODEL_PATH="${KUAVO_IK_MODEL_PATH:-$KUAVO_WS/src/kuavo_assets/models/biped_s${MODEL_VERSION}/urdf/drake/biped_v3_arm.urdf}"
  "$KUAVO_ROS1_PYTHON" "$KUAVO_NOETIC_ROOT/bin/rosparam" set /model_path "$MODEL_PATH"
  "$KUAVO_ROS1_PYTHON" "$KUAVO_NOETIC_ROOT/bin/rosparam" set /robot_version "$ROBOT_VERSION_ARG"
  "$KUAVO_ROS1_PYTHON" "$KUAVO_NOETIC_ROOT/bin/rosparam" set /control_hand_side 1
  "$KUAVO_ROS1_PYTHON" "$KUAVO_NOETIC_ROOT/bin/rosparam" set /print_ik_info false
  "$KUAVO_ROS1_PYTHON" "$KUAVO_NOETIC_ROOT/bin/rosparam" set /enable_ik_vis false
  "$KUAVO_WS/devel_mda319/.private/motion_capture_ik/lib/motion_capture_ik/arms_ik_node" __name:=arms_ik_node >"$LOG_DIR/arms_ik_node.log" 2>&1 &
  IK_PID="$!"
  PIDS+=("$IK_PID")
  for _ in $(seq 1 200); do
    ik_service_ready && break
    if [[ -n "$IK_PID" ]] && ! kill -0 "$IK_PID" >/dev/null 2>&1; then
      break
    fi
    sleep 0.1
  done
fi

if ! ik_service_ready; then
  echo "[kuavo_ik_sidecar] Service did not become available: $SERVICE" >&2
  echo "[kuavo_ik_sidecar] See logs in: $LOG_DIR" >&2
  exit 1
fi

echo "[kuavo_ik_sidecar] Official Kuavo IK service ready: $SERVICE"
echo "[kuavo_ik_sidecar] Starting JSON bridge on $HOST:$PORT"
"$KUAVO_ROS1_PYTHON" "$WORKSPACE_ROOT/task_319_garbage_sort/scripts/kuavo_ik_socket_bridge.py" \
  --host "$HOST" \
  --port "$PORT" \
  --service "$SERVICE" \
  2>&1 | tee "$LOG_DIR/kuavo_ik_socket_bridge.log"
