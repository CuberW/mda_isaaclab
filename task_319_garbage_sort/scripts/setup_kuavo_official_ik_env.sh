#!/usr/bin/env bash
# Source this file before starting the official Kuavo ROS1 IK node.
#
# The current host is Ubuntu 24.04 with ROS2/Jazzy.  The Kuavo IK node is a
# ROS1/Noetic catkin binary, so we assemble a narrow ROS1 runtime without
# putting an old rootfs libc on LD_LIBRARY_PATH.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KUAVO_WS="${KUAVO_WS:-$WORKSPACE_ROOT/kuavo-ros-opensource}"
KUAVO_NOETIC_ROOT="${KUAVO_NOETIC_ROOT:-/home/zhxm/workspace/rl_ros1_ws/dvst2/barn/challenge/official_faithful/drl_vo/artifacts/nav_competition_image_sandbox/opt/ros/noetic}"
if [[ ! -f "$KUAVO_NOETIC_ROOT/setup.bash" && -f /opt/ros/noetic/setup.bash ]]; then
  KUAVO_NOETIC_ROOT=/opt/ros/noetic
fi

if [[ ! -f "$KUAVO_NOETIC_ROOT/setup.bash" ]]; then
  echo "[kuavo_ik_env] ROS1 Noetic setup.bash not found: $KUAVO_NOETIC_ROOT/setup.bash" >&2
  return 1 2>/dev/null || exit 1
fi

DRAKE_LIB_DIR="${DRAKE_LIB_DIR:-/home/zhxm/miniconda3/envs/my_task319_safe/lib/python3.11/site-packages/pydrake/lib}"
if [[ ! -f "$DRAKE_LIB_DIR/libdrake.so" ]]; then
  echo "[kuavo_ik_env] libdrake.so not found in DRAKE_LIB_DIR=$DRAKE_LIB_DIR" >&2
  echo "[kuavo_ik_env] Install with: /home/zhxm/miniconda3/envs/my_task319_safe/bin/python -m pip install drake" >&2
  return 1 2>/dev/null || exit 1
fi
CONDA_SITE_PACKAGES="${CONDA_SITE_PACKAGES:-/home/zhxm/miniconda3/envs/my_task319_safe/lib/python3.11/site-packages}"

KUAVO_COMPAT_LIB_DIR="${KUAVO_COMPAT_LIB_DIR:-/tmp/task319_ros1_compat_libs}"
mkdir -p "$KUAVO_COMPAT_LIB_DIR"

NOETIC_ROOTFS="$(cd "$KUAVO_NOETIC_ROOT/../../.." && pwd)"
KUAVO_ROS1_PYTHON="${KUAVO_ROS1_PYTHON:-$NOETIC_ROOTFS/usr/bin/python3.8}"
if [[ ! -x "$KUAVO_ROS1_PYTHON" ]]; then
  KUAVO_ROS1_PYTHON=python3
fi
link_compat_lib() {
  local soname="$1"
  local src=""
  src="$(find "$NOETIC_ROOTFS/usr/lib" "$NOETIC_ROOTFS/lib" -name "$soname" -print -quit 2>/dev/null || true)"
  if [[ -n "$src" ]]; then
    ln -sf "$src" "$KUAVO_COMPAT_LIB_DIR/$soname"
  fi
}

for lib in \
  libtinyxml2.so.6 \
  libboost_thread.so.1.71.0 \
  libboost_program_options.so.1.71.0 \
  libboost_filesystem.so.1.71.0 \
  libpython3.8.so.1.0 \
  libconsole_bridge.so.0.4 \
  libboost_regex.so.1.71.0 \
  liblog4cxx.so.10 \
  libboost_chrono.so.1.71.0 \
  libapr-1.so.0 \
  libaprutil-1.so.0 \
  libicui18n.so.66 \
  libicuuc.so.66 \
  libicudata.so.66; do
  link_compat_lib "$lib"
done

unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH COLCON_PREFIX_PATH
KUAVO_IK_ENV_NOUNSET_WAS_ON=0
case "$-" in
  *u*) KUAVO_IK_ENV_NOUNSET_WAS_ON=1; set +u ;;
esac
source "$KUAVO_NOETIC_ROOT/setup.bash"

if [[ -f "$KUAVO_WS/devel_mda319/.private/kuavo_msgs/setup.bash" ]]; then
  source "$KUAVO_WS/devel_mda319/.private/kuavo_msgs/setup.bash"
fi
if [[ -f "$KUAVO_WS/devel_mda319/.private/motion_capture_ik/setup.bash" ]]; then
  source "$KUAVO_WS/devel_mda319/.private/motion_capture_ik/setup.bash"
fi
if [[ "$KUAVO_IK_ENV_NOUNSET_WAS_ON" == "1" ]]; then
  set -u
fi

export PATH="$KUAVO_NOETIC_ROOT/bin:$PATH"
export ROS_PACKAGE_PATH="$KUAVO_WS/src/manipulation_nodes:$KUAVO_WS/src:${ROS_PACKAGE_PATH:-}"
export PYTHONPATH="$CONDA_SITE_PACKAGES:$KUAVO_WS/devel_mda319/lib/python3/dist-packages:$KUAVO_NOETIC_ROOT/lib/python3/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$KUAVO_NOETIC_ROOT/lib:$DRAKE_LIB_DIR:$KUAVO_COMPAT_LIB_DIR:$KUAVO_WS/src/manipulation_nodes/motion_capture_ik/lib:$KUAVO_WS/devel_mda319/.private/motion_capture_ik/lib:$KUAVO_WS/devel_mda319/.private/kuavo_msgs/lib:$KUAVO_WS/devel_mda319/lib:$KUAVO_WS/installed/lib:${LD_LIBRARY_PATH:-}"
export ROBOT_VERSION="${ROBOT_VERSION:-43}"
export KUAVO_NOETIC_ROOT NOETIC_ROOTFS KUAVO_ROS1_PYTHON
