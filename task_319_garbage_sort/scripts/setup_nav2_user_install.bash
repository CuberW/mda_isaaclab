#!/usr/bin/env bash
# Source the system ROS 2 environment plus the user-local Nav2 packages unpacked
# under task_319_garbage_sort/output/nav2_user_install/root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_ROOT="${TASK_DIR}/output/nav2_user_install/root"
LOCAL_PREFIX="${LOCAL_ROOT}/opt/ros/jazzy"

source /opt/ros/jazzy/setup.bash

if [ -d "${LOCAL_PREFIX}" ]; then
  export AMENT_PREFIX_PATH="${LOCAL_PREFIX}:${AMENT_PREFIX_PATH:-}"
  export CMAKE_PREFIX_PATH="${LOCAL_PREFIX}:${CMAKE_PREFIX_PATH:-}"
  export PATH="${LOCAL_PREFIX}/bin:${PATH}"
  export LD_LIBRARY_PATH="${LOCAL_PREFIX}/lib:${LOCAL_ROOT}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${LOCAL_PREFIX}/lib/python3.12/site-packages:${PYTHONPATH:-}"
fi
