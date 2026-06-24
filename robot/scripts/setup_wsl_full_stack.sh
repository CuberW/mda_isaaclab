#!/usr/bin/env bash
set -euo pipefail

# Full mature-backend setup for WSL2 Ubuntu 24.04.
# Windows conda remains the light/debug environment; this script prepares the
# CUDA GraspNet + robosuite stack used by full health.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${ROBOT_STACK_VENV:-/opt/robot-stack/venv}"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS_APT_URL="${ROS_APT_URL:-http://packages.ros.org/ros2/ubuntu}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_CUDA_TOOLKIT="${INSTALL_CUDA_TOOLKIT:-auto}"
CUDA_TOOLKIT_PACKAGE="${CUDA_TOOLKIT_PACKAGE:-cuda-toolkit-12-8}"

log() {
  printf '\n==> %s\n' "$*"
}

run() {
  printf '+ %s\n' "$*"
  "$@"
}

apt_install() {
  DEBIAN_FRONTEND=noninteractive run apt-get install -y --no-install-recommends "$@"
}

ensure_ros_apt() {
  if [[ -f /etc/apt/sources.list.d/ros2.list ]]; then
    return
  fi
  log "Configuring ROS2 apt repository (${ROS_DISTRO})"
  apt_install software-properties-common curl gnupg lsb-release ca-certificates
  add-apt-repository -y universe
  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  local codename
  codename="$(. /etc/os-release && echo "${UBUNTU_CODENAME}")"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] ${ROS_APT_URL} ${codename} main" \
    > /etc/apt/sources.list.d/ros2.list
}

install_cuda_toolkit_if_needed() {
  if command -v nvcc >/dev/null 2>&1; then
    log "CUDA compiler already available: $(command -v nvcc)"
    return
  fi
  if [[ "${INSTALL_CUDA_TOOLKIT}" == "0" || "${INSTALL_CUDA_TOOLKIT}" == "false" ]]; then
    log "nvcc is missing and INSTALL_CUDA_TOOLKIT is disabled"
    return
  fi

  if [[ -x "${VENV}/bin/python" ]]; then
    log "Trying pip CUDA compiler package for GraspNet extensions"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    if python -m pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12; then
      local pip_nvcc
      pip_nvcc="$(python - <<'PY'
from pathlib import Path
import site
for root in site.getsitepackages():
    p = Path(root) / "nvidia" / "cuda_nvcc" / "bin" / "nvcc"
    if p.exists():
        print(p)
        break
PY
)"
      if [[ -n "${pip_nvcc}" && -x "${pip_nvcc}" ]]; then
        log "pip nvcc available: ${pip_nvcc}"
        return
      fi
    fi
  fi

  log "Installing apt CUDA toolkit for GraspNet CUDA extensions"
  apt_install wget
  local distro_id="ubuntu2404"
  local pin="/etc/apt/preferences.d/cuda-repository-pin-600"
  if [[ ! -f "${pin}" ]]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/${distro_id}/x86_64/cuda-keyring_1.1-1_all.deb \
      -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
  fi
  run apt-get update
  if ! DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${CUDA_TOOLKIT_PACKAGE}"; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cuda-toolkit
  fi
}

setup_python_env() {
  log "Creating Python environment: ${VENV}"
  apt_install python3 python3-venv python3-pip python3-dev
  mkdir -p "$(dirname "${VENV}")"
  run python3 -m venv --system-site-packages "${VENV}"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  run python -m pip install --upgrade pip "setuptools<80" wheel ninja packaging
  if ! python -m pip install --upgrade torch torchvision --index-url "${TORCH_INDEX_URL}"; then
    python -m pip install --upgrade torch torchvision
  fi
  run python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
  run python -m pip install gdown huggingface_hub modelscope transforms3d pywavefront scikit-image dill h5py cvxopt scikit-learn
  run python -m pip install --no-deps -e "${PROJECT_ROOT}/third_party/graspnetAPI"
  run python -m pip install robosuite
  run python -m pip install "mink>=1.1.1"
}

build_ros_workspace() {
  log "No ROS2 workspace is required by active tasks; skipping"
}

prepare_assets() {
  log "Preparing third-party source and model assets"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  run python "${PROJECT_ROOT}/scripts/setup_full_stack_assets.py"
}

build_graspnet_extensions() {
  log "Building GraspNet CUDA extensions"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  else
    local pip_cuda_home
    pip_cuda_home="$(python - <<'PY'
from pathlib import Path
import site
for root in site.getsitepackages():
    p = Path(root) / "nvidia" / "cuda_nvcc"
    if (p / "bin" / "nvcc").exists():
        print(p)
        break
PY
)"
    if [[ -n "${pip_cuda_home}" ]]; then
      export CUDA_HOME="${CUDA_HOME:-${pip_cuda_home}}"
      export PATH="${CUDA_HOME}/bin:${PATH}"
    fi
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is required to build GraspNet pointnet2. Run stage 'cuda' or set CUDA_HOME." >&2
    exit 1
  fi
  if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    TORCH_CUDA_ARCH_LIST="$(python - <<'PY'
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"{major}.{minor}+PTX")
else:
    print("8.6+PTX")
PY
)"
    export TORCH_CUDA_ARCH_LIST
  fi
  log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  mkdir -p "${PROJECT_ROOT}/third_party/graspnet-baseline/pointnet2/pointnet2"
  touch "${PROJECT_ROOT}/third_party/graspnet-baseline/pointnet2/pointnet2/__init__.py"
  mkdir -p "${PROJECT_ROOT}/third_party/graspnet-baseline/knn/knn_pytorch"
  touch "${PROJECT_ROOT}/third_party/graspnet-baseline/knn/knn_pytorch/__init__.py"
  run python -m pip install --no-build-isolation -e "${PROJECT_ROOT}/third_party/graspnet-baseline/pointnet2"
  run python -m pip install --no-build-isolation -e "${PROJECT_ROOT}/third_party/graspnet-baseline/knn"
}

run_full_health() {
  log "Running full system health"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/graspnet-baseline:${PROJECT_ROOT}/third_party/graspnet-baseline/models:${PROJECT_ROOT}/third_party/graspnet-baseline/pointnet2:${PROJECT_ROOT}/third_party/graspnet-baseline/knn:${PROJECT_ROOT}/third_party/graspnetAPI:${PYTHONPATH:-}"
  run python "${PROJECT_ROOT}/scripts/verify_system_health.py" --full
}

run_base() {
  if [[ "$(id -u)" != "0" ]]; then
    echo "Run this script as root inside WSL, or with sudo." >&2
    exit 1
  fi
  log "Project root: ${PROJECT_ROOT}"
  run apt-get update
  apt_install git git-lfs build-essential cmake ninja-build pkg-config \
    libegl1 libgl1 libglvnd0 libosmesa6 libglew-dev \
    libopenblas-dev libomp-dev libspatialindex-dev
}

run_stage() {
  case "$1" in
    base)
      run_base
      ;;
    ros)
      build_ros_workspace
      ;;
    python)
      setup_python_env
      ;;
    cuda)
      install_cuda_toolkit_if_needed
      ;;
    assets)
      prepare_assets
      ;;
    graspnet)
      build_graspnet_extensions
      ;;
    health)
      run_full_health
      ;;
    all)
      run_base
      build_ros_workspace
      setup_python_env
      install_cuda_toolkit_if_needed
      prepare_assets
      build_graspnet_extensions
      run_full_health
      ;;
    *)
      echo "Unknown stage '$1'. Use: all, base, ros, python, cuda, assets, graspnet, health." >&2
      exit 2
      ;;
  esac
}

main() {
  if [[ $# -eq 0 ]]; then
    run_stage all
    return
  fi
  for stage in "$@"; do
    run_stage "${stage}"
  done
}

main "$@"
