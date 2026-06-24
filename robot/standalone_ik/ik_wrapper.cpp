#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>

// Keep the bridge ROS-free even if future upstream headers pull logging macros.
#ifndef ROS_INFO
#define ROS_INFO(...)
#define ROS_WARN(...)
#define ROS_ERROR(...)
#define ROS_INFO_STREAM(...)
#define ROS_WARN_STREAM(...)
#define ROS_ERROR_STREAM(...)
#endif

#include "humanoid_wheel_interface/FactoryFunctions.h"
#include "humanoid_wheel_interface/ManipulatorModelInfo.h"
#include "humanoid_wheel_interface/common/Types.h"
#include "humanoid_wheel_interface/motion_planner/InverseKinematics.h"

namespace {

using ocs2::PinocchioInterface;
using ocs2::mobile_manipulator::InverseKinematics;
using ocs2::mobile_manipulator::ManipulatorModelInfo;
using ocs2::mobile_manipulator::ManipulatorModelType;
using ocs2::mobile_manipulator::createManipulatorModelInfo;
using ocs2::mobile_manipulator::createPinocchioInterface;
using ocs2::mobile_manipulator::vector3_t;
using ocs2::vector_t;

struct TaskInfoConfig {
  ManipulatorModelType manipulator_model_type;
  std::vector<std::string> remove_joints;
  std::string base_frame;
  std::string torso_frame;
  std::vector<std::string> ee_frames;
};

struct IkHandle {
  std::shared_ptr<PinocchioInterface> pinocchio;
  std::shared_ptr<ManipulatorModelInfo> info;
  InverseKinematics solver;
  int arm_index = 0;
  bool is_whole_body = true;
};

std::vector<std::string> loadIndexedStringList(const boost::property_tree::ptree& pt, const std::string& prefix) {
  std::vector<std::string> values;
  for (std::size_t index = 0;; ++index) {
    const auto key = prefix + ".[" + std::to_string(index) + "]";
    const auto value = pt.get_optional<std::string>(key);
    if (!value.has_value()) {
      break;
    }
    values.push_back(*value);
  }
  return values;
}

TaskInfoConfig loadTaskInfoConfig(const std::string& task_info_path) {
  boost::property_tree::ptree pt;
  boost::property_tree::read_info(task_info_path, pt);

  const auto model_type_value = pt.get<int>("model_information.manipulatorModelType");
  TaskInfoConfig config{
      static_cast<ManipulatorModelType>(model_type_value),
      loadIndexedStringList(pt, "model_information.removeJoints"),
      pt.get<std::string>("model_information.baseFrame"),
      pt.get<std::string>("model_information.torsoFrame"),
      loadIndexedStringList(pt, "model_information.eeFrames"),
  };

  if (config.ee_frames.empty()) {
    throw std::runtime_error("task.info does not define any end-effector frames.");
  }

  return config;
}

std::shared_ptr<PinocchioInterface> createModel(const std::string& urdf_path, const TaskInfoConfig& config) {
  PinocchioInterface interface = config.remove_joints.empty()
                                     ? createPinocchioInterface(urdf_path, config.manipulator_model_type)
                                     : createPinocchioInterface(urdf_path, config.manipulator_model_type, config.remove_joints);
  return std::make_shared<PinocchioInterface>(std::move(interface));
}

std::shared_ptr<ManipulatorModelInfo> createModelInfo(const PinocchioInterface& interface, const TaskInfoConfig& config) {
  auto info = std::make_shared<ManipulatorModelInfo>(
      createManipulatorModelInfo(interface, config.manipulator_model_type, config.base_frame, config.torso_frame, config.ee_frames));
  return info;
}

void writeFailureOutputs(double* out_best_linear_error, double* out_best_angular_error) {
  if (out_best_linear_error != nullptr) {
    *out_best_linear_error = std::numeric_limits<double>::infinity();
  }
  if (out_best_angular_error != nullptr) {
    *out_best_angular_error = std::numeric_limits<double>::infinity();
  }
}

}  // namespace

extern "C" {

void* ik_create(const char* urdf_path, const char* task_info_path, int arm_index, int is_whole_body) {
  if (urdf_path == nullptr || task_info_path == nullptr) {
    return nullptr;
  }

  if (arm_index != 0 && arm_index != 1) {
    return nullptr;
  }

  try {
    const TaskInfoConfig config = loadTaskInfoConfig(task_info_path);
    if (static_cast<std::size_t>(arm_index) >= config.ee_frames.size()) {
      return nullptr;
    }

    auto handle = std::make_unique<IkHandle>();
    handle->pinocchio = createModel(urdf_path, config);
    handle->info = createModelInfo(*handle->pinocchio, config);
    handle->arm_index = arm_index;
    handle->is_whole_body = (is_whole_body != 0);
    handle->solver.setParam(handle->pinocchio, handle->info);
    return static_cast<void*>(handle.release());
  } catch (...) {
    return nullptr;
  }
}

int ik_solve(void* handle,
             const double* target_pose_6d,
             const double* initial_q,
             int initial_q_len,
             double* out_q,
             int out_q_len,
             double* out_best_linear_error,
             double* out_best_angular_error) {
  writeFailureOutputs(out_best_linear_error, out_best_angular_error);

  if (handle == nullptr || target_pose_6d == nullptr || out_q == nullptr) {
    return -1;
  }

  auto* ik_handle = static_cast<IkHandle*>(handle);
  if (!ik_handle->pinocchio || !ik_handle->info) {
    return -2;
  }

  const int state_dim = static_cast<int>(ik_handle->info->stateDim);
  const int arm_dim = static_cast<int>(ik_handle->info->armDim);

  if (out_q_len < arm_dim) {
    return -3;
  }
  std::fill(out_q, out_q + out_q_len, 0.0);

  vector_t q0(state_dim);
  if (initial_q != nullptr && initial_q_len >= state_dim) {
    q0 = Eigen::Map<const vector_t>(initial_q, state_dim);
  } else if (initial_q == nullptr || initial_q_len == 0) {
    q0.setZero();
  } else {
    return -4;
  }

  const vector3_t target_xyz(target_pose_6d[0], target_pose_6d[1], target_pose_6d[2]);
  const vector3_t target_euler_zyx(target_pose_6d[3], target_pose_6d[4], target_pose_6d[5]);

  try {
    vector_t solution = ik_handle->is_whole_body
                            ? ik_handle->solver.computeWholeBodyIK(q0, ik_handle->arm_index, target_xyz, target_euler_zyx, true)
                            : ik_handle->solver.computeHandOnlyIK(q0, ik_handle->arm_index, target_xyz, target_euler_zyx, true);

    if (solution.size() < arm_dim) {
      return -5;
    }

    const vector_t q_best = solution.tail(arm_dim);
    std::memcpy(out_q, q_best.data(), static_cast<std::size_t>(arm_dim) * sizeof(double));

    if (out_best_linear_error != nullptr) {
      *out_best_linear_error = ik_handle->solver.getBestLinearError();
    }
    if (out_best_angular_error != nullptr) {
      *out_best_angular_error = ik_handle->solver.getBestAngularError();
    }

    return 0;
  } catch (...) {
    return -6;
  }
}

int ik_get_dof_name(void* handle, int index, char* out_name, int out_name_len) {
  if (handle == nullptr || out_name == nullptr || out_name_len <= 0 || index < 0) {
    return -1;
  }

  auto* ik_handle = static_cast<IkHandle*>(handle);
  if (!ik_handle->info) {
    return -2;
  }
  if (static_cast<std::size_t>(index) >= ik_handle->info->dofNames.size()) {
    return -3;
  }

  const auto& name = ik_handle->info->dofNames[static_cast<std::size_t>(index)];
  const std::size_t copy_len = std::min(static_cast<std::size_t>(out_name_len - 1), name.size());
  std::memcpy(out_name, name.data(), copy_len);
  out_name[copy_len] = '\0';
  return static_cast<int>(copy_len);
}

void ik_destroy(void* handle) {
  if (handle == nullptr) {
    return;
  }

  auto* ik_handle = static_cast<IkHandle*>(handle);
  delete ik_handle;
}

}  // extern "C"
