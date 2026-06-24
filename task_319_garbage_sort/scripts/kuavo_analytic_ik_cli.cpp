#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "motion_capture_ik/AnalyticArmIk.hpp"
#include "motion_capture_ik/json.hpp"

namespace {

using json = nlohmann::json;
using motion_capture_ik::AnalyticArmIk;

constexpr std::array<double, 7> kRightLower = {
    -3.14159265358979,
    -3.49065850398866,
    -1.48352986419518,
    -2.61799387799149,
    -1.5707963267949,
    -0.698131700797732,
    -0.698131700797732,
};
constexpr std::array<double, 7> kRightUpper = {
    1.5707963267949,
    0.349065850398866,
    1.48352986419518,
    0.0,
    1.5707963267949,
    1.30899693899575,
    0.698131700797732,
};

Eigen::Matrix3d readRotation(const json& value) {
  Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
  if (value.is_array() && value.size() == 9) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        rot(r, c) = value.at(r * 3 + c).get<double>();
      }
    }
    return rot;
  }
  if (value.is_array() && value.size() == 3) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        rot(r, c) = value.at(r).at(c).get<double>();
      }
    }
    return rot;
  }
  throw std::runtime_error("rotation must be a flat 9-list or 3x3 list");
}

Eigen::Vector3d readVector3(const json& value, const char* name) {
  if (!value.is_array() || value.size() != 3) {
    throw std::runtime_error(std::string(name) + " must be a 3-list");
  }
  return Eigen::Vector3d(value.at(0).get<double>(), value.at(1).get<double>(), value.at(2).get<double>());
}

json vectorJson(const Eigen::Matrix<double, 7, 1>& q) {
  json out = json::array();
  for (int i = 0; i < 7; ++i) {
    out.push_back(q[i]);
  }
  return out;
}

json vector3Json(const Eigen::Vector3d& v) {
  return json::array({v.x(), v.y(), v.z()});
}

json matrix3Json(const Eigen::Matrix3d& m) {
  json out = json::array();
  for (int r = 0; r < 3; ++r) {
    json row = json::array();
    for (int c = 0; c < 3; ++c) {
      row.push_back(m(r, c));
    }
    out.push_back(row);
  }
  return out;
}

Eigen::Isometry3d fkForArm(const std::string& arm, const Eigen::Matrix<double, 7, 1>& q) {
  if (arm != "right") {
    return AnalyticArmIk::FKLeftArm(q);
  }

  Eigen::Matrix<double, 7, 1> q_m = q;
  q_m[1] = -q_m[1];
  q_m[2] = -q_m[2];
  q_m[4] = -q_m[4];
  q_m[5] = -q_m[5];
  const Eigen::Isometry3d fk_m = AnalyticArmIk::FKLeftArm(q_m);
  const Eigen::Matrix3d S = (Eigen::Vector3d(1.0, -1.0, 1.0)).asDiagonal();

  Eigen::Isometry3d fk = Eigen::Isometry3d::Identity();
  fk.linear() = S * fk_m.linear() * S;
  fk.translation() = S * fk_m.translation();
  return fk;
}

bool withinRightLimits(const Eigen::Matrix<double, 7, 1>& q, double margin, json& violations) {
  bool ok = true;
  violations = json::array();
  for (int i = 0; i < 7; ++i) {
    const double lower = kRightLower[static_cast<size_t>(i)] - margin;
    const double upper = kRightUpper[static_cast<size_t>(i)] + margin;
    if (q[i] < lower || q[i] > upper) {
      ok = false;
      violations.push_back({
          {"joint_index", i},
          {"value_rad", q[i]},
          {"lower_rad", kRightLower[static_cast<size_t>(i)]},
          {"upper_rad", kRightUpper[static_cast<size_t>(i)]},
      });
    }
  }
  return ok;
}

}  // namespace

int main() {
  try {
    json request;
    std::cin >> request;

    const std::string arm = request.value("arm", "right");
    const Eigen::Matrix3d rot = readRotation(request.at("rotation"));
    const Eigen::Vector3d pos = readVector3(request.at("position"), "position");
    const double limit_margin = request.value("joint_limit_margin_rad", 1e-5);
    const double max_fk_position_error = request.value("max_fk_position_error_m", -1.0);

    AnalyticArmIk::Options opts;
    opts.outer_pos_iter = request.value("outer_pos_iter", opts.outer_pos_iter);
    opts.wrist_refine_iter = request.value("wrist_refine_iter", opts.wrist_refine_iter);
    opts.wrist_orient_gain = request.value("wrist_orient_gain", opts.wrist_orient_gain);
    opts.wrist_posture_weight = request.value("wrist_posture_weight", opts.wrist_posture_weight);
    opts.outer_break_eps = request.value("outer_break_eps", opts.outer_break_eps);

    Eigen::Matrix<double, 7, 1> q = Eigen::Matrix<double, 7, 1>::Zero();
    bool solved = false;
    if (arm == "right") {
      solved = AnalyticArmIk::SolveRightArm(rot, pos, opts, q);
    } else if (arm == "left") {
      solved = AnalyticArmIk::SolveLeftArm(rot, pos, opts, q);
    } else {
      throw std::runtime_error("arm must be 'left' or 'right'");
    }

    json violations = json::array();
    const bool limits_ok = (arm == "right") ? withinRightLimits(q, limit_margin, violations) : true;
    const Eigen::Isometry3d fk = fkForArm(arm, q);
    const double fk_position_error = (fk.translation() - pos).norm();
    const bool fk_ok = max_fk_position_error <= 0.0 || fk_position_error <= max_fk_position_error;
    json response = {
        {"success", solved && limits_ok && fk_ok},
        {"solver", "kuavo_official_analytic_arm_ik"},
        {"arm", arm},
        {"q", vectorJson(q)},
        {"within_joint_limits", limits_ok},
        {"joint_limit_violations", violations},
        {"fk_position", vector3Json(fk.translation())},
        {"fk_rotation", matrix3Json(fk.linear())},
        {"fk_position_error_m", fk_position_error},
        {"max_fk_position_error_m", max_fk_position_error},
        {"fk_position_error_ok", fk_ok},
    };
    if (!solved) {
      response["error_reason"] = "analytic_solver_failed";
    } else if (!limits_ok) {
      response["error_reason"] = "joint_limit_violation";
    } else if (!fk_ok) {
      response["error_reason"] = "fk_position_error_exceeded";
    }
    std::cout << response.dump() << std::endl;
    return (solved && limits_ok) ? 0 : 2;
  } catch (const std::exception& exc) {
    json response = {
        {"success", false},
        {"solver", "kuavo_official_analytic_arm_ik"},
        {"error_reason", exc.what()},
    };
    std::cout << response.dump() << std::endl;
    return 1;
  }
}
