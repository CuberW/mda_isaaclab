"""
Execution Hub - Motion planning, IK, grasping, and navigation.

Components:
  - IK Solver: MinkIKSolver (MuJoCo-native, via mink library)
  - Motion Planner: Trajectory generation (linear, spline, RRT-style)
  - Grasp Executor: Grasp planning and execution
  - Navigation Controller: Differential drive navigation (for task 3.19)
  - Collision Monitor: Real-time collision checking
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from robot_common.infra.logging import logger

# Suppress Mink's noisy joint-limit warnings during IK convergence
import logging
logging.getLogger("root").setLevel(logging.WARNING)

from robot_common.execution.grasp import GraspManager  # noqa: E402
from robot_common.execution.mink_ik import MinkIKSolver  # noqa: E402
from robot_common.execution.robosuite_wbc import RobosuiteWBCController, WBCCommand  # noqa: E402
from robot_common.execution.dual_arm_wbc import DualArmWBCController, WBCStepInfo, WBC_SOURCE  # noqa: E402


# ── IK Solver ────────────────────────────────────────────────
@dataclass
class IKSolution:
    """Result of inverse kinematics computation.
    NOTE: Prefer MinkIKSolver for new code. NumericalIKSolver is deprecated.
    """
    success: bool = False
    joint_positions: np.ndarray = None
    position_error: float = 0.0
    iterations: int = 0

    def __post_init__(self):
        if self.joint_positions is None:
            self.joint_positions = np.array([])


class AnalyticalIKSolver:
    """Analytical IK solver for simple arm geometries (4-DoF Dobot-style).

    For more complex arms (7-DoF Panda), use numerical IK.
    """

    def __init__(self, link_lengths: List[float] = None):
        """
        Args:
            link_lengths: List of link lengths [l1, l2, l3, ...]
        """
        self.link_lengths = link_lengths or [0.3, 0.3, 0.15]  # Default lengths

    def solve(self, target_pos: np.ndarray,
              current_joints: np.ndarray = None,
              orientation: float = 0.0) -> IKSolution:
        """Solve IK for a planar 4-DoF arm (RRR+R base rotation).

        Args:
            target_pos: Target (x, y, z) in arm base frame
            current_joints: Current joint positions for nearest-solution selection
            orientation: Desired end-effector rotation (radians)

        Returns:
            IKSolution with joint positions [base, shoulder, elbow, wrist]
        """
        x, y, z = target_pos

        # 4-DoF: base rotation + 3-DoF planar arm
        # Base rotation: atan2(y, x)
        base = np.arctan2(y, x)

        # Distance in base-frame XY plane
        r = np.sqrt(x**2 + y**2)

        # 3-DoF planar arm in (r, z) plane
        l1, l2, l3 = self.link_lengths[0], self.link_lengths[1], self.link_lengths[2]

        # Wrist position before end-effector
        wx = r - l3 * np.cos(orientation)
        wz = z - l3 * np.sin(orientation)

        # Distance from shoulder to wrist
        d = np.sqrt(wx**2 + wz**2)

        if d > l1 + l2 or d < abs(l1 - l2):
            # Target unreachable
            return IKSolution(success=False)

        # Law of cosines
        cos_elbow = (d**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
        elbow = np.arccos(cos_elbow)  # Always elbow-up for simplicity

        # Shoulder angle
        alpha = np.arctan2(wz, wx)
        beta = np.arctan2(l2 * np.sin(elbow), l1 + l2 * np.cos(elbow))
        shoulder = alpha - beta

        # Wrist angle
        wrist = orientation - shoulder - elbow

        joint_positions = np.array([base, shoulder, elbow, wrist])

        # Check if we should use alternate elbow configuration
        if current_joints is not None:
            # Elbow-down alternative
            alt_base = base
            alt_shoulder = alpha + beta
            alt_elbow = -elbow
            alt_wrist = orientation - alt_shoulder - alt_elbow
            alt_solution = np.array([alt_base, alt_shoulder, alt_elbow, alt_wrist])

            # Choose nearest to current joints
            if np.linalg.norm(alt_solution - current_joints) < np.linalg.norm(joint_positions - current_joints):
                joint_positions = alt_solution

        # Compute end-effector position error
        ee_pos = self.forward_kinematics(joint_positions)[:3]
        target_full = np.array([x, y, z])
        error = np.linalg.norm(ee_pos - target_full)

        return IKSolution(
            success=True,
            joint_positions=joint_positions,
            position_error=float(error),
            iterations=1,
        )

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """Compute end-effector pose from joint angles."""
        base, shoulder, elbow, wrist = joints[0], joints[1], joints[2], joints[3]
        l1, l2, l3 = self.link_lengths

        # Position in (r, z) plane
        r = l1 * np.cos(shoulder) + l2 * np.cos(shoulder + elbow) + l3 * np.cos(shoulder + elbow + wrist)
        z = l1 * np.sin(shoulder) + l2 * np.sin(shoulder + elbow) + l3 * np.sin(shoulder + elbow + wrist)

        # Convert to (x, y, z)
        x = r * np.cos(base)
        y = r * np.sin(base)

        # Orientation
        orientation = shoulder + elbow + wrist

        return np.array([x, y, z, orientation])


class NumericalIKSolver:
    """Numerical IK solver using Jacobian pseudoinverse (for 6-7 DoF arms)."""

    def __init__(self, forward_kinematics_fn, jacobian_fn,
                 max_iterations: int = 100, tolerance: float = 1e-4,
                 damping: float = 0.1):
        self.fk = forward_kinematics_fn
        self.jacobian = jacobian_fn
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping

    def solve(self, target_pose: np.ndarray,
              initial_joints: np.ndarray) -> IKSolution:
        """Solve IK numerically.

        Args:
            target_pose: Target end-effector pose [x, y, z, roll, pitch, yaw]
            initial_joints: Initial guess for joint positions
        """
        q = initial_joints.copy()
        for i in range(self.max_iterations):
            current_pose = self.fk(q)
            error = target_pose - current_pose
            if np.linalg.norm(error[:3]) < self.tolerance:
                return IKSolution(success=True, joint_positions=q,
                                  position_error=float(np.linalg.norm(error[:3])),
                                  iterations=i + 1)

            J = self.jacobian(q)
            # Damped least squares
            n = J.shape[1]
            J_pinv = J.T @ np.linalg.inv(J @ J.T + self.damping * np.eye(J.shape[0]))
            dq = J_pinv @ error
            q += dq

        return IKSolution(success=False, joint_positions=q,
                          position_error=float(np.linalg.norm(
                              target_pose[:3] - self.fk(q)[:3])),
                          iterations=self.max_iterations)


# ── Trajectory Generator ─────────────────────────────────────
class TrajectoryGenerator:
    """Generate smooth trajectories between waypoints."""

    @staticmethod
    def linear_interpolation(start: np.ndarray, end: np.ndarray,
                             num_steps: int) -> np.ndarray:
        """Linear interpolation in joint space."""
        t = np.linspace(0, 1, num_steps)
        return start + np.outer(t, end - start)

    @staticmethod
    def cubic_spline(start: np.ndarray, end: np.ndarray,
                     num_steps: int,
                     start_vel: np.ndarray = None,
                     end_vel: np.ndarray = None) -> np.ndarray:
        """Cubic spline interpolation for smooth velocity profiles."""
        if start_vel is None:
            start_vel = np.zeros_like(start)
        if end_vel is None:
            end_vel = np.zeros_like(end)

        trajectory = np.zeros((num_steps, len(start)))
        for i in range(num_steps):
            t = i / (num_steps - 1)
            # Hermite cubic interpolation
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2

            trajectory[i] = (h00 * start + h10 * start_vel +
                             h01 * end + h11 * end_vel)

        return trajectory

    @staticmethod
    def cartesian_linear(start_pos: np.ndarray, end_pos: np.ndarray,
                         num_steps: int) -> np.ndarray:
        """Linear trajectory in Cartesian space."""
        return TrajectoryGenerator.linear_interpolation(start_pos, end_pos, num_steps)

    @staticmethod
    def cartesian_arc(start: np.ndarray, end: np.ndarray,
                      via_point: np.ndarray, num_steps: int) -> np.ndarray:
        """Arc trajectory through a via point (quadratic Bezier)."""
        t = np.linspace(0, 1, num_steps)
        trajectory = np.zeros((num_steps, len(start)))
        for i, ti in enumerate(t):
            trajectory[i] = ((1 - ti)**2 * start +
                             2 * (1 - ti) * ti * via_point +
                             ti**2 * end)
        return trajectory


# ── Navigation Controller ────────────────────────────────────
class DiffDriveNavigator:
    """Differential drive navigation controller for wheeled robots.

    Used for Task 3.19 (Stretch robot).
    Implements go-to-goal with obstacle avoidance.
    """

    def __init__(self, wheel_radius: float = 0.1, wheel_base: float = 0.5,
                 max_linear_vel: float = 0.5, max_angular_vel: float = 1.0):
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel

    def compute_velocity(self, current_pose: np.ndarray,
                         goal_position: np.ndarray,
                         obstacles: List[np.ndarray] = None) -> Tuple[float, float]:
        """Compute (linear_vel, angular_vel) for go-to-goal.

        Args:
            current_pose: (x, y, theta) current robot pose
            goal_position: (x, y) target position
            obstacles: List of obstacle (x, y) positions

        Returns:
            (linear_velocity, angular_velocity)
        """
        cx, cy, ctheta = current_pose
        gx, gy = goal_position

        # Vector to goal
        dx = gx - cx
        dy = gy - cy
        distance = np.sqrt(dx**2 + dy**2)
        goal_angle = np.arctan2(dy, dx)

        # Angle error (normalized to [-pi, pi])
        angle_error = goal_angle - ctheta
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        # Obstacle avoidance using simple repulsive field
        if obstacles and len(obstacles) > 0:
            for obs in obstacles:
                ox, oy = obs[0], obs[1]
                odx = cx - ox
                ody = cy - oy
                obs_dist = np.sqrt(odx**2 + ody**2)
                if obs_dist < 0.5:  # 50cm safety radius
                    # Add repulsive force
                    rep_angle = np.arctan2(ody, odx)
                    angle_error += 0.5 * (np.pi - abs(angle_error - rep_angle)) * np.sign(angle_error - rep_angle)

        # Proportional control
        if distance < 0.05:  # 5cm threshold
            linear_vel = 0.0
            angular_vel = 0.0
        elif abs(angle_error) > 0.2:  # Rotate first
            linear_vel = 0.0
            angular_vel = np.clip(2.0 * angle_error, -self.max_angular_vel, self.max_angular_vel)
        else:
            linear_vel = np.clip(0.5 * distance, 0.0, self.max_linear_vel)
            angular_vel = np.clip(1.5 * angle_error, -self.max_angular_vel, self.max_angular_vel)

        return linear_vel, angular_vel

    def wheel_velocities(self, linear_vel: float, angular_vel: float) -> Tuple[float, float]:
        """Convert body velocities to wheel velocities."""
        v_left = (linear_vel - angular_vel * self.wheel_base / 2) / self.wheel_radius
        v_right = (linear_vel + angular_vel * self.wheel_base / 2) / self.wheel_radius
        return v_left, v_right


# ── Collision Monitor ────────────────────────────────────────
class CollisionMonitor:
    """Monitor for self-collision and environment collision."""

    def __init__(self, env):
        self.env = env
        self.robot_link_names: List[str] = []

    def set_robot_links(self, link_names: List[str]):
        """Set names of robot body links for self-collision checking."""
        self.robot_link_names = link_names

    def check_self_collision(self) -> bool:
        """Check if the robot is in self-collision."""
        return self.env.check_self_collision()

    def check_environment_collision(self, safe_distance: float = 0.01) -> bool:
        """Check if robot is too close to environment objects."""
        contacts = self.env.get_contacts("")
        for c in contacts:
            if c["distance"] < -safe_distance:
                return True
        return False

    def is_safe(self) -> bool:
        """Check if current state is collision-free."""
        return not self.check_self_collision() and not self.check_environment_collision()
