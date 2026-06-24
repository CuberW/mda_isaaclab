"""StretchGarbageController — low-level Stretch robot actuator control."""

import numpy as np

from control import MuJoCoEnv
from planning import DiffDriveNavigator


class StretchGarbageController:
    """Controller for the Stretch robot in garbage sorting task."""

    # Actuator indices (from stretch.xml tendon + actuator setup)
    FORWARD_IDX = 0
    TURN_IDX = 1
    LIFT_IDX = 2
    ARM_EXTEND_IDX = 3
    WRIST_YAW_IDX = 4
    GRIP_IDX = 5
    HEAD_PAN_IDX = 6
    HEAD_TILT_IDX = 7
    OPEN_GRIP_CTRL = 0.035
    CLOSED_GRIP_CTRL = -0.005

    def __init__(self, env: MuJoCoEnv):
        self.env = env
        self.nu = env.nu
        self._navigator = DiffDriveNavigator(
            wheel_radius=0.05, wheel_base=0.34,
            max_linear_vel=0.3, max_angular_vel=0.8,
        )
        self.max_ctrl_delta = 0.08
        self.base_linear_ctrl_gain = 2.4
        self.base_turn_ctrl_gain = 1.0

    def get_base_pose(self) -> np.ndarray:
        """Get base (x, y, theta) from MuJoCo state."""
        # Stretch base is a free joint
        pos = self.env.get_body_position("base_link")
        quat = self.env.get_body_quat("base_link")
        # Quaternion to yaw
        w, x, y, z = quat
        theta = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([pos[0], pos[1], theta])

    def get_arm_end_effector_pos(self) -> np.ndarray:
        """Get end effector (gripper) position in world frame."""
        # The gripper position from a known body
        try:
            return self.env.get_body_position("link_gripper_finger_left")
        except Exception:
            return self.env.get_site_position("link_gripper_finger_left")

    def set_control(self, ctrl: np.ndarray):
        """Set all actuator controls with a small rate limit."""
        current = self.env.data.ctrl.copy()
        target = np.asarray(ctrl, dtype=float).copy()
        delta = np.clip(target - current, -self.max_ctrl_delta, self.max_ctrl_delta)
        self.env.set_control(current + delta)

    def set_position_targets(self, ctrl: np.ndarray):
        """Set position-style actuator targets directly while preserving trace limits.

        Wheel velocity commands are rate-limited through ``set_control``. The
        Stretch arm/head actuators are position targets; rate-limiting them once
        and then stepping the simulator leaves the target half-updated, which is
        the source of the visible "no motion, then jump" behavior.
        """
        target = np.asarray(ctrl, dtype=float).copy()
        self.env.set_control(target)

    def move_base(self, forward_vel: float, turn_vel: float):
        """Set base velocity, preserving arm state.

        Positive command velocity should move the base in +x; the actuator
        itself is reversed in this MJCF scene.
        """
        # Preserve current arm ctrl values
        current = self.env.data.ctrl.copy()
        current[self.FORWARD_IDX] = float(np.clip(
            -forward_vel * self.base_linear_ctrl_gain,
            -1.0,
            1.0,
        ))
        # Positive turn control rotates the model clockwise, opposite the
        # yaw convention returned by get_base_pose().
        current[self.TURN_IDX] = float(np.clip(
            -turn_vel * self.base_turn_ctrl_gain,
            -1.0,
            1.0,
        ))
        self.set_control(current)

    def stop_base(self):
        """Immediately zero wheel commands while preserving arm/head targets."""
        current = self.env.data.ctrl.copy()
        current[self.FORWARD_IDX] = 0.0
        current[self.TURN_IDX] = 0.0
        self.env.set_control(current)

    def move_arm(self, lift: float = None, extend: float = None,
                 wrist_yaw: float = None, grip: float = None,
                 preserve_base: bool = False):
        """Set arm joint positions.

        Pass None to keep the current value for that joint. Base wheel
        commands are zeroed unless ``preserve_base`` is explicitly requested.
        """
        current = self.env.data.ctrl.copy()
        if not preserve_base:
            current[self.FORWARD_IDX] = 0.0
            current[self.TURN_IDX] = 0.0
        if lift is not None:
            current[self.LIFT_IDX] = np.clip(lift, -0.4, 0.5)
        if extend is not None:
            current[self.ARM_EXTEND_IDX] = np.clip(extend, 0.05, 0.5)
        if wrist_yaw is not None:
            current[self.WRIST_YAW_IDX] = wrist_yaw
        if grip is not None:
            current[self.GRIP_IDX] = grip
        self.set_position_targets(current)

    def look_at(self, pan: float = 0.0, tilt: float = 0.0):
        """Control head camera direction, preserving arm/base state."""
        current = self.env.data.ctrl.copy()
        current[self.FORWARD_IDX] = 0.0
        current[self.TURN_IDX] = 0.0
        current[self.HEAD_PAN_IDX] = pan
        current[self.HEAD_TILT_IDX] = tilt
        self.set_position_targets(current)

    def open_gripper(self):
        """Open the gripper fully (preserves arm position)."""
        self.move_arm(grip=self.OPEN_GRIP_CTRL)

    def close_gripper(self):
        """Close the gripper (preserves arm position)."""
        self.move_arm(grip=self.CLOSED_GRIP_CTRL)
