"""End-effector specifications for task control code.

The task pipelines should not need to remember every weld, pinch frame,
finger body, and actuator name. These specs keep robot-specific names local to
the control seam while preserving the existing MuJoCo behavior.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EndEffectorSpec:
    name: str
    weld_name: str
    pinch_body: str
    primary_gripper_body: str
    extra_gripper_bodies: tuple[str, ...] = ()
    finger_bodies: tuple[str, ...] = ()
    finger_tip_bodies: tuple[str, ...] = ()
    finger_pad_geoms: tuple[str, ...] = ()
    actuator_name: str = ""

    @property
    def gripper_bodies(self) -> tuple[str, ...]:
        return (self.primary_gripper_body, *self.extra_gripper_bodies)


STRETCH_GRIPPER = EndEffectorSpec(
    name="stretch_gripper",
    weld_name="grasp_stretch",
    pinch_body="link_gripper_finger_left",
    primary_gripper_body="link_gripper_finger_left",
    extra_gripper_bodies=(
        "link_gripper_finger_right",
        "rubber_tip_left",
        "rubber_tip_right",
    ),
    finger_bodies=(
        "link_gripper_finger_left",
        "link_gripper_finger_right",
        "rubber_tip_left",
        "rubber_tip_right",
    ),
    finger_tip_bodies=("rubber_tip_left", "rubber_tip_right"),
)

KUAVO_WHEEL_RIGHT_GRIPPER = EndEffectorSpec(
    name="kuavo_wheel_right_gripper",
    weld_name="grasp_kuavo_right",
    pinch_body="r_pinch",
    primary_gripper_body="r_pinch",
    extra_gripper_bodies=("r_f_fingers", "r_b_fingers"),
    finger_bodies=("r_f_fingers", "r_b_fingers"),
    finger_tip_bodies=("r_f_fingers", "r_b_fingers"),
    finger_pad_geoms=("r_f_fingers_pad", "r_b_fingers_pad"),
    actuator_name="r_fingers_actuator",
)

DUAL_PANDA_LEFT_GRIPPER = EndEffectorSpec(
    name="dual_panda_left_gripper",
    weld_name="grasp_left",
    pinch_body="l_pinch",
    primary_gripper_body="l_pinch",
    extra_gripper_bodies=("l_finger_l", "l_finger_r"),
    finger_bodies=("l_finger_l", "l_finger_r"),
    actuator_name="ml_grip",
)

DUAL_PANDA_RIGHT_GRIPPER = EndEffectorSpec(
    name="dual_panda_right_gripper",
    weld_name="grasp_right",
    pinch_body="r_pinch",
    primary_gripper_body="r_pinch",
    extra_gripper_bodies=("r_finger_l", "r_finger_r"),
    finger_bodies=("r_finger_l", "r_finger_r"),
    actuator_name="mr_grip",
)

DUAL_PANDA_GRIPPERS = {
    "left": DUAL_PANDA_LEFT_GRIPPER,
    "right": DUAL_PANDA_RIGHT_GRIPPER,
}

__all__ = [
    "DUAL_PANDA_GRIPPERS",
    "DUAL_PANDA_LEFT_GRIPPER",
    "DUAL_PANDA_RIGHT_GRIPPER",
    "EndEffectorSpec",
    "KUAVO_WHEEL_RIGHT_GRIPPER",
    "STRETCH_GRIPPER",
]
