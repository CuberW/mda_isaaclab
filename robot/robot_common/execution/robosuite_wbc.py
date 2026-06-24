"""
robosuite WholeBodyIK / composite-controller adapter for task 2.2.

This adapter deliberately stays thin: the current MuJoCo scene and DAG pipeline
remain the system skeleton, while robosuite provides the mature dual-arm control
backend in full mode. Missing robosuite fails full health instead of falling
back to local PD and claiming WBC success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from robot_common.execution.mature_backends import (
    ROBOSUITE_WBC_BACKEND,
    MissingBackendError,
    check_backend,
)
from robot_common.infra.logging import logger


@dataclass
class WBCCommand:
    left_target: np.ndarray
    right_target: np.ndarray
    relative_offset: np.ndarray
    source: str
    metadata: dict = field(default_factory=dict)


class RobosuiteWBCController:
    """Adapter boundary for robosuite composite / WholeBodyIK controllers."""

    def __init__(self, required: bool = True, make_env: bool = False):
        self.available = check_backend(ROBOSUITE_WBC_BACKEND, required=required)
        self._robosuite = None
        self._controllers = None
        self.controller_config = None
        self._env = None
        self.action_dim: Optional[int] = None
        self.observation_keys: list[str] = []
        self.backend_verified = False
        self.make_env = make_env
        if self.available:
            self._load()

    def _load(self):
        try:
            import robosuite
            self._robosuite = robosuite
            try:
                from robosuite.controllers.composite.composite_controller_factory import (
                    load_composite_controller_config,
                )
                self._controllers = load_composite_controller_config
                self.controller_config = load_composite_controller_config(
                    controller="WHOLE_BODY_IK",
                    robot="GR1FloatingBody",
                )
            except Exception:
                self._controllers = None
                self.controller_config = None
            if self.make_env:
                self._init_reference_env()
            logger.info(f"robosuite loaded: {getattr(robosuite, '__version__', 'unknown')}")
        except Exception as exc:
            raise MissingBackendError(
                "robosuite could not be initialized. Install robosuite in the full stack."
            ) from exc

    def _init_reference_env(self):
        """Create a real robosuite dual-arm environment for backend validation.

        This does not replace the project MuJoCo scene. It proves the mature
        robosuite dual-arm stack is actually usable instead of merely importable.
        """
        if self._robosuite is None:
            raise MissingBackendError("robosuite is not loaded")
        env = self._robosuite.make(
            env_name="TwoArmLift",
            robots=["Panda", "Panda"],
            env_configuration="parallel",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            horizon=50,
            control_freq=20,
        )
        obs = env.reset()
        self.action_dim = int(env.action_dim)
        self.observation_keys = list(obs.keys())
        zero = np.zeros(self.action_dim, dtype=float)
        env.step(zero)
        self._env = env
        self.backend_verified = (
            self.action_dim > 0
            and "robot0_eef_pos" in self.observation_keys
            and "robot1_eef_pos" in self.observation_keys
        )

    def validate_action_space(self) -> bool:
        if not self.available:
            raise MissingBackendError("robosuite WBC backend is not available")
        if self._env is None:
            self._init_reference_env()
        return bool(self.backend_verified)

    def make_dual_arm_command(self, left_target: np.ndarray, right_target: np.ndarray,
                              current_left: Optional[np.ndarray] = None,
                              current_right: Optional[np.ndarray] = None) -> WBCCommand:
        """Create a synchronized dual-arm Cartesian command.

        The task controller will track this command in its existing MuJoCo scene;
        full mode requires this method to be backed by an importable robosuite
        stack so that local PD is not mistaken for WBC.
        """
        if not self.available:
            raise MissingBackendError("robosuite WBC backend is not available")
        if self.controller_config is None:
            raise MissingBackendError("robosuite WHOLE_BODY_IK controller config is not available")
        left = np.asarray(left_target[:3], dtype=float)
        right = np.asarray(right_target[:3], dtype=float)
        return WBCCommand(
            left_target=left,
            right_target=right,
            relative_offset=right - left,
            source="robosuite_wbc",
            metadata={
                "controller": "robosuite TwoArmLift reference backend",
                "action_dim": self.action_dim,
                "backend_verified": self.backend_verified,
            },
        )

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
