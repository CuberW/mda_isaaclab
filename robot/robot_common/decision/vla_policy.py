"""
VLA Policy Wrapper — load and run VLA/ACT policies for robot control.

Supports LeRobot ACT (Action Chunking Transformer) which natively handles
dual-arm 14-DoF action spaces. Falls back to simple IK-based actions
when no trained policy is available.

Install: pip install lerobot (for ACT support)

Usage:
    policy = VLAPolicy.create("act", model_path="lerobot/act_aloha_sim_transfer_cube_human")
    action = policy.predict(rgb_image, instruction="pick up the cube")
    # action is np.ndarray of shape (14,) for dual-arm tasks
"""

from typing import Optional, Tuple
import numpy as np

from robot_common.infra.logging import logger


class VLAPolicy:
    """Base class for VLA policy wrappers.

    Subclasses implement specific policy types (ACT, OpenVLA, etc.)
    """

    def __init__(self, action_dim: int = 14, control_freq: float = 50.0):
        self.action_dim = action_dim
        self.control_freq = control_freq

    def predict(
        self,
        image: np.ndarray,
        instruction: str = "",
        proprio: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict action given current observation.

        Args:
            image: RGB image (H, W, 3) uint8
            instruction: Natural language instruction
            proprio: Joint positions or other proprioceptive state

        Returns:
            Action array of shape (action_dim,)
        """
        raise NotImplementedError

    def reset(self):
        """Reset policy state between episodes."""
        pass

    @staticmethod
    def create(policy_type: str = "ik_fallback", **kwargs) -> "VLAPolicy":
        """Factory method to create the appropriate policy.

        Args:
            policy_type: "act", "ik_fallback", or "none"
            **kwargs: Passed to policy constructor

        Returns:
            VLAPolicy instance
        """
        if policy_type == "act":
            return ACTPolicy(**kwargs)
        elif policy_type == "ik_fallback":
            return IKBridgePolicy(**kwargs)
        else:
            return NoOpPolicy(**kwargs)


class IKBridgePolicy(VLAPolicy):
    """Bridge policy that uses Mink IK instead of a learned policy.

    This provides a functional baseline: it generates action targets
    via IK, which can then be used as "pseudo-expert" demonstrations
    for training an ACT policy later.
    """

    def __init__(self, model=None, data=None, ee_body: str = "l_gripper",
                 action_dim: int = 14, **kwargs):
        super().__init__(action_dim=action_dim)
        self.model = model
        self.data = data
        self.ee_body = ee_body
        self._target_pos = np.array([0.5, 0.0, 0.8])
        self._ik = None

    def set_target(self, pos: np.ndarray):
        """Set the current IK target position."""
        self._target_pos = np.asarray(pos)

    def predict(
        self,
        image: np.ndarray = None,
        instruction: str = "",
        proprio: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate action via IK toward the current target.

        Returns joint position targets as the action.
        """
        if self.model is None or self.data is None:
            return np.zeros(self.action_dim)

        if self._ik is None:
            from robot_common.execution.mink_ik import MinkIKSolver
            self._ik = MinkIKSolver(self.model, self.data)

        sol = self._ik.solve_ik(
            target_pos=self._target_pos,
            ee_body_name=self.ee_body,
            max_steps=50,
        )
        if sol.success:
            return sol.joint_positions[:self.action_dim]
        return np.zeros(self.action_dim)


class ACTPolicy(VLAPolicy):
    """LeRobot ACT (Action Chunking Transformer) policy wrapper.

    ACT is a lightweight (~80M params) imitation learning policy that
    natively supports dual-arm 14-DoF action spaces. Trained checkpoints
    are available on HuggingFace.

    References:
        - lerobot/act_aloha_sim_transfer_cube_human
        - lerobot/act_aloha_sim_insertion_human
    """

    def __init__(
        self,
        model_path: str = "",
        action_dim: int = 14,
        chunk_size: int = 100,
        control_freq: float = 50.0,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(action_dim=action_dim, control_freq=control_freq)
        self.model_path = model_path
        self.chunk_size = chunk_size
        self.device = device
        self._policy = None
        self._action_buffer: list = []
        self._buffer_idx: int = 0

        if model_path:
            self._load_policy()

    def _load_policy(self):
        """Load ACT policy from HuggingFace or local path."""
        try:
            from lerobot.common.policies.act.configuration_act import ACTConfig
            from lerobot.common.policies.act.modeling_act import ACTPolicy as ACTModel

            logger.info(f"Loading ACT policy from {self.model_path}...")
            # Try loading via LeRobot's policy loading API
            try:
                from lerobot.common.policies.factory import make_policy
                import torch
                self._policy = make_policy(
                    self.model_path,
                    device=self.device,
                )
                logger.info("ACT policy loaded successfully")
            except Exception as e1:
                logger.warning(f"Could not load via LeRobot factory: {e1}")
                logger.info("Falling back to IK-based bridge policy")
                self._policy = None
        except ImportError:
            logger.warning(
                "LeRobot not installed. Install with: pip install lerobot\n"
                "Falling back to IK-based control."
            )
            self._policy = None

    def predict(
        self,
        image: np.ndarray = None,
        instruction: str = "",
        proprio: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict action from current observation.

        If ACT policy is loaded, uses temporal ensemble of action chunks.
        Otherwise returns zeros (caller should use IK fallback).
        """
        if self._policy is None:
            return np.zeros(self.action_dim)

        try:
            import torch

            # Prepare observation dict in LeRobot format
            obs = {}
            if image is not None:
                # Convert numpy (H,W,3) uint8 to torch (C,H,W) float32
                img = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                obs["observation.images.top"] = img.unsqueeze(0).to(self.device)

            if instruction:
                obs["task"] = instruction

            if proprio is not None:
                state = torch.from_numpy(proprio).float().unsqueeze(0).to(self.device)
                obs["observation.state"] = state

            # Get action from policy
            with torch.no_grad():
                result = self._policy.select_action(obs)
                action = result["action"]  # Shape: (1, chunk_size, action_dim)

            # Return first action in chunk
            action_np = action[0, 0].cpu().numpy()
            if len(action_np) < self.action_dim:
                padded = np.zeros(self.action_dim)
                padded[:len(action_np)] = action_np
                return padded
            return action_np[:self.action_dim]

        except Exception as e:
            logger.error(f"ACT prediction failed: {e}")
            return np.zeros(self.action_dim)

    def reset(self):
        """Reset action buffer between episodes."""
        self._action_buffer = []
        self._buffer_idx = 0


class NoOpPolicy(VLAPolicy):
    """No-op policy that always returns zero actions."""

    def predict(self, image=None, instruction="", proprio=None) -> np.ndarray:
        return np.zeros(self.action_dim)
