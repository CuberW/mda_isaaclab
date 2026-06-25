"""Task 319 local grasp RL task registration.

This package intentionally lives outside ``isaaclab_tasks`` so the upstream
IsaacLab tree stays untouched.  Importing the package registers the Gym task.
"""

from __future__ import annotations

import gymnasium as gym

from . import agents


TASK_ID = "Task319-Hover-Descent-Grasp-Kuavo-Direct-v0"
LEGACY_TASK_ID = "Task319-Local-Suction-Grasp-Kuavo-Direct-v0"


for task_id in (TASK_ID, LEGACY_TASK_ID):
    gym.register(
        id=task_id,
        entry_point="task319_local_grasp_rl.local_suction_grasp_env:Task319LocalSuctionGraspEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "task319_local_grasp_rl.local_suction_grasp_env:Task319LocalSuctionGraspEnvCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
            "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        },
    )
