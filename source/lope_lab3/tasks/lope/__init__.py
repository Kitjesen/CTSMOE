"""Register the from-scratch Lope task. Thunder gym ids are not modified."""
import os

import gymnasium as gym

from lope_lab3.spec import TASK_ID


def make_env(cfg, **kwargs):
    """Build the manager env after the caller has composed the config."""
    from isaaclab.envs import ManagerBasedRLEnv

    seed = os.environ.get("LOPE_SEED")
    if seed is not None:
        seed = int(seed)
        if not 0 <= seed < 2**32:
            raise ValueError("LOPE_SEED must be in [0, 2**32)")
        cfg.seed = seed
    return ManagerBasedRLEnv(cfg=cfg, **kwargs)


gym.register(
    id=TASK_ID,
    entry_point="lope_lab3.tasks.lope:make_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "lope_lab3.tasks.lope.env_cfg:LopeFreshEnvCfg",
        "play_env_cfg_entry_point": "lope_lab3.tasks.lope.env_cfg:LopeFreshPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "lope_lab3.tasks.lope.rsl_rl_cfg:LopeFreshMoeCtsRunnerCfg",
    },
)
