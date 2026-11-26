"""Lope pure locomotion task registration."""
import os

try:
    import gymnasium as gym
except ModuleNotFoundError:
    gym = None

from lope_rl_lab.tasks.locomotion.config import TASK_ID


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


def register_tasks():
    """Register the Lope locomotion Gym environment."""
    if gym is None:
        raise ModuleNotFoundError("gymnasium is required to register Lope Gym tasks")
    gym.register(
        id=TASK_ID,
        entry_point="lope_rl_lab.tasks.locomotion:make_env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "lope_rl_lab.tasks.locomotion.env_cfg:LopeLocomotionEnvCfg",
            "play_env_cfg_entry_point": "lope_rl_lab.tasks.locomotion.env_cfg:LopeLocomotionPlayEnvCfg",
            "rsl_rl_cfg_entry_point": "lope_rl_lab.tasks.locomotion.agents.rsl_rl_moe_cts_cfg:LopeLocomotionMoeCtsRunnerCfg",
        },
    )


if gym is not None:
    register_tasks()
