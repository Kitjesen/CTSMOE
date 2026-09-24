# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Gym registration for DOSO Thunder terrain-inward tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="DOSO-Isaac-Velocity-Flat-Thunder-Terrain-Inward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DosoThunderTerrainInwardFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DosoThunderTerrainInwardFlatPPORunnerCfg"
        ),
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DosoThunderTerrainInwardFlatTrainerCfg",
    },
)
gym.register(
    id="DOSO-Isaac-Velocity-Rough-Thunder-Terrain-Inward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_v0_env_cfg:DosoThunderTerrainInwardRoughV0EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DosoThunderTerrainInwardRoughPPORunnerCfg"
        ),
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DosoThunderTerrainInwardRoughTrainerCfg",
    },
)
