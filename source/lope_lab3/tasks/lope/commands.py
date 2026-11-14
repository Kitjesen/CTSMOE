"""Official RLGym velocity command with Lope's height-scaled ranges.

The class, stage iterations, and limit-velocity direction codes come from
go2_rl_lab ``RLGymVelCommandCfg`` and mixed ``CommandsCfg``. Linear speeds and
yaw magnitudes are scaled in ``lope_lab3.spec``. This module does not edit
Thunder and does not use UniformVelocityCommand or the old 1.0 m/s gate.
"""
from copy import deepcopy

from go2_rl_lab.cmd_utils.rl_gym_cmd import RLGymVelCommandCfg
from isaaclab.utils import configclass

from lope_lab3.spec import (
    LOPE_CLIP_RULES,
    LOPE_COMMAND_RANGE_CURRICULUM,
    LOPE_INITIAL_RANGES,
    LOPE_TERRAIN_MAX_COMMANDS,
    OFFICIAL_LIMIT_VEL_CURRICULUM,
)


def lope_velocity_command_cfg():
    """Return one command config. Copies official lists before replacing them."""
    cfg = RLGymVelCommandCfg()
    cfg.ranges.lin_vel_x = LOPE_INITIAL_RANGES["lin_vel_x"]
    cfg.ranges.lin_vel_y = LOPE_INITIAL_RANGES["lin_vel_y"]
    cfg.ranges.ang_vel_yaw = LOPE_INITIAL_RANGES["ang_vel_yaw"]
    cfg.command_range_curriculum = [dict(stage) for stage in LOPE_COMMAND_RANGE_CURRICULUM]
    cfg.limit_vel_curriculum = deepcopy(list(OFFICIAL_LIMIT_VEL_CURRICULUM))
    cfg.terrain_max_command_ranges = {
        name: dict(axes) for name, axes in LOPE_TERRAIN_MAX_COMMANDS.items()
    }
    cfg.command_clip_rules = [dict(rule) for rule in LOPE_CLIP_RULES]
    for rule in cfg.command_clip_rules:
        rule["clip"] = dict(rule["clip"])
    return cfg


@configclass
class CommandsCfg:
    base_velocity = lope_velocity_command_cfg()
