"""Official RLGym velocity command, unscaled, with commanded-direction progress tracking.

Ranges, curricula, clip rules, and the iteration-25000 limit stage are the
go2_rl_lab mixed-task numbers. Terrain caps omit wheels because Lope has none.
"""
from collections.abc import Sequence
from copy import deepcopy

import torch
from go2_rl_lab.cmd_utils.rl_gym_cmd import RLGymVelCommand, RLGymVelCommandCfg
from isaaclab.utils import configclass

from lope_rl_lab.tasks.locomotion.config import (
    LOPE_CLIP_RULES,
    LOPE_COMMAND_RANGE_CURRICULUM,
    LOPE_INITIAL_RANGES,
    LOPE_TERRAIN_MAX_COMMANDS,
    OFFICIAL_LIMIT_VEL_CURRICULUM,
    RESAMPLING_TIME_S,
    ZERO_COMMAND_CURRICULUM,
)


class LopeVelCommand(RLGymVelCommand):
    """RLGymVelCommand extended with yaw-invariant commanded-path progress tracking."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.projected_move_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.commanded_move_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        if env_ids is None or isinstance(env_ids, slice):
            resolved = torch.arange(self.num_envs, device=self.device)[env_ids or slice(None)]
        else:
            resolved = env_ids
        self.projected_move_distance[resolved] = 0.0
        self.commanded_move_distance[resolved] = 0.0
        return super().reset(env_ids)

    def _update_command(self):
        super()._update_command()
        cmd_xy = self.velocity_command_b[:, :2]
        cmd_speed = torch.linalg.vector_norm(cmd_xy, dim=1)
        valid_cmd = cmd_speed > 0.1
        cmd_dir = torch.where(
            valid_cmd.unsqueeze(1),
            cmd_xy / cmd_speed.clamp(min=1.0e-6).unsqueeze(1),
            torch.zeros_like(cmd_xy),
        )
        vel_xy = self.robot.data.root_lin_vel_b.torch[:, :2]
        progress_speed = torch.clamp(torch.sum(vel_xy * cmd_dir, dim=1), min=0.0)
        dt = float(self._env.step_dt)
        zeros = torch.zeros_like(progress_speed)
        self.projected_move_distance.add_(torch.where(valid_cmd, progress_speed * dt, zeros))
        self.commanded_move_distance.add_(torch.where(valid_cmd, cmd_speed * dt, zeros))


@configclass
class LopeVelCommandCfg(RLGymVelCommandCfg):
    class_type: type = LopeVelCommand


def lope_velocity_command_cfg():
    """Return one command config. Copies official lists before replacing them."""
    cfg = LopeVelCommandCfg()
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
    cfg.zero_command_curriculum = dict(ZERO_COMMAND_CURRICULUM)
    cfg.resampling_time = RESAMPLING_TIME_S
    cfg.resampling_time_range = (RESAMPLING_TIME_S, RESAMPLING_TIME_S)
    return cfg


@configclass
class CommandsCfg:
    base_velocity = lope_velocity_command_cfg()
