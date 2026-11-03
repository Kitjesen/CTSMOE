"""Go2 command curricula with Thunder's wheel-leg stair restrictions.

Sampling, boundary modes and current-tile lookup come from the pinned upstream
implementation.  These restrictions are intentional differences from Go2: the
user's global speed cap, forward-only stairs, and a budget integrated from the
commands actually issued after terrain clipping.
"""

from copy import deepcopy
import math

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from go2_rl_lab.cmd_utils.rl_gym_cmd import RLGymVelCommand, RLGymVelCommandCfg

from .mdp.terms import TerrainProgress, env_indices, stairs


# Both pyramid families start in the centre: the inverted one climbs outward.
# Random rough has no identical Go2 category; it uses the conservative user cap.
TERRAIN_COMMAND_SOURCE = {
    "flat": "flat",
    "pyramid_stairs": "stairs_down",
    "pyramid_stairs_inv": "stairs_up",
    "boxes": "small_squares",
    "random_rough": None,
    "hf_pyramid_slope": "slope_down",
    "hf_pyramid_slope_inv": "slope_up",
    "wave": "wave",
}


def terrain_command_ranges():
    """Translate the eight terrain names, intersecting Go2 and user maxima."""
    source = RLGymVelCommandCfg().terrain_max_command_ranges
    cap = {"lin_vel_x": (-1., 1.), "lin_vel_y": (-1., 1.), "ang_vel_yaw": (-1.5, 1.5)}
    result = {}
    for name, upstream_name in TERRAIN_COMMAND_SOURCE.items():
        limits = source[upstream_name] if upstream_name is not None else cap
        result[name] = {
            axis: (max(bounds[0], limits[axis][0]), min(bounds[1], limits[axis][1]))
            for axis, bounds in cap.items()
        }
    for name in ("pyramid_stairs", "pyramid_stairs_inv"):
        result[name] = {"lin_vel_x": (.25, .5), "lin_vel_y": (0., 0.), "ang_vel_yaw": (-.2, .2)}
    return result


class ThunderRLGymVelocityCommand(TerrainProgress, RLGymVelCommand):
    """Use official Go2 sampling while preserving Thunder curriculum telemetry."""

    def __init__(self, cfg, env):
        # Upstream consumes curriculum lists with pop(). Keep declarative env.cfg
        # unchanged for hashes/resume and give each command instance its own list.
        super().__init__(deepcopy(cfg), env)
        self.was_on_stairs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.is_standing_env = torch.zeros_like(self.was_on_stairs)
        self.stair_heading = torch.zeros(self.num_envs, device=self.device)
        self._init_travel_statistics()

    def _set_stair_heading(self, ids):
        if not len(ids):
            return
        yaw = euler_xyz_from_quat(self.robot.data.root_quat_w.torch[ids])[2]
        self.stair_heading[ids] = torch.round(yaw / (math.pi / 2)) * (math.pi / 2)
        self.stair_heading[ids] += torch.empty(len(ids), device=self.device).uniform_(-math.pi / 18, math.pi / 18)

    def _apply_terrain_command_limits(self, env_ids=None):
        ids = env_indices(env_ids, self.num_envs, self.device)
        super()._apply_terrain_command_limits(ids)
        mask = stairs(self._env)
        stair_ids = ids[mask[ids]]
        self._set_stair_heading(ids[mask[ids] & ~self.was_on_stairs[ids]])
        self.was_on_stairs[ids] = mask[ids]
        self.is_standing_env[ids] = (self._requested_velocity_command_b[ids] == 0.).all(dim=1)
        if len(stair_ids):
            yaw = euler_xyz_from_quat(self.robot.data.root_quat_w.torch[stair_ids])[2]
            self.velocity_command_b[stair_ids, 0] = self.velocity_command_b[stair_ids, 0].clamp(.25, .5)
            self.velocity_command_b[stair_ids, 1] = 0.
            self.velocity_command_b[stair_ids, 2] = (
                .5 * wrap_to_pi(self.stair_heading[stair_ids] - yaw)).clamp(-.2, .2)
            # The positive stair sampling interval must not turn standing into
            # creeping forward. Retain the true zero mode selected by upstream.
            self.velocity_command_b[stair_ids[self.is_standing_env[stair_ids]]] = 0.

    def _resample(self, env_ids):
        ids = env_indices(env_ids, self.num_envs, self.device)
        if not len(ids):
            return
        mask = stairs(self._env)
        self._set_stair_heading(ids[mask[ids]])
        self.was_on_stairs[ids] = mask[ids]
        # Apply any due stage before the upstream call, which repeats this check.
        # Its standing-time calculation can then use per-environment *effective*
        # speed rather than assume a staircase can use the global 1 m/s cap.
        self._update_command_range_curriculum()
        self._update_env_command_ranges()
        global_max = self.max_lin_vel
        elapsed_budget = self.commands_xy_accumulation[ids].clone()
        try:
            self.max_lin_vel = self._env_command_ranges[ids, :2].abs().amax(dim=(1, 2))
            super()._resample(ids)
        finally:
            self.max_lin_vel = global_max
            # Parent adds a prospective five-second command. Integrate actual
            # control intervals instead, including short stops and tile changes.
            self.commands_xy_accumulation[ids] = elapsed_budget

    def accumulate(self, env_ids):
        ids = env_indices(env_ids, self.num_envs, self.device)
        ids = ids[(self._env.episode_length_buf[ids] > 0)
                  & (self.travel_step[ids] != self._env.common_step_counter)]
        # Preserve upstream units: its sampler multiplies this by resampling_time
        # to get commanded displacement. This is not measured robot progress.
        self.commands_xy_accumulation[ids] += self.command[ids, :2] * (
            self._env.step_dt / self.cfg.resampling_time)
        super().accumulate(ids)

    def _update_metrics(self):
        super()._update_metrics()
        self.accumulate(None)

    def reset(self, env_ids=None):
        ids = env_indices(env_ids, self.num_envs, self.device)
        # Lab resets this buffer *after* command_manager.reset. The first command
        # must use the new episode's full horizon, not the old episode's tail.
        previous_lengths = self._env.episode_length_buf[ids].clone()
        self._env.episode_length_buf[ids] = 0
        try:
            result = super().reset(ids)
        finally:
            self._env.episode_length_buf[ids] = previous_lengths
        self._reset_travel_statistics(ids)
        return result


@configclass
class ThunderRLGymVelocityCommandCfg(RLGymVelCommandCfg):
    class_type: type = ThunderRLGymVelocityCommand
    command_range_curriculum: list[dict] = [{
        "iter": 5000,
        "lin_vel_x": (-1., 1.),
        "lin_vel_y": (-1., 1.),
        "ang_vel_yaw": (-1.5, 1.5),
    }]
    terrain_max_command_ranges: dict = terrain_command_ranges()

