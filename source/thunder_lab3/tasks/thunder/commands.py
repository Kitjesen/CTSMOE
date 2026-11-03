"""Go2 command sampling with Thunder terrain names and progress bookkeeping.

The pinned upstream implementation owns all command modes and current-tile
clipping, including stairs. Thunder only adapts the names and integrates the
command budget over the control intervals actually elapsed.
"""

from copy import deepcopy

from isaaclab.utils import configclass
from go2_rl_lab.cmd_utils.rl_gym_cmd import RLGymVelCommand, RLGymVelCommandCfg
from go2_rl_lab.tasks.mixed.env_cfg import CommandsCfg as Go2CommandsCfg
from go2_rl_lab.tasks.mixed.mdp.curriculums import TerrainLevelsVelGym

from .mdp.terms import TerrainProgress, env_indices


# Both pyramid families start in the centre: the inverted one climbs outward.
# Random rough has no identical Go2 geometry; reuse the rough_slope speed policy.
TERRAIN_COMMAND_SOURCE = {
    "flat": "flat",
    "pyramid_stairs": "stairs_down",
    "pyramid_stairs_inv": "stairs_up",
    "boxes": "small_squares",
    "random_rough": "rough_slope",
    "hf_pyramid_slope": "slope_down",
    "hf_pyramid_slope_inv": "slope_up",
    "wave": "wave",
}


def terrain_command_ranges():
    """Translate Thunder names without changing the corresponding Go2 limits."""
    source = RLGymVelCommandCfg().terrain_max_command_ranges
    return {name: deepcopy(source[upstream_name]) for name, upstream_name in TERRAIN_COMMAND_SOURCE.items()}


class ThunderRLGymVelocityCommand(TerrainProgress, RLGymVelCommand):
    """Use official Go2 sampling while preserving Thunder curriculum telemetry."""

    def __init__(self, cfg, env):
        # Upstream consumes curriculum lists with pop(). Keep declarative env.cfg
        # unchanged for hashes/resume and give each command instance its own list.
        super().__init__(deepcopy(cfg), env)
        self._init_travel_statistics()

    def _resample(self, env_ids):
        ids = env_indices(env_ids, self.num_envs, self.device)
        if not len(ids):
            return
        # Apply any due stage before the upstream call, which repeats this check.
        # Its standing-time calculation can then use per-environment *effective*
        # speed rather than assume a staircase can use the full flat-ground cap.
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
        # The reset curriculum runs before CommandTerm.compute. Include a final
        # tile-crossing step when it explicitly accumulates the ending episode.
        distance = (self.robot.data.root_pos_w.torch[ids, :2]
                    - self._env.scene.env_origins[ids, :2]).norm(dim=1)
        self.max_move_distance[ids] = self.max_move_distance[ids].maximum(distance)
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


class ThunderTerrainLevels(TerrainLevelsVelGym):
    """Record the terminal frame before applying the unchanged Go2 level rule."""

    def __call__(self, env, env_ids):
        ids = env_indices(env_ids, env.num_envs, env.device)
        env.command_manager.get_term("base_velocity").accumulate(ids)
        return super().__call__(env, ids)


@configclass
class ThunderRLGymVelocityCommandCfg(RLGymVelCommandCfg):
    class_type: type = ThunderRLGymVelocityCommand
    # The mixed task adds its 25k boundary-mode transition to the base sampler.
    # Global speed stages and coupled-axis clipping are inherited unchanged.
    limit_vel_curriculum: list[dict] = deepcopy(Go2CommandsCfg().base_velocity.limit_vel_curriculum)
    terrain_max_command_ranges: dict = terrain_command_ranges()

