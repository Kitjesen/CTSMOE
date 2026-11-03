"""Straight stair commands on the pinned terrain; other terrain keeps upstream commands."""
import math

import torch

from robot_lab.tasks.go2.mdp.commands import Go2RLGymCommand
from robot_lab.tasks.go2.mdp.curriculums import terrain_levels_vel_gym
from robot_lab.tasks.go2.mdp.events import reset_root_state_uniform
from robot_lab.tasks.go2.mdp.utils import is_env_assigned_to_terrain
from stair_speed_logic import PROFILE, SPEED_EDGES, sample_speeds, terrain_decisions


class StraightStairCommand(Go2RLGymCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.speed_bins = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.held_speed = torch.zeros(self.num_envs, device=self.device)
        self.needs_speed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.learner = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.error_sum = torch.zeros(self.num_envs, device=self.device)
        self.error_samples = torch.zeros(self.num_envs, device=self.device)
        # learner, direction, level, speed bin, [episodes, radius success, falls, MAE sum, eligible]
        self.episode_totals = torch.zeros((2, 2, 10, 4, 5), device=self.device, dtype=torch.float64)

    def stair_mask(self):
        return ((self.terrain_idxs == self.terrain_type2idx["stairs_up"])
                | (self.terrain_idxs == self.terrain_type2idx["stairs_down"]))

    def set_student_indices(self, indices):
        self.learner.zero_()
        self.learner[indices] = 1

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.needs_speed[env_ids] = True
        self.error_sum[env_ids] = 0
        self.error_samples[env_ids] = 0
        return super().reset(env_ids)

    def _resample(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        stairs = self.stair_mask()[env_ids]
        # Upstream owns all original random/zero-command branches on non-stairs.
        super()._resample(env_ids[~stairs])
        ids = env_ids[stairs]
        fresh = ids[self.needs_speed[ids]]
        bins, speeds = sample_speeds(len(fresh), self.device)
        self.speed_bins[fresh], self.held_speed[fresh] = bins, speeds
        self.needs_speed[fresh] = False
        self.commands[ids] = 0
        self.commands[ids, 0] = self.held_speed[ids]
        self.last_is_limit_vel[ids] = False
        self.time_left[ids] = 1.0e9  # Hold one randomly sampled speed for the entire episode.

    def _update_metrics(self):
        super()._update_metrics()
        valid = self.stair_mask() & (self._env.episode_length_buf > 0)
        error = (self.robot.data.root_lin_vel_b[:, 0] - self.commands[:, 0]).abs()
        self.error_sum[valid] += error[valid]
        self.error_samples[valid] += 1

    def _update_command(self):
        super()._update_command()
        ids = self.stair_mask().nonzero().flatten()
        elapsed = self._env.episode_length_buf[ids] * self._env.step_dt
        # Preserve the upstream diagnostic field, but use actual elapsed commands.
        self.commands_xy_accumulation[ids, 0] = self.held_speed[ids] * elapsed / self.cfg.resampling_time
        self.commands_xy_accumulation[ids, 1] = 0

    def record_episodes(self, ids, move_up, terminated, eligible):
        terrain = self._env.scene.terrain
        direction = (self.terrain_idxs[ids] == self.terrain_type2idx["stairs_down"]).long()
        key = ((self.learner[ids] * 2 + direction) * 10 + terrain.terrain_levels[ids]) * 4 + self.speed_bins[ids]
        # Command.compute runs after terminal reset; include the terminal sample here.
        final_error = (self.robot.data.root_lin_vel_b[ids, 0] - self.held_speed[ids]).abs()
        mae = (self.error_sum[ids] + final_error) / (self.error_samples[ids] + 1)
        values = torch.stack((torch.ones_like(mae), move_up, terminated, mae, eligible), dim=1)
        self.episode_totals.view(-1, 5).index_add_(0, key, values.double())

    def snapshot(self):
        mask = self.stair_mask()
        commands = self.commands[mask]
        valid = ((commands[:, 0] >= SPEED_EDGES[0]) & (commands[:, 0] <= SPEED_EDGES[-1])
                 & (commands[:, 1:] == 0).all(dim=1))
        if not bool(valid.all()):
            raise RuntimeError("Straight stair command contract violated")
        totals = self.episode_totals.sum(dim=2).cpu().tolist()
        by_group = {}
        for learner, learner_name in enumerate(("teacher", "student")):
            by_group[learner_name] = {}
            for direction, name in enumerate(("stairs_up", "stairs_down")):
                entries = []
                for bin_idx, (episodes, passed, falls, mae, eligible) in enumerate(totals[learner][direction]):
                    entries.append({"speed_range": list(SPEED_EDGES[bin_idx:bin_idx + 2]),
                                    "episodes": int(episodes), "grade_eligible": int(eligible),
                                    "radius_passes": int(passed), "physical_failures": int(falls),
                                    "vx_mae_mps": mae / episodes if episodes else None})
                by_group[learner_name][name] = entries
        return {"profile": PROFILE, "speed_edges": list(SPEED_EDGES),
                "stairs_command_count": len(commands),
                "active_speed_bin_counts": torch.bincount(self.speed_bins[mask], minlength=4).cpu().tolist(),
                "vx_min_max": [float(commands[:, 0].min()), float(commands[:, 0].max())],
                "vy_wz_abs_max": commands[:, 1:].abs().amax(dim=0).cpu().tolist(),
                "episode_statistics": by_group,
                "criterion": "eligible forward exit > 4m without physical termination; not independent stair completion",
                "statistics_mode": "cumulative stochastic training episodes; no automatic speed-weight adjustment"}


def reset_stair_root(env, env_ids, pose_range, velocity_range, asset_cfg):
    """Choose one of four stair-axis headings at reset; physics remains unconstrained."""
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    mask = is_env_assigned_to_terrain(env, "stairs_up") | is_env_assigned_to_terrain(env, "stairs_down")
    other = env_ids[~mask[env_ids]]
    reset_root_state_uniform(env, other, pose_range, velocity_range, asset_cfg)
    stairs = env_ids[mask[env_ids]]
    headings = torch.randint(4, (len(stairs),), device=env.device)
    if not hasattr(env, "stair_axis"):
        env.stair_axis = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env.stair_axis[stairs] = headings
    for axis in range(4):
        ids = stairs[headings == axis]
        aligned_pose = dict(pose_range, yaw=(axis * math.pi / 2, axis * math.pi / 2))
        reset_root_state_uniform(env, ids, aligned_pose, velocity_range, asset_cfg)


def stair_route_exit(env):
    """End the reference traverse at the existing 4 m boundary, before leaving the cell.

    This is a time-limit truncation, not a new reward or proof that all four wheels
    completed a real stair flight. Independent complete-flight evaluation is separate.
    """
    command = env.command_manager.get_term("base_velocity")
    if not hasattr(env, "stair_axis"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    axes = torch.tensor(((1., 0.), (0., 1.), (-1., 0.), (0., -1.)), device=env.device)
    offset = command.robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    progress = (offset * axes[env.stair_axis]).sum(dim=1)
    return command.stair_mask() & (progress > command.terrain_length / 2)


def straight_stair_terrain_levels(env, env_ids):
    """Retain terrain rows; separate held-speed grading from upstream mixed commands."""
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    command = env.command_manager.get_term("base_velocity")
    stairs = command.stair_mask()[env_ids]
    others = env_ids[~stairs]
    if len(others):
        terrain_levels_vel_gym(env, others)
    ids = env_ids[stairs]
    ids = ids[(env.episode_length_buf[ids] > 0) & (command.speed_bins[ids] >= 0)]
    if len(ids):
        current_radius = torch.linalg.vector_norm(
            command.robot.data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2], dim=1)
        radius = torch.maximum(command.max_move_distance[ids], current_radius)
        terminated = env.termination_manager.terminated[ids]
        elapsed = env.episode_length_buf[ids] * env.step_dt
        up, down, eligible = terrain_decisions(command.held_speed[ids], elapsed, radius, terminated,
                                               env.max_episode_length_s, command.terrain_length / 2)
        # A sideways drift beyond the radius is not a straight stair exit.
        up &= stair_route_exit(env)[ids]
        from startup_repair import PHASE_PROFILES
        if getattr(env.cfg, 'startup_repair_profile', 'off') in PHASE_PROFILES:
            from phase_transition import measure
            phase = measure(env)
            up &= (phase['stable_s'][ids] >= .4) & (phase['weight'][ids] >= .8)
        command.record_episodes(ids, up, terminated, eligible)
        env.scene.terrain.update_env_origins(ids, up, down)
    return env.scene.terrain.terrain_levels.float().mean()


def configure_stair_speed(cfg, profile):
    if profile == "original":
        return
    if profile != PROFILE:
        raise ValueError(f"Unknown stair command profile: {profile}")
    if cfg.commands.base_velocity.ranges.lin_vel_x[1] < SPEED_EDGES[-1]:
        raise ValueError("Enable the existing intermediate command range before stair speed coverage")
    cfg.commands.base_velocity.class_type = StraightStairCommand
    cfg.events.reset_base.func = reset_stair_root
    # The original event omits asset_cfg and relies on its default.
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
    cfg.events.reset_base.params.setdefault("asset_cfg", SceneEntityCfg("robot"))
    cfg.curriculum.terrain_levels.func = straight_stair_terrain_levels
    cfg.terminations.stair_route_exit = TerminationTermCfg(func=stair_route_exit, time_out=True)
