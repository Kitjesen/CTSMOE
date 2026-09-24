"""Reset-boundary implementation of bounded mixed-terrain allocation."""
from collections import Counter

import torch

from terrain_allocation import AllocationCourse, BASE, PROFILE, reassign


class AdaptiveAllocation:
    def __init__(self, env, names, groups, iteration, saved=None):
        self.env, self.names, self.groups = env, names, groups
        self.command = env.command_manager.get_term('base_velocity')
        self.keys = list(BASE)
        self.course = AllocationCourse(last_check=iteration)
        self.stats = torch.zeros((9, 22, 6, 3, 3), device=env.device, dtype=torch.float64)
        self.sums = torch.zeros((env.num_envs, 4), device=env.device)
        self.pending = {}
        self.moves_applied = 0
        if saved is not None:
            if saved['profile'] != PROFILE:
                raise ValueError('Unknown allocation checkpoint')
            self.course = AllocationCourse(**saved['course'])
            self.stats.copy_(saved['stats'].to(env.device))
            self.pending = saved['pending']
            self.moves_applied = saved['moves_applied']

    def step(self):
        c = self.command
        xy = c.robot.data.root_lin_vel_b[:, :2]
        values = torch.stack(((xy-c.commands[:, :2]).norm(dim=1),
                              c.commands[:, :2].norm(dim=1),
                              c.commands[:, 2].abs(), torch.ones(self.env.num_envs, device=self.env.device)), -1)
        self.sums += values

    def record(self, ids):
        """Called before terrain curricula change rows and before reset events."""
        env, c = self.env, self.command
        ids = ids[(env.episode_length_buf[ids] > 0) & (c.learner[ids] == 1)]
        if not len(ids):
            return
        counts = self.sums[ids, 3].clamp_min(1)
        mean_error = self.sums[ids, 0]/counts
        speed = self.sums[ids, 1]/counts
        speed = torch.where(c.stair_mask()[ids], c.held_speed[ids], speed)
        speed = torch.where(c.pit.active[ids], c.pit.held_speed[ids], speed)
        yaw = self.sums[ids, 2]/counts
        lookup = torch.tensor([self.keys.index(name) for name in c.terrain_types], device=env.device)
        types = lookup[c.terrain_idxs[ids]]
        levels = env.scene.terrain.terrain_levels[ids]
        bins = (speed/.25).long().clamp(0, 5)
        yaw_bins = (yaw/.5).long().clamp(0, 2)
        failed = env.termination_manager.terminated[ids] & ~(
            c.pit.active[ids] & c.pit.success[ids])
        index = ((types*22 + levels)*6 + bins)*3 + yaw_bins
        values = torch.stack((torch.ones_like(mean_error), failed, mean_error), -1)
        self.stats.view(-1, 3).index_add_(0, index, values.double())

    def after_curriculum(self, ids):
        """Only ended environments move; spawning and history reset follow normally."""
        env, c = self.env, self.command
        self.sums[ids] = 0
        if not self.pending:
            return
        moves = [(i, self.pending[i]) for i in ids.cpu().tolist() if i in self.pending]
        if not moves:
            return
        terrain = env.scene.terrain
        moved = torch.tensor([i for i, _ in moves], device=env.device)
        terrain.terrain_levels[moved] = torch.tensor([v[0] for _, v in moves], device=env.device)
        terrain.terrain_types[moved] = torch.tensor([v[1] for _, v in moves], device=env.device)
        terrain.env_origins[moved] = terrain.terrain_origins[
            terrain.terrain_levels[moved], terrain.terrain_types[moved]]
        env.scene.env_origins[moved] = terrain.env_origins[moved]
        # Clear departing pit state so pit-only rewards cannot leak into old types.
        for name in ('hold', 'success', 'climb_hold', 'climb_seen', 'climb_event',
                     'platform_clear', 'settle_steps', 'stop_on_platform', 'front_seen',
                     'all_seen', 'progress', 'supported', 'physical', 'outside', 'body_contact'):
            getattr(c.pit, name)[moved] = 0
        c.refresh_membership()
        c.pit.last_refresh = -1
        for i, _ in moves:
            del self.pending[i]
        self.moves_applied += len(moves)

    def check(self, iteration):
        if iteration-self.course.last_check < 500 or self.pending:
            return None
        values = self.stats.cpu().tolist()
        current = {}
        for t, name in enumerate(self.keys):
            current[name] = {f'{level}:{speed}:{yaw}': values[t][level][speed][yaw]
                             for level in range(22) for speed in range(6) for yaw in range(3)
                             if values[t][level][speed][yaw][0]}
        decision = self.course.evaluate(current, iteration)
        self.stats.zero_()
        if decision['move']:
            terrain = self.env.scene.terrain
            levels, columns, receipt = reassign(terrain.terrain_levels.cpu().tolist(),
                terrain.terrain_types.cpu().tolist(), self.names, self.groups, self.course.weights,
                seed=iteration)
            self.pending = {i: (levels[i], columns[i]) for i, _ in receipt['moves']}
            decision['pending_moves'] = len(self.pending)
        return decision

    def snapshot(self):
        columns = self.env.scene.terrain.terrain_types.cpu().tolist()
        return dict(profile=PROFILE, target_weights=dict(self.course.weights),
                    actual_counts=dict(Counter(self.names[col] for col in columns)),
                    pending_moves=len(self.pending), moves_applied=self.moves_applied,
                    last_check=self.course.last_check, check_interval=500,
                    evidence='matched training strata, not independent held-out validation',
                    decisions=self.course.decisions[-2:])

    def state_dict(self):
        return dict(profile=PROFILE, course=self.course.state_dict(), stats=self.stats,
                    pending=self.pending, moves_applied=self.moves_applied)
