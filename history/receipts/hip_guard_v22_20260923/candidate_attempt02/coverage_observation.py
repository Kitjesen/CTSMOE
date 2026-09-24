"""Read-only preflight observations of outer-band exposure and existing cached gates."""


class CoverageObservation:
    def __init__(self, env, torch):
        self.torch, self.dt = torch, float(env.step_dt)
        command = env.command_manager.get_term('base_velocity')
        columns = env.scene.terrain.terrain_types
        self.wheel_ids = command.pit.wheel_ids
        self.groups = {}
        for learner, label in ((0, 'teacher'), (1, 'student')):
            for column, kind in ((8, 'stairs_up'), (11, 'stairs_down')):
                mask = (command.learner == learner) & (columns == column)
                self.groups[label + '/' + kind] = dict(mask=mask.clone(), initial_count=int(mask.sum()))
        n, device = env.num_envs, env.device
        self.names = ('valid', 'root_outer_band', 'wheel_envelope_outer_band',
                      'cached_phase_ready', 'outer_band_phase_ready', 'outer_band_pose_weight_ge_0p9')
        self.counts = {name: torch.zeros(n, dtype=torch.int64, device=device) for name in self.names}
        self.streak = {name: torch.zeros(n, dtype=torch.int64, device=device) for name in self.names[1:]}
        self.longest = {name: torch.zeros(n, dtype=torch.int64, device=device) for name in self.names[1:]}
        self.phase_weight_max = torch.zeros(n, device=device)
        self.pose_weight_max = torch.zeros(n, device=device)
        self.cache_missing_frames = 0

    def observe(self, env, done):
        # Reading the existing reward caches cannot advance recovery timers or add reward calls.
        if (getattr(env, '_phase_step', -1) != env.common_step_counter
                or getattr(env, '_pose_v18_step', -1) != env.common_step_counter):
            self.cache_missing_frames += 1
            for value in self.streak.values():
                value.zero_()
            return
        torch = self.torch
        phase, pose = env.phase_state, env._pose_v18_state
        columns = env.scene.terrain.terrain_types
        selected = (columns == 8) | (columns == 11)
        valid = selected & ~done.bool() & (env.episode_length_buf > 1)
        data, origin = env.scene['robot'].data, env.scene.env_origins
        root_radius = (data.root_pos_w[:, :2] - origin[:, :2]).abs().amax(-1)
        wheel_radius = (data.body_pos_w[:, self.wheel_ids, :2] - origin[:, None, :2]).abs().amax(-1)
        outer = (root_radius >= 2.9) & (root_radius <= 4.4)
        # A conservative horizontal wheel-circle envelope, not a contact or full robot hull test.
        wheels = (wheel_radius.amin(-1) >= 2.9 + .093) & (wheel_radius.amax(-1) <= 4.4 - .093)
        conditions = dict(valid=valid, root_outer_band=valid & outer,
                          wheel_envelope_outer_band=valid & wheels,
                          cached_phase_ready=valid & phase['ready'],
                          outer_band_phase_ready=valid & outer & phase['ready'],
                          outer_band_pose_weight_ge_0p9=valid & outer & (pose['weight'] >= .9))
        for name, condition in conditions.items():
            self.counts[name] += condition.long()
            if name in self.streak:
                self.streak[name] = torch.where(condition, self.streak[name] + 1, 0)
                self.longest[name] = torch.maximum(self.longest[name], self.streak[name])
        self.phase_weight_max = torch.maximum(self.phase_weight_max, torch.where(valid & outer, phase['weight'], 0.))
        self.pose_weight_max = torch.maximum(self.pose_weight_max, torch.where(valid & outer, pose['weight'], 0.))

    def snapshot(self):
        groups = {}
        for label, group in self.groups.items():
            mask = group['mask']
            measures = {}
            for name in self.names:
                count = self.counts[name][mask]
                item = dict(env_steps=int(count.sum()), unique_envs=int((count > 0).sum()))
                if name in self.longest:
                    item['longest_continuous_seconds'] = float(self.longest[name][mask].max()) * self.dt if len(count) else 0.
                measures[name] = item
            groups[label] = dict(initial_env_count=group['initial_count'], measures=measures,
                                 outer_band_phase_weight_max=float(self.phase_weight_max[mask].max()) if mask.any() else None,
                                 outer_band_pose_weight_max=float(self.pose_weight_max[mask].max()) if mask.any() else None)
        return dict(groups=groups, cache_missing_control_steps=self.cache_missing_frames,
                    control_dt_s=self.dt, outer_band_chebyshev_radius_m=[2.9, 4.4],
                    wheel_horizontal_envelope_radius_m=.093,
                    excluded_frames='done or episode_length<=1; missing cache interrupts every continuous streak',
                    scope='Read-only root/wheel XY geometry and existing cached phase/pose gates. '
                          'Not contact-supported landing, full collision hull containment, behavior success, '
                          'or evidence of sufficient learning dwell. Initial cohort IDs are held fixed.')
