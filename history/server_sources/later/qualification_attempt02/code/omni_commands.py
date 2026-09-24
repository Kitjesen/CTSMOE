"""Explicit flat-only command coverage, preserving complete 25 second episodes."""
import torch

PROFILE = 'omni_full_episode_v19'
NAMES = ('forward', 'backward', 'left', 'right', 'yaw_left', 'yaw_right',
         'diagonal_a', 'diagonal_b', 'longitudinal_turn', 'lateral_turn', 'stop')
PROBABILITIES = (.20, .10, .10, .10, .10, .10, .05, .05, .05, .05, .10)
RAMP_S = .2


def sample_goals(count, device='cpu', generator=None, moving_only=False):
    probs = torch.tensor(PROBABILITIES, device=device)
    if moving_only:
        probs[-1] = 0.
    kinds = torch.multinomial(probs, count, replacement=True, generator=generator)
    vx = .2 + .8 * torch.rand(count, device=device, generator=generator)
    vy = .2 + .4 * torch.rand(count, device=device, generator=generator)
    yaw = .3 + .7 * torch.rand(count, device=device, generator=generator)
    signs = torch.where(torch.rand(count, 3, device=device, generator=generator) < .5, -1., 1.)
    goals = torch.zeros(count, 3, device=device)
    goals[:, 0] = torch.where(kinds == 0, vx, torch.where(kinds == 1, -vx, 0.))
    goals[:, 1] = torch.where(kinds == 2, vy, torch.where(kinds == 3, -vy, 0.))
    goals[:, 2] = torch.where(kinds == 4, yaw, torch.where(kinds == 5, -yaw, 0.))
    goals[:, 0] += torch.where((kinds == 6) | (kinds == 8), vx, torch.where(kinds == 7, -vx, 0.))
    goals[:, 1] += torch.where((kinds == 6) | (kinds == 9), vy, torch.where(kinds == 7, -vy, 0.))
    goals[:, 2] += torch.where((kinds == 8) | (kinds == 9), yaw, 0.)
    goals = torch.where(((kinds >= 6) & (kinds <= 9))[:, None], goals.abs() * signs, goals)
    return goals, kinds


def scheduled_stop(sequence_mask, segments):
    return sequence_mask & ((segments % 2) == 1)


def command_class(base):
    class OmniCommand(base):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.omni_segment = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
            self.omni_resetting = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self.omni_goal = torch.zeros_like(self.commands)
            self.omni_ramp = torch.zeros_like(self.omni_resetting)
            self.omni_started = torch.zeros_like(self.omni_segment)
            self.omni_counts = torch.zeros((2, len(NAMES)), device=self.device, dtype=torch.long)
            self.omni_transitions = torch.zeros((2, 2), device=self.device, dtype=torch.long)
            self.omni_sequence = torch.zeros_like(self.omni_resetting)
            self._select_sequence()

        def _select_sequence(self):
            generator = torch.Generator(device=self.device).manual_seed(190922)
            self.omni_sequence.zero_()
            for group in (0, 1):
                ids = (self.learner == group).nonzero().flatten()
                order = torch.randperm(len(ids), device=self.device, generator=generator)
                self.omni_sequence[ids[order[:len(ids)//5]]] = True

        def set_student_indices(self, indices):
            super().set_student_indices(indices)
            self._select_sequence()

        def reset(self, env_ids=None):
            ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None
                   else torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
            self.omni_segment[ids] = -1
            self.omni_resetting[ids] = True
            try:
                return super().reset(ids)
            finally:
                self.omni_resetting[ids] = False

        def _resample(self, env_ids):
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            integral_before = self.commands_xy_accumulation[ids].clone()
            old_stop = (self.commands[ids] == 0).all(-1)
            super()._resample(ids)
            flat = self.terrain_idxs[ids] == self.terrain_type2idx['flat']
            chosen = ids[flat]
            if not len(chosen):
                return
            self.startup_active[chosen] = False
            self.omni_segment[chosen] += 1
            goals, kinds = sample_goals(len(chosen), self.device)
            seq = self.omni_sequence[chosen]
            if bool(seq.any()):
                moving, moving_kinds = sample_goals(int(seq.sum()), self.device, moving_only=True)
                goals[seq], kinds[seq] = moving, moving_kinds
            stopped = scheduled_stop(seq, self.omni_segment[chosen])
            goals[stopped] = 0.
            kinds[stopped] = 10
            from flat_boundary import constrain
            goals = constrain(self, chosen, goals)
            # Reflections exchange cardinal labels; keep actual coverage truthful.
            kinds = torch.where((kinds == 0) | (kinds == 1), torch.where(goals[:, 0] >= 0, 0, 1), kinds)
            kinds = torch.where((kinds == 2) | (kinds == 3), torch.where(goals[:, 1] >= 0, 2, 3), kinds)
            initial = self.omni_resetting[chosen]
            ramp = initial & (torch.rand(len(chosen), device=self.device) < .5)
            self.omni_goal[chosen] = goals
            self.omni_ramp[chosen] = ramp
            self.omni_started[chosen] = self._env.common_step_counter
            self.time_left[chosen] = self.cfg.resampling_time
            self.last_is_limit_vel[chosen] = False
            mean_factor = 1. - ramp.float() * RAMP_S / (2 * self.cfg.resampling_time)
            # Replace all inherited integral changes on flat, including startup ramps.
            self.commands_xy_accumulation[chosen] = integral_before[flat] + goals[:, :2]*mean_factor[:, None]
            key = self.learner[chosen]*len(NAMES) + kinds
            self.omni_counts.view(-1).index_add_(0, key, torch.ones_like(key))
            new_stop = (goals == 0).all(-1)
            for group in (0, 1):
                same = (self.learner[chosen] == group) & ~initial
                self.omni_transitions[group, 0] += (same & ~old_stop[flat] & new_stop).sum()
                self.omni_transitions[group, 1] += (same & old_stop[flat] & ~new_stop).sum()
            self._apply_omni(chosen)

        def _apply_omni(self, ids):
            elapsed = (self._env.common_step_counter-self.omni_started[ids])*self._env.step_dt
            factor = torch.where(self.omni_ramp[ids], (elapsed/RAMP_S).clamp(0., 1.), 1.)
            self.commands[ids] = self.omni_goal[ids]*factor[:, None]

        def _update_command(self):
            super()._update_command()
            flat = self.terrain_idxs == self.terrain_type2idx['flat']
            self._apply_omni(flat.nonzero().flatten())
            from flat_boundary import check
            check(self)

        def omni_snapshot(self):
            from flat_boundary import snapshot as boundary_snapshot
            flat = self.terrain_idxs == self.terrain_type2idx['flat']
            return dict(profile=PROFILE, segment_s=5., episode_s=25., boundary=boundary_snapshot(self),
                        sequence_fraction=.2, sampled_probabilities=dict(zip(NAMES, PROBABILITIES)),
                        kind_counts_by_learner=self.omni_counts.cpu().tolist(),
                        stop_restart_by_learner=self.omni_transitions.cpu().tolist(),
                        current_full_zero=int((flat & (self.commands == 0).all(-1)).sum()),
                        current_pure_yaw=int((flat & (self.commands[:, :2] == 0).all(-1)
                                             & (self.commands[:, 2] != 0)).sum()))
    return OmniCommand


def configure(cfg):
    if cfg.episode_length_s != 25. or cfg.commands.base_velocity.resampling_time != 5.:
        raise ValueError('V19 requires full 25s episodes and 5s segments')
    cfg.commands.base_velocity.class_type = command_class(cfg.commands.base_velocity.class_type)
    cfg.omni_command_profile = PROFILE
