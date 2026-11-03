"""Explicit flat startup coverage for the V13 full-episode experiment.

Tensor helpers can be tested without Isaac. The lazy command class only changes
the first command of selected flat normal-reset episodes; all later commands,
stairs and pit phases retain the existing MixedCommand implementation.
"""
from __future__ import annotations

import torch


KIND_NAMES = (
    "forward", "backward", "left", "right", "yaw_left", "yaw_right",
    "diagonal_a", "diagonal_b", "longitudinal_turn", "lateral_turn",
)


def sample_explicit_goals(
    count: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(goals, kinds)`` for ten balanced explicit primitives.

    Kind ids are sampled uniformly. Magnitudes stay inside the existing V6
    anchor domain: vx in [.2, 1.2], vy in [.2, .5], yaw in [.3, .8].  Every
    combination component is signed symmetrically. Cardinal kinds 0..5 have
    fixed signs, paired across kinds; absent components are exactly zero.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    kinds = torch.randint(len(KIND_NAMES), (count,), device=device, generator=generator)
    vx = 0.2 + torch.rand(count, device=device, generator=generator)
    vy = 0.2 + 0.3 * torch.rand(count, device=device, generator=generator)
    yaw = 0.3 + 0.5 * torch.rand(count, device=device, generator=generator)
    signs = torch.where(torch.rand(count, 3, device=device, generator=generator) < 0.5, -1.0, 1.0)
    goals = torch.zeros(count, 3, device=device)
    goals[:, 0] = torch.where(kinds == 0, vx, torch.where(kinds == 1, -vx, 0.0))
    goals[:, 1] = torch.where(kinds == 2, vy, torch.where(kinds == 3, -vy, 0.0))
    goals[:, 2] = torch.where(kinds == 4, yaw, torch.where(kinds == 5, -yaw, 0.0))
    goals[:, 0] += torch.where((kinds == 6) | (kinds == 8), vx, torch.where(kinds == 7, -vx, 0.0))
    goals[:, 1] += torch.where((kinds == 6) | (kinds == 9), vy, torch.where(kinds == 7, -vy, 0.0))
    goals[:, 2] += torch.where((kinds == 8) | (kinds == 9), yaw, 0.0)
    # Kinds 0..5 have fixed semantic signs. Combination kinds only use random signs.
    combo = kinds >= 6
    goals = torch.where(combo[:, None], goals.abs() * signs, goals)
    return goals, kinds


def explicit_selection(cohort_size: int, *, device: torch.device | str = "cpu", generator: torch.Generator | None = None) -> torch.Tensor:
    """Select half of one complete fixed cohort once; reuse its global mask on resets."""
    if cohort_size < 0:
        raise ValueError("cohort_size must be non-negative")
    selected = torch.zeros(cohort_size, dtype=torch.bool, device=device)
    if cohort_size:
        order = torch.randperm(cohort_size, device=device, generator=generator)
        selected[order[: cohort_size // 2]] = True
    return selected


def reset_selection(reset_ids: torch.Tensor, global_mask: torch.Tensor) -> torch.Tensor:
    """Return selected reset ids from a precomputed global cohort mask."""
    ids = reset_ids.to(device=global_mask.device, dtype=torch.long)
    if ids.numel() and (ids.min() < 0 or ids.max() >= global_mask.numel()):
        raise IndexError("reset id outside global mask")
    return ids[global_mask[ids]]


RAMP_SECONDS = 0.2
SAMPLING_VERSION = 'explicit_full_episode_v13'


def command_class(base_class=None):
    if base_class is None:
        from mixed_task import MixedCommand
        base_class = MixedCommand

    class ExplicitStartupCommand(base_class):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.startup_goal = torch.zeros_like(self.commands)
            self.startup_kind = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
            self.startup_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self.startup_ramp = torch.zeros_like(self.startup_active)
            self.startup_selected = torch.zeros_like(self.startup_active)
            self.startup_pending_reset = torch.zeros_like(self.startup_active)
            self.startup_started_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.startup_reset_counts = torch.zeros((2, 10, 2), device=self.device, dtype=torch.long)
            self._select_startup_cohorts()

        def _select_startup_cohorts(self):
            from startup_repair import learner_cohort_mask
            cohort = learner_cohort_mask(self.learner)
            # A private generator makes membership stable across restarts without
            # consuming or reseeding the simulation/training random stream.
            generator = torch.Generator(device=self.device).manual_seed(42)
            self.startup_selected.zero_()
            for group in (0, 1):
                members = (cohort & (self.learner == group)).nonzero().flatten()
                selected = explicit_selection(len(members), device=self.device, generator=generator)
                self.startup_selected[members[selected]] = True

        def set_student_indices(self, indices):
            super().set_student_indices(indices)
            self._select_startup_cohorts()
            self.startup_reset_counts.zero_()

        def reset(self, env_ids=None):
            ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None
                   else torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
            # Command.reset runs before episode_length_buf is cleared. An explicit
            # reset marker, not episode age, distinguishes initial from 5 s samples.
            self.startup_pending_reset[ids] = True
            try:
                return super().reset(ids)
            finally:
                self.startup_pending_reset[ids] = False

        def _resample(self, env_ids):
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self.startup_active[ids] = False
            self.startup_kind[ids] = -1
            super()._resample(ids)
            flat = self.terrain_idxs[ids] == self.terrain_type2idx['flat']
            chosen = ids[flat & self.startup_selected[ids] & self.startup_pending_reset[ids]]
            if not len(chosen):
                return
            goals, kinds = sample_explicit_goals(len(chosen), device=self.device)
            ramp = torch.rand(len(chosen), device=self.device) < 0.5
            native = self.commands[chosen, :2].clone()
            self.startup_goal[chosen] = goals
            self.startup_kind[chosen] = kinds
            self.startup_ramp[chosen] = ramp
            self.startup_active[chosen] = True
            self.startup_started_step[chosen] = self._env.common_step_counter
            self.time_left[chosen] = self.cfg.resampling_time
            self.last_is_limit_vel[chosen] = False
            # Upstream stores command integrals divided by the 5 s period for
            # terrain curriculum accounting. Replace its just-sampled command,
            # including the triangular area lost during the optional 0.2 s ramp.
            mean_factor = 1.0 - ramp.float() * RAMP_SECONDS / (2*self.cfg.resampling_time)
            self.commands_xy_accumulation[chosen] += goals[:, :2]*mean_factor[:, None]-native
            key = (self.learner[chosen]*10+kinds)*2+ramp.long()
            self.startup_reset_counts.view(-1).index_add_(0, key, torch.ones_like(key))
            self._apply_startup_commands(chosen)

        def _apply_startup_commands(self, ids):
            elapsed = (self._env.common_step_counter-self.startup_started_step[ids])*self._env.step_dt
            factor = torch.where(self.startup_ramp[ids], (elapsed/RAMP_SECONDS).clamp(0., 1.), 1.)
            self.commands[ids] = self.startup_goal[ids]*factor[:, None]

        def _update_command(self):
            super()._update_command()
            self._apply_startup_commands(self.startup_active.nonzero().flatten())

        def startup_sampling_snapshot(self):
            return dict(version=SAMPLING_VERSION, ramp_seconds=RAMP_SECONDS,
                        explicit_fraction_of_normal_cohort=0.5,
                        ramp_probability=0.5, kind_names=list(KIND_NAMES),
                        reset_counts=self.startup_reset_counts.cpu().tolist(),
                        selected_by_learner=[int((self.startup_selected & (self.learner == i)).sum())
                                             for i in (0, 1)],
                        active_count=int(self.startup_active.sum()))

    return ExplicitStartupCommand
