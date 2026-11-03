from types import SimpleNamespace

import torch

from startup_repair import learner_cohort_mask
from startup_sampling import command_class, sample_explicit_goals


class BaseCommand:
    """Minimal native-command contract: reset samples; compute advances externally."""
    def __init__(self, cfg, env):
        self.cfg, self._env, self.device, self.num_envs = cfg, env, 'cpu', 4096
        self.commands = torch.zeros(self.num_envs, 3)
        self.commands_xy_accumulation = torch.zeros(self.num_envs, 2)
        self.time_left = torch.zeros(self.num_envs)
        self.last_is_limit_vel = torch.zeros(self.num_envs, dtype=torch.bool)
        self.learner = torch.zeros(self.num_envs, dtype=torch.long)
        self.terrain_idxs = torch.zeros_like(self.learner)
        self.terrain_type2idx = {'flat': 0, 'stairs_up': 1, 'pit': 2}
        self.terrain_idxs[::5] = 1
        self.terrain_idxs[::7] = 2

    def set_student_indices(self, ids):
        self.learner.zero_()
        self.learner[ids] = 1

    def reset(self, ids):
        self.commands_xy_accumulation[ids] = 0
        self._resample(ids)
        return {'native_reset': True}

    def _resample(self, ids):
        self.commands[ids] = torch.tensor([1.7, 0.8, -1.1])
        self.commands_xy_accumulation[ids] += self.commands[ids, :2]
        self.time_left[ids] = self.cfg.resampling_time
        self.last_is_limit_vel[ids] = True

    def _update_command(self):
        pass


def make_command():
    torch.manual_seed(11)
    env = SimpleNamespace(common_step_counter=5000, step_dt=.02)
    command = command_class(BaseCommand)(SimpleNamespace(resampling_time=5.), env)
    command.set_student_indices(torch.arange(0, 4096, 4))
    command.reset()
    return command


def test_flat_reset_coverage_both_learners_direct_and_ramp():
    c = make_command()
    normal = learner_cohort_mask(c.learner)
    assert not (c.startup_active & ~normal).any()
    assert not (c.startup_active & (c.terrain_idxs != 0)).any()
    for group in (0, 1):
        assert int((c.startup_selected & (c.learner == group)).sum()) == int((normal & (c.learner == group)).sum())//2
        assert torch.all(c.startup_reset_counts[group].sum(-1) > 0)
        assert torch.all(c.startup_reset_counts[group].sum(0) > 0)
    untouched = ~c.startup_active
    torch.testing.assert_close(c.commands[untouched], torch.tensor([1.7, .8, -1.1]).expand(int(untouched.sum()), -1))


def test_single_reset_remains_selected_and_does_not_change_other_envs():
    c = make_command()
    ids = c.startup_active.nonzero().flatten()[:1]
    before = c.commands.clone()
    count = int(c.startup_reset_counts.sum())
    assert c.reset(ids) == {'native_reset': True}
    assert c.startup_active[ids].all()
    assert int(c.startup_reset_counts.sum()) == count+1
    keep = torch.arange(c.num_envs) != ids[0]
    torch.testing.assert_close(c.commands[keep], before[keep], rtol=0, atol=0)


def test_ramp_reaches_goal_at_point_two_seconds_and_holds_until_resample():
    c = make_command()
    ramp, direct = c.startup_active & c.startup_ramp, c.startup_active & ~c.startup_ramp
    assert torch.all(c.commands[ramp] == 0)
    torch.testing.assert_close(c.commands[direct], c.startup_goal[direct])
    for steps, scale in ((5, .5), (10, 1.), (100, 1.)):
        c._env.common_step_counter = 5000+steps
        c._update_command()
        torch.testing.assert_close(c.commands[ramp], c.startup_goal[ramp]*scale)
    # At native 5 s resampling, stop overriding the new command.
    c._env.common_step_counter = 5250
    ids = c.startup_active.nonzero().flatten()
    c._resample(ids)
    c._update_command()
    assert not c.startup_active.any()
    torch.testing.assert_close(c.commands[ids], torch.tensor([1.7, .8, -1.1]).expand(len(ids), -1))


def test_command_integral_replaces_native_first_sample_for_curriculum():
    c = make_command()
    ids = c.startup_active.nonzero().flatten()
    factor = torch.where(c.startup_ramp[ids], .98, 1.)
    torch.testing.assert_close(c.commands_xy_accumulation[ids], c.startup_goal[ids, :2]*factor[:, None])
    assert torch.all(c.time_left[ids] == 5.)
    assert not c.last_is_limit_vel[ids].any()


def test_primitive_exact_zeros_and_anchor_bounds():
    goals, kinds = sample_explicit_goals(10000, generator=torch.Generator().manual_seed(7))
    components = ((1,0,0), (1,0,0), (0,1,0), (0,1,0), (0,0,1), (0,0,1),
                  (1,1,0), (1,1,0), (1,0,1), (0,1,1))
    assert (goals.abs() <= torch.tensor([1.2, .5, .8])).all()
    for kind, mask in enumerate(components):
        selected = goals[kinds == kind]
        assert len(selected) > 0
        assert (selected.ne(0) == torch.tensor(mask, dtype=torch.bool)).all()
        if kind < 6:
            assert (selected[:, kind//2]*(1 if kind % 2 == 0 else -1) > 0).all()
