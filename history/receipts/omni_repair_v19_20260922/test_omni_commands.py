from types import SimpleNamespace
import torch
import omni_commands
from omni_commands import (
    NAMES,
    PROBABILITIES,
    command_class,
    sample_goals,
    scheduled_stop,
    truthful_kinds,
)


def test_samples_cover_stop_pure_yaw_and_cardinal_directions():
    assert abs(sum(PROBABILITIES)-1.) < 1e-8
    goals, kinds = sample_goals(20000, generator=torch.Generator().manual_seed(42))
    assert set(kinds.tolist()) == set(range(len(NAMES)))
    assert bool((goals[kinds == 10] == 0).all())
    assert bool((goals[(kinds == 4) | (kinds == 5), :2] == 0).all())
    assert bool((goals[kinds == 4, 2] >= .3).all())
    assert bool((goals[kinds == 5, 2] <= -.3).all())
    assert bool((goals[:, 0].abs() <= 1.).all())
    assert bool((goals[:, 1].abs() <= .6).all())
    assert bool((goals[:, 2].abs() <= 1.).all())
    _, moving = sample_goals(2000, moving_only=True)
    assert not bool((moving == 10).any())


def test_sequence_five_full_segments_no_episode_termination():
    assert scheduled_stop(torch.ones(5, dtype=torch.bool), torch.arange(5)).tolist() == [False, True, False, True, False]


def test_truthful_labels_follow_reflected_cardinal_and_yaw_signs():
    goals = torch.tensor([[-.2, 0., 0.], [.2, 0., 0.], [0., -.2, 0.],
                          [0., .2, 0.], [0., 0., -.3], [0., 0., .3]])
    kinds = torch.tensor([0, 1, 2, 3, 4, 5])
    assert truthful_kinds(goals, kinds).tolist() == [1, 0, 3, 2, 5, 4]


class Base:
    def __init__(self, cfg, env):
        self.cfg, self._env, self.num_envs, self.device = cfg, env, 20, 'cpu'
        self.learner = torch.arange(20) % 2
        self.commands = torch.zeros(20, 3)
        self.commands_xy_accumulation = torch.zeros(20, 2)
        self.terrain_idxs = torch.tensor([0]*18+[1, 2])
        self.terrain_type2idx = {'flat': 0, 'stairs': 1, 'pit': 2}
        self.startup_active = torch.ones(20, dtype=torch.bool)
        self.time_left = torch.zeros(20)
        self.last_is_limit_vel = torch.zeros(20, dtype=torch.bool)
        self.flat_band_bounds = (torch.tensor([-100., -100.]), torch.tensor([100., 100.]))
        self.flat_boundary_reflections = self.flat_boundary_crossings = 0
        quaternion = torch.zeros(20, 4)
        quaternion[:, 0] = 1.
        self.robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(20, 3), root_quat_w=quaternion))

    def set_student_indices(self, indices):
        pass

    def _resample(self, ids):
        self.commands[ids] = 9.
        self.commands_xy_accumulation[ids] += 9.

    def reset(self, ids):
        self.commands_xy_accumulation[ids] = 0.
        self._resample(ids)

    def _update_command(self):
        pass


def test_wrapping_preserves_nonflat_and_integral_and_episode_sequence():
    env = SimpleNamespace(common_step_counter=0, step_dt=.02)
    sampler = command_class(Base)(SimpleNamespace(resampling_time=5.), env)
    ids = torch.arange(20)
    sampler.reset(ids)
    flat = torch.arange(18)
    assert bool((sampler.commands[18:] == 9.).all())
    assert bool((sampler.commands_xy_accumulation[18:] == 9.).all())
    assert not bool(sampler.startup_active[flat].any())
    expect = sampler.omni_goal[flat, :2]*(1.-sampler.omni_ramp[flat, None].float()*.02)
    torch.testing.assert_close(sampler.commands_xy_accumulation[flat], expect)
    env.common_step_counter = 250
    before = sampler.commands_xy_accumulation.clone()
    sampler._resample(ids)
    seq = sampler.omni_sequence & (sampler.terrain_idxs == 0)
    assert bool((sampler.commands[seq] == 0.).all())
    torch.testing.assert_close(sampler.commands_xy_accumulation[flat], before[flat]+sampler.omni_goal[flat, :2])
    env.common_step_counter = 500
    sampler._resample(ids)
    assert bool((sampler.commands[seq].abs().sum(-1) > 0).all())
    assert bool((sampler.omni_transitions.sum(0) > 0).all())
    sampler.reset(ids)
    assert bool((sampler.omni_segment[flat] == 0).all())
    assert bool((sampler.omni_segment[18:] == -1).all())


def test_four_failed_resamples_use_legal_cardinal_recovery_and_truthful_counts(monkeypatch):
    def impossible(count, device='cpu', generator=None, moving_only=False):
        return torch.tensor([[1., 0., 0.]]).repeat(count, 1), torch.zeros(count, dtype=torch.long)

    monkeypatch.setattr(omni_commands, 'sample_goals', impossible)
    env = SimpleNamespace(common_step_counter=0, step_dt=.02)
    sampler = command_class(Base)(SimpleNamespace(resampling_time=5.), env)
    sampler.flat_band_bounds = (torch.tensor([-2.3, -2.3]), torch.tensor([2.3, 2.3]))
    sampler.reset(torch.arange(20))
    flat = torch.arange(18)
    assert sampler.flat_boundary_resamples == 4 * len(flat)
    assert sampler.flat_boundary_cardinal_recoveries == len(flat)
    torch.testing.assert_close(sampler.omni_goal[flat].norm(dim=-1), torch.full((18,), .2))
    assert bool((sampler.omni_goal[flat, 2] == 0).all())
    # Recovery candidate order breaks the center tie toward truthful forward.
    assert sampler.omni_counts[:, 0].sum() == len(flat)
