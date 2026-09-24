import math
from types import SimpleNamespace

import pytest
import torch

import flat_boundary
import omni_commands
from omni_commands import NONZERO_ABS_RANGES, command_class, sample_goals
from omni_lift import LANDING_EVENT_WEIGHT, LIFT_EVENT_WEIGHT, command_categories, command_tracking
from test_omni_commands_v20 import Base


def test_sampled_probabilities_and_full_ranges_without_clamping():
    goals, kinds = sample_goals(60000, generator=torch.Generator().manual_seed(20260922))
    frequencies = torch.bincount(kinds, minlength=11) / len(kinds)
    torch.testing.assert_close(frequencies, torch.tensor(omni_commands.PROBABILITIES), atol=.006, rtol=0)
    for axis, (low, high) in enumerate(NONZERO_ABS_RANGES.values()):
        values = goals[:, axis].abs()
        values = values[values > 0]
        assert values.min() >= low
        assert values.max() <= high
        assert values.min() < low + .002
        assert values.max() > high - .002
        assert values.mean() == pytest.approx((low + high) / 2, abs=.015)
    lateral = (kinds == 2) | (kinds == 3)
    assert bool((goals[lateral, 0] == 0).all())
    assert bool((goals[lateral, 2] == 0).all())


def test_config_reports_flat_override_without_changing_other_terrain_ranges(monkeypatch):
    inherited = SimpleNamespace(lin_vel_x=(-1., 1.), lin_vel_y=(-1., 1.), ang_vel_yaw=(-1.5, 1.5))
    cfg = SimpleNamespace(episode_length_s=25., commands=SimpleNamespace(
        base_velocity=SimpleNamespace(class_type=Base, resampling_time=5., ranges=inherited)))
    called = []
    monkeypatch.setattr(flat_boundary, 'configure', lambda value: called.append(value))
    omni_commands.configure(cfg)
    assert called == [cfg]
    assert cfg.omni_command_profile == 'omni_full_episode_v20'
    assert cfg.omni_command_sampling['nonzero_abs_ranges'] == NONZERO_ABS_RANGES
    assert cfg.omni_command_sampling['scope'] == 'flat_only'
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (-1., 1.)


@pytest.mark.parametrize(('episode', 'segment'), [(24., 5.), (25., 4.)])
def test_segment_or_episode_duration_cannot_silently_change(episode, segment):
    cfg = SimpleNamespace(episode_length_s=episode, commands=SimpleNamespace(
        base_velocity=SimpleNamespace(resampling_time=segment)))
    with pytest.raises(ValueError, match='25s episodes and 5s segments'):
        omni_commands.configure(cfg)


def test_first_half_ramp_then_full_speed_and_five_segment_sequence(monkeypatch):
    def fixed(count, device='cpu', generator=None, moving_only=False):
        return torch.tensor([[2., 0., 0.]], device=device).repeat(count, 1), torch.zeros(count, dtype=torch.long)

    monkeypatch.setattr(omni_commands, 'sample_goals', fixed)
    monkeypatch.setattr(torch, 'rand', lambda count, device: torch.arange(count, device=device) % 2)
    env = SimpleNamespace(common_step_counter=0, step_dt=.02)
    sampler = command_class(Base)(SimpleNamespace(resampling_time=5.), env)
    for learner in (0, 1):
        mask = sampler.learner == learner
        assert sampler.omni_sequence[mask].sum() == 2
    ids, flat = torch.arange(20), torch.arange(18)
    sampler.reset(ids)
    ramp = sampler.omni_ramp.clone()
    assert ramp.sum() == 9
    assert bool((sampler.commands[flat[ramp[flat]]] == 0).all())
    env.common_step_counter = 5
    sampler._update_command()
    torch.testing.assert_close(sampler.commands[flat[ramp[flat]], 0], torch.ones(9))
    env.common_step_counter = 10
    sampler._update_command()
    torch.testing.assert_close(sampler.commands[flat, 0], torch.full((18,), 2.))
    for segment in range(1, 5):
        env.common_step_counter = segment * 250
        sampler._resample(ids)
        assert bool((sampler.time_left[flat] == 5.).all())
        assert not bool(sampler.omni_ramp[flat].any())
        seq = sampler.omni_sequence & (sampler.terrain_idxs == 0)
        assert bool((sampler.commands[seq, 0] == (0. if segment % 2 else 2.)).all())
    assert bool((sampler.omni_segment[flat] == 4).all())
    snapshot = sampler.omni_snapshot()
    assert snapshot['segment_s'] == 5. and snapshot['episode_s'] == 25.
    assert snapshot['sampling_contract']['nonzero_abs_ranges'] == NONZERO_ABS_RANGES
    assert snapshot['current_goal_nonzero_abs_ranges']['lin_vel_x'] == [2., 2.]
    assert snapshot['current_command_nonzero_abs_ranges']['lin_vel_x'] == [2., 2.]


def test_snapshot_distinguishes_ramp_values_from_goal_range():
    assert omni_commands.observed_nonzero_abs_ranges(torch.tensor([[.05, 0., .1]])) == {
        'lin_vel_x': [pytest.approx(.05), pytest.approx(.05)],
        'lin_vel_y': None,
        'ang_vel_yaw': [pytest.approx(.1), pytest.approx(.1)],
    }


@pytest.mark.parametrize('goal', [(2., 0., 0.), (2., 1., 0.), (2., 0., .2), (2., 0., 1.5)])
def test_max_speed_five_second_paths_fit_actual_band_center_at_every_heading(goal):
    # Frozen recovery01/PREFLIGHT_RUN.json records [-99,-27] to [99,0].
    lower, upper = torch.tensor([-97.8, -25.8]), torch.tensor([97.8, -1.2])
    yaw = torch.linspace(-math.pi, math.pi, 2049)
    position = ((lower + upper) / 2).repeat(len(yaw), 1)
    goals = torch.tensor([goal]).repeat(len(yaw), 1)
    result, _, resolved = flat_boundary.reflect_goals(position, yaw, goals, lower, upper)
    assert bool(resolved.all())
    torch.testing.assert_close(result, goals)
    assert 5. * math.hypot(2., 1.) < float((upper - lower).min()) / 2


def test_corner_infeasibility_remains_explicit_without_reducing_speed():
    lower, upper = torch.tensor([-97.8, -25.8]), torch.tensor([97.8, -1.2])
    position = (upper - .01)[None]
    goal = torch.tensor([[2., 0., 0.]])
    result, _, resolved = flat_boundary.reflect_goals(
        position, torch.tensor([-math.pi / 4]), goal, lower, upper)
    assert resolved.tolist() == [False]
    torch.testing.assert_close(result.abs(), goal.abs())


def test_new_low_yaw_threshold_keeps_tracking_gate_and_reward_weights():
    commands = torch.tensor([[0., 0., .2], [0., 0., -.2], [0., 0., .199], [0., 0., .2]])
    actual = commands.clone()
    actual[-1] = 0.
    result = command_tracking(commands, actual)
    assert result['pure_yaw'].tolist() == [True, True, False, True]
    assert result['tracking'].tolist() == [True, True, False, False]
    assert command_categories(commands).tolist() == [2, 3, 4, 2]
    assert LIFT_EVENT_WEIGHT == .50 and LANDING_EVENT_WEIGHT == 1.00
