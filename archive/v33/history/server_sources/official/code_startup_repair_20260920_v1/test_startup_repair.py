import pytest
import torch

from startup_repair import sample_goals, target_excursion, transition_command
from terrain_allocation import BASE, STARTUP_PROFILE, AllocationCourse, validate_weights


def test_all_direction_sampler_includes_sides_turns_and_signed_combinations():
    torch.manual_seed(31)
    goals = sample_goals(4096, 'cpu')
    assert (goals[:, 0] > 0).any() and (goals[:, 0] < 0).any()
    assert (goals[:, 1] > 0).any() and (goals[:, 1] < 0).any()
    pure_turn = goals[:, :2].norm(dim=-1) == 0
    assert (goals[pure_turn, 2] > 0).any() and (goals[pure_turn, 2] < 0).any()
    assert ((goals[:, :2].norm(dim=-1) > 0) & (goals[:, 2] != 0)).any()
    assert torch.all(goals.abs().amax(0) <= torch.tensor([1.2, .6, .9])+1e-6)


def test_stop_clears_all_axes_and_immediate_starts_are_not_hidden_by_a_ramp():
    age = torch.tensor([0., 1., 2.1, 4.1])
    goal = torch.tensor([[.6, .3, .5]]).expand(4, -1)
    cmd, _ = transition_command(age, torch.zeros(4), torch.full((4,), 2.), goal,
                                 torch.zeros(4, dtype=torch.bool))
    torch.testing.assert_close(cmd[0], goal[0])
    torch.testing.assert_close(cmd[2], torch.zeros(3))
    torch.testing.assert_close(cmd[3], goal[0])


def test_settle_and_ramp_are_distinct_from_immediate_start():
    goal = torch.ones(3, 3)
    cmd, _ = transition_command(torch.tensor([.5, 1.3, 1.6]), torch.ones(3),
                                 torch.full((3,), 2.), goal, torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(cmd[0], torch.zeros(3))
    torch.testing.assert_close(cmd[1], torch.full((3,), .5))
    torch.testing.assert_close(cmd[2], torch.ones(3))


def test_target_cost_preserves_targets_and_keeps_gait_deadband():
    target = torch.tensor([[.1, .2, .3], [.1, 1.6, .3]])
    before = target.clone()
    cost = target_excursion(target, torch.zeros_like(target), torch.tensor([.25, .45, .55]))
    assert cost[0] == 0 and cost[1] > 2
    torch.testing.assert_close(before, target)


def test_repair_allocation_has_explicit_bounds_and_original_bounds_stay_unchanged():
    weights = {**BASE, 'pit': .35, 'flat': .08}
    validate_weights(weights, STARTUP_PROFILE)
    with pytest.raises(ValueError):
        validate_weights(weights)
    course = AllocationCourse(profile=STARTUP_PROFILE, weights=weights)
    restored = AllocationCourse(**course.state_dict())
    restored.evaluate({}, 500)
    assert restored.weights == weights
