import pytest
import torch

from guard_math import bounded_position_target, guard_costs


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_motor_specific_normalization_and_float_outputs(dtype):
    torque_limits = torch.tensor([120.0, 17.0], dtype=dtype)
    speed_limits = torch.tensor([17.48, 44.0], dtype=dtype)
    applied = torque_limits * 0.5
    costs = guard_costs(applied, applied, speed_limits * 0.5, torque_limits, speed_limits)
    torch.testing.assert_close(costs["torque_mean_square"], torch.tensor(0.25, dtype=dtype))
    assert costs["speed_excess"].item() == 0
    assert costs["saturation"].item() == 0
    assert all(cost.dtype == dtype for cost in costs.values())


def test_costs_are_symmetric_when_motion_and_torque_signs_reverse():
    applied = torch.tensor([[30.0, -12.0], [-80.0, 14.0]])
    computed = torch.tensor([[55.0, 15.0], [-140.0, 20.0]])
    velocity = torch.tensor([[18.0, -50.0], [-10.0, 30.0]])
    limits = torch.tensor([120.0, 17.0])
    speeds = torch.tensor([17.48, 44.0])
    positive = guard_costs(applied, computed, velocity, limits, speeds)
    negative = guard_costs(-applied, -computed, -velocity, limits, speeds)
    for name in positive:
        torch.testing.assert_close(positive[name], negative[name])


def test_speed_margin_is_unpenalized_through_eighty_percent_then_quadratic():
    velocity = torch.tensor([[0.0, -10.0], [8.0, -16.0], [9.0, -24.0]])
    costs = guard_costs(torch.zeros_like(velocity), torch.zeros_like(velocity), velocity,
                       torch.tensor([100.0, 20.0]), torch.tensor([10.0, 20.0]))
    torch.testing.assert_close(costs["speed_excess"], torch.tensor([0.0, 0.0, 0.17]))


def test_transient_speed_cost_uses_ninety_percent_margin_without_changing_default():
    velocity = torch.tensor([[13.5, 27.0], [15.6, 0.0]])
    zeros = torch.zeros_like(velocity)
    speed = torch.tensor([15.0, 30.0])
    default = guard_costs(zeros, zeros, velocity, 1.0, speed)
    transient = guard_costs(zeros, zeros, velocity, 1.0, speed, speed_penalty_fraction=0.9)
    torch.testing.assert_close(transient["speed_excess"], torch.tensor([0.0, (15.6 / 15 - 0.9) ** 2]))
    assert default["speed_excess"][0] > 0


def test_saturation_keeps_sign_and_does_not_hide_opposing_torques():
    applied = torch.tensor([[5.0, -5.0], [5.0, -5.0]])
    computed = torch.tensor([[-5.0, 5.0], [5.0, -5.0]])
    costs = guard_costs(applied, computed, torch.zeros_like(applied), 10.0, 20.0)
    torch.testing.assert_close(costs["saturation"], torch.tensor([1.0, 0.0]))


def test_large_excursions_are_not_clipped():
    applied = torch.tensor([[1000.0]])
    computed = torch.tensor([[2000.0]])
    velocity = torch.tensor([[100.0]])
    costs = guard_costs(applied, computed, velocity, 10.0, 1.0)
    torch.testing.assert_close(costs["torque_mean_square"], torch.tensor([10000.0]))
    torch.testing.assert_close(costs["saturation"], torch.tensor([10000.0]))
    torch.testing.assert_close(costs["speed_excess"], torch.tensor([99.2**2]))


def test_load_reference_is_independent_of_hard_torque_and_saturation_limits():
    applied = torch.tensor([[70.0, -12.0]])
    computed = torch.tensor([[190.0, -29.0]])
    hard = torch.tensor([120.0, 17.0])
    reference = torch.tensor([35.0, 6.0])
    velocity = torch.tensor([[15.0, -30.0]])
    costs = guard_costs(applied, computed, velocity, hard, torch.tensor([15.0, 30.0]), reference)
    torch.testing.assert_close(costs["torque_mean_square"], torch.tensor([4.0]))
    torch.testing.assert_close(costs["saturation"], torch.tensor([1.0]))
    torch.testing.assert_close(costs["speed_excess"], torch.tensor([0.08]))

    hard_normalized = guard_costs(applied, computed, velocity, hard, torch.tensor([15.0, 30.0]))
    assert costs["torque_mean_square"].item() > hard_normalized["torque_mean_square"].item()
    torch.testing.assert_close(costs["saturation"], hard_normalized["saturation"])
    torch.testing.assert_close(costs["speed_excess"], hard_normalized["speed_excess"])


def test_explicit_load_reference_matching_hard_limit_preserves_legacy_costs():
    applied = torch.tensor([[30.0, -5.0], [-34.0, 5.5]])
    computed = applied * 1.5
    velocity = torch.tensor([[13.0, -28.0], [-16.0, 31.0]])
    hard = torch.tensor([35.0, 6.0])
    speed_limits = torch.tensor([15.0, 30.0])
    legacy = guard_costs(applied, computed, velocity, hard, speed_limits)
    explicit = guard_costs(applied, computed, velocity, hard, speed_limits, hard)
    for name in legacy:
        torch.testing.assert_close(legacy[name], explicit[name])


def test_targets_respect_broadcast_joint_bounds_and_per_sample_change_limits():
    previous = torch.tensor([[0.9, -1.9, 0.0], [-0.9, 1.9, 0.0]])
    requested = torch.tensor([[4.0, -5.0, 0.8], [-4.0, 5.0, -0.8]])
    lower = torch.tensor([-1.0, -2.0, -0.5])
    upper = torch.tensor([1.0, 2.0, 0.5])
    delta = torch.tensor([[0.2], [0.1]])
    actual = bounded_position_target(requested, previous, lower, upper, delta)
    torch.testing.assert_close(actual, torch.tensor([[1.0, -2.0, 0.2], [-1.0, 2.0, -0.1]]))
    assert torch.all(actual >= lower)
    assert torch.all(actual <= upper)
    assert torch.all((actual - previous).abs() <= delta + 1e-6)


def test_out_of_bounds_previous_is_projected_before_rate_limiting_without_mutation():
    previous = torch.tensor([[5.0, -5.0]])
    requested = torch.tensor([[-5.0, 5.0]])
    previous_copy = previous.clone()
    requested_copy = requested.clone()
    actual = bounded_position_target(requested, previous, -1.0, 1.0, 0.2)
    torch.testing.assert_close(actual, torch.tensor([[0.8, -0.8]]))
    torch.testing.assert_close(previous, previous_copy)
    torch.testing.assert_close(requested, requested_copy)


def test_zero_delta_holds_projected_previous_and_supports_fixed_joint_bounds():
    previous = torch.tensor([2.0, -0.3])
    requested = torch.tensor([-2.0, 0.3])
    actual = bounded_position_target(requested, previous, torch.tensor([0.5, -1.0]),
                                     torch.tensor([0.5, 1.0]), 0.0)
    torch.testing.assert_close(actual, torch.tensor([0.5, -0.3]))


def test_one_environment_reset_cannot_change_another_environments_targets_or_costs():
    previous = torch.tensor([[0.0, 0.0], [0.2, -0.2]])
    requested = torch.tensor([[1.0, -1.0], [0.4, -0.4]])
    original = bounded_position_target(requested, previous, -1.0, 1.0, 0.1)
    previous[0] = torch.tensor([10.0, -10.0])
    requested[0] = -requested[0]
    changed = bounded_position_target(requested, previous, -1.0, 1.0, 0.1)
    torch.testing.assert_close(original[1], changed[1])
    assert not torch.equal(original[0], changed[0])

    limits = torch.tensor([120.0, 17.0])
    applied = torch.tensor([[30.0, 5.0], [60.0, 8.5]])
    computed = applied + 1.0
    velocity = torch.tensor([[10.0, 10.0], [15.0, 40.0]])
    before = guard_costs(applied, computed, velocity, limits, torch.tensor([17.48, 44.0]))
    applied[0] *= 100.0
    computed[0] *= -100.0
    velocity[0] *= 100.0
    after = guard_costs(applied, computed, velocity, limits, torch.tensor([17.48, 44.0]))
    for name in before:
        assert before[name].shape == (2,)
        torch.testing.assert_close(before[name][1], after[name][1])
        assert before[name][0] != after[name][0]
