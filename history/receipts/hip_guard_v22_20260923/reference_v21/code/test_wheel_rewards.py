"""Physical invariants for the wheel adaptation, independent of Isaac Sim."""

import math

import torch

from wheel_rewards import wheel_contact_velocity, wheel_regulation_penalty


def test_forward_and_reverse_pure_rolling_have_stationary_contact():
    radius = 0.093
    linear = torch.tensor([[[1.0, 0.0, 0.0]], [[-2.0, 0.0, 0.0]]])
    angular = torch.zeros_like(linear)
    angular[..., 1] = linear[..., 0] / radius
    contact_velocity = wheel_contact_velocity(
        linear, angular, radius, torch.tensor([0.0, 0.0, -1.0])
    )
    torch.testing.assert_close(contact_velocity, torch.zeros_like(linear))


def test_stalled_wheel_and_lateral_slide_retain_contact_speed():
    linear = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.7, 0.0]]])
    contact_velocity = wheel_contact_velocity(
        linear, torch.zeros_like(linear), 0.093, torch.tensor([0.0, 0.0, -1.0])
    )
    torch.testing.assert_close(contact_velocity, linear)


def test_contact_velocity_uses_world_gravity_direction():
    # Rotate the entire forward-rolling setup by 90 degrees around world X.
    linear = torch.tensor([[[1.0, 0.0, 0.0]]])
    angular = torch.tensor([[[0.0, 0.0, 10.0]]])
    contact_velocity = wheel_contact_velocity(
        linear, angular, 0.1, torch.tensor([0.0, 1.0, 0.0])
    )
    torch.testing.assert_close(contact_velocity, torch.zeros_like(linear))


def test_ground_contact_and_lifted_wheel_keep_upstream_height_weight():
    radius = 0.093
    base_height = 0.46
    decay_height = 0.025 * base_height
    # Both wheels slide at 2 m/s: one on the ground, one raised by one
    # exponential decay length. Wheel CENTER height must not suppress contact.
    center_positions = torch.tensor(
        [[[0.0, 0.0, radius]], [[0.0, 0.0, radius + decay_height]]]
    )
    linear = torch.tensor([[[2.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
    result = wheel_regulation_penalty(
        center_positions,
        linear,
        torch.zeros_like(linear),
        torch.tensor([[0.0, 0.0, base_height]]).repeat(2, 1),
        torch.full((2,), base_height),
        torch.tensor([0.0, 0.0, -9.81]),
        base_height,
        radius,
    )
    torch.testing.assert_close(result, torch.tensor([4.0, 4.0 / math.e]))


def test_rolling_wheels_are_zero_and_slipping_wheels_sum_independently():
    radius = 0.093
    centers = torch.tensor([[[0.0, 0.0, radius]]]).repeat(2, 4, 1)
    linear = torch.zeros_like(centers)
    linear[..., 0] = 1.0
    angular = torch.zeros_like(linear)
    angular[0, :, 1] = 1.0 / radius
    result = wheel_regulation_penalty(
        centers,
        linear,
        angular,
        torch.tensor([[0.0, 0.0, 0.46]]).repeat(2, 1),
        torch.full((2,), 0.46),
        torch.tensor([0.0, 0.0, -9.81]),
        0.46,
        radius,
    )
    torch.testing.assert_close(result, torch.tensor([0.0, 4.0]))
