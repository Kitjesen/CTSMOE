import torch
from torch import nn

from anchor_objective import (
    deterministic_mean,
    gradient_ratio_coefficient,
    masked_action_loss,
    normal_start_mask,
)


class _StudentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, obs):
        return self.linear(obs), torch.zeros(obs.shape[0], 1)


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.single_obs_normalizer = nn.Identity()
        self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()
        self.teacher_encoder = nn.Linear(2, 2, bias=False)
        self.student_moe_encoder = _StudentEncoder()
        self.actor = nn.Linear(4, 16, bias=False)
        self.state_dependent_std = False

    @staticmethod
    def get_actor_obs(obs):
        return obs["actor_obs"]

    @staticmethod
    def get_critic_obs(obs):
        return obs["critic_obs"]


def test_normal_start_mask_selects_only_finite_flat_samples_in_valid_window():
    flat = torch.tensor([True, True, True, True, False, True, True, True])
    age = torch.tensor([0.0, 0.999, 1.0, -0.01, 0.2, 0.2, 0.2, float("nan")])
    command = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.2, -0.5, 0.8],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.2001, 0.0, 0.0],
            [0.0, float("inf"), 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    selected = normal_start_mask(flat, age, command)

    torch.testing.assert_close(
        selected,
        torch.tensor([True, True, False, False, False, False, False, False]),
    )


def test_masked_action_loss_normalizes_physical_error_and_detaches_reference():
    current = torch.zeros((2, 16), requires_grad=True)
    reference = torch.zeros((2, 16), requires_grad=True)
    with torch.no_grad():
        current[0, 3] = 2.0
        current[0, 11] = -1.0
        reference[0, 3] = 1.0
    mask = torch.tensor([True, False])
    action_scales = torch.arange(1.0, 17.0)
    physical_scales = action_scales * 2.0

    loss = masked_action_loss(
        current,
        reference,
        mask,
        action_scales,
        physical_scales,
    )
    loss.backward()

    # Every channel is normalized independently; no leg/wheel order is assumed.
    expected = ((0.5**2) + ((-1.0 * 12.0 / 24.0) ** 2)) / 16.0
    torch.testing.assert_close(loss, torch.tensor(expected))
    assert current.grad is not None
    assert reference.grad is None


def test_masked_action_loss_empty_mask_returns_connected_zero():
    current = torch.randn((3, 16), requires_grad=True)

    loss = masked_action_loss(
        current,
        torch.randn((3, 16)),
        torch.zeros(3, dtype=torch.bool),
        torch.ones(16),
        torch.ones(16),
    )
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(current.grad, torch.zeros_like(current))


def test_deterministic_mean_routes_gradients_through_selected_encoder():
    policy = _Policy()
    obs = {
        "single_obs": torch.ones(2, 2),
        "actor_obs": torch.ones(2, 2),
        "critic_obs": torch.ones(2, 2),
    }

    deterministic_mean(policy, obs, is_teacher=False).sum().backward()
    assert policy.student_moe_encoder.linear.weight.grad is not None
    assert policy.teacher_encoder.weight.grad is None
    assert policy.actor.weight.grad is not None

    policy.zero_grad(set_to_none=True)
    deterministic_mean(policy, obs, is_teacher=True).sum().backward()
    assert policy.teacher_encoder.weight.grad is not None
    assert policy.student_moe_encoder.linear.weight.grad is None
    assert policy.actor.weight.grad is not None


def test_deterministic_mean_can_freeze_actor_without_blocking_latent_gradient():
    policy = _Policy()
    obs = {
        "single_obs": torch.ones(2, 2),
        "actor_obs": torch.ones(2, 2),
        "critic_obs": torch.ones(2, 2),
    }

    deterministic_mean(
        policy,
        obs,
        is_teacher=False,
        freeze_actor=True,
    ).sum().backward()

    assert policy.student_moe_encoder.linear.weight.grad is not None
    assert policy.actor.weight.grad is None


def test_deterministic_mean_can_detach_student_encoder_only():
    policy = _Policy()
    obs = {
        "single_obs": torch.ones(2, 2),
        "actor_obs": torch.ones(2, 2),
        "critic_obs": torch.ones(2, 2),
    }

    deterministic_mean(
        policy,
        obs,
        is_teacher=False,
        detach_student=True,
    ).sum().backward()

    assert policy.student_moe_encoder.linear.weight.grad is None
    assert policy.actor.weight.grad is not None


def test_masked_action_loss_keeps_frozen_reference_without_gradients():
    current_policy = _Policy()
    reference_policy = _Policy().eval()
    reference_policy.requires_grad_(False)
    obs = {
        "single_obs": torch.ones(2, 2),
        "actor_obs": torch.ones(2, 2),
        "critic_obs": torch.ones(2, 2),
    }
    current = deterministic_mean(current_policy, obs, is_teacher=False)
    reference = deterministic_mean(reference_policy, obs, is_teacher=False)

    loss = masked_action_loss(
        current,
        reference,
        torch.ones(2, dtype=torch.bool),
        torch.ones(16),
        torch.ones(16),
    )
    loss.backward()

    assert current_policy.actor.weight.grad is not None
    assert all(parameter.grad is None for parameter in reference_policy.parameters())


def test_gradient_ratio_coefficient_matches_gradient_norm_ratio_without_side_effects():
    parameter = nn.Parameter(torch.tensor([2.0, -1.0]))
    primary_loss = (2.0 * parameter).sum()
    anchor_loss = (4.0 * parameter).sum()

    coefficient = gradient_ratio_coefficient(
        primary_loss,
        anchor_loss,
        [parameter],
        ratio=0.1,
        max_coefficient=1.0,
    )

    assert coefficient == 0.05
    assert parameter.grad is None


def test_gradient_ratio_coefficient_returns_zero_for_zero_anchor_gradient():
    parameter = nn.Parameter(torch.tensor([2.0]))

    coefficient = gradient_ratio_coefficient(
        (2.0 * parameter).sum(),
        (0.0 * parameter).sum(),
        [parameter],
    )

    assert coefficient == 0.0
    assert parameter.grad is None
