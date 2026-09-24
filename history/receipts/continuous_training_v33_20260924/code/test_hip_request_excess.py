"""Unit contract for the isolated V24 inward-request reward candidate."""

import pytest
import torch
from types import SimpleNamespace

from hip_request_excess import excess_per_environment, reward


def test_no_cost_inside_existing_motion_envelope_including_new_default():
    requests = torch.tensor([[.12, -.12, -.12, .12],
                             [.369, -.369, -.369, .369],
                             [-.30, .30, .30, -.30]])
    signs = torch.tensor([1., -1., -1., 1.])
    torch.testing.assert_close(excess_per_environment(requests, [0, 1, 2, 3], signs),
                               torch.zeros(3))


def test_only_unreachable_inward_excess_is_costly_and_graded():
    requests = torch.tensor([[1.369, -.869, 0., .369],
                             [.469, -.369, -.469, .369]])
    signs = torch.tensor([1., -1., -1., 1.])
    result = excess_per_environment(requests, [0, 1, 2, 3], signs)
    torch.testing.assert_close(result, torch.tensor([1.5, .2]))


def test_rejects_nonhip_or_unbatched_inputs():
    with pytest.raises(ValueError):
        excess_per_environment(torch.zeros(2, 4), [0, 1, 2], torch.ones(3))
    with pytest.raises(ValueError):
        excess_per_environment(torch.zeros(4), [0, 1, 2, 3], torch.ones(4))


def test_runtime_reward_uses_robot_joint_names_not_action_column_order():
    names = ["fr_hip_joint", "fl_hip_joint", "rr_hip_joint", "rl_hip_joint", "other_joint"]
    # Action output is deliberately shuffled and contains a non-hip joint.
    term = SimpleNamespace(joint_indices=[4, 2, 0, 3, 1],
                           requested_actions=torch.tensor([[9.0, 0.769, -0.569, -0.969, 0.469]]))
    env = SimpleNamespace(device="cpu", scene={"robot": SimpleNamespace(joint_names=names)},
                          action_manager=SimpleNamespace(get_term=lambda name: term))
    # Four inward amplitudes are .569, .469, .769, .969; excess sum is 1.30 rad.
    torch.testing.assert_close(reward(env), torch.tensor([1.30]))
