from types import SimpleNamespace

import torch

from flat_outlier import components, posture_cost, replace_flat_cost, reward


def test_concentrated_error_costs_more_than_equal_mean_distributed_error():
    q = torch.full((2, 12), .2)
    q[1] = 0.
    q[1, 0] = 2.4
    mean, tail = components(q, torch.zeros_like(q), torch.ones(12))
    torch.testing.assert_close(mean[0], mean[1])
    assert tail[1] > tail[0]*40
    assert posture_cost(q, 0., 1.)[1] > posture_cost(q, 0., 1.)[0]*4


def test_continuous_symmetric_and_no_state_mutation():
    q = torch.arange(12).float()[None]*.001
    saved = q.clone()
    c = posture_cost(q, 0., 1.)
    assert c > 0
    assert posture_cost(q*2, 0., 1.) > c*2
    assert posture_cost(torch.zeros_like(q), 0., 1.) == 0
    torch.testing.assert_close(posture_cost(-q.flip(-1), 0., 1.), c)
    torch.testing.assert_close(q, saved)


def test_nonflat_reward_and_state_are_exactly_preserved():
    old = torch.tensor([.3, .8, 0., 1.5])
    flat = torch.tensor([True, False, False, False])
    q = torch.full((4, 12), .5)
    result = replace_flat_cost(old, flat, q, torch.zeros_like(q), torch.ones(12))
    assert torch.equal(result[~flat], old[~flat])
    assert result[0] > old[0]


def test_real_reward_adapter_keeps_nonflat_v8_and_has_no_command_gate(monkeypatch):
    import flat_posture
    import phase_transition
    monkeypatch.setattr(flat_posture, 'flat_mask', lambda env: torch.tensor([True, False]))
    monkeypatch.setattr(phase_transition, 'joint_reward', lambda env, kind: torch.tensor([0., .7]))
    q = torch.full((2, 12), .5)
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(
        joint_pos=q, default_joint_pos=torch.zeros_like(q)))},
        _surface_posture_ids=list(range(12)), _surface_posture_tolerance=torch.ones(12))
    torch.testing.assert_close(reward(env, 'surface_joint'), torch.tensor([.625, .7]))


def test_v10_inherits_phase_and_both_learners_startup_sampling():
    from startup_repair import PHASE_PROFILES, FOCUSED_PROFILES, REQUEST_PROFILES, REPEAT_PROFILES
    for profiles in (PHASE_PROFILES, FOCUSED_PROFILES, REQUEST_PROFILES, REPEAT_PROFILES):
        assert 'omni_start_v10' in profiles
