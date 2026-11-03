from types import SimpleNamespace

import pytest
import torch

from flat_posture import amplify_on_flat, surface_joint_outside_flat
from startup_repair import CONTINUOUS_PROFILE, FLAT_FOCUS_PROFILE, reward


def environment():
    # Flat, a level staircase platform, and pit: geometric flatness is not
    # enough to opt another terrain into this repair.
    command = SimpleNamespace(terrain_idxs=torch.tensor([0, 1, 2]),
        terrain_type2idx={'flat': 0, 'stairs_up': 1, 'pit': 2},
        pit=SimpleNamespace(active=torch.tensor([False, False, True])))
    robot = SimpleNamespace(joint_names=['fr_hip_joint'], data=SimpleNamespace(
        default_joint_pos=torch.zeros(3, 1), joint_pos_target=torch.ones(3, 1),
        root_lin_vel_w=torch.ones(3, 3)))
    return SimpleNamespace(cfg=SimpleNamespace(startup_repair_profile=CONTINUOUS_PROFILE),
        scene={'robot': robot}, device='cpu', _surface_posture_ids=[0],
        _surface_posture_tolerance=torch.tensor([.25]),
        action_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(
            requested_actions=torch.ones(3, 1), joint_indices=[0])),
        command_manager=SimpleNamespace(get_term=lambda name: command))


def test_continuous_cost_penalizes_small_errors_and_preserves_other_terrain():
    env = environment()
    for error in (0., .0001, .1, .5):
        cost = torch.full((3,), error)
        torch.testing.assert_close(amplify_on_flat(env, cost),
                                   torch.tensor([2*error, error, error]))


def test_old_surface_cost_removed_only_on_flat(monkeypatch):
    import surface_posture
    monkeypatch.setattr(surface_posture, 'reward', lambda env, kind: torch.tensor([1., 2., 0.]))
    torch.testing.assert_close(surface_joint_outside_flat(environment(), 'joint'),
                               torch.tensor([0., 2., 0.]))


@pytest.mark.parametrize('kind', ['target', 'request', 'vertical'])
def test_flat_proxy_replacement_preserves_nonflat_rewards_and_height_dynamics(monkeypatch, kind):
    import stance_strength
    monkeypatch.setattr(stance_strength, 'measure', lambda env: (torch.ones(3), None, None))
    env = environment()
    current = reward(env, kind)
    env.cfg.startup_repair_profile = FLAT_FOCUS_PROFILE
    previous = reward(env, kind)
    torch.testing.assert_close(current[1:], previous[1:])
    if kind == 'vertical':
        torch.testing.assert_close(current, previous)
    else:
        assert current[0] == 0 and previous[0] > 0
    if kind == 'request':
        assert current[2] == 0
