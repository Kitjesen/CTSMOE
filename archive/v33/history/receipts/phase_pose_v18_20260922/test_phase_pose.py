import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import phase_pose as module


def test_approach_is_bounded_continuous_and_monotone():
    gap = torch.linspace(-1., 2., 501)
    weight = module.approach_weight(gap)
    assert bool(((weight >= 0) & (weight <= 1)).all())
    assert bool((weight[1:] >= weight[:-1]).all())
    assert module.approach_weight(torch.tensor(.85)) == 1
    assert module.approach_weight(torch.tensor(.35)) == 0
    assert module.approach_weight(torch.tensor(.60)) == pytest.approx(.5)


def test_gap_has_no_current_joint_or_pitch_input(monkeypatch):
    geometry = ModuleType('pit_geometry')
    geometry.EDGE_OFFSET = 1.5
    monkeypatch.setitem(sys.modules, 'pit_geometry', geometry)
    torch.testing.assert_close(module.reference_gap(torch.tensor([0., .4]),
        torch.zeros(2), .43), torch.tensor([1.07, .67]))


def test_recovery_waits_support_and_cannot_jump_on_landing():
    recovery = module.Recovery(1, 'cpu')
    recovery.update(torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), .02)
    for _ in range(20):
        assert recovery.update(torch.ones(1), torch.zeros(1, dtype=torch.bool),
                               torch.zeros(1, dtype=torch.bool), .02) == 0
    last = 0.
    for _ in range(90):
        value = float(recovery.update(torch.ones(1), torch.ones(1, dtype=torch.bool),
                                     torch.zeros(1, dtype=torch.bool), .02))
        assert 0 <= value - last <= .02 / module.RECOVERY_S + 1e-6
        last = value
    assert last == 1.
    value = recovery.update(torch.ones(1), torch.zeros(1, dtype=torch.bool),
                            torch.zeros(1, dtype=torch.bool), .02)
    assert value == 1.  # transient support loss holds an already recovered value
    assert recovery.update(torch.zeros(1), torch.zeros(1, dtype=torch.bool),
                           torch.ones(1, dtype=torch.bool), .02) == 0


def test_actual_l1_has_no_deadzone_and_does_not_penalize_requested_angle(monkeypatch):
    names = [f'{side}_{joint}_joint' for side in ('fl','fr','rl','rr')
             for joint in ('hip','thigh','calf')]
    q = torch.full((3, 12), .01)
    env = SimpleNamespace(scene={'robot': SimpleNamespace(joint_names=names,
        data=SimpleNamespace(joint_pos=q, default_joint_pos=torch.zeros_like(q)))})
    monkeypatch.setattr(module, 'measure', lambda env: {'weight': torch.tensor([1., .5, 0.])})
    torch.testing.assert_close(module.reward(env,'hip'), torch.tensor([.04,.02,0.]))
    torch.testing.assert_close(module.reward(env,'thigh_calf'), torch.tensor([.08,.04,0.]))


def test_config_reuses_disabled_terms_and_rejects_stacking():
    cfg = SimpleNamespace(posture_stop_profile='posture_stop_v17', rewards=SimpleNamespace(
        early_actual_hip_l1=SimpleNamespace(weight=0.),
        early_actual_thigh_calf_l1=SimpleNamespace(weight=0.)))
    module.configure(cfg)
    assert cfg.rewards.early_actual_hip_l1.weight == -.2
    assert cfg.rewards.early_actual_thigh_calf_l1.func is module.reward
    with pytest.raises(ValueError, match='stack'):
        module.configure(cfg)
