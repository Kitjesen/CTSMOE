import torch
from types import SimpleNamespace

from pit_posture import phase_weight, stance_values
from stance_strength import stationary_cost, zero_command
from surface_posture import surface_weight


def test_zero_translation_with_yaw_is_not_standing():
    cmd = torch.tensor([[0., 0., 0.], [.03, .03, .04], [0., 0., .2], [.06, 0., 0.]])
    assert zero_command(cmd).tolist() == [True, True, False, False]


def test_stop_penalizes_deviations_inside_old_locomotion_deadband():
    q = torch.tensor([[.1, .2, .3], [.1, .2, .3], [0., 0., 0.]])
    nominal = torch.zeros_like(q)
    tol = torch.tensor([.25, .45, .55])
    cmd = torch.tensor([[0., 0., 0.], [.6, 0., 0.], [0., 0., 0.]])
    old, _, _, _ = stance_values(q, nominal, tol, torch.full((3,), .46))
    new = stationary_cost(q, nominal, tol, cmd)
    assert old.sum() == 0 and new[0] > 0 and new[1] == 0 and new[2] == 0


def test_height_remains_active_at_fast_translation_and_turning_only_on_level_ground():
    rays = torch.tensor([[0., 0.], [0., .25], [0., 0.], [0., float('nan')]])
    cmd = torch.tensor([[2., 0., 1.5]]).expand(4, -1)
    pit = torch.tensor([False, False, True, False])
    old, _, _ = surface_weight(rays, cmd, pit)
    new, _, _ = surface_weight(rays, cmd, pit, command_gating=False)
    torch.testing.assert_close(old, torch.zeros(4))
    torch.testing.assert_close(new, torch.tensor([1., 0., 0., 0.]))


def test_pit_climb_excluded_even_if_command_zero_and_pose_abnormal():
    weight = phase_weight(torch.tensor([1., .1, -1.]),
                          torch.tensor([False, False, True]), torch.ones(3, dtype=torch.bool))
    cost = stationary_cost(torch.ones(3, 12), torch.zeros(3, 12),
                           torch.tensor([.25, .45, .55]*4), torch.zeros(3, 3))
    assert (weight*cost > 0).tolist() == [True, False, True]


def test_height_penalty_quadruples_for_same_measured_error():
    _, cost, _, _ = stance_values(torch.zeros(1, 12), torch.zeros(1, 12),
                                  torch.ones(12), torch.tensor([.36]))
    torch.testing.assert_close(-2.*cost, 4*(-.5*cost))


def test_snapshot_reports_live_reward_weights(monkeypatch):
    import stance_strength
    command = SimpleNamespace(pit=SimpleNamespace(active=torch.tensor([False, True])))
    weights = {'surface_stance_height': -3., 'zero_command_nominal_stance': -4.}
    env = SimpleNamespace(cfg=SimpleNamespace(stance_strength_profile='strong_stand_v1'),
                          command_manager=SimpleNamespace(get_term=lambda _: command),
                          reward_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(weight=weights[name])))
    monkeypatch.setattr(stance_strength, 'measure', lambda _: (torch.ones(2), torch.ones(2), torch.ones(2, dtype=torch.bool)))
    result = stance_strength.snapshot(env)
    assert result['height_weight'] == -3. and result['zero_command_weight'] == -4.
