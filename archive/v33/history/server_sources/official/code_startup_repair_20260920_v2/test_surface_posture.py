import torch

from pit_posture import stance_values
from surface_posture import surface_weight


def test_level_ground_only_no_pit_double_penalty_and_no_invalid_plane():
    rays = torch.tensor([[0., 0.], [2., 2.04], [0., .07], [0., .10],
                         [0., .25], [0., 0.], [0., float('inf')]])
    pit = torch.tensor([False]*5+[True, False])
    weight, ground, _ = surface_weight(rays, torch.zeros(7, 3), pit)
    torch.testing.assert_close(weight, torch.tensor([1., 1., .5, 0., 0., 0., 0.]))
    assert torch.isfinite(ground).all()


def test_new_objective_does_not_constrain_high_speed_or_turning_gaits():
    commands = torch.tensor([[.6, 0., 0.], [1.15, 0., 0.], [1.5, 0., 0.],
                             [0., 0., .75], [0., 0., 1.]])
    weight, _, _ = surface_weight(torch.zeros(5, 4), commands, torch.zeros(5, dtype=torch.bool))
    torch.testing.assert_close(weight, torch.tensor([1., .5, 0., .5, 0.]))


def test_startup_excursion_costs_more_than_normal_motion_and_preserves_inputs():
    # A 1.8 rad single-leg excursion was measured in the direct-start audit.
    q = torch.zeros(2, 12)
    q[0, 0] = 1.8
    q[1] = .2
    original = q.clone()
    tolerance = torch.tensor([.25, .45, .55]*4)
    joint, height, valid, _ = stance_values(q, torch.zeros_like(q), tolerance, torch.tensor([.28, .46]))
    assert joint[0] > 0 and height[0] > 0 and not valid[0]
    assert joint[1] == 0 and height[1] == 0 and valid[1]
    torch.testing.assert_close(q, original)
