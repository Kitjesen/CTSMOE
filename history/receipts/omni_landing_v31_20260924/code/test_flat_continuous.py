import torch

from flat_continuous import continuous_cost, replace_flat_cost
from pit_posture import stance_values


def test_small_real_deviations_cost_something_without_a_free_range():
    scales = torch.tensor([.25, .45, .55])
    q = torch.tensor([[0., 0., 0.], [.01, -.02, .03]])
    before = q.clone()
    cost = continuous_cost(q, torch.zeros_like(q), scales)
    assert cost[0] == 0 and cost[1] > 0
    torch.testing.assert_close(q, before)
    doubled = continuous_cost(q*2, torch.zeros_like(q), scales)
    torch.testing.assert_close(doubled, cost*2)


def test_v8_stairs_and_pit_costs_are_bitwise_preserved():
    previous = torch.tensor([.07, .42, 0., .9, 1.7])
    flat = torch.tensor([True, False, False, False, False])
    q = torch.full((5, 12), .35)
    actual = replace_flat_cost(previous, flat, q, torch.zeros_like(q), torch.ones(12))
    assert torch.equal(actual[~flat], previous[~flat])
    torch.testing.assert_close(actual[flat], torch.tensor([.35]))


def test_measured_rear_thigh_excursion_gets_continuous_cost():
    scales = torch.tensor([.25]*4+[.45]*4+[.55]*4)
    q = torch.tensor([[.210, -.210, .252, -.414, -.201, -.484, -.867, .365,
                       .155, .110, .404, .310]])
    original = stance_values(q, torch.zeros_like(q), scales, torch.tensor([.383]))[0]
    current = continuous_cost(q, torch.zeros_like(q), scales)
    assert current.item() > original.item()*3
    assert continuous_cost(q*.5, torch.zeros_like(q), scales) < current


def test_v9_keeps_v8_phase_family_and_repair_cohort():
    from startup_repair import PHASE_PROFILES, REPEAT_PROFILES, REQUEST_PROFILES, FOCUSED_PROFILES
    assert 'omni_start_v8' in PHASE_PROFILES and 'omni_start_v9' in PHASE_PROFILES
    for profiles in (REPEAT_PROFILES, REQUEST_PROFILES, FOCUSED_PROFILES):
        assert all(profile in profiles for profile in PHASE_PROFILES)
