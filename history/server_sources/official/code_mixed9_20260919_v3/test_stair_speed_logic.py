import torch

from stair_speed_logic import SPEED_EDGES, sample_speeds, terrain_decisions


def test_all_speed_bands_are_sampled_and_respect_boundaries():
    torch.manual_seed(42)
    bins, speeds = sample_speeds(4096, "cpu")
    edges = torch.tensor(SPEED_EDGES)
    assert torch.all(speeds >= edges[bins])
    assert torch.all(speeds < edges[bins + 1])
    assert torch.all(torch.bincount(bins, minlength=4) > 800)


def test_slow_budget_does_not_inflate_or_reduce_difficulty_but_falls_still_count():
    speed = torch.tensor([0.2, 0.2, 0.2, 0.6, 0.6, 0.6])
    distance = torch.tensor([1.0, 4.5, 1.0, 4.5, 1.0, 4.5])
    failed = torch.tensor([False, False, True, False, False, True])
    up, down, eligible = terrain_decisions(speed, torch.full_like(speed, 25), distance, failed, 25, 4)
    assert eligible.tolist() == [False, False, False, True, True, True]
    assert up.tolist() == [False, False, False, True, False, False]
    assert down.tolist() == [False, False, True, False, True, True]


def test_downgrade_uses_elapsed_command_not_full_future_episode():
    speed = torch.tensor([0.8, 0.8])
    up, down, _ = terrain_decisions(speed, torch.tensor([2., 25.]), torch.tensor([1., 1.]),
                                    torch.tensor([False, False]), 25, 4)
    assert not up.any()
    assert down.tolist() == [False, True]
