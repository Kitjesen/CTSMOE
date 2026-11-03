"""A brief necessary hip excursion remains possible; prolonged exposure costs more."""

import torch

from pit_hip_margin import margin_cost


def test_margin_cost_starts_before_soft_limit_and_grows_with_exposure():
    signs = torch.tensor([-1., 1., 1., -1.])
    inward = torch.tensor([
        [.12, .12, .12, .12],
        [.34, .12, .12, .12],
        [.39, .12, .12, .12],
        [.41, .12, .12, .12],
    ])
    cost = margin_cost(inward * signs, signs)
    torch.testing.assert_close(cost, torch.tensor([0., 0., (.05 / .07) ** 2, 1.]),
                               rtol=0, atol=1e-6)
