import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
from continuous_runtime import next_route_levels


def test_only_top_success_randomizes_and_failure_downgrades():
    torch.manual_seed(71)
    result = next_route_levels(
        torch.tensor([9, 9, 8, 0, 3]),
        torch.tensor([True, False, True, False, False]),
        torch.tensor([False, True, False, True, True]),
    )
    assert 0 <= result[0] < 9
    assert result[1:].tolist() == [8, 9, 0, 2]


def test_top_reset_covers_all_lower_levels_and_rng_reproducible():
    old = torch.full((9000,), 9)
    success = torch.ones(9000, dtype=torch.bool)
    fail = ~success
    torch.manual_seed(71)
    state = torch.get_rng_state()
    a = next_route_levels(old, success, fail)
    torch.set_rng_state(state)
    b = next_route_levels(old, success, fail)
    assert torch.equal(a, b)
    counts = torch.bincount(a, minlength=9)
    assert len(counts) == 9 and (counts > 800).all() and (counts < 1200).all()
    assert old.eq(9).all()


def test_no_top_success_does_not_consume_rng():
    state = torch.get_rng_state()
    next_route_levels(
        torch.tensor([8, 9]), torch.tensor([True, False]), torch.tensor([False, True])
    )
    assert torch.equal(state, torch.get_rng_state())


def test_reject_contradictory_outcome():
    with pytest.raises(ValueError):
        next_route_levels(torch.tensor([9]), torch.tensor([True]), torch.tensor([True]))
