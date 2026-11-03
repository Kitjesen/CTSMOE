import copy

import torch

from pit_individual import IndividualPitCourse


def test_success_and_failure_streaks_are_independent_per_env_and_speed():
    c = IndividualPitCourse(torch.tensor([9, 9]))
    ids = torch.tensor([0, 1])
    c.bins[:] = 0
    c.record(ids, torch.tensor([9, 9]), torch.tensor([True, False]))
    assert c.levels[:, 0].tolist() == [9, 9]
    c.record(ids, torch.tensor([9, 9]), torch.tensor([True, False]))
    assert c.levels[:, 0].tolist() == [10, 8]
    assert c.levels[:, 1:].tolist() == [[9,9,9], [9,9,9]]
    assert c.promotions.sum() == 1 and c.demotions.sum() == 1


def test_review_never_promotes_target_bank():
    c = IndividualPitCourse(torch.tensor([9]))
    c.bins[:] = 0
    c.review[:] = True
    for _ in range(5):
        c.record(torch.tensor([0]), torch.tensor([5]), torch.tensor([True]))
    assert c.levels[0,0] == 9 and c.wins.sum() == 0


def test_can_reach_120cm_without_global_gate_and_never_exceeds_it():
    c = IndividualPitCourse(torch.tensor([9, 0]))
    c.bins[:] = 0
    ids = torch.tensor([0,1])
    for _ in range(40):
        c.record(ids, c.levels[ids, c.bins].clone(), torch.tensor([True, False]))
    assert c.levels[:,0].tolist() == [21,0]
    assert .15+.05*c.levels[0,0].item() == 1.2


def test_switching_speed_does_not_transfer_success():
    c = IndividualPitCourse(torch.tensor([9]))
    ids = torch.tensor([0])
    c.bins[:] = 0
    c.record(ids, torch.tensor([9]), torch.tensor([True]))
    c.bins[:] = 3
    c.record(ids, torch.tensor([9]), torch.tensor([True]))
    assert c.levels.tolist() == [[9,9,9,9]]
    assert c.wins.tolist() == [[1,0,0,1]]


def test_sampling_matches_bank_and_restore_is_exact():
    torch.manual_seed(7)
    c = IndividualPitCourse(torch.full((4096,), 9))
    c.levels[:, 3] = 12
    ids = torch.arange(4096)
    actual = c.sample(ids)
    bank = c.levels[ids, c.bins]
    assert torch.equal(actual[~c.review], bank[~c.review])
    assert bool((actual[c.review] < bank[c.review]).all())
    assert .17 < float(c.review.float().mean()) < .23
    restored = IndividualPitCourse(torch.zeros(4096, dtype=torch.long))
    restored.load(copy.deepcopy(c.state_dict()))
    for name, value in c.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, restored.state_dict()[name])
    restored.initialize_ids(torch.tensor([3]), torch.tensor([10]))
    assert restored.levels[3].tolist() == [10]*4
    assert not restored.review[3]
