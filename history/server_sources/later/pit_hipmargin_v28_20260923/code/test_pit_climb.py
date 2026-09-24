import torch

from pit_climb import confirm_climb, stability_reward
from pit_logic import PitCourse


def test_climb_needs_full_support_and_body_but_not_quiet_angular_speed():
    support = torch.ones((4, 4), dtype=torch.bool)
    support[1, 3] = False
    clear = torch.tensor([True, True, False, True])
    invalid = torch.tensor([False, False, False, True])
    good = torch.ones(4, dtype=torch.bool)
    seen = torch.zeros(4, dtype=torch.bool)
    hold = torch.zeros(4)
    for _ in range(9):
        hold, seen, event = confirm_climb(hold, seen, support, clear, good, invalid, good, .02)
        assert not event.any()
    hold, seen, event = confirm_climb(hold, seen, support, clear, good, invalid, good, .02)
    assert event.tolist() == [True, False, False, False]
    for _ in range(5):
        hold, seen, event = confirm_climb(hold, seen, support, clear, good, invalid, good, .02)
        assert not event.any()  # Do not repeatedly pay the climb reward.


def test_climb_interrupted_support_restarts_confirmation_and_requires_pit_start():
    hold = torch.tensor([9., 9.])
    seen = torch.zeros(2, dtype=torch.bool)
    support = torch.ones((2, 4), dtype=torch.bool)
    support[0, 0] = False
    good = torch.ones(2, dtype=torch.bool)
    hold, seen, _ = confirm_climb(hold, seen, support, good, good, ~good,
                                  torch.tensor([True, False]), .02)
    assert hold.tolist() == [0, 0] and not seen.any()


def test_post_climb_shaping_rewards_support_and_less_sway_only_when_enabled():
    supported = torch.ones((4, 4), dtype=torch.bool)
    supported[1, 3] = False
    angular = torch.zeros((4, 3))
    angular[2, 0] = .3
    reward = stability_reward(supported, -torch.ones(4), angular,
                              torch.tensor([True, True, True, False]))
    assert reward[0] == 1 and reward[1] == .75
    assert 0 < reward[2] < reward[1] and reward[3] == 0


def test_course_can_advance_on_climbs_while_parking_is_a_separate_metric():
    course = PitCourse()
    bins = list(range(4)) * 64
    assert not course.observe(bins, [True] * 256, [False] * 256)
    assert course.observe(bins, [True] * 256, [False] * 256)
    assert course.frontier == 3
