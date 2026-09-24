"""CPU checks for the bounded, continuously commanded pit-platform task."""

import torch

from pit_traverse_logic import PROFILE, command_speed, finish, platform_tracking_reward
from train import inherited_optional_profile


def _finish(**changes):
    values = dict(
        entered=torch.tensor([True]),
        entry_x=torch.tensor([2.1]),
        root_x=torch.tensor([2.75]),
        elapsed_s=torch.tensor([0.8]),
        tracked_s=torch.tensor([0.8]),
        error_integral=torch.tensor([0.04]),
        command_vx=torch.tensor([0.6]),
        supported=torch.tensor([[True, False, True, False]]),
        body_clear=torch.tensor([True]),
        upright=torch.tensor([True]),
        physical=torch.tensor([False]),
        outside=torch.tensor([False]),
        started_in_pit=torch.tensor([True]),
    )
    values.update(changes)
    return finish(**values).item()


def test_top_command_does_not_cut_to_zero():
    held = torch.tensor([0.2, 0.6, 1.0])
    latched = torch.tensor([True, True, False])
    torch.testing.assert_close(command_speed(held, latched, PROFILE), held)
    torch.testing.assert_close(
        command_speed(held, latched, "strict_v1"), torch.tensor([0.0, 0.0, 1.0]))


def test_older_source_has_no_trained_optional_reward_to_drop():
    assert inherited_optional_profile(None, "hip_request_profile") == "off"
    assert inherited_optional_profile({"hip_request_profile": None}, "hip_request_profile") == "off"
    assert inherited_optional_profile({"hip_request_profile": "off"}, "hip_request_profile") == "off"
    assert inherited_optional_profile({"hip_request_profile": "inward_unreachable_request_v1"},
                                      "hip_request_profile") == "inward_unreachable_request_v1"


def test_traverse_accepts_tracking_with_gait_support():
    assert _finish()
    assert not _finish(root_x=torch.tensor([2.65]))  # Too little top travel.
    assert not _finish(elapsed_s=torch.tensor([0.4]))
    assert not _finish(tracked_s=torch.tensor([0.4]))
    assert not _finish(error_integral=torch.tensor([0.16]))
    assert not _finish(root_x=torch.tensor([3.71]))  # Before the existing x=4 guard.
    assert not _finish(supported=torch.tensor([[True, True, False, False]]))
    assert not _finish(body_clear=torch.tensor([False]))
    assert not _finish(upright=torch.tensor([False]))
    assert not _finish(physical=torch.tensor([True]))
    assert not _finish(outside=torch.tensor([True]))
    assert not _finish(started_in_pit=torch.tensor([False]))


def test_platform_reward_requires_real_top_support_and_tracking():
    entered = torch.tensor([True, True, True, False])
    clear = torch.tensor([True, True, False, True])
    command = torch.full((4,), 0.6)
    actual = torch.tensor([0.6, 0.2, 0.6, 0.6])
    support = torch.tensor([[True, False, True, False]] * 4)
    result = platform_tracking_reward(entered, clear, command, actual, support)
    assert result[0].item() == 1.0
    assert 0.0 < result[1].item() < 0.1
    assert result[2:].tolist() == [0.0, 0.0]
    support[0, 2:] = False
    assert platform_tracking_reward(entered, clear, command, actual, support)[0].item() == 0.0
