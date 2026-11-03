from types import SimpleNamespace

import torch

from flat_speed import apply_flat_range, snapshot, speed_limit


def test_schedule_resumes_from_absolute_steps():
    course = {"start_step": 240000}
    assert speed_limit(course, 240000, 24) == 1.5
    assert speed_limit(course, 264000, 24) == 1.75
    assert speed_limit(course, 288000, 24) == 2.
    assert speed_limit(course, 400000, 24) == 2.


def test_only_flat_x_changes_and_membership_can_move():
    ranges = {k: torch.tensor([[-1., 1.]] * 4) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")}
    command = SimpleNamespace(
        _env=SimpleNamespace(cfg=SimpleNamespace(flat_speed_course={"start_step": 0}), common_step_counter=24000),
        cfg=SimpleNamespace(num_steps_per_iter=24), terrain_idxs=torch.tensor([0, 1, 2, 1]),
        terrain_type2idx={"flat": 1}, env_command_ranges=ranges, commands=torch.zeros(4, 3))
    apply_flat_range(command)
    torch.testing.assert_close(ranges["lin_vel_x"], torch.tensor([[-1., 1.], [-1.75, 1.75], [-1., 1.], [-1.75, 1.75]]))
    for axis in ("lin_vel_y", "ang_vel_yaw"):
        torch.testing.assert_close(ranges[axis], torch.tensor([[-1., 1.]] * 4))
    # Upstream refresh restores ordinary ranges before the flat override.
    ranges["lin_vel_x"][:] = torch.tensor([-1., 1.])
    command.terrain_idxs[:] = torch.tensor([1, 0, 2, 0])
    apply_flat_range(command)
    assert snapshot(command)["count"] == 1
    torch.testing.assert_close(ranges["lin_vel_x"][1:], torch.tensor([[-1., 1.]] * 3))
