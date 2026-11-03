"""Check that early expansion reuses the official stage and keeps later limits."""
from types import SimpleNamespace

import pytest

from train import apply_command_curriculum_start


def command_cfg():
    return SimpleNamespace(
        ranges=SimpleNamespace(lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5), ang_vel_yaw=(-1., 1.)),
        command_range_curriculum=[
            dict(iter=20000, lin_vel_x=[-1., 1.], lin_vel_y=[-1., 1.], ang_vel_yaw=[-1.5, 1.5]),
            dict(iter=50000, lin_vel_x=[-2., 2.], lin_vel_y=[-1., 1.], ang_vel_yaw=[-2., 2.]),
        ],
        terrain_max_command_ranges={"stairs_up": {"lin_vel_x": [-1., 1.]}},
    )


def test_original_keeps_existing_course():
    cfg = command_cfg()
    apply_command_curriculum_start(cfg, "original")
    assert cfg.ranges.lin_vel_x == (-0.5, 0.5)
    assert [item['iter'] for item in cfg.command_range_curriculum] == [20000, 50000]


def test_intermediate_keeps_future_course_and_terrain_caps():
    cfg = command_cfg()
    apply_command_curriculum_start(cfg, "intermediate")
    assert vars(cfg.ranges) == dict(lin_vel_x=(-1., 1.), lin_vel_y=(-1., 1.), ang_vel_yaw=(-1.5, 1.5))
    assert [item['iter'] for item in cfg.command_range_curriculum] == [50000]
    assert cfg.terrain_max_command_ranges == {"stairs_up": {"lin_vel_x": [-1., 1.]}}


def test_unknown_stage_rejected():
    with pytest.raises(ValueError):
        apply_command_curriculum_start(command_cfg(), "unknown")
