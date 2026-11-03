from types import SimpleNamespace

import torch

from uphill_speed import apply_uphill_range, snapshot


def test_uphill_only_and_official_later_stage_is_retained():
    command = SimpleNamespace(
        _env=SimpleNamespace(cfg=SimpleNamespace(uphill_speed_limit=1.2)),
        cfg=SimpleNamespace(terrain_max_command_ranges={"slope_up": {"lin_vel_x": [-1.5,1.5]}}),
        terrain_idxs=torch.tensor([0,1,2]), terrain_type2idx={"slope_up":1},
        command_ranges={"lin_vel_x": [-1.,1.]},
        env_command_ranges={"lin_vel_x":torch.tensor([[-1.6,1.6],[-1.,1.],[-1.,1.]])},
        commands=torch.tensor([[1.6,0.,0.],[-1.2,.5,.5],[.8,0.,0.]]))
    apply_uphill_range(command)
    torch.testing.assert_close(command.env_command_ranges["lin_vel_x"],torch.tensor([[-1.6,1.6],[-1.2,1.2],[-1.,1.]]))
    assert snapshot(command)["sampled_min"][0] < -1.
    command.command_ranges["lin_vel_x"] = [-2.,2.]
    apply_uphill_range(command)
    torch.testing.assert_close(command.env_command_ranges["lin_vel_x"][1],torch.tensor([-1.5,1.5]))
