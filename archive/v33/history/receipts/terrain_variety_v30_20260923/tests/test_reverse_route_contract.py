"""CPU isolation of the actual reset and command methods; no simulator substitution in production."""
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


def test_reverse_body_heading_preserves_world_exit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'code'))

    class Base:
        def _resample(self, ids):
            pass

    n = 140
    columns = torch.tensor([6, 7, 8, 9, 10, 11, 12] * 20)
    headings = torch.zeros(n)

    def reset(env, ids, pose_range, velocity_range, asset_cfg):
        if len(ids):
            headings[ids] = pose_range['yaw'][0]

    monkeypatch.setitem(sys.modules, 'stop_commands', SimpleNamespace(Go2RLGymCommand=Base))
    monkeypatch.setitem(sys.modules, 'robot_lab.tasks.go2.mdp.curriculums', SimpleNamespace(terrain_levels_vel_gym=lambda *a: None))
    monkeypatch.setitem(sys.modules, 'robot_lab.tasks.go2.mdp.events', SimpleNamespace(reset_root_state_uniform=reset))
    monkeypatch.setitem(sys.modules, 'robot_lab.tasks.go2.mdp.utils', SimpleNamespace(is_env_assigned_to_terrain=lambda env, name: torch.ones(n, dtype=torch.bool)))
    sys.modules.pop('stair_speed', None)
    module = importlib.import_module('stair_speed')
    env = SimpleNamespace(num_envs=n, device='cpu', cfg=SimpleNamespace(terrain_variety_profile='radial_routes_v30'),
                          scene=SimpleNamespace(terrain=SimpleNamespace(terrain_types=columns)))
    torch.manual_seed(230930)
    module.reset_stair_root(env, torch.arange(n), {'yaw': (0., 0.)}, {}, None)
    assert (env.stair_motion_sign < 0).any()
    assert (env.stair_motion_sign[(columns == 8) | (columns == 11)] == 1).all()
    command = module.StraightStairCommand.__new__(module.StraightStairCommand)
    command._env, command.device = env, 'cpu'
    command.stair_mask = lambda: torch.ones(n, dtype=torch.bool)
    command.needs_speed = torch.ones(n, dtype=torch.bool)
    command.speed_bins = torch.zeros(n, dtype=torch.long)
    command.held_speed = torch.zeros(n)
    command.commands = torch.zeros(n, 3)
    command.last_is_limit_vel = torch.zeros(n, dtype=torch.bool)
    command.time_left = torch.zeros(n)
    command._resample(torch.arange(n))
    direction = torch.stack((headings.cos(), headings.sin()), -1) * command.commands[:, :1]
    exit_heading = env.stair_axis.float() * math.pi / 2
    expected = torch.stack((exit_heading.cos(), exit_heading.sin()), -1) * command.commands[:, :1].abs()
    torch.testing.assert_close(direction, expected, atol=1e-6, rtol=1e-5)
    assert (command.commands[:, 1:] == 0).all()
