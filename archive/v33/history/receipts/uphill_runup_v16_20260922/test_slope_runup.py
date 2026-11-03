import sys
from types import SimpleNamespace as NS, ModuleType

import pytest
import torch

from runup_protocol import build_cases, summarize
from slope_runup import configure, place


def test_non_uphill_and_original_quarter_keep_parent_commands(monkeypatch):
    managers = ModuleType("isaaclab.managers")
    managers.EventTermCfg = managers.TerminationTermCfg = NS
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    class Parent:
        def __init__(self, cfg, env):
            self.device, self.num_envs, self._env, self.cfg = "cpu", 12, env, cfg
            self.terrain_idxs = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
            self.terrain_type2idx = {"slope_up": 0}
            self.commands = torch.zeros(12, 3)
            self.commands_xy_accumulation = torch.zeros_like(self.commands)
            self.time_left = torch.zeros(12)
        def _resample(self, ids):
            self.commands[ids] = torch.rand(len(ids), 3)
            self.time_left[ids] = 5.
        def _update_command(self):
            self.commands[:] = .123
    cfg = NS(commands=NS(base_velocity=NS(class_type=Parent, resampling_time=5.)), events=NS(), terminations=NS())
    configure(cfg)
    env = NS(episode_length_buf=torch.ones(12), step_dt=.02)
    baseline = Parent(cfg.commands.base_velocity, env)
    candidate = cfg.commands.base_velocity.class_type(cfg.commands.base_velocity, env)
    ids = torch.arange(12)
    torch.manual_seed(123)
    baseline._resample(ids)
    baseline_rng = torch.get_rng_state()
    torch.manual_seed(123)
    candidate._resample(ids)
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    unchanged = torch.tensor([0, 4, 5, 6, 7, 8, 9, 10, 11])
    torch.testing.assert_close(candidate.commands[unchanged], baseline.commands[unchanged])
    torch.testing.assert_close(candidate.time_left[unchanged], baseline.time_left[unchanged])
    torch.testing.assert_close(candidate.commands[1:4], torch.tensor([[.6, 0., 0.]]).expand(3, -1))
    baseline._update_command()
    candidate._update_command()
    torch.testing.assert_close(candidate.commands[unchanged], baseline.commands[unchanged])


def placement_env(width=.9):
    class Geometry:
        def world_points(self, q, pose):
            points = torch.tensor([[x, y, -.46] for x in (-width / 2, width / 2) for y in (-.25, .25)])
            c, s = 1 - 2 * pose[:, 6].square(), 2 * pose[:, 3] * pose[:, 6]
            x = c[:, None] * points[None, :, 0] - s[:, None] * points[None, :, 1]
            y = s[:, None] * points[None, :, 0] + c[:, None] * points[None, :, 1]
            z = points[None, :, 2].expand(len(pose), -1)
            return torch.stack((x, y, z), -1) + pose[:, None, :3]
    class Robot:
        data = NS(default_root_state=torch.zeros(8, 13), joint_pos=torch.zeros(8, 16))
        def write_root_pose_to_sim(self, pose, env_ids):
            self.pose = pose
        def write_root_velocity_to_sim(self, velocity, env_ids):
            assert torch.count_nonzero(velocity) == 0
    robot = Robot()
    class Scene(dict):
        env_origins = torch.zeros(8, 3)
    env = NS(scene=Scene(robot=robot), device="cpu", command_manager=NS(get_term=lambda _: NS(pit=NS(geometry=Geometry()))))
    return env


def test_full_collision_placement_all_axes_and_both_approaches():
    env = placement_env()
    gaps = torch.tensor([.1, 1.5] * 4)
    receipt = place(env, torch.arange(8), gaps, torch.arange(4).repeat_interleave(2))
    assert receipt["collision_platform_margin_min_m"] >= .099
    assert receipt["actual_collision_clearance_min_m"] >= .0299
    assert receipt["front_gap_max_m"] == pytest.approx(1.5)


def test_oversized_collision_is_rejected_before_writing_pose():
    env = placement_env(width=4.)
    with pytest.raises(RuntimeError, match="does not fit"):
        place(env, torch.arange(8), torch.full((8,), 1.5), torch.zeros(8, dtype=torch.long))
    assert not hasattr(env.scene["robot"], "pose")


def test_paired_protocol_counts_and_missing_evidence_rejected():
    rows = build_cases([dict(teacher=False), dict(teacher=True)], "terrain83")
    assert len(rows) == 180
    for row in rows:
        row.update(passed_4m=False, physical=False, dx_m=0., entry_speed_mps=None)
    summary = summarize(dict(cases=rows))
    assert len(summary["groups"]) == 60
    with pytest.raises(ValueError, match="Missing matched"):
        summarize(dict(cases=rows[:-1]))
