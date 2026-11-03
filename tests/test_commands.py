"""Run the real pinned Go2 sampler against a CPU scene and current-tile lookup.

These tests exercise manager reset/compute and command budgets, not simulation
stability or learned performance; those require the native Newton probe.
"""

from copy import deepcopy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from go2_rl_lab.cmd_utils import rl_gym_cmd as upstream
from thunder_lab3.tasks.thunder.commands import (
    TERRAIN_COMMAND_SOURCE, ThunderRLGymVelocityCommand, ThunderRLGymVelocityCommandCfg,
)


class Scene(dict):
    pass


def fixture(terrain_names=("flat",), *, counter=0, cfg=None):
    names = tuple(TERRAIN_COMMAND_SOURCE)
    n = len(terrain_names)
    positions = torch.zeros(n, 3)
    positions[:, 1] = torch.tensor([-28. + 8. * names.index(name) for name in terrain_names])
    robot = NS(data=NS(root_pos_w=NS(torch=positions),
        root_quat_w=NS(torch=torch.tensor([[0., 0., 0., 1.]]).repeat(n, 1)),
        projected_gravity_b=NS(torch=torch.tensor([[0., 0., -1.]]).repeat(n, 1))))
    scene = Scene(robot=robot)
    scene.env_origins = positions.clone()
    scene.terrain = NS(cfg=NS(terrain_type="generator", terrain_generator=NS(
        curriculum=True, num_rows=1, num_cols=8, size=(8., 8.),
        sub_terrains={name: NS(proportion=1. / 8.) for name in names})))
    env = NS(num_envs=n, device="cpu", scene=scene, common_step_counter=counter,
        step_dt=.02, max_episode_length=1000, max_episode_length_s=20.,
        episode_length_buf=torch.zeros(n, dtype=torch.long),
        sim=NS(vis_marker_registry=NS(clear_debug_vis_callback=lambda _: None)))
    cfg = cfg if cfg is not None else ThunderRLGymVelocityCommandCfg(debug_vis=False)
    term = ThunderRLGymVelocityCommand(cfg, env)
    return term, env, cfg


class CommandTests(unittest.TestCase):
    def test_official_mechanism_and_explicit_user_caps(self):
        cfg = ThunderRLGymVelocityCommandCfg()
        official = upstream.RLGymVelCommandCfg()
        self.assertTrue(issubclass(cfg.class_type, upstream.RLGymVelCommand))
        for name in ("zero_command_curriculum", "limit_vel_curriculum", "limit_vel_prob",
                     "limit_ang_vel_at_zero_command_prob", "resampling_time", "num_steps_per_iter",
                     "dynamic_resample_commands", "limit_vel_invert_when_continuous"):
            self.assertEqual(getattr(cfg, name), getattr(official, name))
        self.assertEqual((cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_yaw),
                         ((-.5, .5), (-.5, .5), (-1., 1.)))
        self.assertEqual(cfg.command_range_curriculum, [{"iter": 5000,
            "lin_vel_x": (-1., 1.), "lin_vel_y": (-1., 1.), "ang_vel_yaw": (-1.5, 1.5)}])
        self.assertEqual(set(cfg.terrain_max_command_ranges), set(TERRAIN_COMMAND_SOURCE))
        self.assertIsNone(TERRAIN_COMMAND_SOURCE["random_rough"])
        self.assertEqual(cfg.terrain_max_command_ranges["random_rough"], {
            "lin_vel_x": (-1., 1.), "lin_vel_y": (-1., 1.), "ang_vel_yaw": (-1.5, 1.5)})
        self.assertFalse(hasattr(cfg, "rel_pure_lateral_envs"))
        self.assertFalse(hasattr(cfg, "rel_pure_yaw_envs"))

    def test_exact_iteration_boundary_and_resume_do_not_mutate_declarative_cfg(self):
        cfg = ThunderRLGymVelocityCommandCfg(debug_vis=False)
        original = deepcopy(cfg.to_dict())
        term, env, _ = fixture(cfg=cfg)
        for iteration, remainder, expected in ((0, 0, .5), (4999, 23, .5), (5000, 0, 1.), (20000, 0, 1.)):
            env.common_step_counter = iteration * 24 + remainder
            term.reset()
            self.assertEqual(term.command_ranges["lin_vel_x"], (-expected, expected))
        self.assertEqual(cfg.to_dict(), original)
        resumed, resume_env, _ = fixture(cfg=cfg)
        # Mirrors restore_course: construct first, restore counter, then reset.
        resume_env.common_step_counter = 12345 * 24
        resumed.reset()
        self.assertEqual(resumed.command_ranges, term.command_ranges)
        self.assertEqual(resumed.zero_command_prob, .1)
        self.assertEqual(cfg.to_dict(), original)

    def test_current_tile_clips_stairs_then_restores_saved_flat_request(self):
        term, env, _ = fixture(counter=5000 * 24)
        term.reset()
        request = torch.tensor([[.9, -.8, 1.4]])
        term._requested_velocity_command_b[:] = request
        term._update_command()
        torch.testing.assert_close(term.command, request)
        env.scene["robot"].data.root_pos_w.torch[:, 1] = -12.  # inverted pyramid: upstairs
        term._update_command()
        term.stair_heading[:] = 1.
        term._update_command()
        torch.testing.assert_close(term.command, torch.tensor([[.5, 0., .2]]))
        torch.testing.assert_close(term._requested_velocity_command_b, request)
        env.scene["robot"].data.root_pos_w.torch[:, 1] = -28.
        term._update_command()
        torch.testing.assert_close(term.command, request)

    def test_stair_zero_survives_positive_moving_interval(self):
        term, _, _ = fixture(("pyramid_stairs", "pyramid_stairs_inv"))
        term.reset()
        term._requested_velocity_command_b[:] = 0.
        term._update_command()
        self.assertTrue(bool(term.is_standing_env.all()))
        torch.testing.assert_close(term.command, torch.zeros(2, 3))

    def test_all_eight_terrains_stay_within_cap_through_training(self):
        torch.manual_seed(42)
        term, env, _ = fixture(tuple(TERRAIN_COMMAND_SOURCE) * 16)
        stair = torch.tensor([name.startswith("pyramid_stairs") for name in tuple(TERRAIN_COMMAND_SOURCE) * 16])
        for iteration in (0, 750, 1500, 4999, 5000, 19999):
            env.common_step_counter = iteration * 24
            term.reset()
            env.episode_length_buf[:] = 1
            env.common_step_counter += 1
            term.compute(env.step_dt)
            self.assertTrue(bool(torch.isfinite(term.command).all()))
            self.assertTrue(bool((term.command[:, :2].abs() <= 1.).all()))
            self.assertTrue(bool((term.command[:, 2].abs() <= 1.5).all()))
            moving_stair = stair & ~term.is_standing_env
            self.assertTrue(bool((term.command[moving_stair, 0] >= .25).all()))
            self.assertTrue(bool((term.command[moving_stair, 0] <= .5).all()))
            self.assertTrue(bool((term.command[stair, 1] == 0.).all()))
            self.assertTrue(bool((term.command[stair, 2].abs() <= .2).all()))

    def test_applied_budget_counts_once_and_partial_reset_clears_only_selected(self):
        term, env, _ = fixture(("flat", "pyramid_stairs_inv"), counter=5000 * 24)
        term.reset()
        torch.testing.assert_close(term.commands_xy_accumulation, torch.zeros(2, 2))
        term._requested_velocity_command_b[:] = torch.tensor([1., 1., 1.5])
        term._update_command()
        env.episode_length_buf[:] = 1
        env.common_step_counter += 1
        term.compute(env.step_dt)
        term.accumulate(None)
        # The stair budget is .5 m/s forward, not the requested sqrt(2) m/s.
        torch.testing.assert_close(term.commands_xy_accumulation * 5., torch.tensor([[.02, .02], [.01, 0.]]))
        torch.testing.assert_close(term.travel_expected, torch.tensor([2.**.5 * .02, .01]))
        second_budget = term.commands_xy_accumulation[1].clone()
        term.reset([0])
        torch.testing.assert_close(term.commands_xy_accumulation[0], torch.zeros(2))
        torch.testing.assert_close(term.commands_xy_accumulation[1], second_budget)
        self.assertEqual(term.travel_expected[0].item(), 0.)
        self.assertAlmostEqual(term.travel_expected[1].item(), .01)

    def test_dynamic_sampling_uses_elapsed_applied_displacement(self):
        term, env, _ = fixture(("pyramid_stairs_inv",), counter=5000 * 24)
        term.reset()
        env.episode_length_buf[:] = 500
        term.commands_xy_accumulation[:] = torch.tensor([.4, 0.])  # 2 commanded metres already elapsed
        with patch.object(upstream, "_sample_disjoint_intervals", wraps=upstream._sample_disjoint_intervals) as sample:
            term._resample([0])
        lower_bound = sample.call_args_list[0].args[1]
        torch.testing.assert_close(lower_bound, torch.tensor([3. / (501 * .02)]))
        torch.testing.assert_close(term.commands_xy_accumulation, torch.tensor([[.4, 0.]]))

    def test_episode_reset_samples_against_full_horizon_before_lab_clears_buffer(self):
        term, env, _ = fixture(("flat", "pyramid_stairs_inv"), counter=5000 * 24)
        env.episode_length_buf[:] = torch.tensor([1000, 400])
        with patch.object(upstream, "_sample_disjoint_intervals", wraps=upstream._sample_disjoint_intervals) as sample:
            term.reset([0])
        torch.testing.assert_close(sample.call_args_list[0].args[1], torch.tensor([5. / (1001 * .02)]))
        # Preserve the framework's ordering and do not touch unselected episodes.
        self.assertEqual(env.episode_length_buf.tolist(), [1000, 400])

    def test_standing_time_reserves_motion_at_the_effective_stair_cap(self):
        term, env, _ = fixture(("flat", "pyramid_stairs_inv"), counter=5000 * 24)
        term.reset()
        env.episode_length_buf[:] = 500
        term.commands_xy_accumulation[:] = torch.tensor([.4, 0.])
        # .25 selects zero mode after the .20 boundary region, but not its .20 yaw addition.
        with patch.object(upstream.torch, "rand", side_effect=lambda n, device=None: torch.full((n,), .25, device=device)):
            term._resample([0, 1])
        torch.testing.assert_close(term.command, torch.zeros(2, 3))
        # 10 s left, 3 m command budget left: flat reserves 3/.8, stairs 3/.4.
        torch.testing.assert_close(term.time_left, torch.tensor([5., 2.5]))
        self.assertEqual(term.max_lin_vel, 1.)


if __name__ == "__main__":
    unittest.main()
