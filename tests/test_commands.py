"""Run the real pinned Go2 sampler against a CPU scene and current-tile lookup.

These tests exercise manager reset/compute and command budgets, not simulation
stability or learned performance; those require the native Newton probe.
"""

from copy import deepcopy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from isaaclab.managers import CurriculumTermCfg
from go2_rl_lab.cmd_utils import rl_gym_cmd as upstream
from go2_rl_lab.tasks.mixed.env_cfg import CommandsCfg as Go2CommandsCfg
from thunder_lab3.tasks.thunder.commands import (
    TERRAIN_COMMAND_SOURCE, ThunderRLGymVelocityCommand, ThunderRLGymVelocityCommandCfg, ThunderTerrainLevels,
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


def differing_terrain_caps_cfg():
    """Exercise budget correctness when flat-ground commands exceed stair limits."""
    cfg = ThunderRLGymVelocityCommandCfg(debug_vis=False)
    cfg.ranges = upstream.RLGymVelCommandCfg.Ranges(
        lin_vel_x=(-2., 2.), lin_vel_y=(-1., 1.), ang_vel_yaw=(-3., 3.))
    cfg.command_range_curriculum = []
    source = upstream.RLGymVelCommandCfg().terrain_max_command_ranges
    cfg.terrain_max_command_ranges = {
        terrain: deepcopy(source[original]) for terrain, original in TERRAIN_COMMAND_SOURCE.items()}
    return cfg


class CommandTests(unittest.TestCase):
    def test_official_mechanism_full_curriculum_and_terrain_caps(self):
        cfg = ThunderRLGymVelocityCommandCfg()
        official = upstream.RLGymVelCommandCfg()
        self.assertTrue(issubclass(cfg.class_type, upstream.RLGymVelCommand))
        for name in ("zero_command_curriculum", "limit_vel_prob", "command_range_curriculum", "command_clip_rules",
                     "limit_ang_vel_at_zero_command_prob", "resampling_time", "num_steps_per_iter",
                     "dynamic_resample_commands", "limit_vel_invert_when_continuous"):
            self.assertEqual(getattr(cfg, name), getattr(official, name))
        self.assertEqual((cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_yaw),
                         ((-.5, .5), (-.5, .5), (-1., 1.)))
        self.assertEqual(cfg.limit_vel_curriculum, Go2CommandsCfg().base_velocity.limit_vel_curriculum)
        self.assertEqual(set(cfg.terrain_max_command_ranges), set(TERRAIN_COMMAND_SOURCE))
        self.assertEqual(TERRAIN_COMMAND_SOURCE["random_rough"], "rough_slope")
        self.assertEqual(cfg.terrain_max_command_ranges["random_rough"], {
            "lin_vel_x": (-1.5, 1.5), "lin_vel_y": (-1., 1.), "ang_vel_yaw": (-1.5, 1.5)})
        self.assertFalse(hasattr(cfg, "rel_pure_lateral_envs"))
        self.assertFalse(hasattr(cfg, "rel_pure_yaw_envs"))
        self.assertIs(ThunderRLGymVelocityCommand._apply_terrain_command_limits,
                      upstream.RLGymVelCommand._apply_terrain_command_limits)
        for terrain, original in TERRAIN_COMMAND_SOURCE.items():
            self.assertEqual(cfg.terrain_max_command_ranges[terrain], official.terrain_max_command_ranges[original])

    def test_exact_iteration_boundary_and_resume_do_not_mutate_declarative_cfg(self):
        cfg = ThunderRLGymVelocityCommandCfg(debug_vis=False)
        original = deepcopy(cfg.to_dict())
        term, env, _ = fixture(cfg=cfg)
        stages = [(0, (-.5, .5), (-.5, .5), (-1., 1.)),
                  (5000, (-1., 1.), (-1., 1.), (-2., 2.)),
                  (15000, (-2., 2.), (-1., 1.), (-3., 3.)),
                  (25000, (-3., 3.), (-1., 1.), (-3., 3.)),
                  (35000, (-4., 4.), (-1., 1.), (-3., 3.)),
                  (45000, (-4.5, 4.2), (-1., 1.), (-3., 3.))]
        checkpoints = [(0, 0, stages[0][1:])]
        for prior, current in zip(stages, stages[1:]):
            checkpoints.extend(((current[0] - 1, 23, prior[1:]), (current[0], 0, current[1:])))
        for iteration, remainder, expected in checkpoints:
            env.common_step_counter = iteration * 24 + remainder
            term.reset()
            self.assertEqual(tuple(tuple(term.command_ranges[axis]) for axis in
                ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")), expected)
            self.assertEqual(term.limit_vel["lin_vel_y"], [0] if iteration >= 25000 else [-1, 1])
        self.assertEqual(cfg.to_dict(), original)
        for iteration, *expected in stages[1:]:
            resumed, resume_env, _ = fixture(cfg=cfg)
            # Mirrors restore_course: construct first, restore counter, then reset.
            resume_env.common_step_counter = iteration * 24
            resumed.reset()
            self.assertEqual(tuple(tuple(resumed.command_ranges[axis]) for axis in
                ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")), tuple(expected))
            self.assertEqual(resumed.limit_vel["lin_vel_y"], [0] if iteration >= 25000 else [-1, 1])
            self.assertEqual(resumed.zero_command_prob, .1)
        self.assertEqual(cfg.to_dict(), original)

    def test_current_tile_clips_stairs_then_restores_saved_flat_request(self):
        term, env, _ = fixture(counter=5000 * 24, cfg=differing_terrain_caps_cfg())
        term.reset()
        for direction in (-1., 1.):
            request = torch.tensor([[1.8, -.8, 1.4]]) * direction
            term._requested_velocity_command_b[:] = request
            term._update_command()
            torch.testing.assert_close(term.command, request)
            env.scene["robot"].data.root_pos_w.torch[:, 1] = -12.  # inverted pyramid: upstairs
            term._update_command()
            torch.testing.assert_close(term.command, torch.tensor([[1., -.8, 1.4]]) * direction)
            torch.testing.assert_close(term._requested_velocity_command_b, request)
            env.scene["robot"].data.root_pos_w.torch[:, 1] = -28.
            term._update_command()
            torch.testing.assert_close(term.command, request)

    def test_stairs_preserve_zero_command(self):
        term, _, _ = fixture(("pyramid_stairs", "pyramid_stairs_inv"))
        term.reset()
        term._requested_velocity_command_b[:] = 0.
        term._update_command()
        torch.testing.assert_close(term.command, torch.zeros(2, 3))

    def test_all_eight_terrains_stay_within_cap_through_training(self):
        torch.manual_seed(42)
        term, env, _ = fixture(tuple(TERRAIN_COMMAND_SOURCE) * 16)
        stair = torch.tensor([name.startswith("pyramid_stairs") for name in tuple(TERRAIN_COMMAND_SOURCE) * 16])
        for iteration in (0, 750, 1500, 4999, 5000, 15000, 19999, 25000, 35000, 45000):
            env.common_step_counter = iteration * 24
            term.reset()
            env.episode_length_buf[:] = 1
            env.common_step_counter += 1
            term.compute(env.step_dt)
            self.assertTrue(bool(torch.isfinite(term.command).all()))
            self.assertTrue(bool((term.command >= term._env_command_ranges[:, :, 0]).all()))
            self.assertTrue(bool((term.command <= term._env_command_ranges[:, :, 1]).all()))
            self.assertTrue(bool((term.command[stair, 0] < 0.).any()))
            self.assertTrue(bool((term.command[stair, 1].abs() > .2).any()))
            self.assertTrue(bool((term.command[stair, 2].abs() > .2).any()))

    def test_final_caps_and_ordered_high_speed_coupling(self):
        names = tuple(TERRAIN_COMMAND_SOURCE)
        term, _, _ = fixture(names * 2, counter=45000 * 24)
        term.reset()
        term._requested_velocity_command_b[:8] = 100.
        term._requested_velocity_command_b[8:] = -100.
        term._update_command()
        upper = torch.tensor([[4.2, .5, 1.], [1., 1., 1.5], [1., 1., 1.5], [1., 1., 1.5],
                              [1.5, 1., 1.5], [1.5, 1., 1.5], [1.5, 1., 1.5], [1.5, 1., 1.5]])
        lower = -upper.clone()
        lower[0, 0] = -4.5
        torch.testing.assert_close(term.command, torch.cat((upper, lower)))
        flat, _, _ = fixture(("flat",) * 4, counter=45000 * 24)
        flat.reset()
        flat._requested_velocity_command_b[:] = torch.tensor(
            [[2., 1., 2.], [2.0001, 1., 2.5], [-2.0001, -1., -2.5], [1.8, .8, 2.5]])
        flat._update_command()
        torch.testing.assert_close(flat.command, torch.tensor(
            [[2., 1., 2.], [2.0001, .5, 1.], [-2.0001, -.5, -1.], [1.8, .8, 2.5]]))

    def test_applied_budget_counts_once_and_partial_reset_clears_only_selected(self):
        term, env, _ = fixture(("flat", "pyramid_stairs_inv"), counter=5000 * 24,
                               cfg=differing_terrain_caps_cfg())
        term.reset()
        torch.testing.assert_close(term.commands_xy_accumulation, torch.zeros(2, 2))
        term._requested_velocity_command_b[:] = torch.tensor([1.8, .8, 1.4])
        term._update_command()
        env.episode_length_buf[:] = 1
        env.common_step_counter += 1
        term.compute(env.step_dt)
        term.accumulate(None)
        # Integrate the clipped stair command, not the faster requested flat one.
        expected_xy = torch.tensor([[1.8, .8], [1., .8]]) * .02
        torch.testing.assert_close(term.commands_xy_accumulation * 5., expected_xy)
        torch.testing.assert_close(term.travel_expected, expected_xy.norm(dim=1))
        second_budget = term.commands_xy_accumulation[1].clone()
        term.reset([0])
        torch.testing.assert_close(term.commands_xy_accumulation[0], torch.zeros(2))
        torch.testing.assert_close(term.commands_xy_accumulation[1], second_budget)
        self.assertEqual(term.travel_expected[0].item(), 0.)
        self.assertAlmostEqual(term.travel_expected[1].item(), expected_xy[1].norm().item())

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
        term, env, _ = fixture(("flat", "pyramid_stairs_inv"), counter=5000 * 24,
                               cfg=differing_terrain_caps_cfg())
        term.reset()
        env.episode_length_buf[:] = 700
        term.commands_xy_accumulation[:] = torch.tensor([.4, 0.])
        # .25 selects zero mode after the .20 boundary region, but not its .20 yaw addition.
        with patch.object(upstream.torch, "rand", side_effect=lambda n, device=None: torch.full((n,), .25, device=device)):
            term._resample([0, 1])
        torch.testing.assert_close(term.command, torch.zeros(2, 3))
        # 6 s left, 3 m command budget left: flat reserves 3/1.6, stairs 3/.8.
        torch.testing.assert_close(term.time_left, torch.tensor([4.125, 2.25]))
        self.assertEqual(term.max_lin_vel, 2.)

    def test_terminal_stair_crossing_promotes_forward_reverse_and_lateral(self):
        for counter in (4999 * 24 + 23, 5000 * 24):
            term, env, _ = fixture(("pyramid_stairs_inv",) * 4, counter=counter)
            term.reset()
            terrain = env.scene.terrain
            terrain.terrain_types = torch.full((4,), 2, dtype=torch.long)
            terrain.terrain_levels = torch.full((4,), 3, dtype=torch.long)
            decisions = []

            def update_origins(ids, up, down):
                decisions.append((up.clone(), down.clone()))
                terrain.terrain_levels[ids] += up.long() - down.long()

            terrain.update_env_origins = update_origins
            env.command_manager = NS(get_term=lambda _: term)
            course = ThunderTerrainLevels(CurriculumTermCfg(func=ThunderTerrainLevels), env)
            directions = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [1., 0.]])
            term._requested_velocity_command_b[:, :2] = directions
            term._requested_velocity_command_b[:, 2] = 0.
            term._update_command()
            term.commands_xy_accumulation[:] = directions * .8
            before_budget = term.commands_xy_accumulation.clone()
            # Previous frame was exactly on the 4 m boundary. The reset happens
            # on the next frame, before CommandTerm.compute can record 4.01 m.
            term.max_move_distance[:] = torch.tensor([4., 4., 4., .1])
            env.scene["robot"].data.root_pos_w.torch[:, :2] += (
                directions * torch.tensor([[4.01], [4.01], [4.01], [.1]]))
            env.episode_length_buf[:] = 1000
            env.common_step_counter += 1
            result = course(env, torch.arange(4))
            self.assertEqual(decisions[-1][0].tolist(), [True, True, True, False])
            self.assertEqual(decisions[-1][1].tolist(), [False, False, False, True])
            self.assertEqual(terrain.terrain_levels.tolist(), [4, 4, 4, 2])
            self.assertEqual(result["pyramid_stairs_inv/max_level_reached"].item(), 4.)
            torch.testing.assert_close(term.commands_xy_accumulation,
                before_budget + term.command[:, :2] * (.02 / 5.))


if __name__ == "__main__":
    unittest.main()
