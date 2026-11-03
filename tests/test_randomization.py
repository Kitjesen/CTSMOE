"""CPU contracts for native Go2 randomization on the Thunder wheel-leg layout."""

import re
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch
from isaaclab_newton.physics import NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab.utils.string import ResolvableString
from scripts.probe_alignment import qualified_class

from go2_rl_lab.physics.randomize import apply_body_com_buckets
from go2_rl_lab.tasks.mixed.mdp.events import randomize_action_joint_pos_offset
from thunder_lab3.tasks.thunder.mdp.randomization import configure_randomization
from thunder_lab3.tasks.thunder.mdp.privileged import LEG_JOINTS, JOINTS


def configured():
    cfg = NS(events=NS(reset_scene=object(), reset_root=object()),
             sim=NS(physics=NewtonCfg(collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
                                     default_shape_cfg=NewtonShapeCfg(margin=.01), num_substeps=1)))
    configure_randomization(cfg)
    return cfg


class RandomizationTest(unittest.TestCase):
    def test_actual_lazy_manager_is_resolved_before_reporting_its_class(self):
        from collections import Counter
        proxy = ResolvableString('collections:Counter')
        self.assertIsNone(proxy._resolved_callable)
        self.assertEqual(qualified_class(proxy), 'collections:Counter')
        self.assertIs(proxy._resolved_callable, Counter)

    def test_parent_manager_dispatch_tracks_custom_solver_and_preserves_physics(self):
        physics = configured().sim.physics
        expected = 'go2_rl_lab.physics.resettable_mjwarp_manager:NewtonMJWarpResettableManager'
        self.assertEqual(physics.solver_cfg.class_type, expected)
        self.assertEqual(physics.class_type, expected)
        self.assertEqual(physics.collision_cfg.max_triangle_pairs, 2_500_000)
        self.assertEqual(physics.default_shape_cfg.margin, .01)
        self.assertEqual(physics.num_substeps, 1)

    def test_preserves_reset_events_and_scales_mass_with_inertia(self):
        cfg = configured()
        previous = (cfg.events.reset_scene, cfg.events.reset_root)
        configure_randomization(cfg)
        self.assertEqual((cfg.events.reset_scene, cfg.events.reset_root), previous)
        for name in ("randomize_rigid_body_mass_base", "randomize_rigid_body_mass_others"):
            term = getattr(cfg.events, name)
            self.assertEqual(term.mode, "startup")
            self.assertEqual(term.params["mass_distribution_params"], (0.9, 1.1))
            self.assertEqual(term.params["operation"], "scale")
            self.assertTrue(term.params["recompute_inertia"])
        pattern = cfg.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names
        self.assertIsNone(re.fullmatch(pattern, "base_link"))
        self.assertIsNotNone(re.fullmatch(pattern, "fr_foot_Link"))

    def test_leg_and_wheel_gains_have_disjoint_ownership(self):
        cfg = configured()
        legs = cfg.events.randomize_actuator_gains
        wheels = cfg.events.randomize_wheel_gains
        self.assertEqual((legs.mode, wheels.mode), ("reset", "startup"))
        self.assertEqual(legs.params["asset_cfg"].joint_names, list(LEG_JOINTS))
        self.assertEqual(wheels.params["asset_cfg"].joint_names, list(JOINTS[12:]))
        self.assertTrue(set(legs.params["asset_cfg"].joint_names).isdisjoint(
            wheels.params["asset_cfg"].joint_names))
        self.assertEqual(legs.params["stiffness_distribution_params"], (0.9, 1.1))
        self.assertIsNone(wheels.params["stiffness_distribution_params"])
        self.assertEqual(wheels.params["damping_distribution_params"], (0.9, 1.1))
        offset = cfg.events.randomize_motor_zero_offset
        self.assertIs(offset.func, randomize_action_joint_pos_offset)
        self.assertEqual(offset.params["action_term_name"], "joint_pos")

    def test_upstream_motor_zero_offset_is_partial_and_does_not_accumulate(self):
        torch.manual_seed(42)
        baseline = torch.arange(36, dtype=torch.float32).reshape(3, 12) / 20
        action = NS(_offset=baseline.clone())
        env = NS(scene=NS(num_envs=3), device="cpu",
                 action_manager=NS(get_term=lambda name: action if name == "joint_pos" else None))
        term = configured().events.randomize_motor_zero_offset
        for _ in range(20):
            term.func(env, [0, 2], **term.params)
            self.assertLessEqual((action._offset - baseline).abs().max().item(), 0.035001)
            torch.testing.assert_close(action._offset[1], baseline[1], rtol=0, atol=0)

    def test_material_and_push_use_official_ranges(self):
        from go2_rl_lab.tasks.mixed.env_cfg import EventCfg
        cfg = configured()
        official = EventCfg()
        for name in ("randomize_rigid_body_material", "randomize_push_robot"):
            ours, reference = getattr(cfg.events, name), getattr(official, name)
            self.assertIs(ours.func, reference.func)
            self.assertEqual(ours.mode, reference.mode)
            for key, value in reference.params.items():
                if key != "asset_cfg":
                    self.assertEqual(ours.params[key], value)
        self.assertEqual(cfg.events.randomize_push_robot.interval_range_s, (4., 4.))

    def test_actual_com_bucket_assignment_is_bounded_and_idempotent(self):
        cfg = configured().sim.physics.solver_cfg.body_com_buckets
        count = 1001
        builder = NS(world_count=count, body_world=list(range(count)) + [-1],
                     body_label=[f"/World/envs/env_{i}/Robot/Geometry/base_link" for i in range(count)] + ["ground"],
                     body_com=[(0., 0., 0.) for _ in range(count)] + [(9., 9., 9.)])
        report = apply_body_com_buckets(builder, cfg)
        values = np.asarray([list(v) for v in builder.body_com[:count]])
        self.assertTrue(np.all((values >= [0., -.05, 0.]) & (values <= [.1, .05, .05])))
        self.assertLessEqual(max(report.joint_bucket_counts) - min(report.joint_bucket_counts), 1)
        self.assertEqual(builder.body_com[-1], (9., 9., 9.))
        self.assertTrue(apply_body_com_buckets(builder, cfg).already_applied)
        np.testing.assert_array_equal(values, np.asarray([list(v) for v in builder.body_com[:count]]))

    def test_com_bucket_rejects_unmatched_robot_asset(self):
        cfg = configured().sim.physics.solver_cfg.body_com_buckets
        builder = NS(world_count=1, body_world=[0], body_label=["/Robot/base"], body_com=[(0., 0., 0.)])
        with self.assertRaisesRegex(ValueError, "exactly one body"):
            apply_body_com_buckets(builder, cfg)


if __name__ == "__main__":
    unittest.main()
