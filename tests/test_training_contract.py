"""Construct the actual fresh Lab task on CPU and validate its checkpoint contract."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from thunder_lab3.training.fresh_state import canonical_configuration, training_contract


class ConfigurationEncodingTests(unittest.TestCase):
    def encode(self, value):
        return json.dumps(canonical_configuration(value), sort_keys=True, allow_nan=False)

    def test_resolved_types_and_nonstring_keys_are_stable_and_distinct(self):
        items = {("hip", 0): slice(None, np.int64(12), 2), "device": torch.device("cpu"),
                 "dtype": torch.float32, "values": np.array([1., 2.], dtype=np.float32)}
        self.assertEqual(self.encode(items), self.encode(dict(reversed(list(items.items())))))
        self.assertNotEqual(self.encode({1: "a"}), self.encode({"1": "a"}))
        self.assertNotEqual(self.encode(slice(0, 3)), self.encode(slice(0, 4)))
        with self.assertRaises(ValueError):
            self.encode({"bad": np.array([float("nan")])})
        with self.assertRaises(ValueError):
            self.encode({"bad": float("inf")})
        with self.assertRaises(TypeError):
            self.encode(object())

    def test_real_fresh_config_hashes_on_cpu_and_detects_reward_change(self):
        from isaacsim.asset.importer.urdf import URDFImporter  # noqa: F401
        from thunder_lab3.tasks.thunder.env_cfg import ThunderFreshEnvCfg

        project = Path(__file__).resolve().parents[1]
        asset = project / "source/thunder_lab3/assets/thunder/urdf/thunder_v4.urdf"
        with tempfile.TemporaryDirectory(prefix="thunder_cfg_cpu_") as cache, patch.dict(os.environ, {
                "NEWPOSE_URDF": str(asset), "NEWPOSE_USD_CACHE": cache}):
            cfg = ThunderFreshEnvCfg()
            env = SimpleNamespace(cfg=cfg)
            first = training_contract(env)
            self.assertEqual(first, training_contract(env))
            self.assertEqual(first["physics_schema"], "newton_mjwarp_triangle_v1")
            cfg.rewards.track_lin_vel_xy_exp.weight += .1
            self.assertNotEqual(first["sha256"], training_contract(env)["sha256"])
            before=training_contract(env)
            cfg.scene.height_scanner.offset.pos=(0.,0.,19.)
            self.assertNotEqual(before['sha256'],training_contract(env)['sha256'])
            before=training_contract(env)
            cfg.scene.contact_forces.update_period=.01
            self.assertNotEqual(before['sha256'],training_contract(env)['sha256'])
            before=training_contract(env)
            read_bytes=Path.read_bytes
            def changed_runner(path):
                data=read_bytes(path)
                return data+b'\n# simulated runner change\n' if path.name=='runner.py' else data
            with patch.object(Path,'read_bytes',changed_runner):
                self.assertNotEqual(before['sha256'],training_contract(env)['sha256'])

    def test_fresh_runner_matches_official_core_and_keeps_legacy_separate(self):
        import gymnasium as gym
        from go2_rl_lab.tasks.mixed.rsl_rl_cfg import Go2MoECTSRunnerCfg
        from thunder_lab3.tasks.thunder import FRESH_TASK_ID,TASK_ID
        from thunder_lab3.tasks.thunder.rsl_rl_cfg import ThunderFreshMoeCtsRunnerCfg,ThunderMoeCtsRunnerCfg

        fresh=ThunderFreshMoeCtsRunnerCfg();official=Go2MoECTSRunnerCfg();legacy=ThunderMoeCtsRunnerCfg()
        for field in ('num_steps_per_env','clip_rewards','save_interval','obs_groups'):
            self.assertEqual(getattr(fresh,field),getattr(official,field),field)
        for field in ('init_noise_std','noise_std_type','actor_hidden_dims','critic_hidden_dims',
                'teacher_encoder_hidden_dims','student_encoder_hidden_dims','expert_num','latent_dim',
                'norm_type','activation'):
            self.assertEqual(getattr(fresh.policy,field),getattr(official.policy,field),field)
        for field in ('value_loss_coef','load_balance_coef','use_clipped_value_loss','clip_param','entropy_coef',
                'num_learning_epochs','num_mini_batches','learning_rate','student_encoder_learning_rate',
                'schedule','gamma','lam','betas','weight_decay','desired_kl','max_grad_norm',
                'teacher_env_ratio','normalize_advantage_per_mini_batch'):
            self.assertEqual(getattr(fresh.algorithm,field),getattr(official.algorithm,field),field)
        self.assertTrue(fresh.algorithm.symmetry_cfg.use_symmetric_augmentation)
        self.assertEqual(fresh.algorithm.symmetry_cfg.symmetry_class,
            'thunder_lab3.tasks.thunder.mdp.symmetry:ThunderSymmetry')
        self.assertIsNotNone(official.algorithm.symmetry_cfg)
        self.assertEqual(fresh.max_iterations,100000)
        self.assertEqual(fresh.newpose_compat['cumulative_target'],100000)
        self.assertEqual(official.max_iterations,100000)
        self.assertEqual(legacy.policy.noise_std_type,'log')
        self.assertIsNone(legacy.clip_rewards)
        self.assertEqual(legacy.save_interval,100)
        self.assertIsNone(legacy.algorithm.symmetry_cfg)
        self.assertEqual(legacy.newpose_compat['cumulative_target'],20000)
        self.assertEqual(gym.spec(FRESH_TASK_ID).kwargs['rsl_rl_cfg_entry_point'],
            'thunder_lab3.tasks.thunder.rsl_rl_cfg:ThunderFreshMoeCtsRunnerCfg')
        self.assertEqual(gym.spec(TASK_ID).kwargs['rsl_rl_cfg_entry_point'],
            'thunder_lab3.tasks.thunder.rsl_rl_cfg:ThunderMoeCtsRunnerCfg')
        fresh.policy.actor_hidden_dims[0]=1
        self.assertEqual(legacy.policy.actor_hidden_dims,[512,256,128])
        self.assertEqual(ThunderFreshMoeCtsRunnerCfg().policy.actor_hidden_dims,[512,256,128])


if __name__ == "__main__":
    unittest.main()
