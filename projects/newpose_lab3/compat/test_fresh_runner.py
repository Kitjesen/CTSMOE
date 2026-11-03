"""Real upstream CTS CPU update/resume/export test: python -m unittest compat.test_fresh_runner."""

import copy
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from .checkpoint import FRESH_FORMAT, load_checkpoint, normalize_checkpoint, validate_for_algorithm
from .history import FrameHistory
from .runner import ThunderMoeCtsRunner
from .verify_bridge import equal_nested, runner_config, verify_timeout_boundary
from .verify_timeout_runner import TimeoutEnv


class NativeSyntheticEnv(TimeoutEnv):
    def __init__(self):
        super().__init__()
        self.cfg.newpose_physics_schema = "newton_mjwarp_triangle_v1"
        self.cfg.newpose_terrain_schema = "thunder_fresh_10level_v1"
        self.cfg.newpose_test_contract = {"fixture": "native_synthetic_v1"}
        origins = torch.arange(10 * 2 * 3, dtype=torch.float32).reshape(10, 2, 3)
        levels, types = torch.arange(8) % 10, torch.arange(8) % 2
        self.scene = SimpleNamespace(terrain=SimpleNamespace(
            terrain_origins=origins, terrain_levels=levels, terrain_types=types,
            env_origins=origins[levels, types].clone()))
        self.common_step_counter = 0
        self._sim_step_counter = 0
        self.curriculum_calls = self.reset_calls = 0
        self.curriculum_manager = SimpleNamespace(compute=self._advance_curriculum)

    def _advance_curriculum(self, **kwargs):
        self.curriculum_calls += 1
        self.scene.terrain.terrain_levels.zero_()

    def reset(self):
        self.curriculum_manager.compute(env_ids=slice(None))
        self.reset_calls += 1
        self.episode_length_buf.zero_()
        self.history = FrameHistory(torch.randn(8, 53) * .1)
        self.critic = torch.randn(8, 309) * .1
        return self.get_observations(), {}

    def step(self, actions):
        self.common_step_counter += 1
        self._sim_step_counter += 8
        return super().step(actions)


class FreshRunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.mode = patch.dict(os.environ, {"NEWPOSE_RUN_MODE": "train"})
        self.mode.start()

    def tearDown(self):
        self.mode.stop()

    def test_actual_cts_updates_roundtrip_and_export(self):
        verify_timeout_boundary()
        with tempfile.TemporaryDirectory(prefix="thunder_fresh_cpu_") as directory:
            runner = ThunderMoeCtsRunner(NativeSyntheticEnv(), runner_config(), directory, "cpu")
            before = {key: value.clone() for key, value in runner.alg.policy.state_dict().items()}
            self.assertFalse(runner._checkpoint_loaded)
            self.assertFalse(runner.alg.optimizer.state)
            self.assertFalse(runner.alg.optimizer_stu_enc.state)
            losses = []
            original_log = runner.logger.log

            def record(**kwargs):
                losses.append(kwargs["loss_dict"])
                return original_log(**kwargs)

            runner.logger.log = record
            runner.learn(2)
            self.assertEqual(runner.completed_updates, 2)
            self.assertEqual(runner.current_learning_iteration, 2)
            self.assertEqual(runner.terminal_bootstraps, 28)
            self.assertEqual(len(losses), 2)
            self.assertTrue(all(math.isfinite(value) for row in losses for value in row.values()))
            after = runner.alg.policy.state_dict()
            for prefix in ("teacher_encoder.", "actor.", "critic.", "student_moe_encoder."):
                self.assertTrue(any(not torch.equal(value, after[key]) for key, value in before.items() if key.startswith(prefix)))
            for optimizer in (runner.alg.optimizer, runner.alg.optimizer_stu_enc):
                self.assertTrue(optimizer.state)
                self.assertTrue(all(int(value["step"]) == 8 for value in optimizer.state.values()))
            self.assertTrue(all(torch.isfinite(value).all() for value in after.values()))
            initial = load_checkpoint(Path(directory) / "model_0.pt")
            self.assertEqual(initial["completed_updates"], 0)
            validate_for_algorithm(initial, runner.alg)
            saved = load_checkpoint(Path(directory) / "model_2.pt")
            self.assertEqual(saved["compat_format"], FRESH_FORMAT)
            self.assertEqual(saved["iter"], 2)
            self.assertEqual(saved["environment_state"]["common_step_counter"], 8)
            self.assertNotIn("formal_source_sha256", saved["compat_provenance"])

            resumed = ThunderMoeCtsRunner(NativeSyntheticEnv(), runner_config(), str(Path(directory) / "resume"), "cpu")
            resumed.load(Path(directory) / "model_2.pt")
            self.assertTrue(equal_nested(resumed.alg.optimizer.state_dict(), runner.alg.optimizer.state_dict()))
            self.assertTrue(equal_nested(resumed.alg.optimizer_stu_enc.state_dict(), runner.alg.optimizer_stu_enc.state_dict()))
            resumed.env.common_step_counter = 999  # upstream train.py's derived counter must not win
            resumed.learn(1, init_at_random_ep_len=True)
            self.assertEqual(resumed.completed_updates, 3)
            self.assertEqual(resumed.env.common_step_counter, 12)
            self.assertEqual(resumed.env._sim_step_counter, 96)
            self.assertEqual(resumed.env.reset_calls, 1)
            self.assertEqual(resumed.env.curriculum_calls, 0)
            self.assertTrue(torch.equal(resumed.env.scene.terrain.terrain_levels, saved["environment_state"]["terrain_levels"]))
            self.assertFalse(resumed.environment_resume_status["physics_state_restored"])

            obs = resumed.env.get_observations()
            expected = resumed.get_inference_policy("cpu")(obs)
            path = resumed.export_policy_as_jit(directory)
            serialized = torch.jit.load(str(path))
            with torch.inference_mode():
                torch.testing.assert_close(serialized(obs["policy"]), expected)
                self.assertEqual(tuple(serialized(torch.zeros(3, 530)).shape), (3, 16))
            with self.assertRaises(ValueError):
                resumed.learn(20000)
            invalid = copy.deepcopy(saved)
            invalid["iter"] = 3
            with self.assertRaises(ValueError):
                normalize_checkpoint(invalid, "unused")

            diagnostic = ThunderMoeCtsRunner(NativeSyntheticEnv(), runner_config(), device="cpu")
            diagnostic.load(Path(directory) / "model_2.pt", load_optimizer=False)
            with self.assertRaises(ValueError):
                diagnostic.learn(1)
            with self.assertRaises(ValueError):
                diagnostic.save(Path(directory) / "diagnostic.pt")

    def test_resume_rejects_different_task_before_model_mutation(self):
        with tempfile.TemporaryDirectory(prefix="thunder_contract_cpu_") as directory:
            runner = ThunderMoeCtsRunner(NativeSyntheticEnv(), runner_config(), device="cpu")
            runner.save(Path(directory) / "model_0.pt")
            env = NativeSyntheticEnv()
            env.cfg.newpose_test_contract["changed_reward"] = True
            other = ThunderMoeCtsRunner(env, runner_config(), device="cpu")
            before = {key: value.clone() for key, value in other.alg.policy.state_dict().items()}
            with self.assertRaises(ValueError):
                other.load(Path(directory) / "model_0.pt")
            self.assertTrue(equal_nested(before, other.alg.policy.state_dict()))


if __name__ == "__main__":
    unittest.main()
