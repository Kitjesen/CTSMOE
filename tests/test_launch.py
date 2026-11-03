"""Dependency-free CLI boundary checks: python -m unittest tests.test_launch."""

import json
from pathlib import Path
import subprocess
import sys
import unittest


class LaunchTests(unittest.TestCase):
    def plan(self, *args, success=True):
        entry = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/launch.py"
        result = subprocess.run([sys.executable, str(entry), *args, "--urdf", "robot.urdf"],
                                capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)

    def test_fresh_training_never_loads_checkpoint(self):
        plan = self.plan("train", "--updates", "2", "--run-name", "smoke")
        self.assertIsNone(plan["checkpoint_source"])
        self.assertNotIn("--resume", plan["argv"])
        self.assertNotIn("--checkpoint", plan["argv"])
        self.assertIn("Thunder-Newpose-Newton3-Fresh-v0", plan["argv"])
        self.assertEqual(plan["training_state"], "fresh_random_initialization_from_0")

    def test_resume_is_explicit(self):
        plan = self.plan("train", "--updates", "2", "--resume", "--checkpoint", "model_2.pt")
        self.assertIn("--resume", plan["argv"])
        self.plan("train", "--updates", "2", "--checkpoint", "model_2.pt", success=False)
        self.plan("train", "--updates", "2", "--resume", success=False)
        self.plan("train", "--updates", "2", "--legacy", success=False)

    def test_environment_count_uses_task_default_until_explicitly_overridden(self):
        default = self.plan("train", "--updates", "2")
        self.assertNotIn("--num_envs", default["argv"])
        explicit = self.plan("train", "--updates", "2", "--num-envs", "32")
        index = explicit["argv"].index("--num_envs")
        self.assertEqual(explicit["argv"][index + 1], "32")
        for invalid in ("0", "-4", "3", "6"):
            with self.subTest(invalid=invalid):
                self.plan("train", "--updates", "2", "--num-envs", invalid, success=False)
        self.assertEqual(default["cumulative_target"], 20000)

    def test_play_requires_model_and_distinguishes_legacy(self):
        self.plan("play", success=False)
        fresh = self.plan("play", "--checkpoint", "model_2.pt", "--terrain", "stairs_up", "--level", "9")
        self.assertIn("Thunder-Newpose-Newton3-Fresh-v0", fresh["argv"])
        self.assertNotIn("--seed", fresh["argv"])
        old = self.plan("play", "--checkpoint", "model_19538.pt", "--legacy")
        self.assertIn("Thunder-Newpose-Newton3-v0", old["argv"])
        self.assertNotIn("--num_envs", fresh["argv"])
        self.assertNotIn("--num_envs", old["argv"])


if __name__ == "__main__":
    unittest.main()
