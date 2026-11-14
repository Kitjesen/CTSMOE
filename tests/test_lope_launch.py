"""Plan-only checks for the Lope launcher. They do not start a simulator."""
import json
from pathlib import Path
import subprocess
import sys
import unittest


class LopeLaunchTests(unittest.TestCase):
    def plan(self, *args, success=True):
        entry = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/launch_lope.py"
        urdf = Path(__file__).resolve().parents[1] / "source/lope_lab3/assets/lope/urdf/lope.urdf"
        result = subprocess.run([sys.executable, str(entry), *args, "--urdf", str(urdf)],
                                capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_fresh_plan_is_lope_and_loads_no_checkpoint(self):
        plan = self.plan("train", "--updates", "2", "--run-name", "lope-smoke")
        self.assertEqual(plan["task"], "Lope-Newton3-Fresh-v0")
        self.assertEqual(plan["argv"][plan["argv"].index("--task") + 1], "Lope-Newton3-Fresh-v0")
        self.assertIsNone(plan["checkpoint_source"])
        self.assertNotIn("--resume", plan["argv"])
        self.assertNotIn("--checkpoint", plan["argv"])
        self.assertEqual(plan["action_dim"], 12)
        self.assertEqual(plan["wheel_actions"], 0)
        self.assertFalse(plan["thunder_checkpoint"])
        self.assertFalse(plan["thunder_task"])
        self.assertNotIn("Thunder", " ".join(plan["argv"]))
        self.assertEqual(plan["training_state"], "fresh_random_initialization_from_0")
        self.assertEqual(plan["cumulative_target"], 3000)

    def test_resume_must_be_explicit_and_budget_is_the_rough_task_cap(self):
        plan = self.plan("train", "--updates", "2", "--resume", "--checkpoint", "model_2.pt")
        self.assertIn("--resume", plan["argv"])
        self.plan("train", "--updates", "2", "--checkpoint", "model_2.pt", success=False)
        self.plan("train", "--updates", "3001", success=False)
        self.plan("train", "--updates", "0", success=False)
        accepted = self.plan("train", "--updates", "3000")
        self.assertIn("3000", accepted["argv"])

    def test_playback_selects_lope_terrain_keys(self):
        plan = self.plan("play", "--checkpoint", "model_2.pt", "--terrain", "stairs_up", "--level", "7")
        self.assertEqual(plan["play_terrain"], "stairs_up")
        self.assertEqual(plan["play_fixed_level"], 7)
        self.assertIn("Lope-Newton3-Fresh-v0", plan["argv"])
        self.plan("play", "--checkpoint", "model_2.pt", "--level", "8", success=False)
        self.plan("play", success=False)


if __name__ == "__main__":
    unittest.main()
