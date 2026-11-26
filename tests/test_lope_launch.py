"""Plan-only checks for the Lope launcher. They do not start a simulator."""
import json
from pathlib import Path
import subprocess
import sys
import unittest


class LopeLaunchTests(unittest.TestCase):
    def _launcher(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl"
        lope_specific = scripts / "launch_lope.py"
        return lope_specific if lope_specific.is_file() else scripts / "launch.py"

    def plan(self, *args, include_urdf=True, success=True):
        entry = self._launcher()
        urdf = Path(__file__).resolve().parents[1] / "source/lope_rl_lab/lope_rl_lab/assets/data/urdf/lope.urdf"
        cmd = [sys.executable, str(entry), *args]
        if include_urdf:
            cmd += ["--urdf", str(urdf)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_fresh_plan_is_lope_and_loads_no_checkpoint(self):
        plan = self.plan("train", "--updates", "2", "--run-name", "lope-smoke")
        self.assertEqual(plan["task"], "lope-locomotion-mixed")
        self.assertEqual(plan["argv"][plan["argv"].index("--task") + 1], "lope-locomotion-mixed")
        self.assertIsNone(plan["checkpoint_source"])
        self.assertNotIn("--resume", plan["argv"])
        self.assertNotIn("--checkpoint", plan["argv"])
        self.assertEqual(plan["action_dim"], 12)
        self.assertEqual(plan["wheel_actions"], 0)
        self.assertFalse(plan["thunder_checkpoint"])
        self.assertFalse(plan["thunder_task"])
        self.assertNotIn("Thunder", " ".join(plan["argv"]))
        self.assertEqual(plan["training_state"], "fresh_random_initialization_from_0")
        self.assertEqual(plan["cumulative_target"], 100000)

        default_urdf_plan = self.plan("train", "--updates", "2", include_urdf=False)
        self.assertEqual(default_urdf_plan["urdf"], plan["urdf"])

    def test_resume_must_be_explicit_and_budget_is_the_official_cap(self):
        plan = self.plan("train", "--updates", "2", "--resume", "--checkpoint", "model_2.pt")
        self.assertIn("--resume", plan["argv"])
        self.plan("train", "--updates", "2", "--checkpoint", "model_2.pt", success=False)
        self.plan("train", "--updates", "100001", success=False)
        self.plan("train", "--updates", "0", success=False)
        accepted = self.plan("train", "--updates", "100000")
        self.assertIn("100000", accepted["argv"])

    def test_playback_selects_lope_terrain_keys(self):
        plan = self.plan("play", "--checkpoint", "model_2.pt", "--terrain", "stairs_up", "--level", "7")
        self.assertEqual(plan["play_terrain"], "stairs_up")
        self.assertEqual(plan["play_fixed_level"], 7)
        self.assertIn("lope-locomotion-mixed", plan["argv"])
        self.plan("play", "--checkpoint", "model_2.pt", "--level", "9")
        self.plan("play", "--checkpoint", "model_2.pt", "--level", "10", success=False)
        self.plan("play", success=False)


if __name__ == "__main__":
    unittest.main()
