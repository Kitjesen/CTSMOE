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


if __name__ == "__main__":
    unittest.main()
