import math
import random
import types
import unittest

import torch

from thunder_lab3.tasks.thunder.boxes import amplitude, with_amplitude
from thunder_lab3.tasks.thunder.mdp.terms import HeightOptions, HeightReference
from thunder_lab3.tasks.thunder.mdp.privileged import tensor


OPTIONS = HeightOptions(target_height=.42, tolerance=.02, max_reference_speed=.5,
                        max_reference_lag=.04)


class HeightTest(unittest.TestCase):
    def make(self, count=2):
        return HeightReference(count, "cpu", torch.float64, .02, OPTIONS)

    def test_cost_band_and_translation_invariance(self):
        base = torch.tensor([.40, .50], dtype=torch.float64)
        rays = torch.zeros((2, 15), dtype=torch.float64)
        low = self.make().step(base, rays)
        high = self.make().step(base + 37, rays + 37)
        torch.testing.assert_close(low, torch.tensor([0, .06**2], dtype=torch.float64), atol=1e-12, rtol=0)
        torch.testing.assert_close(low, high, atol=1e-12, rtol=0)

    def test_missing_then_valid_and_partial_reset(self):
        state = self.make()
        missing = torch.full((2, 3), torch.nan, dtype=torch.float64)
        cost = state.step(torch.ones(2, dtype=torch.float64), missing)
        self.assertTrue(torch.equal(cost, torch.zeros_like(cost)))
        self.assertFalse(bool(state.initialized.any()))
        state.step(torch.tensor([.42, 1.42]), torch.tensor([[0., 0.], [1., 1.]]))
        state.reset(torch.tensor([0]))
        self.assertEqual(state.ground_z.tolist(), [0., 1.])
        self.assertEqual(state.initialized.tolist(), [False, True])
        cost = state.step(torch.tensor([10.42, 1.42]), torch.tensor([[10., 10.], [1., 1.]]))
        torch.testing.assert_close(cost, torch.zeros_like(cost), atol=1e-10, rtol=0)
        self.assertEqual(state.ground_z.tolist(), [10., 1.])

    def test_step_edge_is_suppressed_until_reference_catches_up(self):
        state = self.make(1)
        state.step(torch.tensor([.42]), torch.zeros((1, 15)))
        last = state.ground_z.clone()
        cost = state.step(torch.tensor([.42]), torch.full((1, 15), -.239))
        self.assertLessEqual(abs(float(state.ground_z[0] - last[0])), .01 + 1e-12)
        self.assertFalse(bool(state.active[0]))
        self.assertEqual(float(cost[0]), 0)
        for _ in range(24):
            state.step(torch.tensor([.42 - .239]), torch.full((1, 15), -.239))
        self.assertTrue(bool(state.active[0]))
        self.assertLess(abs(float(state.relative_height[0]) - .42), 1e-7)

    def test_bad_sensor_frames_never_create_nan_penalty(self):
        state = self.make()
        state.step(torch.tensor([.42, .42]), torch.zeros((2, 3)))
        cost = state.step(torch.tensor([float('nan'), .42]),
                          torch.tensor([[0., 0., 0.], [0., float('inf'), 0.]]))
        self.assertEqual(cost.tolist(), [0., 0.])
        self.assertFalse(bool(state.active.any()))
        self.assertEqual(state.ground_z.tolist(), [0., 0.])

    def test_settings_and_shapes_reject_invalid(self):
        for value in (0, -.1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                HeightOptions(value, .02, .5, .04)
        with self.assertRaises(ValueError):
            self.make().step(torch.zeros(2), torch.empty((2, 0)))
        with self.assertRaises(ValueError):
            self.make().step(torch.zeros(1), torch.zeros((1, 3)))
        with self.assertRaises(ValueError):
            HeightOptions(.42, .5, .5, .04)

    def test_composed_options_are_revalidated_and_runtime_is_isolated(self):
        options = HeightOptions(.42, .02, .5, .04)
        options.target_height = .46
        state = HeightReference(1, "cpu", torch.float64, .02, options)
        options.target_height = -.1
        self.assertEqual(state.options.target_height, .46)
        with self.assertRaises(ValueError):
            HeightReference(1, "cpu", torch.float64, .02, options)


class BoxesTest(unittest.TestCase):
    def test_bounds_and_monotonicity(self):
        last = amplitude(0)
        self.assertEqual(last, .01)
        for i in range(1, 1001):
            value = amplitude(i / 1000)
            self.assertGreater(value, last)
            self.assertLessEqual(value, .20)
            last = value
        self.assertAlmostEqual(amplitude(.3), .046)
        self.assertAlmostEqual(last, .20)
        for d in (-.1, 1.01, math.nan):
            with self.assertRaises(ValueError):
                amplitude(d)

    def test_native_call_preserves_layout_rng_and_config(self):
        cfg = types.SimpleNamespace(grid_height_range=(.05, .20), grid_width=.45)
        seen = []
        def native(d, c):
            seen.append((d, c.grid_width, c.grid_height_range))
            return [random.random() for _ in range(16)]
        random.seed(42)
        result = with_amplitude(.25, cfg, native)
        random.seed(42)
        expected = native(.25, cfg)
        self.assertEqual(result, expected)
        self.assertEqual(cfg.grid_height_range, (.05, .20))
        self.assertEqual(seen[0][2], (amplitude(.25), amplitude(.25)))
        self.assertEqual(seen[0][:2], (.25, .45))

class AdapterTest(unittest.TestCase):
    def test_tensor_and_proxy_views(self):
        class Proxy:
            def __init__(self, value): self.torch = value
        x = torch.zeros((1, 3))
        self.assertIs(tensor(x), x)
        self.assertIs(tensor(Proxy(x)), x)
        with self.assertRaises(TypeError):
            tensor(Proxy(None))


if __name__ == "__main__":
    unittest.main()
