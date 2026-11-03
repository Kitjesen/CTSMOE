"""Guard against counting resets as actuator oscillation or continuous survival."""
from pathlib import Path
import importlib.util
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('alignment_probe', Path(__file__).resolve().parents[1] / 'scripts/probe_alignment.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class AlignmentProbeTest(unittest.TestCase):
    def test_resolved_class_and_string_config_compare_identically(self):
        expected = f'{AlignmentProbeTest.__module__}:{AlignmentProbeTest.__qualname__}'
        self.assertEqual(probe.qualified_class(expected), expected)
        self.assertEqual(probe.qualified_class(AlignmentProbeTest), expected)
        self.assertEqual(probe.qualified_class(AlignmentProbeTest()), expected)

    def test_reset_discontinuities_are_censored(self):
        count = 5
        data = dict(episode_index=np.array([[0], [0], [1], [1], [1]]),
                    torque_proxy=np.zeros((count, 1, 16)),
                    contact_normal=np.zeros((count, 1, 7, 3)),
                    gravity=np.tile([0., 0., -1.], (count, 1, 1)),
                    root_position=np.zeros((count, 1, 3)))
        data['torque_proxy'][:, 0, 0] = [10., 10., -10., 120., -120.]
        metrics = probe.summarize(data, .005, [0, 1, 2, 3], [4, 5], [6])
        self.assertEqual(metrics['first_episode_seconds_per_env'], [.01])
        self.assertEqual(metrics['adjacent_leg_torque_signflip_over5Nm_counts'], [0]*12)
        self.assertEqual(metrics['leg_torque_120Nm_samples'], [0]*12)
        data['torque_proxy'][1, 0, 0] = -10.
        metrics = probe.summarize(data, .005, [0, 1, 2, 3], [4, 5], [6])
        self.assertEqual(metrics['adjacent_leg_torque_signflip_over5Nm_counts'], [1]+[0]*11)
        self.assertEqual(metrics['signflip_env_ids'], [0])


if __name__ == '__main__':
    unittest.main()
