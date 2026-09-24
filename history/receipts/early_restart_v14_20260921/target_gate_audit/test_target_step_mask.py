import unittest

import numpy as np

from target_step_mask import observable_target_deltas


class TargetStepMaskTests(unittest.TestCase):
    def test_termination_excludes_overwritten_terminal_sample(self):
        target = np.array([[0.1], [0.2], [0.9]])

        deltas, step_ids = observable_target_deltas(
            target,
            np.array([0.0]),
            end_step=2,
            termination_terms=["time_out"],
        )

        np.testing.assert_array_equal(step_ids, [0, 1])
        np.testing.assert_allclose(deltas[:, 0], [0.1, 0.1])

    def test_no_termination_terms_retains_last_sample(self):
        target = np.array([[0.1], [0.2], [0.3]])

        deltas, step_ids = observable_target_deltas(
            target,
            np.array([0.0]),
            end_step=2,
            termination_terms=[],
        )

        np.testing.assert_array_equal(step_ids, [0, 1, 2])
        np.testing.assert_allclose(deltas[:, 0], [0.1, 0.1, 0.1])

    def test_first_step_still_uses_initial_q_difference(self):
        target = np.array([[0.34], [0.40]])

        deltas, step_ids = observable_target_deltas(
            target,
            np.array([0.10]),
            end_step=1,
            termination_terms=[],
        )

        np.testing.assert_array_equal(step_ids, [0, 1])
        np.testing.assert_allclose(deltas[:, 0], [0.24, 0.06])


if __name__ == "__main__":
    unittest.main()
