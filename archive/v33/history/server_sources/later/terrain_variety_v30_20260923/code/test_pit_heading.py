"""Check the pit heading command's sign, gain and physical command bound."""

import math

import torch

from pit_heading import heading_rate


def test_heading_rate_returns_toward_world_x_and_saturates():
    yaw = torch.tensor([0., .04, -.20, 1.0, -1.0])
    q = torch.zeros((len(yaw), 4))
    q[:, 0] = torch.cos(yaw / 2)
    q[:, 3] = torch.sin(yaw / 2)
    actual = heading_rate(q)
    expected = torch.tensor([0., -.06, .3, -.6, .6])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    assert math.isclose(float(actual.abs().max()), .6, abs_tol=1e-6)
