"""Bounded yaw-rate command that holds the pit's world +X traversal axis."""

import torch


GAIN = 1.5
MAX_YAW_RATE = 0.6


def heading_rate(root_quat_w):
    """Return a body yaw-rate target from world-frame wxyz root quaternions."""
    q = root_quat_w
    yaw = torch.atan2(
        2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1 - 2 * (q[:, 2].square() + q[:, 3].square()),
    )
    return (-GAIN * yaw).clamp(-MAX_YAW_RATE, MAX_YAW_RATE)
