"""Privileged 12-DoF signals. Foot contact replaces every wheel feature."""
from dataclasses import dataclass

import torch

from lope_lab3.spec import FOOT_BODIES, JOINT_NAMES

FEATURE_NAMES = (
    "joint_acc_12",
    "joint_applied_torque_12",
    "foot_normal_contact_force_magnitude_4",
)


@dataclass(frozen=True)
class FeatureScales:
    acceleration: float = 1e-4
    torque: float = 0.01
    contact_force: float = 1e-3


def tensor(value):
    """Isaac Lab 3 ProxyArray or a Tensor. Do not copy to CPU."""
    result = value if isinstance(value, torch.Tensor) else getattr(value, "torch", None)
    if not isinstance(result, torch.Tensor):
        raise TypeError("Expected torch Tensor or ProxyArray.torch")
    return result


def joint_indices(robot):
    return [robot.joint_names.index(name) for name in JOINT_NAMES]


def foot_indices(names):
    return [list(names).index(name) for name in FOOT_BODIES]
