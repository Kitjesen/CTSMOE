"""Append force/actuator features to the teacher and critic, never to the student."""
from dataclasses import dataclass

import torch

LEGS = ('fr', 'fl', 'rr', 'rl')
LEG_JOINTS = tuple(f'{leg}_{joint}_joint' for leg in LEGS for joint in ('hip', 'thigh', 'calf'))
JOINTS = LEG_JOINTS + tuple(f'{leg}_foot_joint' for leg in LEGS)
WHEEL_BODIES = tuple(f'{leg}_foot_Link' for leg in LEGS)
FEATURE_NAMES = ('joint_acc_16', 'joint_applied_torque_or_proxy_16', 'wheel_normal_contact_force_magnitude_4')


@dataclass(frozen=True)
class FeatureScales:
    acceleration: float = 1e-4
    torque: float = 0.01
    contact_force: float = 0.001
    clip: float = 10.0

    def __post_init__(self):
        if not all(0 < value < float('inf') for value in vars(self).values()):
            raise ValueError('Feature scales and clip must be finite and positive')


def tensor(value):
    """Isaac Lab 2 Tensor or Isaac Lab 3 ProxyArray; no CPU copy."""
    result = value if isinstance(value, torch.Tensor) else getattr(value, 'torch', None)
    if not isinstance(result, torch.Tensor):
        raise TypeError('Expected torch Tensor or ProxyArray.torch')
    return result


def normalized_features(acceleration, torque, force_w, reset_mask, scales=FeatureScales()):
    """Force is the norm of each wheel's net normal-contact vector, not a sum of point magnitudes."""
    acceleration, torque, force_w = map(tensor, (acceleration, torque, force_w))
    n = acceleration.shape[0]
    if acceleration.shape != (n, 16) or torque.shape != (n, 16) or force_w.shape != (n, 4, 3):
        raise ValueError('Expected canonical (N,16), (N,16), (N,4,3) features')
    reset_mask = tensor(reset_mask).bool()
    if reset_mask.shape != (n,):
        raise ValueError('Reset mask must have one entry per environment')
    if not all(torch.isfinite(value).all() for value in (acceleration, torque, force_w)):
        raise ValueError('Nonfinite privileged physics signal')
    # Reset velocities are discontinuous and must not become acceleration evidence.
    acceleration = torch.where(reset_mask[:, None], 0., acceleration)
    features = torch.cat((acceleration * scales.acceleration, torque * scales.torque,
                          torch.linalg.vector_norm(force_w, dim=-1) * scales.contact_force), dim=-1)
    return features.clamp(-scales.clip, scales.clip)
